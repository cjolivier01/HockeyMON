from pathlib import Path
from typing import Any

import torch

from hmlib.config import prepend_root_dir
from hmlib.log import get_logger
from hmlib.utils.numpy_pickle_compat import numpy2_pickle_compat
from hmlib.utils.torch_tensorrt_runtime import compile_torch_tensorrt

from .base import Plugin

logger = get_logger(__name__)


def _decode_pose_predictions(model, features, data_samples):
    """Preserve MMPose head options and crop-to-image coordinate restoration."""
    predictions = model.head.predict(features, data_samples, test_cfg=model.test_cfg)
    if isinstance(predictions, tuple):
        instances, fields = predictions
    else:
        instances, fields = predictions, None
    return model.add_pred_to_datasample(instances, fields, data_samples)


class PoseInferencerFactoryPlugin(Plugin):
    """
    Builds and caches an MMPoseInferencer and exposes it via context['pose_inferencer'].

    Params:
      - pose_config: str path to pose2d config (mmengine-style path or alias)
      - pose_checkpoint: Optional[str] checkpoint path/URL
      - device: Optional[str] device string (overrides context['device'] if provided)
      - show_progress: bool (default False)
      - filter_args: Optional[Dict] default preprocess/forward args (bbox_thr, nms_thr, etc.)
    """

    def __init__(
        self,
        pose_config: str | None = None,
        pose_checkpoint: str | None = None,
        device: str | None = None,
        show_progress: bool = False,
        filter_args: dict[str, Any] | None = None,
        disable_internal_detector: bool = False,
        enabled: bool = True,
        onnx: dict[str, Any] | None = None,
        trt: dict[str, Any] | None = None,
        pose_checkpoint_sha256: str | None = None,
    ):
        super().__init__(enabled=enabled)
        self._pose_config = pose_config
        self._pose_checkpoint = pose_checkpoint
        self._pose_checkpoint_sha256 = pose_checkpoint_sha256
        self._device = device
        self._show_progress = bool(show_progress)
        self._filter_args = dict(filter_args or {})
        self._disable_internal_detector = bool(disable_internal_detector)
        self._inferencer = None
        # ONNX runtime/export options for pose (backbone+neck)
        self._onnx_cfg: dict[str, Any] = dict(onnx or {})
        self._onnx_runner: _OnnxPoseRunner | None = None
        # TensorRT options for pose (backbone+neck)
        self._trt_cfg: dict[str, Any] = dict(trt or {})
        self._trt_runner: _TrtPoseRunner | None = None

    def _default_filter_args(self, pose_config: str | None) -> dict[str, Any]:
        # Defaults taken from hmtrack CLI logic
        args = {"bbox_thr": 0.2, "nms_thr": 0.3, "pose_based_nms": False}
        if not pose_config:
            return args
        # Customize by model string
        spec = {
            "yoloxpose": {"bbox_thr": 0.01, "nms_thr": 0.65, "pose_based_nms": True},
            "rtmo": {"bbox_thr": 0.1, "nms_thr": 0.65, "pose_based_nms": True},
            "rtmp": {"kpt_thr": 0.3, "pose_based_nms": False, "disable_norm_pose_2d": False},
        }
        for k, v in spec.items():
            if k in pose_config:
                args.update(v)
                break
        return args

    @staticmethod
    def _wrap_model_test_step(model: torch.nn.Module | None, runner: Any | None) -> None:
        if model is None or runner is None:
            raise ValueError("An accelerated pose runner requires a pose model")
        if getattr(model, "_hm_test_step_wrapped", False):
            raise ValueError("The pose model already has an accelerator attached")
        orig_test_step = getattr(model, "test_step", None)
        if not callable(orig_test_step):
            raise TypeError("The pose model does not expose test_step")

        def _test_step(data, *args, **kwargs):
            preds = runner.forward(data)
            if not isinstance(preds, (list, tuple)):
                raise TypeError("Accelerated pose inference must return a prediction sequence")
            return list(preds)

        model._hm_test_step_wrapped = True
        model._hm_orig_test_step = orig_test_step
        model.test_step = _test_step

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if self._inferencer is None:
            from mmpose.apis.inferencers import MMPoseInferencer

            cfg = self._pose_config
            if (
                isinstance(cfg, str)
                and cfg
                and "://" not in cfg
                and not cfg.startswith("/")
                and cfg.endswith((".py", ".yaml", ".yml"))
            ):
                cfg = prepend_root_dir(cfg)
            ckpt = self._pose_checkpoint
            if isinstance(ckpt, str) and ckpt and "://" not in ckpt and not ckpt.startswith("/"):
                ckpt = prepend_root_dir(ckpt)
            if self._pose_checkpoint_sha256:
                from pathlib import Path

                from hmlib.models.detector_selection import download_pinned_checkpoint

                if not isinstance(ckpt, str) or not ckpt.startswith("https://"):
                    raise ValueError("A pinned pose checkpoint requires its HTTPS source URL")
                ckpt = download_pinned_checkpoint(
                    ckpt, self._pose_checkpoint_sha256, Path(ckpt).name
                )
            dev = self._device
            if dev is None:
                device_obj = context.get("device")
                if device_obj is not None:
                    dev = str(device_obj)
            # Skip initializing MMPose's own detector when requested or when
            # detections are already provided upstream.
            det_model = None
            if self._disable_internal_detector or bool(
                context.get("using_precalculated_detection", False)
            ):
                det_model = "whole_image"

            with numpy2_pickle_compat():
                inferencer = MMPoseInferencer(
                    pose2d=cfg,
                    pose2d_weights=ckpt,
                    device=dev,
                    det_model=det_model,
                    show_progress=self._show_progress,
                )
            # Merge default + provided filter args
            fa = self._default_filter_args(cfg)
            fa.update(self._filter_args)
            inferencer.filter_args = fa
            # An explicitly selected accelerator must initialize and attach
            # successfully before this factory publishes the inferencer.
            use_trt = bool(self._trt_cfg.get("enable", False))
            use_onnx = bool(self._onnx_cfg.get("enable", False))
            backend = "TensorRT" if use_trt else "ONNX"
            if use_trt or use_onnx:
                try:
                    pose_impl = inferencer.inferencer
                    if getattr(pose_impl.cfg, "data_mode", None) != "topdown":
                        raise ValueError(
                            f"{backend} pose acceleration requires a topdown pose model"
                        )
                    if use_trt:
                        self._trt_runner = _TrtPoseRunner(
                            model=pose_impl.model,
                            engine_path=str(self._trt_cfg.get("engine", "pose.engine")),
                            force_build=bool(self._trt_cfg.get("force_build", False)),
                            fp16=bool(self._trt_cfg.get("fp16", True)),
                            int8=bool(self._trt_cfg.get("int8", False)),
                            calib_frames=int(self._trt_cfg.get("calib_frames", 0)),
                            batch_size=int(self._trt_cfg.get("batch_size", 32)),
                        )
                        inferencer._hm_trt_runner = self._trt_runner
                        runner = self._trt_runner
                    else:
                        self._onnx_runner = _OnnxPoseRunner(
                            model=pose_impl.model,
                            onnx_path=str(self._onnx_cfg.get("path", "pose.onnx")),
                            force_export=bool(self._onnx_cfg.get("force_export", False)),
                            quantize_int8=bool(self._onnx_cfg.get("quantize_int8", False)),
                            calib_frames=int(self._onnx_cfg.get("calib_frames", 0)),
                        )
                        inferencer._hm_onnx_runner = self._onnx_runner
                        runner = self._onnx_runner
                    self._wrap_model_test_step(pose_impl.model, runner)
                    inferencer._hm_pose_runner = runner
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to initialize requested {backend} pose backend"
                    ) from exc
            self._inferencer = inferencer

        context["pose_inferencer"] = self._inferencer
        return {"pose_inferencer": self._inferencer}

    def input_keys(self):
        return {"device", "using_precalculated_detection"}

    def output_keys(self):
        return {"pose_inferencer"}


class _BackboneNeckWrapper(torch.nn.Module):
    """Expose backbone+neck forward for pose export.

    Returns a single feature map tensor or a tuple of tensors, depending on the model.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.neck = getattr(model, "neck", None)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats


class _OnnxPoseRunner:
    """Run pose model with ONNX Runtime for backbone+neck, then decode with PyTorch head.

    This integrates with PosePlugin's bypass path by attaching to the inferencer
    as `._hm_onnx_runner` so PosePlugin can optionally use it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        onnx_path: str,
        force_export: bool = False,
        quantize_int8: bool = False,
        calib_frames: int = 0,
    ):
        self.model = model
        self.onnx_path = onnx_path
        self.force_export = force_export
        self.quantize_int8 = bool(quantize_int8)
        self.calib_target = int(calib_frames or 0)
        self._calib_inputs: list[Any] = []
        self._quantized = str(onnx_path).lower().endswith(".int8.onnx")
        self._quantized_path = onnx_path if self._quantized else None
        if self._quantized and (force_export or not Path(onnx_path).is_file()):
            raise ValueError("Supply an existing INT8 pose model without force_export")
        if self.quantize_int8 and not self._quantized and self.calib_target <= 0:
            raise ValueError("Pose INT8 quantization requires positive calib_frames")
        self._ort_sess = None
        self._providers = None
        self._use_cuda = False
        self._fallback_test_step = None
        self._mean = None
        self._std = None
        self._mean_t = None
        self._std_t = None
        self._swap_rgb = False
        self._init_preprocess()
        self._export_and_create_session()

    def set_fallback_test_step(self, fn) -> None:
        self._fallback_test_step = fn

    def _init_preprocess(self) -> None:
        dp = getattr(self.model, "data_preprocessor", None)
        if dp is None:
            return
        mean = getattr(dp, "mean", None)
        std = getattr(dp, "std", None)
        if mean is not None and not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean)
        if std is not None and not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std)
        if isinstance(mean, torch.Tensor):
            if mean.ndim == 1:
                mean = mean.view(1, -1, 1, 1)
            elif mean.ndim == 3:
                mean = mean.unsqueeze(0)
            elif mean.ndim != 4:
                mean = None
        if isinstance(std, torch.Tensor):
            if std.ndim == 1:
                std = std.view(1, -1, 1, 1)
            elif std.ndim == 3:
                std = std.unsqueeze(0)
            elif std.ndim != 4:
                std = None
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            self._mean = mean.detach().cpu().numpy()
            self._std = std.detach().cpu().numpy()
            self._mean_t = mean.detach()
            self._std_t = std.detach()
        bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
        rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
        self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)

    @staticmethod
    def _ensure_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty((0,))
            tensors: list[torch.Tensor] = []
            for item in x:
                t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                if t.ndim == 4 and t.size(0) == 1:
                    t = t.squeeze(0)
                tensors.append(t)
            if len(tensors) == 1:
                t = tensors[0]
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                return t
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(x)

    def _preprocess(self, x: torch.Tensor) -> Any:
        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean is not None and self._std is not None:
            mean = torch.from_numpy(self._mean).to(device=x.device, dtype=x.dtype)
            std = torch.from_numpy(self._std).to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        x_cpu = x.detach().to("cpu")
        return x_cpu.numpy()

    def _preprocess_gpu(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean_t is not None and self._std_t is not None:
            mean = self._mean_t.to(device=x.device, dtype=x.dtype)
            std = self._std_t.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def _export_and_create_session(self) -> None:
        import os

        export_needed = self.force_export or (not os.path.exists(self.onnx_path))
        if export_needed:
            wrapper = _BackboneNeckWrapper(self.model).eval()
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            wrapper.to(device)
            # Use a typical person crop size; dynamic H/W enabled
            sample = torch.randn(1, 3, 256, 192, device=device, dtype=torch.float32)
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    sample,
                    self.onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["feats"],
                    dynamic_axes={
                        "images": {0: "batch", 2: "height", 3: "width"},
                        "feats": {0: "batch"},
                    },
                )
        import onnxruntime as ort  # type: ignore

        # Session options with higher optimization level
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._use_cuda = True
        else:
            self._providers = ["CPUExecutionProvider"]
            self._use_cuda = False
        self._ort_sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=self._providers)  # type: ignore

    def _run_feats(
        self, x: torch.Tensor, *, preprocessed: bool = False, collect_calibration: bool = True
    ) -> Any:
        import numpy as np

        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        need_calib = (
            collect_calibration
            and self.quantize_int8
            and not self._quantized
            and self.calib_target > 0
        )
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        if self._use_cuda and x.is_cuda:
            xc = x.contiguous() if preprocessed else self._preprocess_gpu(x)
            if need_calib and len(self._calib_inputs) < self.calib_target:
                x_cpu = xc.detach().to("cpu")
                remaining = max(0, self.calib_target - len(self._calib_inputs))
                take = min(int(x_cpu.shape[0]), remaining)
                for i in range(take):
                    self._calib_inputs.append(x_cpu[i : i + 1].numpy().copy())
            io = self._ort_sess.io_binding()
            input_name = self._ort_sess.get_inputs()[0].name
            io.bind_input(
                name=input_name,
                device_type="cuda",
                device_id=int(xc.device.index or 0),
                element_type=np.float32,
                shape=list(xc.shape),
                buffer_ptr=int(xc.data_ptr()),
            )
            for out in self._ort_sess.get_outputs():
                io.bind_output(out.name, device_type="cuda")
            self._ort_sess.run_with_iobinding(io)
            ort_outs = io.get_outputs()
            outs_t: list[torch.Tensor] = []
            try:
                from torch.utils.dlpack import from_dlpack  # type: ignore

                for ov in ort_outs:
                    outs_t.append(from_dlpack(ov.to_dlpack()).to(device=dev))
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.warning(
                    "Pose ONNX DLPack transfer unavailable; copying outputs through CPU: %s", exc
                )
                outs_t = [torch.from_numpy(ov.numpy()).to(device=dev) for ov in ort_outs]
            if len(outs_t) == 1:
                return outs_t[0]
            return outs_t
        # CPU fallback
        x_np = x.detach().cpu().numpy() if preprocessed else self._preprocess(x)
        if need_calib and len(self._calib_inputs) < self.calib_target:
            remaining = max(0, self.calib_target - len(self._calib_inputs))
            take = min(int(x_np.shape[0]), remaining)
            for i in range(take):
                self._calib_inputs.append(x_np[i : i + 1].copy())
        outputs = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: x_np})
        if len(outputs) == 1:
            out = torch.from_numpy(outputs[0]).to(device=dev)
            return out
        return [torch.from_numpy(o).to(device=dev) for o in outputs]

    def _maybe_quantize_after_calib(self) -> None:
        if not self.quantize_int8 or self._quantized:
            return
        if self.calib_target <= 0 or len(self._calib_inputs) < self.calib_target:
            return
        try:
            from onnxruntime.quantization import (  # type: ignore
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )
        except ImportError as exc:
            raise RuntimeError("Pose INT8 quantization requires onnxruntime.quantization") from exc
        input_name = self._ort_sess.get_inputs()[0].name

        class _Reader(CalibrationDataReader):  # type: ignore
            def __init__(self, name, samples):
                self.name, self._it = name, iter(samples)

            def get_next(self):
                try:
                    x = next(self._it)
                    return {self.name: x}
                except StopIteration:
                    return None

        q_path = str(Path(self.onnx_path).with_suffix(".int8.onnx"))
        try:
            quantize_static(
                model_input=self.onnx_path,
                model_output=q_path,
                calibration_data_reader=_Reader(input_name, self._calib_inputs),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            import onnxruntime as ort  # type: ignore

            self._ort_sess = ort.InferenceSession(q_path, providers=self._providers)
            self._quantized = True
            self._quantized_path = q_path
        except Exception as exc:
            raise RuntimeError(f"Pose INT8 quantization failed: {self.onnx_path}") from exc
        finally:
            self._calib_inputs = []

    def forward(self, proc_inputs: dict[str, Any]):
        """Run the requested ONNX backend and decode with the original pose head."""
        try:
            processed = self.model.data_preprocessor(proc_inputs, training=False)
            inputs = processed["inputs"]
            datas = processed["data_samples"]
            feats = self._run_feats(inputs, preprocessed=True)
            if self.model.test_cfg.get("flip_test", False):
                flipped = self._run_feats(
                    inputs.flip(-1), preprocessed=True, collect_calibration=False
                )
                feats = [feats, flipped]
            with torch.no_grad():
                preds = _decode_pose_predictions(self.model, feats, datas)
            self._maybe_quantize_after_calib()
            return preds
        except Exception as exc:
            raise RuntimeError(f"Requested ONNX pose execution failed: {self.onnx_path}") from exc


class _TrtPoseRunner:
    """Compile pose backbone+neck with Torch-TensorRT; decode with PyTorch."""

    def __init__(
        self,
        model: torch.nn.Module,
        engine_path: str,
        force_build: bool = False,
        fp16: bool = True,
        int8: bool = False,
        calib_frames: int = 0,
        batch_size: int = 32,
    ):
        self.model = model
        self.engine_path = engine_path
        self.force_build = bool(force_build)
        self.fp16 = bool(fp16)
        self.int8 = bool(int8)
        # Retain the constructor argument for existing Aspen configurations;
        # Torch-TensorRT's Dynamo frontend cannot consume legacy frame calibration.
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Torch-TensorRT pose batch size must be positive")
        self._trt_module = None
        self._trt_modules: dict[tuple[tuple[int, ...], torch.dtype, str], torch.nn.Module] = {}
        self._trt_input_buffers: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
        self._wrapper_module = None
        self._fallback_test_step = None
        self._mean = None
        self._std = None
        self._mean_t = None
        self._std_t = None
        self._swap_rgb = False
        self._init_preprocess()

    def set_fallback_test_step(self, fn) -> None:
        self._fallback_test_step = fn

    def _init_preprocess(self) -> None:
        dp = getattr(self.model, "data_preprocessor", None)
        if dp is None:
            return
        mean = getattr(dp, "mean", None)
        std = getattr(dp, "std", None)
        if mean is not None and not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean)
        if std is not None and not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std)
        if isinstance(mean, torch.Tensor):
            if mean.ndim == 1:
                mean = mean.view(1, -1, 1, 1)
            elif mean.ndim == 3:
                mean = mean.unsqueeze(0)
            elif mean.ndim != 4:
                mean = None
        if isinstance(std, torch.Tensor):
            if std.ndim == 1:
                std = std.view(1, -1, 1, 1)
            elif std.ndim == 3:
                std = std.unsqueeze(0)
            elif std.ndim != 4:
                std = None
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            self._mean = mean.detach().cpu()
            self._std = std.detach().cpu()
            self._mean_t = mean.detach()
            self._std_t = std.detach()
        bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
        rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
        self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)

    @staticmethod
    def _ensure_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty((0,))
            tensors: list[torch.Tensor] = []
            for item in x:
                t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                if t.ndim == 4 and t.size(0) == 1:
                    t = t.squeeze(0)
                tensors.append(t)
            if len(tensors) == 1:
                t = tensors[0]
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                return t
            return torch.stack(tensors, dim=0)
        return torch.as_tensor(x)

    def _prepare_inputs(self, x: Any, *, preprocessed: bool = False) -> torch.Tensor:
        x = self._ensure_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.device != dev:
            x = x.to(device=dev, non_blocking=True)
        if not preprocessed and self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if not preprocessed and self._mean_t is not None and self._std_t is not None:
            mean = self._mean_t.to(device=x.device, dtype=x.dtype)
            std = self._std_t.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def _static_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pad one pose chunk to the configured, engine-specialized batch size."""
        if inputs.size(0) > self.batch_size:
            raise ValueError(
                f"Pose chunk has {inputs.size(0)} inputs, exceeding batch size {self.batch_size}"
            )
        if inputs.size(0) == self.batch_size:
            return inputs

        shape = (self.batch_size, *inputs.shape[1:])
        key = (shape, inputs.dtype, str(inputs.device))
        buffer = self._trt_input_buffers.get(key)
        if buffer is None:
            buffer = inputs.new_zeros(shape)
            self._trt_input_buffers[key] = buffer
        else:
            buffer.zero_()
        if inputs.size(0) > 0:
            buffer[: inputs.size(0)].copy_(inputs)
        return buffer

    def _ensure_trt_engine(self, sample: torch.Tensor) -> None:
        sample = self._static_batch(sample[: self.batch_size])
        cache_key = (tuple(sample.shape), sample.dtype, str(sample.device))
        cached_module = self._trt_modules.get(cache_key)
        if cached_module is not None:
            self._trt_module = cached_module
            return
        try:
            try:
                dev = next(self.model.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wrapper = _BackboneNeckWrapper(self.model).eval().to(dev)
            self._wrapper_module = wrapper
            with torch.inference_mode():
                compiled = compile_torch_tensorrt(
                    wrapper,
                    [sample],
                    engine_path=self.engine_path,
                    force_build=self.force_build,
                    fp16=self.fp16,
                    int8=self.int8,
                )
            self._trt_modules[cache_key] = compiled
            self._trt_module = compiled
        except Exception as exc:
            raise RuntimeError(
                f"Requested TensorRT pose compilation failed: {self.engine_path}"
            ) from exc

    @staticmethod
    def _slice_feature_batch(feats: Any, valid_count: int, *, clone: bool = False) -> Any:
        if isinstance(feats, torch.Tensor):
            result = feats[:valid_count]
            return result.clone() if clone else result
        if isinstance(feats, tuple):
            return tuple(
                _TrtPoseRunner._slice_feature_batch(value, valid_count, clone=clone)
                for value in feats
            )
        if isinstance(feats, list):
            return [
                _TrtPoseRunner._slice_feature_batch(value, valid_count, clone=clone)
                for value in feats
            ]
        raise TypeError(f"Unexpected pose feature output type: {type(feats).__name__}")

    def _predict_trt_chunks(self, inputs: torch.Tensor, datas: Any):
        predictions: list[Any] = []
        assert self._trt_module is not None
        for start in range(0, inputs.size(0), self.batch_size):
            chunk = inputs[start : start + self.batch_size]
            static_chunk = self._static_batch(chunk)
            try:
                with torch.inference_mode():
                    feats = self._trt_module(static_chunk)
                    flipped = None
                    if self.model.test_cfg.get("flip_test", False):
                        # Torch-TensorRT reuses output buffers. Keep the normal
                        # features before the mirrored pass overwrites them.
                        feats = self._slice_feature_batch(feats, chunk.size(0), clone=True)
                        flipped = self._trt_module(static_chunk.flip(-1))
            except Exception as exc:
                raise RuntimeError(
                    f"Requested TensorRT pose execution failed: {self.engine_path}"
                ) from exc
            feats = self._slice_feature_batch(feats, chunk.size(0))
            if flipped is not None:
                feats = [feats, self._slice_feature_batch(flipped, chunk.size(0))]
            with torch.inference_mode():
                chunk_predictions = _decode_pose_predictions(
                    self.model,
                    feats,
                    datas[start : start + chunk.size(0)],
                )
            if not isinstance(chunk_predictions, (list, tuple)):
                raise TypeError(
                    "Pose head.predict must return a sequence for batched TensorRT inference"
                )
            predictions.extend(chunk_predictions)
        return predictions

    def forward(self, proc_inputs: dict[str, Any]):
        processed = self.model.data_preprocessor(proc_inputs, training=False)
        inputs = self._prepare_inputs(processed["inputs"], preprocessed=True)
        datas = processed["data_samples"]
        if inputs.size(0) == 0:
            return []
        self._ensure_trt_engine(inputs)
        try:
            return self._predict_trt_chunks(inputs, datas)
        except Exception as exc:
            raise RuntimeError(
                f"Requested TensorRT pose inference failed: {self.engine_path}"
            ) from exc
