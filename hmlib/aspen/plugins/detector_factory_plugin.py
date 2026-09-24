from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.structures import InstanceData

from hmlib.config import prepend_root_dir
from hmlib.log import get_logger
from hmlib.utils.cuda_graph import CudaGraphCallable
from hmlib.utils.nms import DetectorNMS
from hmlib.utils.numpy_pickle_compat import numpy2_pickle_compat
from hmlib.utils.torch_tensorrt_runtime import compile_torch_tensorrt

from .base import Plugin

logger = get_logger(__name__)


def _is_rocm_runtime() -> bool:
    """Return True when running on a HIP/ROCm PyTorch build."""
    try:
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def _runtime_supports_tensorrt() -> bool:
    """TensorRT is only supported on NVIDIA CUDA builds, not ROCm."""
    try:
        return not _is_rocm_runtime() and bool(getattr(torch.version, "cuda", None))
    except Exception:
        return False


def _runtime_supports_torchvision_nms() -> bool:
    """ROCm currently segfaults in torchvision.ops.nms for this detector path."""
    return not _is_rocm_runtime()


def _resolve_detector_nms_backend(requested_backend: str, *, owner: str) -> str:
    """Resolve detector NMS to a backend that is valid for the active runtime."""
    backend = str(requested_backend or "trt").lower()
    if backend == "trt" and not _runtime_supports_tensorrt():
        runtime = "ROCm/HIP" if _is_rocm_runtime() else "non-CUDA"
        logger.warning(
            "%s requested TensorRT NMS on a %s runtime; using bbox-head NMS instead.",
            owner,
            runtime,
        )
        return "head"
    if backend == "torchvision" and not _runtime_supports_torchvision_nms():
        logger.warning(
            "%s requested torchvision NMS on ROCm/HIP; using bbox-head NMS instead "
            "because torchvision.ops.nms is unstable on this runtime.",
            owner,
        )
        return "head"
    return backend


def _detector_nms_compare_backends(primary_backend: str, nms_test: bool) -> List[str]:
    if not nms_test:
        return []
    if primary_backend == "trt" and _runtime_supports_torchvision_nms():
        return ["torchvision"]
    return []


def _strip_static_padding(instances: InstanceData, strip: bool) -> InstanceData:
    """Remove padded detections when static outputs are enabled."""
    if instances is None or not strip:
        return instances
    num_valid = getattr(instances, "num_valid_after_nms", getattr(instances, "num_valid", None))
    if isinstance(num_valid, torch.Tensor):
        if num_valid.numel() == 0:
            return instances[:0]
        if num_valid.device.type == "cpu":
            try:
                keep = int(num_valid)
            except Exception:
                return instances
        else:
            device = None
            for attr in ("bboxes", "scores", "labels"):
                try:
                    tensor = getattr(instances, attr)
                except Exception:
                    tensor = None
                if torch.is_tensor(tensor):
                    device = tensor.device
                    break
            if device is None:
                return instances
            keep_mask = torch.arange(len(instances), device=device) < num_valid.to(device=device)
            try:
                return instances[keep_mask]
            except Exception:
                return instances
    else:
        try:
            keep = int(num_valid)
        except Exception:
            return instances
    if keep < 0:
        keep = 0
    if keep >= len(instances):
        return instances
    return instances[:keep]


class DetectorFactoryPlugin(Plugin):
    """
    Builds and caches a pure detection model (e.g., YOLOX) from Aspen YAML.

    Params:
      - detector: dict model config (optional)
      - detector_yaml: str path to YAML with a top-level detector definition (optional)
      - data_preprocessor: dict (optional), passed to model as TrackDataPreprocessor-compatible
      - to_device: bool (default True)

    Outputs in context:
      - detector_model: the constructed detection model (eval mode)
    """

    def __init__(
        self,
        detector: Optional[Dict[str, Any]] = None,
        detector_yaml: Optional[str] = None,
        data_preprocessor: Optional[Dict[str, Any]] = None,
        to_device: bool = True,
        enabled: bool = True,
        onnx: Optional[Dict[str, Any]] = None,
        trt: Optional[Dict[str, Any]] = None,
        static_detections: Optional[Dict[str, Any]] = None,
        # Optional NMS configuration applied across backends (PyTorch, ONNX, TRT).
        # When not provided, defaults mirror the CLI flags in hm_opts (--detector-nms-backend
        # and --detector-trt-nms-plugin), with TensorRT EfficientNMS as the default.
        nms_backend: str = "trt",
        nms_test: bool = False,
        nms_plugin: str = "efficient",
        cuda_graph: bool = True,
        checkpoint: Optional[str] = None,
    ):
        super().__init__(enabled=enabled)
        self._detector_dict = detector
        self._detector_yaml = detector_yaml
        self._checkpoint = checkpoint
        self._data_preprocessor = data_preprocessor
        self._to_device = to_device
        self._model = None
        # ONNX runtime/export options
        self._onnx_cfg: Dict[str, Any] = dict(onnx or {})
        self._onnx_wrapper: Optional[_OnnxDetectorWrapper] = None
        # TensorRT options
        self._trt_cfg: Dict[str, Any] = dict(trt or {})
        self._trt_wrapper: Optional[_TrtDetectorWrapper] = None
        # Optional static detection outputs (fixed-shape top-k)
        try:
            trt_static = dict(self._trt_cfg.get("static_detections", {}) or {})
        except Exception:
            trt_static = {}
        self._static_detections: Dict[str, Any] = dict(trt_static)
        if static_detections:
            self._static_detections.update(static_detections)
        # Unified NMS configuration shared across all detector backends.
        self._nms_backend: str = _resolve_detector_nms_backend(
            nms_backend, owner="DetectorFactoryPlugin"
        )
        self._nms_test: bool = bool(nms_test)
        self._nms_plugin: str = str(nms_plugin or "efficient")
        self._cuda_graph_enabled: bool = bool(cuda_graph)
        self._logged_trt_unavailable: bool = False
        # Optional cached wrapper for pure-PyTorch YOLOX detectors so we don't
        # rebuild the wrapper on every forward().
        self._torch_wrapper: Optional[_TorchDetectorWrapper] = None
        # Fully-resolved runtime detector (PyTorch, ONNX, or TRT). This is
        # selected once after the base model is built and reused for all
        # subsequent forward() calls.
        self._runtime_detector: Optional[Any] = None
        self._pass: int = 0

    def _load_detector_from_yaml(self, path: str) -> Dict[str, Any]:
        import yaml

        with open(prepend_root_dir(path), "r") as f:
            y = yaml.safe_load(f)
        if isinstance(y, dict) and "detector" in y:
            return y["detector"]
        return y

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if bool(context.get("using_precalculated_detection", False)):
            # Skip building a detector if detections are provided externally
            return {}
        if self._model is None:
            from mmengine.config import ConfigDict
            from mmengine.registry import DefaultScope
            from mmyolo.registry import MODELS

            if MODELS is None:
                raise RuntimeError("mmdet MODELS registry is unavailable; cannot build detector.")

            detector_cfg = self._detector_dict
            if detector_cfg is None and self._detector_yaml:
                detector_cfg = self._load_detector_from_yaml(self._detector_yaml)

            # Convert to ConfigDict recursively for mmengine
            def _to_cfg(x):
                if isinstance(x, dict):
                    return ConfigDict({k: _to_cfg(v) for k, v in x.items()})
                if isinstance(x, list):
                    return [_to_cfg(v) for v in x]
                return x

            model_cfg = _to_cfg(detector_cfg)
            if self._checkpoint:
                model_cfg["init_cfg"] = None
            if self._data_preprocessor is not None:
                # Attach data_preprocessor to the model config before build
                if not isinstance(model_cfg, dict):
                    # Convert to dict to inject
                    model_cfg = ConfigDict(model_cfg)
                model_cfg["data_preprocessor"] = _to_cfg(self._data_preprocessor)

            if "yolodetector" in model_cfg.get("type", "").lower():
                # Ensure we have mmyolo installed for YOLOv8 support
                try:
                    pass  # type: ignore
                except Exception as ex:
                    raise RuntimeError(
                        "mmyolo is required for YOLOv8 models but is not installed."
                    ) from ex

            model_type = str(model_cfg.get("type", "")).lower()
            model_scope = (
                DefaultScope.overwrite_default_scope("mmyolo")
                if "mmyolo" in model_type or model_type == "yolodetector"
                else nullcontext()
            )
            with model_scope:
                model = MODELS.build(model_cfg)

            if hasattr(model, "init_weights"):
                with numpy2_pickle_compat():
                    model.init_weights()

            if self._checkpoint:
                from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

                with numpy2_pickle_compat():
                    checkpoint = _load_checkpoint(self._checkpoint, map_location="cpu")
                weights = checkpoint.get("state_dict", checkpoint)
                if any(key.startswith("student.") for key in weights):
                    weights = {
                        key.removeprefix("student."): value
                        for key, value in weights.items()
                        if key.startswith("student.")
                    }
                load_state_dict(model, weights, strict=True)
                logger.info("Loaded detector checkpoint: %s", self._checkpoint)

            # Enable static-shape detection outputs whenever the head supports
            # it (e.g., YOLOXHead). This avoids dynamic mask-based selects and
            # plays nicely with the TensorRT NMS path.
            head = getattr(model, "bbox_head", None)
            setter = getattr(head, "set_static_detections", None)
            if callable(setter):
                if self._static_detections:
                    setter(**self._static_detections)
                else:
                    # Defaults: enable static detections and, when possible,
                    # mirror max_per_img from test_cfg.
                    static_kwargs: Dict[str, Any] = {"enable": True}
                    try:
                        test_cfg = getattr(head, "test_cfg", None)
                        if test_cfg is not None:
                            max_per_img = int(test_cfg.get("max_per_img", 0) or 0)
                        else:
                            max_per_img = 0
                    except Exception:
                        max_per_img = 0
                    if max_per_img > 0:
                        static_kwargs["max_detections"] = max_per_img
                    setter(**static_kwargs)

            if (
                self._to_device
                and "device" in context
                and isinstance(context["device"], torch.device)
            ):
                model = model.to(context["device"])  # type: ignore[assignment]
            model.eval()
            self._model = model

        # Decide backend: TensorRT (highest), ONNX, or PyTorch exactly once
        # after the base model is constructed. Subsequent forward() calls reuse
        # the resolved runtime detector to avoid per-step backend selection.
        if self._runtime_detector is None:
            detector_model: Any = self._model

            # Prefer TensorRT backbone+neck when explicitly enabled.
            use_trt_requested = bool(self._trt_cfg.get("enable", False))
            use_trt = use_trt_requested and _runtime_supports_tensorrt()
            if use_trt_requested and not use_trt and not self._logged_trt_unavailable:
                runtime = "ROCm/HIP" if _is_rocm_runtime() else "non-CUDA"
                logger.warning(
                    "TensorRT detector path is enabled, but TensorRT is unavailable on this %s "
                    "runtime; using the PyTorch detector path instead.",
                    runtime,
                )
                self._logged_trt_unavailable = True
            if use_trt:
                if self._trt_wrapper is None:
                    try:
                        self._trt_wrapper = _TrtDetectorWrapper(
                            model=self._model,
                            engine_path=str(
                                self._trt_cfg.get("engine")
                                or Path(str(context.get("work_dir") or ".")) / "detector.engine"
                            ),
                            force_build=bool(self._trt_cfg.get("force_build", False)),
                            fp16=bool(self._trt_cfg.get("fp16", True)),
                            int8=bool(self._trt_cfg.get("int8", False)),
                            calib_frames=int(self._trt_cfg.get("calib_frames", 0)),
                            profiler=self._profiler,
                            nms_backend=_resolve_detector_nms_backend(
                                str(self._trt_cfg.get("nms_backend", self._nms_backend)),
                                owner="TensorRT detector wrapper",
                            ),
                            nms_test=bool(self._trt_cfg.get("nms_test", self._nms_test)),
                            nms_plugin=str(self._trt_cfg.get("nms_plugin", self._nms_plugin)),
                        )
                    except Exception as ex:
                        # Fall back to next option if TRT init fails
                        logger.warning("Failed to initialize TensorRT detector wrapper: %s", ex)
                        self._trt_wrapper = None
                if self._trt_wrapper is not None:
                    detector_model = self._trt_wrapper

            # If no TensorRT wrapper is active, consider ONNX backbone+neck.
            if detector_model is self._model:
                use_onnx = bool(self._onnx_cfg.get("enable", False))
                if use_onnx:
                    if self._onnx_wrapper is None:
                        try:
                            self._onnx_wrapper = _OnnxDetectorWrapper(
                                model=self._model,
                                onnx_path=str(self._onnx_cfg.get("path", "detector.onnx")),
                                force_export=bool(self._onnx_cfg.get("force_export", False)),
                                quantize_int8=bool(self._onnx_cfg.get("quantize_int8", False)),
                                calib_frames=int(self._onnx_cfg.get("calib_frames", 0)),
                                game_id=str(context.get("game_id", "")),
                                profiler=self._profiler,
                                nms_backend=_resolve_detector_nms_backend(
                                    str(self._onnx_cfg.get("nms_backend", self._nms_backend)),
                                    owner="ONNX detector wrapper",
                                ),
                                nms_test=bool(self._onnx_cfg.get("nms_test", self._nms_test)),
                                nms_plugin=str(self._onnx_cfg.get("nms_plugin", self._nms_plugin)),
                            )
                        except Exception as ex:
                            # Fall back to PyTorch if ONNX init fails
                            logger.warning("Failed to initialize ONNX detector wrapper: %s", ex)
                            self._onnx_wrapper = None
                    if self._onnx_wrapper is not None:
                        detector_model = self._onnx_wrapper

            # Default: PyTorch detector. For YOLOX-style heads, wrap the model so
            # that we can reuse the DetectorNMS path (TensorRT / static shapes)
            # even when backbone+neck are running in pure PyTorch.
            if detector_model is self._model:
                try:
                    bbox_head = getattr(self._model, "bbox_head", None)
                    if bbox_head is not None and bbox_head.__class__.__name__ == "YOLOXHead":
                        if self._torch_wrapper is None:
                            self._torch_wrapper = _TorchDetectorWrapper(
                                model=self._model,
                                profiler=self._profiler,
                                nms_backend=self._nms_backend,
                                nms_test=self._nms_test,
                                nms_plugin=self._nms_plugin,
                                cuda_graph=self._cuda_graph_enabled,
                            )
                        else:
                            self._torch_wrapper.set_cuda_graph_enabled(self._cuda_graph_enabled)
                        detector_model = self._torch_wrapper
                except Exception:
                    detector_model = self._model

            self._runtime_detector = detector_model

        context["detector_model"] = self._runtime_detector
        return {"detector_model": self._runtime_detector}

    def input_keys(self):
        return {"device", "using_precalculated_detection", "game_id", "work_dir"}

    def output_keys(self):
        return {"detector_model"}

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if self._torch_wrapper is not None:
            self._torch_wrapper.set_cuda_graph_enabled(enabled)
        runtime = self._runtime_detector
        if hasattr(runtime, "set_cuda_graph_enabled"):
            runtime.set_cuda_graph_enabled(enabled)
        return True


class _BackboneNeckWrapper(torch.nn.Module):
    """Thin wrapper that exposes backbone+neck forward for export.

    Returns three feature maps (small->large stride) as individual outputs.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        feats = self.backbone(x)
        feats = self.neck(feats)
        if isinstance(feats, (list, tuple)):
            assert len(feats) >= 3, "Expected at least 3 FPN levels"
            return feats[0], feats[1], feats[2]
        # Some necks may return a single tensor; mirror into three
        return feats, feats, feats


class _ProfilerMixin:
    def __init__(self, profiler: Optional[Any], label: str):
        self._profiler = profiler
        self._profile_label = label

    def _profile_scope(self, suffix: Optional[str] = None):
        if self._profiler is None:
            return nullcontext()
        name = self._profile_label if suffix is None else f"{self._profile_label}.{suffix}"
        return self._profiler.rf(name)


class _OnnxDetectorWrapper(_ProfilerMixin):
    """Wraps YOLOX detector to run backbone+neck in ONNX Runtime, then decodes
    with the existing PyTorch YOLOX head to produce InstanceData results.

    - Exposes a `.predict(images, [data_sample])` compatible with mmdet
      SingleStageDetector.predict() return type expectations used by
      DetectorInferencePlugin (list with object having .pred_instances).
    - Optional on-the-fly static INT8 quantization after collecting N frames.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        onnx_path: str,
        force_export: bool = False,
        quantize_int8: bool = False,
        calib_frames: int = 0,
        game_id: str = "",
        profiler: Optional[Any] = None,
        nms_backend: str = "trt",
        nms_test: bool = False,
        nms_plugin: str = "efficient",
    ):
        super().__init__(profiler=profiler, label="onnx_predictor")
        self.model = model
        self.onnx_path = onnx_path
        self.force_export = force_export
        self.quantize_int8 = bool(quantize_int8)
        self.calib_target = int(calib_frames or 0)
        self._calib_inputs: List[Any] = []
        self._quantized = False
        self._quantized_path: Optional[str] = None
        self._use_cuda = False
        # NMS configuration and dispatcher (mirrors TensorRT wrapper).
        self._nms_backend: str = _resolve_detector_nms_backend(
            nms_backend, owner="ONNX detector wrapper"
        )
        self._nms_test: bool = bool(nms_test)
        self._nms_plugin: str = str(nms_plugin or "efficient")
        compare_backends = _detector_nms_compare_backends(self._nms_backend, self._nms_test)
        self._nms = DetectorNMS(
            bbox_head=getattr(model, "bbox_head", model),
            backend=self._nms_backend,
            compare_backends=compare_backends,
            trt_plugin=self._nms_plugin,
        )

        # Gather preproc config
        self._mean = None
        self._std = None
        self._mean_t: Optional[torch.Tensor] = None
        self._std_t: Optional[torch.Tensor] = None
        self._swap_rgb = False
        try:
            dp = getattr(model, "data_preprocessor", None)
            if dp is not None:
                mean = getattr(dp, "mean", None)
                std = getattr(dp, "std", None)
                # TrackDataPreprocessor may store (1,C,1,1)
                if isinstance(mean, torch.Tensor):
                    if mean.ndim == 1:
                        self._mean = mean.view(1, -1, 1, 1).detach().cpu().numpy()
                        self._mean_t = mean.view(1, -1, 1, 1).detach()
                    elif mean.ndim == 4:
                        self._mean = mean.detach().cpu().numpy()
                        self._mean_t = mean.detach()
                if isinstance(std, torch.Tensor):
                    if std.ndim == 1:
                        self._std = std.view(1, -1, 1, 1).detach().cpu().numpy()
                        self._std_t = std.view(1, -1, 1, 1).detach()
                    elif std.ndim == 4:
                        self._std = std.detach().cpu().numpy()
                        self._std_t = std.detach()
                bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
                rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
                self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)
        except Exception:
            pass

        # Export ONNX if necessary
        self._ensure_onnx_export()
        # Create an ORT session
        self._create_session(self.onnx_path)

    def _ensure_onnx_export(self) -> None:
        import os

        export_needed = self.force_export or (not os.path.exists(self.onnx_path))
        if not export_needed:
            return
        # Prepare export module and sample input (dynamic axes set)
        wrapper = _BackboneNeckWrapper(self.model).eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        wrapper.to(device)
        # Choose a reasonable default input size; graph is dynamic on H/W
        sample = torch.randn(1, 3, 480, 1312, device=device, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample,
                self.onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["feat0", "feat1", "feat2"],
                dynamic_axes={
                    "images": {0: "batch", 2: "height", 3: "width"},
                    "feat0": {0: "batch", 2: "h0", 3: "w0"},
                    "feat1": {0: "batch", 2: "h1", 3: "w1"},
                    "feat2": {0: "batch", 2: "h2", 3: "w2"},
                },
            )

    def _create_session(self, path: str) -> None:
        import onnxruntime as ort  # type: ignore

        # Session options for maximum graph optimizations
        so = ort.SessionOptions()
        try:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        providers = []
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._use_cuda = True
            else:
                providers = ["CPUExecutionProvider"]
                self._use_cuda = False
        except Exception:
            providers = ["CPUExecutionProvider"]
            self._use_cuda = False
        self._ort_providers = providers
        self._ort_sess = ort.InferenceSession(path, sess_options=so, providers=providers)  # type: ignore

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
        except Exception:
            # Quantization not available; skip
            self._calib_inputs = []
            return

        # Create a simple calibration data reader
        input_name = self._ort_sess.get_inputs()[0].name

        class _Reader(CalibrationDataReader):  # type: ignore
            def __init__(self, name: str, samples: List[Any]):
                self.name = name
                self._iter = iter(samples)

            def get_next(self):  # type: ignore
                try:
                    x = next(self._iter)
                    return {self.name: x}
                except StopIteration:
                    return None

        reader = _Reader(input_name, self._calib_inputs)
        q_path = self.onnx_path.replace(".onnx", ".int8.onnx")
        try:
            quantize_static(
                model_input=self.onnx_path,
                model_output=q_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            # Reload session with the quantized model
            self._create_session(q_path)
            self._quantized = True
            self._quantized_path = q_path
        except Exception:
            # If quant fails, continue with fp32
            pass
        finally:
            self._calib_inputs = []

    def _preprocess(self, x: torch.Tensor) -> Any:
        # x: (1, C, H, W) torch tensor on any device
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean is not None and self._std is not None:
            mean = torch.from_numpy(self._mean).to(device=x.device, dtype=x.dtype)
            std = torch.from_numpy(self._std).to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        # ORT expects numpy on CPU
        x_cpu = x.detach().to("cpu")
        return x_cpu.numpy()

    def _preprocess_gpu(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean_t is not None and self._std_t is not None:
            mean = self._mean_t.to(device=x.device, dtype=x.dtype)
            std = self._std_t.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def _run_ort(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run ORT either via CUDA I/O binding (if available) or CPU numpy path.
        Returns a list of torch tensors on the model device.
        """
        import numpy as np

        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        if self._use_cuda and x.is_cuda:
            with self._profile_scope("preprocess"):
                xc = self._preprocess_gpu(x)
            with self._profile_scope("engine"):
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
            with self._profile_scope("postprocess"):
                outs_t: List[torch.Tensor] = []
                try:
                    from torch.utils.dlpack import from_dlpack  # type: ignore

                    for ov in ort_outs:
                        outs_t.append(from_dlpack(ov.to_dlpack()).to(device=dev))
                except Exception:
                    for ov in ort_outs:
                        np_arr = np.array(ov)
                        outs_t.append(torch.from_numpy(np_arr).to(device=dev))
                return outs_t
        with self._profile_scope("preprocess"):
            x_np = self._preprocess(x)
        with self._profile_scope("engine"):
            outs = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: x_np})
        with self._profile_scope("postprocess"):
            return [torch.from_numpy(o).to(device=dev) for o in outs]

    def predict(self, imgs: torch.Tensor, data_samples: List[Any]):  # type: ignore[override]
        with self._profile_scope():
            assert isinstance(data_samples, (list, tuple)) and len(data_samples) == imgs.size(0)
            N = imgs.size(0)
            if self.quantize_int8 and not self._quantized and self.calib_target > 0:
                remaining = max(0, self.calib_target - len(self._calib_inputs))
                if remaining > 0:
                    take = min(N, remaining)
                    with self._profile_scope("calibration"):
                        for i in range(take):
                            x_np = self._preprocess(imgs[i : i + 1])
                            self._calib_inputs.append(x_np.copy())
            outs = self._run_ort(imgs)
            feats: List[torch.Tensor] = [o for o in outs[:3]]
            with torch.inference_mode():
                with self._profile_scope("head"):
                    cls_scores, bbox_preds, objectnesses = self.model.bbox_head(tuple(feats))
                    metas: List[Dict[str, Any]] = []
                    for i in range(N):
                        try:
                            metas.append(getattr(data_samples[i], "metainfo", {}))
                        except Exception:
                            metas.append({})
                    # Decode boxes/scores but defer NMS to DetectorNMS so that
                    # the same TensorRT/static-shape path can be used across
                    # ONNX and pure PyTorch backends.
                    result_list: List[InstanceData] = self.model.bbox_head.predict_by_feat(
                        cls_scores=cls_scores,
                        bbox_preds=bbox_preds,
                        objectnesses=objectnesses,
                        batch_img_metas=metas,
                        cfg=None,
                        rescale=True,
                        with_nms=False,
                    )
            results: List[Any] = []
            with self._profile_scope("postprocess"):
                self._maybe_quantize_after_calib()
                for inst, meta in zip(result_list, metas):
                    # Apply configured NMS backend (TensorRT by default).
                    with self._profile_scope("nms"):
                        inst = self._nms.run_single(inst, meta)

                    class _Wrap:
                        def __init__(self, inst_):
                            self.pred_instances = inst_

                    results.append(
                        _Wrap(_strip_static_padding(inst, strip=self._nms_backend != "trt"))
                    )
            return results


class _TrtDetectorWrapper(_ProfilerMixin):
    """Compile backbone+neck with Torch-TensorRT and decode with the PyTorch head.

    Engines are specialized lazily from the first input shape and persisted in
    Torch-TensorRT's disk cache.  The wrapper exposes the ``predict`` interface
    expected by downstream mmdetection code.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        engine_path: str,
        force_build: bool = False,
        fp16: bool = True,
        int8: bool = False,
        calib_frames: int = 0,
        profiler: Optional[Any] = None,
        nms_backend: str = "trt",
        nms_test: bool = False,
        nms_plugin: str = "efficient",
    ):
        super().__init__(profiler=profiler, label="trt_predictor")
        self.model = model
        self.engine_path = Path(engine_path)
        self.force_build = bool(force_build)
        self.fp16 = bool(fp16)
        self.int8 = bool(int8)
        # Retain the constructor argument for existing Aspen configurations;
        # Torch-TensorRT's Dynamo frontend cannot consume legacy frame calibration.
        self._trt_module = None
        self._trt_modules: Dict[Tuple[Tuple[int, ...], torch.dtype, str], torch.nn.Module] = {}
        self._wrapper_module: Optional[_BackboneNeckWrapper] = None
        self._mean = None
        self._std = None
        self._swap_rgb = False
        self._pass: int = 0
        self._trt_disabled_reason: Optional[str] = None
        self._nms_backend: str = _resolve_detector_nms_backend(
            nms_backend, owner="TensorRT detector wrapper"
        )
        self._nms_test: bool = bool(nms_test)
        self._nms_plugin: str = str(nms_plugin or "efficient")
        compare_backends = _detector_nms_compare_backends(self._nms_backend.lower(), self._nms_test)
        # Centralized NMS dispatcher; handles 'trt', 'torchvision', and 'head'.
        self._nms = DetectorNMS(
            bbox_head=getattr(model, "bbox_head", model),
            backend=self._nms_backend,
            compare_backends=compare_backends,
            trt_plugin=self._nms_plugin,
        )

        # Gather data_preprocessor normalization like ONNX path
        try:
            dp = getattr(model, "data_preprocessor", None)
            if dp is not None:
                mean = getattr(dp, "mean", None)
                std = getattr(dp, "std", None)
                if mean is not None:
                    if not isinstance(mean, torch.Tensor):
                        try:
                            mean = torch.as_tensor(mean)
                        except Exception:
                            mean = None
                    if isinstance(mean, torch.Tensor):
                        if mean.ndim == 1:
                            self._mean = mean.view(1, -1, 1, 1).detach().cpu()
                        elif mean.ndim == 4:
                            self._mean = mean.detach().cpu()
                if std is not None:
                    if not isinstance(std, torch.Tensor):
                        try:
                            std = torch.as_tensor(std)
                        except Exception:
                            std = None
                    if isinstance(std, torch.Tensor):
                        if std.ndim == 1:
                            self._std = std.view(1, -1, 1, 1).detach().cpu()
                        elif std.ndim == 4:
                            self._std = std.detach().cpu()
                bgr_to_rgb = bool(getattr(dp, "bgr_to_rgb", False))
                rgb_to_bgr = bool(getattr(dp, "rgb_to_bgr", False))
                self._swap_rgb = bool(bgr_to_rgb or rgb_to_bgr)
        except Exception as exc:
            logger.warning(
                "Could not read detector preprocessing metadata; TensorRT inputs "
                "will not be normalized by the wrapper: %s",
                exc,
            )

    def _ensure_trt_engine(self, sample: torch.Tensor) -> None:
        cache_key = (tuple(sample.shape), sample.dtype, str(sample.device))
        cached_module = self._trt_modules.get(cache_key)
        if cached_module is not None:
            self._trt_module = cached_module
            return
        dev = next(self.model.parameters()).device
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

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize similar to data_preprocessor if present
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._swap_rgb and x.size(1) == 3:
            x = x[:, [2, 1, 0], :, :]
        if self._mean is not None and self._std is not None:
            mean = self._mean.to(device=x.device, dtype=x.dtype)
            std = self._std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x.contiguous()

    def predict(self, imgs: torch.Tensor, data_samples: List[Any]):  # type: ignore[override]
        do_trace: bool = self._pass == 10
        if do_trace:
            pass
        with self._profile_scope():
            assert isinstance(data_samples, (list, tuple)) and len(data_samples) == imgs.size(0)
            try:
                dev = next(self.model.parameters()).device
            except StopIteration:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_metas = [getattr(data_sample, "metainfo", {}) for data_sample in data_samples]
            with self._profile_scope("preprocess"):
                x = imgs.to(device=dev, non_blocking=True)
                x = self._preprocess(x)
            with self._profile_scope("engine"):
                if self._trt_disabled_reason is None:
                    try:
                        self._ensure_trt_engine(x)
                    except Exception as ex:
                        self._disable_trt(ex)
                if self._trt_module is not None:
                    try:
                        feats = self._trt_module(x)
                    except Exception as ex:
                        self._disable_trt(ex)
                        assert self._wrapper_module is not None
                        with torch.inference_mode():
                            feats = self._wrapper_module(x)
                else:
                    if self._wrapper_module is None:
                        self._wrapper_module = _BackboneNeckWrapper(self.model).eval().to(dev)
                    with torch.inference_mode():
                        feats = self._wrapper_module(x)
                if isinstance(feats, torch.Tensor):
                    feats = [feats, feats, feats]
                elif isinstance(feats, (list, tuple)):
                    feats = list(feats)
                else:
                    feats = [torch.as_tensor(feats)]
            with torch.inference_mode():
                with self._profile_scope("head"):
                    bbox_head_results = self.model.bbox_head(tuple(feats))
                    objectnesses = None
                    if len(bbox_head_results) == 3:
                        cls_scores, bbox_preds, objectnesses = bbox_head_results
                    elif len(bbox_head_results) == 2:
                        cls_scores, bbox_preds = bbox_head_results
                    else:
                        raise RuntimeError(
                            "Unexpected number of outputs from bbox_head: "
                            f"{len(bbox_head_results)}"
                        )
                    # Decode the complete batch once, then retain the existing
                    # per-image NMS behavior in the centralized dispatcher.
                    result_list: List[InstanceData] = self.model.bbox_head.predict_by_feat(
                        cls_scores=cls_scores,
                        bbox_preds=bbox_preds,
                        objectnesses=objectnesses,
                        batch_img_metas=img_metas,
                        cfg=None,
                        rescale=True,
                        with_nms=False,
                    )

            with self._profile_scope("nms"):
                result_list = self._nms.run_batch(result_list, img_metas)

            class _Wrap:
                def __init__(self, inst_):
                    self.pred_instances = inst_

            with self._profile_scope("postprocess"):
                results = [
                    _Wrap(_strip_static_padding(inst, strip=self._nms_backend != "trt"))
                    for inst in result_list
                ]

            if do_trace == 10:
                pass

            self._pass += 1
            return results

    def _disable_trt(self, exc: Exception) -> None:
        self._trt_disabled_reason = str(exc)
        self._trt_module = None
        if self._nms_backend.lower() == "trt":
            self._nms_backend = _resolve_detector_nms_backend(
                "torchvision", owner="TensorRT detector wrapper fallback"
            )
            self._nms = DetectorNMS(
                bbox_head=getattr(self.model, "bbox_head", self.model),
                backend=self._nms_backend,
                compare_backends=[],
                trt_plugin=self._nms_plugin,
            )
        logger.warning(
            "Disabling TensorRT detector path and falling back to PyTorch backbone/neck "
            "with %s NMS: %s",
            self._nms_backend,
            exc,
        )


class _TorchDetectorWrapper(_ProfilerMixin):
    """Pure-PyTorch detector wrapper that mirrors the YOLOX + DetectorNMS flow.

    Used for YOLOX-based detectors when neither ONNX nor TensorRT backbone/neck
    acceleration is enabled, so that the same TensorRT-based NMS (or alternate
    backends) and static-shape decoding paths can be reused without changing
    the upstream Aspen graph.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        profiler: Optional[Any] = None,
        nms_backend: str = "trt",
        nms_test: bool = False,
        nms_plugin: str = "efficient",
        cuda_graph: bool = True,
    ):
        super().__init__(profiler=profiler, label="torch_predictor")
        self.model = model
        self._nms_backend: str = _resolve_detector_nms_backend(
            nms_backend, owner="PyTorch detector wrapper"
        )
        self._nms_test: bool = bool(nms_test)
        self._nms_plugin: str = str(nms_plugin or "efficient")
        self._cuda_graph_enabled: bool = bool(cuda_graph)
        self._cg: Optional[CudaGraphCallable] = None
        self._cg_device: Optional[torch.device] = None
        compare_backends = _detector_nms_compare_backends(self._nms_backend, self._nms_test)
        self._nms = DetectorNMS(
            bbox_head=getattr(model, "bbox_head", model),
            backend=self._nms_backend,
            compare_backends=compare_backends,
            trt_plugin=self._nms_plugin,
        )

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if not self._cuda_graph_enabled:
            self._cg = None
            self._cg_device = None
        return True

    def predict(self, imgs: torch.Tensor, data_samples: List[Any]):  # type: ignore[override]
        with self._profile_scope():
            assert isinstance(data_samples, (list, tuple)) and len(data_samples) == imgs.size(0)
            N = imgs.size(0)
            try:
                dev = next(self.model.parameters()).device
            except StopIteration:
                dev = imgs.device

            results: List[Any] = []
            with torch.inference_mode():
                # CUDA graph fast path: batch=1, TensorRT NMS backend, and CUDA inputs.
                if (
                    self._cuda_graph_enabled
                    and imgs.is_cuda
                    and N == 1
                    and self._nms_backend == "trt"
                    and not self._nms_test
                ):
                    meta0 = getattr(data_samples[0], "metainfo", {}) or {}

                    def _fn(x: torch.Tensor):
                        feats = self.model.extract_feat(x)
                        head_out = self.model.bbox_head(feats)
                        if isinstance(head_out, (list, tuple)) and len(head_out) == 2:
                            cls_scores, bbox_preds = head_out
                            objectnesses = None
                        elif isinstance(head_out, (list, tuple)) and len(head_out) == 3:
                            cls_scores, bbox_preds, objectnesses = head_out
                        else:
                            raise RuntimeError(
                                f"Unexpected bbox_head output shape: {type(head_out)}"
                            )
                        inst_list: List[InstanceData] = self.model.bbox_head.predict_by_feat(
                            cls_scores=cls_scores,
                            bbox_preds=bbox_preds,
                            objectnesses=objectnesses,
                            batch_img_metas=[meta0],
                            cfg=None,
                            rescale=True,
                            with_nms=False,
                        )
                        inst0 = inst_list[0]
                        out_list = self._nms._run_trt([inst0], [meta0])  # type: ignore[attr-defined]
                        out0 = out_list[0]
                        num_valid = getattr(out0, "num_valid_after_nms", None)
                        if not isinstance(num_valid, torch.Tensor):
                            num_valid = torch.as_tensor(
                                int(getattr(out0, "num_valid", 0) or 0),
                                device=out0.bboxes.device,
                                dtype=torch.int32,
                            )
                        return out0.bboxes, out0.scores, out0.labels, num_valid

                    if self._cg is None or self._cg_device != imgs.device:
                        self._cg = CudaGraphCallable(_fn, (imgs[:1].to(dev),), name="detector")
                        self._cg_device = imgs.device

                    bboxes, scores, labels, num_valid = self._cg(imgs[:1].to(dev))

                    inst = InstanceData()
                    inst.bboxes = bboxes
                    inst.scores = scores
                    inst.labels = labels
                    inst.set_metainfo(
                        dict(
                            num_valid_after_nms=num_valid,
                            max_detections=int(bboxes.shape[0]),
                        )
                    )

                    class _Wrap:
                        def __init__(self, inst_):
                            self.pred_instances = inst_

                    results.append(_Wrap(_strip_static_padding(inst, strip=False)))
                    return results

                with self._profile_scope("backbone"):
                    feats = self.model.extract_feat(imgs.to(device=dev, non_blocking=True))
                with self._profile_scope("head"):
                    head_out = self.model.bbox_head(feats)
                    if isinstance(head_out, (list, tuple)) and len(head_out) == 2:
                        cls_scores, bbox_preds = head_out
                        objectnesses = None
                    elif isinstance(head_out, (list, tuple)) and len(head_out) == 3:
                        cls_scores, bbox_preds, objectnesses = head_out
                    else:
                        raise RuntimeError(f"Unexpected bbox_head output type: {type(head_out)}")
                    metas: List[Dict[str, Any]] = []
                    for i in range(N):
                        try:
                            metas.append(getattr(data_samples[i], "metainfo", {}))
                        except Exception:
                            metas.append({})
                    result_list: List[InstanceData] = self.model.bbox_head.predict_by_feat(
                        cls_scores=cls_scores,
                        bbox_preds=bbox_preds,
                        objectnesses=objectnesses,
                        batch_img_metas=metas,
                        cfg=None,
                        rescale=True,
                        with_nms=False,
                    )

                with self._profile_scope("postprocess"):
                    for inst, meta in zip(result_list, metas):
                        with self._profile_scope("nms"):
                            inst = self._nms.run_single(inst, meta)

                        class _Wrap:
                            def __init__(self, inst_):
                                self.pred_instances = inst_

                        results.append(
                            _Wrap(_strip_static_padding(inst, strip=self._nms_backend != "trt"))
                        )

            return results
