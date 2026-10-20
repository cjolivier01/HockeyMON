from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.structures import InstanceData

from hmlib.log import get_logger
from hmlib.utils.tensor import make_const_tensor, new_zeros

_TRT_LOGGER = None


def get_trt_logger(trt_module):
    """Return a process-wide TensorRT logger instance.

    TensorRT uses a single global logger behind the scenes. Creating
    multiple Logger instances and passing them to Builder/Runtime can
    trigger warnings about mismatched loggers. To avoid this, we create
    and cache a single Logger here and reuse it for all engines.
    """
    global _TRT_LOGGER
    if _TRT_LOGGER is not None:
        return _TRT_LOGGER

    # Use a dedicated logger with elevated severity so that noisy TensorRT
    # warnings (e.g., BatchedNMSPlugin deprecation messages) do not spam
    # the console during inference. Errors are still reported.
    try:
        _TRT_LOGGER = trt_module.Logger(trt_module.Logger.ERROR)
    except Exception:
        # Fallback to default WARNING severity if the constructor signature
        # differs on older TensorRT versions.
        _TRT_LOGGER = trt_module.Logger()

    # Best-effort plugin initialization; non-fatal if this fails or if
    # plugins were already initialized elsewhere.
    try:
        trt_module.init_libnvinfer_plugins(_TRT_LOGGER, "")
    except Exception:
        pass

    return _TRT_LOGGER


def _get_plugin_creator(registry, name: str):
    """Look up a TensorRT plugin across the pre-11 and 11+ registry APIs."""
    # Prefer enumerating creators: TensorRT 11 logs an error when get_creator
    # probes a plugin removed from the registry, even when absence is expected.
    for collection_name in ("all_creators", "plugin_creator_list"):
        creators = getattr(registry, collection_name, None)
        if creators is None:
            continue
        for creator in creators:
            creator_name = getattr(creator, "name", None)
            creator_version = getattr(creator, "plugin_version", None)
            creator_namespace = getattr(creator, "plugin_namespace", None)
            if (creator_name, creator_version, creator_namespace) == (name, "1", ""):
                return creator
        return None

    get_creator = getattr(registry, "get_creator", None)
    if get_creator is None:
        get_creator = getattr(registry, "get_plugin_creator", None)
    if get_creator is None:
        raise RuntimeError("TensorRT plugin registry does not expose a creator lookup API")
    return get_creator(name, "1", "")


@dataclass
class TrtNmsConfig:
    num_classes: int
    max_num_boxes: int  # static input/output size per image
    top_k: int  # plugin topK (per class * image)
    keep_top_k: int  # plugin keepTopK (per image, before our own max_per_img)
    score_threshold: float
    iou_threshold: float
    max_per_img: int
    share_location: bool = True
    background_label_id: int = -1
    # Suppress across classes rather than within each. Only a multi-class head
    # needs it -- two anchors claiming one skater under different labels --
    # and it stays off by default so single-class profiles are unchanged.
    class_agnostic: bool = False
    is_normalized: bool = False
    clip_boxes: bool = False
    score_bits: int = 16
    caffe_semantics: bool = True
    plugin: str = "efficient"


class TrtBatchedNMS:
    """Thin TensorRT NMS wrapper.

    Builds a tiny TensorRT engine that only contains the EfficientNMS_TRT
    plugin and exposes a callable that takes an InstanceData and returns a
    new InstanceData with NMS applied. The engine is static-shape
    (no dynamic dims) to avoid runtime shape updates and stream syncs.
    """

    def __init__(self, cfg: TrtNmsConfig, stream: Optional[torch.cuda.Stream]):
        self.cfg = cfg
        self._engine = None
        self._context = None
        # Optional external stream; when None, the caller's current stream is
        # used for execution so that no explicit cross-stream waits are needed.
        self._stream: Optional[torch.cuda.Stream] = stream
        self._trt = None
        self._np = None
        self._output_dtypes: dict[str, torch.dtype] = {}

    @staticmethod
    def _zero_padded_outputs(
        boxes_out: torch.Tensor,
        scores_out: torch.Tensor,
        labels_out: torch.Tensor,
        num_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_valid.numel() < 1:
            raise ValueError("num_valid tensor must contain at least one value")
        valid_count = num_valid.reshape(-1)[:1].to(
            device=boxes_out.device,
            dtype=torch.long,
            non_blocking=True,
        )
        valid_mask = (
            torch.arange(boxes_out.shape[0], device=boxes_out.device, dtype=torch.long)
            < valid_count[0]
        )
        boxes_out = torch.where(valid_mask.view(-1, 1), boxes_out, torch.zeros_like(boxes_out))
        scores_mask = valid_mask
        if scores_mask.device != scores_out.device:
            scores_mask = scores_mask.to(device=scores_out.device, non_blocking=True)
        scores_out = torch.where(scores_mask, scores_out, torch.zeros_like(scores_out))
        labels_mask = valid_mask
        if labels_mask.device != labels_out.device:
            labels_mask = labels_mask.to(device=labels_out.device, non_blocking=True)
        labels_out = torch.where(labels_mask, labels_out, torch.zeros_like(labels_out))
        return boxes_out, scores_out, labels_out

    @staticmethod
    def from_bbox_head(
        bbox_head: torch.nn.Module,
        max_num_boxes: int,
        stream: Optional[torch.cuda.Stream] = None,
        max_per_img: int = 250,
        plugin: str = "batched",
    ) -> TrtBatchedNMS:
        """Construct a config from an mmdet/mmyolo bbox head."""
        num_classes = int(getattr(bbox_head, "num_classes", 1))
        test_cfg = getattr(bbox_head, "test_cfg", None)

        score_thr = 0.0
        iou_thr = 0.5
        class_agnostic = False
        nms_pre = max_num_boxes

        try:
            if test_cfg is not None:
                # mmengine ConfigDict behaves like a dict
                score_thr = float(test_cfg.get("score_thr", score_thr))
                max_per_img = int(test_cfg.get("max_per_img", max_per_img))
                nms_cfg = test_cfg.get("nms", {}) or {}
                if "iou_threshold" in nms_cfg:
                    iou_thr = float(nms_cfg["iou_threshold"])
                elif "iou_thr" in nms_cfg:
                    iou_thr = float(nms_cfg["iou_thr"])
                class_agnostic = bool(nms_cfg.get("class_agnostic", class_agnostic))
                nms_pre = int(test_cfg.get("nms_pre", nms_pre))
        except Exception:
            pass

        # Prefer static max_detections when available (YOLOX static path).
        static_max = 0
        try:
            if bool(getattr(bbox_head, "static_det_enabled", False)):
                static_max = int(getattr(bbox_head, "static_det_max", 0) or 0)
        except Exception:
            static_max = 0

        max_top_k = 512 * 8
        max_num_boxes = int(max_num_boxes)
        nms_pre = int(nms_pre)
        # Bound max_num_boxes by pre-NMS clamp and plugin limits.
        if static_max > 0:
            max_num_boxes = static_max
        max_num_boxes = max(1, min(max_num_boxes, nms_pre, max_top_k))

        # Use full max_num_boxes inside the plugin, and apply max_per_img
        # explicitly after TensorRT NMS for closer parity with mmcv.batched_nms.
        top_k = max_num_boxes
        # Prefer enforcing max_per_img inside the plugin when available.
        keep_top_k = max_per_img if max_per_img > 0 else max_num_boxes

        cfg = TrtNmsConfig(
            num_classes=num_classes,
            max_num_boxes=max_num_boxes,
            top_k=top_k,
            keep_top_k=keep_top_k,
            score_threshold=float(score_thr),
            iou_threshold=float(iou_thr),
            max_per_img=int(max_per_img),
            plugin=str(plugin or "batched").lower(),
            class_agnostic=bool(class_agnostic),
        )
        return TrtBatchedNMS(cfg, stream=stream)

    def _lazy_imports(self):
        if self._trt is not None:
            return
        import importlib

        try:
            trt = importlib.import_module("tensorrt")
        except Exception as ex:
            raise RuntimeError("TensorRT Python package is required for TrtBatchedNMS") from ex
        try:
            import numpy as np  # type: ignore
        except Exception as ex:
            raise RuntimeError("NumPy is required for TrtBatchedNMS") from ex

        self._trt = trt
        self._np = np

    def _build_engine(self, device: torch.device) -> None:
        self._lazy_imports()
        trt = self._trt
        np = self._np

        logger = get_trt_logger(trt)

        builder = trt.Builder(logger)
        # TensorRT 10 and earlier require this flag; TensorRT 11 removes it
        # because explicit batch is the only supported network mode.
        explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", 0)
        flags = int(explicit_batch)
        network = builder.create_network(flags)
        config = builder.create_builder_config()
        # NMS network is tiny; 32MB is plenty.
        try:
            # TensorRT >= 8.0 style API.
            workspace_type = trt.MemoryPoolType.WORKSPACE
            config.set_memory_pool_limit(workspace_type, 1 << 25)
        except Exception:
            # Fallback for older TensorRT versions, if ever encountered.
            if hasattr(config, "max_workspace_size"):
                config.max_workspace_size = 1 << 25

        B = 1
        N = int(self.cfg.max_num_boxes)
        C = int(self.cfg.num_classes)

        # TensorRT 11 removes BatchedNMSDynamic_TRT, while older releases may
        # lack EfficientNMS_TRT. Resolve the requested plugin before defining
        # inputs so the runtime bindings use the selected plugin's shapes.
        plugin_kind = getattr(self.cfg, "plugin", "efficient").lower()
        registry = trt.get_plugin_registry()
        efficient_creator = _get_plugin_creator(registry, "EfficientNMS_TRT")
        batched_creator = _get_plugin_creator(registry, "BatchedNMSDynamic_TRT")

        if plugin_kind == "efficient" and efficient_creator is None:
            if batched_creator is None:
                raise RuntimeError("No supported NMS plugin found in TensorRT registry")
            get_logger(__name__).warning(
                "EfficientNMS_TRT is unavailable; falling back to BatchedNMSDynamic_TRT"
            )
            plugin_kind = "batched"
        elif plugin_kind != "efficient" and batched_creator is None:
            if efficient_creator is None:
                raise RuntimeError("No supported NMS plugin found in TensorRT registry")
            get_logger(__name__).warning(
                "BatchedNMSDynamic_TRT is unavailable; falling back to EfficientNMS_TRT"
            )
            plugin_kind = "efficient"

        self.cfg.plugin = plugin_kind

        if plugin_kind == "efficient":
            # EfficientNMS expects boxes in shape [B, num_boxes, 4] and scores
            # in shape [B, num_boxes, num_classes].
            boxes = network.add_input(
                name="boxes",
                dtype=trt.DataType.FLOAT,
                shape=(B, N, 4),
            )
            scores = network.add_input(
                name="scores",
                dtype=trt.DataType.FLOAT,
                shape=(B, N, C),
            )

            fields = []

            def add_field(name: str, value, ftype):
                arr = np.array([value], dtype=np.int32 if "INT" in ftype.name else np.float32)
                fields.append(trt.PluginField(name, arr, ftype))

            # EfficientNMS parameters
            add_field(
                "score_threshold", float(self.cfg.score_threshold), trt.PluginFieldType.FLOAT32
            )
            add_field("iou_threshold", float(self.cfg.iou_threshold), trt.PluginFieldType.FLOAT32)
            add_field("max_output_boxes", int(self.cfg.keep_top_k), trt.PluginFieldType.INT32)
            add_field(
                "background_class", int(self.cfg.background_label_id), trt.PluginFieldType.INT32
            )
            # 0 = no activation (scores already in [0,1]), 1 = sigmoid
            add_field("score_activation", 0, trt.PluginFieldType.INT32)
            # 0 = per-class NMS, 1 = class-agnostic
            add_field(
                "class_agnostic", int(bool(self.cfg.class_agnostic)), trt.PluginFieldType.INT32
            )
            # 0 = boxes in [x1, y1, x2, y2], 1 = [x, y, w, h]
            add_field("box_coding", 0, trt.PluginFieldType.INT32)

            assert efficient_creator is not None
            plugin = efficient_creator.create_plugin(
                "efficient_nms", trt.PluginFieldCollection(fields)
            )
            layer = network.add_plugin_v2([boxes, scores], plugin)

        if plugin_kind != "efficient":
            # Legacy BatchedNMSDynamic_TRT plugin.
            boxes = network.add_input(
                name="boxes",
                dtype=trt.DataType.FLOAT,
                shape=(B, N, 1, 4),
            )
            scores = network.add_input(
                name="scores",
                dtype=trt.DataType.FLOAT,
                shape=(B, N, C),
            )

            fields = []

            def add_field(name: str, value, ftype):
                arr = np.array([value], dtype=np.int32 if "INT" in ftype.name else np.float32)
                fields.append(trt.PluginField(name, arr, ftype))

            add_field(
                "shareLocation", int(bool(self.cfg.share_location)), trt.PluginFieldType.INT32
            )
            add_field(
                "backgroundLabelId", int(self.cfg.background_label_id), trt.PluginFieldType.INT32
            )
            add_field("numClasses", int(self.cfg.num_classes), trt.PluginFieldType.INT32)
            add_field("topK", int(self.cfg.top_k), trt.PluginFieldType.INT32)
            add_field("keepTopK", int(self.cfg.keep_top_k), trt.PluginFieldType.INT32)
            add_field(
                "scoreThreshold", float(self.cfg.score_threshold), trt.PluginFieldType.FLOAT32
            )
            add_field("iouThreshold", float(self.cfg.iou_threshold), trt.PluginFieldType.FLOAT32)
            add_field("isNormalized", int(bool(self.cfg.is_normalized)), trt.PluginFieldType.INT32)
            add_field("clipBoxes", int(bool(self.cfg.clip_boxes)), trt.PluginFieldType.INT32)
            add_field("scoreBits", int(self.cfg.score_bits), trt.PluginFieldType.INT32)
            add_field(
                "caffeSemantics", int(bool(self.cfg.caffe_semantics)), trt.PluginFieldType.INT32
            )

            assert batched_creator is not None
            plugin = batched_creator.create_plugin("batched_nms", trt.PluginFieldCollection(fields))
            layer = network.add_plugin_v2([boxes, scores], plugin)

        num_det = layer.get_output(0)
        nmsed_boxes = layer.get_output(1)
        nmsed_scores = layer.get_output(2)
        nmsed_classes = layer.get_output(3)

        num_det.name = "num_detections"
        nmsed_boxes.name = "nmsed_boxes"
        nmsed_scores.name = "nmsed_scores"
        nmsed_classes.name = "nmsed_classes"

        network.mark_output(num_det)
        network.mark_output(nmsed_boxes)
        network.mark_output(nmsed_scores)
        network.mark_output(nmsed_classes)

        # Static shapes; single profile with fixed min/opt/max.
        profile = builder.create_optimization_profile()
        if plugin_kind == "efficient":
            profile.set_shape("boxes", (B, N, 4), (B, N, 4), (B, N, 4))
            profile.set_shape("scores", (B, N, C), (B, N, C), (B, N, C))
        else:
            profile.set_shape("boxes", (B, N, 1, 4), (B, N, 1, 4), (B, N, 1, 4))
            profile.set_shape("scores", (B, N, C), (B, N, C), (B, N, C))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT NMS network")

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT NMS engine")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT NMS execution context")

        self._engine = engine
        self._context = context
        self._output_dtypes = {
            name: torch.from_numpy(
                np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))
            ).dtype
            for name in (
                "num_detections",
                "nmsed_boxes",
                "nmsed_scores",
                "nmsed_classes",
            )
        }

    def _ensure_engine(self, device: torch.device) -> None:
        if self._engine is None or self._context is None:
            self._build_engine(device)

    def _infer(self, boxes: torch.Tensor, scores: torch.Tensor):
        assert boxes.is_cuda and scores.is_cuda
        assert boxes.dtype == torch.float32
        assert scores.dtype == torch.float32

        device = boxes.device
        self._ensure_engine(device)

        engine = self._engine
        context = self._context
        assert engine is not None and context is not None

        B = 1
        N = self.cfg.max_num_boxes
        C = self.cfg.num_classes
        K = self.cfg.keep_top_k
        plugin_kind = getattr(self.cfg, "plugin", "efficient").lower()

        num_boxes = int(boxes.shape[0])
        if num_boxes > N:
            # Truncate to the maximum supported by the engine.
            boxes = boxes[:N]
            scores = scores[:N]

        # Pad inputs up to static engine shapes.
        if plugin_kind == "efficient":
            boxes_pad = torch.zeros((B, N, 4), device=device, dtype=torch.float32)
        else:
            boxes_pad = torch.zeros((B, N, 1, 4), device=device, dtype=torch.float32)
        scores_pad = torch.zeros((B, N, C), device=device, dtype=torch.float32)
        if plugin_kind == "efficient":
            boxes_pad[0, : boxes.shape[0], :] = boxes
        else:
            boxes_pad[0, : boxes.shape[0], 0, :] = boxes
        scores_pad[0, : scores.shape[0], :] = scores

        num_det = torch.empty((B, 1), device=device, dtype=self._output_dtypes["num_detections"])
        out_boxes = torch.empty((B, K, 4), device=device, dtype=self._output_dtypes["nmsed_boxes"])
        out_scores = torch.empty((B, K), device=device, dtype=self._output_dtypes["nmsed_scores"])
        out_classes = torch.empty((B, K), device=device, dtype=self._output_dtypes["nmsed_classes"])

        # Bind buffers by tensor name.
        context.set_tensor_address("boxes", boxes_pad.data_ptr())
        context.set_tensor_address("scores", scores_pad.data_ptr())
        context.set_tensor_address("num_detections", num_det.data_ptr())
        context.set_tensor_address("nmsed_boxes", out_boxes.data_ptr())
        context.set_tensor_address("nmsed_scores", out_scores.data_ptr())
        context.set_tensor_address("nmsed_classes", out_classes.data_ptr())

        # Execute on the caller's current stream so that the entire NMS path
        # stays on a single CUDA stream with no host-side synchronization.
        current_stream = torch.cuda.current_stream(device)
        ok = context.execute_async_v3(current_stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT NMS execution failed")

        return num_det, out_boxes, out_scores, out_classes

    def __call__(self, instances: InstanceData) -> InstanceData:
        bboxes = getattr(instances, "bboxes", None)
        scores = getattr(instances, "scores", None)
        labels = getattr(instances, "labels", None)
        if bboxes is None or scores is None or labels is None:
            return instances
        if (
            not torch.is_tensor(bboxes)
            or not torch.is_tensor(scores)
            or not torch.is_tensor(labels)
        ):
            return instances
        if bboxes.numel() == 0:
            return instances

        device = bboxes.device
        num_boxes = bboxes.shape[0]
        num_classes = self.cfg.num_classes

        # Build per-class score matrix from per-box scores + labels.
        scores_full = torch.zeros(
            (num_boxes, num_classes),
            device=device,
            dtype=torch.float32,
        )
        idx = torch.arange(num_boxes, device=device, dtype=torch.long)
        # Avoid in-place ops on tensors that may have been created under
        # torch.inference_mode() by using out-of-place clamp.
        labels_clamped = labels.to(torch.long).clamp(0, num_classes - 1)
        scores_full[idx, labels_clamped] = scores.to(torch.float32)

        num_det, out_boxes, out_scores, out_classes = self._infer(
            bboxes.to(torch.float32), scores_full
        )

        # Use plugin outputs directly; all tensors are already on the correct
        # CUDA stream. TensorRT leaves rows beyond num_detections unspecified
        # for some plugin versions, so explicitly zero the padded tail before
        # downstream fixed-shape pruning can rank those rows.
        boxes_out = out_boxes[0]
        scores_out = out_scores[0]
        labels_out = out_classes[0].to(torch.long)
        num_valid_tensor = num_det.view(-1)[0].to(device=device, dtype=torch.int32)
        boxes_out, scores_out, labels_out = self._zero_padded_outputs(
            boxes_out,
            scores_out,
            labels_out,
            num_valid_tensor,
        )

        new_inst = InstanceData()
        new_inst.bboxes = boxes_out
        new_inst.scores = scores_out
        new_inst.labels = labels_out

        # Propagate static padding metadata so _strip_static_padding can
        # trim back to num_valid on the GPU. Keep num_valid on device to
        # avoid a host-side sync here.
        new_inst.set_metainfo(
            dict(
                num_valid_after_nms=num_valid_tensor,
                max_detections=int(self.cfg.max_num_boxes),
                num_valid_before_nms=int(num_boxes),
            )
        )
        return new_inst


class DetectorNMS:
    """Unified NMS dispatcher for detector outputs.

    Handles multiple backends:
      - 'trt': TensorRT NMS plugin (EfficientNMS_TRT by default) via TrtBatchedNMS
      - 'torchvision': torchvision.ops.nms per class
      - 'head': reuse the detection head's own _bbox_post_process NMS

    Can also compare results against one or more auxiliary backends for
    debugging. All methods operate on batches represented as sequences
    of InstanceData objects.
    """

    def __init__(
        self,
        bbox_head: torch.nn.Module,
        backend: str = "trt",
        compare_backends: Optional[Sequence[str]] = None,
        trt_plugin: str = "efficient",
    ) -> None:
        self.bbox_head = bbox_head
        self.backend: str = str(backend or "trt").lower()
        self.compare_backends: List[str] = [b.lower() for b in (compare_backends or [])]
        self.trt_plugin: str = str(trt_plugin or "efficient").lower()
        self._trt_nms: Optional[TrtBatchedNMS] = None

    def _ensure_trt(self, max_num_boxes: int, stream: Optional[torch.cuda.Stream]) -> None:
        if self._trt_nms is None:
            self._trt_nms = TrtBatchedNMS.from_bbox_head(
                self.bbox_head,
                max_num_boxes=max_num_boxes,
                stream=stream,
                plugin=self.trt_plugin,
            )
            assert self._trt_nms is not None
        # else:
        #     # If we ever see a larger pre-NMS set, rebuild with the larger
        #     # capacity to avoid truncation.
        #     if max_num_boxes > self._trt_nms.cfg.max_num_boxes:
        #         self._trt_nms = TrtBatchedNMS.from_bbox_head(self.bbox_head, max_num_boxes=max_num_boxes, stream=stream)

    def _run_trt(
        self, instances: Sequence[InstanceData], img_metas: Sequence[Dict[str, Any]]
    ) -> List[InstanceData]:
        if not instances:
            return []
        # Determine maximum pre-NMS count across the batch so the plugin
        # capacity is sufficient for all images.
        max_num_boxes = 1
        for inst in instances:
            b = getattr(inst, "bboxes", None)
            if torch.is_tensor(b) and b.ndim == 2 and b.shape[0] > max_num_boxes:
                max_num_boxes = int(b.shape[0])
        stream = torch.cuda.current_stream(instances[0].bboxes.device) if instances else None
        self._ensure_trt(max_num_boxes=max_num_boxes, stream=stream)
        assert self._trt_nms is not None
        return [self._trt_nms(inst) for inst in instances]

    def _run_torchvision(
        self,
        instances: Sequence[InstanceData],
        img_metas: Sequence[Dict[str, Any]],
    ) -> List[InstanceData]:
        try:
            from torchvision.ops import nms as tv_nms  # type: ignore
        except Exception as ex:
            raise RuntimeError(
                "torchvision.ops.nms is required for torchvision NMS backend"
            ) from ex

        def _single(inst: InstanceData, img_meta: Dict[str, Any]) -> InstanceData:
            bboxes = getattr(inst, "bboxes", None)
            scores = getattr(inst, "scores", None)
            labels = getattr(inst, "labels", None)
            if bboxes is None or scores is None or labels is None:
                return inst
            if (
                not torch.is_tensor(bboxes)
                or not torch.is_tensor(scores)
                or not torch.is_tensor(labels)
            ):
                return inst

            device = bboxes.device

            # Drop padded entries (e.g., from YOLOX static_detections) and invalid labels.
            valid_mask = torch.isfinite(scores)
            valid_mask = valid_mask & (labels >= 0)
            if valid_mask.ndim > 1:
                valid_mask = valid_mask.view(-1)
            if valid_mask.numel() != bboxes.shape[0]:
                valid_mask = torch.ones(bboxes.shape[0], dtype=torch.bool, device=device)

            bboxes_valid = bboxes[valid_mask]
            scores_valid = scores[valid_mask]
            labels_valid = labels[valid_mask].to(torch.long)
            num_before = int(bboxes_valid.shape[0])
            if num_before == 0:
                empty = InstanceData()
                empty.bboxes = bboxes[:0]
                empty.scores = scores[:0]
                empty.labels = labels[:0]
                return empty

            # Infer NMS config from the bbox head when available.
            iou_thr = 0.5
            max_per_img: Optional[int] = None
            try:
                test_cfg = getattr(self.bbox_head, "test_cfg", None)
                if test_cfg is not None:
                    nms_cfg = test_cfg.get("nms", {}) or {}
                    if "iou_threshold" in nms_cfg:
                        iou_thr = float(nms_cfg["iou_threshold"])
                    elif "iou_thr" in nms_cfg:
                        iou_thr = float(nms_cfg["iou_thr"])
                    max_per_img = int(test_cfg.get("max_per_img", 0) or 0)
            except Exception:
                pass

            keep_global: List[torch.Tensor] = []
            for cls_id in torch.unique(labels_valid):
                cls_id_int = int(cls_id)
                cls_mask = labels_valid == cls_id_int
                idxs = cls_mask.nonzero(as_tuple=False).view(-1)
                if idxs.numel() == 0:
                    continue
                boxes_cls = bboxes_valid[idxs]
                scores_cls = scores_valid[idxs]
                keep_rel = tv_nms(boxes_cls, scores_cls, iou_thr)
                if keep_rel.numel() == 0:
                    continue
                keep_global.append(idxs[keep_rel])

            if not keep_global:
                kept = new_zeros(bboxes_valid, (0, 4))
                kept_scores = new_zeros(scores_valid, (0,))
                kept_labels = new_zeros(labels_valid, (0,))
            else:
                keep_idx = torch.cat(keep_global, dim=0)
                kept = bboxes_valid[keep_idx]
                kept_scores = scores_valid[keep_idx]
                kept_labels = labels_valid[keep_idx]

            # Sort by score descending and apply max_per_img if set.
            if kept_scores.numel() > 0:
                order = torch.argsort(kept_scores, descending=True)
                kept = kept[order]
                kept_scores = kept_scores[order]
                kept_labels = kept_labels[order]
            if max_per_img and kept.shape[0] > max_per_img:
                kept = kept[:max_per_img]
                kept_scores = kept_scores[:max_per_img]
                kept_labels = kept_labels[:max_per_img]

            num_valid = int(kept.shape[0])

            new_inst = InstanceData()
            new_inst.bboxes = kept
            new_inst.scores = kept_scores
            new_inst.labels = kept_labels

            try:
                new_inst.set_metainfo(
                    dict(
                        num_valid_after_nms=make_const_tensor(
                            num_valid, device=device, dtype=torch.int32
                        ),
                        num_valid_before_nms=int(num_before),
                    )
                )
            except Exception:
                pass
            return new_inst

        return [_single(inst, meta) for inst, meta in zip(instances, img_metas)]

    def _run_head(
        self,
        instances: Sequence[InstanceData],
        img_metas: Sequence[Dict[str, Any]],
    ) -> List[InstanceData]:
        """Reuse the bbox head's own _bbox_post_process NMS."""
        if not hasattr(self.bbox_head, "_bbox_post_process"):
            raise RuntimeError(
                "bbox_head does not implement _bbox_post_process; cannot use 'head' NMS backend"
            )

        test_cfg = getattr(self.bbox_head, "test_cfg", None)
        if test_cfg is None:
            raise RuntimeError(
                "bbox_head.test_cfg is None; cannot infer NMS config for 'head' backend"
            )

        out: List[InstanceData] = []
        for inst, img_meta in zip(instances, img_metas):
            # Determine static_max_detections mirroring YOLOX static path.
            static_max: Optional[int] = None
            try:
                if bool(getattr(self.bbox_head, "static_det_enabled", False)):
                    pad_after = bool(getattr(self.bbox_head, "static_det_pad_after_nms", True))
                    if pad_after:
                        static_max = int(getattr(self.bbox_head, "static_det_max", 0) or 0) or None
            except Exception:
                static_max = None
            # If metadata carries max_detections, prefer that.
            try:
                meta_obj = getattr(inst, "metainfo", None)
                if isinstance(meta_obj, dict) and "max_detections" in meta_obj:
                    static_max = int(meta_obj["max_detections"])
            except Exception:
                pass

            # Bboxes in inst are already in final image space when produced via
            # predict_by_feat(..., rescale=True, with_nms=False), so we call
            # _bbox_post_process with rescale=False.
            new_inst = self.bbox_head._bbox_post_process(  # type: ignore[attr-defined]
                results=inst,
                cfg=test_cfg,
                rescale=False,
                with_nms=True,
                img_meta=img_meta,
                static_max_detections=static_max,
            )
            out.append(new_inst)
        return out

    def run_batch(
        self,
        instances: Sequence[InstanceData],
        img_metas: Sequence[Dict[str, Any]],
    ) -> List[InstanceData]:
        """Apply the configured primary backend (and optional comparators) to a batch."""
        if len(instances) != len(img_metas):
            raise ValueError(
                f"DetectorNMS.run_batch expected equal lengths, got {len(instances)} and {len(img_metas)}"
            )

        backend = self.backend
        if backend == "trt":
            primary = self._run_trt(instances, img_metas)
        elif backend == "torchvision":
            primary = self._run_torchvision(instances, img_metas)
        elif backend == "head":
            primary = self._run_head(instances, img_metas)
        else:
            raise ValueError(f"Unknown NMS backend '{backend}'")

        # Optional comparison against additional backends for debugging.
        for cmp_backend in self.compare_backends:
            if cmp_backend == backend:
                continue
            if cmp_backend == "trt":
                other = self._run_trt(instances, img_metas)
            elif cmp_backend == "torchvision":
                other = self._run_torchvision(instances, img_metas)
            elif cmp_backend == "head":
                other = self._run_head(instances, img_metas)
            else:
                raise ValueError(f"Unknown comparison NMS backend '{cmp_backend}'")
            # Very lightweight comparison: log count differences per image.
            for idx, (p, o) in enumerate(zip(primary, other)):
                n_p = int(getattr(p.bboxes, "shape", [0])[0])
                n_o = int(getattr(o.bboxes, "shape", [0])[0])
                if n_p != n_o:
                    get_logger(__name__).info(
                        "[NMS-COMPARE] img %d: backend=%s kept %d boxes, %s kept %d boxes",
                        idx,
                        backend,
                        n_p,
                        cmp_backend,
                        n_o,
                    )

        return primary

    def run_single(self, instance: InstanceData, img_meta: Dict[str, Any]) -> InstanceData:
        return self.run_batch([instance], [img_meta])[0]


__all__ = ["DetectorNMS"]
