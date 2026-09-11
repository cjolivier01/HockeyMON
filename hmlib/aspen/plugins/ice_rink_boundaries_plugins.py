import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from mmengine.structures import InstanceData

from hmlib.utils.cuda_graph import CudaGraphCallable
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor

from .base import Plugin


class IceRinkSegmBoundariesPlugin(Plugin):
    """
    Prune detections using the ice rink segmentation mask and optionally draw it.

    Runs after the detector trunk and before the tracker trunk. Uses the
    `rink_profile` provided by the inference pipeline (IceRinkSegmConfig)
    via the TrackDataSample metainfo.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - original_images: optional image tensor for drawing
      - game_id, original_clip_box (optional)
      - plot_ice_mask: bool (optional)

    Produces in context:
      - Side-effects: pred_instances filtered by rink mask; optionally updates original_images when drawing
    """

    def __init__(
        self,
        raise_bbox_center_by_height_ratio: float,
        lower_bbox_bottom_by_height_ratio: float,
        enabled: bool = True,
        det_thresh: float = 0.05,
        max_detections_in_mask: Optional[int] = None,
        plot_ice_mask: bool = False,
        cuda_graph: bool = False,
    ):
        super().__init__(enabled=enabled)
        self._det_thresh = float(det_thresh)
        self._max_detections_in_mask: Optional[int] = (
            int(max_detections_in_mask) if max_detections_in_mask is not None else None
        )
        self._segm = None
        self._default_plot_ice_mask: bool = bool(plot_ice_mask)
        self._cuda_graph_enabled = False
        self._cg: Optional[CudaGraphCallable] = None
        self._cg_device: Optional[torch.device] = None
        self._raise_bbox_center_by_height_ratio = raise_bbox_center_by_height_ratio
        self._lower_bbox_bottom_by_height_ratio = lower_bbox_bottom_by_height_ratio

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        # The rink-prune graph uses dynamic top-k/index_select over mixed dtypes.
        # Under threaded per-plugin streams, captured label outputs have shown
        # intermittent aliasing with bbox float storage. Keep this stage on the
        # plugin stream; it is small and still runs asynchronously.
        self._cuda_graph_enabled = False
        return False

    def _ensure_pipeline(self, context: Dict[str, Any], draw: bool):
        if self._segm is not None:
            # Update draw flag dynamically if needed
            try:
                self._segm._draw = bool(draw)
            except Exception:
                pass
            return
        # Lazily create the same logic as pipeline-based IceRinkSegmBoundaries
        from hmlib.tracking_utils.ice_rink_segm_boundaries import IceRinkSegmBoundaries

        self._segm = IceRinkSegmBoundaries(
            game_id=context.get("game_id"),
            original_clip_box=context.get("original_clip_box"),
            det_thresh=self._det_thresh,
            max_detections_in_mask=self._max_detections_in_mask,
            draw=bool(draw),
            raise_bbox_center_by_height_ratio=self._raise_bbox_center_by_height_ratio,
            lower_bbox_bottom_by_height_ratio=self._lower_bbox_bottom_by_height_ratio,
        )

    @staticmethod
    def _detector_num_detections(det_instances: InstanceData):
        for name in ("num_detections", "num_valid_after_nms", "num_valid"):
            value = getattr(det_instances, name, None)
            if value is not None:
                return value
        metainfo = getattr(det_instances, "metainfo", None)
        if isinstance(metainfo, dict):
            for name in ("num_detections", "num_valid_after_nms", "num_valid"):
                value = metainfo.get(name)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _num_detections_tensor(num_detections, det_bboxes: torch.Tensor) -> torch.Tensor:
        if isinstance(num_detections, StreamTensorBase):
            num_detections = unwrap_tensor(num_detections)
        if num_detections is None:
            return torch.full(
                (1,),
                int(det_bboxes.shape[0]),
                device=det_bboxes.device,
                dtype=torch.int32,
            )
        if torch.is_tensor(num_detections):
            if num_detections.numel() < 1:
                raise ValueError("num_detections tensor must contain at least one value")
            return num_detections.reshape(-1)[:1].to(
                device=det_bboxes.device,
                dtype=torch.int32,
                non_blocking=True,
            )
        count = int(num_detections)
        if count < 0 or count > int(det_bboxes.shape[0]):
            raise ValueError(
                f"num_detections={count} is outside valid range [0, {int(det_bboxes.shape[0])}]"
            )
        return torch.tensor([count], device=det_bboxes.device, dtype=torch.int32)

    @staticmethod
    def _num_detections_int(num_detections, total: int) -> Optional[int]:
        if num_detections is None:
            return None
        if isinstance(num_detections, StreamTensorBase):
            num_detections = unwrap_tensor(num_detections, get=True)
        if torch.is_tensor(num_detections):
            if num_detections.numel() < 1:
                raise ValueError("num_detections tensor must contain at least one value")
            count = int(num_detections.detach().reshape(-1)[:1].cpu().item())
        else:
            count = int(num_detections)
        if count < 0 or count > total:
            raise ValueError(f"num_detections={count} is outside valid range [0, {total}]")
        return count

    @staticmethod
    def _sanitize_pruned_detections(
        bboxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        num_detections,
        max_items: Optional[int],
    ):
        total = int(bboxes.shape[0])
        count = IceRinkSegmBoundariesPlugin._num_detections_int(num_detections, total)
        if count is None:
            count = total

        bboxes = bboxes[:count].to(dtype=torch.float32)
        labels = labels[:count].to(dtype=torch.long)
        scores = scores[:count].to(dtype=torch.float32)

        if labels.numel():
            max_cpp_int = torch.iinfo(torch.int32).max
            valid_labels = (labels >= 0) & (labels <= max_cpp_int)
            if not bool(valid_labels.all().detach().cpu().item()):
                labels = labels.clone()
                labels[~valid_labels] = 0

        if bboxes.numel():
            finite = torch.isfinite(bboxes).all(dim=1)
            ordered = (bboxes[:, 2] >= bboxes[:, 0]) & (bboxes[:, 3] >= bboxes[:, 1])
            valid_boxes = finite & ordered
            if not bool(valid_boxes.all().detach().cpu().item()):
                bboxes = bboxes[valid_boxes]
                labels = labels[valid_boxes]
                scores = scores[valid_boxes]

        kept = int(bboxes.shape[0])
        if max_items is not None:
            max_items = max(int(max_items), 0)
            kept = min(kept, max_items)
            bboxes = bboxes[:kept]
            labels = labels[:kept]
            scores = scores[:kept]
            if bboxes.shape[0] < max_items:
                padded_bboxes = bboxes.new_zeros((max_items, 4))
                padded_labels = labels.new_zeros((max_items,))
                padded_scores = scores.new_zeros((max_items,))
                if bboxes.numel():
                    padded_bboxes[: bboxes.shape[0]].copy_(bboxes)
                    padded_labels[: labels.shape[0]].copy_(labels)
                    padded_scores[: scores.shape[0]].copy_(scores)
                bboxes, labels, scores = padded_bboxes, padded_labels, padded_scores

        num_valid = torch.tensor([kept], device=bboxes.device, dtype=torch.int32)
        return bboxes, labels, scores, num_valid

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        # Access TrackDataSample list
        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        draw_mask: bool = bool(
            context.get("plot_ice_mask")
            or context.get("shared", {}).get("plot_ice_mask", self._default_plot_ice_mask)
        )
        self._ensure_pipeline(context, draw_mask)

        # Optional original_images for overlay
        original_images = context.get("original_images")
        updated_original_images = None

        # Iterate frames and prune detections using rink mask
        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            det_instances = getattr(img_data_sample, "pred_instances", None)
            if det_instances is None:
                continue

            det_bboxes = unwrap_tensor(det_instances.bboxes)
            det_labels = unwrap_tensor(det_instances.labels)
            det_scores = unwrap_tensor(det_instances.scores)
            num_detections = self._detector_num_detections(det_instances)

            # Fast path: CUDA graph the pruning operation (no drawing, fixed max outputs).
            if (
                self._cuda_graph_enabled
                and self._iter_num > 1
                and not draw_mask
                and self._max_detections_in_mask is not None
                and isinstance(det_bboxes, torch.Tensor)
                and det_bboxes.is_cuda
                and det_bboxes.numel() > 0
            ):
                if self._cg is None or self._cg_device != det_bboxes.device:
                    # Build a graph for the current input signature.
                    def _fn(
                        b: torch.Tensor,
                        labels_t: torch.Tensor,
                        s: torch.Tensor,
                        n: torch.Tensor,
                    ):
                        out_b, out_l, out_s, num_valid = self._segm.prune_detections_static(
                            det_bboxes=b,
                            labels=labels_t,
                            scores=s,
                            num_detections=n,
                        )
                        return out_b, out_l, out_s, num_valid

                    num_detections_t = self._num_detections_tensor(num_detections, det_bboxes)
                    self._cg = CudaGraphCallable(
                        _fn,
                        (det_bboxes, det_labels, det_scores, num_detections_t),
                        name="rink_prune",
                    )
                    self._cg_device = det_bboxes.device
                num_detections_t = self._num_detections_tensor(num_detections, det_bboxes)
                new_bboxes, new_labels, new_scores, num_valid = self._cg(
                    det_bboxes,
                    det_labels,
                    det_scores,
                    num_detections_t,
                )
                new_bboxes, new_labels, new_scores, num_valid = self._sanitize_pruned_detections(
                    new_bboxes,
                    new_labels,
                    new_scores,
                    num_valid,
                    self._max_detections_in_mask,
                )
                new_inst = InstanceData(
                    bboxes=wrap_tensor(new_bboxes),
                    labels=wrap_tensor(new_labels),
                    scores=wrap_tensor(new_scores),
                )
                if num_valid is not None:
                    new_inst.set_metainfo({"num_detections": num_valid})
                img_data_sample.pred_instances = new_inst
                continue

            pd_input: Dict[str, Any] = {
                "det_bboxes": det_bboxes,
                "labels": det_labels,
                "scores": det_scores,
                "prune_list": ["det_bboxes", "labels", "scores"],
                "data_samples": track_samples,
                "rink_profile": context.get("rink_profile"),
            }
            if num_detections is not None:
                pd_input["num_detections"] = num_detections
            if original_images is not None:
                pd_input["original_images"] = original_images

            # Reuse the pipeline module's forward for pruning and optional drawing
            out = self._segm(pd_input)

            new_bboxes = out["det_bboxes"]
            new_labels = out["labels"]
            new_scores = out["scores"]
            num_detections = out.get("num_detections")
            new_bboxes, new_labels, new_scores, num_detections = self._sanitize_pruned_detections(
                new_bboxes,
                new_labels,
                new_scores,
                num_detections,
                self._max_detections_in_mask,
            )

            new_inst = InstanceData(
                bboxes=wrap_tensor(new_bboxes),
                labels=wrap_tensor(new_labels),
                scores=wrap_tensor(new_scores),
            )
            if num_detections is not None:
                new_inst.set_metainfo({"num_detections": num_detections})
            img_data_sample.pred_instances = new_inst

            # Update original_images if overlay updated
            if draw_mask:
                updated_original_images = wrap_tensor(out["original_images"])

        return (
            {"original_images": updated_original_images}
            if updated_original_images is not None
            else {}
        )

    def input_keys(self):
        return {
            "data_samples",
            "original_images",
            "game_id",
            "original_clip_box",
            "plot_ice_mask",
            "rink_profile",
        }

    def output_keys(self):
        return {"original_images"}


class IceRinkSegmConfigPlugin(Plugin):
    """
    Compute and attach the rink segmentation profile to TrackDataSample metainfo.

    Replaces the old pipeline transform `IceRinkSegmConfig`.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - original_images: optional image tensor
      - game_id: str (from shared or context)
      - device: torch.device

    Produces in context:
      - rink_profile: dict with rink mask + metadata (for downstream pruning/camera seeding)
    """

    def __init__(self, enabled: bool = True, require_geometry_provenance: bool = False):
        super().__init__(enabled=enabled)
        self._require_geometry_provenance = bool(require_geometry_provenance)
        self._rink_profile = None
        self._rink_geometry_key = None
        self._snapshot_mask = None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        if len(track_data_sample) == 0:
            return {}

        game_id = context.get("game_id") or context.get("shared", {}).get("game_id")
        geometry = context.get("camera_input_geometry") or {}
        revision = (
            geometry.get("stitched_geometry_revision")
            if self._require_geometry_provenance
            else None
        )
        if self._require_geometry_provenance and revision is None:
            raise ValueError("Static rink provenance requires a stitched geometry revision")
        img = unwrap_tensor(context.get("original_images"))
        original_frame = img[0] if torch.is_tensor(img) and img.ndim == 4 else img
        strict_geometry = revision is not None
        if strict_geometry and (not torch.is_tensor(original_frame) or original_frame.ndim != 3):
            raise ValueError("Proven rink geometry requires the original stitched frame")
        geometry_key = (
            game_id,
            revision,
            tuple(original_frame.shape) if torch.is_tensor(original_frame) else None,
        )
        if geometry_key != self._rink_geometry_key:
            self._rink_profile = None
        if self._rink_profile is None:
            from hmlib.segm.ice_rink import configure_ice_rink_mask

            # Never certify a saved mask from shape alone. With authoritative
            # live geometry, regenerate in memory from that exact coordinate plane.
            # Legacy non-stitching pipelines may still use their saved profile,
            # but cannot use a grid checkpoint without a proven original frame.
            frame0 = original_frame
            if frame0 is None:
                fallback = unwrap_tensor(context.get("img"))
                frame0 = (
                    fallback[0] if torch.is_tensor(fallback) and fallback.ndim == 4 else fallback
                )
            exp_shape = None
            if torch.is_tensor(frame0) and frame0.ndim == 3:
                exp_shape = torch.Size(
                    frame0.shape[:2] if frame0.shape[-1] in (3, 4) else frame0.shape[-2:]
                )
            self._rink_profile = configure_ice_rink_mask(
                game_id=game_id,
                device=torch.device("cpu"),
                expected_shape=exp_shape,
                image=frame0,
                force=strict_geometry,
                persist=not strict_geometry,
            )
            if self._rink_profile is not None and strict_geometry:
                mask = self._rink_profile["combined_mask"]
                if mask is None or tuple(mask.shape) != tuple(exp_shape):
                    raise ValueError("Rink profile does not match the original stitched frame")
                raw = mask.detach().cpu().contiguous().numpy()
                self._rink_profile["coordinate_space"] = "original_stitched_pixels"
                self._rink_profile["frame_size"] = [int(raw.shape[1]), int(raw.shape[0])]
                self._rink_profile["geometry_revision"] = hashlib.sha256(
                    revision.encode() + memoryview(raw).tobytes()
                ).hexdigest()
            self._rink_geometry_key = geometry_key
            work_dir = context.get("work_dir") or context.get("shared", {}).get("work_dir")
            if self._rink_profile is not None and work_dir:
                from hmlib.segm.ice_rink import save_boolean_tensor_as_png

                # configure_ice_rink_mask returns a CPU mask. Preserve that
                # exact calibration image once, before later runs can change
                # the game-directory mask. This does not read video frames.
                mask = self._rink_profile["combined_mask"]
                if mask is not None:
                    if self._snapshot_mask is not None:
                        if not torch.equal(mask, self._snapshot_mask):
                            raise ValueError(
                                "Rink mask changed during a run with an output snapshot"
                            )
                        return {"rink_profile": self._rink_profile}
                    destination = Path(work_dir) / "rink_mask_0.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", dir=destination.parent, delete=False
                    ) as temporary:
                        snapshot = Path(temporary.name)
                    try:
                        save_boolean_tensor_as_png(mask, str(snapshot))
                        os.replace(snapshot, destination)
                        self._snapshot_mask = mask.clone()
                    finally:
                        snapshot.unlink(missing_ok=True)
        if self._rink_profile is None:
            return {}
        return {"rink_profile": self._rink_profile}

    def input_keys(self):
        return {
            "data_samples",
            "original_images",
            "img",
            "inputs",
            "game_id",
            "camera_input_geometry",
            "work_dir",
        }

    def output_keys(self):
        return {"rink_profile"}
