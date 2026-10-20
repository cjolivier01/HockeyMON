"""Team affiliation trunk: labels each track by which side it plays for.

Runs after the tracker, takes a pose-guided torso crop per track, and feeds a
per-game two-way clusterer. The per-frame answer is noisy, so a
:class:`~hmlib.team.team_tracker.TeamTracker` accumulates decayed evidence per
track and only the settled answer is published.

The trunk is read-only with respect to tracking: it publishes ``team_results``
and changes nothing upstream, so a wrong label costs a wrong badge and nothing
else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision.ops import roi_align

from hmlib.aspen.plugins.base import Plugin
from hmlib.bbox.torso import bbox_upper_torso_fallback, torso_roi_from_pose
from hmlib.log import logger
from hmlib.team.backends import CROP_SIZE, build_backend
from hmlib.team.clusterer import OnlineTeamClusterer
from hmlib.team.team_tracker import TeamTracker
from hmlib.team.types import TrackTeamInfo
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, make_channels_first

#: Minimum IoU before a pose is considered to describe a given track.
_POSE_MATCH_IOU = 0.1


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, StreamTensorBase):
        return x.wait()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"cannot convert {type(x).__name__} to a tensor")


def _bbox_from_keypoints(kpts: torch.Tensor) -> torch.Tensor | None:
    if kpts.ndim != 2 or kpts.shape[0] == 0:
        return None
    xs, ys = kpts[:, 0], kpts[:, 1]
    return torch.stack((xs.min(), ys.min(), xs.max(), ys.max()))


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0.0 else 0.0


def _extract_pose_instances(pose_result_item: Any) -> Any | None:
    preds = pose_result_item.get("predictions") if isinstance(pose_result_item, dict) else None
    if isinstance(preds, list) and preds:
        return getattr(preds[0], "pred_instances", None)
    return None


class TeamAffiliationPlugin(Plugin):
    """Assigns tracks to teams and marks officials.

    Expects in context:
      - ``data_samples``: TrackDataSample (or a single-element list)
      - ``original_images``: Tensor [T, C, H, W] or [T, H, W, C]
      - ``pose_results``: optional; falls back to a bbox torso heuristic

    Produces in context:
      - ``team_results``: ``List[List[TrackTeamInfo]]``, one list per frame
    """

    def __init__(
        self,
        enabled: bool = True,
        backend: str = "prtreid",
        warmup_samples: int = 200,
        refit_interval: int = 600,
        min_margin: float = 0.12,
        frame_stride: int = 1,
        min_track_height: int = 60,
        home_is_dark: bool | None = None,
        prtreid_weights: str = "",
        prtreid_backbone_weights: str = "",
        prtreid_device: str = "cuda",
    ) -> None:
        super().__init__(enabled=enabled)
        self._backend_name = str(backend)
        self._frame_stride = max(1, int(frame_stride))
        self._min_track_height = int(min_track_height)
        self._prtreid_kwargs = {
            "prtreid_weights": prtreid_weights,
            "prtreid_backbone_weights": prtreid_backbone_weights,
            "prtreid_device": prtreid_device,
        }

        self._backend = None
        self._clusterer = OnlineTeamClusterer(
            warmup_samples=warmup_samples,
            refit_interval=refit_interval,
            min_margin=min_margin,
        )
        self._tracker = TeamTracker(home_is_dark=home_is_dark)

    def _ensure_backend(self):
        if self._backend is None:
            # Built lazily so constructing the graph never imports PRTReid, and
            # so a missing install surfaces at the first frame with context
            # rather than at import time.
            self._backend = build_backend(self._backend_name, **self._prtreid_kwargs)
            logger.info("Team affiliation using the %r backend", self._backend.name)
        return self._backend

    @property
    def team_tracker(self) -> TeamTracker:
        """Shared aggregator, so the overlay can render the same state."""
        return self._tracker

    def _torso_rois(
        self,
        boxes: torch.Tensor,
        pose_inst: Any | None,
        img_w: int,
        img_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """One torso ROI per track, pose-guided where a pose matches."""
        if pose_inst is None or not hasattr(pose_inst, "keypoints"):
            return [
                bbox_upper_torso_fallback(boxes[i].to(dtype=torch.int64), img_w, img_h)
                for i in range(boxes.shape[0])
            ]

        kpts_all = pose_inst.keypoints
        kps_all = getattr(pose_inst, "keypoint_scores", None)
        pose_boxes = [_bbox_from_keypoints(kpts_all[i]) for i in range(kpts_all.shape[0])]

        rois: list[tuple[int, int, int, int]] = []
        for ti in range(boxes.shape[0]):
            track_box = boxes[ti]
            best_iou, best_j = 0.0, -1
            for pj, pose_box in enumerate(pose_boxes):
                if pose_box is None:
                    continue
                iou = _iou(track_box, pose_box)
                if iou > best_iou:
                    best_iou, best_j = iou, pj

            roi = None
            if best_j >= 0 and best_iou > _POSE_MATCH_IOU:
                roi = torso_roi_from_pose(
                    kpts_all[best_j],
                    None if kps_all is None else kps_all[best_j],
                    img_w,
                    img_h,
                )
            if roi is None:
                roi = bbox_upper_torso_fallback(track_box.to(dtype=torch.int64), img_w, img_h)
            rois.append(roi)
        return rois

    @staticmethod
    def _cut_crops(
        frame: torch.Tensor,
        rois: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Extract every ROI into one uniform ``(N, 3, H, W)`` batch.

        `roi_align` crops and resizes all of them in a single fused kernel,
        which replaces a Python loop that launched a slice and an interpolate
        per skater. Both backends consume the same batch, so the frame is read
        once however many of them are enabled.

        The result is float in 0-255, which is the convention every backend
        assumes. Letting them sniff the range instead would put a `.max()`
        -- a GPU->CPU sync -- on the per-frame path.
        """
        boxes = torch.tensor(
            [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in rois],
            dtype=torch.float32,
            device=frame.device,
        )
        # roi_align wants a batch index column; every ROI comes from this frame.
        indexed = torch.cat((torch.zeros((boxes.shape[0], 1), device=boxes.device), boxes), dim=1)
        return roi_align(
            frame.to(dtype=torch.float32).unsqueeze(0),
            indexed,
            output_size=CROP_SIZE,
            aligned=True,
        )

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            if not track_samples:
                return {}
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        original_images = context.get("original_images")
        if original_images is None:
            return {}
        original_images = make_channels_first(_to_tensor(original_images))
        img_w = int(image_width(original_images))
        img_h = int(image_height(original_images))

        pose_results: list[Any] | None = context.get("pose_results")
        frame_id0 = int(context.get("frame_id", 0))
        backend = self._ensure_backend()

        all_results: list[list[TrackTeamInfo]] = []
        for frame_index, img_data_sample in enumerate(track_data_sample.video_data_samples):
            pred_tracks = getattr(img_data_sample, "pred_track_instances", None)
            if pred_tracks is None:
                all_results.append([])
                continue

            base_mask = get_track_mask(pred_tracks)
            boxes = pred_tracks.bboxes
            if isinstance(boxes, StreamTensorBase):
                boxes = boxes.wait()
            ids = pred_tracks.instances_id
            if isinstance(ids, StreamTensorBase):
                ids = ids.wait()
            if isinstance(base_mask, torch.Tensor):
                boxes = boxes[base_mask.to(device=boxes.device)]
                ids = ids[base_mask.to(device=ids.device)]
            if boxes.numel() == 0:
                all_results.append([])
                continue

            if (frame_index + frame_id0) % self._frame_stride != 0:
                # Still republish the settled labels so the overlay does not
                # blink on skipped frames.
                all_results.append(self._settled_results(ids))
                continue

            pose_inst = None
            if pose_results and frame_index < len(pose_results):
                pose_inst = _extract_pose_instances(pose_results[frame_index])

            frame_img = original_images[frame_index]
            rois = self._torso_rois(boxes, pose_inst, img_w, img_h)

            keep_rois: list[tuple[int, int, int, int]] = []
            crop_track_index: list[int] = []
            for ti, (x1, y1, x2, y2) in enumerate(rois):
                if (y2 - y1) < 4 or (x2 - x1) < 4:
                    continue
                if float(boxes[ti][3] - boxes[ti][1]) < self._min_track_height:
                    # Distant players give a torso a few pixels tall; the mean
                    # colour is mostly ice and would poison the clusterer.
                    continue
                keep_rois.append((x1, y1, x2, y2))
                crop_track_index.append(ti)

            if not keep_rois:
                all_results.append(self._settled_results(ids))
                continue

            out = backend.extract(self._cut_crops(frame_img, keep_rois))
            assignment = self._clusterer.observe(
                features=out.features,
                lightness=out.lightness,
            )
            self._tracker.set_dark_cluster(self._clusterer.dark_cluster)

            frame_id = frame_id0 + frame_index
            for k, ti in enumerate(crop_track_index):
                self._tracker.observe(
                    frame_id=frame_id,
                    info=TrackTeamInfo(
                        tracking_id=int(ids[ti]),
                        cluster=int(assignment.clusters[k]),
                        margin=float(assignment.margins[k]),
                        score=1.0,
                    ),
                )

            all_results.append(self._settled_results(ids))

        return {"team_results": all_results}

    def _settled_results(self, ids: torch.Tensor) -> list[TrackTeamInfo]:
        """Current aggregated label per visible track."""
        results: list[TrackTeamInfo] = []
        for i in range(int(ids.shape[0])):
            tid = int(ids[i])
            results.append(
                TrackTeamInfo(
                    tracking_id=tid,
                    label=self._tracker.label_for(tid),
                )
            )
        return results

    def input_keys(self):
        return {"data_samples", "original_images", "pose_results", "frame_id", "shared"}

    def output_keys(self):
        return {"team_results"}
