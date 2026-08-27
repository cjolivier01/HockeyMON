from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    from hockeymon.core import HmByteTrackConfig  # type: ignore
except Exception:  # pragma: no cover - optional fallback for non-native envs
    HmByteTrackConfig = None  # type: ignore[assignment, misc]

kFrameIdKey = "frame_id"
kBBoxesKey = "bboxes"
kLabelsKey = "labels"
kScoresKey = "scores"
kIdsKey = "ids"
kNumDetectionsKey = "num_detections"
kNumTracksKey = "num_tracks"


@dataclass(frozen=True)
class _ByteTrackConfig:
    init_track_thr: float = 0.7
    obj_score_thrs_low: float = 0.1
    obj_score_thrs_high: float = 0.6
    match_iou_thrs_high: float = 0.1
    match_iou_thrs_low: float = 0.5
    match_iou_thrs_tentative: float = 0.3
    num_frames_to_keep_lost_tracks: int = 30
    weight_iou_with_det_scores: bool = True
    num_tentatives: int = 3

    @staticmethod
    def from_obj(config: object) -> "_ByteTrackConfig":
        def _f(name: str, default: float) -> float:
            try:
                return float(getattr(config, name))
            except Exception:
                return float(default)

        def _i(name: str, default: int) -> int:
            try:
                return int(getattr(config, name))
            except Exception:
                return int(default)

        def _b(name: str, default: bool) -> bool:
            try:
                return bool(getattr(config, name))
            except Exception:
                return bool(default)

        track_buffer = _i("track_buffer_size", 30)
        return _ByteTrackConfig(
            init_track_thr=_f("init_track_thr", 0.7),
            obj_score_thrs_low=_f("obj_score_thrs_low", 0.1),
            obj_score_thrs_high=_f("obj_score_thrs_high", 0.6),
            match_iou_thrs_high=_f("match_iou_thrs_high", 0.1),
            match_iou_thrs_low=_f("match_iou_thrs_low", 0.5),
            match_iou_thrs_tentative=_f("match_iou_thrs_tentative", 0.3),
            num_frames_to_keep_lost_tracks=_i("num_frames_to_keep_lost_tracks", track_buffer),
            weight_iou_with_det_scores=_b("weight_iou_with_det_scores", True),
            num_tentatives=_i("num_tentatives", 3),
        )


def _copy_prefix(dst: torch.Tensor, src: torch.Tensor, count: int) -> None:
    if count <= 0:
        return
    dst[:count].copy_(src)


def _stable_mask_order(mask: torch.Tensor) -> torch.Tensor:
    """Return indices that move True entries to the front, preserving order."""
    if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    mask_bool = mask.to(dtype=torch.bool)
    n = int(mask_bool.shape[0])
    idx = torch.arange(n, device=mask_bool.device, dtype=torch.long)
    # Key ensures stable ordering: True -> 0..n-1, False -> (n+1)+idx.
    key = (~mask_bool).to(dtype=idx.dtype) * (n + 1) + idx
    return torch.argsort(key)


def _bbox_xyxy_to_cxcyah(bboxes: torch.Tensor) -> torch.Tensor:
    if bboxes.numel() == 0:
        return bboxes.reshape(0, 4)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise RuntimeError("bboxes must have shape [N, 4]")
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    aspect_ratio = w / torch.clamp(h, min=1e-6)
    return torch.stack((cx, cy, aspect_ratio, h), dim=1)


def _bbox_cxcyah_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    if bboxes.numel() == 0:
        return bboxes.reshape(0, 4)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise RuntimeError("bboxes must have shape [N, 4] (cx, cy, a, h)")
    cx = bboxes[:, 0:1]
    cy = bboxes[:, 1:2]
    ratio = bboxes[:, 2:3]
    h = bboxes[:, 3:4]
    w = ratio * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.cat((x1, y1, x2, y2), dim=1)


def _bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    if boxes1.ndim != 2 or boxes1.shape[1] != 4 or boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise RuntimeError("boxes must have shape [N, 4]")

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = torch.clamp(rb - lt, min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    area1_wh = torch.clamp(boxes1[:, 2:] - boxes1[:, :2], min=0.0)
    area2_wh = torch.clamp(boxes2[:, 2:] - boxes2[:, :2], min=0.0)
    area1 = area1_wh[:, 0] * area1_wh[:, 1]
    area2 = area2_wh[:, 0] * area2_wh[:, 1]

    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _hungarian_assign(
    cost: torch.Tensor,
    *,
    cost_limit: float,
    row_mask: Optional[torch.Tensor] = None,
    col_mask: Optional[torch.Tensor] = None,
    big: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (track_to_det, det_to_track) with `-1` meaning unmatched.

    Torch-only greedy assignment (no NumPy/SciPy). Minimizes total cost
    subject to `cost <= cost_limit`, leaving unmatched as `-1`.
    """
    n_tracks = int(cost.shape[0])
    n_dets = int(cost.shape[1])
    if n_tracks <= 0:
        return (
            cost.new_empty((0,), dtype=torch.long),
            cost.new_full((n_dets,), -1, dtype=torch.long),
        )
    if n_dets <= 0:
        return (
            cost.new_full((n_tracks,), -1, dtype=torch.long),
            cost.new_empty((0,), dtype=torch.long),
        )

    if big is None:
        big = cost.new_full((), 1e9)
    elif big.device != cost.device or big.dtype != cost.dtype:
        big = big.to(device=cost.device, dtype=cost.dtype)
    cost_work = torch.where(cost <= float(cost_limit), cost, big)

    track_to_det = cost.new_full((n_tracks,), -1, dtype=torch.long)
    det_to_track = cost.new_full((n_dets,), -1, dtype=torch.long)
    row_used = cost.new_zeros((n_tracks,), dtype=torch.bool)
    col_used = cost.new_zeros((n_dets,), dtype=torch.bool)
    if row_mask is not None:
        row_used |= ~row_mask.to(device=cost.device, dtype=torch.bool)
    if col_mask is not None:
        col_used |= ~col_mask.to(device=cost.device, dtype=torch.bool)

    for _ in range(min(n_tracks, n_dets)):
        row_pen = row_used.to(dtype=cost.dtype).unsqueeze(1) * big
        col_pen = col_used.to(dtype=cost.dtype).unsqueeze(0) * big
        masked_cost = cost_work + row_pen + col_pen

        flat_idx = masked_cost.reshape(-1).argmin()
        flat_idx_1 = flat_idx.view(1)
        min_cost = masked_cost.reshape(-1).gather(0, flat_idx_1)

        row = torch.div(flat_idx, n_dets, rounding_mode="floor").to(dtype=torch.long)
        col = flat_idx.remainder(n_dets).to(dtype=torch.long)
        row_1 = row.view(1)
        col_1 = col.view(1)

        is_valid = min_cost <= float(cost_limit)

        cur_td = track_to_det.gather(0, row_1)
        cur_dt = det_to_track.gather(0, col_1)
        track_to_det.scatter_(0, row_1, torch.where(is_valid, col_1, cur_td))
        det_to_track.scatter_(0, col_1, torch.where(is_valid, row_1, cur_dt))

        cur_ru = row_used.gather(0, row_1)
        cur_cu = col_used.gather(0, col_1)
        row_used.scatter_(0, row_1, cur_ru | is_valid)
        col_used.scatter_(0, col_1, cur_cu | is_valid)

    return track_to_det, det_to_track


class HmByteTrackerCuda:
    """Pure-Python BYTETracker implementation (torch-only, GPU friendly)."""

    _STATE_TENTATIVE = 0
    _STATE_TRACKING = 1
    _STATE_LOST = 2

    def __init__(
        self,
        config: Optional[HmByteTrackConfig] = None,
        *,
        max_detections: int | None = None,
        max_tracks: int | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        if config is None:
            config = HmByteTrackConfig() if HmByteTrackConfig is not None else object()  # type: ignore[call-arg]
        self._config = _ByteTrackConfig.from_obj(config)
        self._device = torch.device(device)

        def _maybe_int_attr(name: str, default: int) -> int:
            try:
                return int(getattr(config, name))
            except Exception:
                return int(default)

        self._max_detections = (
            int(max_detections)
            if max_detections is not None
            else _maybe_int_attr("max_detections", 256)
        )
        self._max_tracks = (
            int(max_tracks)
            if max_tracks is not None
            else _maybe_int_attr("max_tracks", min(self._max_detections, 256))
        )
        if self._max_tracks > self._max_detections:
            self._max_tracks = self._max_detections

        self._motion_mat = torch.eye(8, device=self._device, dtype=torch.float32)
        for i in range(4):
            self._motion_mat[i, 4 + i] = 1.0
        self._motion_mat_T = self._motion_mat.transpose(0, 1).contiguous()

        self._update_mat = torch.zeros((4, 8), device=self._device, dtype=torch.float32)
        self._update_mat[:, :4] = torch.eye(4, device=self._device, dtype=torch.float32)
        self._update_mat_T = self._update_mat.transpose(0, 1).contiguous()

        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

        self._static_tracker: Optional[HmByteTrackerCudaStatic] = None
        self.reset()
        if self._device.type == "cuda":
            self._static_tracker = HmByteTrackerCudaStatic(
                config,
                max_detections=self._max_detections,
                max_tracks=self._max_tracks,
                device=self._device,
            )

    def reset(self) -> None:
        self._track_ids = torch.empty((0,), dtype=torch.long, device=self._device)
        self._track_states = torch.empty((0,), dtype=torch.long, device=self._device)
        self._track_labels = torch.empty((0,), dtype=torch.long, device=self._device)
        self._track_scores = torch.empty((0,), dtype=torch.float32, device=self._device)
        self._track_last_frame = torch.empty((0,), dtype=torch.long, device=self._device)
        self._track_hits = torch.empty((0,), dtype=torch.long, device=self._device)
        self._track_mean = torch.empty((0, 8), dtype=torch.float32, device=self._device)
        self._track_covariance = torch.empty((0, 8, 8), dtype=torch.float32, device=self._device)
        self._next_track_id = 0
        self._track_calls_since_last_empty = 0
        if self._static_tracker is not None:
            self._static_tracker.reset()

    @property
    def max_detections(self) -> int:
        return self._max_detections

    @property
    def max_tracks(self) -> int:
        return self._max_tracks

    def num_tracks(self) -> int:
        if self._static_tracker is not None:
            return int(self._static_tracker.num_tracks())
        return int(self._track_ids.numel())

    def _track_static(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._static_tracker is None:
            raise RuntimeError("static tracker is not initialized")
        frame_id_tensor = data[kFrameIdKey]

        bboxes = self._ensure_bboxes(data[kBBoxesKey])
        labels = self._ensure_vector(data[kLabelsKey], dtype=torch.long)
        scores = self._ensure_vector(data[kScoresKey], dtype=torch.float32)

        if bboxes.shape[0] != labels.shape[0] or labels.shape[0] != scores.shape[0]:
            raise RuntimeError("bboxes/labels/scores must have matching length")

        if (
            kNumDetectionsKey in data
            and bboxes.shape[0] == self._max_detections
            and labels.shape[0] == self._max_detections
            and scores.shape[0] == self._max_detections
        ):
            num_detections = (
                data[kNumDetectionsKey].to(device=self._device, dtype=torch.long).reshape(-1)[:1]
            )
            payload = {
                kFrameIdKey: frame_id_tensor,
                kBBoxesKey: bboxes,
                kLabelsKey: labels,
                kScoresKey: scores,
                kNumDetectionsKey: num_detections,
            }
            return self._static_tracker.track(payload)

        total = int(bboxes.shape[0])
        kept = min(total, self._max_detections)
        if total > self._max_detections:
            keep_idx = torch.topk(scores, k=self._max_detections).indices
            keep_idx, _ = torch.sort(keep_idx)
            bboxes = bboxes.index_select(0, keep_idx)
            labels = labels.index_select(0, keep_idx)
            scores = scores.index_select(0, keep_idx)

        padded_bboxes = bboxes.new_zeros((self._max_detections, 4))
        padded_labels = labels.new_zeros((self._max_detections,))
        padded_scores = scores.new_zeros((self._max_detections,))
        if kept:
            padded_bboxes[:kept].copy_(bboxes[:kept])
            padded_labels[:kept].copy_(labels[:kept])
            padded_scores[:kept].copy_(scores[:kept])

        kept_t = bboxes.new_tensor([kept], dtype=torch.long)
        if kNumDetectionsKey in data:
            num_detections = (
                data[kNumDetectionsKey].to(device=self._device, dtype=torch.long).reshape(-1)[:1]
            )
            num_detections = torch.min(num_detections, kept_t)
        else:
            num_detections = kept_t

        payload = {
            kFrameIdKey: frame_id_tensor,
            kBBoxesKey: padded_bboxes,
            kLabelsKey: padded_labels,
            kScoresKey: padded_scores,
            kNumDetectionsKey: num_detections,
        }
        return self._static_tracker.track(payload)

    def _ensure_bboxes(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.to(device=self._device, dtype=torch.float32).reshape(0, 4)
        if tensor.ndim == 1:
            if tensor.shape[0] != 4:
                raise RuntimeError("bbox tensor must have 4 elements")
            return tensor.to(device=self._device, dtype=torch.float32).unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != 4:
            raise RuntimeError("bbox tensor must have shape [N, 4]")
        return tensor.to(device=self._device, dtype=torch.float32)

    def _ensure_vector(self, tensor: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor.to(device=self._device, dtype=dtype).reshape(0)
        if tensor.ndim == 0:
            return tensor.to(device=self._device, dtype=dtype).unsqueeze(0)
        return tensor.to(device=self._device, dtype=dtype)

    def _mask_indices(self, mask: torch.Tensor) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self._device)
        mask_bool = mask.to(dtype=torch.bool)
        indices = torch.arange(mask_bool.shape[0], dtype=torch.long, device=mask_bool.device)
        return indices.masked_select(mask_bool)

    def _kalman_initiate(
        self, measurements_cxcyah: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        count = int(measurements_cxcyah.shape[0])
        mean = measurements_cxcyah.new_zeros((count, 8))
        mean[:, 0:4] = measurements_cxcyah
        h = measurements_cxcyah[:, 3]
        std_pos = torch.stack(
            (
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                torch.full_like(h, 1e-2),
                2 * self._std_weight_position * h,
            ),
            dim=1,
        )
        std_vel = torch.stack(
            (
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                torch.full_like(h, 1e-5),
                10 * self._std_weight_velocity * h,
            ),
            dim=1,
        )
        std = torch.cat((std_pos, std_vel), dim=1)
        cov = torch.diag_embed(std.pow(2))
        return mean, cov

    def _kalman_predict(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean.numel() == 0:
            return mean, covariance
        h = mean[:, 3]
        std_pos = torch.stack(
            (
                self._std_weight_position * h,
                self._std_weight_position * h,
                torch.full_like(h, 1e-2),
                self._std_weight_position * h,
            ),
            dim=1,
        )
        std_vel = torch.stack(
            (
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                torch.full_like(h, 1e-5),
                self._std_weight_velocity * h,
            ),
            dim=1,
        )
        std = torch.cat((std_pos, std_vel), dim=1)
        motion_cov = torch.diag_embed(std.pow(2))

        mean = mean @ self._motion_mat_T
        motion = self._motion_mat.unsqueeze(0).expand_as(covariance)
        motion_T = self._motion_mat_T.unsqueeze(0).expand_as(covariance)
        covariance = motion @ (covariance @ motion_T) + motion_cov
        return mean, covariance

    def _kalman_project(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        measurement_mean = mean[:, 0:4]
        h = mean[:, 3]
        std = torch.stack(
            (
                self._std_weight_position * h,
                self._std_weight_position * h,
                torch.full_like(h, 1e-1),
                self._std_weight_position * h,
            ),
            dim=1,
        )
        projected_cov = covariance[:, 0:4, 0:4] + torch.diag_embed(std.pow(2))
        return measurement_mean, projected_cov

    def _kalman_update(
        self, mean: torch.Tensor, covariance: torch.Tensor, measurement_cxcyah: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_mean, projected_cov = self._kalman_project(mean, covariance)

        B = covariance @ self._update_mat_T
        BT = B.transpose(1, 2)
        chol, _ = torch.linalg.cholesky_ex(projected_cov, check_errors=False)
        sol = torch.cholesky_solve(BT, chol)
        kalman_gain = sol.transpose(1, 2)

        innovation = measurement_cxcyah - projected_mean
        delta = (innovation.unsqueeze(1) @ kalman_gain.transpose(1, 2)).squeeze(1)
        new_mean = mean + delta

        temp = projected_cov @ kalman_gain.transpose(1, 2)
        new_covariance = covariance - (kalman_gain @ temp)
        return new_mean, new_covariance

    def _predict_tracks(self, frame_id: torch.Tensor) -> None:
        if self._track_ids.numel() == 0:
            return
        confirmed = self._track_states.ne(self._STATE_TENTATIVE)
        not_prev_frame = self._track_last_frame.ne(frame_id - 1)
        degrade = confirmed & not_prev_frame
        degrade_idx = self._mask_indices(degrade)
        if degrade_idx.numel() > 0:
            self._track_mean[degrade_idx, 7] = self._track_mean[degrade_idx, 7] / 2.0
        self._track_mean, self._track_covariance = self._kalman_predict(
            self._track_mean, self._track_covariance
        )

    def _remove_stale_tracks(self, frame_id: torch.Tensor) -> None:
        if self._track_ids.numel() == 0:
            return
        lost_mask = self._track_states.eq(self._STATE_LOST)
        frame_delta = frame_id - self._track_last_frame
        too_long = lost_mask & frame_delta.ge(int(self._config.num_frames_to_keep_lost_tracks))
        stale_tent = self._track_states.eq(self._STATE_TENTATIVE) & self._track_last_frame.ne(
            frame_id
        )
        remove_mask = too_long | stale_tent
        remove_idx = self._mask_indices(remove_mask)
        if remove_idx.numel() == 0:
            return
        keep_idx = self._mask_indices(~remove_mask)
        self._track_ids = self._track_ids.index_select(0, keep_idx)
        self._track_states = self._track_states.index_select(0, keep_idx)
        self._track_labels = self._track_labels.index_select(0, keep_idx)
        self._track_scores = self._track_scores.index_select(0, keep_idx)
        self._track_last_frame = self._track_last_frame.index_select(0, keep_idx)
        self._track_hits = self._track_hits.index_select(0, keep_idx)
        self._track_mean = self._track_mean.index_select(0, keep_idx)
        self._track_covariance = self._track_covariance.index_select(0, keep_idx)
        if self._track_ids.numel() == 0:
            self._track_calls_since_last_empty = 0

    def _mark_unmatched_tracking(self, matched_indices: torch.Tensor) -> None:
        tracking_mask = self._track_states.eq(self._STATE_TRACKING)
        matched_mask = torch.zeros_like(self._track_ids, dtype=torch.bool)
        if matched_indices.numel() > 0:
            matched_mask[matched_indices] = True
        to_lose = tracking_mask & (~matched_mask)
        lose_idx = self._mask_indices(to_lose)
        if lose_idx.numel() > 0:
            self._track_states[lose_idx] = self._STATE_LOST

    def _update_tracks(
        self,
        track_indices: torch.Tensor,
        detections_xyxy: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        frame_id: torch.Tensor,
    ) -> None:
        if track_indices.numel() == 0:
            return
        measurements = _bbox_xyxy_to_cxcyah(detections_xyxy)
        mean_subset = self._track_mean.index_select(0, track_indices)
        cov_subset = self._track_covariance.index_select(0, track_indices)
        new_mean, new_cov = self._kalman_update(mean_subset, cov_subset, measurements)

        self._track_mean.index_put_((track_indices,), new_mean)
        self._track_covariance.index_put_((track_indices,), new_cov)
        self._track_scores.index_put_((track_indices,), scores)
        self._track_labels.index_put_((track_indices,), labels)
        self._track_last_frame[track_indices] = frame_id

        hits = self._track_hits.index_select(0, track_indices) + torch.ones_like(track_indices)
        self._track_hits.index_put_((track_indices,), hits)

        states = self._track_states.index_select(0, track_indices)
        tent_mask = states.eq(self._STATE_TENTATIVE) & hits.ge(int(self._config.num_tentatives))
        tent_idx = self._mask_indices(tent_mask)
        if tent_idx.numel() > 0:
            to_activate = track_indices.index_select(0, tent_idx)
            self._track_states[to_activate] = self._STATE_TRACKING

        lost_mask = states.eq(self._STATE_LOST)
        lost_idx = self._mask_indices(lost_mask)
        if lost_idx.numel() > 0:
            to_activate = track_indices.index_select(0, lost_idx)
            self._track_states[to_activate] = self._STATE_TRACKING

    def _init_new_tracks(
        self,
        ids: torch.Tensor,
        detections_xyxy: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        frame_id: torch.Tensor,
    ) -> None:
        if ids.numel() == 0:
            return
        measurements = _bbox_xyxy_to_cxcyah(detections_xyxy)
        new_mean, new_cov = self._kalman_initiate(measurements)

        if self._track_ids.numel() == 0:
            self._track_ids = ids.clone()
            state_val = (
                self._STATE_TRACKING
                if self._track_calls_since_last_empty == 0
                else self._STATE_TENTATIVE
            )
            self._track_states = torch.full(
                (ids.shape[0],), int(state_val), dtype=torch.long, device=self._device
            )
            self._track_labels = labels.clone()
            self._track_scores = scores.clone()
            self._track_last_frame = ids.new_zeros(ids.shape) + frame_id
            self._track_hits = torch.ones_like(ids)
            self._track_mean = new_mean
            self._track_covariance = new_cov
            return

        self._track_ids = torch.cat((self._track_ids, ids), dim=0)
        state_val = (
            self._STATE_TRACKING
            if self._track_calls_since_last_empty == 0
            else self._STATE_TENTATIVE
        )
        new_states = torch.full(
            (ids.shape[0],), int(state_val), dtype=torch.long, device=self._device
        )
        self._track_states = torch.cat((self._track_states, new_states), dim=0)
        self._track_labels = torch.cat((self._track_labels, labels), dim=0)
        self._track_scores = torch.cat((self._track_scores, scores), dim=0)
        frame_tensor = ids.new_zeros(ids.shape) + frame_id
        self._track_last_frame = torch.cat((self._track_last_frame, frame_tensor), dim=0)
        self._track_hits = torch.cat((self._track_hits, torch.ones_like(ids)), dim=0)
        self._track_mean = torch.cat((self._track_mean, new_mean), dim=0)
        self._track_covariance = torch.cat((self._track_covariance, new_cov), dim=0)

    def _assign_tracks(
        self,
        track_indices: torch.Tensor,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        *,
        weight_with_scores: bool,
        iou_thr: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if track_indices.numel() == 0 or det_bboxes.shape[0] == 0:
            return (
                torch.full((track_indices.shape[0],), -1, dtype=torch.long, device=self._device),
                torch.full((det_bboxes.shape[0],), -1, dtype=torch.long, device=self._device),
            )

        means = self._track_mean.index_select(0, track_indices)[:, 0:4]
        track_boxes = _bbox_cxcyah_to_xyxy(means)
        det_boxes = det_bboxes
        ious = _bbox_iou(track_boxes, det_boxes)
        if weight_with_scores and det_scores.numel() > 0:
            ious = ious * det_scores.unsqueeze(0)

        track_labels = self._track_labels.index_select(0, track_indices)
        label_ok = (track_labels.unsqueeze(1) == det_labels.unsqueeze(0)).to(dtype=ious.dtype)
        cate_cost = (1.0 - label_ok) * 1e6
        cost = (1.0 - ious) + cate_cost
        cost_limit = 1.0 - float(iou_thr)
        return _hungarian_assign(cost, cost_limit=cost_limit)

    def track(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._device.type == "cuda" and self._static_tracker is not None:
            return self._track_static(data)
        frame_id_tensor = data[kFrameIdKey]
        if frame_id_tensor.numel() < 1:
            raise RuntimeError("frame_id tensor must contain a value")
        frame_id = frame_id_tensor.to(device=self._device, dtype=torch.long).reshape(-1)[:1]
        if frame_id.numel() < 1:
            raise RuntimeError("frame_id tensor must contain a value")
        frame_id = frame_id[0]

        bboxes = self._ensure_bboxes(data[kBBoxesKey])
        labels = self._ensure_vector(data[kLabelsKey], dtype=torch.long)
        scores = self._ensure_vector(data[kScoresKey], dtype=torch.float32)

        if bboxes.shape[0] != labels.shape[0] or labels.shape[0] != scores.shape[0]:
            raise RuntimeError("bboxes/labels/scores must have matching length")

        if self._track_ids.numel() == 0:
            self._track_calls_since_last_empty = 0
            keep = scores > float(self._config.init_track_thr)
            keep_idx = self._mask_indices(keep)
            if keep_idx.numel() == 0:
                data[kIdsKey] = torch.empty((0,), dtype=torch.long, device=self._device)
                data[kBBoxesKey] = torch.empty((0, 4), dtype=torch.float32, device=self._device)
                data[kLabelsKey] = torch.empty((0,), dtype=torch.long, device=self._device)
                data[kScoresKey] = torch.empty((0,), dtype=torch.float32, device=self._device)
                return data

            bboxes = bboxes.index_select(0, keep_idx)
            labels = labels.index_select(0, keep_idx)
            scores = scores.index_select(0, keep_idx)

            num = int(bboxes.shape[0])
            ids = torch.arange(
                self._next_track_id,
                self._next_track_id + num,
                dtype=torch.long,
                device=self._device,
            )
            self._next_track_id += num
            self._init_new_tracks(ids, bboxes, labels, scores, frame_id)

            data[kIdsKey] = ids
            data[kBBoxesKey] = bboxes
            data[kLabelsKey] = labels
            data[kScoresKey] = scores
            return data

        self._track_calls_since_last_empty += 1
        if bboxes.shape[0] == 0:
            ids = torch.empty((0,), dtype=torch.long, device=self._device)
            self._mark_unmatched_tracking(torch.empty((0,), dtype=torch.long, device=self._device))
            self._remove_stale_tracks(frame_id)
            data[kIdsKey] = ids
            data[kBBoxesKey] = bboxes
            data[kLabelsKey] = labels
            data[kScoresKey] = scores
            return data

        det_count = int(bboxes.shape[0])
        ids = torch.full((det_count,), -1, dtype=torch.long, device=self._device)

        first_mask = scores > float(self._config.obj_score_thrs_high)
        second_mask = (~first_mask) & (scores > float(self._config.obj_score_thrs_low))
        first_idx = self._mask_indices(first_mask)
        second_idx = self._mask_indices(second_mask)

        first_bboxes = bboxes.index_select(0, first_idx) if first_idx.numel() else bboxes[:0]
        first_labels = labels.index_select(0, first_idx) if first_idx.numel() else labels[:0]
        first_scores = scores.index_select(0, first_idx) if first_idx.numel() else scores[:0]

        second_bboxes = bboxes.index_select(0, second_idx) if second_idx.numel() else bboxes[:0]
        second_labels = labels.index_select(0, second_idx) if second_idx.numel() else labels[:0]
        second_scores = scores.index_select(0, second_idx) if second_idx.numel() else scores[:0]

        self._predict_tracks(frame_id)

        confirmed_mask = self._track_states.ne(self._STATE_TENTATIVE)
        unconfirmed_mask = self._track_states.eq(self._STATE_TENTATIVE)
        confirmed_idx = self._mask_indices(confirmed_mask)
        unconfirmed_idx = self._mask_indices(unconfirmed_mask)

        first_det_ids = torch.full(
            (first_bboxes.shape[0],), -1, dtype=torch.long, device=self._device
        )
        first_track_indices = torch.full_like(first_det_ids, -1)

        first_track_to_det, first_det_to_track = self._assign_tracks(
            confirmed_idx,
            first_bboxes,
            first_labels,
            first_scores,
            weight_with_scores=bool(self._config.weight_iou_with_det_scores),
            iou_thr=float(self._config.match_iou_thrs_high),
        )

        if first_det_to_track.numel() > 0:
            valid_det = first_det_to_track.ge(0)
            det_valid_idx = self._mask_indices(valid_det)
            if det_valid_idx.numel() > 0:
                matched_tracks_subset = first_det_to_track.index_select(0, det_valid_idx)
                matched_tracks_global = confirmed_idx.index_select(0, matched_tracks_subset)
                matched_track_ids = self._track_ids.index_select(0, matched_tracks_global)
                matched_det_global = first_idx.index_select(0, det_valid_idx)
                ids.index_put_((matched_det_global,), matched_track_ids)
                first_det_ids.index_put_((det_valid_idx,), matched_track_ids)
                first_track_indices.index_put_((det_valid_idx,), matched_tracks_global)

        unmatched_first_mask = first_det_ids.lt(0)
        unmatched_first_idx = self._mask_indices(unmatched_first_mask)
        first_unmatch_bboxes = (
            first_bboxes.index_select(0, unmatched_first_idx)
            if unmatched_first_idx.numel()
            else first_bboxes[:0]
        )
        first_unmatch_labels = (
            first_labels.index_select(0, unmatched_first_idx)
            if unmatched_first_idx.numel()
            else first_labels[:0]
        )
        first_unmatch_scores = (
            first_scores.index_select(0, unmatched_first_idx)
            if unmatched_first_idx.numel()
            else first_scores[:0]
        )
        first_unmatch_det_ids = (
            first_det_ids.index_select(0, unmatched_first_idx).clone()
            if unmatched_first_idx.numel()
            else first_det_ids[:0]
        )
        first_unmatch_track_indices = (
            first_track_indices.index_select(0, unmatched_first_idx).clone()
            if unmatched_first_idx.numel()
            else first_track_indices[:0]
        )

        if unconfirmed_idx.numel() > 0 and first_unmatch_bboxes.shape[0] > 0:
            tent_track_to_det, tent_det_to_track = self._assign_tracks(
                unconfirmed_idx,
                first_unmatch_bboxes,
                first_unmatch_labels,
                first_unmatch_scores,
                weight_with_scores=bool(self._config.weight_iou_with_det_scores),
                iou_thr=float(self._config.match_iou_thrs_tentative),
            )
            if tent_det_to_track.numel() > 0:
                valid_det = tent_det_to_track.ge(0)
                det_valid_idx = self._mask_indices(valid_det)
                if det_valid_idx.numel() > 0:
                    matched_tracks_subset = tent_det_to_track.index_select(0, det_valid_idx)
                    matched_tracks_global = unconfirmed_idx.index_select(0, matched_tracks_subset)
                    matched_track_ids = self._track_ids.index_select(0, matched_tracks_global)
                    global_det_idx = unmatched_first_idx.index_select(0, det_valid_idx)
                    ids.index_put_((first_idx.index_select(0, global_det_idx),), matched_track_ids)
                    first_unmatch_det_ids.index_put_((det_valid_idx,), matched_track_ids)
                    first_unmatch_track_indices.index_put_((det_valid_idx,), matched_tracks_global)

        track_unmatched_mask = first_track_to_det.lt(0)
        recent_mask = self._track_last_frame.index_select(0, confirmed_idx).eq(frame_id - 1)
        selectable_mask = track_unmatched_mask & recent_mask
        second_selected_idx = self._mask_indices(selectable_mask)

        second_det_ids = torch.full(
            (second_bboxes.shape[0],), -1, dtype=torch.long, device=self._device
        )
        second_track_indices = torch.full_like(second_det_ids, -1)
        if second_selected_idx.numel() > 0 and second_bboxes.shape[0] > 0:
            selected_tracks = confirmed_idx.index_select(0, second_selected_idx)
            second_track_to_det, second_det_to_track = self._assign_tracks(
                selected_tracks,
                second_bboxes,
                second_labels,
                second_scores,
                weight_with_scores=False,
                iou_thr=float(self._config.match_iou_thrs_low),
            )
            if second_det_to_track.numel() > 0:
                valid_det = second_det_to_track.ge(0)
                det_valid_idx = self._mask_indices(valid_det)
                if det_valid_idx.numel() > 0:
                    matched_tracks_subset = second_det_to_track.index_select(0, det_valid_idx)
                    matched_tracks_global = selected_tracks.index_select(0, matched_tracks_subset)
                    matched_track_ids = self._track_ids.index_select(0, matched_tracks_global)
                    matched_det_global = second_idx.index_select(0, det_valid_idx)
                    ids.index_put_((matched_det_global,), matched_track_ids)
                    second_det_ids.index_put_((det_valid_idx,), matched_track_ids)
                    second_track_indices.index_put_((det_valid_idx,), matched_tracks_global)

        first_match_valid = first_det_ids.ge(0)
        first_match_valid_idx = self._mask_indices(first_match_valid)
        first_match_det_bboxes = (
            first_bboxes.index_select(0, first_match_valid_idx)
            if first_match_valid_idx.numel()
            else first_bboxes[:0]
        )
        first_match_det_labels = (
            first_labels.index_select(0, first_match_valid_idx)
            if first_match_valid_idx.numel()
            else first_labels[:0]
        )
        first_match_det_scores = (
            first_scores.index_select(0, first_match_valid_idx)
            if first_match_valid_idx.numel()
            else first_scores[:0]
        )
        first_match_det_ids = (
            first_det_ids.index_select(0, first_match_valid_idx)
            if first_match_valid_idx.numel()
            else first_det_ids[:0]
        )
        first_match_track_indices = (
            first_track_indices.index_select(0, first_match_valid_idx)
            if first_match_valid_idx.numel()
            else first_track_indices[:0]
        )

        second_valid = second_det_ids.ge(0)
        second_valid_idx = self._mask_indices(second_valid)
        second_match_bboxes = (
            second_bboxes.index_select(0, second_valid_idx)
            if second_valid_idx.numel()
            else second_bboxes[:0]
        )
        second_match_labels = (
            second_labels.index_select(0, second_valid_idx)
            if second_valid_idx.numel()
            else second_labels[:0]
        )
        second_match_scores = (
            second_scores.index_select(0, second_valid_idx)
            if second_valid_idx.numel()
            else second_scores[:0]
        )
        second_match_ids = (
            second_det_ids.index_select(0, second_valid_idx)
            if second_valid_idx.numel()
            else second_det_ids[:0]
        )
        second_match_track_indices = (
            second_track_indices.index_select(0, second_valid_idx)
            if second_valid_idx.numel()
            else second_track_indices[:0]
        )

        out_bboxes = torch.cat(
            (first_match_det_bboxes, first_unmatch_bboxes, second_match_bboxes), dim=0
        )
        out_labels = torch.cat(
            (first_match_det_labels, first_unmatch_labels, second_match_labels), dim=0
        )
        out_scores = torch.cat(
            (first_match_det_scores, first_unmatch_scores, second_match_scores), dim=0
        )
        out_ids = torch.cat((first_match_det_ids, first_unmatch_det_ids, second_match_ids), dim=0)
        out_track_indices = torch.cat(
            (first_match_track_indices, first_unmatch_track_indices, second_match_track_indices),
            dim=0,
        )

        new_track_mask = out_ids.lt(0)
        new_track_idx = self._mask_indices(new_track_mask)
        if new_track_idx.numel() > 0:
            new_count = int(new_track_idx.numel())
            new_ids = torch.arange(
                self._next_track_id,
                self._next_track_id + new_count,
                dtype=torch.long,
                device=self._device,
            )
            self._next_track_id += new_count
            out_ids.index_put_((new_track_idx,), new_ids)

        matched_mask = out_track_indices.ge(0)
        matched_det_idx = self._mask_indices(matched_mask)
        matched_global_idx = (
            out_track_indices.index_select(0, matched_det_idx)
            if matched_det_idx.numel()
            else out_track_indices[:0]
        )
        self._mark_unmatched_tracking(matched_global_idx)

        if matched_det_idx.numel() > 0:
            update_track_indices = out_track_indices.index_select(0, matched_det_idx)
            self._update_tracks(
                update_track_indices,
                out_bboxes.index_select(0, matched_det_idx),
                out_labels.index_select(0, matched_det_idx),
                out_scores.index_select(0, matched_det_idx),
                frame_id,
            )

        if new_track_idx.numel() > 0:
            self._init_new_tracks(
                out_ids.index_select(0, new_track_idx),
                out_bboxes.index_select(0, new_track_idx),
                out_labels.index_select(0, new_track_idx),
                out_scores.index_select(0, new_track_idx),
                frame_id,
            )

        self._remove_stale_tracks(frame_id)

        data[kIdsKey] = out_ids
        data[kBBoxesKey] = out_bboxes
        data[kLabelsKey] = out_labels
        data[kScoresKey] = out_scores
        return data


class HmByteTrackerCudaStatic:
    """Static-shape CUDA BYTETracker implemented in Python (no dynamic indexing)."""

    _STATE_TENTATIVE = 0
    _STATE_TRACKING = 1
    _STATE_LOST = 2

    def __init__(
        self,
        config: HmByteTrackConfig | None = None,
        max_detections: int = 256,
        max_tracks: int = 256,
        device: str | torch.device = "cuda:0",
    ) -> None:
        if config is None:
            config = HmByteTrackConfig() if HmByteTrackConfig is not None else object()  # type: ignore[call-arg]
        self._config = _ByteTrackConfig.from_obj(config)
        self._max_detections = int(max_detections)
        self._max_tracks = int(max_tracks)
        if self._max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if self._max_tracks <= 0:
            raise ValueError("max_tracks must be positive")
        if self._max_tracks > self._max_detections:
            raise ValueError("max_tracks must be <= max_detections for static tracker")
        self._device = torch.device(device)

        self._motion_mat = torch.eye(8, device=self._device, dtype=torch.float32)
        for i in range(4):
            self._motion_mat[i, 4 + i] = 1.0
        self._motion_mat_T = self._motion_mat.transpose(0, 1).contiguous()

        self._update_mat = torch.zeros((4, 8), device=self._device, dtype=torch.float32)
        self._update_mat[:, :4] = torch.eye(4, device=self._device, dtype=torch.float32)
        self._update_mat_T = self._update_mat.transpose(0, 1).contiguous()

        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

        self.reset()

    @property
    def max_detections(self) -> int:
        return self._max_detections

    @property
    def max_tracks(self) -> int:
        return self._max_tracks

    def num_tracks(self) -> int:
        return int(self._track_active.sum().item())

    def reset(self) -> None:
        self._track_ids = torch.full(
            (self._max_tracks,),
            -1,
            dtype=torch.long,
            device=self._device,
        )
        self._track_active = torch.zeros((self._max_tracks,), dtype=torch.bool, device=self._device)
        self._track_states = torch.full(
            (self._max_tracks,),
            int(self._STATE_LOST),
            dtype=torch.long,
            device=self._device,
        )
        self._track_labels = torch.zeros(
            (self._max_tracks,),
            dtype=torch.long,
            device=self._device,
        )
        self._track_scores = torch.zeros(
            (self._max_tracks,),
            dtype=torch.float32,
            device=self._device,
        )
        self._track_last_frame = torch.zeros(
            (self._max_tracks,),
            dtype=torch.long,
            device=self._device,
        )
        self._track_hits = torch.zeros(
            (self._max_tracks,),
            dtype=torch.long,
            device=self._device,
        )
        self._track_mean = torch.zeros(
            (self._max_tracks, 8), dtype=torch.float32, device=self._device
        )
        self._track_covariance = torch.zeros(
            (self._max_tracks, 8, 8), dtype=torch.float32, device=self._device
        )
        self._next_track_id = torch.zeros((1,), dtype=torch.long, device=self._device)
        self._track_index = torch.arange(self._max_tracks, device=self._device, dtype=torch.long)
        self._det_index = torch.arange(self._max_detections, device=self._device, dtype=torch.long)
        self._big_cost = torch.full((), 1e9, device=self._device, dtype=torch.float32)
        self._state_tracking_t = torch.full(
            (), int(self._STATE_TRACKING), device=self._device, dtype=self._track_states.dtype
        )
        self._state_tentative_t = torch.full(
            (), int(self._STATE_TENTATIVE), device=self._device, dtype=self._track_states.dtype
        )
        self._state_lost_t = torch.full(
            (), int(self._STATE_LOST), device=self._device, dtype=self._track_states.dtype
        )
        self._neg_one_long = torch.full((), -1, device=self._device, dtype=torch.long)
        self._zero_long = torch.zeros((), device=self._device, dtype=torch.long)
        self._zero_float = torch.zeros((), device=self._device, dtype=torch.float32)

    def _kalman_initiate(
        self, measurements_cxcyah: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        count = int(measurements_cxcyah.shape[0])
        mean = measurements_cxcyah.new_zeros((count, 8))
        mean[:, 0:4] = measurements_cxcyah
        h = measurements_cxcyah[:, 3]
        std_pos = torch.stack(
            (
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                torch.full_like(h, 1e-2),
                2 * self._std_weight_position * h,
            ),
            dim=1,
        )
        std_vel = torch.stack(
            (
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                torch.full_like(h, 1e-5),
                10 * self._std_weight_velocity * h,
            ),
            dim=1,
        )
        std = torch.cat((std_pos, std_vel), dim=1)
        cov = torch.diag_embed(std.pow(2))
        return mean, cov

    def _kalman_predict(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean.numel() == 0:
            return mean, covariance
        h = mean[:, 3]
        std_pos = torch.stack(
            (
                self._std_weight_position * h,
                self._std_weight_position * h,
                torch.full_like(h, 1e-2),
                self._std_weight_position * h,
            ),
            dim=1,
        )
        std_vel = torch.stack(
            (
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                torch.full_like(h, 1e-5),
                self._std_weight_velocity * h,
            ),
            dim=1,
        )
        std = torch.cat((std_pos, std_vel), dim=1)
        motion_cov = torch.diag_embed(std.pow(2))

        mean = mean @ self._motion_mat_T
        motion = self._motion_mat.unsqueeze(0).expand_as(covariance)
        motion_T = self._motion_mat_T.unsqueeze(0).expand_as(covariance)
        covariance = motion @ (covariance @ motion_T) + motion_cov
        return mean, covariance

    def _kalman_project(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        measurement_mean = mean[:, 0:4]
        h = mean[:, 3]
        std = torch.stack(
            (
                self._std_weight_position * h,
                self._std_weight_position * h,
                torch.full_like(h, 1e-1),
                self._std_weight_position * h,
            ),
            dim=1,
        )
        projected_cov = covariance[:, 0:4, 0:4] + torch.diag_embed(std.pow(2))
        return measurement_mean, projected_cov

    def _kalman_update(
        self, mean: torch.Tensor, covariance: torch.Tensor, measurement_cxcyah: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_mean, projected_cov = self._kalman_project(mean, covariance)

        B = covariance @ self._update_mat_T
        BT = B.transpose(1, 2)
        chol, _ = torch.linalg.cholesky_ex(projected_cov, check_errors=False)
        sol = torch.cholesky_solve(BT, chol)
        kalman_gain = sol.transpose(1, 2)

        innovation = measurement_cxcyah - projected_mean
        delta = (innovation.unsqueeze(1) @ kalman_gain.transpose(1, 2)).squeeze(1)
        new_mean = mean + delta

        temp = projected_cov @ kalman_gain.transpose(1, 2)
        new_covariance = covariance - (kalman_gain @ temp)
        return new_mean, new_covariance

    def _predict_tracks(self, frame_id: torch.Tensor) -> None:
        confirmed = self._track_active & self._track_states.ne(self._STATE_TENTATIVE)
        not_prev_frame = self._track_last_frame.ne(frame_id - 1)
        degrade = confirmed & not_prev_frame
        self._track_mean[:, 7] = torch.where(
            degrade, self._track_mean[:, 7] / 2.0, self._track_mean[:, 7]
        )
        self._track_mean, self._track_covariance = self._kalman_predict(
            self._track_mean, self._track_covariance
        )

    def _remove_stale_tracks(self, frame_id: torch.Tensor) -> None:
        active_mask = self._track_active
        lost_mask = active_mask & self._track_states.eq(self._STATE_LOST)
        frame_delta = frame_id - self._track_last_frame
        too_long = lost_mask & frame_delta.ge(int(self._config.num_frames_to_keep_lost_tracks))
        stale_tent = (
            active_mask
            & self._track_states.eq(self._STATE_TENTATIVE)
            & self._track_last_frame.ne(frame_id)
        )
        remove_mask = too_long | stale_tent
        keep_mask = ~remove_mask

        self._track_active = active_mask & keep_mask
        self._track_ids = torch.where(
            keep_mask, self._track_ids, self._track_ids.new_full(self._track_ids.shape, -1)
        )
        self._track_states = torch.where(
            keep_mask,
            self._track_states,
            self._track_states.new_full(self._track_states.shape, int(self._STATE_LOST)),
        )
        self._track_labels = torch.where(
            keep_mask, self._track_labels, self._track_labels.new_zeros(self._track_labels.shape)
        )
        self._track_scores = torch.where(
            keep_mask, self._track_scores, self._track_scores.new_zeros(self._track_scores.shape)
        )
        self._track_last_frame = torch.where(
            keep_mask,
            self._track_last_frame,
            self._track_last_frame.new_zeros(self._track_last_frame.shape),
        )
        self._track_hits = torch.where(
            keep_mask, self._track_hits, self._track_hits.new_zeros(self._track_hits.shape)
        )
        self._track_mean = torch.where(
            keep_mask.unsqueeze(1),
            self._track_mean,
            self._track_mean.new_zeros(self._track_mean.shape),
        )
        self._track_covariance = torch.where(
            keep_mask.unsqueeze(1).unsqueeze(2),
            self._track_covariance,
            self._track_covariance.new_zeros(self._track_covariance.shape),
        )

    def _mark_unmatched_tracking(self, matched_mask: torch.Tensor) -> None:
        tracking_mask = self._track_active & self._track_states.eq(self._STATE_TRACKING)
        to_lose = tracking_mask & (~matched_mask)
        self._track_states = torch.where(to_lose, self._state_lost_t, self._track_states)

    def _assign_tracks_masked(
        self,
        track_mask: torch.Tensor,
        det_mask: torch.Tensor,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        *,
        weight_with_scores: bool,
        iou_thr: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        track_boxes = _bbox_cxcyah_to_xyxy(self._track_mean[:, 0:4])
        ious = _bbox_iou(track_boxes, det_bboxes)
        if weight_with_scores and det_scores.numel() > 0:
            ious = ious * det_scores.unsqueeze(0)

        track_labels = self._track_labels
        label_ok = (track_labels.unsqueeze(1) == det_labels.unsqueeze(0)).to(dtype=ious.dtype)
        cate_cost = (1.0 - label_ok) * 1e6
        cost = (1.0 - ious) + cate_cost
        cost_limit = 1.0 - float(iou_thr)

        row_mask = track_mask.to(dtype=torch.bool)
        col_mask = det_mask.to(dtype=torch.bool)
        big = self._big_cost
        cost = torch.where(row_mask.unsqueeze(1) & col_mask.unsqueeze(0), cost, big)
        return _hungarian_assign(
            cost,
            cost_limit=cost_limit,
            row_mask=row_mask,
            col_mask=col_mask,
            big=big,
        )

    def _update_tracks(
        self,
        track_to_det: torch.Tensor,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        frame_id: torch.Tensor,
    ) -> None:
        matched_mask = track_to_det.ge(0)
        safe_det_idx = torch.where(matched_mask, track_to_det, torch.zeros_like(track_to_det))

        det_bboxes_sel = det_bboxes.index_select(0, safe_det_idx)
        det_labels_sel = det_labels.index_select(0, safe_det_idx)
        det_scores_sel = det_scores.index_select(0, safe_det_idx)

        measurements = _bbox_xyxy_to_cxcyah(det_bboxes_sel)
        new_mean, new_cov = self._kalman_update(
            self._track_mean, self._track_covariance, measurements
        )

        mask_mean = matched_mask.unsqueeze(1)
        mask_cov = matched_mask.unsqueeze(1).unsqueeze(2)
        self._track_mean = torch.where(mask_mean, new_mean, self._track_mean)
        self._track_covariance = torch.where(mask_cov, new_cov, self._track_covariance)
        self._track_scores = torch.where(matched_mask, det_scores_sel, self._track_scores)
        self._track_labels = torch.where(matched_mask, det_labels_sel, self._track_labels)
        self._track_last_frame = torch.where(matched_mask, frame_id, self._track_last_frame)

        new_hits = torch.where(matched_mask, self._track_hits + 1, self._track_hits)
        self._track_hits = new_hits

        tent_mask = matched_mask & self._track_states.eq(self._STATE_TENTATIVE)
        tent_mask &= new_hits.ge(int(self._config.num_tentatives))
        lost_mask = matched_mask & self._track_states.eq(self._STATE_LOST)

        to_track = tent_mask | lost_mask
        self._track_states = torch.where(to_track, self._state_tracking_t, self._track_states)

    def _init_new_tracks(
        self,
        new_det_mask: torch.Tensor,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        ids: torch.Tensor,
        frame_id: torch.Tensor,
    ) -> torch.Tensor:
        active_count = self._track_active.sum()
        state_val = torch.where(active_count.eq(0), self._state_tracking_t, self._state_tentative_t)

        num_new = new_det_mask.sum()
        free_mask = ~self._track_active
        free_count = free_mask.sum()
        assign_count = torch.minimum(num_new, free_count)

        assign_mask = self._track_index < assign_count
        new_order = _stable_mask_order(new_det_mask)[: self._max_tracks]
        free_order = _stable_mask_order(free_mask)

        new_prefix_mask = self._track_index < num_new
        free_prefix_mask = self._track_index < free_count
        pair_mask = assign_mask & new_prefix_mask & free_prefix_mask

        new_bboxes = det_bboxes.index_select(0, new_order)
        new_labels = det_labels.index_select(0, new_order)
        new_scores = det_scores.index_select(0, new_order)

        new_ids_base = self._next_track_id + torch.arange(
            self._max_tracks, device=self._device, dtype=torch.long
        )

        det_existing = ids.index_select(0, new_order)
        det_updated = torch.where(pair_mask, new_ids_base, det_existing)
        ids.index_put_((new_order,), det_updated)

        track_idx = free_order
        track_ids_existing = self._track_ids.index_select(0, track_idx)
        track_ids_updated = torch.where(pair_mask, new_ids_base, track_ids_existing)
        self._track_ids.index_put_((track_idx,), track_ids_updated)

        active_existing = self._track_active.index_select(0, track_idx)
        active_updated = torch.where(pair_mask, torch.ones_like(active_existing), active_existing)
        self._track_active.index_put_((track_idx,), active_updated)

        states_existing = self._track_states.index_select(0, track_idx)
        states_updated = torch.where(
            pair_mask, state_val.expand_as(states_existing), states_existing
        )
        self._track_states.index_put_((track_idx,), states_updated)

        labels_existing = self._track_labels.index_select(0, track_idx)
        labels_updated = torch.where(pair_mask, new_labels, labels_existing)
        self._track_labels.index_put_((track_idx,), labels_updated)

        scores_existing = self._track_scores.index_select(0, track_idx)
        scores_updated = torch.where(pair_mask, new_scores, scores_existing)
        self._track_scores.index_put_((track_idx,), scores_updated)

        last_existing = self._track_last_frame.index_select(0, track_idx)
        last_updated = torch.where(pair_mask, frame_id, last_existing)
        self._track_last_frame.index_put_((track_idx,), last_updated)

        hits_existing = self._track_hits.index_select(0, track_idx)
        hits_updated = torch.where(pair_mask, torch.ones_like(hits_existing), hits_existing)
        self._track_hits.index_put_((track_idx,), hits_updated)

        measurements = _bbox_xyxy_to_cxcyah(new_bboxes)
        mean_new, cov_new = self._kalman_initiate(measurements)

        mean_existing = self._track_mean.index_select(0, track_idx)
        mean_updated = torch.where(pair_mask.unsqueeze(1), mean_new, mean_existing)
        self._track_mean.index_put_((track_idx,), mean_updated)

        cov_existing = self._track_covariance.index_select(0, track_idx)
        cov_updated = torch.where(pair_mask.unsqueeze(1).unsqueeze(2), cov_new, cov_existing)
        self._track_covariance.index_put_((track_idx,), cov_updated)

        self._next_track_id = self._next_track_id + assign_count
        return ids

    def track(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        frame_id_tensor = data[kFrameIdKey]
        if frame_id_tensor.numel() < 1:
            raise RuntimeError("frame_id tensor must contain a value")
        frame_id = frame_id_tensor.to(device=self._device, dtype=torch.long).reshape(-1)[:1]
        if frame_id.numel() < 1:
            raise RuntimeError("frame_id tensor must contain a value")
        frame_id = frame_id[0]

        if kNumDetectionsKey not in data:
            raise RuntimeError("data must contain 'num_detections' entry")
        num_detections = (
            data[kNumDetectionsKey].to(device=self._device, dtype=torch.long).reshape(-1)[:1]
        )
        if num_detections.numel() < 1:
            raise RuntimeError("num_detections tensor must contain a value")
        num_detections = torch.clamp(num_detections, min=0, max=int(self._max_detections))

        det_bboxes = data[kBBoxesKey]
        det_labels = data[kLabelsKey]
        det_scores = data[kScoresKey]

        if det_bboxes.ndim != 2 or det_bboxes.shape[1] != 4:
            raise RuntimeError("bboxes tensor must have shape [max_detections, 4]")
        if det_labels.ndim != 1:
            raise RuntimeError("labels tensor must be 1-D")
        if det_scores.ndim != 1:
            raise RuntimeError("scores tensor must be 1-D")
        if det_bboxes.shape[0] != self._max_detections:
            raise RuntimeError("bboxes tensor first dimension must equal max_detections")
        if det_labels.shape[0] != self._max_detections:
            raise RuntimeError("labels tensor first dimension must equal max_detections")
        if det_scores.shape[0] != self._max_detections:
            raise RuntimeError("scores tensor first dimension must equal max_detections")

        det_bboxes = det_bboxes.to(device=self._device, dtype=torch.float32)
        det_labels = det_labels.to(device=self._device, dtype=torch.long)
        det_scores = det_scores.to(device=self._device, dtype=torch.float32)

        det_valid_mask = self._det_index < num_detections[0]
        valid_f = det_valid_mask.to(dtype=det_bboxes.dtype)
        det_bboxes = det_bboxes * valid_f.unsqueeze(1)
        det_labels = det_labels * det_valid_mask.to(dtype=det_labels.dtype)
        det_scores = det_scores * valid_f

        ids = det_labels.new_full((self._max_detections,), -1, dtype=torch.long)

        first_mask = det_valid_mask & det_scores.gt(float(self._config.obj_score_thrs_high))
        second_mask = (
            det_valid_mask & (~first_mask) & det_scores.gt(float(self._config.obj_score_thrs_low))
        )

        self._predict_tracks(frame_id)

        confirmed_mask = self._track_active & self._track_states.ne(self._STATE_TENTATIVE)
        unconfirmed_mask = self._track_active & self._track_states.eq(self._STATE_TENTATIVE)

        first_track_to_det, first_det_to_track = self._assign_tracks_masked(
            confirmed_mask,
            first_mask,
            det_bboxes,
            det_labels,
            det_scores,
            weight_with_scores=bool(self._config.weight_iou_with_det_scores),
            iou_thr=float(self._config.match_iou_thrs_high),
        )

        first_match_mask = first_mask & first_det_to_track.ge(0)
        safe_track_idx = torch.where(
            first_match_mask, first_det_to_track, torch.zeros_like(first_det_to_track)
        )
        matched_track_ids = self._track_ids.index_select(0, safe_track_idx)
        ids = torch.where(first_match_mask, matched_track_ids, ids)

        unmatched_first_mask = first_mask & first_det_to_track.lt(0)
        tent_track_to_det, tent_det_to_track = self._assign_tracks_masked(
            unconfirmed_mask,
            unmatched_first_mask,
            det_bboxes,
            det_labels,
            det_scores,
            weight_with_scores=bool(self._config.weight_iou_with_det_scores),
            iou_thr=float(self._config.match_iou_thrs_tentative),
        )
        tent_match_mask = unmatched_first_mask & tent_det_to_track.ge(0)
        safe_track_idx = torch.where(
            tent_match_mask, tent_det_to_track, torch.zeros_like(tent_det_to_track)
        )
        matched_track_ids = self._track_ids.index_select(0, safe_track_idx)
        ids = torch.where(tent_match_mask, matched_track_ids, ids)

        track_unmatched_mask = confirmed_mask & first_track_to_det.lt(0)
        recent_mask = self._track_last_frame.eq(frame_id - 1)
        selectable_mask = track_unmatched_mask & recent_mask
        second_track_to_det, second_det_to_track = self._assign_tracks_masked(
            selectable_mask,
            second_mask,
            det_bboxes,
            det_labels,
            det_scores,
            weight_with_scores=False,
            iou_thr=float(self._config.match_iou_thrs_low),
        )
        second_match_mask = second_mask & second_det_to_track.ge(0)
        safe_track_idx = torch.where(
            second_match_mask, second_det_to_track, torch.zeros_like(second_det_to_track)
        )
        matched_track_ids = self._track_ids.index_select(0, safe_track_idx)
        ids = torch.where(second_match_mask, matched_track_ids, ids)

        matched_tracks_mask = first_track_to_det.ge(0) | second_track_to_det.ge(0)
        self._mark_unmatched_tracking(matched_tracks_mask)

        self._update_tracks(first_track_to_det, det_bboxes, det_labels, det_scores, frame_id)
        self._update_tracks(tent_track_to_det, det_bboxes, det_labels, det_scores, frame_id)
        self._update_tracks(second_track_to_det, det_bboxes, det_labels, det_scores, frame_id)

        new_track_mask = first_mask & ids.lt(0)
        ids = self._init_new_tracks(
            new_track_mask, det_bboxes, det_labels, det_scores, ids, frame_id
        )

        self._remove_stale_tracks(frame_id)

        output_mask = ids.ge(0)
        order = _stable_mask_order(output_mask)
        ordered_ids = ids.index_select(0, order)
        ordered_bboxes = det_bboxes.index_select(0, order)
        ordered_labels = det_labels.index_select(0, order)
        ordered_scores = det_scores.index_select(0, order)

        num_tracks = torch.clamp(output_mask.sum(), max=int(self._max_tracks))
        prefix_ids = ordered_ids[: self._max_tracks]
        prefix_bboxes = ordered_bboxes[: self._max_tracks]
        prefix_labels = ordered_labels[: self._max_tracks]
        prefix_scores = ordered_scores[: self._max_tracks]

        assign_mask = self._track_index < num_tracks
        out_ids = torch.where(assign_mask, prefix_ids, prefix_ids.new_full((self._max_tracks,), -1))
        out_labels = torch.where(
            assign_mask, prefix_labels, prefix_labels.new_zeros((self._max_tracks,))
        )
        out_scores = torch.where(
            assign_mask, prefix_scores, prefix_scores.new_zeros((self._max_tracks,))
        )
        out_bboxes = torch.where(
            assign_mask.unsqueeze(1),
            prefix_bboxes,
            prefix_bboxes.new_zeros((self._max_tracks, 4)),
        )

        data[kIdsKey] = out_ids
        data[kBBoxesKey] = out_bboxes
        data[kLabelsKey] = out_labels
        data[kScoresKey] = out_scores
        data[kNumTracksKey] = num_tracks.reshape(1)
        data[kNumDetectionsKey] = num_detections.to(device=self._device, dtype=torch.long)
        data["max_id"] = torch.clamp(self._next_track_id - 1, min=0).reshape(1)

        return data


__all__ = ["HmByteTrackerCuda", "HmByteTrackerCudaStatic"]
