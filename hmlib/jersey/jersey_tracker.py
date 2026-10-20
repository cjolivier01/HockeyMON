from __future__ import absolute_import, division, print_function

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from hmlib.log import logger
from hmlib.utils.image import is_channels_last
from hmlib.vis.labels import draw_label_box

from .number_classifier import TrackJerseyInfo


@dataclass
class TrackingIdNumberInfo:
    tracking_id: int = -1
    current_number: int = -1
    current_score: float = 0.0
    last_frame_id: int = -1
    # number -> decayed evidence score
    number_scores: Dict[int, float] = field(default_factory=dict)
    # smoothed anchor point for drawing (x, y) in image coordinates
    anchor_ema: Optional[Tuple[float, float]] = None


class JerseyTracker:
    def __init__(
        self,
        show: bool = False,
        evidence_decay: float = 0.92,
        min_display_evidence: float = 1.0,
        switch_ratio: float = 1.35,
        anchor_ema_alpha: float = 0.55,
    ) -> None:
        self._show = show
        self._evidence_decay = float(evidence_decay)
        self._min_display_evidence = float(min_display_evidence)
        self._switch_ratio = float(switch_ratio)
        self._anchor_ema_alpha = float(anchor_ema_alpha)
        self._tracking_id_jersey: Dict[int, TrackingIdNumberInfo] = {}
        self._jersey_number_to_tracking_id: Dict[int, int] = {}

    def _update_anchor(self, tracking_id: int, anchor_xy: Tuple[float, float]) -> None:
        info = self._tracking_id_jersey.get(tracking_id)
        if info is None:
            return
        if info.anchor_ema is None:
            info.anchor_ema = (float(anchor_xy[0]), float(anchor_xy[1]))
            return
        ax, ay = info.anchor_ema
        alpha = self._anchor_ema_alpha
        info.anchor_ema = (
            float(alpha * anchor_xy[0] + (1.0 - alpha) * ax),
            float(alpha * anchor_xy[1] + (1.0 - alpha) * ay),
        )

    def _decay_scores(self, info: TrackingIdNumberInfo, frame_id: int) -> None:
        if info.last_frame_id < 0:
            return
        dt = max(1, int(frame_id) - int(info.last_frame_id))
        decay = float(self._evidence_decay**dt)
        if decay >= 0.999:
            return
        for k in list(info.number_scores.keys()):
            v = float(info.number_scores.get(k, 0.0)) * decay
            if v < 1e-4:
                info.number_scores.pop(k, None)
            else:
                info.number_scores[k] = v

    def _select_number(self, info: TrackingIdNumberInfo) -> None:
        if not info.number_scores:
            info.current_number = -1
            info.current_score = 0.0
            return
        sorted_items = sorted(info.number_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_num, best_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        current_score = float(info.number_scores.get(info.current_number, 0.0))

        # Initial acquisition.
        if info.current_number < 0:
            if best_score >= self._min_display_evidence:
                info.current_number = int(best_num)
                info.current_score = float(best_score)
            return

        # Keep current unless a contender wins decisively.
        if best_num == info.current_number:
            info.current_score = float(best_score)
            return

        should_switch = (
            best_score >= self._min_display_evidence
            and best_score >= (current_score * self._switch_ratio)
            and best_score >= (second_score * 1.05)
        )
        if should_switch:
            logger.info(
                "Jersey switch: ID %s #%s -> #%s (%.3f -> %.3f)",
                info.tracking_id,
                info.current_number,
                int(best_num),
                current_score,
                best_score,
            )
            info.current_number = int(best_num)
            info.current_score = float(best_score)

    def draw(
        self, image: torch.Tensor, tracking_ids: torch.Tensor, bboxes: torch.Tensor
    ) -> torch.Tensor:
        if not self._show:
            return image
        assert len(tracking_ids) == len(bboxes)
        # `image` is usually CHW torch, but handle HWC for robustness.
        if isinstance(image, torch.Tensor):
            img_w = int(image.shape[-1] if not is_channels_last(image) else image.shape[1])
        else:
            img_w = int(image.shape[1])
        font_scale = max(0.6, float(img_w) / 1800.0)

        for (x1, y1, w, _), tracking_id in zip(bboxes, tracking_ids):
            tracking_id = int(tracking_id)
            info: TrackingIdNumberInfo = self._tracking_id_jersey.get(tracking_id)
            if info is None or info.current_number <= 0:
                continue
            anchor = (float(x1 + (w * 0.5)), float(y1))
            self._update_anchor(tracking_id=tracking_id, anchor_xy=anchor)
            ax, ay = info.anchor_ema if info.anchor_ema is not None else anchor

            # Fade background with evidence (keeps display stable but less noisy).
            ev = float(info.current_score)
            bg_alpha = int(max(80, min(200, 80 + (ev * 60.0))))
            image = draw_label_box(
                image=image,
                text=str(info.current_number),
                center_x=int(ax),
                bottom_y=int(ay),
                font_scale=font_scale,
                fg=(255, 255, 255),
                bg=(0, 0, 0),
                bg_alpha=bg_alpha,
            )
        return image

    def observe_tracking_id_number_info(self, frame_id: int, info: TrackJerseyInfo) -> None:
        """Consume per-frame jersey predictions and keep a stable per-track display state."""
        tracking_id = int(info.tracking_id)
        number = int(info.number)
        score = float(info.score)
        if tracking_id < 0 or number <= 0:
            return

        prev_info = self._tracking_id_jersey.get(tracking_id)
        if prev_info is None:
            prev_info = TrackingIdNumberInfo(tracking_id=tracking_id)
            self._tracking_id_jersey[tracking_id] = prev_info

        self._decay_scores(prev_info, frame_id=frame_id)
        prev_info.last_frame_id = int(frame_id)

        prev_info.number_scores[number] = float(
            prev_info.number_scores.get(number, 0.0) + max(score, 0.0)
        )
        self._select_number(prev_info)

        # Best-effort mapping (not used for control, but handy for debugging).
        if prev_info.current_number > 0:
            self._jersey_number_to_tracking_id[prev_info.current_number] = tracking_id
