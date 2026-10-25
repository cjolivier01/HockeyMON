"""Overlay action labels on tracked players in video frames.

This helper keeps the last seen action per tracking id and draws text
above bounding boxes using the tracking utilities visualization helpers.

@see @ref hmlib.tracking_utils.visualization "tracking_utils.visualization" for text rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import torch

from hmlib.tracking_utils import visualization as vis


@dataclass
class TrackingIdActionInfo:
    tracking_id: int
    label: str
    label_index: int
    score: float


class ActionTracker:
    """
    Maintains latest action label per tracking id and overlays text on frames.
    """

    def __init__(self, show: bool = False) -> None:
        self._show = show
        # tracking_id -> last seen action info
        self._actions: dict[int, TrackingIdActionInfo] = {}

    def replace(self, frame_id: int, infos: list[dict]) -> None:
        """Replace display state with current, already-settled backend results."""
        self._actions = {
            int(info["tracking_id"]): TrackingIdActionInfo(
                tracking_id=int(info["tracking_id"]),
                label=str(info["label"]),
                label_index=int(info["label_index"]),
                score=float(info["score"]),
            )
            for info in infos
        }

    def observe(self, frame_id: int, infos: list[dict] | None) -> None:
        if not infos:
            return
        for a in infos:
            tid = int(a.get("tracking_id", -1))
            if tid < 0:
                continue
            label = str(a.get("label", ""))
            label_index = int(a.get("label_index", -1))
            score = float(a.get("score", 0.0))
            self._actions[tid] = TrackingIdActionInfo(
                tracking_id=tid, label=label, label_index=label_index, score=score
            )

    def draw(
        self, image: torch.Tensor, tracking_ids: torch.Tensor, bboxes_tlwh: torch.Tensor
    ) -> torch.Tensor:
        if not self._show:
            return image
        assert len(tracking_ids) == len(bboxes_tlwh)
        text_scale = max(2, image.shape[1] / 2500.0)
        for (x1, y1, w, _), tracking_id in zip(bboxes_tlwh, tracking_ids):
            tid = int(tracking_id)
            info = self._actions.get(tid)
            if info is None or not info.label:
                continue
            xc = int(x1 + w // 2)
            yc = int(y1)  # top of box
            label_text = f"{info.label}"
            image = vis.my_put_text(
                image,
                label_text,
                (xc, yc - 20),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 200, 0),  # green
                thickness=3,
            )
        return image
