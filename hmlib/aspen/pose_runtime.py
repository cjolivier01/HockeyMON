"""Source-time pose scheduling and compact display holds.

Fresh predictions stay separate from display estimates: jersey crops and action
history must never consume a skeleton merely reprojected for drawing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

MAX_POSE_TRACKS = 256
MAX_POSE_DUE_TRACKS = 32
POSE_HOLD_SECONDS = 0.250


@lru_cache(maxsize=1)
def _baseline_pose_settings() -> dict:
    from hmlib.config import ROOT_DIR, baseline_config

    return baseline_config(ROOT_DIR)["pipeline"]["player-analytics"]["pose"]


def pose_settings(config: Mapping[str, Any]) -> tuple[float | None, bool]:
    defaults = _baseline_pose_settings()
    settings = config.get("pipeline", {}).get("player-analytics", {}).get("pose", {})
    rate = settings.get("rate-hz", defaults["rate-hz"])
    if isinstance(rate, str) and rate.lower().strip() == "fps":
        rate_hz = None
    else:
        if isinstance(rate, bool):
            raise ValueError("Pose rate-hz must be a positive number or 'fps'")
        rate_hz = float(rate)
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("Pose rate-hz must be a positive number or 'fps'")
    continuous = settings.get("continuous-drawing", defaults["continuous-drawing"])
    if not isinstance(continuous, bool):
        raise TypeError("Pose continuous-drawing must be a boolean")
    return rate_hz, continuous


@dataclass
class PoseObservation:
    timestamp: float
    box: torch.Tensor
    keypoints: torch.Tensor
    scores: torch.Tensor


class PoseRuntime:
    """Bound per-track work and retain only box-relative pose coordinates."""

    def __init__(self) -> None:
        self.next_due: dict[int, float] = {}
        self.observations: dict[int, PoseObservation] = {}
        self.timestamp: float | None = None
        self.source: Any = None
        self.rate_hz: float | None = None
        self.resets = 0

    def reset(self) -> None:
        self.next_due.clear()
        self.observations.clear()
        self.timestamp = None
        self.resets += 1

    def due(
        self,
        timestamp: float,
        track_ids: Sequence[int],
        *,
        source: Any,
        rate_hz: float | None,
        limit: int = MAX_POSE_DUE_TRACKS,
    ) -> list[int]:
        if not math.isfinite(timestamp) or timestamp < 0:
            self.reset()
            raise ValueError("Pose scheduling requires finite, nonnegative source timestamps")
        if (
            source != self.source
            or rate_hz != self.rate_hz
            or (self.timestamp is not None and timestamp <= self.timestamp)
        ):
            self.reset()
        self.source = source
        self.rate_hz = rate_hz
        self.timestamp = timestamp
        active = {int(track_id) for track_id in track_ids[:MAX_POSE_TRACKS]}
        self.next_due = {key: value for key, value in self.next_due.items() if key in active}
        self.observations = {
            key: value
            for key, value in self.observations.items()
            if key in active and timestamp - value.timestamp <= POSE_HOLD_SECONDS
        }
        due = sorted(
            (
                track_id
                for track_id in active
                if timestamp + 1e-9 >= self.next_due.get(track_id, -math.inf)
            ),
            key=lambda track_id: (self.next_due.get(track_id, -math.inf), track_id),
        )[: max(1, min(limit, MAX_POSE_DUE_TRACKS))]
        for track_id in due:
            self.next_due[track_id] = timestamp if rate_hz is None else timestamp + 1 / rate_hz
        return due

    def observe(
        self,
        track_id: int,
        timestamp: float,
        box: torch.Tensor,
        keypoints: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        if track_id not in self.next_due:
            return
        if keypoints.ndim != 2 or keypoints.shape[-1] != 2:
            raise ValueError("Pose keypoints must have shape [joints, 2]")
        if scores.shape != keypoints.shape[:1]:
            raise ValueError("Pose confidence must contain one score per joint")
        if not torch.isfinite(box).all() or not torch.isfinite(keypoints).all():
            raise ValueError("Pose drawing received non-finite coordinates")
        self.observations[track_id] = PoseObservation(
            timestamp=timestamp,
            box=box.detach().clone(),
            keypoints=keypoints.detach().clone(),
            scores=scores.detach().clone(),
        )

    def drawing(
        self,
        timestamp: float,
        boxes: Mapping[int, torch.Tensor],
        *,
        continuous: bool,
    ) -> list[tuple[int, PoseObservation]]:
        result = []
        for track_id, box in boxes.items():
            observation = self.observations.get(track_id)
            if observation is None:
                continue
            age = timestamp - observation.timestamp
            if age < 0 or age > POSE_HOLD_SECONDS or (age > 1e-9 and not continuous):
                continue
            old_box = observation.box
            new_box = box.to(device=old_box.device, dtype=old_box.dtype)
            old_size = old_box[2:4] - old_box[:2]
            new_size = new_box[2:4] - new_box[:2]
            if (old_size <= 0).any() or (new_size <= 0).any():
                continue
            points = (observation.keypoints - old_box[:2]) / old_size
            points = points * new_size + new_box[:2]
            result.append(
                (
                    track_id,
                    PoseObservation(
                        timestamp=observation.timestamp,
                        box=new_box,
                        keypoints=points,
                        scores=observation.scores,
                    ),
                )
            )
        return result
