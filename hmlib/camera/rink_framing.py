"""Rink-aware camera guidance over the existing camera trajectory.

The policy follows HStream's RinkCameraGuide: robust per-player rink velocity,
bounded anticipation, optional goal framing and asymmetric zoom smoothing.
It changes only the output crop, never observations or the tracker's state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hmlib.ui.calibration_workspace import project_rink_points, unproject_rink_points


@dataclass
class _History:
    reference: float
    seen: float
    point: np.ndarray
    pixel: np.ndarray
    velocity: np.ndarray | None = None
    pixel_velocity: np.ndarray | None = None


def _blend(seconds: float, constant: float) -> float:
    return -math.expm1(-seconds / constant)


class RinkCameraGuide:
    def __init__(self, mapping: dict, settings: dict, image_size: tuple[int, int]) -> None:
        self.mapping = mapping
        self.settings = settings
        self.image_size = image_size
        self.previous_time = None
        self.previous_box = None
        self.history: dict[int, _History] = {}
        self.offset = np.zeros(2)
        self.scale = 1.0
        self.info = {}
        length, width = mapping["rink_size"]
        goal_inset = mapping.get("layout", {}).get("goal_inset", 3.3528)
        self.goals = unproject_rink_points(
            [[goal_inset, width / 2], [length - goal_inset, width / 2]],
            mapping,
            image_size=image_size,
        )

    def step(self, seconds: float, baseline, arena, boxes, ids, cluster=None) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        arena = np.asarray(arena, dtype=float)
        if not np.isfinite(baseline).all() or np.any(baseline[2:] <= baseline[:2]):
            raise ValueError("Rink guidance requires a finite, nonempty camera crop")
        if not np.isfinite(arena).all() or np.any(arena[2:] <= arena[:2]):
            raise ValueError("Rink guidance requires a finite, nonempty arena")
        if self.previous_time == seconds:
            return self.previous_box.copy()
        reset = self.previous_time is None or not 0 < seconds - self.previous_time <= 0.3
        dt = 0.0 if reset else seconds - self.previous_time
        if reset:
            self.history.clear()
            self.offset[:] = 0
            self.scale = 1.0
        self.previous_time = seconds
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)[:256]
        pixels = (boxes[:, :2] + boxes[:, 2:]) / 2
        pixels[:, 1] = boxes[:, 3]
        points = np.asarray(
            project_rink_points(
                pixels.tolist(), self.mapping, image_size=self.image_size, allow_missing=True
            )
        )
        located, heights, velocities, pixel_velocities = [], [], [], []
        for index, (identity, pixel, point) in enumerate(zip(ids, pixels, points)):
            identity = int(identity)
            if identity < 0 or not np.isfinite(point).all():
                continue
            if cluster is not None and (np.any(pixel < cluster[:2]) or np.any(pixel > cluster[2:])):
                continue
            located.append(point)
            heights.append(boxes[index, 3] - boxes[index, 1])
            history = self.history.get(identity)
            if history is None or seconds - history.seen > 0.3:
                self.history[identity] = _History(seconds, seconds, point, pixel)
                continue
            history.seen = seconds
            elapsed = seconds - history.reference
            if elapsed >= 0.12:
                velocity = (point - history.point) / elapsed
                pv = (pixel - history.pixel) / elapsed
                if elapsed <= 0.4 and np.linalg.norm(velocity) <= 14:
                    alpha = _blend(elapsed, 0.35) if history.velocity is not None else 1
                    history.velocity = (
                        velocity
                        if history.velocity is None
                        else history.velocity + (velocity - history.velocity) * alpha
                    )
                    history.pixel_velocity = (
                        pv
                        if history.pixel_velocity is None
                        else history.pixel_velocity + (pv - history.pixel_velocity) * alpha
                    )
                else:
                    history.velocity = history.pixel_velocity = None
                history.point, history.pixel, history.reference = point, pixel, seconds
            if history.velocity is not None:
                velocities.append(history.velocity)
                pixel_velocities.append(history.pixel_velocity)
        self.history = {
            key: value for key, value in self.history.items() if seconds - value.seen <= 0.3
        }
        desired_offset, desired_scale = np.zeros(2), 1.0
        bw, bh = baseline[2:] - baseline[:2]
        aspect = bw / bh
        active = not reset and len(located) >= 3 and len(velocities) >= 3 and cluster is not None
        speed = 0.0
        if active:
            speed = float(np.linalg.norm(np.median(velocities, axis=0)))
            rush = np.clip((speed - 1.5) / 4, 0, 1)
            if self.settings.get("anticipate", True):
                desired_offset = np.clip(
                    np.median(pixel_velocities, axis=0) * min(0.6, 4 / max(speed, 0.1)) * rush,
                    [-bw * 0.16, -bh * 0.06],
                    [bw * 0.16, bh * 0.06],
                )
            if self.settings.get("adaptive_zoom", False):
                cluster_size = np.asarray(cluster[2:]) - np.asarray(cluster[:2])
                desired_scale = max(
                    0.94 + 0.24 * rush, min(1.18, max(cluster_size / [bw, bh]) * 1.08)
                )
            if self.settings.get("frame_goals", False):
                length = self.mapping["rink_size"][0]
                x = float(np.median(np.asarray(located)[:, 0])) - length / 2
                # IIHF/NHL blue lines are approximately 7.6m from centre ice.
                weight = np.clip((abs(x) - 9.6) / 8, 0, 1)
                goal = np.asarray(self.goals[int(x > 0)])
                if np.isfinite(goal).all():
                    height = float(np.median(heights))
                    low, high = goal - height * [0.3, 1.2], goal + height * [0.3, 0.2]
                    centre = (baseline[:2] + baseline[2:]) / 2 + desired_offset
                    required = 2 * max(
                        centre[0] - low[0],
                        high[0] - centre[0],
                        (centre[1] - low[1]) * aspect,
                        (high[1] - centre[1]) * aspect,
                    )
                    desired_scale += (
                        np.clip(required / bw, desired_scale, 1.35) - desired_scale
                    ) * weight
                    half = np.array([bw, bh]) * desired_scale / 2
                    for axis in range(2):
                        if high[axis] - half[axis] <= low[axis] + half[axis]:
                            target = np.clip(
                                centre[axis], high[axis] - half[axis], low[axis] + half[axis]
                            )
                            desired_offset[axis] += (target - centre[axis]) * weight
        desired_offset = np.clip(desired_offset, [-bw * 0.2, -bh * 0.1], [bw * 0.2, bh * 0.1])
        self.offset += (desired_offset - self.offset) * _blend(dt, 0.45)
        self.scale += (desired_scale - self.scale) * _blend(
            dt, 0.35 if desired_scale > self.scale else 1.5
        )
        width = min(bw * self.scale, arena[2] - arena[0], (arena[3] - arena[1]) * aspect)
        size = np.array([width, width / aspect])
        centre = (baseline[:2] + baseline[2:]) / 2 + self.offset
        low = np.clip(centre - size / 2, arena[:2], arena[2:] - size)
        self.previous_box = (
            baseline.copy()
            if reset or (self.scale == 1 and not self.offset.any())
            else np.r_[low, low + size]
        )
        self.info = {
            "active": active,
            "speed_mps": speed,
            "located": len(located),
            "zoom_ratio": self.scale,
        }
        return self.previous_box.copy()


def draw_rink_map(image, overlay: dict):
    """Draw a bounded desktop map using already-projected metadata."""
    import cv2
    import torch

    from hmlib.utils.image import make_channels_last

    frame = make_channels_last(image)
    batched = frame.ndim == 4
    frames = frame if batched else frame.unsqueeze(0)
    height, width = frames.shape[-3:-1]
    sizes = {"icon": 120, "compact": 220, "standard": 320, "large": 480}
    map_size = overlay.get("map_size", "standard")
    position = overlay.get("position", "upper-right")
    if map_size not in sizes or position not in {"upper-left", "upper-right", "lower-left"}:
        raise ValueError("Unknown rink mini-map size or position")
    map_width = min(sizes[map_size], max(40, width // 3))
    map_height = max(20, round(map_width * overlay["size"][1] / overlay["size"][0]))
    margin = min(12, height // 10, width // 10)
    if map_height + 2 * margin > height or map_width + 2 * margin > width:
        return image
    for index, points in enumerate(overlay["points"]):
        if index >= len(frames):
            break
        tile = np.full((map_height, map_width, 3), 235, dtype=np.uint8)
        cv2.rectangle(tile, (0, 0), (map_width - 1, map_height - 1), (20, 20, 20), 2)
        cv2.line(tile, (map_width // 2, 0), (map_width // 2, map_height - 1), (30, 30, 210), 1)
        for fraction in (0.375, 0.625):
            cv2.line(
                tile,
                (round(map_width * fraction), 0),
                (round(map_width * fraction), map_height - 1),
                (210, 60, 30),
                1,
            )
        for x, y in points:
            if not np.isfinite([x, y]).all():
                continue
            px, py = round(x / overlay["size"][0] * map_width), round(
                y / overlay["size"][1] * map_height
            )
            if 0 <= px < map_width and 0 <= py < map_height:
                cv2.circle(tile, (px, py), 3, (20, 100, 20), -1)
        tensor = torch.as_tensor(tile, device=frames.device, dtype=frames.dtype)
        if frames.is_floating_point() and float(frames[index].max()) <= 1:
            tensor = tensor / 255
        x = width - margin - map_width if position == "upper-right" else margin
        y = height - margin - map_height if position == "lower-left" else margin
        frames[index, y : y + map_height, x : x + map_width, :3] = tensor
    return frames if batched else frames[0]
