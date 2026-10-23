"""Source-time state and preprocessing shared by native player analytics."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from PIL import Image

SAMPLE_NS = 100_000_000
MAX_GAP_NS = 150_000_000


class ActionHistory:
    """100 causal COCO17 samples; interpolation only between fresh observations."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=100)
        self.previous = None
        self.previous_ns = None
        self.next_ns = None
        self.size = None

    def observe(self, timestamp_ns: int, pose: np.ndarray, width: int, height: int) -> bool:
        if pose.shape != (17, 3):
            raise ValueError("Native activity recognition requires COCO17 keypoints and confidence")
        if timestamp_ns < 0 or width <= 0 or height <= 0:
            self.reset()
            raise ValueError(
                "Activity history requires a nonnegative timestamp and positive canvas"
            )
        if self.previous_ns is not None and (
            timestamp_ns < self.previous_ns
            or timestamp_ns - self.previous_ns > MAX_GAP_NS
            or self.size != (width, height)
        ):
            self.reset()
        if timestamp_ns == self.previous_ns:
            return False
        confidence = pose[:, 2]
        if not np.isfinite(pose).all() or np.any((confidence < 0) | (confidence > 1)):
            self.reset()
            return False
        visible = confidence >= 0.3
        if visible.sum() < 8:
            self.reset()
            return False
        normalized = np.zeros((17, 3), dtype=np.float32)
        normalized[visible, :2] = 2 * pose[visible, :2] / [width, height] - 1
        normalized[visible, 2] = confidence[visible]
        if self.previous_ns is None:
            self.samples.append((timestamp_ns, normalized))
            self.next_ns = timestamp_ns + SAMPLE_NS
        else:
            while self.next_ns <= timestamp_ns:
                weight = (self.next_ns - self.previous_ns) / (timestamp_ns - self.previous_ns)
                confidences = (
                    normalized[:, 2]
                    if self.next_ns == timestamp_ns
                    else np.minimum(self.previous[:, 2], normalized[:, 2])
                )
                valid = confidences > 0
                if valid.sum() < 8:
                    self.reset()
                    return False
                sample = np.zeros((17, 3), dtype=np.float32)
                sample[valid, :2] = self.previous[valid, :2] + weight * (
                    normalized[valid, :2] - self.previous[valid, :2]
                )
                sample[valid, 2] = confidences[valid]
                self.samples.append((self.next_ns, sample))
                self.next_ns += SAMPLE_NS
        self.previous_ns = timestamp_ns
        self.previous = normalized
        self.size = (width, height)
        return True

    def tensor(self, now_ns: int) -> np.ndarray | None:
        if len(self.samples) != 100 or not 0 <= now_ns - self.previous_ns <= MAX_GAP_NS:
            return None
        result = np.zeros((2, 100, 17, 3), dtype=np.float32)
        result[0] = np.stack([pose for _, pose in self.samples])
        return result


class JerseyConsensus:
    """Native two-second half-life, three-second expiry and switching hysteresis."""

    def __init__(self):
        self.evidence = {}
        self.observed = {}
        self.current = None
        self.now = None
        self.last_attempt = None

    def advance(self, timestamp_ns: int):
        if self.now is not None and timestamp_ns < self.now:
            self.__init__()
        if self.now is not None:
            decay = math.exp2(-(timestamp_ns - self.now) / 2e9)
            self.evidence = {
                text: score * decay
                for text, score in self.evidence.items()
                if timestamp_ns - self.observed[text] <= 3e9 and score * decay >= 0.0001
            }
        self.now = timestamp_ns
        best = max(self.evidence, key=self.evidence.get, default=None)
        if best is None or self.evidence[best] < 1:
            self.current = None
        elif (
            self.current is None or self.evidence[best] >= self.evidence.get(self.current, 0) * 1.35
        ):
            self.current = best

    def observe(self, timestamp_ns: int, text: str, confidence: float):
        self.advance(timestamp_ns)
        if timestamp_ns == self.last_attempt:
            return
        self.last_attempt = timestamp_ns
        self.evidence[text] = min(1000, self.evidence.get(text, 0) + confidence)
        self.observed[text] = timestamp_ns
        self.advance(timestamp_ns)

    def result(self, timestamp_ns: int):
        self.advance(timestamp_ns)
        if self.current is None:
            return None
        return self.current, self.evidence[self.current] / sum(self.evidence.values())


def jersey_roi(box, pose: np.ndarray | None, mode: str, width: int, height: int):
    box = np.asarray(box, dtype=np.float32)
    if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return None
    if mode == "bbox":
        size = box[2:] - box[:2]
        low = box[:2] + size * [0.2, 0.25]
        high = box[:2] + size * [0.8, 0.95]
    elif mode == "pose":
        if pose is None:
            return None
        if pose.shape != (17, 3):
            raise ValueError("Pose-guided hockey PARSeq requires COCO17 keypoints")
        torso = pose[[5, 6, 11, 12]]
        if not np.isfinite(torso).all() or np.any((torso[:, 2] < 0.4) | (torso[:, 2] > 1)):
            return None
        low, high = torso[:, :2].min(axis=0), torso[:, :2].max(axis=0)
        padding = (high - low) * 0.05
        low, high = low - padding, high + padding
    else:
        raise ValueError("Jersey ROI mode must be bbox or pose")
    low, high = np.floor(low).astype(int), np.ceil(high).astype(int)
    extent = high - low
    if (
        np.any(extent <= 0)
        or np.any(extent > 8192)
        or high[0] <= 0
        or high[1] <= 0
        or low[0] >= width
        or low[1] >= height
    ):
        return None
    return (*low, *high)


def parseq_input(image_bgr: np.ndarray, roi) -> np.ndarray:
    """Pillow's uint8 bicubic path matches native half-pixel antialiased resizing."""
    image = Image.fromarray(np.ascontiguousarray(image_bgr[..., ::-1]))
    crop = image.crop(tuple(map(int, roi))).resize((128, 32), Image.Resampling.BICUBIC)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1


def decode_parseq(logits: np.ndarray) -> tuple[str, float] | None:
    if logits.shape != (3, 95) or not np.isfinite(logits).all():
        raise ValueError("Hockey PARSeq requires finite [3,95] token logits")
    distribution = np.exp(logits - logits.max(axis=-1, keepdims=True))
    distribution /= distribution.sum(axis=-1, keepdims=True)
    text, confidence = "", 1.0
    for position, token in enumerate(distribution.argmax(axis=-1)):
        confidence *= float(distribution[position, token])
        if token == 0:
            return (text, confidence) if text else None
        if position == 2 or not 1 <= token <= 10:
            return None
        text += str(token - 1)
    return None
