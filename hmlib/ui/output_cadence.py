"""Convert output cadence without changing the source analytics timeline."""

from __future__ import annotations

import math
from fractions import Fraction


def recording_fps(config: dict, source_fps: float) -> float:
    value = config.get("video_out", {}).get("output_fps")
    if value is None:
        value = config.get("camera", {}).get("output-fps")
    value = float(value) if value is not None else source_fps
    if value == 0:
        value = source_fps
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Recording FPS must be finite and positive, or 0 for source cadence")
    return value


class OutputCadence:
    """Select source frames at each output timestamp, holding frames when needed.

    Integer ratios keep fractional rates and batch boundaries deterministic.
    The output duration differs by less than one output frame from the input.
    """

    def __init__(self, source_fps: float, output_fps: float) -> None:
        if not all(math.isfinite(rate) and rate > 0 for rate in (source_fps, output_fps)):
            raise ValueError("Source and output frame rates must be finite and positive")
        self.source_fps = source_fps
        self.output_fps = output_fps
        ratio = Fraction(str(output_fps)) / Fraction(str(source_fps))
        self._numerator, self._denominator = ratio.numerator, ratio.denominator
        self.source_count = 0
        self.output_count = 0

    def select(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("Frame batch size cannot be negative")
        end = self.source_count + count
        target = (end * self._numerator + self._denominator - 1) // self._denominator
        indices = [
            index * self._denominator // self._numerator - self.source_count
            for index in range(self.output_count, target)
        ]
        self.source_count = end
        self.output_count = target
        return indices
