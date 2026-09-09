"""Sample synchronized frames and rank independent calibration candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import cv2
import torch

logger = logging.getLogger(__name__)


class CalibrationAlignmentError(RuntimeError):
    """A frame's correspondences failed geometric alignment; another may work."""


@dataclass(frozen=True)
class CalibrationCandidate:
    images: tuple[Path, Path]
    points: Mapping[str, torch.Tensor]
    label: str


def sample_frame_indices(
    left_start: int, right_start: int, count: int, left_frames: int, right_frames: int
) -> list[tuple[int, int]]:
    """Use the first synchronized pairs at the requested time, clipped at EOF."""
    if left_start < 0 or right_start < 0:
        raise ValueError("Calibration frame offsets must be nonnegative")
    if not 1 <= count <= 64:
        raise ValueError("calibration_frame_count must be between 1 and 64")
    available = min(left_frames - left_start, right_frames - right_start)
    if available <= 0:
        raise ValueError("Calibration start lies beyond the synchronized video range")
    if available < count:
        logger.warning(
            "Only %d synchronized calibration pairs remain; requested %d", available, count
        )
    return [(left_start + index, right_start + index) for index in range(min(count, available))]


def calibration_candidates(
    image_pairs: Sequence[tuple[Path, Path]],
    max_control_points: int,
    matcher: Callable[..., Mapping[str, torch.Tensor]],
    matcher_name: str,
) -> Iterator[CalibrationCandidate]:
    """Try pooled correspondences first, then strongest individual frame pairs.

    Only explicit geometric failures can skip a pair. Missing files, model
    failures, invalid tensors and memory errors retain their original cause.
    """
    candidates: list[CalibrationCandidate] = []
    dimensions = None
    for index, images in enumerate(image_pairs):
        sizes = []
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Cannot decode calibration image {path}")
            sizes.append(image.shape[:2])
        if dimensions is not None and sizes != dimensions:
            raise ValueError("Calibration frame pairs must have stable source dimensions")
        dimensions = sizes
        try:
            points = matcher(
                str(images[0]),
                str(images[1]),
                max_control_points=max_control_points,
                matcher=matcher_name,
            )
        except CalibrationAlignmentError as exc:
            logger.warning("Skipping calibration pair %d/%d: %s", index + 1, len(image_pairs), exc)
            continue
        left, right = points["m_kpts0"], points["m_kpts1"]
        if left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape:
            raise ValueError("Calibration matcher returned malformed point pairs")
        for values, (height, width) in zip((left, right), sizes):
            if (
                not torch.isfinite(values).all()
                or (values < 0).any()
                or (values[:, 0] >= width).any()
                or (values[:, 1] >= height).any()
            ):
                raise ValueError("Calibration matcher returned points outside source images")
        if len(left) < 4:
            logger.warning("Skipping calibration pair %d: fewer than four matches", index + 1)
            continue
        candidates.append(CalibrationCandidate(images, points, f"frame pair {index + 1}"))
    if not candidates:
        raise CalibrationAlignmentError("No synchronized calibration pair produced usable matches")
    candidates.sort(key=lambda candidate: len(candidate.points["m_kpts0"]), reverse=True)
    if len(candidates) > 1:
        from hmlib.stitching.control_points import select_evenly_spaced

        pooled = {
            key: torch.cat([candidate.points[key] for candidate in candidates])
            for key in ("m_kpts0", "m_kpts1")
        }
        # Repeated static points must not crowd different correspondences out.
        paired = torch.cat((pooled["m_kpts0"], pooled["m_kpts1"]), dim=1)
        paired = torch.unique(paired, dim=0)
        indices = select_evenly_spaced(paired[:, :2], max_control_points)
        pooled = {"m_kpts0": paired[indices, :2], "m_kpts1": paired[indices, 2:]}
        yield CalibrationCandidate(
            candidates[0].images, pooled, f"pooled matches from {len(candidates)} frame pairs"
        )
    yield from candidates
