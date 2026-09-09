"""Model-free AKAZE matching and optional paired GoPro KB4 lens calibration."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from hmlib.stitching.calibration import CalibrationAlignmentError

logger = logging.getLogger(__name__)
MAXIMUM_DETECTION_DIMENSION = 1920


@dataclass(frozen=True)
class FisheyeCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]

    def native_values(self) -> list[float]:
        return [self.width, self.height, self.fx, self.fy, self.cx, self.cy, *self.distortion]

    def camera_matrix(self, width: int, height: int) -> np.ndarray:
        sx, sy = width / self.width, height / self.height
        return np.array(
            [[self.fx * sx, 0, self.cx * sx], [0, self.fy * sy, self.cy * sy], [0, 0, 1]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LensCalibrationPair:
    left: FisheyeCalibration
    right: FisheyeCalibration
    fingerprint: str


def _parse_camera(document: Any, key: str) -> FisheyeCalibration:
    if not isinstance(document, dict) or not isinstance(document.get(key), dict):
        raise ValueError(f"AKAZE lens calibration is missing {key}")
    camera = document[key]
    values = []
    for field in ("width", "height", "fx", "fy", "cx", "cy"):
        value = camera.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"AKAZE lens calibration {key}.{field} must be finite")
        values.append(float(value))
    distortion = camera.get("d")
    if (
        not isinstance(distortion, list)
        or len(distortion) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in distortion
        )
    ):
        raise ValueError(f"AKAZE lens calibration {key}.d must contain four finite coefficients")
    if (
        not values[0].is_integer()
        or not values[1].is_integer()
        or any(value <= 0 for value in values[:4])
    ):
        raise ValueError(f"AKAZE lens calibration {key} has invalid dimensions or focal lengths")
    return FisheyeCalibration(
        int(values[0]), int(values[1]), *values[2:], tuple(map(float, distortion))
    )


def load_lens_calibration(game_dir: str | Path) -> LensCalibrationPair | None:
    """Pin, bound and fingerprint the exact profile used by matching and mapping."""
    path = Path(game_dir) / "left_calibration.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError:
        logger.warning(
            "AKAZE lens calibration missing at %s; matching original camera images", path
        )
        return None
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024 * 1024:
            raise ValueError(f"AKAZE profile must be a nonempty regular file at most 1 MiB: {path}")
        contents = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if len(contents) != before.st_size or (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise OSError(f"AKAZE lens profile changed while reading: {path}")
    try:
        document = json.loads(contents)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid AKAZE lens calibration JSON: {path}") from exc
    result = LensCalibrationPair(
        _parse_camera(document, "left_uniforms"),
        _parse_camera(document, "right_uniforms"),
        hashlib.sha256(contents).hexdigest(),
    )
    logger.info("Using paired KB4 lens calibration from %s", path)
    return result


def _detect_akaze(gray: np.ndarray, mask: np.ndarray):
    from hockeymon.core import detect_akaze_features

    return detect_akaze_features(gray, mask)


def _features(image: torch.Tensor, left: bool, calibration: FisheyeCalibration | None):
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    gray = cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    source_height, source_width = gray.shape
    scale = min(1.0, MAXIMUM_DETECTION_DIMENSION / max(gray.shape))
    width, height = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    if (height, width) != gray.shape:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    if calibration is not None:
        camera = calibration.camera_matrix(width, height)
        x_map, y_map = cv2.fisheye.initUndistortRectifyMap(
            camera,
            np.array(calibration.distortion),
            np.eye(3),
            camera,
            (width, height),
            cv2.CV_32FC1,
        )
        gray = cv2.remap(gray, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = np.zeros((height, width), np.uint8)
    x0, x1 = (width // 2, width) if left else (0, (width + 1) // 2)
    mask[math.floor(height * 0.05) : math.ceil(height * 0.95), x0:x1] = 255
    keypoints, descriptors = _detect_akaze(gray, mask)
    return keypoints, descriptors, (width, height), (source_width, source_height)


def _ratio_matches(query, train):
    neighbors = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(query, train, k=2)
    return {
        pair[0].queryIdx: pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[1].distance > 0 and pair[0].distance < 0.75 * pair[1].distance
    }


def match_akaze(
    image0: torch.Tensor, image1: torch.Tensor, calibration: LensCalibrationPair | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-sized rectified points when a KB4 profile is supplied."""
    left, descriptors0, size0, source0 = _features(
        image0, True, calibration.left if calibration else None
    )
    right, descriptors1, size1, source1 = _features(
        image1, False, calibration.right if calibration else None
    )
    if (
        descriptors0 is None
        or descriptors1 is None
        or len(descriptors0) < 2
        or len(descriptors1) < 2
    ):
        raise CalibrationAlignmentError("AKAZE produced no usable M-LDB descriptors")
    forward, reverse = _ratio_matches(descriptors0, descriptors1), _ratio_matches(
        descriptors1, descriptors0
    )
    accepted = []
    for index, match in forward.items():
        if match.trainIdx not in reverse or reverse[match.trainIdx].trainIdx != index:
            continue
        point0, point1 = np.array(left[index]), np.array(right[match.trainIdx])
        normalized0, normalized1 = point0 / size0, point1 / size1
        if (
            normalized0[0] < 0.5
            or normalized1[0] > 0.5
            or not (0.2 <= normalized0[1] <= 0.8 and 0.2 <= normalized1[1] <= 0.8)
            or abs(normalized0[1] - normalized1[1]) > 0.08
        ):
            continue
        accepted.append((point0, point1, match.distance, index, match.trainIdx))
    if len(accepted) < 8:
        raise CalibrationAlignmentError("AKAZE found fewer than eight mutual overlap matches")
    points0, points1 = (
        np.array([entry[index] for entry in accepted], dtype=np.float32) for index in (0, 1)
    )
    fundamental, inliers = cv2.findFundamentalMat(points0, points1, cv2.FM_RANSAC, 1.0, 0.99, 2000)
    if fundamental is None or inliers is None or inliers.size != len(accepted):
        raise CalibrationAlignmentError("AKAZE could not estimate overlap epipolar geometry")
    accepted = [entry for entry, keep in zip(accepted, inliers.ravel()) if keep]
    if len(accepted) < 6:
        raise CalibrationAlignmentError("AKAZE epipolar filtering retained fewer than six matches")
    accepted.sort(key=lambda entry: (entry[2], entry[3], entry[4]))
    return tuple(
        torch.from_numpy(
            np.array([entry[index] for entry in accepted], dtype=np.float32)
            * np.array(source, dtype=np.float32)
            / np.array(size, dtype=np.float32)
        )
        for index, source, size in ((0, source0, size0), (1, source1, size1))
    )
