"""Native-compatible geometric rejection in detector pixel coordinates."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from hmlib.stitching.calibration import CalibrationAlignmentError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlPointFiltering:
    epipolar_ransac: bool = True
    homography_consistency: bool = True


def read_control_point_filtering(value=None) -> ControlPointFiltering:
    if isinstance(value, ControlPointFiltering):
        return value
    if value is None:
        return ControlPointFiltering()
    if not isinstance(value, Mapping) or set(value) - set(
        ControlPointFiltering.__dataclass_fields__
    ):
        raise ValueError(
            "stitching.control_point_filtering must contain only the two supported switches"
        )
    selected = {key: item for key, item in value.items() if item is not None}
    if any(not isinstance(item, bool) for item in selected.values()):
        raise ValueError("Control-point filtering switches must be boolean")
    return ControlPointFiltering(**selected)


def geometric_inliers(left, right, filtering=None, matcher="matcher") -> np.ndarray:
    """Reject epipolar outliers and gross contradictions to a strongly supported plane.

    Homography fitting uses all candidate matches, independently of epipolar
    filtering. The generous second threshold preserves ordinary rink parallax.
    """
    filtering = read_control_point_filtering(filtering)
    left, right = (np.asarray(points, dtype=np.float32) for points in (left, right))
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError("Geometric filtering requires paired Nx2 coordinates")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Geometric filtering requires finite coordinates")
    keep = np.ones(len(left), dtype=bool)
    if not filtering.epipolar_ransac and not filtering.homography_consistency:
        return np.flatnonzero(keep)
    if filtering.epipolar_ransac:
        if len(left) < 8:
            raise CalibrationAlignmentError(
                f"{matcher} produced fewer than eight matches for epipolar filtering"
            )
        matrix, mask = cv2.findFundamentalMat(left, right, cv2.FM_RANSAC, 1.0, 0.99, 2000)
        if matrix is None or mask is None or mask.size != len(left):
            raise CalibrationAlignmentError(f"{matcher} could not estimate epipolar geometry")
        keep = mask.ravel().astype(bool)
        if not keep.all():
            logger.warning(
                "%s epipolar filtering rejected %d of %d matches", matcher, (~keep).sum(), len(left)
            )
    if filtering.homography_consistency and len(left) >= 16:
        matrix, mask = cv2.findHomography(
            left, right, cv2.USAC_MAGSAC, 8.0, maxIters=2000, confidence=0.99
        )
        if (
            matrix is not None
            and mask is not None
            and mask.size == len(left)
            and np.count_nonzero(mask) >= 0.8 * len(left)
            and np.isfinite(matrix).all()
        ):
            invertible, inverse = cv2.invert(matrix)
            if invertible and np.isfinite(inverse).all():

                def error(transform, source, target):
                    projected = np.column_stack((source, np.ones(len(source)))) @ transform.T
                    denominator = projected[:, 2]
                    valid = np.abs(denominator) >= 1e-12
                    errors = np.full(len(source), np.inf)
                    errors[valid] = np.linalg.norm(
                        projected[valid, :2] / denominator[valid, None] - target[valid], axis=1
                    )
                    return errors

                consistent = (error(matrix, left, right) <= 32) & (
                    error(inverse, right, left) <= 32
                )
                rejected = np.count_nonzero(keep & ~consistent)
                if rejected:
                    logger.warning(
                        "%s homography consistency rejected %d gross contradictions",
                        matcher,
                        rejected,
                    )
                keep &= consistent
    indices = np.flatnonzero(keep)
    if len(indices) < 6:
        raise CalibrationAlignmentError(
            f"{matcher} geometric filtering retained fewer than six matches"
        )
    return indices
