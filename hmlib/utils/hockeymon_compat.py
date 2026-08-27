"""Shared `hockeymon` native-extension symbols for Python callers."""

from __future__ import annotations

from hockeymon.core import (
    BlenderConfig,
    CudaStitchPanoF32,
    CudaStitchPanoNF32,
    CudaStitchPanoNU8,
    CudaStitchPanoU8,
    HmByteTrackConfig,
    HmTracker,
    HmTrackerPredictionMode,
    ImageBlender,
    ImageBlenderMode,
    ImageRemapper,
    RemapImageInfo,
    WHDims,
)

try:
    from hockeymon.core import EnBlender
except ImportError:
    EnBlender = None

HOCKEYMON_AVAILABLE = True
HOCKEYMON_IMPORT_ERROR = None


def hockeymon_error_message() -> str:
    return ""


def require_hockeymon(feature: str) -> None:
    del feature


__all__ = [
    "BlenderConfig",
    "CudaStitchPanoF32",
    "CudaStitchPanoNF32",
    "CudaStitchPanoNU8",
    "CudaStitchPanoU8",
    "EnBlender",
    "HmByteTrackConfig",
    "HmTracker",
    "HmTrackerPredictionMode",
    "HOCKEYMON_AVAILABLE",
    "HOCKEYMON_IMPORT_ERROR",
    "ImageBlender",
    "ImageBlenderMode",
    "ImageRemapper",
    "RemapImageInfo",
    "WHDims",
    "hockeymon_error_message",
    "require_hockeymon",
]
