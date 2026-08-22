"""Type stubs for `hockeymom.core`.

NOTE: These signatures are approximate because the underlying implementations
are provided by a pybind11 extension. Adjust types as the native API evolves.

This file is kept in sync with `core.py` which adds runtime docstrings.
"""

from __future__ import annotations

import enum
from typing import Any, Optional, Sequence, Tuple

import torch

class WHDims:
    width: int
    height: int
    def __init__(self, width: int, height: int) -> None: ...

class RemapperConfig:
    src_width: int
    src_height: int
    x_pos: int
    y_pos: int
    col_map: Optional[torch.Tensor]
    row_map: Optional[torch.Tensor]
    dtype: str
    add_alpha_channel: bool
    interpolation: str
    batch_size: int
    device: str
    def __init__(self) -> None: ...

class BlenderConfig:
    mode: str
    levels: int
    seam: Optional[torch.Tensor]
    xor_map: Optional[torch.Tensor]
    lazy_init: bool
    interpolation: str
    device: str
    def __init__(self) -> None: ...

class RemapImageInfo: ...
class StitchImageInfo: ...

class ImageRemapper:
    def __init__(self, config: RemapperConfig): ...
    def remap(self, image: torch.Tensor) -> torch.Tensor: ...

class ImageBlenderMode: ...  # enum-like

class ImageBlender:
    def __init__(self, config: BlenderConfig): ...
    def blend(self, images: Sequence[torch.Tensor]) -> torch.Tensor: ...

class CudaStitchPanoU8:
    def __init__(
        self,
        game_dir: str,
        batch_size: int,
        num_levels: int,
        input1: WHDims,
        input2: WHDims,
        minimize_blend: bool = True,
        max_output_width: int = 0,
    ): ...
    def process(
        self, d_input1: int, d_input2: int, d_canvas: int, stream: Optional[int]
    ) -> None: ...

class CudaStitchPanoF32(CudaStitchPanoU8): ...

class CudaStitchPano3U8:
    def __init__(
        self,
        game_dir: str,
        batch_size: int,
        num_levels: int,
        inputs: Sequence[WHDims],
    ): ...
    def process(self, d_inputs: Sequence[int], d_canvas: int, stream: Optional[int]) -> None: ...

class CudaStitchPano3F32(CudaStitchPano3U8): ...

class CudaStitchPanoNU8:
    def __init__(
        self,
        game_dir: str,
        batch_size: int,
        num_levels: int,
        input_sizes: Sequence[WHDims],
        minimize_blend: bool,
        quiet: bool = False,
    ): ...
    def process(self, d_inputs: Sequence[int], d_canvas: int, stream: Optional[int]) -> None: ...

class CudaStitchPanoNF32(CudaStitchPanoNU8): ...
class HmTrackerPredictionMode: ...  # enum-like

class HmLogLevel(enum.Enum):
    DEBUG: "HmLogLevel"
    INFO: "HmLogLevel"
    WARNING: "HmLogLevel"
    ERROR: "HmLogLevel"

class HmLogMessage:
    level: HmLogLevel
    message: str

class HmTracker:
    def update(self, detections: torch.Tensor) -> Any: ...

class HmByteTrackConfig: ...
class HmByteTracker(HmTracker): ...
class HmByteTrackerCuda(HmTracker): ...
class HmByteTrackerCudaStatic(HmTracker): ...
class HmDcfTrackerCudaStatic(HmTracker): ...
class PlayTrackerConfig: ...

class PlayTracker:
    def __init__(self, config: PlayTrackerConfig): ...
    def update(self, frame_id: int, tracks: torch.Tensor) -> Any: ...

class AllLivingBoxConfig: ...
class LivingBox: ...
class BBox: ...
class GrowShrink: ...  # enum-like

# AspenNet progress sampling helper (pybind11)
class AspenGraphSampler:
    def __init__(
        self,
        max_samples: int = ...,
        min_interval_ms: int = ...,
        max_interval_ms: int = ...,
    ) -> None: ...
    def configure_graph(
        self,
        names: Sequence[str],
        degrees: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def enter_index(self, index: int) -> None: ...
    def exit_index(self, index: int) -> None: ...
    def pop_samples(self, max_items: int = ...) -> list[dict[str, Any]]: ...

# Optional blender implementation
class EnBlender(ImageBlender): ...  # may be None at runtime

def compute_kmeans_clusters(
    data: torch.Tensor, k: int, max_iter: int = ...
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def create_homography_maps(
    left_points: Sequence[Tuple[float, float]],
    right_points: Sequence[Tuple[float, float]],
    left_width: int,
    left_height: int,
    right_width: int,
    right_height: int,
    reprojection_threshold: float = ...,
    confidence: float = ...,
    max_iterations: int = ...,
    max_output_dimension: int = ...,
) -> dict[str, Any]: ...
def create_affine_ransac_maps(
    left_points: Sequence[Tuple[float, float]],
    right_points: Sequence[Tuple[float, float]],
    left_width: int,
    left_height: int,
    right_width: int,
    right_height: int,
    reprojection_threshold: float = ...,
    confidence: float = ...,
    max_iterations: int = ...,
    refine_iterations: int = ...,
    max_output_dimension: int = ...,
) -> dict[str, Any]: ...
def show_cuda_tensor(
    label: str, img_cuda: torch.Tensor, wait: bool = ..., stream: Optional[int] = ...
) -> None: ...
def bgr_to_i420_cuda(bgr_hwc: torch.Tensor) -> torch.Tensor: ...

__all__ = [
    "ImageRemapper",
    "ImageBlender",
    "ImageBlenderMode",
    "CudaStitchPanoU8",
    "CudaStitchPanoF32",
    "CudaStitchPano3U8",
    "CudaStitchPano3F32",
    "BlenderConfig",
    "RemapImageInfo",
    "HmTracker",
    "HmByteTracker",
    "HmByteTrackerCuda",
    "HmByteTrackerCudaStatic",
    "HmDcfTrackerCudaStatic",
    "HmByteTrackConfig",
    "RemapperConfig",
    "HmTrackerPredictionMode",
    "HmLogLevel",
    "HmLogMessage",
    "StitchImageInfo",
    "EnBlender",
    "AspenGraphSampler",
    "PlayTracker",
    "PlayTrackerConfig",
    "AllLivingBoxConfig",
    "BBox",
    "LivingBox",
    "WHDims",
    "GrowShrink",
    "compute_kmeans_clusters",
    "create_affine_ransac_maps",
    "create_homography_maps",
    "bgr_to_i420_cuda",
    "show_cuda_tensor",
]
