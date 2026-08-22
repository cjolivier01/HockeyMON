"""HockeyMOM Python core re‑exports.

This module provides a thin, documented Python façade over the C++ / CUDA
implementations that are exposed via the compiled extension module
`hockeymom._hockeymom` (pybind11 + Torch).  All heavy‑lifting (image
remapping, multi‑camera stitching, blending, tracking, play detection, k‑means
clustering utilities, etc.) lives in optimized native code; here we **only**
re‑export public symbols and attach rich docstrings so that IDEs, static type
checkers and generated docs have something useful to display.

Implementation note:
    * Runtime types & methods are defined in C++; we cannot change their
        signatures here, but we can attach / override the `__doc__` attributes.
    * For tooling (mypy / editors) a parallel stub file `core.pyi` is generated
        with the same public API and type hints.  Update **both** this file and the
        stub if you add or remove symbols.
    * Docstrings use a hybrid style: first line summary, followed by extended
        description and `Args / Returns / Raises` (Google style) plus lightweight
        Doxygen tags (e.g. ``@param``) so that either documentation toolchain can
        parse them.

Environment assumptions:
    * Most tensor arguments are expected to be `torch.Tensor` with layouts noted
        per method (commonly HWC for uint8 image data or CHW for float batches).
    * CUDA vs CPU execution is controlled by each config's `.device` or by the
        tensor device.

High‑level components:
    * Remapping / Stitching (`RemapperConfig`, `ImageRemapper`, `ImageStitcher`,
        `CudaStitchPano*`) – geometric warping & panorama composition.
    * Blending (`BlenderConfig`, `ImageBlender`, optional `EnBlender`) – seam
        handling & Laplacian / multi‑band / exposure compensation.
    * Tracking (`HmTracker`, `HmByteTracker`, `PlayTracker`, `LivingBox`,
        `BBox`) – object & play sequence tracking utilities.
    * Utilities (`compute_kmeans_clusters`, `show_cuda_tensor`, `WHDims`,
        `GrowShrink`) – helper operations and simple data containers.

If you need details not covered here, inspect `hockeymom/src/PythonBindings.cpp`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path


def _preload_torch_shared_libraries() -> None:
    """
    Ensure Torch's shared libraries are loaded before importing `hockeymom._hockeymom`.

    When this package is imported outside Bazel, the dynamic loader does not know
    about Torch's `torch/lib` directory, so importing our extension can fail with
    missing dependencies (e.g. `libcaffe2_nvrtc.so`, `libtorch_cuda_linalg.so`).
    Preloading a small set of Torch libs with `RTLD_GLOBAL` makes the symbols
    available for the extension module to resolve.
    """

    try:
        import torch
    except Exception:
        return

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib_dir.is_dir():
        return

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", None)
    if rtld_global is None:
        return

    # Keep this list small and stable; only add libraries as needed.
    for name in (
        "libtorch_global_deps.so",
        "libcaffe2_nvrtc.so",
        "libtorch_cuda_linalg.so",
        "libtorch_cuda.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libc10.so",
        "libtorch.so",
    ):
        path = torch_lib_dir / name
        if not path.exists():
            continue
        try:
            ctypes.CDLL(str(path), mode=rtld_global)
        except OSError:
            # Best-effort: missing/ABI-mismatched libs should surface when importing the extension.
            pass


_preload_torch_shared_libraries()

from . import _hockeymom as _native_hockeymom
from ._hockeymom import (
    AllLivingBoxConfig,
    AspenGraphSampler,
    BBox,
    BlenderConfig,
    CudaStitchPano3F32,
    CudaStitchPano3U8,
    CudaStitchPanoF32,
    CudaStitchPanoNF32,
    CudaStitchPanoNU8,
    CudaStitchPanoU8,
    GrowShrink,
    HmByteTrackConfig,
    HmByteTracker,
    HmByteTrackerCuda,
    HmByteTrackerCudaStatic,
    HmDcfTrackerCudaStatic,
    HmLogLevel,
    HmLogMessage,
    HmTracker,
    HmTrackerPredictionMode,
    ImageBlender,
    ImageBlenderMode,
    ImageRemapper,
    ImageStitcher,
    LivingBox,
    PlayTracker,
    PlayTrackerConfig,
    RemapImageInfo,
    RemapperConfig,
    StitchImageInfo,
    WHDims,
    bgr_to_i420_cuda,
    compute_kmeans_clusters,
    show_cuda_tensor,
)

try:
    from ._hockeymom import EnBlender
except Exception:
    EnBlender = None


# REMOVEME(colivier)
# class HmByteTrackerCuda:
#     pass


def create_homography_maps(
    left_points,
    right_points,
    left_width,
    left_height,
    right_width,
    right_height,
    reprojection_threshold=3.0,
    confidence=0.999,
    max_iterations=10000,
    max_output_dimension=0,
):
    """Call the native MAGSAC++ homography-map builder."""
    native_function = getattr(_native_hockeymom, "create_homography_maps", None)
    if native_function is None:
        raise RuntimeError(
            "The installed HockeyMOM extension does not provide "
            "create_homography_maps; rebuild the native extension"
        )
    return native_function(
        left_points,
        right_points,
        left_width,
        left_height,
        right_width,
        right_height,
        reprojection_threshold,
        confidence,
        max_iterations,
        max_output_dimension,
    )


def create_affine_ransac_maps(
    left_points,
    right_points,
    left_width,
    left_height,
    right_width,
    right_height,
    reprojection_threshold=10.0,
    confidence=0.999,
    max_iterations=10000,
    refine_iterations=10,
    max_output_dimension=0,
):
    """Call the native affine RANSAC coordinate-map builder."""
    native_function = getattr(_native_hockeymom, "create_affine_ransac_maps", None)
    if native_function is None:
        raise RuntimeError(
            "The installed HockeyMOM extension does not provide "
            "create_affine_ransac_maps; rebuild the native extension"
        )
    return native_function(
        left_points,
        right_points,
        left_width,
        left_height,
        right_width,
        right_height,
        reprojection_threshold,
        confidence,
        max_iterations,
        refine_iterations,
        max_output_dimension,
    )


__all__ = [
    "ImageRemapper",
    "ImageBlender",
    "ImageBlenderMode",
    "CudaStitchPanoU8",
    "CudaStitchPanoF32",
    "CudaStitchPano3U8",
    "CudaStitchPano3F32",
    "CudaStitchPanoNU8",
    "CudaStitchPanoNF32",
    "BlenderConfig",
    "AspenGraphSampler",
    "ImageStitcher",
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


def _doc(target, text: str):  # pragma: no cover - trivial helper
    """Attach a docstring to a pybind11 exported object if possible."""
    try:
        if getattr(target, "__doc__", None):  # keep original at end
            target.__doc__ = text.rstrip() + "\n\nOriginal:\n" + target.__doc__
        else:
            target.__doc__ = text
    except Exception:
        pass
    return target


_doc(
    RemapperConfig,
    """Configuration for a single image remap / warp.

    @param src_width: Source image width in pixels.
    @param src_height: Source image height in pixels.
    @param x_pos: X offset of the (possibly remapped) image inside a panorama canvas.
    @param y_pos: Y offset of the (possibly remapped) image inside a panorama canvas.
    @param col_map: Optional (H, W) float/half tensor of destination X coordinates.
    @param row_map: Optional (H, W) float/half tensor of destination Y coordinates.
    @param dtype: Torch dtype string name for output (e.g. 'uint8', 'float32').
    @param add_alpha_channel: If True, append an opaque alpha channel to output.
    @param interpolation: Interpolation mode ('nearest', 'bilinear', ...).
    @param batch_size: Number of images processed in parallel.
    @param device: 'cpu' or 'cuda:<idx>'.
    """,
)

_doc(
    create_homography_maps,
    """Estimate a MAGSAC++ homography and build inverse panorama coordinate maps.

    The right control points are mapped onto the left image. The returned
    dictionary contains the robust inlier mask, panorama bounds, image
    placements, and uint16 inverse X/Y maps compatible with HockeyMOM's
    stitching remappers.
    """,
)

_doc(
    create_affine_ransac_maps,
    """Estimate an affine transform with RANSAC and build inverse coordinate maps.

    The right control points are mapped onto the left image. The returned
    dictionary has the same panorama geometry, inlier mask, image placement,
    and uint16 inverse X/Y map structure as ``create_homography_maps``.
    """,
)

_doc(
    BlenderConfig,
    """Multi‑band / seam blending configuration.

    Attributes:
        mode: Blend strategy – 'gpu-hard-seam', 'laplacian', maybe 'multiblend'.
        levels: Number of pyramid levels (0 = auto / minimal).
        seam: Optional torch.Tensor seam mask (H, W) or (H, W, 1/3).
        xor_map: Optional debugging seam XOR map.
        lazy_init: Defer GPU allocations until first use.
        interpolation: Interpolation for internal resampling.
        device: Target device (e.g. 'cuda:0').
    """,
)

_doc(
    ImageRemapper,
    """Warp an input image tensor into a destination canvas using `RemapperConfig`.

    Typical usage:
        cfg = RemapperConfig(src_width=1920, src_height=1080, ...)
        remapper = ImageRemapper(cfg)
        out = remapper.remap(image)  # image: [H, W, C] or batched variant.
    """,
)

_doc(
    ImageBlender,
    """Blend (composite) multiple already remapped images into a seamless panorama.

    Expects images with overlapping regions; uses seams or multi‑band blending
    depending on `BlenderConfig.mode`.
    """,
)

_doc(
    ImageBlenderMode,
    """Enum‑like container of blend mode constants exposed from C++.""",
)

_doc(
    ImageStitcher,
    """High‑level convenience wrapper orchestrating remap + blend for 2+ cameras.""",
)

_doc(
    CudaStitchPanoU8,
    """Fast CUDA panorama pipeline (uint8) – feeds raw device pointers (HWC/BGR).

    See also: `CudaStitchPanoF32`, multi‑input variants `CudaStitchPano3*`.
    """,
)
_doc(
    CudaStitchPanoF32,
    """Fast CUDA panorama pipeline (float32).  Allows higher precision blending.""",
)
_doc(
    CudaStitchPano3U8,
    """Three‑camera CUDA panorama (uint8).""",
)
_doc(
    CudaStitchPano3F32,
    """Three‑camera CUDA panorama (float32).""",
)

_doc(
    RemapImageInfo,
    """Lightweight struct describing an image remap (sizes, offsets, maybe maps).""",
)
_doc(
    StitchImageInfo,
    """Struct describing stitched panorama metadata (canvas size, etc.).""",
)

_doc(
    WHDims,
    """Width / Height dimensions helper.

    Fields:
        width (int)
        height (int)
    """,
)

_doc(
    GrowShrink,
    """Enum / configuration for growth vs shrink logic (tracking or mask ops).""",
)

_doc(
    HmTrackerPredictionMode,
    """Enum controlling tracker state prediction strategy (e.g. linear, kalman).""",
)

_doc(
    HmTracker,
    """Base multi‑object tracker (Kalman / BYTE fusion) exposed from C++.

    Provides frame‑wise `update(detections)` interface (see C++ binding for exact
    argument contract).  Tracks are typically returned as tensors / lists of
    structs with IDs, boxes, confidences.
    """,
)
_doc(
    HmByteTrackConfig,
    """Configuration knobs for BYTETracker (iou thresholds, track age limits, etc.).""",
)
_doc(
    HmByteTracker,
    """BYTETracker variant integrated with HockeyMOM pipeline (player detection).""",
)
_doc(
    HmByteTrackerCuda,
    """High-performance CUDA BYTETracker.

    Args:
        config: `HmByteTrackConfig` with thresholds / buffering options.
        device: Torch device string (default ``"cuda:0"``).

    Returns tracked tensors directly on the target CUDA device without host
    synchronization for the heavy math (Kalman, IoU, assignment).
    """,
)

_doc(
    HmByteTrackerCudaStatic,
    """Static-shape CUDA BYTETracker wrapper.

    Creates a CUDA tracker whose inputs/outputs keep fixed shapes suitable for
    CUDA graph capture or engines requiring static dimensions. Detection tensors
    must be padded to ``max_detections`` and provide a ``num_detections`` scalar;
    outputs are padded to ``max_tracks`` with a matching ``num_tracks`` entry.

    Args:
        config: ``HmByteTrackConfig`` thresholds and tracker knobs.
        max_detections: Maximum number of detections accepted per frame.
        max_tracks: Maximum number of active tracks returned.
        device: Torch device string (default ``"cuda:0"``).
    """,
)
_doc(
    HmDcfTrackerCudaStatic,
    """Static-shape CUDA DCF tracker with optional ReID features.

    Uses fixed-size buffers to avoid dynamic shapes and combines IoU with
    per-track appearance templates (when provided) for matching. The interface
    mirrors BYTETracker static mode (`track(data)` with padded inputs/outputs).
    """,
)

_doc(
    PlayTrackerConfig,
    """Configuration for play / sequence level tracker (heuristics & thresholds).""",
)
_doc(
    PlayTracker,
    """High‑level play detector that aggregates object tracks into semantic plays.""",
)

_doc(
    AllLivingBoxConfig,
    """Aggregate config affecting all `LivingBox` instances (e.g., decay, merge).""",
)
_doc(
    LivingBox,
    """Temporal box primitive with lifecycle (appears, persists, expires).""",
)
_doc(
    BBox,
    """Axis‑aligned bounding box container (x1, y1, x2, y2[, score, id]).""",
)

_doc(
    AspenGraphSampler,
    """Background AspenNet sampler that captures plugin activity without the GIL.""",
)

_doc(
    compute_kmeans_clusters,
    """Run k‑means on a (N, D) feature tensor.

    Args:
        data: torch.Tensor[N, D] float – input features.
        k: Number of clusters.
        max_iter: Optional iteration cap.

    Returns:
        (centers, assignments) – implementation specific; see C++ binding.
    """,
)

_doc(
    bgr_to_i420_cuda,
    """Convert a CUDA BGR uint8 image [H, W, 3] to planar I420/YUV420 layout.

    The output is a contiguous CUDA tensor of shape [H * 3 / 2, W] laid out as
    Y plane followed by U and V planes (4:2:0 subsampling).  Intended for use
    with GPU encoders that consume YUV420 surfaces (for example PyNvVideoCodec).
    """,
)

_doc(
    show_cuda_tensor,
    """Render a CUDA uint8 H×W×3 tensor in a native window (debug visualization).

    Args:
        label: Window title / render surface key.
        img_cuda: torch.ByteTensor[H, W, 3] on CUDA device (BGR ordering assumed).
        wait: If True, block for a short period / event loop iteration.
        stream: Optional CUDA stream pointer (int) for synchronization.
    """,
)

if EnBlender is not None:  # pragma: no cover - optional component
    _doc(
        EnBlender,
        """Alternative (likely CPU / external lib) blender implementation.

        Present only when compiled without NO_CPP_BLENDING. API mirrors `ImageBlender`.
        """,
    )
