"""Native OpenCV mapping-file generation for two-camera stitching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import tifffile
import torch

INVALID_MAP_COORDINATE = np.iinfo(np.uint16).max
MAXIMUM_MAP_DIMENSION = int(INVALID_MAP_COORDINATE) - 1
_TIFF_RESOLUTION = 150


def _native_create_homography_maps(
    left_points: Sequence[Sequence[float]],
    right_points: Sequence[Sequence[float]],
    left_width: int,
    left_height: int,
    right_width: int,
    right_height: int,
    max_output_dimension: int,
) -> Mapping[str, Any]:
    try:
        from hockeymon.core import create_homography_maps
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The opencv-magsac mapping backend requires a HockeyMON extension "
            "built with create_homography_maps support"
        ) from exc

    return create_homography_maps(
        left_points,
        right_points,
        left_width,
        left_height,
        right_width,
        right_height,
        reprojection_threshold=3.0,
        confidence=0.999,
        max_iterations=10000,
        max_output_dimension=max_output_dimension,
    )


def _native_create_affine_ransac_maps(
    left_points: Sequence[Sequence[float]],
    right_points: Sequence[Sequence[float]],
    left_width: int,
    left_height: int,
    right_width: int,
    right_height: int,
    max_output_dimension: int,
) -> Mapping[str, Any]:
    try:
        from hockeymon.core import create_affine_ransac_maps
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The opencv-affine-ransac mapping backend requires a HockeyMON "
            "extension built with create_affine_ransac_maps support"
        ) from exc

    return create_affine_ransac_maps(
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
        max_output_dimension=max_output_dimension,
    )


def _position_tags(
    x_position: int,
    y_position: int,
    canvas_width: int,
    canvas_height: int,
) -> list[tuple[int, str, int, Any, bool]]:
    return [
        (286, "2I", 1, (int(x_position), _TIFF_RESOLUTION), False),
        (287, "2I", 1, (int(y_position), _TIFF_RESOLUTION), False),
        (33300, "I", 1, int(canvas_width), False),
        (33301, "I", 1, int(canvas_height), False),
    ]


def _write_positioned_rgba_tiff(
    path: Path,
    image: np.ndarray,
    x_position: int,
    y_position: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        planarconfig="contig",
        extrasamples="unassalpha",
        compression=None,
        resolution=(_TIFF_RESOLUTION, _TIFF_RESOLUTION),
        metadata=None,
        extratags=_position_tags(
            x_position,
            y_position,
            canvas_width,
            canvas_height,
        ),
        bigtiff=image.nbytes >= (2**32 - 2**25),
    )


def _write_coordinate_tiff(path: Path, coordinate_map: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        coordinate_map,
        photometric="minisblack",
        compression=None,
        resolution=(_TIFF_RESOLUTION, _TIFF_RESOLUTION),
        metadata=None,
        bigtiff=coordinate_map.nbytes >= (2**32 - 2**25),
    )


def _remap_reference_image(
    image_bgr: np.ndarray,
    x_map: np.ndarray,
    y_map: np.ndarray,
) -> np.ndarray:
    invalid = (x_map == INVALID_MAP_COORDINATE) | (y_map == INVALID_MAP_COORDINATE)
    map_x = x_map.astype(np.float32)
    map_y = y_map.astype(np.float32)
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    remapped = cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = np.where(invalid, 0, 255).astype(np.uint8)
    return np.dstack((remapped, alpha))


def _create_opencv_mapping_files(
    image_files: Sequence[str],
    control_points: Mapping[str, torch.Tensor],
    output_directory: str | Path,
    max_output_dimension: int | None,
    native_builder: Callable[..., Mapping[str, Any]],
    minimum_points: int,
    estimator_name: str,
) -> list[str]:
    if len(image_files) != 2:
        raise ValueError("Exactly two input images are required")
    if max_output_dimension is not None:
        max_output_dimension = int(max_output_dimension)
        if not 0 < max_output_dimension <= MAXIMUM_MAP_DIMENSION:
            raise ValueError(
                "max_output_dimension must be between 1 and " f"{MAXIMUM_MAP_DIMENSION}"
            )
    points0 = control_points["m_kpts0"].detach().cpu().to(torch.float64).tolist()
    points1 = control_points["m_kpts1"].detach().cpu().to(torch.float64).tolist()
    if len(points0) != len(points1):
        raise ValueError("Left and right control-point counts must match")
    if len(points0) < minimum_points:
        raise ValueError(
            f"At least {minimum_points} control-point pairs are required for {estimator_name}"
        )

    images: list[np.ndarray] = []
    for image_file in image_files:
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read stitching reference image: {image_file}")
        images.append(image)

    left_height, left_width = images[0].shape[:2]
    right_height, right_width = images[1].shape[:2]
    result = native_builder(
        points0,
        points1,
        left_width,
        left_height,
        right_width,
        right_height,
        int(max_output_dimension or 0),
    )
    image_maps = result["image_maps"]
    if len(image_maps) != 2:
        raise RuntimeError("Native OpenCV mapping returned an invalid image-map count")

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_files: list[str] = []
    for index, (image, image_map) in enumerate(zip(images, image_maps)):
        x_map = np.asarray(image_map["x_map"], dtype=np.uint16)
        y_map = np.asarray(image_map["y_map"], dtype=np.uint16)
        if x_map.ndim != 2 or x_map.shape != y_map.shape:
            raise RuntimeError("Native OpenCV coordinate maps have invalid shapes")

        basename = output_dir / f"mapping_{index:04d}"
        mapping_file = basename.with_suffix(".tif")
        rgba = _remap_reference_image(image, x_map, y_map)
        _write_positioned_rgba_tiff(
            mapping_file,
            rgba,
            int(image_map["x_position"]),
            int(image_map["y_position"]),
            int(result["canvas_width"]),
            int(result["canvas_height"]),
        )
        _write_coordinate_tiff(output_dir / f"{basename.name}_x.tif", x_map)
        _write_coordinate_tiff(output_dir / f"{basename.name}_y.tif", y_map)
        mapping_files.append(str(mapping_file))

    return mapping_files


def create_opencv_magsac_mapping_files(
    image_files: Sequence[str],
    control_points: Mapping[str, torch.Tensor],
    output_directory: str | Path,
    max_output_dimension: int | None = None,
) -> list[str]:
    """Create nona-compatible TIFF maps from a native MAGSAC++ homography."""
    return _create_opencv_mapping_files(
        image_files,
        control_points,
        output_directory,
        max_output_dimension,
        _native_create_homography_maps,
        minimum_points=4,
        estimator_name="a homography",
    )


def create_opencv_affine_ransac_mapping_files(
    image_files: Sequence[str],
    control_points: Mapping[str, torch.Tensor],
    output_directory: str | Path,
    max_output_dimension: int | None = None,
) -> list[str]:
    """Create nona-compatible TIFF maps from a native affine RANSAC fit."""
    return _create_opencv_mapping_files(
        image_files,
        control_points,
        output_directory,
        max_output_dimension,
        _native_create_affine_ransac_maps,
        minimum_points=3,
        estimator_name="an affine transform",
    )
