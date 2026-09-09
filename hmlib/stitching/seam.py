"""Load enblend seam masks at their PNG-declared canvas position."""

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import tifffile

from hmlib.stitching.artifact_validation import (
    MAX_SEAM_BYTES,
    bounded_file,
    validate_canvas,
    validate_mapping_tiff,
)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PngLayout:
    """Raster dimensions and pixel origin declared by a PNG file."""

    width: int
    height: int
    offset_x: int = 0
    offset_y: int = 0


def _read_exact(file, size: int, description: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated PNG {description}")
    return data


def read_png_layout(path: PathLike) -> PngLayout:
    """Read and validate PNG dimensions plus an optional pixel-unit ``oFFs`` chunk."""

    path = Path(path)
    bounded_file(path, MAX_SEAM_BYTES)
    with path.open("rb") as png_file:
        if _read_exact(png_file, 8, "signature") != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG signature: {path}")

        layout = None
        have_offset = False
        have_image_data = False
        have_end = False
        first_chunk = True

        while not have_end:
            chunk_header = png_file.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError(f"Truncated PNG chunk header: {path}")

            length, chunk_type = struct.unpack(">I4s", chunk_header)
            if first_chunk and chunk_type != b"IHDR":
                raise ValueError(f"PNG IHDR is not the first chunk: {path}")
            first_chunk = False

            expected_lengths = {b"IHDR": 13, b"oFFs": 9, b"IEND": 0}
            if chunk_type in expected_lengths and length != expected_lengths[chunk_type]:
                raise ValueError(f"Invalid PNG {chunk_type!r} chunk length: {path}")
            if length > MAX_SEAM_BYTES:
                raise ValueError(f"Oversized PNG chunk: {path}")
            crc = zlib.crc32(chunk_type)
            retained_data = bytearray()
            remaining = length
            while remaining:
                block = _read_exact(png_file, min(remaining, 64 * 1024), "chunk data")
                crc = zlib.crc32(block, crc)
                if chunk_type in (b"IHDR", b"oFFs"):
                    retained_data.extend(block)
                remaining -= len(block)

            expected_crc = struct.unpack(">I", _read_exact(png_file, 4, "chunk CRC"))[0]
            if crc != expected_crc:
                name = chunk_type.decode("ascii", errors="replace")
                raise ValueError(f"PNG {name} chunk has an invalid CRC: {path}")

            if chunk_type == b"IHDR":
                if layout is not None or length != 13:
                    raise ValueError(f"Invalid PNG IHDR chunk: {path}")
                width, height, depth, color, compression, filtering, interlace = struct.unpack(
                    ">IIBBBBB", retained_data
                )
                if (
                    color != 0
                    or depth not in (1, 2, 4, 8, 16)
                    or compression != 0
                    or filtering != 0
                    or interlace not in (0, 1)
                ):
                    raise ValueError(f"Unsupported grayscale PNG seam format: {path}")
                if width == 0 or height == 0:
                    raise ValueError(f"Invalid PNG dimensions: {path}")
                validate_canvas(width, height)
                layout = PngLayout(width=width, height=height)
            elif chunk_type == b"oFFs":
                if layout is None or have_offset or have_image_data or length != 9:
                    raise ValueError(f"Invalid PNG oFFs chunk: {path}")
                offset_x, offset_y, unit = struct.unpack(">iiB", retained_data)
                if unit != 0:
                    raise ValueError(f"PNG seam offset is not expressed in pixels: {path}")
                layout = PngLayout(
                    width=layout.width,
                    height=layout.height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
                have_offset = True
            elif chunk_type == b"IDAT":
                have_image_data = True
            elif chunk_type == b"IEND":
                if length != 0:
                    raise ValueError(f"Invalid PNG IEND chunk: {path}")
                have_end = True

        if layout is None:
            raise ValueError(f"PNG is missing its IHDR chunk: {path}")
        if not have_image_data:
            raise ValueError(f"PNG is missing its IDAT chunk: {path}")
        if not have_end:
            raise ValueError(f"PNG is missing its IEND chunk: {path}")
        return layout


def load_canvas_seam_mask(path: PathLike, canvas_width: int, canvas_height: int) -> np.ndarray:
    """Decode a seam and replicate its edges around the PNG crop on the full canvas."""

    validate_canvas(canvas_width, canvas_height)

    path = Path(path)
    layout = read_png_layout(path)
    right = layout.offset_x + layout.width
    bottom = layout.offset_y + layout.height
    if layout.offset_x < 0 or layout.offset_y < 0 or right > canvas_width or bottom > canvas_height:
        raise ValueError(
            f"PNG seam crop lies outside its mapping canvas: {path} "
            f"crop={layout.width}x{layout.height}+{layout.offset_x}+{layout.offset_y} "
            f"canvas={canvas_width}x{canvas_height}"
        )

    seam = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if seam is None or seam.ndim != 2 or seam.shape != (layout.height, layout.width):
        raise ValueError(f"PNG seam is not a decodable grayscale image: {path}")

    padding = (
        (layout.offset_y, canvas_height - bottom),
        (layout.offset_x, canvas_width - right),
    )
    if any(before or after for before, after in padding):
        seam = np.pad(seam, padding, mode="edge")
    return np.ascontiguousarray(seam)


def _tiff_tag_number(tag, default: float) -> float:
    if tag is None:
        return default
    value = tag.value
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            numerator, denominator = value
            return float(numerator) / float(denominator)
        if len(value) == 1:
            return float(value[0])
    return float(value)


def read_mapping_canvas_size(mapping_files: Sequence[PathLike]) -> tuple[int, int]:
    """Return the common canvas size described by positioned Hugin mapping TIFFs."""

    placements = []
    for mapping_file in mapping_files:
        validate_mapping_tiff(mapping_file)
        with tifffile.TiffFile(mapping_file) as tif:
            page = tif.pages[0]
            tags = page.tags
            x_resolution = _tiff_tag_number(tags.get("XResolution"), 1.0)
            y_resolution = _tiff_tag_number(tags.get("YResolution"), 1.0)
            x_position = _tiff_tag_number(tags.get("XPosition"), 0.0)
            y_position = _tiff_tag_number(tags.get("YPosition"), 0.0)
            if not all(
                math.isfinite(value)
                for value in (x_resolution, y_resolution, x_position, y_position)
            ):
                raise ValueError(f"Non-finite TIFF placement metadata: {mapping_file}")
            if x_resolution <= 0 or y_resolution <= 0:
                raise ValueError(f"Invalid TIFF placement resolution: {mapping_file}")
            # Playback's get_image_geo_position() quantizes each absolute
            # position before get_canvas_info() normalizes their common origin.
            # Match that ordering so the seam and runtime canvases cannot differ
            # by a pixel for fractional TIFF rationals.
            x = int(x_position * x_resolution + 0.5)
            y = int(y_position * y_resolution + 0.5)
            placements.append((x, y, int(page.imagewidth), int(page.imagelength)))

    if not placements:
        raise ValueError("No Hugin mapping TIFFs were provided")
    min_x = min(x for x, _, _, _ in placements)
    min_y = min(y for _, y, _, _ in placements)
    width = max(x - min_x + width for x, _, width, _ in placements)
    height = max(y - min_y + height for _, y, _, height in placements)
    validate_canvas(width, height)
    return width, height
