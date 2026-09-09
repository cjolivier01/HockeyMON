"""Bound stitching metadata before allocating decoded maps or canvas arrays."""

import stat
import struct
from pathlib import Path

import numpy as np
import tifffile

MAX_CANVAS_EDGE = 65536
MAX_CANVAS_PIXELS = 256 * 1024 * 1024
MAX_PLACEMENT_BYTES = 2 * 1024**3
MAX_SEAM_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 1024**2


def validate_canvas(width: int, height: int) -> None:
    if (
        width <= 0
        or height <= 0
        or max(width, height) > MAX_CANVAS_EDGE
        or width * height > MAX_CANVAS_PIXELS
    ):
        raise ValueError(f"Stitching canvas exceeds supported bounds: {width}x{height}")


def bounded_file(path: str | Path, maximum: int) -> int:
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise ValueError(f"Invalid or oversized stitching artifact: {path}")
    return info.st_size


def _bounded_tiff_ifd(path: Path, size: int) -> None:
    """Bound IFD/tag payloads before tifffile can materialize metadata arrays."""
    with path.open("rb") as stream:

        def read(length):
            data = stream.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated stitching TIFF metadata: {path}")
            return data

        endian = read(2)
        if endian not in (b"II", b"MM"):
            raise ValueError(f"Invalid stitching TIFF byte order: {path}")
        order = "<" if endian == b"II" else ">"
        version = struct.unpack(order + "H", read(2))[0]
        if version == 42:
            offset_format, count_format, entry_format, inline_size = "I", "H", "HHII", 4
        elif version == 43 and read(4) == struct.pack(order + "HH", 8, 0):
            offset_format, count_format, entry_format, inline_size = "Q", "Q", "HHQQ", 8
        else:
            raise ValueError(f"Unsupported stitching TIFF header: {path}")
        offset_size = struct.calcsize(offset_format)
        offset = struct.unpack(order + offset_format, read(offset_size))[0]
        if offset < stream.tell() or offset >= size:
            raise ValueError(f"Invalid stitching TIFF IFD offset: {path}")
        stream.seek(offset)
        count = struct.unpack(order + count_format, read(struct.calcsize(count_format)))[0]
        if not 0 < count <= 4096:
            raise ValueError(f"Oversized stitching TIFF IFD: {path}")
        type_sizes = {
            1: 1,
            2: 1,
            3: 2,
            4: 4,
            5: 8,
            6: 1,
            7: 1,
            8: 2,
            9: 4,
            10: 8,
            11: 4,
            12: 8,
            13: 4,
            16: 8,
            17: 8,
            18: 8,
        }
        for _ in range(count):
            tag, kind, length, value = struct.unpack(
                order + entry_format, read(struct.calcsize(entry_format))
            )
            if kind not in type_sizes:
                raise ValueError(f"Unsupported stitching TIFF tag type {kind}: {path}")
            payload = length * type_sizes[kind]
            if payload > MAX_METADATA_BYTES:
                raise ValueError(f"Oversized stitching TIFF tag {tag}: {path}")
            if payload > inline_size and (value > size or payload > size - value):
                raise ValueError(f"Truncated stitching TIFF tag {tag}: {path}")
        if struct.unpack(order + offset_format, read(offset_size))[0] != 0:
            raise ValueError(f"Expected a single-page stitching TIFF: {path}")


def validate_mapping_tiff(path: str | Path, *, coordinates: bool = False) -> tuple[int, int]:
    path = Path(path)
    size = bounded_file(path, MAX_PLACEMENT_BYTES)
    _bounded_tiff_ifd(path, size)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        width, height = int(page.imagewidth), int(page.imagelength)
        validate_canvas(width, height)
        if coordinates:
            if page.samplesperpixel != 1 or page.dtype != np.dtype("uint16"):
                raise ValueError(f"Expected a uint16 single-channel coordinate map: {path}")
            if size > width * height * 4 + 32 * 1024**2:
                raise ValueError(f"Oversized stitching coordinate TIFF: {path}")
        for offset, length in zip(page.dataoffsets, page.databytecounts, strict=True):
            if offset <= 0 or length <= 0 or offset > size or length > size - offset:
                raise ValueError(f"Truncated stitching TIFF raster: {path}")
    return width, height


def validate_artifact_generation(
    directory: str | Path,
    *,
    project_name: str | None = None,
    basenames: tuple[str, ...] = ("mapping_0000", "mapping_0001"),
) -> tuple[int, int]:
    """Validate all required files without decoding placement images or allocating maps."""
    from hmlib.stitching.seam import read_mapping_canvas_size, read_png_layout

    directory = Path(directory)
    if project_name is not None:
        for name in (project_name, "autooptimiser_out.pto"):
            bounded_file(directory / name, 32 * 1024**2)
    mapping_paths = []
    for name in basenames:
        mapping = directory / f"{name}.tif"
        dimensions = validate_mapping_tiff(mapping)
        for axis in ("x", "y"):
            if (
                validate_mapping_tiff(directory / f"{name}_{axis}.tif", coordinates=True)
                != dimensions
            ):
                raise ValueError(f"Mismatched stitching coordinate map dimensions: {name}_{axis}")
        mapping_paths.append(mapping)
    width, height = read_mapping_canvas_size(mapping_paths)
    seam = read_png_layout(directory / "seam_file.png")
    if (
        seam.offset_x < 0
        or seam.offset_y < 0
        or seam.offset_x + seam.width > width
        or seam.offset_y + seam.height > height
    ):
        raise ValueError(f"Stitching seam lies outside the {width}x{height} canvas")
    return width, height
