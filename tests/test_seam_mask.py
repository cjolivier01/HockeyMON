import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from hmlib.stitching.seam import (
    PngLayout,
    load_canvas_seam_mask,
    normalize_canvas_seam_mask,
    read_mapping_canvas_size,
    read_png_layout,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data))
    )


def _write_grayscale_png(
    path: Path,
    pixels: list[list[int]],
    offset: tuple[int, int] | None = None,
    offset_after_image_data: bool = False,
) -> None:
    height = len(pixels)
    width = len(pixels[0])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    image_data = zlib.compress(b"".join(b"\x00" + bytes(row) for row in pixels))
    chunks = [_png_chunk(b"IHDR", ihdr)]
    offset_chunk = None
    if offset is not None:
        offset_chunk = _png_chunk(b"oFFs", struct.pack(">iiB", *offset, 0))
        if not offset_after_image_data:
            chunks.append(offset_chunk)
    chunks.append(_png_chunk(b"IDAT", image_data))
    if offset_chunk is not None and offset_after_image_data:
        chunks.append(offset_chunk)
    chunks.append(_png_chunk(b"IEND", b""))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def should_place_cropped_seam_at_png_offset(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]], offset=(1, 2))

    seam = load_canvas_seam_mask(seam_file, canvas_width=5, canvas_height=5)

    assert seam.tolist() == [
        [10, 10, 20, 20, 20],
        [10, 10, 20, 20, 20],
        [10, 10, 20, 20, 20],
        [30, 30, 40, 40, 40],
        [30, 30, 40, 40, 40],
    ]


@pytest.mark.parametrize("offset", [(1, 2), (0, 0)])
def should_publish_positioned_seam_for_native_loading(tmp_path, offset):
    path = tmp_path / "seam_file.png"
    _write_grayscale_png(path, [[0, 255], [255, 0]], offset=offset)
    path.chmod(0o640)
    expected = load_canvas_seam_mask(path, 5, 5)

    normalize_canvas_seam_mask(path, 5, 5)

    assert read_png_layout(path) == PngLayout(5, 5)
    np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), expected)
    assert path.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.glob(".stitching-stage-*"))


def should_preserve_an_already_normalized_seam(tmp_path):
    path = tmp_path / "seam_file.png"
    _write_grayscale_png(path, [[0, 255], [255, 0]])
    original, info = path.read_bytes(), path.stat()

    normalize_canvas_seam_mask(path, 2, 2)

    assert path.read_bytes() == original
    assert path.stat().st_ino == info.st_ino
    assert path.stat().st_mtime_ns == info.st_mtime_ns


def should_normalize_hstream_one_pixel_crop_without_offset(tmp_path):
    path = tmp_path / "seam_file.png"
    _write_grayscale_png(path, [[0, 255], [255, 0]])

    normalize_canvas_seam_mask(path, 3, 3)

    assert cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).tolist() == [
        [0, 255, 255],
        [255, 0, 0],
        [255, 0, 0],
    ]


@pytest.mark.parametrize(
    "pixels,offset,error",
    [
        ([[0, 255], [255, 0]], None, "no crop offset"),
        ([[0, 255], [255, 0]], (4, 0), "outside its mapping canvas"),
        ([[0, 0], [0, 0]], (1, 1), "uniform"),
    ],
)
def should_preserve_invalid_seams_when_normalization_fails(tmp_path, pixels, offset, error):
    path = tmp_path / "seam_file.png"
    _write_grayscale_png(path, pixels, offset=offset)
    original = path.read_bytes()

    with pytest.raises(ValueError, match=error):
        normalize_canvas_seam_mask(path, 5, 5)

    assert path.read_bytes() == original


def should_preserve_original_seam_if_normalized_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "seam_file.png"
    _write_grayscale_png(path, [[0, 255], [255, 0]], offset=(1, 1))
    original = path.read_bytes()
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)

    with pytest.raises(OSError, match="Could not write normalized"):
        normalize_canvas_seam_mask(path, 5, 5)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".stitching-stage-*"))


def should_treat_a_missing_png_offset_as_the_canvas_origin(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]])

    seam = load_canvas_seam_mask(seam_file, canvas_width=4, canvas_height=3)

    assert seam.tolist() == [
        [10, 20, 20, 20],
        [30, 40, 40, 40],
        [30, 40, 40, 40],
    ]


@pytest.mark.parametrize("offset", [(-1, 0), (0, -1), (4, 0), (0, 4)])
def should_reject_a_seam_crop_outside_the_canvas(tmp_path: Path, offset: tuple[int, int]) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]], offset=offset)

    with pytest.raises(ValueError, match="outside its mapping canvas"):
        load_canvas_seam_mask(seam_file, canvas_width=5, canvas_height=5)


def should_reject_an_offset_after_image_data(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(
        seam_file,
        [[10, 20], [30, 40]],
        offset=(1, 1),
        offset_after_image_data=True,
    )

    with pytest.raises(ValueError, match="Invalid PNG oFFs chunk"):
        read_png_layout(seam_file)


def should_reject_a_corrupt_offset_chunk(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]], offset=(1, 1))
    png = bytearray(seam_file.read_bytes())
    offset_data = png.index(b"oFFs") + 4
    png[offset_data] ^= 1
    seam_file.write_bytes(png)

    with pytest.raises(ValueError, match="invalid CRC"):
        read_png_layout(seam_file)


def should_read_the_common_canvas_from_positioned_mapping_tiffs(tmp_path: Path) -> None:
    def write_mapping(
        path: Path,
        width: int,
        height: int,
        x: tuple[int, int],
        y: tuple[int, int],
    ) -> None:
        tifffile.imwrite(
            path,
            np.zeros((height, width), dtype=np.uint8),
            resolution=(1, 1),
            extratags=[
                (286, 5, 1, x, False),
                (287, 5, 1, y, False),
            ],
        )

    left = tmp_path / "mapping_0000.tif"
    right = tmp_path / "mapping_0001.tif"
    write_mapping(left, width=4, height=3, x=(15, 1), y=(26, 1))
    write_mapping(right, width=5, height=4, x=(18, 1), y=(24, 1))

    assert read_mapping_canvas_size([left, right]) == (8, 5)


def should_quantize_mapping_positions_before_normalizing_the_canvas(tmp_path: Path) -> None:
    def write_mapping(path: Path, x: tuple[int, int]) -> None:
        tifffile.imwrite(
            path,
            np.zeros((100, 200), dtype=np.uint8),
            resolution=(1, 1),
            extratags=[
                (286, 5, 1, x, False),
                (287, 5, 1, (0, 1), False),
            ],
        )

    left = tmp_path / "mapping_0000.tif"
    right = tmp_path / "mapping_0001.tif"
    write_mapping(left, x=(106, 10))
    write_mapping(right, x=(604, 10))

    # Playback rounds the absolute positions to 11 and 60 before subtracting
    # the common origin, giving a 49-pixel displacement and a 249-pixel canvas.
    assert read_mapping_canvas_size([left, right]) == (249, 100)
