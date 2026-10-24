"""Fit archive pixels to the selected hardware encoder without changing the canvas."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hmlib.utils.torch_backend import is_rocm_backend


def fit_encoder_dimensions(width: int, height: int, limits: Mapping[str, int]) -> tuple[int, int]:
    """Apply HStream's even, proportional NVENC raster and macroblock limits."""
    required = ("width_min", "height_min", "width_max", "height_max", "mb_num_max")
    if width < 2 or height < 2 or any(limits.get(key, 0) <= 0 for key in required):
        raise ValueError("Archive encoder dimensions require positive source and hardware limits")
    scale = min(1.0, limits["width_max"] / width, limits["height_max"] / height)

    def dimensions(factor: float) -> tuple[int, int]:
        return int(width * factor) & ~1, int(height * factor) & ~1

    def within_budget(size: tuple[int, int]) -> bool:
        return ((size[0] + 15) // 16) * ((size[1] + 15) // 16) <= limits["mb_num_max"]

    fitted = dimensions(scale)
    if not within_budget(fitted):
        low, high = 0.0, scale
        for _ in range(48):
            middle = (low + high) / 2
            if within_budget(dimensions(middle)):
                low = middle
            else:
                high = middle
        fitted = dimensions(low)
    if fitted[0] < max(2, limits["width_min"]) or fitted[1] < max(2, limits["height_min"]):
        raise ValueError(
            f"Stitched canvas {width}x{height} cannot fit the encoder's minimum dimensions "
            f"{limits['width_min']}x{limits['height_min']} without upscaling"
        )
    return fitted


def stitched_archive_dimensions(width: int, height: int, device: torch.device) -> tuple[int, int]:
    """Query the CUDA HEVC encoder used by the stitched archive; retain software behavior."""
    if device.type != "cuda" or is_rocm_backend():
        return width, height
    import PyNvVideoCodec

    gpu = device.index if device.index is not None else torch.cuda.current_device()
    try:
        limits = PyNvVideoCodec.GetEncoderCaps(gpuid=gpu, codec="hevc")
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot query the stitched archive HEVC encoder on GPU {gpu}"
        ) from error
    return fit_encoder_dimensions(width, height, limits)
