"""Hue-preserving Rec. 709 shadow lift shared by CPU and GPU color grading.

The full-strength luma gamma (0.812) and optional black-point toe match
hstream's ShadowToneCurve. Inputs are floating RGB/BGR tensors or arrays in
the 0..255 range, with channels on the third-to-last axis.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def shadow_lift_settings(percent: float | None, black_point: bool | None) -> tuple[float, bool]:
    """Validate runtime settings, treating null as the disabled default."""
    amount = 0.0 if percent is None else float(percent)
    if not math.isfinite(amount) or not 0.0 <= amount <= 100.0:
        raise ValueError("shadow_lift must be a finite percentage between 0 and 100")
    if black_point is None:
        black_point = False
    if not isinstance(black_point, bool):
        raise ValueError("shadow_lift_black_point must be a boolean")
    return amount, black_point


def lift_tensor(
    image: torch.Tensor, percent: float, black_point: bool, channel_order: str
) -> torch.Tensor:
    """Apply the reference curve without transferring pixels off their device."""
    if percent == 0.0:
        return image
    amount = percent / 100.0
    gamma = 0.812**amount
    weights = (0.2126, 0.7152, 0.0722)
    if channel_order == "bgr":
        weights = weights[::-1]
    luma = sum(image[..., index : index + 1, :, :] * weight for index, weight in enumerate(weights))
    luma = luma / 255.0
    scale = torch.where(
        (luma > 0.0) & (luma < 1.0),
        luma.clamp_min(torch.finfo(image.dtype).tiny).pow(gamma - 1.0),
        1.0,
    )
    lifted = image * scale
    if black_point:
        toe = (1.0 - luma / 0.6).clamp(0.0, 1.0).square() * (amount * 0.15 * 255.0)
        lifted = lifted + torch.where(luma >= 0.0, toe, 0.0)
    return lifted.clamp(0.0, 255.0)


def lift_numpy(
    image: np.ndarray, percent: float, black_point: bool, channel_order: str
) -> np.ndarray:
    """Apply the same reference curve to a CPU image."""
    if percent == 0.0:
        return image
    amount = percent / 100.0
    gamma = 0.812**amount
    weights = (0.2126, 0.7152, 0.0722)
    if channel_order == "bgr":
        weights = weights[::-1]
    luma = sum(image[..., index : index + 1, :, :] * weight for index, weight in enumerate(weights))
    luma = luma / 255.0
    scale = np.where(
        (luma > 0.0) & (luma < 1.0),
        np.maximum(luma, np.finfo(image.dtype).tiny) ** (gamma - 1.0),
        1.0,
    )
    lifted = image * scale
    if black_point:
        toe = np.clip(1.0 - luma / 0.6, 0.0, 1.0) ** 2 * (amount * 0.15 * 255.0)
        lifted = lifted + np.where(luma >= 0.0, toe, 0.0)
    return np.clip(lifted, 0.0, 255.0)
