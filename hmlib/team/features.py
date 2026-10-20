"""Appearance features for team affiliation.

One analytic signal, training-free: the mean CIELAB colour of a torso crop.
Lightness alone separates the common dark-versus-white hockey matchup and is
what names the two clusters; the a/b channels carry the rest when the two
sweaters sit at similar luminance.

This is the feature the ``lab`` backend clusters on, and -- whichever backend is
selected -- it is always what supplies the lightness used to rank the two groups
dark-vs-light, because a learned embedding has no photometric axis to rank by.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert float RGB in [0, 1], shaped ``(..., 3)``, to CIELAB (D65).

    L is in [0, 100]; a and b are roughly [-128, 127]. Implemented in torch so
    the crop never has to leave the device it was produced on.
    """
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected a trailing RGB axis, got shape {tuple(rgb.shape)}")

    lin = _srgb_to_linear(rgb.clamp(0.0, 1.0))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    # sRGB D65 -> XYZ, then normalised by the D65 white point.
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.clamp_min(1e-12) ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return torch.stack(
        (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)),
        dim=-1,
    )


def _as_hwc_float(crop: torch.Tensor) -> torch.Tensor:
    """Normalise a crop to ``(H, W, 3)`` float RGB in [0, 1]."""
    if crop.ndim != 3:
        raise ValueError(f"expected a 3-D crop, got shape {tuple(crop.shape)}")
    # Channels-first crops are the common case out of the tiling helpers.
    if crop.shape[0] == 3 and crop.shape[-1] != 3:
        crop = crop.permute(1, 2, 0)
    if crop.shape[-1] != 3:
        raise ValueError(f"expected a 3-channel crop, got shape {tuple(crop.shape)}")

    out = crop.to(dtype=torch.float32)
    if crop.dtype == torch.uint8:
        return out / 255.0
    # Float crops arrive either already normalised or in 0..255.
    return out / 255.0 if float(out.max()) > 1.0 else out


#: Side length the crop is pooled to before the CIELAB conversion.
#:
#: Only the crop's mean colour is wanted, and converting every pixel of a
#: ~320x300 crop to compute one 3-vector is almost all wasted work: measured
#: over 12 crops, full-resolution per-crop conversion cost 3.07 ms against
#: 0.40 ms for the pooled batch. Area-pooling first is not exactly the same
#: number -- CIELAB is non-linear, so this is the mean of locally-averaged
#: colours rather than the mean of per-pixel ones -- but on a sweater crop,
#: which is what the ROI is cut to be, the two agree closely.
LAB_POOL_SIZE = 32


def batched_lab_mean(batch: torch.Tensor) -> torch.Tensor:
    """Mean CIELAB of each crop in an ``(N, 3, H, W)`` batch, as ``(N, 3)``.

    One pooling op, one conversion and one reduction over the whole batch,
    rather than a Python loop that launches the same kernels per crop and
    synchronises each time.
    """
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"expected an (N, 3, H, W) batch, got {tuple(batch.shape)}")
    if batch.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=batch.device)

    # Crop batches are 0-255 float by convention (see
    # TeamAffiliationPlugin._cut_crops). Testing the data instead would mean
    # a `.max()` per frame, and that is a GPU->CPU sync on the hot path.
    pooled = F.adaptive_avg_pool2d(batch.to(dtype=torch.float32), LAB_POOL_SIZE) / 255.0
    lab = rgb_to_lab(pooled.permute(0, 2, 3, 1))
    return lab.reshape(batch.shape[0], -1, 3).mean(dim=1)


def lab_torso_feature(crop: torch.Tensor) -> torch.Tensor:
    """Mean CIELAB colour of ``crop``, returned as a ``(3,)`` float tensor.

    The mean is a weak summary when the crop is not purely sweater -- measured
    over 2293 real crops, a band that also caught pants, skates and the ice
    between the legs averaged a gold sweater down to |a|,|b| ~= 2-3. That is why
    :func:`hmlib.bbox.torso.bbox_upper_torso_fallback` exists; keep the crop
    tight and this stays informative.
    """
    lab = rgb_to_lab(_as_hwc_float(crop))
    return lab.reshape(-1, 3).mean(dim=0)
