"""Appearance features used for team affiliation."""

from __future__ import annotations

import torch

from hmlib.team.features import lab_torso_feature, rgb_to_lab


def _solid(rgb, height: int = 48, width: int = 32) -> torch.Tensor:
    """A uint8 HWC crop of one flat colour."""
    crop = torch.zeros((height, width, 3), dtype=torch.uint8)
    for channel, value in enumerate(rgb):
        crop[:, :, channel] = value
    return crop


def should_convert_known_colours_to_lab():
    white = rgb_to_lab(torch.ones((1, 1, 3), dtype=torch.float32))
    black = rgb_to_lab(torch.zeros((1, 1, 3), dtype=torch.float32))
    assert float(white[0, 0, 0]) == torch.tensor(100.0).item()
    assert abs(float(white[0, 0, 1])) < 0.1 and abs(float(white[0, 0, 2])) < 0.1
    assert abs(float(black[0, 0, 0])) < 1e-4


def should_match_reference_lab_values_for_saturated_colours():
    """White and black alone cannot catch a chroma-scaling bug, so pin real hues.

    Reference values are the standard sRGB D65 conversions; an earlier version
    of this file only checked the achromatic ends and would have passed with
    the a/b axes scaled by any constant.
    """
    cases = {
        (218, 165, 32): (70.8, 8.5, 68.8),  # gold
        (200, 30, 30): (43.2, 63.1, 45.2),  # red
        (30, 40, 200): (29.3, 53.5, -81.3),  # blue
        (128, 128, 128): (53.6, 0.0, 0.0),  # mid grey
    }
    for rgb, expected in cases.items():
        got = rgb_to_lab(torch.tensor(rgb, dtype=torch.float32).view(1, 1, 3) / 255.0)[0, 0]
        for axis, (value, want) in enumerate(zip(got.tolist(), expected)):
            assert abs(value - want) < 0.5, f"{rgb} axis {axis}: {value:.1f} != {want}"


def should_rank_dark_and_light_sweaters_by_lightness():
    dark = lab_torso_feature(_solid((18, 18, 24)))
    light = lab_torso_feature(_solid((240, 240, 235)))
    assert float(dark[0]) < float(light[0])


def should_separate_two_team_colours_in_lab_space():
    red = lab_torso_feature(_solid((200, 30, 30)))
    blue = lab_torso_feature(_solid((30, 40, 200)))
    # Red and blue sit at similar lightness, which is exactly the case a
    # lightness-only split cannot resolve -- the a/b axes have to carry it.
    assert abs(float(red[0]) - float(blue[0])) < 30.0
    assert torch.linalg.vector_norm(red[1:] - blue[1:]) > 60.0


def should_accept_channels_first_and_uint8_and_float_crops():
    hwc_u8 = _solid((200, 30, 30))
    chw_u8 = hwc_u8.permute(2, 0, 1)
    hwc_f = hwc_u8.to(torch.float32) / 255.0

    a = lab_torso_feature(hwc_u8)
    b = lab_torso_feature(chw_u8)
    c = lab_torso_feature(hwc_f)
    assert torch.allclose(a, b, atol=1e-4)
    assert torch.allclose(a, c, atol=1e-4)


def should_refuse_a_torso_band_that_left_the_frame():
    """A box mostly off-canvas has no torso to sample; say so, do not substitute.

    The old behaviour returned the whole detection box, which `roi_align`
    zero-fills: on a uniform frame the crop read near-black, joined the dark
    cluster and dragged its centroid. The caller cannot tell a body from a
    torso, so the band has to decline.
    """
    import torch

    from hmlib.bbox.torso import bbox_upper_torso_fallback

    inside = torch.tensor([100, 100, 180, 300])
    assert bbox_upper_torso_fallback(inside, 1920, 1080) is not None

    # Off the right edge, and off the left edge.
    assert bbox_upper_torso_fallback(torch.tensor([1900, 300, 1990, 500]), 1920, 1080) is None
    assert bbox_upper_torso_fallback(torch.tensor([-90, 300, 10, 500]), 1920, 1080) is None


def should_treat_crop_batches_as_0_255():
    """`batched_lab_mean` is the production entry point and assumes 0-255.

    The whole CIELAB path hangs on that convention -- `_cut_crops` documents
    it, and sniffing the range instead would cost a GPU sync per frame -- but
    only the per-crop helpers were covered, so dropping the `/255` scaling
    went unnoticed.
    """
    import torch

    from hmlib.team.features import batched_lab_mean, rgb_to_lab

    mid = torch.full((1, 3, 8, 8), 128.0)
    got = batched_lab_mean(mid)
    want = rgb_to_lab(torch.full((1, 3), 128.0 / 255.0))
    assert torch.allclose(got, want, atol=1e-3), f"{got} != {want}"
    # Mid grey is L~53; without the scaling it saturates to 100.
    assert 50.0 < float(got[0, 0]) < 56.0


def should_report_lightness_from_the_l_axis():
    """The backend's `lightness` must be L, not a chroma axis.

    The dark/light ranking is the only thing that turns two clusters into
    HOME and AWAY, and it reads this field. A chroma axis would rank by hue.
    """
    import torch

    from hmlib.team.backends import LabBackend

    dark = torch.full((1, 3, 8, 8), 20.0)
    light = torch.full((1, 3, 8, 8), 230.0)
    backend = LabBackend()
    assert float(backend.extract(dark).lightness[0]) < float(backend.extract(light).lightness[0])
    # And it is on the CIELAB scale, not raw bytes.
    assert 0.0 <= float(backend.extract(light).lightness[0]) <= 100.0


def should_keep_the_measured_torso_band():
    """The band constants carry a measurement table; pin them.

    Widening toward the jersey band cost 8 points of team accuracy. Nothing
    held the numbers, so loosening them to the whole box was invisible.
    """
    import torch

    from hmlib.bbox.torso import bbox_upper_torso_fallback

    box = torch.tensor([0, 0, 100, 200])
    x1, y1, x2, y2 = bbox_upper_torso_fallback(box, 1920, 1080)
    assert (x1, y1, x2, y2) == (32, 40, 68, 90), "the measured 32-68% / 20-45% band moved"
    # Strictly inside the detection box on every side.
    assert x1 > 0 and y1 > 0 and x2 < 100 and y2 < 200
