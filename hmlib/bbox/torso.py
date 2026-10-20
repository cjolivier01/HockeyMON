"""Pose-guided torso ROI derivation.

Both the jersey-number pipeline and team affiliation want the same crop: the
quadrilateral spanned by the shoulders and hips. Keeping one implementation here
means the two cannot drift apart.

Torso-only matters more in hockey than it looks. Helmets, gloves, pants and socks
are routinely a different colour from the sweater, and the ice is a large bright
background that leaks into any crop taken from the whole detection box -- so
appearance statistics over a full bbox say more about the rink than the team.
"""

from __future__ import annotations

import torch

#: Pixels of slack around the shoulder/hip hull. Matches the jersey pipeline.
PADDING = 5

#: Minimum per-keypoint confidence before a torso ROI is trusted.
CONFIDENCE_THRESHOLD = 0.4

#: COCO keypoint indices: right shoulder, left shoulder, left hip, right hip.
_TORSO_KEYPOINTS = (6, 5, 11, 12)


def torso_roi_from_pose(
    kpts: torch.Tensor,
    kps: torch.Tensor | None,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Torso ROI from COCO keypoints, or ``None`` when the pose is not usable.

    ``kpts`` is ``(K, 2)`` and ``kps`` the matching ``(K,)`` scores. Returns an
    ``(x1, y1, x2, y2)`` box clamped to the image. ``None`` means a caller should
    fall back to :func:`bbox_torso_fallback` rather than trust a degenerate crop.

    A keypoint set too short to contain the hips is a pose this function cannot
    answer for, not an error -- some estimators emit reduced skeletons -- so it
    returns ``None`` alongside the low-confidence case.
    """
    if kpts.ndim != 2 or kpts.shape[0] <= max(_TORSO_KEYPOINTS):
        return None

    px = kpts.clone().detach()
    scores = None
    if kps is not None and torch.is_tensor(kps) and kps.shape[0] == kpts.shape[0]:
        scores = kps

    pts: list[tuple[float, float, float]] = []
    for i in _TORSO_KEYPOINTS:
        x = float(px[i, 0].item())
        y = float(px[i, 1].item())
        s = float(scores[i].item()) if scores is not None else 1.0
        pts.append((x, y, s))

    if any(p[2] < CONFIDENCE_THRESHOLD for p in pts):
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = int(max(0, min(xs) - PADDING))
    x_max = int(min(img_w - 1, max(xs) + PADDING))
    y_min = int(max(0, min(ys) - PADDING))
    y_max = int(min(img_h - 1, max(ys)))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def _band(
    box: torch.Tensor,
    img_w: int,
    img_h: int,
    top: float,
    bottom: float,
    side: float,
) -> tuple[int, int, int, int] | None:
    """Fractional sub-box of ``box`` clamped to the image, or ``None``.

    ``None`` when the band does not survive clamping -- a skater far enough
    off the edge that the torso is not in frame. The old fallback returned the
    whole detection box, which is the one thing this must not do: the caller
    cannot tell it got a body instead of a torso, and
    `bbox_upper_torso_fallback`'s own docstring calls a whole-body crop
    actively harmful here. Measured on a 1920-wide frame, a box at
    ``(1900, 300, 1990, 500)`` came back unchanged at 77% outside the image,
    so `roi_align` zero-filled it and a uniform-120 frame produced a crop
    reading near-black -- it then joined the dark cluster and dragged that
    centroid toward it.
    """
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    w = x2 - x1
    h = y2 - y1
    new_x1 = max(0, x1 + int(side * w))
    new_x2 = min(img_w - 1, x2 - int(side * w))
    new_y1 = max(0, y1 + int(top * h))
    new_y2 = min(img_h - 1, y1 + int(bottom * h))
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None
    return (new_x1, new_y1, new_x2, new_y2)


def bbox_torso_fallback(
    box: torch.Tensor,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Jersey-number ROI from a detection box when no pose is available.

    Keeps the vertical band [25%, 95%] and trims 20% from each side. That band
    is deliberately tall because a number can sit anywhere from the upper back
    to the pants, so it is the right ROI for OCR and the wrong one for colour --
    see :func:`bbox_upper_torso_fallback`.

    ``None`` when the band does not survive clamping to the frame.
    """
    return _band(box, img_w, img_h, top=0.25, bottom=0.95, side=0.2)


def bbox_upper_torso_fallback(
    box: torch.Tensor,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """Sweater-only ROI from a detection box, for appearance and team colour.

    Keeps [20%, 45%] vertically and trims 32% per side, which on a standing
    skater is roughly the chest.

    Measured by sweeping the band over 38 hand-labelled crops from
    tv-14-2-p1 (navy versus white squads), scoring the two-means split:

        band                       lab     prtreid
        [0.15, 0.50] side 0.25   76.3%      78.9%
        [0.20, 0.45] side 0.32   76.3%      86.8%   <- this
        [0.22, 0.40] side 0.35   76.3%      86.8%
        [0.10, 0.60] side 0.18   68.4%      68.4%

    Tightening past this plateaus and widening hurts both. The mean-Lab
    backend does not benefit at all -- it sits at 76% whatever the crop -- so
    this band is chosen for the embedding backend, which is the one that can
    use the extra cleanliness.

    The wider jersey band is actively harmful here. Measured over 2293 crops
    from a real clip it spans sweater, pants, skates and the ice between the
    legs, and mean torso chroma came out at |a|,|b| ~= 2-3 on a gold-versus-black
    matchup -- grey, because the three regions average out, which leaves the
    colour feature carrying almost no team signal.

    ``None`` when the band does not survive clamping to the frame.
    """
    return _band(box, img_w, img_h, top=0.20, bottom=0.45, side=0.32)
