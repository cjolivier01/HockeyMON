"""Anchored text label with a translucent backing box.

Extracted from the jersey overlay so jersey numbers and team affiliation render
identically -- two differently-styled labels stacked over the same player read
as two unrelated systems.

Handles both the torch (channels-first or channels-last) and numpy/cv2 image
paths, because the overlay trunks work in torch while some callers still hand
over cv2 frames.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

import hmlib.tracking_utils.visualization as vis
import hmlib.vis.pt_text as ptt
import hmlib.vis.pt_visualization as ptv
from hmlib.utils.image import is_channels_last, make_channels_first, make_channels_last

ImageLike = torch.Tensor | np.ndarray


def draw_label_box(
    image: ImageLike,
    text: str,
    center_x: int,
    bottom_y: int,
    font_scale: float,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    bg_alpha: int,
) -> ImageLike:
    """Draw ``text`` centred at ``center_x`` with its baseline at ``bottom_y``.

    ``bg_alpha`` is 0-255 over the filled backing box; the border is drawn at a
    fixed higher opacity so the label stays legible against the ice.
    """
    if isinstance(image, torch.Tensor):
        was_channels_last = is_channels_last(image)
        if was_channels_last:
            image = make_channels_first(image)
        font_size = max(1, int(font_scale * 2))
        text_w, text_h = ptt.measure_text(text=text, font_size=font_size)
        pad_x = max(3, int(font_scale * 6))
        pad_y = max(2, int(font_scale * 4))
        x1 = int(center_x - (text_w // 2) - pad_x)
        x2 = int(center_x + (text_w // 2) + pad_x)
        y2 = int(bottom_y + pad_y)
        y1 = int(bottom_y - text_h - pad_y)
        image = ptv.draw_box(
            image=image, tlbr=(x1, y1, x2, y2), color=bg, filled=True, alpha=bg_alpha
        )
        image = ptv.draw_box(
            image=image,
            tlbr=(x1, y1, x2, y2),
            color=fg,
            filled=False,
            alpha=220,
            thickness=max(1, int(font_scale)),
        )
        image = ptt.draw_text(
            image=image,
            x=int(center_x - text_w // 2),
            y=int(bottom_y),
            text=text,
            font_size=font_size,
            color=fg,
            position_is_text_bottom=True,
        )
        if was_channels_last:
            image = make_channels_last(image)
        return image

    # numpy/cv2 path
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    pad_x = max(3, int(font_scale * 6))
    pad_y = max(2, int(font_scale * 4))
    x1 = int(center_x - (tw // 2) - pad_x)
    x2 = int(center_x + (tw // 2) + pad_x)
    y2 = int(bottom_y + pad_y)
    y1 = int(bottom_y - th - baseline - pad_y)
    img = vis.to_cv2(image)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, thickness=-1)
    alpha = float(max(0, min(255, bg_alpha))) / 255.0
    img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)
    cv2.rectangle(img, (x1, y1), (x2, y2), fg, thickness=max(1, int(font_scale)))
    cv2.putText(
        img,
        text,
        (int(center_x - tw // 2), int(bottom_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        fg,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return img
