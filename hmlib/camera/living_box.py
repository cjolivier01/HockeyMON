"""Python-side helpers for visualizing and working with LivingBox instances."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from hmlib.bbox.box_functions import center, make_box_at_center, move_box_to_center, scale_box
from hmlib.tracking_utils import visualization as vis
from hockeymon.core import BBox, LivingBox, WHDims


def to_bbox(tensor: torch.Tensor, is_cpp: bool) -> BBox:
    """Convert a tensor `[x1, y1, x2, y2]` into a BBox (if using C++ boxes)."""
    if not is_cpp:
        return tensor
    if isinstance(tensor, BBox):
        return tensor
    bbox = BBox()
    bbox.left = tensor[0].item()
    bbox.top = tensor[1].item()
    bbox.right = tensor[2].item()
    bbox.bottom = tensor[3].item()
    return bbox


def from_bbox(bbox: BBox, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if isinstance(bbox, torch.Tensor):
        return bbox
    return torch.tensor(
        [bbox.left, bbox.top, bbox.right, bbox.bottom], dtype=torch.float, device=device
    )


class PyLivingBox(LivingBox):
    """Subclass of C++ `LivingBox` with extra drawing helpers for debugging."""

    def __init__(self, *args, color: Tuple[int, int, int], thickness: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._color = color
        self._thickness = thickness

    @staticmethod
    def _draw_resizing_state(
        live_box: LivingBox,
        img: torch.Tensor,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ):
        rconfig = live_box.resizing_config()
        rstate = live_box.resizing_state()
        lconfig = live_box.living_config()
        lstate = live_box.living_state()

        if rconfig.sticky_sizing:
            assert following_box is not None  # why?
            my_bbox = live_box.bounding_box()
            my_bbox_t: torch.Tensor = from_bbox(my_bbox)
            center_tensor = center(my_bbox_t)
            my_width = my_bbox.width()
            my_height = my_bbox.height()
            following_bbox = from_bbox(following_box.bounding_box())
            # dashed box representing the following box inscribed at our center

            if rstate.size_is_frozen:
                corner_box = scale_box(box=my_bbox_t, scale_width=0.98, scale_height=0.98)
                img = vis.draw_corner_boxes(
                    image=img, bbox=corner_box, color=(255, 255, 255), thickness=1
                )

            scaled_following_box = scale_box(
                following_bbox,
                scale_width=lconfig.scale_dest_width,
                scale_height=lconfig.scale_dest_height,
            )

            inscribed = move_box_to_center(scaled_following_box.clone(), center(my_bbox_t))
            img = vis.draw_dashed_rectangle(
                img,
                box=inscribed,
                color=(255, 255, 255) if not lstate.was_size_constrained else (0, 0, 255),
                thickness=2,
            )

            # Sizing thresholds
            if draw_thresholds:
                gs = live_box.get_grow_shrink_wh(my_bbox)
                grow_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width + gs.grow_width,
                    h=my_height + gs.grow_height,
                )
                shrink_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width - gs.shrink_width,
                    h=my_height - gs.shrink_height,
                )

                img = vis.draw_centered_lines(img, bbox=grow_box, thickness=4, color=(0, 255, 0))
                img = vis.draw_centered_lines(img, bbox=shrink_box, thickness=4, color=(0, 0, 255))
        return img

    @staticmethod
    def _draw_translation_state(
        live_box: LivingBox,
        img: torch.Tensor | np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ):
        tconfig = live_box.translation_config()
        tstate = live_box.translation_state()
        draw_box = from_bbox(live_box.bounding_box())
        img = vis.plot_rectangle(
            img,
            draw_box,
            color=color if not tstate.translation_is_frozen else (128, 128, 128),
            thickness=thickness,
            label=live_box.name(),
            text_scale=2,
        )
        # Small visual cue: if cancel-on-opposite triggered this frame, flash a thin cyan border
        try:
            if getattr(tstate, "canceled_stop_x", False) or getattr(
                tstate, "canceled_stop_y", False
            ):
                img = vis.plot_rectangle(
                    img,
                    draw_box,
                    color=(0, 255, 255),
                    thickness=2,
                    label=None,
                    text_scale=1,
                )
        except Exception:
            pass
        # Debug overlay: show active per-axis braking state for C++ boxes
        try:
            # Optional[IntValue] fields come through as None or int
            stop_x = getattr(tstate, "stop_delay_x", None)
            stop_y = getattr(tstate, "stop_delay_y", None)
            if (stop_x and int(stop_x) > 0) or (stop_y and int(stop_y) > 0):
                intbox = [int(i) for i in draw_box]
                x1, y1 = intbox[0], intbox[1]
                text_height = 100
                y_offset = text_height * 5
                if stop_x and int(stop_x) > 0:
                    cnt_x = int(getattr(tstate, "stop_delay_x_counter", 0))
                    img = vis.plot_text(
                        img,
                        f"BrakeX {cnt_x}/{int(stop_x)}",
                        (x1 + 5, y1 + y_offset),
                        cv2.FONT_HERSHEY_PLAIN,
                        5,
                        (0, 255, 255),
                        thickness=2,
                    )
                    y_offset += text_height
                if stop_y and int(stop_y) > 0:
                    cnt_y = int(getattr(tstate, "stop_delay_y_counter", 0))
                    img = vis.plot_text(
                        img,
                        f"BrakeY {cnt_y}/{int(stop_y)}",
                        (x1 + 5, y1 + y_offset),
                        cv2.FONT_HERSHEY_PLAIN,
                        5,
                        (0, 255, 255),
                        thickness=2,
                    )
                    y_offset += text_height
                # Show hysteresis threshold and cooldown counters
                hyst = int(getattr(tconfig, "cancel_stop_hysteresis_frames", 0))
                if hyst > 0:
                    img = vis.plot_text(
                        img,
                        f"Hyst {hyst}",
                        (x1 + 5, y1 + y_offset),
                        cv2.FONT_HERSHEY_PLAIN,
                        5,
                        (128, 255, 128),
                        thickness=2,
                    )
                    y_offset += text_height
                cd_x = int(getattr(tstate, "cooldown_x_counter", 0))
                cd_y = int(getattr(tstate, "cooldown_y_counter", 0))
                if cd_x > 0 or cd_y > 0:
                    img = vis.plot_text(
                        img,
                        f"Cd X{cd_x} Y{cd_y}",
                        (x1 + 5, y1 + y_offset),
                        cv2.FONT_HERSHEY_PLAIN,
                        5,
                        (255, 200, 0),
                        thickness=2,
                    )
        except Exception:
            pass

        # img = vis.draw_arrows(img, bbox=draw_box, horizontal=True, vertical=True)
        if draw_thresholds and tconfig.sticky_translation:
            sticky, unsticky = live_box.get_sticky_translation_sizes()
            bbox_t = from_bbox(live_box.bounding_box())
            center_tensor = center(bbox_t)
            my_center = [int(i) for i in center_tensor]
            img = vis.plot_circle(
                img,
                my_center,
                radius=int(sticky),
                color=(255, 0, 0),
                thickness=3,
            )
            img = vis.plot_circle(
                img,
                my_center,
                radius=int(unsticky),
                color=(255, 0, 255),
                thickness=3,
            )
            if following_box is not None:
                following_bbox = from_bbox(following_box.bounding_box())
                following_bbox_center = center(following_bbox)

                # Line from center of this box to the center of the box that it is following,
                # with little circle nubs at each end.
                co = [int(i) for i in following_bbox_center]
                img = vis.plot_circle(
                    img,
                    my_center,
                    radius=5,
                    color=(255, 255, 0),
                    thickness=cv2.FILLED,
                )
                img = vis.plot_circle(
                    img,
                    co,
                    radius=5,
                    color=(0, 255, 128),
                    thickness=cv2.FILLED,
                )
                img = vis.plot_line(img, my_center, co, color=(255, 255, 0), thickness=10)
                # X
                img = vis.plot_line(
                    img,
                    my_center,
                    [co[0], my_center[1]],
                    color=(255, 255, 0),
                    thickness=3,
                )
                # Y
                img = vis.plot_line(
                    img,
                    my_center,
                    [my_center[0], co[1]],
                    color=(255, 255, 0),
                    thickness=3,
                )
        return img

    @staticmethod
    def draw_impl(
        live_box: LivingBox,
        img: torch.Tensor,
        color: Tuple[int, int, int],
        thickness: int,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ) -> torch.Tensor:
        img = PyLivingBox._draw_resizing_state(
            live_box=live_box, img=img, draw_thresholds=draw_thresholds, following_box=following_box
        )
        img = PyLivingBox._draw_translation_state(
            live_box=live_box,
            img=img,
            draw_thresholds=draw_thresholds,
            following_box=following_box,
            color=color,
            thickness=thickness,
        )
        return img

    def draw(
        self,
        img: torch.Tensor,
        draw_thresholds: bool = False,
        following_box: Optional[LivingBox] = None,
    ) -> torch.Tensor:
        return PyLivingBox.draw_impl(
            live_box=self,
            img=img,
            draw_thresholds=draw_thresholds,
            following_box=following_box,
            color=self._color,
            thickness=self._thickness,
        )

    def get_size_scale(self) -> Tuple[float, float]:
        size_scale: WHDims = super().get_size_scale()
        return size_scale.width, size_scale.height
