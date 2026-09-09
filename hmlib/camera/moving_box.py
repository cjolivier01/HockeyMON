"""Camera box motion primitives used for smoothing pan/zoom behavior."""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.bbox.box_functions import (
    aspect_ratio,
    center,
    check_for_box_overshoot,
    clamp_box,
    height,
    make_box_at_center,
    move_box_to_center,
    scale_box,
    shift_box_to_edge,
    width,
)
from hmlib.tracking_utils import visualization as vis
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution


class BasicBox(torch.nn.Module):
    def __init__(self, bbox: torch.Tensor, device: Union[torch.device, str, None] = None):
        super(BasicBox, self).__init__()
        if device and isinstance(device, str):
            device = torch.device(device)
        self.device = bbox.device if device is None else device
        self._zero_float_tensor = torch.tensor(0, dtype=torch.float, device=self.device)
        self._zero_int_tensor = torch.tensor(0, dtype=torch.int64, device=self.device)
        self._one_float_tensor = torch.tensor(1, dtype=torch.int64, device=self.device)
        self._true = torch.tensor(True, dtype=torch.bool, device=self.device)
        self._false = torch.tensor(False, dtype=torch.bool, device=self.device)

    def set_bbox(self, bbox: torch.Tensor):
        self._bbox = bbox

    @property
    def one(self):
        return self._one_float_tensor.clone()

    @property
    def _zero(self):
        return self._zero_float_tensor.clone()

    @property
    def _zero_int(self):
        return self._zero_int_tensor.clone()

    def bounding_box(self):
        return self._bbox.clone()


def _as_scalar_float_tensor(
    val: Union[torch.Tensor, int, float], device: torch.device
) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val
    return torch.tensor(float(val), dtype=torch.float, device=device)


class ResizingBox(BasicBox):
    # TODO: move resizing stuff here
    def __init__(
        self,
        bbox: torch.Tensor,
        max_speed_w: Union[torch.Tensor, int, float],
        max_speed_h: Union[torch.Tensor, int, float],
        max_accel_w: Union[torch.Tensor, int, float],
        max_accel_h: Union[torch.Tensor, int, float],
        min_width: Union[torch.Tensor, int, float],
        min_height: Union[torch.Tensor, int, float],
        max_width: Union[torch.Tensor, int, float],
        max_height: Union[torch.Tensor, int, float],
        stop_on_dir_change: bool,
        stop_on_dir_change_delay: int = 0,
        resize_stop_on_dir_change: bool = False,
        resize_stop_on_dir_change_delay: int = 0,
        resize_cancel_stop_on_opposite_dir: bool = False,
        resize_stop_cancel_hysteresis_frames: int = 0,
        resize_stop_delay_cooldown_frames: int = 0,
        resize_time_to_dest_speed_limit_frames: int = 10,
        resize_time_to_dest_stop_speed_threshold: float = 0.0,
        sticky_sizing: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(ResizingBox, self).__init__(bbox=bbox, device=device)
        self._stop_on_dir_change = stop_on_dir_change
        self._resize_stop_on_dir_change = bool(resize_stop_on_dir_change)

        self._sticky_sizing = sticky_sizing

        self._max_speed_w = max_speed_w
        self._max_speed_h = max_speed_h
        self._max_accel_w = max_accel_w
        self._max_accel_h = max_accel_h

        self._min_width = _as_scalar_float_tensor(min_width, device=device)
        self._min_height = _as_scalar_float_tensor(min_height, device=device)
        self._max_width = _as_scalar_float_tensor(max_width, device=device)
        self._max_height = _as_scalar_float_tensor(max_height, device=device)
        self._size_constrained = False

        #
        # Sticky sizing thresholds
        #
        # Threshold to grow width (ratio of bbox)
        self._size_ratio_thresh_grow_dw = 0.05
        # Threshold to grow height (ratio of bbox)
        self._size_ratio_thresh_grow_dh = 0.1
        # Threshold to shrink width (ratio of bbox)
        self._size_ratio_thresh_shrink_dw = 0.08
        # Threshold to shrink height (ratio of bbox)
        self._size_ratio_thresh_shrink_dh = 0.1

        self._size_is_frozen = False
        # One-frame visual cue flags when cancel-on-opposite triggers
        self._cancel_stop_x_flash = False
        self._cancel_stop_y_flash = False
        self._resize_cancel_stop_w_flash = False
        self._resize_cancel_stop_h_flash = False

        # Resizing stop-on-direction-change braking config
        self._resize_stop_on_dir_change_delay = torch.tensor(
            int(resize_stop_on_dir_change_delay), dtype=torch.int64, device=self.device
        )
        self._resize_cancel_stop_on_opposite_dir = bool(resize_cancel_stop_on_opposite_dir)
        self._resize_cancel_hysteresis_frames = torch.tensor(
            int(resize_stop_cancel_hysteresis_frames), dtype=torch.int64, device=self.device
        )
        self._resize_stop_delay_cooldown_frames = torch.tensor(
            int(resize_stop_delay_cooldown_frames), dtype=torch.int64, device=self.device
        )
        self._resize_ttg_limit_frames = torch.tensor(
            int(resize_time_to_dest_speed_limit_frames), dtype=torch.int64, device=self.device
        )
        self._resize_ttg_stop_speed_threshold = float(resize_time_to_dest_stop_speed_threshold)

        self._resize_stop_delay_w = self._zero_int.clone()
        self._resize_stop_delay_h = self._zero_int.clone()
        self._resize_stop_delay_w_counter = self._zero_int.clone()
        self._resize_stop_delay_h_counter = self._zero_int.clone()
        self._resize_stop_decel_w = self._zero.clone()
        self._resize_stop_decel_h = self._zero.clone()
        self._resize_stop_trigger_dir_w = self._zero.clone()
        self._resize_stop_trigger_dir_h = self._zero.clone()
        self._resize_cancel_opp_w_count = self._zero_int.clone()
        self._resize_cancel_opp_h_count = self._zero_int.clone()
        self._resize_cooldown_w_counter = self._zero_int.clone()
        self._resize_cooldown_h_counter = self._zero_int.clone()

    def draw(
        self,
        img: np.array,
        draw_thresholds: bool = True,
        following_box: Optional[BasicBox] = None,
    ):
        if self._sticky_sizing:
            assert following_box is not None  # why?
            my_bbox = self.bounding_box()
            center_tensor = center(my_bbox)
            my_width = width(my_bbox)
            my_height = height(my_bbox)
            following_bbox = following_box.bounding_box()
            # dashed box representing the following box inscribed at our center

            if self._size_is_frozen:
                corner_box = scale_box(box=my_bbox.clone(), scale_width=0.98, scale_height=0.98)
                img = vis.draw_corner_boxes(
                    image=img, bbox=corner_box, color=(255, 255, 255), thickness=1
                )

            if hasattr(self, "_scale_width"):
                scaled_following_box = scale_box(
                    following_bbox,
                    scale_width=self._scale_width,
                    scale_height=self._scale_height,
                )

            inscribed = move_box_to_center(scaled_following_box.clone(), center(my_bbox))
            img = vis.draw_dashed_rectangle(
                img,
                box=inscribed,
                color=(255, 255, 255) if not self._size_constrained else (0, 0, 255),
                thickness=2,
            )

            # Sizing thresholds
            if draw_thresholds:
                (
                    grow_width,
                    grow_height,
                    shrink_width,
                    shrink_height,
                ) = self._get_grow_wh_and_shrink_wh(bbox=my_bbox)
                grow_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width + grow_width,
                    h=my_height + grow_height,
                )
                shrink_box = make_box_at_center(
                    center_point=center_tensor,
                    w=my_width - shrink_width,
                    h=my_height - shrink_height,
                )

                img = vis.draw_centered_lines(img, bbox=grow_box, thickness=4, color=(0, 255, 0))
                img = vis.draw_centered_lines(img, bbox=shrink_box, thickness=4, color=(0, 0, 255))
        return img

    def set_resizing_shrink_thresholds(self, width_ratio: float, height_ratio: float) -> None:
        """Tune sticky shrink thresholds while retaining motion/braking state."""
        self._size_ratio_thresh_shrink_dw = float(width_ratio)
        self._size_ratio_thresh_shrink_dh = float(height_ratio)

    def _get_grow_wh_and_shrink_wh(self, bbox: torch.Tensor):
        my_width = width(bbox)
        my_height = height(bbox)
        grow_width = my_width * self._size_ratio_thresh_grow_dw
        grow_height = my_height * self._size_ratio_thresh_grow_dh
        shrink_width = my_width * self._size_ratio_thresh_shrink_dw
        shrink_height = my_height * self._size_ratio_thresh_shrink_dh
        return grow_width, grow_height, shrink_width, shrink_height

    def _clamp_resizing(self):
        self._current_speed_w = torch.clamp(
            self._current_speed_w, min=-self._max_speed_w, max=self._max_speed_w
        )
        self._current_speed_h = torch.clamp(
            self._current_speed_h, min=-self._max_speed_h, max=self._max_speed_h
        )

    def _update_resize_stop_delays(self):
        if self._resize_stop_delay_w != self._zero_int:
            self._resize_stop_delay_w_counter += 1
            if self._resize_stop_delay_w_counter >= self._resize_stop_delay_w:
                self._resize_stop_delay_w = self._zero_int.clone()
                self._resize_stop_delay_w_counter = self._zero_int.clone()
                self._resize_stop_decel_w = self._zero.clone()
                self._current_speed_w = self._zero.clone()
                if self._resize_stop_delay_cooldown_frames > self._zero_int:
                    self._resize_cooldown_w_counter = (
                        self._resize_stop_delay_cooldown_frames.clone()
                    )

        if self._resize_stop_delay_h != self._zero_int:
            self._resize_stop_delay_h_counter += 1
            if self._resize_stop_delay_h_counter >= self._resize_stop_delay_h:
                self._resize_stop_delay_h = self._zero_int.clone()
                self._resize_stop_delay_h_counter = self._zero_int.clone()
                self._resize_stop_decel_h = self._zero.clone()
                self._current_speed_h = self._zero.clone()
                if self._resize_stop_delay_cooldown_frames > self._zero_int:
                    self._resize_cooldown_h_counter = (
                        self._resize_stop_delay_cooldown_frames.clone()
                    )

        if self._resize_cooldown_w_counter > self._zero_int:
            self._resize_cooldown_w_counter -= 1
        if self._resize_cooldown_h_counter > self._zero_int:
            self._resize_cooldown_h_counter -= 1

    def _adjust_size(
        self,
        accel_w: Optional[torch.Tensor] = None,
        accel_h: Optional[torch.Tensor] = None,
        use_constraints: bool = True,
    ):
        if self._size_is_frozen:
            return

        if use_constraints:
            # Growing is allowed at a higher rate than shrinking
            resize_larger_scale = 2.0
            max_accel_wh = torch.tensor([self._max_accel_w, self._max_accel_h])
            max_accel_wh = torch.where(
                torch.tensor([accel_w, accel_h]) > 0,
                max_accel_wh * resize_larger_scale,
                max_accel_wh,
            )

            if accel_w is not None:
                accel_w = torch.clamp(accel_w, min=-max_accel_wh[0], max=max_accel_wh[0])
            if accel_h is not None:
                accel_h = torch.clamp(accel_h, min=-max_accel_wh[1], max=max_accel_wh[1])
        if accel_w is not None:
            self._current_speed_w += accel_w

        if accel_h is not None:
            self._current_speed_h += accel_h

        if use_constraints:
            self._clamp_resizing()

    def set_destination(self, dest_box: torch.Tensor):
        self._set_destination_size(
            dest_width=width(dest_box),
            dest_height=height(dest_box),
        )

    def _set_destination_size(
        self,
        dest_width: torch.Tensor,
        dest_height: torch.Tensor,
    ):
        bbox = self.bounding_box()
        current_w = width(bbox)
        current_h = height(bbox)

        dw = dest_width - current_w
        dh = dest_height - current_h

        #
        # BEGIN size threshhold
        #
        if self._sticky_sizing:

            grow_width, grow_height, shrink_width, shrink_height = self._get_grow_wh_and_shrink_wh(
                bbox=bbox
            )

            dw_thresh = torch.logical_and(dw < 0, dw < -shrink_width)
            want_bigger_w = torch.logical_and(dw > 0, dw > grow_width)
            dw_thresh = torch.logical_or(dw_thresh, want_bigger_w)
            dw = torch.where(dw_thresh, dw, 0)

            dh_thresh = torch.logical_and(dh < 0, dh < -shrink_height)
            want_bigger_h = torch.logical_and(dh > 0, dh > grow_height)
            dh_thresh = torch.logical_or(dh_thresh, want_bigger_h)
            dh = torch.where(dh_thresh, dh, 0)

            self._size_is_frozen = torch.logical_and(
                torch.logical_not(dw_thresh), torch.logical_not(dh_thresh)
            )
            both_thresh = torch.logical_and(dw_thresh, dh_thresh)

            # prioritize width thresh
            if True:
                both_thresh = torch.logical_or(dw_thresh, both_thresh)

            any_thresh = torch.logical_or(dw_thresh, dh_thresh)
            want_bigger = torch.logical_and(
                torch.logical_or(want_bigger_w, want_bigger_h), any_thresh
            )
            self._size_is_frozen = torch.logical_or(
                self._size_is_frozen,
                torch.logical_not(torch.logical_or(both_thresh, want_bigger)),
            )
        #
        # END size threshhold
        #

        assert self._zero.item() == 0

        # Reset one-frame cancel flash indicators for resizing
        self._resize_cancel_stop_w_flash = False
        self._resize_cancel_stop_h_flash = False

        accel_w = dw
        accel_h = dh

        # Trigger resizing braking on direction changes
        if (
            self._resize_stop_delay_w == self._zero_int
            and self._resize_cooldown_w_counter == self._zero_int
            and different_directions(dw, self._current_speed_w)
        ):
            moving_enough_w = torch.abs(self._current_speed_w) >= (self._max_speed_w / 6)
            if self._resize_stop_on_dir_change_delay > self._zero_int and moving_enough_w:
                self._resize_stop_delay_w = self._resize_stop_on_dir_change_delay.clone()
                self._resize_stop_delay_w_counter = self._zero_int.clone()
                self._resize_stop_decel_w = -self._current_speed_w / self._resize_stop_delay_w.to(
                    self._current_speed_w.dtype
                )
                self._resize_stop_trigger_dir_w = torch.sign(dw).to(self._current_speed_w.dtype)
            else:
                self._current_speed_w = torch.where(
                    torch.abs(self._current_speed_w) < self._max_speed_w / 6,
                    0,
                    self._current_speed_w / 2,
                )
                if self._resize_stop_on_dir_change:
                    accel_w = dw * 0.25

        if (
            self._resize_stop_delay_h == self._zero_int
            and self._resize_cooldown_h_counter == self._zero_int
            and different_directions(dh, self._current_speed_h)
        ):
            moving_enough_h = torch.abs(self._current_speed_h) >= (self._max_speed_h / 6)
            if self._resize_stop_on_dir_change_delay > self._zero_int and moving_enough_h:
                self._resize_stop_delay_h = self._resize_stop_on_dir_change_delay.clone()
                self._resize_stop_delay_h_counter = self._zero_int.clone()
                self._resize_stop_decel_h = -self._current_speed_h / self._resize_stop_delay_h.to(
                    self._current_speed_h.dtype
                )
                self._resize_stop_trigger_dir_h = torch.sign(dh).to(self._current_speed_h.dtype)
            else:
                self._current_speed_h = torch.where(
                    torch.abs(self._current_speed_h) < self._max_speed_h / 6,
                    0,
                    self._current_speed_h / 2,
                )
                if self._resize_stop_on_dir_change:
                    accel_h = dh * 0.25

        # If braking is active on an axis, ignore input accel for that axis
        if self._resize_stop_delay_w != self._zero_int:
            if (
                self._resize_cancel_stop_on_opposite_dir
                and torch.sign(dw) != 0
                and (torch.sign(dw) == -self._resize_stop_trigger_dir_w)
            ):
                if self._resize_cancel_hysteresis_frames > self._zero_int:
                    self._resize_cancel_opp_w_count += 1
                    if self._resize_cancel_opp_w_count >= self._resize_cancel_hysteresis_frames:
                        self._resize_stop_delay_w = self._zero_int.clone()
                        self._resize_stop_delay_w_counter = self._zero_int.clone()
                        self._resize_stop_decel_w = self._zero.clone()
                        self._resize_cancel_stop_w_flash = True
                        self._resize_cancel_opp_w_count = self._zero_int.clone()
                        self._resize_cooldown_w_counter = (
                            self._resize_stop_delay_cooldown_frames.clone()
                        )
                else:
                    self._resize_stop_delay_w = self._zero_int.clone()
                    self._resize_stop_delay_w_counter = self._zero_int.clone()
                    self._resize_stop_decel_w = self._zero.clone()
                    self._resize_cancel_stop_w_flash = True
                    self._resize_cooldown_w_counter = (
                        self._resize_stop_delay_cooldown_frames.clone()
                    )
            else:
                self._resize_cancel_opp_w_count = self._zero_int.clone()
                accel_w = self._resize_stop_decel_w

        if self._resize_stop_delay_h != self._zero_int:
            if (
                self._resize_cancel_stop_on_opposite_dir
                and torch.sign(dh) != 0
                and (torch.sign(dh) == -self._resize_stop_trigger_dir_h)
            ):
                if self._resize_cancel_hysteresis_frames > self._zero_int:
                    self._resize_cancel_opp_h_count += 1
                    if self._resize_cancel_opp_h_count >= self._resize_cancel_hysteresis_frames:
                        self._resize_stop_delay_h = self._zero_int.clone()
                        self._resize_stop_delay_h_counter = self._zero_int.clone()
                        self._resize_stop_decel_h = self._zero.clone()
                        self._resize_cancel_stop_h_flash = True
                        self._resize_cancel_opp_h_count = self._zero_int.clone()
                        self._resize_cooldown_h_counter = (
                            self._resize_stop_delay_cooldown_frames.clone()
                        )
                else:
                    self._resize_stop_delay_h = self._zero_int.clone()
                    self._resize_stop_delay_h_counter = self._zero_int.clone()
                    self._resize_stop_decel_h = self._zero.clone()
                    self._resize_cancel_stop_h_flash = True
                    self._resize_cooldown_h_counter = (
                        self._resize_stop_delay_cooldown_frames.clone()
                    )
            else:
                self._resize_cancel_opp_h_count = self._zero_int.clone()
                accel_h = self._resize_stop_decel_h

        prev_sw = self._current_speed_w.clone()
        prev_sh = self._current_speed_h.clone()

        self._adjust_size(accel_w=accel_w, accel_h=accel_h, use_constraints=True)

        # Time-to-destination speed limiting (per-axis)
        def _limit_speed_ttg(v, prev_v, dist, frames):
            if frames <= 0:
                return v
            sgn = torch.sign(dist)
            if sgn == 0:
                return v
            increasing = torch.abs(v) > torch.abs(prev_v)
            if torch.sign(v) == sgn and increasing:
                limit = torch.abs(dist) / frames.to(v.dtype)
                vmax = limit
                v = torch.clamp(v, min=-vmax, max=vmax)
                if (
                    self._resize_ttg_stop_speed_threshold > 0.0
                    and torch.abs(dist)
                    <= self._resize_ttg_stop_speed_threshold * frames.to(v.dtype)
                    and torch.abs(v) <= self._resize_ttg_stop_speed_threshold
                ):
                    v = self._zero.clone()
            return v

        self._current_speed_w = _limit_speed_ttg(
            self._current_speed_w, prev_sw, dw, self._resize_ttg_limit_frames
        )
        self._current_speed_h = _limit_speed_ttg(
            self._current_speed_h, prev_sh, dh, self._resize_ttg_limit_frames
        )

        # Clamp overshoot during braking to avoid reversing direction
        if self._resize_stop_delay_w != self._zero_int:
            if torch.abs(self._current_speed_w) < torch.abs(self._resize_stop_decel_w):
                self._current_speed_w = self._zero.clone()
        if self._resize_stop_delay_h != self._zero_int:
            if torch.abs(self._current_speed_h) < torch.abs(self._resize_stop_decel_h):
                self._current_speed_h = self._zero.clone()


# @HM.register_module()
class MovingBox(ResizingBox):
    def __init__(
        self,
        label: str,
        bbox: torch.Tensor,
        max_speed_x: torch.Tensor,
        max_speed_y: torch.Tensor,
        max_accel_x: torch.Tensor,
        max_accel_y: torch.Tensor,
        max_width: torch.Tensor,
        max_height: torch.Tensor,
        stop_on_dir_change: bool,
        stop_on_dir_change_delay: int = 0,
        cancel_stop_on_opposite_dir: bool = False,
        min_width: int = 10,
        min_height: int = 10,
        max_speed_w: Optional[torch.Tensor] = None,
        max_speed_h: Optional[torch.Tensor] = None,
        scale_width: Optional[torch.Tensor] = None,
        scale_height: Optional[torch.Tensor] = None,
        arena_box: Optional[torch.Tensor] = None,
        fixed_aspect_ratio: Optional[torch.Tensor] = None,
        sticky_translation: bool = False,
        sticky_size_ratio_to_frame_width: float = 10.0,
        sticky_translation_gaussian_mult: float = 5.0,
        unsticky_translation_size_ratio: float = 0.75,
        pan_smoothing_alpha: float = 0.18,
        post_nonstop_stop_delay: int = 0,
        cancel_hysteresis_frames: int = 0,
        stop_delay_cooldown_frames: int = 0,
        time_to_dest_speed_limit_frames: int = 10,
        time_to_dest_stop_speed_threshold: float = 0.0,
        resize_stop_on_dir_change: bool = False,
        resize_stop_on_dir_change_delay: int = 0,
        resize_cancel_stop_on_opposite_dir: bool = False,
        resize_stop_cancel_hysteresis_frames: int = 0,
        resize_stop_delay_cooldown_frames: int = 0,
        resize_time_to_dest_speed_limit_frames: int = 10,
        resize_time_to_dest_stop_speed_threshold: float = 0.0,
        sticky_sizing: bool = False,
        color: Tuple[int, int, int] = (255, 0, 0),
        frozen_color: Tuple[int, int, int] = (64, 64, 64),
        thickness: int = 2,
        device: Optional[str] = None,
        clamp_scaled_input_box: bool = True,  # EXPERIMENTAL
    ):
        super().__init__(
            bbox=bbox,
            device=device,
            max_speed_w=max_speed_w if max_speed_w is not None else max_speed_x / 1.8,
            max_speed_h=max_speed_h if max_speed_h is not None else max_speed_y / 1.8,
            max_accel_w=max_accel_x,
            max_accel_h=max_accel_y,
            stop_on_dir_change=stop_on_dir_change,
            resize_stop_on_dir_change=resize_stop_on_dir_change,
            resize_stop_on_dir_change_delay=resize_stop_on_dir_change_delay,
            resize_cancel_stop_on_opposite_dir=resize_cancel_stop_on_opposite_dir,
            resize_stop_cancel_hysteresis_frames=resize_stop_cancel_hysteresis_frames,
            resize_stop_delay_cooldown_frames=resize_stop_delay_cooldown_frames,
            resize_time_to_dest_speed_limit_frames=resize_time_to_dest_speed_limit_frames,
            resize_time_to_dest_stop_speed_threshold=resize_time_to_dest_stop_speed_threshold,
            sticky_sizing=sticky_sizing,
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
        )
        self._label = label
        self._color = color
        self._frozen_color = frozen_color
        self._thickness = thickness
        self._sticky_translation = sticky_translation
        self._sticky_translation_gaussian_mult = sticky_translation_gaussian_mult
        self._sticky_size_ratio_to_frame_width = sticky_size_ratio_to_frame_width
        self._unsticky_translation_size_ratio = unsticky_translation_size_ratio
        self._pan_smoothing_alpha = float(pan_smoothing_alpha)
        self._filtered_target_center: Optional[torch.Tensor] = None
        self._clamp_scaled_input_box = clamp_scaled_input_box
        self._cancel_stop_on_opposite_dir = bool(cancel_stop_on_opposite_dir)
        self._post_nonstop_stop_delay = torch.tensor(
            int(post_nonstop_stop_delay), dtype=torch.int64, device=self.device
        )
        self._cancel_hysteresis_frames = torch.tensor(
            int(cancel_hysteresis_frames), dtype=torch.int64, device=self.device
        )
        self._stop_delay_cooldown_frames = torch.tensor(
            int(stop_delay_cooldown_frames), dtype=torch.int64, device=self.device
        )
        self._ttg_limit_frames = torch.tensor(
            int(time_to_dest_speed_limit_frames), dtype=torch.int64, device=self.device
        )
        self._ttg_stop_speed_threshold = float(time_to_dest_stop_speed_threshold)
        self._cancel_opp_x_count = self._zero_int.clone()
        self._cancel_opp_y_count = self._zero_int.clone()
        self._cooldown_x_counter = self._zero_int.clone()
        self._cooldown_y_counter = self._zero_int.clone()

        self._line_thickness_tensor = torch.tensor(
            [2, 2, -1, -2], dtype=torch.float, device=self.device
        )
        self._inflate_arena_for_unsticky_edges = torch.tensor(
            [1, 1, -1, -1], dtype=torch.float, device=self.device
        )

        self._size_is_frozen = False

        self._scale_width = self._one_float_tensor if scale_width is None else scale_width
        self._scale_height = self._one_float_tensor if scale_height is None else scale_height
        self.set_bbox(
            make_box_at_center(
                center(bbox),
                w=width(bbox) * self._scale_width,
                h=height(bbox) * self._scale_height,
            )
        )

        self._arena_box = arena_box
        self._fixed_aspect_ratio = fixed_aspect_ratio
        self._current_speed_x = self._zero
        self._current_speed_y = self._zero
        self._current_speed_w = self._zero
        self._current_speed_h = self._zero
        self._nonstop_delay = self._zero
        self._nonstop_delay_counter = self._zero
        self._previous_area = width(bbox) * height(bbox)
        # Per-axis braking state for stop-on-direction-change
        self._stop_on_dir_change_delay = torch.tensor(
            int(stop_on_dir_change_delay), dtype=torch.int64, device=self.device
        )
        self._stop_on_dir_change_delay = torch.tensor(
            int(stop_on_dir_change_delay), dtype=torch.int64, device=self.device
        )
        self._stop_delay_x = self._zero_int.clone()
        self._stop_delay_y = self._zero_int.clone()
        self._stop_delay_x_counter = self._zero_int.clone()
        self._stop_delay_y_counter = self._zero_int.clone()
        self._stop_decel_x = self._zero.clone()
        self._stop_decel_y = self._zero.clone()
        self._stop_trigger_dir_x = self._zero.clone()
        self._stop_trigger_dir_y = self._zero.clone()

        if self._arena_box is not None:
            arena_width = width(self._arena_box)
            arena_width_int = (
                int(arena_width.detach().cpu().item())
                if isinstance(arena_width, torch.Tensor)
                else int(arena_width)
            )
            self._horizontal_image_gaussian_distribution = ImageHorizontalGaussianDistribution(
                arena_width_int, invert=True, device=self.device
            )
        else:
            self._horizontal_image_gaussian_distribution = None

        self._translation_is_frozen = False

        # Constraints
        self._max_speed_x = max_speed_x
        self._max_speed_y = max_speed_y
        self._max_accel_x = max_accel_x
        self._max_accel_y = max_accel_y

        if self._arena_box is not None:
            self._gaussian_x_clamp = self._arena_box[[0, 2]].to(dtype=torch.float32).clone()
        else:
            self._gaussian_x_clamp = None

        # Set up some values to help us to efficiently calculate the current zoom level
        # as well as other rules based upon zoom level (i.e. gaussian wrt center position)
        self._full_aspect_ratio_size = None
        if self._arena_box is not None and self._fixed_aspect_ratio:
            ah = width(self._arena_box)
            aw = height(self._arena_box)
            # Figure out which is the constraining edge
            ar_w = ah * self._fixed_aspect_ratio
            if ar_w <= aw:
                ww = ar_w
                hh = ah
            else:
                ww = aw
                hh = aw / self._fixed_aspect_ratio
                assert hh <= ah
            self._full_aspect_ratio_size = torch.tensor([ww, hh])
            # Gaussian clamp to center x of max aspect ratio box
            self._gaussian_x_clamp[0] += ww / 2
            self._gaussian_x_clamp[1] -= ww / 2

    def draw(
        self,
        img: np.array,
        draw_thresholds: bool = False,
        following_box: Optional[BasicBox] = None,
    ):
        super().draw(img=img, draw_thresholds=draw_thresholds, following_box=following_box)
        draw_box = self.bounding_box()
        img = vis.plot_rectangle(
            img,
            draw_box,
            color=self._color if not self._translation_is_frozen else (128, 128, 128),
            thickness=self._thickness,
            label=self._make_label(),
            text_scale=2,
        )
        # Small visual cue: if cancel-on-opposite triggered this frame, flash a thin cyan border
        if self._cancel_stop_x_flash or self._cancel_stop_y_flash:
            img = vis.plot_rectangle(
                img,
                draw_box,
                color=(0, 255, 255),
                thickness=2,
            )
        # Debug overlay: show active per-axis braking state
        if self._stop_delay_x != self._zero_int or self._stop_delay_y != self._zero_int:
            intbox = [int(i) for i in draw_box]
            x1, y1 = intbox[0], intbox[1]
            y_offset = 50
            if self._stop_delay_x != self._zero_int:
                text = f"BrakeX {int(self._stop_delay_x_counter.item())}/{int(self._stop_delay_x.item())}"
                img = vis.plot_text(
                    img,
                    text,
                    (x1 + 5, y1 + y_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 255),
                    thickness=2,
                )
                y_offset += 28
            if self._stop_delay_y != self._zero_int:
                text = f"BrakeY {int(self._stop_delay_y_counter.item())}/{int(self._stop_delay_y.item())}"
                img = vis.plot_text(
                    img,
                    text,
                    (x1 + 5, y1 + y_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 255),
                    thickness=2,
                )
                y_offset += 28
            # Hysteresis/cooldown indicators
            if self._cancel_hysteresis_frames > self._zero_int:
                text = f"Hyst {int(self._cancel_hysteresis_frames.item())}"
                img = vis.plot_text(
                    img,
                    text,
                    (x1 + 5, y1 + y_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (128, 255, 128),
                    thickness=2,
                )
                y_offset += 28
            if (
                self._cooldown_x_counter > self._zero_int
                or self._cooldown_y_counter > self._zero_int
            ):
                text = f"Cd X{int(self._cooldown_x_counter.item())} Y{int(self._cooldown_y_counter.item())}"
                img = vis.plot_text(
                    img,
                    text,
                    (x1 + 5, y1 + y_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 200, 0),
                    thickness=2,
                )
        # img = vis.draw_arrows(img, bbox=draw_box, horizontal=True, vertical=True)
        if draw_thresholds and self._sticky_translation:
            sticky, unsticky = self._get_sticky_translation_sizes()
            center_tensor = center(self.bounding_box())
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
                following_bbox = following_box.bounding_box()
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

    def get_gaussian_y_about_width_center(self, x) -> float:

        # return 1.0

        if self._horizontal_image_gaussian_distribution is None:
            return 1.0
        else:
            if self._gaussian_x_clamp is not None:
                x = torch.clamp(x, min=self._gaussian_x_clamp[0], max=self._gaussian_x_clamp[1])
            return (
                self._horizontal_image_gaussian_distribution.get_gaussian_y_from_image_x_position(x)
            )

    def _get_sticky_translation_sizes(self):
        gaussian_factor = 1 - self.get_gaussian_y_about_width_center(center(self.bounding_box())[0])
        gaussian_mult = 6
        gaussian_add = gaussian_factor * gaussian_mult

        max_sticky_size = self._max_speed_x * self._sticky_translation_gaussian_mult + gaussian_add
        sticky_size = width(self.bounding_box()) / self._sticky_size_ratio_to_frame_width
        sticky_size = min(sticky_size, max_sticky_size)

        ratio = float(self._unsticky_translation_size_ratio)
        if ratio < 1.0:
            # Interpret values <1 as inverted to enforce proper hysteresis
            ratio = 1.0 / max(ratio, 1e-3)
        unsticky_size = sticky_size * ratio

        return sticky_size, unsticky_size

    def _make_label(self):
        return f"dx={self._current_speed_x.item():.1f}, dy={self._current_speed_y.item()}, {self._label}"

    def _clamp_speed(self, scale: float = 1.0):
        mx = self._max_speed_x * scale
        my = self._max_speed_y * scale
        self._current_speed_x = torch.clamp(self._current_speed_x, min=-mx, max=mx)
        self._current_speed_y = torch.clamp(self._current_speed_y, min=-my, max=my)

    def adjust_speed(
        self,
        accel_x: Optional[torch.Tensor] = None,
        accel_y: Optional[torch.Tensor] = None,
        scale_constraints: Optional[float] = None,
        nonstop_delay: Optional[torch.Tensor] = None,
    ):
        if scale_constraints is not None:
            mult = scale_constraints
            if accel_x is not None:
                accel_x = torch.clamp(
                    accel_x, min=-self._max_accel_x * mult, max=self._max_accel_x * mult
                )
            if accel_y is not None:
                accel_y = torch.clamp(
                    accel_y, min=-self._max_accel_y * mult, max=self._max_accel_y * mult
                )
        if accel_x is not None:
            self._current_speed_x += accel_x

        if accel_y is not None:
            self._current_speed_y += accel_y

        if scale_constraints is not None:
            self._clamp_speed(scale=scale_constraints)

        if nonstop_delay is not None:
            self._nonstop_delay = nonstop_delay
            self._nonstop_delay_counter = self._zero.clone()

    # Public: begin a per-axis stop delay (mirror of C++ API for parity)
    def begin_stop_delay(
        self,
        delay_x: Optional[int] = None,
        delay_y: Optional[int] = None,
    ):
        # Repeated overshoot requests must retain the original braking deadline.
        if delay_x is not None and int(delay_x) > 0 and self._stop_delay_x == self._zero_int:
            self._stop_delay_x = torch.tensor(int(delay_x), dtype=torch.int64, device=self.device)
            self._stop_delay_x_counter = self._zero_int.clone()
            self._stop_decel_x = -self._current_speed_x / self._stop_delay_x.to(
                self._current_speed_x.dtype
            )
            self._stop_trigger_dir_x = torch.sign(self._current_speed_x)
        if delay_y is not None and int(delay_y) > 0 and self._stop_delay_y == self._zero_int:
            self._stop_delay_y = torch.tensor(int(delay_y), dtype=torch.int64, device=self.device)
            self._stop_delay_y_counter = self._zero_int.clone()
            self._stop_decel_y = -self._current_speed_y / self._stop_delay_y.to(
                self._current_speed_y.dtype
            )
            self._stop_trigger_dir_y = torch.sign(self._current_speed_y)

    def scale_speed(
        self,
        ratio_x: Optional[torch.Tensor] = None,
        ratio_y: Optional[torch.Tensor] = None,
        clamp_to_max: Optional[bool] = False,
    ):
        if clamp_to_max:
            self._clamp_speed()
        if ratio_x is not None:
            self._current_speed_x *= ratio_x

        if ratio_y is not None:
            self._current_speed_y *= ratio_y

    def get_size_scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the width and height scale that we adjust the initial target rectangle to
        """
        return self._scale_width, self._scale_height

    def set_destination(self, dest_box: torch.Tensor):
        """
        We try to go to the given box's position, given
        our current velocity and constraints
        """
        if isinstance(dest_box, BasicBox):
            dest_box = dest_box.bounding_box()

        # print(f"{self._label}: set_destination({dest_box})")
        # Reset one-frame cancel flash indicators
        self._cancel_stop_x_flash = False
        self._cancel_stop_y_flash = False

        bbox = self.bounding_box()
        center_current = center(bbox)
        center_dest = center(dest_box)
        # Low-pass filter the destination center to smooth panning.
        if self._pan_smoothing_alpha > 0.0:
            if self._filtered_target_center is None:
                self._filtered_target_center = center_dest.clone()
            else:
                a = min(max(self._pan_smoothing_alpha, 0.0), 1.0)
                self._filtered_target_center = self._filtered_target_center + a * (
                    center_dest - self._filtered_target_center
                )
            center_dest = self._filtered_target_center
        total_diff = center_dest - center_current
        # print(f"{self._label}: {total_diff=}")
        # If both the dest box and our current box are on an edge, we zero-out
        # the magnitude in the direction of that edge so that the size
        # differences of the box don't keep us in the un-stuck mode,
        # even though we can't move anymore in that direction
        # TODO: do cleverly with pytorch tensors
        if self._arena_box is not None:
            edge_ok = torch.logical_not(
                check_for_box_overshoot(
                    box=bbox,
                    bounding_box=self._arena_box + self._inflate_arena_for_unsticky_edges,
                    movement_directions=total_diff,
                    epsilon=0.1,
                )
            )
            self._current_speed_x *= edge_ok[0]
            self._current_speed_y *= edge_ok[1]
            total_diff *= edge_ok
            # logger.info(total_diff)

        # BEGIN Sticky Translation
        if self._sticky_translation and not self.is_nonstop():
            diff_magnitude = torch.linalg.norm(total_diff)

            # Check if the new center is in a direction opposed to our current velocity
            # velocity = torch.tensor(
            #     [self._current_speed_x, self._current_speed_y],
            #     device=self._current_speed_x.device,
            # )
            # s1 = torch.sign(total_diff)
            # s2 = torch.sign(velocity)
            # changed_direction = s1 * s2

            # # Reduce velocity on axes that changed direction
            # velocity = torch.where(changed_direction < 0, self._zero, velocity)

            sticky, unsticky = self._get_sticky_translation_sizes()
            if not self._translation_is_frozen and diff_magnitude <= sticky:
                self._translation_is_frozen = True
                self._current_speed_x = self._zero.clone()
                self._current_speed_y = self._zero.clone()
            elif self._translation_is_frozen and diff_magnitude >= unsticky:
                self._translation_is_frozen = False
                # Unstick at zero velocity
                self._current_speed_x = self._zero.clone()
                self._current_speed_y = self._zero.clone()
        # END Sticky Translation

        # Build per-axis accel, allowing stop-delay braking to override input
        accel_x = total_diff[0]
        accel_y = total_diff[1]

        if not self.is_nonstop():
            # Trigger braking if not already active and direction changes
            if (
                self._stop_delay_x == self._zero_int
                and self._cooldown_x_counter == self._zero_int
                and different_directions(total_diff[0], self._current_speed_x)
            ):
                moving_enough_x = torch.abs(self._current_speed_x) >= (self._max_speed_x / 6)
                if self._stop_on_dir_change_delay > self._zero_int and moving_enough_x:
                    self._stop_delay_x = self._stop_on_dir_change_delay.clone()
                    self._stop_delay_x_counter = self._zero_int.clone()
                    # Decelerate linearly to zero in N frames
                    self._stop_decel_x = -self._current_speed_x / self._stop_delay_x.to(
                        self._current_speed_x.dtype
                    )
                    self._stop_trigger_dir_x = torch.sign(total_diff[0]).to(
                        self._current_speed_x.dtype
                    )
                else:
                    # legacy mild damping behavior
                    self._current_speed_x = torch.where(
                        torch.abs(self._current_speed_x) < self._max_speed_x / 6,
                        0,
                        self._current_speed_x / 2,
                    )
                    if self._stop_on_dir_change:
                        accel_x = total_diff[0] * 0.25

            if (
                self._stop_delay_y == self._zero_int
                and self._cooldown_y_counter == self._zero_int
                and different_directions(total_diff[1], self._current_speed_y)
            ):
                moving_enough_y = torch.abs(self._current_speed_y) >= (self._max_speed_y / 6)
                if self._stop_on_dir_change_delay > self._zero_int and moving_enough_y:
                    self._stop_delay_y = self._stop_on_dir_change_delay.clone()
                    self._stop_delay_y_counter = self._zero_int.clone()
                    self._stop_decel_y = -self._current_speed_y / self._stop_delay_y.to(
                        self._current_speed_y.dtype
                    )
                    self._stop_trigger_dir_y = torch.sign(total_diff[1]).to(
                        self._current_speed_y.dtype
                    )
                else:
                    self._current_speed_y = torch.where(
                        torch.abs(self._current_speed_y) < self._max_speed_y / 6,
                        0,
                        self._current_speed_y / 2,
                    )
                    if self._stop_on_dir_change:
                        accel_y = total_diff[1] * 0.25

        # If braking is active on an axis, ignore input accel for that axis
        if self._stop_delay_x != self._zero_int:
            if (
                self._cancel_stop_on_opposite_dir
                and torch.sign(total_diff[0]) != 0
                and (torch.sign(total_diff[0]) == -self._stop_trigger_dir_x)
            ):
                # Hysteresis: require consecutive frames before cancel
                if self._cancel_hysteresis_frames > self._zero_int:
                    self._cancel_opp_x_count += 1
                    if self._cancel_opp_x_count >= self._cancel_hysteresis_frames:
                        self._stop_delay_x = self._zero_int.clone()
                        self._stop_delay_x_counter = self._zero_int.clone()
                        self._stop_decel_x = self._zero.clone()
                        self._cancel_stop_x_flash = True
                        self._cancel_opp_x_count = self._zero_int.clone()
                        self._cooldown_x_counter = self._stop_delay_cooldown_frames.clone()
                else:
                    # Cancel immediately
                    self._stop_delay_x = self._zero_int.clone()
                    self._stop_delay_x_counter = self._zero_int.clone()
                    self._stop_decel_x = self._zero.clone()
                    self._cancel_stop_x_flash = True
                    self._cooldown_x_counter = self._stop_delay_cooldown_frames.clone()
            else:
                self._cancel_opp_x_count = self._zero_int.clone()
                accel_x = self._stop_decel_x
        if self._stop_delay_y != self._zero_int:
            if (
                self._cancel_stop_on_opposite_dir
                and torch.sign(total_diff[1]) != 0
                and (torch.sign(total_diff[1]) == -self._stop_trigger_dir_y)
            ):
                if self._cancel_hysteresis_frames > self._zero_int:
                    self._cancel_opp_y_count += 1
                    if self._cancel_opp_y_count >= self._cancel_hysteresis_frames:
                        self._stop_delay_y = self._zero_int.clone()
                        self._stop_delay_y_counter = self._zero_int.clone()
                        self._stop_decel_y = self._zero.clone()
                        self._cancel_stop_y_flash = True
                        self._cancel_opp_y_count = self._zero_int.clone()
                        self._cooldown_y_counter = self._stop_delay_cooldown_frames.clone()
                else:
                    # Cancel immediately
                    self._stop_delay_y = self._zero_int.clone()
                    self._stop_delay_y_counter = self._zero_int.clone()
                    self._stop_decel_y = self._zero.clone()
                    self._cancel_stop_y_flash = True
                    self._cooldown_y_counter = self._stop_delay_cooldown_frames.clone()
            else:
                self._cancel_opp_y_count = self._zero_int.clone()
                accel_y = self._stop_decel_y

        self.adjust_speed(
            accel_x=accel_x,
            accel_y=accel_y,
            scale_constraints=1.0,
        )

        # Time-to-destination speed limiting (per-axis)
        def _limit_speed_ttg(v, dist, frames):
            if frames <= 0:
                return v
            sgn = torch.sign(dist)
            if sgn == 0:
                return v
            if torch.sign(v) == sgn:
                limit = torch.abs(dist) / frames.to(v.dtype)
                vmax = limit
                v = torch.clamp(v, min=-vmax, max=vmax)
                if (
                    self._ttg_stop_speed_threshold > 0.0
                    and torch.abs(dist) <= self._ttg_stop_speed_threshold * frames.to(v.dtype)
                    and torch.abs(v) <= self._ttg_stop_speed_threshold
                ):
                    v = self._zero.clone()
            return v

        self._current_speed_x = _limit_speed_ttg(
            self._current_speed_x, total_diff[0], self._ttg_limit_frames
        )
        self._current_speed_y = _limit_speed_ttg(
            self._current_speed_y, total_diff[1], self._ttg_limit_frames
        )

        super(MovingBox, self).set_destination(dest_box=dest_box)

    def forward(self, dest_box: torch.Tensor):
        if self._scale_width is not None or self._scale_height is not None:
            dest_box = scale_box(
                dest_box, scale_width=self._scale_width, scale_height=self._scale_height
            )
            if self._clamp_scaled_input_box:
                # Clamp it to the image, since any translation back into the frame will
                # end up including parts that we don't want
                dest_box = clamp_box(box=dest_box, clamp_box=self._arena_box)
                # pass

        self.set_destination(dest_box=dest_box)
        return self.next_position()

    def next_position(self) -> torch.Tensor:
        arena_box = self._arena_box

        # BEGIN Sticky Translation
        if self._translation_is_frozen and not self.is_nonstop():
            dx = self._zero
            dy = self._zero
        else:
            dx = self._current_speed_x
            dy = self._current_speed_y

        if self._size_is_frozen:
            dw = self._zero
            dh = self._zero
        else:
            dw = self._current_speed_w / 2
            dh = self._current_speed_h / 2
        # END Sticky Translation

        # print(f"{self._label}: {self._current_speed_w=}, {self._current_speed_h=}")

        new_box = self._bbox
        new_box += torch.tensor(
            [
                dx - dw,
                dy - dh,
                dx + dw,
                dy + dh,
            ],
            dtype=self._bbox.dtype,
            device=self._bbox.device,
        )

        # Constrain size
        new_ww = width(new_box)
        new_hh = height(new_box)
        ww = max(new_ww, self._min_width)
        hh = max(new_hh, self._min_height)
        self._size_constrained = ww != new_ww or hh != new_hh
        new_box = make_box_at_center(center_point=center(new_box), w=ww, h=hh)

        # Assign new bounding box
        self._bbox = new_box
        self._update_resize_stop_delays()

        if self._nonstop_delay != self._zero:
            # logger.info(f"self._nonstop_delay_counter={self._nonstop_delay_counter.item()}")
            self._nonstop_delay_counter += 1
            if self._nonstop_delay_counter > self._nonstop_delay:
                self._nonstop_delay = self._zero
                self._nonstop_delay_counter = self._zero.clone()
                # Optional braking after nonstop completes
                if self._post_nonstop_stop_delay > self._zero_int and torch.abs(
                    self._current_speed_x
                ) >= (self._max_speed_x / 6):
                    self._stop_delay_x = self._post_nonstop_stop_delay.clone()
                    self._stop_delay_x_counter = self._zero_int.clone()
                    self._stop_decel_x = -self._current_speed_x / self._stop_delay_x.to(
                        self._current_speed_x.dtype
                    )
                    self._stop_trigger_dir_x = torch.sign(self._current_speed_x)
        # Update per-axis stop delays
        if self._stop_delay_x != self._zero_int:
            self._stop_delay_x_counter += 1
            if self._stop_delay_x_counter >= self._stop_delay_x:
                self._stop_delay_x = self._zero_int.clone()
                self._stop_delay_x_counter = self._zero_int.clone()
                self._stop_decel_x = self._zero.clone()
                self._current_speed_x = self._zero.clone()
                if self._stop_delay_cooldown_frames > self._zero_int:
                    self._cooldown_x_counter = self._stop_delay_cooldown_frames.clone()
        if self._stop_delay_y != self._zero_int:
            self._stop_delay_y_counter += 1
            if self._stop_delay_y_counter >= self._stop_delay_y:
                self._stop_delay_y = self._zero_int.clone()
                self._stop_delay_y_counter = self._zero_int.clone()
                self._stop_decel_y = self._zero.clone()
                self._current_speed_y = self._zero.clone()
                if self._stop_delay_cooldown_frames > self._zero_int:
                    self._cooldown_y_counter = self._stop_delay_cooldown_frames.clone()
        # Decrement cooldowns after updating stop delays (matches C++ behavior)
        if self._cooldown_x_counter > self._zero_int:
            self._cooldown_x_counter -= 1
        if self._cooldown_y_counter > self._zero_int:
            self._cooldown_y_counter -= 1
        self.stop_translation_if_out_of_arena()
        self.clamp_to_arena()
        if self._fixed_aspect_ratio is not None:
            self._bbox = self.set_aspect_ratio(self._bbox, self._fixed_aspect_ratio)
        self.clamp_size_scaled()  # will maintain aspect ratio
        if arena_box is not None:
            self._bbox, was_shifted_x, was_shifted_y = shift_box_to_edge(self._bbox, arena_box)
            if was_shifted_x:
                # Abrupt slow-down at the edge: keep immediate halving rather than braking
                self._current_speed_x /= 2
            if was_shifted_y:
                self._current_speed_y /= 2

        if self._fixed_aspect_ratio is not None:
            assert torch.isclose(aspect_ratio(self._bbox), self._fixed_aspect_ratio)
        return self._bbox

    def clamp_size_scaled(self) -> None:
        w = width(self._bbox).unsqueeze(0)
        h = height(self._bbox).unsqueeze(0)
        wscale = self._zero
        hscale = self._zero
        if w > self._max_width:
            wscale = self._max_width / w
        if h > self._max_height:
            hscale = self._max_height / h
        final_scale = torch.max(wscale, hscale)
        if final_scale != self._zero:
            w *= final_scale
            h *= final_scale
            self._bbox = make_box_at_center(center(self._bbox), w=w, h=h)

    def clamp_to_arena(self):
        if self._arena_box is None:
            return
        self._bbox = clamp_box(box=self._bbox, clamp_box=self._arena_box)

    def stop_translation_if_out_of_arena(self) -> None:
        if self._arena_box is None:
            return
        edge_ok = torch.logical_not(
            check_for_box_overshoot(
                box=self._bbox,
                bounding_box=self._arena_box + self._inflate_arena_for_unsticky_edges,
                movement_directions=torch.tensor([self._current_speed_x, self._current_speed_y]),
                epsilon=0.1,
            )
        )
        self._current_speed_x *= edge_ok[0]
        self._current_speed_y *= edge_ok[1]

    def set_aspect_ratio(
        self, setting_box: torch.Tensor, aspect_ratio: torch.Tensor
    ) -> torch.Tensor:
        if self._arena_box is not None:
            setting_box = clamp_box(setting_box, self._arena_box)
        w = width(setting_box)
        h = height(setting_box)
        if w / h < aspect_ratio:
            # Constrain by height
            new_h = h
            new_w = new_h * aspect_ratio
        else:
            # Constrain by width
            new_w = w
            new_h = new_w / aspect_ratio
        assert new_w >= w
        return make_box_at_center(center_point=center(setting_box), w=new_w, h=new_h)

    def is_nonstop(self):
        return self._nonstop_delay != self._zero


def different_directions(d1: torch.Tensor, d2: torch.Tensor):
    return torch.sign(d1) * torch.sign(d2) < 0
