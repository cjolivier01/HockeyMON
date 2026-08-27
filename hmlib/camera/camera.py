"""Camera viewport helpers and trajectory history for pan/zoom control.

Provides utilities for tracking bounding-box histories and computing simple
speed/trajectory metrics used by the camera controller.

@see @ref hmlib.camera.camera_transformer "camera_transformer" for transformer-based control.
"""

from typing import List, Set, Union

import numpy as np
import torch

from hmlib.bbox.box_functions import tlwh_to_tlbr_single
from hmlib.constants import WIDTH_NORMALIZATION_SIZE
from hmlib.log import logger
from hmlib.tracking_utils.visualization import plot_trajectory

# nosec B101

X_SPEED_HISTORY_LENGTH = 5


def _as_scalar(tensor):
    return tensor


def is_equal(f1, f2):
    return torch.isclose(f1, f2, rtol=0.01, atol=0.01)


class VideoFrame(object):
    def __init__(self, image_width: int, image_height: int):
        # Make sure shape indexes haven't been screwed up
        assert bool(image_width > 10)
        assert bool(image_height > 10)
        assert bool(image_height < image_width)  # Usually the case, but not required
        self._image_width = _as_scalar(image_width)
        self._image_height = _as_scalar(image_height)
        self._bbox = torch.tensor(
            # (0, 0, self._image_width - 1, self._image_height - 1),
            (0, 0, self._image_width, self._image_height),
            dtype=torch.float,
            device=self._image_width.device,
        )

    def bounding_box(self):
        return self._bbox.clone()

    @property
    def width(self):
        return self._image_width.clone()

    @property
    def height(self):
        return self._image_height.clone()


class TlwhHistory(object):
    def __init__(self, id: int, max_history_length: int = 25):
        self.id_ = id
        self._max_history_length = max_history_length
        self._image_position_history: List[torch.Tensor] = list()
        self._spatial_distance_sum = 0.0
        self._current_spatial_x_speed = 0.0
        self._image_distance_sum = 0.0
        self._current_image_speed = 0.0
        self._current_image_x_speed = 0.0

    @property
    def id(self):
        return self.id_

    def draw(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        tracking_id = self.id_
        positions = torch.stack(self._image_position_history)
        return plot_trajectory(image, tracking_id, positions)

    def append(self, image_position: torch.Tensor):
        current_historty_length = len(self._image_position_history)
        if current_historty_length >= self._max_history_length - 1:
            assert current_historty_length == self._max_history_length - 1
            removing_image_point = self._image_position_history[0]
            self._image_position_history = self._image_position_history[1:]
            old_image_distance = torch.linalg.norm(
                removing_image_point - self._image_position_history[0]
            )
            self._image_distance_sum -= old_image_distance
        if len(self._image_position_history) > 0:
            new_image_distance = torch.linalg.norm(
                image_position - self._image_position_history[-1]
            )
            self._image_distance_sum += new_image_distance
        self._image_position_history.append(image_position)
        if len(self._image_position_history) > 1:
            self._current_image_speed = self._image_distance_sum / (
                len(self._image_position_history) - 1
            )
            if len(self._image_position_history) >= X_SPEED_HISTORY_LENGTH:
                self._current_image_x_speed = (
                    self._image_position_history[-1][0]
                    - self._image_position_history[-X_SPEED_HISTORY_LENGTH][0]
                ) / float(
                    X_SPEED_HISTORY_LENGTH
                )  # type: ignore
            else:
                self._current_image_x_speed = 0.0
        else:
            self._current_image_speed = 0.0
            self._current_image_x_speed = 0.0

    @property
    def current_image_x_speed(self):
        return self._current_image_x_speed

    @property
    def image_position_history(self):
        return self._image_position_history

    @staticmethod
    def center_point(tlwh: torch.Tensor):
        assert tlwh.ndim == 1  # Not a batch
        top = tlwh[0]
        left = tlwh[1]
        width = tlwh[2]
        height = tlwh[3]
        x_center = left + (width / 2.0)
        y_center = top + (height / 2.0)
        return torch.tensor((x_center, y_center), dtype=torch.float)

    def __len__(self):
        length = len(self._image_position_history)
        return length

    @property
    def image_speed(self):
        return self._current_image_speed


CAMERA_TYPE_MAX_SPEEDS = {
    "GoPro": 200.0,
    "Zhiwei": 200.0,
    "LiveBarn": 300.0,
}


def should_unsharp_mask_camera(camera_name: str) -> bool:
    # return camera_name in ["LiveBarn", "Zhiwei"]
    return False


class HockeyMON:
    """
    The Observer (The Mon)
    """

    # Frames per second that normal speeds are calculated for
    BASE_FPS: float = 29.97

    def __init__(
        self,
        image_width: int,
        image_height: int,
        fps: float,
        device: torch.device,
        camera_name: str,
        max_history: int = 26,
    ):
        self._device = device
        self._video_frame = VideoFrame(
            image_width=self._to_scalar_float(image_width),
            image_height=self._to_scalar_float(image_height),
        )
        self._max_history = max_history
        self._online_ids: Set[int] = set()
        self._id_to_tlwhs_history_map = dict()
        self._fps_speed_scale: float = fps / HockeyMON.BASE_FPS
        self._image_size_speed_scale: float = image_width / WIDTH_NORMALIZATION_SIZE
        self._speed_scale = self._fps_speed_scale / self._image_size_speed_scale

        self._online_image_center_points = []

        self._camera_box_max_speed_x = self._to_scalar_float(
            max(image_width / CAMERA_TYPE_MAX_SPEEDS[camera_name], 12.0)
        )
        self._camera_box_max_speed_y = self._to_scalar_float(
            max(image_height / CAMERA_TYPE_MAX_SPEEDS[camera_name], 12.0)
        )
        logger.info(
            f"Camera Max speeds: x={self._camera_box_max_speed_x}, y={self._camera_box_max_speed_y}"
        )

        self._camera_box_max_accel_x = self._to_scalar_float(1) / self._speed_scale
        self._camera_box_max_accel_y = self._to_scalar_float(1) / self._speed_scale
        logger.info(
            f"Camera Max acceleration: dx={self._camera_box_max_accel_x}, dy={self._camera_box_max_accel_y}"
        )

    def _to_scalar_float(self, scalar_float):
        return torch.tensor(scalar_float, dtype=torch.float, device=self._device)

    def get_history(self, tracking_id: int) -> Union[TlwhHistory, None]:
        return self._id_to_tlwhs_history_map.get(int(tracking_id))

    def append_online_objects(self, online_ids, online_tlws):
        if len(online_ids) == 0:
            return
        assert isinstance(online_ids, torch.Tensor)
        assert isinstance(online_tlws, torch.Tensor)
        self._online_ids = online_ids
        self._online_tlws = online_tlws
        if len(online_tlws) != 0:
            self._online_image_center_points = torch.stack(
                [TlwhHistory.center_point(twls) for twls in online_tlws]
            )
        else:
            self._online_image_center_points = []

        # Add to history
        prev_dict = self._id_to_tlwhs_history_map
        self._id_to_tlwhs_history_map = dict()
        for id, image_pos in zip(self._online_ids, self._online_tlws):
            hist = prev_dict.get(
                id.item(), TlwhHistory(id=id.item(), max_history_length=self._max_history)
            )
            hist.append(image_position=image_pos)
            self._id_to_tlwhs_history_map[id.item()] = hist

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    @property
    def fps_speed_scale(self) -> float:
        return self._fps_speed_scale

    @property
    def video(self):
        return self._video_frame

    def get_tlwh(self, id: int) -> torch.Tensor:
        position_history = self._id_to_tlwhs_history_map[id.item()]
        return position_history.image_position_history[-1]

    def get_group_x_velocity(
        self, min_considered_velocity: float = 0.01, group_threshhold: float = 0.6
    ):
        neg_count = 0
        pos_count = 0
        idle_count = 0
        neg_sum = 0.0
        pos_sum = 0.0
        leftmost_center = None
        rightmost_center = None
        for _, hist in self._id_to_tlwhs_history_map.items():
            pos = hist.image_position_history[-1]
            this_pos_center = [pos[0] + pos[2] / 2, pos[1] + pos[3] / 2]
            if leftmost_center is None:
                leftmost_center = this_pos_center
            if rightmost_center is None:
                rightmost_center = this_pos_center
            image_speed = hist.current_image_x_speed * self._fps_speed_scale
            if image_speed < -min_considered_velocity:
                neg_count += 1
                neg_sum += image_speed
                if this_pos_center[0] < leftmost_center[0]:
                    leftmost_center = this_pos_center
            elif image_speed > min_considered_velocity:
                pos_count += 1
                pos_sum += image_speed
                if this_pos_center[0] > rightmost_center[0]:
                    rightmost_center = this_pos_center
            else:
                idle_count += 1
        total_ids = len(self._id_to_tlwhs_history_map)
        if total_ids and total_ids > 4:  # Don't just consider two players
            if float(pos_count) / total_ids > group_threshhold:
                avg_x_speed = pos_sum / pos_count
                return avg_x_speed, rightmost_center
            elif float(neg_count) / total_ids > group_threshhold:
                avg_x_speed = neg_sum / neg_count
                return avg_x_speed, leftmost_center
        return 0, None

    def get_current_bounding_box(self, ids=None):
        if ids is None:
            ids = self._online_ids
        if len(ids) == 0:
            return self._video_frame.bounding_box()

        tlwh_list = []
        for id in ids:
            # Can this be a select or gather one day?
            tlwh = self.get_tlwh(id)
            tlwh_list.append(tlwh_to_tlbr_single(tlwh))
        bounding_boxes = torch.stack(tlwh_list)
        min_tl, _ = torch.min(bounding_boxes[:, :2], dim=0)
        max_br, _ = torch.max(bounding_boxes[:, 2:], dim=0)
        # The containing bounding box is then
        containing_box = torch.cat([min_tl, max_br])
        return containing_box
