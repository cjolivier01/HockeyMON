from typing import Optional, Tuple, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.bbox.box_functions import center, height, width
from hmlib.utils.distributions import ImageHorizontalGaussianDistribution
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, rotate_image, to_float_image


def _slow_to_tensor(tensor: Union[torch.Tensor, StreamTensorBase]) -> torch.Tensor:
    """
    Give up on the stream and get the sync'd tensor
    """
    if isinstance(tensor, StreamTensorBase):
        tensor._verbose = True
        return tensor.wait()
    return tensor


def image_wh(image: torch.Tensor):
    if image.shape[-1] in [3, 4]:
        return torch.tensor(
            [image.shape[-2], image.shape[-3]], dtype=torch.float, device=image.device
        )
    assert image.shape[1] in [3, 4]
    return torch.tensor([image.shape[-1], image.shape[-2]], dtype=torch.float, device=image.device)


@TRANSFORMS.register_module()
class HmPerspectiveRotation:
    def __init__(
        self,
        enabled: bool = True,
        fixed_edge_rotation: Optional[bool] = None,
        fixed_edge_rotation_angle: Union[float, Tuple[float, float]] = 0.0,
        dtype: torch.dtype = torch.float,
        pre_clip: bool = False,
        image_label: str = "img",
        bbox_label: str = "camera_box",
        fixed_crop_half_width: bool = False,
        crop_half_width_scale: float = 1.0,
    ):
        if fixed_edge_rotation is not None:
            enabled = bool(fixed_edge_rotation)
        self._enabled = enabled
        self._camera_space_leveling = False
        self._pre_clip = pre_clip
        self._dtype = dtype
        self.set_fixed_edge_rotation_angle(fixed_edge_rotation_angle)
        self._image_label = image_label
        self._bbox_label = bbox_label
        self._fixed_crop_half_width = fixed_crop_half_width
        self._crop_half_width_scale = float(crop_half_width_scale)
        self._horizontal_image_gaussian_distribution = None
        self._zero_uint8 = None
        self._crop_half_width: Optional[float] = None

    def set_fixed_edge_rotation_angle(self, value: Union[float, Tuple[float, float]]) -> None:
        """Update the live left/right edge rotation angles."""
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("fixed_edge_rotation_angle requires exactly two side values")
            self._fixed_edge_rotation_angle = (float(value[0]), float(value[1]))
        else:
            self._fixed_edge_rotation_angle = float(value)

    def _rotation_disabled(self) -> bool:
        value = self._fixed_edge_rotation_angle
        if isinstance(value, (list, tuple)):
            return all(float(item) == 0.0 for item in value)
        return float(value) == 0.0

    def set_camera_space_leveling(self, enabled: bool) -> None:
        """Suppress Program edge correction without discarding its saved angles."""
        self._camera_space_leveling = bool(enabled)

    def __call__(self, results):
        if not self._enabled or self._camera_space_leveling or self._rotation_disabled():
            return results
        online_im = results.pop(self._image_label)
        current_box = results.pop(self._bbox_label)
        online_im = _slow_to_tensor(online_im)
        online_im = to_float_image(
            online_im,
            non_blocking=True,
            dtype=self._dtype,
        )
        src_image_width = image_width(online_im)
        crop_half_width = None
        if self._pre_clip:
            bbox_w = current_box[:, 2] - current_box[:, 0]
            bbox_h = current_box[:, 3] - current_box[:, 1]
            crop_half_width = torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2.0
            crop_half_width = crop_half_width.max().item() * self._crop_half_width_scale
            if self._fixed_crop_half_width:
                if self._crop_half_width is None:
                    self._crop_half_width = float(crop_half_width)
                crop_half_width = float(self._crop_half_width)
        rotated_images = []
        current_boxes = []
        for img, bbox in zip(online_im, current_box):
            rotation_point = [int(i) for i in center(bbox)]
            width_center = src_image_width / 2
            if rotation_point[0] < width_center:
                mult = -1
            else:
                mult = 1

            gaussian = 1 - self._get_gaussian(src_image_width).get_gaussian_y_from_image_x_position(
                rotation_point[0],
                wide=True,
            )

            fixed_edge_rotation_angle = self._fixed_edge_rotation_angle
            if isinstance(fixed_edge_rotation_angle, (list, tuple)):
                assert len(fixed_edge_rotation_angle) == 2
                if rotation_point[0] < src_image_width // 2:
                    fixed_edge_rotation_angle = float(self._fixed_edge_rotation_angle[0])
                else:
                    fixed_edge_rotation_angle = float(self._fixed_edge_rotation_angle[1])

            angle = fixed_edge_rotation_angle - fixed_edge_rotation_angle * gaussian
            angle *= mult

            # BEGIN PERFORMANCE HACK
            #
            # Chop off edges of image that won't be visible after a final crop
            # before we rotate in order to reduce the computation necessary
            # for the rotation (as well as other subsequent operations)
            #
            if self._pre_clip:
                pre_size = image_width(img), image_height(img)
                img, bbox = self.crop_working_image_width(
                    image=img, current_box=bbox, crop_half_width=crop_half_width
                )
                post_size = image_width(img), image_height(img)
                if pre_size != post_size:
                    results["rotation_crop"] = {"from": pre_size, "to": post_size}
                rotation_point = [int(i) for i in center(bbox)]
            #
            # END PERFORMANCE HACK
            #

            img = rotate_image(
                img=img,
                angle=float(angle),
                rotation_point=rotation_point,
            )
            rotated_images.append(img)
            current_boxes.append(bbox)
        del online_im
        results[self._image_label] = torch.stack(rotated_images)
        results[self._bbox_label] = torch.stack(current_boxes)

        return results

    def crop_working_image_width(
        self, image: torch.Tensor, current_box: torch.Tensor, crop_half_width: Optional[float]
    ):
        """
        We try to only retain enough image to supply an arbitrary rotation
        about the center of the given bounding box with pixels,
        and offset that bounding box to be relative to the new (hopefully smaller)
        image
        """
        if self._zero_uint8 is None:
            self._zero_uint8 = torch.tensor(0, dtype=torch.uint8, device=image.device)

        bbox_w = width(current_box)
        bbox_h = height(current_box)
        bbox_c = center(current_box)
        assert bbox_w > 10  # Sanity
        assert bbox_h > 10  # Sanity
        # make sure we're the expected (albeit arbitrary) channels first
        assert image.shape[-1] in [3, 4]
        img_wh = image_wh(image)
        min_width_per_side = (
            torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2
            if crop_half_width is None
            else torch.tensor(crop_half_width, dtype=torch.float, device=current_box.device)
        )
        desired_width = int(torch.ceil(2.0 * min_width_per_side).item())
        img_width = int(img_wh[0].item())
        if desired_width >= img_width:
            clip_left = 0
            clip_right = img_width
        else:
            center_x = float(bbox_c[0].item())
            clip_left = int(round(center_x - desired_width / 2.0))
            clip_right = clip_left + desired_width
            if clip_left < 0:
                clip_right -= clip_left
                clip_left = 0
            if clip_right > img_width:
                clip_left -= clip_right - img_width
                clip_right = img_width
                if clip_left < 0:
                    clip_left = 0
        if image.ndim == 3:
            # no batch dimension
            image = image[:, clip_left:clip_right, :]
        else:
            # with batch dimension
            image = image[:, :, clip_left:clip_right, :]
        current_box[0] -= clip_left
        current_box[2] -= clip_left

        return image, current_box

    def _get_gaussian(self, image_width: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = ImageHorizontalGaussianDistribution(
                image_width, invert=True, show=False
            )
        else:
            assert image_width == self._horizontal_image_gaussian_distribution.width
        return self._horizontal_image_gaussian_distribution
