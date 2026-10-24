"""Transforms that operate on per-frame video tensors (cropping, sharpening).

These classes are registered as mmengine transforms and typically appear in
Aspen pipelines to prepare frames for downstream models.

@see @ref hmlib.hm_opts.hm_opts "hm_opts" for CLI flags that configure video-frame behavior.
"""

from typing import Any

import numpy as np
import torch
from mmengine.registry import TRANSFORMS

from hmlib.algo.unsharp_mask import unsharp_mask
from hmlib.log import logger
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.utils.image import (
    crop_image,
    image_height,
    image_width,
    make_channels_last,
    resize_image,
    to_float_image,
    to_uint8_image,
)


@TRANSFORMS.register_module()
class HmCropToVideoFrame:
    """Crop camera images to a fixed video frame aspect and size.

    Expects ``results`` to contain a list of images under ``\"img\"`` and
    bounding boxes under ``\"camera_box\"``. The cropped, resized tensors
    are stacked back into ``results[\"img\"]``.

    @see @ref HmUnsharpMask "HmUnsharpMask" for optional sharpening later in the pipeline.
    """

    def __init__(
        self, crop_image: bool = True, unsharp_mask: bool = True, enabled: bool = True
    ) -> None:
        self._crop_image = crop_image
        self._unsharp_mask = unsharp_mask
        self._enabled = enabled
        self._pass: int = 0

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if not self._enabled:
            return results
        self._pass += 1
        video_frame_cfg = results["video_frame_cfg"]
        images = results.pop("img")
        if not self._crop_image:
            if isinstance(images, list):
                # since no cropping, they should be the same size
                results["img"] = torch.stack(images)
            else:
                # Just put it back
                results["img"] = images
            # Not cropping, so set output size to input size
            video_frame_cfg["output_frame_width"] = image_width(images)
            video_frame_cfg["output_frame_height"] = image_height(images)
            return results
        current_boxes = results["camera_box"]
        cropped_images: list[torch.Tensor] = []
        src_image_height = image_height(images)
        src_image_width = image_width(images)
        images = unwrap_tensor(images)
        if not torch.is_floating_point(images):
            images = to_float_image(images, dtype=torch.float)
        for index, (img, bbox) in enumerate(zip(images, current_boxes)):
            # box_h = height(bbox)
            # box_w = width(bbox)
            # box_ar = box_w / box_h
            # assert np.isclose(float(box_ar), float(video_frame_cfg["output_aspect_ratio"]))
            if video_frame_cfg["no_crop"]:
                intbox = [0, 0, src_image_width, src_image_height]
            else:
                intbox = [int(i) for i in bbox]
            x1 = intbox[0]
            y1 = intbox[1]
            y2 = intbox[3]
            x2 = intbox[2]
            # x2 = int(x1 + int(float(y2 - y1) * video_frame_cfg["output_aspect_ratio"]))
            # if True:
            #     if not (y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0):
            #         pass
            assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
            if y1 > src_image_height or y2 > src_image_height:
                # assert False
                logger.warning(
                    "y1 (%d) or y2 (%d) is too large, should be < %d",
                    y1,
                    y2,
                    src_image_height,
                )
                y1 = min(y1, src_image_height)
                y2 = min(y2, src_image_height)
            if x1 > src_image_width or x2 > src_image_width:
                logger.warning(
                    "x1 (%d) or x2 (%d) is too large, should be < %d",
                    x1,
                    x2,
                    src_image_width,
                )
                x1 = min(x1, src_image_width)
                x2 = min(x2, src_image_width)
                # assert False

            img = crop_image(img, x1, y1, x2, y2)
            video_frame_width = int(video_frame_cfg["output_frame_width"])
            video_frame_height = int(video_frame_cfg["output_frame_height"])
            if results.get("shared_preview_metadata") is not None:
                from hmlib.ui.preview_overlays import compose_transform

                scale_x = video_frame_width / (x2 - x1)
                scale_y = video_frame_height / (y2 - y1)
                transform = np.array(
                    [
                        [scale_x, 0, (0.5 - x1) * scale_x - 0.5],
                        [0, scale_y, (0.5 - y1) * scale_y - 0.5],
                        [0, 0, 1],
                    ]
                )
                compose_transform(results["shared_preview_metadata"], index, transform)
            image_h = image_height(img)
            image_w = image_width(img)
            # Make sure they have the same aspect ratio
            image_ar = image_w / image_h
            vf_ar = video_frame_width / video_frame_height
            assert np.isclose(image_ar, vf_ar, 0.005, 0.005), (
                f"Image aspect ratio {image_ar} does not match " f"video frame aspect ratio {vf_ar}"
            )
            if image_h != video_frame_height or image_w != video_frame_width:
                img = resize_image(
                    img=img,
                    new_width=video_frame_width,
                    new_height=video_frame_height,
                )
            # By here, our image size should be exactly the video frame output size
            assert image_height(img) == video_frame_height
            assert image_width(img) == video_frame_width
            cropped_images.append(img)
        results["img"] = torch.stack(cropped_images)
        return results


@TRANSFORMS.register_module()
class HmUnsharpMask:
    """Apply an unsharp mask filter to images in a pipeline result dict.

    @param enabled: If ``True``, apply the filter to each call.
    @param image_label: Key inside ``results`` that holds the image tensor.
    @see @ref hmlib.algo.unsharp_mask.unsharp_mask "unsharp_mask" for implementation details.
    """

    def __init__(self, enabled: bool = False, image_label: str = "img") -> None:
        self._enabled = enabled
        self._image_label = image_label

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if self._enabled:
            results[self._image_label] = unsharp_mask(results[self._image_label])
        return results


@TRANSFORMS.register_module()
class HmMakeVisibleImage:
    """Convert image tensors to a visible format for saving or display.

    This transform ensures that the image tensor has
    channels-last memory layout, and is in uint8 format with values in [0, 255].
    Also makes it contiguous in memory.

    Expects ``results`` to contain an image tensor under ``\"img\"``.

    @param enabled: If ``True``, apply the conversion to each call.
    """

    def __init__(
        self, enabled: bool = True, image_labels: tuple[str, ...] = ("img", "ez_img")
    ) -> None:
        self._enabled = enabled
        self._image_labels = image_labels

    @staticmethod
    def _make_visible_image(
        img: torch.Tensor | StreamTensorBase,
    ) -> torch.Tensor:
        """Convert an input image tensor to a visible uint8 format.

        This method assumes the input is either:
          - A floating-point tensor in [0, 1] range.
          - An integer tensor in [0, 255] range.
        """
        img = unwrap_tensor(img)
        img = make_channels_last(img)
        img = to_uint8_image(img)
        img = img.contiguous()
        return img

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if not self._enabled:
            return results
        for label in self._image_labels:
            img = results.get(label)
            if img is None:
                continue
            results[label] = self._make_visible_image(img)
        return results
