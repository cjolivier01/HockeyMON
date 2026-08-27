"""Experimental high-level stitching CLI combining remap + blend stages.

Supports both Python and CUDA-based blending paths, seam-mask generation
and optional Laplacian blending via :class:`hmlib.stitching.laplacian_blend.LaplacianBlend`.
"""

import argparse
import datetime
import os
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.hm_opts import copy_opts, hm_opts, preferred_arg
from hmlib.orientation import configure_game_videos
from hmlib.stitching.configure_stitching import get_image_geo_position
from hmlib.stitching.image_remapper import ImageRemapper, RemapImageInfoEx
from hmlib.stitching.laplacian_blend import LaplacianBlend, simple_make_full
from hmlib.stitching.seam import load_canvas_seam_mask, read_mapping_canvas_size
from hmlib.stitching.synchronize import synchronize_by_audio
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import show_image
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.hockeymon_compat import (
    BlenderConfig,
    CudaStitchPanoF32,
    CudaStitchPanoNF32,
    CudaStitchPanoNU8,
    CudaStitchPanoU8,
    EnBlender,
    ImageBlender,
    ImageBlenderMode,
    WHDims,
)
from hmlib.utils.image import image_height, image_width, make_channels_first, make_channels_last
from hmlib.utils.progress_bar import convert_hms_to_seconds
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_out import VideoOutput
from hmlib.video.video_stream import VideoStreamReader, VideoStreamWriter
from hmlib.vis.pt_visualization import draw_box

ROOT_DIR = os.getcwd()

from hmlib.log import get_logger

logger = get_logger(__name__)


def make_parser():
    """Build an :class:`argparse.ArgumentParser` for the stitcher CLI."""
    parser = argparse.ArgumentParser("Image Remapper")
    parser = hm_opts.parser(parser)
    parser.add_argument(
        "-b",
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--files",
        "-f",
        dest="files",
        default=None,
        type=str,
        help="Queue size",
    )
    parser.add_argument(
        "-q",
        "--queue-size",
        dest="queue_size",
        default=1,
        type=int,
        help="Queue size",
    )
    parser.add_argument(
        "--start-frame-number",
        default=0,
        type=int,
        help="Start frame number",
    )
    parser.add_argument(
        "--python",
        action="store_true",
        help="Python blending path",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw boxes",
    )
    return parser


@dataclass
class BlendImageInfo:
    def __init__(self, remapped_width: int, remapped_height: int, xpos: int, ypos: int):
        self.remapped_width: int = remapped_width
        self.remapped_height: int = remapped_height
        self.xpos: int = xpos
        self.ypos: int = ypos


@dataclass
class ImageAndPos:
    def __init__(self, image: torch.Tensor, xpos: int, ypos: int):
        self.image: np.ndarray = image
        self.xpos = xpos
        self.ypos = ypos


class PtImageBlender(torch.nn.Module):
    def __init__(
        self,
        images_info: List[BlendImageInfo],
        seam_mask: torch.Tensor,
        xor_mask: torch.Tensor,
        dtype: torch.dtype,
        laplacian_blend: False,
        max_levels: int = 6,
        add_alpha_channel: bool = False,
    ) -> None:
        super().__init__()
        self._image_positions: List[int] = []
        for bli in images_info:
            self._image_positions.append((bli.xpos, bli.ypos))
        self._seam_mask = seam_mask.clone()
        # self._xor_mask = xor_mask.clone() if xor_mask is not None else None
        self._xor_mask = None
        self._dtype: torch.dtype = dtype
        self.max_levels: int = max_levels if laplacian_blend else 0
        if laplacian_blend:
            self._laplacian_blend: bool = LaplacianBlend(
                max_levels=self.max_levels,
                channels=3 if not add_alpha_channel else 4,
                seam_mask=self._seam_mask,
                xor_mask=self._xor_mask,
                dtype=self._dtype,
            )
            self._laplacian_blend.to(seam_mask.device)
        else:
            self._laplacian_blend: bool = None
        # assert self._seam_mask.shape[1] == self._xor_mask.shape[1]
        # assert self._seam_mask.shape[0] == self._xor_mask.shape[0]

        # Final misc init
        self._unique_values = None
        self._left_value = None
        self._right_value = None
        self.init()

    def init(self):
        # Check some sanity
        print(
            f"Blending image size: {image_width(self._seam_mask)} x {image_height(self._seam_mask)}"
        )
        unique_values = torch.unique(self._seam_mask)
        if unique_values.numel() >= 2:
            self._left_value = unique_values[0]
            self._right_value = unique_values[-1]
        else:
            self._left_value = torch.tensor(
                1.0, dtype=self._seam_mask.dtype, device=self._seam_mask.device
            )
            self._right_value = torch.tensor(
                0.0, dtype=self._seam_mask.dtype, device=self._seam_mask.device
            )
        self._unique_values = torch.stack([self._left_value, self._right_value])
        print("Initialized")

    # @torch.jit.script_method
    def forward(
        self,
        image_1: torch.Tensor,
        alpha_mask_1: torch.Tensor,
        image_2: torch.Tensor,
        alpha_mask_2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = image_1.shape[0]
        channels = image_1.shape[1]

        x1 = self._image_positions[0][0]
        y1 = self._image_positions[0][1]
        x2 = self._image_positions[1][0]
        y2 = self._image_positions[1][1]

        assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
        if y1 <= y2:
            y2 -= y1
            y1 = 0
        elif y2 < y1:
            y1 -= y2
            y2 = 0
        if x1 <= x2:
            x2 -= x1
            x1 = 0
        elif x2 < x1:
            x1 -= x2
            x2 = 0

        assert int(x1) == 0 or int(x2) == 0  # for now this is the case
        assert int(y1) == 0 or int(y2) == 0  # for now this is the case

        canvas_h = self._seam_mask.shape[0]
        canvas_w = self._seam_mask.shape[1]

        # Build full-canvas versions of images/masks so we can prevent unmapped
        # pixels from participating in blending.
        (
            full_left,
            alpha_mask_left_full,
            full_right,
            alpha_mask_right_full,
        ) = simple_make_full(
            img_1=image_1,
            mask_1=alpha_mask_1,
            x1=x1,
            y1=y1,
            img_2=image_2,
            mask_2=alpha_mask_2,
            x2=x2,
            y2=y2,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
        if alpha_mask_left_full is not None and alpha_mask_right_full is not None:
            left_valid = ~alpha_mask_left_full
            right_valid = ~alpha_mask_right_full
            only_left = left_valid & ~right_valid
            only_right = right_valid & ~left_valid
            neither = ~(left_valid | right_valid)
        else:
            only_left = only_right = neither = None

        if self._laplacian_blend is not None:
            canvas = self._laplacian_blend.forward(
                left=image_1,
                alpha_mask_left=alpha_mask_1,
                x1=x1,
                y1=y1,
                right=image_2,
                alpha_mask_right=alpha_mask_2,
                x2=x2,
                y2=y2,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
            )
        else:
            canvas = torch.empty(
                size=(
                    batch_size,
                    channels,
                    canvas_h,
                    canvas_w,
                ),
                dtype=image_1.dtype,
                device=image_1.device,
            )
            canvas[:, :, self._seam_mask == self._left_value] = full_left[
                :, :, self._seam_mask == self._left_value
            ]
            canvas[:, :, self._seam_mask == self._right_value] = full_right[
                :, :, self._seam_mask == self._right_value
            ]

        # Override blended results where only one side has valid pixels.
        if only_left is not None and only_left.any():
            canvas[:, :, only_left] = full_left[:, :, only_left]
        if only_right is not None and only_right.any():
            canvas[:, :, only_right] = full_right[:, :, only_right]
        if neither is not None and neither.any():
            canvas[:, :, neither] = 0

        return canvas


def make_cv_compatible_tensor(tensor):
    """Convert a tensor or array to a contiguous H×W×C NumPy array."""
    if isinstance(tensor, torch.Tensor):
        assert tensor.dim() == 3
        if tensor.size(0) == 3 or tensor.size(0) == 4:
            # Need to make channels-last
            tensor = tensor.permute(1, 2, 0)
        return tensor.contiguous().cpu().numpy()
    if tensor.shape[0] == 3 or tensor.shape[0] == 4:
        tensor = tensor.transpose(1, 2, 0)
    return np.ascontiguousarray(tensor)


def get_images_and_positions(dir_name: str, basename: str = "mapping_") -> List[ImageAndPos]:
    xpos_1, ypos_1, _, _ = get_mapping(dir_name=dir_name, basename=f"{basename}0000")
    xpos_2, ypos_2, _, _ = get_mapping(dir_name=dir_name, basename=f"{basename}0001")

    images_and_positions = [
        ImageAndPos(
            image=cv2.imread(os.path.join(dir_name, f"{basename}0000.tif")),
            xpos=xpos_1,
            ypos=ypos_1,
        ),
        ImageAndPos(
            image=cv2.imread(os.path.join(dir_name, f"{basename}0001.tif")),
            xpos=xpos_2,
            ypos=ypos_2,
        ),
    ]
    return images_and_positions


def create_generic_seam_mask(img1_size, img2_size, pos1, pos2):
    # Calculate canvas size
    assert pos1[0] == 0 or pos2[0] == 0
    assert pos1[1] == 0 or pos2[1] == 0

    width = max(pos1[0] + img1_size[0], pos2[0] + img2_size[0])
    height = max(pos1[1] + img1_size[1], pos2[1] + img2_size[1])

    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill regions
    mask[pos1[1] : pos1[1] + img1_size[1], pos1[0] : pos1[0] + img1_size[0]] = 76
    mask[pos2[1] : pos2[1] + img2_size[1], pos2[0] : pos2[0] + img2_size[0]] = 128

    return mask


def make_seam_and_xor_masks(
    dir_name: str,
    basename: str,
    images_and_positions: Optional[List[ImageAndPos]] = None,
    force: bool = False,
    use_enblend_tool: bool = True,
    # use_enblend_tool: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute seam and XOR masks for a pair of mapping images.

    @param dir_name: Directory containing mapping TIFFs and seam files.
    @param basename: Mapping base name, e.g. ``"mapping_"``.
    @param images_and_positions: Optional pre-loaded image/position structs.
    @param force: If True, regenerate even when seam file exists and is fresh.
    @param use_enblend_tool: If True, call external `enblend` for seam masks.
    @return: Pair ``(seam_tensor, xor_mask_tensor)``.
    """
    assert images_and_positions is None or len(images_and_positions) == 2
    seam_filename = os.path.join(dir_name, "seam_file.png")
    xor_filename = os.path.join(dir_name, "xor_file.png")
    seam_tensor = None
    if not force and os.path.isfile(seam_filename):
        mapping_file = os.path.join(dir_name, "mapping_0000.tif")
        if os.path.exists(mapping_file):
            mapping_file_mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(mapping_file)
            ).isoformat()
            seam_file_mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(seam_filename)
            ).isoformat()
            force = mapping_file_mtime >= seam_file_mtime
            if force:
                print(f"Recreating seam files because mapping file is newer ({mapping_file})")
        else:
            print(f"Warning: no mapping file found: {mapping_file}")
    if force or not os.path.isfile(seam_filename):
        if not use_enblend_tool and EnBlender is not None:
            blender = EnBlender(
                args=[
                    "--save-seams",
                    seam_filename,
                    "--save-xor",
                    xor_filename,
                ]
            )

            if not images_and_positions:
                images_and_positions = get_images_and_positions(
                    dir_name=dir_name, basename=basename
                )

            # Blend one image to create the seam file
            _ = blender.blend_images(
                left_image=make_cv_compatible_tensor(images_and_positions[0].image),
                left_xy_pos=[images_and_positions[0].xpos, images_and_positions[0].ypos],
                right_image=make_cv_compatible_tensor(images_and_positions[1].image),
                right_xy_pos=[images_and_positions[1].xpos, images_and_positions[1].ypos],
            )
        else:
            curr_dir = os.getcwd()
            os.chdir(dir_name)
            try:
                cmd: List[str] = [
                    "enblend",
                    f"--save-masks={seam_filename}",
                    "-o",
                    f"{os.path.join(dir_name, 'panorama.tif')}",
                    f"{os.path.join(dir_name, 'mapping_????.tif')}",
                ]
                os.system(" ".join(cmd))
            finally:
                os.chdir(curr_dir)

    if os.path.exists(seam_filename):
        mapping_files = sorted(Path(dir_name).glob(f"{basename}????.tif"))
        canvas_width, canvas_height = read_mapping_canvas_size(mapping_files)
        seam_tensor = torch.from_numpy(
            load_canvas_seam_mask(seam_filename, canvas_width, canvas_height)
        )

    if False:
        seam_w = int(image_width(seam_tensor))
        v1 = seam_tensor[0][0]
        v2 = seam_tensor[0][seam_w - 1]
        seam_tensor[:, : seam_w // 2] = v1
        seam_tensor[:, seam_w // 2 :] = v2
    if os.path.exists(xor_filename):
        xor_tensor = torch.from_numpy(cv2.imread(xor_filename, cv2.IMREAD_ANYDEPTH))
    else:
        xor_tensor = None
    return seam_tensor, xor_tensor


def create_blender_config(
    mode: str,
    dir_name: str,
    basename: str,
    device: torch.device,
    levels: int = 10,
    lazy_init: bool = False,
    interpolation: str = "bilinear",
) -> BlenderConfig:
    """Construct a :class:`BlenderConfig` for Laplacian or hard-seam."""
    config = BlenderConfig()
    if not mode or mode == "multiblend":
        # Nothing needed for legacy multiblend mode
        return config
    config.mode = mode
    if mode in ("hard-seam", "gpu-hard-seam"):
        config.levels = 0
    else:
        config.levels = levels
    config.device = str(device)
    config.seam, _ = make_seam_and_xor_masks(dir_name=dir_name, basename=basename)
    config.lazy_init = lazy_init
    config.interpolation = interpolation
    return config


def get_dims_for_output_video(height: int, width: int, max_width: int, allow_resize: bool = True):
    if allow_resize and max_width and width > max_width:
        hh = float(height)
        ww = float(width)
        ar = ww / hh
        new_h = float(max_width) / ar
        return int(new_h), int(max_width)
    return int(height), int(width)


def my_draw_box(
    image: torch.Tensor,
    x1: int | None,
    y1: int | None,
    x2: int | None,
    y2: int | None,
    color: Tuple[int, int, int],
    thickness: int = 4,
) -> torch.Tensor:
    """Thin wrapper around :func:`draw_box` that tolerates None bounds."""
    if x1 is None:
        x1 = 0
    if y1 is None:
        y1 = 0
    w, h = image_width(image), image_height(image)
    if x2 is None:
        x2 = w - 1
    elif x2 == w:
        x2 -= 1
    if y2 is None:
        y2 = h - 1
    elif y2 == h:
        y2 -= 1
    return draw_box(
        image=image, tlbr=[int(x1), int(y1), int(x2), int(y2)], color=color, thickness=thickness
    )


@dataclass
class Point:
    x: int
    y: int


@dataclass
class CanvasInfo:
    positions: Union[List[Point], None] = None
    width: int = 0
    height: int = 0


def get_canvas_info(
    size_1: List[int], xy_pos_1: List[int], size_2: List[int], xy_pos_2: List[int]
) -> CanvasInfo:
    """Return canvas dimensions and remapped positions for two images."""
    h1, w1 = size_1[-2:]
    h2, w2 = size_2[-2:]
    x1, y1 = xy_pos_1
    x2, y2 = xy_pos_2
    assert x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0
    if x1 <= x2:
        x2 -= x1
        x1 = 0
    else:
        x1 -= x2
        x2 = 0
    if y1 <= y2:
        y2 -= y1
        y1 = 0
    else:
        y1 -= y2
        y2 = 0
    canvas_w = max(x1 + w1, x2 + w2)
    canvas_h = max(y1 + h1, y2 + h2)
    return CanvasInfo(
        positions=[Point(x=x1, y=y1), Point(x=x2, y=y2)], width=canvas_w, height=canvas_h
    )


class SmartRemapperBlender(torch.nn.Module):
    def __init__(
        self,
        remapper_1: ImageRemapper,
        remapper_2: ImageRemapper,
        minimize_blend: bool,
        use_python_blender: bool,
        blend_levels: int,
        dtype: torch.dtype,
        overlap_pad: int,
        draw: bool,
        blend_mode: str,
        seam_tensor: torch.Tensor,
        device: torch.device,
        add_alpha_channel: bool = False,
    ) -> None:
        """Compose remapping of two images and blending into a single module."""
        super().__init__()
        self._remapper_1 = remapper_1
        self._remapper_2 = remapper_2
        self._use_python_blender = use_python_blender
        self._canvas_info: CanvasInfo = get_canvas_info(
            size_1=[self._remapper_1.height, self._remapper_1.width],
            xy_pos_1=[self._remapper_1.xpos, self._remapper_1.ypos],
            size_2=[self._remapper_2.height, self._remapper_2.width],
            xy_pos_2=[self._remapper_2.xpos, self._remapper_2.ypos],
        )
        self._dtype = dtype
        self._overlap_pad = overlap_pad
        self._draw = draw
        self._add_alpha_channel = add_alpha_channel
        self._minimize_blend = minimize_blend
        self._blend_levels = blend_levels
        self._padded_blended_tlbr = None
        self._blend_mode = blend_mode
        self._device = device
        self._overlapping_width = None
        self._empty_image_pixel_value: int = 0
        if self._minimize_blend:
            width_1 = self._remapper_1.width
            x2 = self._canvas_info.positions[1].x
            max_valid_pad_canvas = max(0, self._canvas_info.width - width_1)
            max_valid_pad_left = max(0, width_1 - x2)
            effective_pad = min(self._overlap_pad, x2, max_valid_pad_canvas, max_valid_pad_left)
            if effective_pad != self._overlap_pad:
                self._overlap_pad = int(effective_pad)
        if seam_tensor is not None:
            self.register_buffer("_seam_tensor", self.convert_mask_tensor(seam_tensor))
        else:
            self._seam_tensor = None
        self._xor_mask_tensor = None
        # if xor_mask_tensor is not None:
        #     self.register_buffer(
        #         "_xor_mask_tensor",
        #         self.convert_mask_tensor(xor_mask_tensor if xor_mask_tensor is not None else None),
        #     )
        # else:
        #     self._xor_mask_tensor = None
        self._init()

    def _init(self) -> None:
        if self._minimize_blend:
            self._x1, self._y1, self._x2, self._y2 = (
                self._canvas_info.positions[0].x,
                self._canvas_info.positions[0].y,
                self._canvas_info.positions[1].x,
                self._canvas_info.positions[1].y,
            )

            self._remapper_1.xpos = self._x1
            self._remapper_2.xpos = self._x1 + self._overlap_pad  # start overlapping right away
            width_1 = self._remapper_1.width
            self._overlapping_width = width_1 - self._x2
            assert width_1 > self._x2
            # seam tensor box (box we'll be blending)
            self._padded_blended_tlbr = [
                self._x2 - self._overlap_pad,  # x1
                max(0, min(self._y1, self._y2) - self._overlap_pad),  # y1
                width_1 + self._overlap_pad,  # x2
                min(
                    self._canvas_info.height,
                    max(self._y1 + self._remapper_1.height, self._y2 + self._remapper_2.height)
                    + self._overlap_pad,
                ),  # y2
            ]
            assert self._x2 - self._overlap_pad >= 0
            assert width_1 + self._overlap_pad <= self._canvas_info.width

        if not self._use_python_blender:
            # assert False  # Not interested in this path atm
            self._blender = ImageBlender(
                mode=(
                    ImageBlenderMode.Laplacian
                    if self._blend_mode == "laplacian"
                    else ImageBlenderMode.HardSeam
                ),
                half=False,
                levels=self._blend_levels,
                seam=self._seam_tensor,
                lazy_init=True,
                interpolation="bilinear",
                # add_alpha_channel=self._add_alpha_channel,
            )
            self._blender.to(self._device)
        else:
            # this appears to leave shadows atm
            self._blender = PtImageBlender(
                images_info=[
                    BlendImageInfo(
                        remapped_width=0,
                        remapped_height=0,
                        xpos=self._remapper_1.xpos,
                        ypos=self._remapper_1.ypos,
                    ),
                    BlendImageInfo(
                        remapped_width=0,
                        remapped_height=0,
                        xpos=self._remapper_2.xpos,
                        ypos=self._remapper_2.ypos,
                    ),
                ],
                seam_mask=self._seam_tensor.contiguous().to(self._device),
                xor_mask=(
                    self._xor_mask_tensor.contiguous().to(self._device)
                    if self._xor_mask_tensor is not None
                    else None
                ),
                laplacian_blend=self._blend_mode == "laplacian",
                max_levels=self._blend_levels,
                dtype=self._dtype,
            )

    def convert_mask_tensor(self, mask: torch.Tensor) -> torch.Tensor:
        # Mask should be the same size as our canvas
        padw: int = 0
        padh: int = 0
        mwidth: int = image_width(mask)
        mheight: int = image_height(mask)

        assert mwidth <= self._canvas_info.width
        assert mheight <= self._canvas_info.height

        if mwidth < self._canvas_info.width:
            padw = self._canvas_info.width - mwidth
        if mheight < self._canvas_info.height:
            padh = self._canvas_info.height - mheight

        if padw or padh:
            mask = torch.nn.functional.pad(
                mask.unsqueeze(0),
                [
                    0,
                    padw,
                    0,
                    padh,
                ],
                mode="replicate",
            ).squeeze(0)

        assert image_width(mask) == self._canvas_info.width
        assert image_height(mask) == self._canvas_info.height
        if not self._minimize_blend:
            return mask
        return mask[
            ...,
            self._canvas_info.positions[1].x
            - self._overlap_pad : self._remapper_1.width
            + self._overlap_pad,
        ]

    def draw(self, image: torch.Tensor) -> torch.Tensor:
        # left box
        image = my_draw_box(
            image,
            x1=None,
            y1=self._y1,
            x2=self._x2 + self._overlap_pad - 1,
            y2=self._remapper_1.height + self._y1 - 1,
            color=(255, 255, 0),
        )
        # right box
        image = my_draw_box(
            image,
            x1=self._x2 + self._overlapping_width - self._overlap_pad,
            y1=self._y2,
            x2=None,
            y2=self._remapper_2.height + self._y2 - 1,
            color=(0, 255, 255),
        )
        # Blended box
        image = my_draw_box(
            image,
            *self._padded_blended_tlbr,
            color=(255, 0, 0),
        )
        return image

    def forward(
        self, remapped_image_1: torch.Tensor, remapped_image_2: torch.Tensor
    ) -> torch.Tensor:

        # show_image("seam", self._seam_tensor, wait=False, enable_resizing=0.1)
        # show_image("xor", self._xor_mask_tensor, wait=False, enable_resizing=0.1)
        # print(torch.unique(self._xor_mask_tensor))

        alpha_mask_1 = self._remapper_1.unmapped_mask
        alpha_mask_2 = self._remapper_2.unmapped_mask

        # show_image("alpha_mask_1", alpha_mask_1, wait=False, scale=0.2)

        if self._minimize_blend:
            # assert image_width(remapped_image_1) == self._remapper_1.width  # sanity
            canvas = (
                torch.zeros(
                    size=(
                        remapped_image_1.shape[0],
                        remapped_image_1.shape[1],
                        self._canvas_info.height,
                        self._canvas_info.width,
                    ),
                    dtype=remapped_image_1.dtype,
                    device=remapped_image_1.device,
                )
                + self._empty_image_pixel_value
            )
            partial_1 = remapped_image_1[:, :, :, : self._x2 + self._overlap_pad]
            partial_2 = remapped_image_2[:, :, :, self._overlapping_width - self._overlap_pad :]

            assert remapped_image_1.shape[-2:] == alpha_mask_1.shape
            remapped_image_1 = remapped_image_1[
                :, :, :, self._x2 - self._overlap_pad :
            ]  # self._remapper_1.width
            alpha_mask_1 = alpha_mask_1[
                :,
                self._x2 - self._overlap_pad :,
                # self._remapper_1.width
            ]
            assert remapped_image_1.shape[-2:] == alpha_mask_1.shape

            assert remapped_image_2.shape[-2:] == alpha_mask_2.shape
            remapped_image_2 = remapped_image_2[
                :, :, :, : self._overlapping_width + self._overlap_pad
            ]
            alpha_mask_2 = alpha_mask_2[:, : self._overlapping_width + self._overlap_pad]
            assert remapped_image_2.shape[-2:] == alpha_mask_2.shape
        fwd_args = OrderedDict(
            image_1=remapped_image_1.to(self._dtype, non_blocking=True),
            image_2=remapped_image_2.to(self._dtype, non_blocking=True),
        )
        if self._use_python_blender:
            fwd_args.update(
                OrderedDict(
                    alpha_mask_1=alpha_mask_1,
                    alpha_mask_2=alpha_mask_2,
                )
            )
        else:
            fwd_args.update(
                OrderedDict(
                    xy_pos_1=[self._remapper_1.xpos, self._remapper_1.ypos],
                    xy_pos_2=[self._remapper_2.xpos, self._remapper_2.ypos],
                )
            )

        blended_img = self._blender.forward(**fwd_args)

        if self._minimize_blend:
            canvas[
                :,
                :,
                :,
                self._x2
                - self._overlap_pad : self._x2
                + self._overlapping_width
                + self._overlap_pad,
            ] = blended_img.clamp(min=0, max=255).to(dtype=canvas.dtype, non_blocking=True)
            canvas[
                :, :, self._y1 : self._remapper_1.height + self._y1, : self._x2 + self._overlap_pad
            ] = partial_1
            canvas[
                :,
                :,
                self._y2 : self._remapper_2.height + self._y2,
                self._x2 + self._overlapping_width - self._overlap_pad :,
            ] = partial_2
            blended = canvas
            if self._draw:
                blended = self.draw(blended)
        else:
            blended = blended_img

        return blended


@dataclass
class StitchImageInfo:
    image: torch.Tensor
    xy_pos: Tuple[int, int]


class ImageStitcher(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        remap_image_info: List[RemapImageInfoEx],
        blender_config: BlenderConfig,
        dtype: torch.dtype,
        channels: int = 3,
        minimize_blend: bool = True,
        overlap_pad: int = None,
        use_python_blender: bool = False,
        draw: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._remap_image_info = remap_image_info
        self._blender_config = blender_config
        self._dtype = dtype
        self._device = device

        if overlap_pad is None:
            overlap_pad = calculate_required_blend_overlap(blender_config.levels)

        self._remapper_1 = ImageRemapper(
            remap_info=remap_image_info[0],
            dtype=self._dtype,
            channels=channels,
            batch_size=batch_size,
            interpolation=self._blender_config.interpolation,
            use_cpp_remap_op=False,
            debug=False,
        )

        self._remapper_2 = ImageRemapper(
            remap_info=remap_image_info[1],
            dtype=self._dtype,
            channels=channels,
            batch_size=batch_size,
            interpolation=self._blender_config.interpolation,
            use_cpp_remap_op=False,
            debug=False,
        )

        self._smart_remapper_blender = SmartRemapperBlender(
            remapper_1=self._remapper_1,
            remapper_2=self._remapper_2,
            dtype=self._dtype,
            minimize_blend=minimize_blend,
            blend_levels=blender_config.levels,
            overlap_pad=overlap_pad,
            draw=draw,
            use_python_blender=use_python_blender,
            blend_mode=blender_config.mode,
            seam_tensor=self._blender_config.seam,
            device=self._device,
        )
        self.to(device=self._device)

    def to(self, *args, device: torch.device, **kwargs):
        assert isinstance(device, (torch.device, str))
        return super().to(device, **kwargs)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        remapped_tensor_1 = self._remapper_1.forward(source_tensor=inputs[0])
        remapped_tensor_2 = self._remapper_2.forward(source_tensor=inputs[1])
        return self._smart_remapper_blender.forward(remapped_tensor_1, remapped_tensor_2)


def get_mapping(dir_name: str, basename: str):
    x_file = os.path.join(dir_name, f"{basename}_x.tif")
    y_file = os.path.join(dir_name, f"{basename}_y.tif")
    xpos, ypos = get_image_geo_position(os.path.join(dir_name, f"{basename}.tif"))

    x_map = cv2.imread(x_file, cv2.IMREAD_ANYDEPTH)
    y_map = cv2.imread(y_file, cv2.IMREAD_ANYDEPTH)
    if x_map is None:
        raise AssertionError(f"Could not read mapping file: {x_file}")
    if y_map is None:
        raise AssertionError(f"Could not read mapping file: {y_file}")
    col_map = torch.from_numpy(x_map.astype(np.int64))
    row_map = torch.from_numpy(y_map.astype(np.int64))
    return xpos, ypos, col_map, row_map


def calculate_required_blend_overlap(blend_levels: int) -> int:
    """
    Calculate how much overlap we need for the given width of a
    remapped image (after remapping) and the number of levels
    that we intend to blend.
    The idea is that a pixel in the smallest pyramid image should
    be the padding amount when it is scaled up to the original image

    i.e.

       XXXXXXXX
    1: XXXXXXXX
       XXXXXXXX

       XXXX
    2: XXXX
       XXXX

    3: XX
       XX

    One pixel X in the third (2x2) imge represents 4 pixels in the original image

    Then, maybe we adjust it a little...
    """
    return 2**blend_levels * 2


def create_stitcher(
    dir_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    left_image_size_wh: Tuple[int, int],
    right_image_size_wh: Tuple[int, int],
    input_image_sizes_wh: Optional[List[Tuple[int, int]]] = None,
    python_blender: bool = True,
    minimize_blend: bool = True,
    max_output_width: Optional[int] = None,
    mapping_basename_1: str = "mapping_0000",
    mapping_basename_2: str = "mapping_0001",
    remapped_basename: str = "mapping_",
    blend_mode: str = "laplacian",
    interpolation: str = "bilinear",
    levels: int = 11,
    draw: bool = False,
    use_cuda_pano: bool = True,
    use_cuda_pano_n: bool = False,
):
    """Create an ImageStitcher or CUDA panorama stitcher from mapping files."""
    if use_cuda_pano:
        assert dir_name
        if input_image_sizes_wh is None:
            input_image_sizes_wh = [left_image_size_wh, right_image_size_wh]
        if len(input_image_sizes_wh) < 2:
            raise ValueError("Expected at least 2 input views for stitching")
        input_sizes = [WHDims(w, h) for (w, h) in input_image_sizes_wh]
        size1 = input_sizes[0]
        size2 = input_sizes[1]
        if blend_mode != "laplacian":
            # Hard seam
            levels = 0
        max_output_width_i = int(max_output_width) if max_output_width else 0
        if len(input_sizes) == 2 and not use_cuda_pano_n:
            if dtype == torch.float32:
                stitcher = CudaStitchPanoF32(
                    str(dir_name),
                    batch_size,
                    levels,
                    size1,
                    size2,
                    minimize_blend=minimize_blend,
                    max_output_width=max_output_width_i,
                )
            elif dtype == torch.uint8:
                stitcher = CudaStitchPanoU8(
                    str(dir_name),
                    batch_size,
                    levels,
                    size1,
                    size2,
                    minimize_blend=minimize_blend,
                    max_output_width=max_output_width_i,
                )
            else:
                raise ValueError(f"Unsupported dtype for cuda pano: {dtype}")
            return stitcher

        if python_blender:
            raise NotImplementedError("python_blender only supports 2 input views")

        if dtype == torch.float32:
            return CudaStitchPanoNF32(
                str(dir_name),
                batch_size,
                levels,
                input_sizes,
                minimize_blend=minimize_blend,
                quiet=False,
            )
        if dtype == torch.uint8:
            return CudaStitchPanoNU8(
                str(dir_name),
                batch_size,
                levels,
                input_sizes,
                minimize_blend=minimize_blend,
                quiet=False,
            )
        raise ValueError(f"Unsupported dtype for cuda pano N: {dtype}")

    blender_config: BlenderConfig = create_blender_config(
        mode=blend_mode,
        dir_name=dir_name,
        basename=remapped_basename,
        device=device,
        levels=levels,
        lazy_init=False,
        interpolation=interpolation,
    )

    xpos_1, ypos_1, col_map_1, row_map_1 = get_mapping(dir_name, mapping_basename_1)
    xpos_2, ypos_2, col_map_2, row_map_2 = get_mapping(dir_name, mapping_basename_2)

    remap_info_1 = RemapImageInfoEx()
    remap_info_1.src_width = int(left_image_size_wh[0])
    remap_info_1.src_height = int(left_image_size_wh[1])
    remap_info_1.col_map = col_map_1
    remap_info_1.row_map = row_map_1
    remap_info_1.xpos = xpos_1
    remap_info_1.ypos = ypos_1

    remap_info_2 = RemapImageInfoEx()
    remap_info_2.src_width = int(right_image_size_wh[0])
    remap_info_2.src_height = int(right_image_size_wh[1])
    remap_info_2.col_map = col_map_2
    remap_info_2.row_map = row_map_2
    remap_info_2.xpos = xpos_2
    remap_info_2.ypos = ypos_2

    stitcher = ImageStitcher(
        batch_size=batch_size,
        device=device,
        remap_image_info=[remap_info_1, remap_info_2],
        blender_config=blender_config,
        dtype=dtype,
        use_python_blender=python_blender,
        minimize_blend=minimize_blend,
        draw=draw,
    )
    if device is not None:
        stitcher = stitcher.to(device=device)
    return stitcher


def ensure_rgba(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensures that an input image tensor is in RGBA format.

    Supports both:
      - Single image: (C, H, W)
      - Batched images: (B, C, H, W)

    Adds an alpha channel with full opacity if tensor is RGB.

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)

    Returns:
        torch.Tensor: Image tensor with 4 channels (RGBA)
    """
    if tensor.ndim == 3:
        # Single image case
        tensor = tensor.unsqueeze(0)  # Convert to (1, C, H, W)
        squeezed = True
    elif tensor.ndim == 4:
        squeezed = False
    else:
        raise ValueError(f"Expected tensor of shape (C, H, W) or (B, C, H, W), got {tensor.shape}")

    B, C, H, W = tensor.shape
    if C == 4:
        return tensor[0] if squeezed else tensor  # Already RGBA
    elif C == 3:
        alpha = torch.ones((B, 1, H, W), dtype=tensor.dtype, device=tensor.device)
        if tensor.dtype == torch.uint8:
            alpha *= 255
        out = torch.cat([tensor, alpha], dim=1)
        return out[0] if squeezed else out
    else:
        raise ValueError(f"Expected 3 (RGB) or 4 (RGBA) channels, got {C}")


def blend_video(
    opts: object,
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    device: torch.device,
    dtype: torch.dtype,
    lfo: float = None,
    rfo: float = None,
    show: bool = False,
    show_scaled: float | None = None,
    start_frame_number: int = 0,
    output_video: str = None,
    max_width: int = 9999,
    batch_size: int = 8,
    skip_final_video_save: bool = False,
    blend_mode: str = "laplacian",
    queue_size: int = 1,
    minimize_blend: bool = True,
    add_alpha_channel: bool = False,
    overlap_pad: int = 120,
    draw: bool = False,
    use_cuda_pano: bool = False,
) -> None:
    """Blend two camera videos into a stitched panorama video.

    @param opts: Parsed CLI options (hm_opts).
    @param video_file_1: Left video path (or basename under ``dir_name``).
    @param video_file_2: Right video path.
    @param dir_name: Base directory for videos and mapping files.
    @param device: CUDA device used for blending.
    @param dtype: Torch dtype for processing (float/half/uint8).
    @param lfo: Optional left frame offset.
    @param rfo: Optional right frame offset.
    @param show: If True, display frames during processing.
    @param show_scaled: Optional scaling factor for display.
    @param start_frame_number: Start frame index for reading.
    @param output_video: Optional output video path; if None, no file is written.
    @param max_width: Clamp output width for display/encoding.
    @param batch_size: Number of frames processed per batch.
    @param skip_final_video_save: Skip final file save for VideoOutput.
    @param blend_mode: Blend mode ('laplacian' or 'hard-seam').
    @param queue_size: VideoOutput queue size.
    @param minimize_blend: If True, restrict blending to overlap region.
    @param add_alpha_channel: If True, include alpha channel in output.
    @param overlap_pad: Padding in pixels around seam overlap.
    @param draw: If True, draw debug boxes over blends.
    @param use_cuda_pano: If True, use CUDA panorama stitcher instead of Python.
    """
    if "/" not in video_file_1:
        video_file_1 = os.path.join(dir_name, video_file_1)
    if "/" not in video_file_2:
        video_file_2 = os.path.join(dir_name, video_file_2)

    vidinfo_1 = BasicVideoInfo(video_file_1)
    vidinfo_2 = BasicVideoInfo(video_file_2)

    max_blend_levels = getattr(opts, "max_blend_levels", None)

    max_frames = getattr(opts, "max_frames", None)
    if (max_frames in (None, 0)) and getattr(opts, "max_time", None):
        try:
            seconds = convert_hms_to_seconds(opts.max_time)
            if seconds > 0 and vidinfo_1.fps > 0:
                max_frames = int(seconds * vidinfo_1.fps)
        except Exception:
            max_frames = None

    if use_cuda_pano:
        size1 = WHDims(vidinfo_1.width, vidinfo_1.height)
        size2 = WHDims(vidinfo_2.width, vidinfo_2.height)
        if blend_mode == "laplacian":
            num_levels: int = (
                int(max_blend_levels)
                if max_blend_levels is not None and max_blend_levels > 0
                else 11
            )
        else:
            # Hard-seam or other non-laplacian GPU modes: force single level.
            num_levels = 0
        stitcher: CudaStitchPanoU8 = CudaStitchPanoU8(
            dir_name, batch_size, num_levels, size1, size2, minimize_blend
        )
        canvas_width = stitcher.canvas_width()
        canvas_height = stitcher.canvas_height()
    else:
        if blend_mode == "laplacian":
            levels_arg = (
                int(max_blend_levels)
                if max_blend_levels is not None and max_blend_levels > 0
                else 11
            )
        else:
            levels_arg = 0
        stitcher: ImageStitcher = create_stitcher(
            dir_name=dir_name,
            batch_size=batch_size,
            left_image_size_wh=(vidinfo_1.width, vidinfo_1.height),
            right_image_size_wh=(vidinfo_2.width, vidinfo_2.height),
            minimize_blend=minimize_blend,
            device=device,
            dtype=dtype,
            blend_mode=blend_mode,
            draw=draw,
            add_alpha_channel=add_alpha_channel,
            levels=levels_arg,
            use_cuda_pano=use_cuda_pano,
        )

    if lfo is None or rfo is None:
        lfo, rfo = synchronize_by_audio(video_file_1, video_file_2)

    stream = torch.cuda.current_stream(device)
    assert stream is not None

    cap_1 = VideoStreamReader(
        os.path.join(dir_name, video_file_1),
        type=opts.video_stream_decode_method,
        device=f"cuda:{gpu_index(want=2)}",
        batch_size=batch_size,
    )
    if not cap_1 or not cap_1.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_1}")
    if lfo or start_frame_number:
        cap_1.seek(frame_number=lfo + start_frame_number)

    cap_2 = VideoStreamReader(
        os.path.join(dir_name, video_file_2),
        type=opts.video_stream_decode_method,
        device=f"cuda:{gpu_index(want=3)}",
        batch_size=batch_size,
    )

    if not cap_2 or not cap_2.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_2}")
    if rfo or start_frame_number:
        cap_2.seek(frame_number=rfo + start_frame_number)

    v1_iter = iter(cap_1)
    v2_iter = iter(cap_2)

    source_tensor_1 = make_channels_first(next(v1_iter))
    source_tensor_2 = make_channels_first(next(v2_iter))

    video_out = None

    timer = Timer()
    frame_count = 0
    frame_id = start_frame_number
    try:
        while True:
            if isinstance(source_tensor_1, np.ndarray):
                source_tensor_1 = torch.from_numpy(source_tensor_1).to(device, non_blocking=True)
            if isinstance(source_tensor_2, np.ndarray):
                source_tensor_2 = torch.from_numpy(source_tensor_2).to(device, non_blocking=True)

            if use_cuda_pano:
                source_tensor_1 = make_channels_last(ensure_rgba(source_tensor_1)).contiguous()
                source_tensor_2 = make_channels_last(ensure_rgba(source_tensor_2)).contiguous()
                canvas_width = stitcher.canvas_width()
                canvas_height = stitcher.canvas_height()
                blended = torch.zeros(
                    [
                        batch_size,
                        canvas_height,
                        canvas_width,
                        source_tensor_1.shape[-1],
                    ],
                    dtype=source_tensor_1.dtype,
                    device=source_tensor_1.device,
                )
                stitcher.process(source_tensor_1, source_tensor_2, blended, stream.cuda_stream)
                torch.cuda.synchronize()
            else:
                blended = stitcher.forward(inputs=(source_tensor_1, source_tensor_2))

            if output_video:
                if video_out is None:
                    video_dim_height, video_dim_width = get_dims_for_output_video(
                        height=blended.shape[-2],
                        width=blended.shape[-1],
                        max_width=max_width,
                    )
                    fps = cap_1.fps
                    video_out = VideoOutput(
                        output_video_path=output_video,
                        fps=fps,
                        skip_final_save=skip_final_video_save,
                        fourcc="nvenc_av1",
                        # batch_size=batch_size,
                        cache_size=queue_size,
                        name="StitchedOutput",
                        device=blended.device,
                        show_image=bool(show),
                        show_scaled=show_scaled,
                        show_youtube=bool(opts.show_youtube),
                        youtube_stream_url=opts.youtube_stream_url,
                        youtube_stream_key=opts.youtube_stream_key,
                        headless_preview_host=opts.headless_preview_host or "0.0.0.0",
                        headless_preview_port=int(opts.headless_preview_port or 0),
                    )

            if video_out is not None:
                # VideoOutput is a nn.Module; calling it writes frames synchronously.
                video_out(
                    {
                        "frame_id": torch.tensor(frame_id, dtype=torch.int64),
                        "img": blended,
                        "current_box": None,
                    }
                )
                frame_id += len(blended)

            frame_id += 1
            frame_count += 1

            torch.cuda.synchronize()

            if frame_count != 1:
                timer.toc()

            if frame_count % 20 == 0:
                print(
                    "Stitching: {:.2f} fps".format(batch_size * 1.0 / max(1e-5, timer.average_time))
                )
                if frame_count % 50 == 0:
                    timer = Timer()

            if max_frames is not None and frame_count >= max_frames:
                break

            if show:
                for this_blended in blended:
                    show_image(
                        "this_blended",
                        this_blended if use_cuda_pano else this_blended.cpu(),
                        wait=False,
                        enable_resizing=show_scaled,
                    )

            source_tensor_1 = make_channels_first(next(v1_iter))
            source_tensor_2 = make_channels_first(next(v2_iter))
            timer.tic()
            del blended

    except StopIteration:
        # All done.
        pass
    finally:
        if video_out is not None:
            if isinstance(video_out, VideoStreamWriter):
                video_out.flush()
                video_out.close()
            else:
                video_out.stop()


def gpu_index(want: int = 1):
    return min(torch.cuda.device_count() - 1, want)


#
# Combined FPS=XY/(X+Y)
#


def main(args):
    """Entry point for the `blender2.py` stitching CLI."""
    opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    gpu_allocator = GpuAllocator(gpus=args.gpus)
    if not args.video_dir and args.game_id:
        args.video_dir = os.path.join(os.environ["HOME"], "Videos", args.game_id)
    fast_gpu = torch.device("cuda", gpu_allocator.allocate_fast())

    file_list = []
    if args.files:
        file_list = args.files.split(",")
    if not file_list:
        game_videos = configure_game_videos(
            game_id=args.game_id,
            write_results=True,
            force=False,
            inference_scale=getattr(args, "ice_rink_inference_scale", None),
        )
        if "left" in game_videos and game_videos["left"]:
            game_videos["left"] = game_videos["left"][:1][0]
        if "right" in game_videos and game_videos["right"]:
            game_videos["right"] = game_videos["right"][:1][0]
        file_list = [game_videos["left"], game_videos["right"]]

    assert len(file_list) == 2

    HalfFloatType = torch.float16
    # HalfFloatType = torch.bfloat16

    if args.fp16:
        torch.set_default_dtype(HalfFloatType)

    with torch.no_grad():
        blend_video(
            opts,
            video_file_1=file_list[0],
            video_file_2=file_list[1],
            dir_name=args.video_dir,
            lfo=args.lfo,
            rfo=args.rfo,
            start_frame_number=args.start_frame_number,
            show=args.show_image,
            show_scaled=args.show_scaled,
            output_video=args.output_file,
            batch_size=args.batch_size,
            skip_final_video_save=args.skip_final_video_save,
            queue_size=args.queue_size,
            device=fast_gpu,
            dtype=HalfFloatType if args.fp16 else torch.float,
            draw=args.draw,
            minimize_blend=preferred_arg(args.minimize_blend, True),
            blend_mode=args.blend_mode,
            use_cuda_pano=not args.python_blender,
        )


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args = hm_opts.init(args, parser)
    try:
        main(args)
        print("Done.")
    except Exception:
        traceback.print_exc()
