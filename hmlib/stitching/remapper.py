"""CLI entry point for remapping video frames using stitching mapping TIFFs."""

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import torch

import hockeymon.core as core
from hmlib.log import get_root_logger
from hmlib.stitching.configure_stitching import get_image_geo_position
from hmlib.stitching.image_remapper import ImageRemapper
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import show_image
from hmlib.utils.image import make_channels_first
from hmlib.video.video_stream import VideoStreamReader

ROOT_DIR = os.getcwd()

logger = get_root_logger()


def make_parser():
    """Build an :class:`argparse.ArgumentParser` for the remapper CLI."""
    parser = argparse.ArgumentParser("Image Remapper")
    return parser


def read_frame_batch(
    video_iter,
    batch_size: int,
):
    frame_list = []
    frame = next(video_iter)
    assert frame.ndim == 4  # Must have batch dimension
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
        frame = make_channels_first(frame)
    if batch_size == 1:
        return frame
    frame_list.append(frame)
    for i in range(batch_size - 1):
        frame = next(video_iter)
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
            frame = make_channels_first(frame)
        frame_list.append(frame)
    tensor = torch.cat(frame_list, dim=0)
    return tensor


def create_remapper_config(
    dir_name: str,
    basename: str,
    image_index: int,
    batch_size: int,
    source_hw: Tuple[int],
    device: str,
    interpolation: str = "bilinear",
) -> core.RemapperConfig:
    """Build a :class:`core.RemapperConfig` from mapping TIFFs and image size."""
    x_file = os.path.join(dir_name, f"{basename}000{image_index}_x.tif")
    y_file = os.path.join(dir_name, f"{basename}000{image_index}_y.tif")
    x_map = cv2.imread(x_file, cv2.IMREAD_ANYDEPTH)
    y_map = cv2.imread(y_file, cv2.IMREAD_ANYDEPTH)
    if x_map is None:
        raise AssertionError(f"Could not read mapping file: {x_file}")
    if y_map is None:
        raise AssertionError(f"Could not read mapping file: {y_file}")
    config = core.RemapperConfig()
    config.src_height = source_hw[0]
    config.src_width = source_hw[1]
    config.x_pos, config.y_pos = get_image_geo_position(
        os.path.join(dir_name, f"{basename}000{image_index}.tif")
    )
    config.device = str(device)
    config.col_map = torch.from_numpy(x_map.astype(np.int64))
    config.row_map = torch.from_numpy(y_map.astype(np.int64))
    config.batch_size = batch_size
    config.interpolation = interpolation
    return config


def remap_video(
    opts: argparse.Namespace,
    video_file: str,
    dir_name: str,
    basename: str,
    interpolation: str = None,
    show: bool = False,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
):
    """Stream a video, remap each batch of frames, and optionally preview."""
    cap = VideoStreamReader(os.path.join(dir_name, video_file), type="cv2")
    if not cap or not cap.isOpened():
        raise AssertionError(f"Could not open video file: {os.path.join(dir_name, video_file)}")
    video_iter = iter(cap)
    source_tensor = read_frame_batch(video_iter, batch_size=batch_size)
    source_tensor = source_tensor.to(device)

    remapper = ImageRemapper(
        dir_name=dir_name,
        basename=basename,
        source_hw=source_tensor.shape[-2:],
        channels=source_tensor.shape[1],
        interpolation=interpolation,
        batch_size=batch_size,
        use_cpp_remap_op=False,
        dtype=torch.float16,
    )
    remapper.to(device=device)

    timer = Timer()
    frame_count = 0
    while True:
        with torch.no_grad():
            destination_tensor = remapper(source_tensor)
            destination_tensor = destination_tensor.detach().contiguous().cpu()

        frame_count += 1
        if frame_count != 1:
            timer.toc()

        if frame_count % 20 == 0:
            logger.info(
                "Remapping: {:.2f} fps".format(batch_size * 1.0 / max(1e-5, timer.average_time))
            )
            if frame_count % 50 == 0:
                timer = Timer()

        if show:
            for this_image in destination_tensor:
                show_image("mapped image", this_image, enable_resizing=opts.show_scaled)

        source_tensor = read_frame_batch(video_iter, batch_size=batch_size)
        source_tensor = source_tensor.to(device, non_blocking=True)
        timer.tic()


def main(args) -> None:
    """Main entry point for the `remapper` CLI."""
    remap_video(
        args,
        "GX010100.MP4",
        args.video_dir,
        "mapping_0000",
        interpolation="bilinear",
        show=False,
        device=torch.device("cuda", 0),
    )


if __name__ == "__main__":
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(make_parser())
    args = parser.parse_args()
    args = hm_opts.init(args, parser)

    if not args.video_dir:
        args.video_dir = "/mnt/ripper-data/Videos/ev-stockton-ss"

    main(args)
    print("Done.")
