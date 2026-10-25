"""Discover and classify left/right camera videos for a given game.

This module scans game directories for vendor-specific or left/right
clips, runs ice-rink segmentation to infer orientation, and writes the
final left/right chapter lists into the game private config.
"""

import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from hmlib.config import (
    get_game_config_private,
    get_game_dir,
    get_nested_value,
    prepend_root_dir,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.log import get_logger
from hmlib.models.loader import get_model_config
from hmlib.segm.ice_rink import find_ice_rink_masks
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.image import image_height, image_width
from hmlib.utils.video import load_first_video_frame

# GoPro pattern is GXzzxxxx.mp4, where zz is chapter number and zzzz is video
# number
GOPRO_FILE_PATTERN: str = r"^G[A-Z][0-9]{6}\.(MP4|mp4)$"
INSTA360_FILE_PATTERN: str = r"^VID_[0-9]{8}_[0-9]{6}_[0-9]{3}\.(MP4|mp4)$"
VIDEO_EXT_PRIORITY: tuple[str, ...] = ("mp4", "mkv", "m4v")
LEFT_PART_FILE_PATTERN: str = r"left-[0-9]\.{ext}$"
RIGHT_PART_FILE_PATTERN: str = r"right-[0-9]\.{ext}$"
LEFT_FILE_PATTERN: str = r"left\.{ext}$"
RIGHT_FILE_PATTERN: str = r"right\.{ext}$"


VideoChapter = dict[int | str, Any]
VideosDictKey = int | str
VideosDict = dict[VideosDictKey, VideoChapter]


logger = get_logger(__name__)


def gopro_get_video_and_chapter(filename: Path) -> tuple[int, int]:
    """Return (video number, chapter number) from a GoPro filename."""
    name = Path(filename).stem
    assert name[0] == "G"
    assert name[1] in {"H", "X"}
    return int(name[4:8]), int(name[2:4])


def insta360_get_video_and_chapter(filename: Path) -> tuple[int, int]:
    """Return (video identifier, chapter number) for Insta360 clips."""
    name = Path(filename).stem
    tokens = name.split("_")
    assert tokens[0] == "VID"
    assert len(tokens) >= 4
    date_token, video_token, chapter_token = tokens[1], tokens[2], tokens[3]
    video_id = int(f"{date_token}{video_token}")
    return video_id, int(chapter_token)


def get_lr_part_number(filename: str) -> int:
    """Extract the numeric part index from a left/right-part filename."""
    name = Path(filename).stem
    tokens = name.split("-")
    return int(tokens[-1])


def find_matching_files(re_pattern: str, directory: str) -> list[str]:
    """Return sorted list of filenames in ``directory`` matching ``re_pattern``."""
    # Regex to match the file format 'GLXXXXXX.mp4'
    pattern = re.compile(re_pattern)

    # List to store the names of matching files
    matching_files: list[str] = []

    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        if pattern.match(filename):
            matching_files.append(os.path.join(directory, filename))

    return sorted(matching_files)


def prune_chapters(videos: VideosDict) -> tuple[VideosDict, VideosDict]:
    """Prune out videos that don't have matching chapters.

    @return: Tuple (kept_videos, discarded_videos).
    """
    for video, chapters in videos.items():
        # Not pruning anything yet
        pass
    return videos, {}


def _find_vendor_chapter_pairs(directory: str) -> list[tuple[tuple[int, int], str]]:
    """Scan a directory for known vendor MP4 chapter files and return ordered pairs.

    Returns a list of ((video_id, chapter_num), full_path) sorted by (video_id, chapter_num).
    Supports GoPro and Insta360 naming conventions.
    """
    pairs: list[tuple[tuple[int, int], str]] = []

    # GoPro
    for f in find_matching_files(re_pattern=GOPRO_FILE_PATTERN, directory=directory):
        vid, ch = gopro_get_video_and_chapter(f)
        pairs.append(((vid, ch), f))

    # Insta360
    for f in find_matching_files(re_pattern=INSTA360_FILE_PATTERN, directory=directory):
        vid, ch = insta360_get_video_and_chapter(f)
        pairs.append(((vid, ch), f))

    pairs.sort(key=lambda x: x[0])
    return pairs


def _pairs_to_linear_chapter_map(pairs: list[tuple[tuple[int, int], str]]) -> VideoChapter:
    """Flatten vendor chapter pairs into a linear 1..N->file mapping."""
    return {i + 1: f for i, ((_, _), f) in enumerate(pairs)}


def _collect_lr_chapters(directory: str, side: str, renumber: bool = False) -> VideoChapter | None:
    """Collect left/right single or part files into a chapter map.

    If renumber=True, chapters are returned as 1..N in file order; otherwise
    they retain the part number from the filename (e.g., left-3.mp4 -> 3).
    """
    side = side.lower()
    assert side in {"left", "right"}

    if side == "left":
        single_pat_base, parts_pat_base = LEFT_FILE_PATTERN, LEFT_PART_FILE_PATTERN
    else:
        single_pat_base, parts_pat_base = RIGHT_FILE_PATTERN, RIGHT_PART_FILE_PATTERN

    for ext in VIDEO_EXT_PRIORITY:
        single_pat = single_pat_base.format(ext=ext)
        files = find_matching_files(re_pattern=single_pat, directory=directory)
        if files:
            return {1: files[0]}

    for ext in VIDEO_EXT_PRIORITY:
        parts_pat = parts_pat_base.format(ext=ext)
        files = find_matching_files(re_pattern=parts_pat, directory=directory)
        if files:
            files = sorted(files, key=lambda f: get_lr_part_number(f))
            if renumber:
                return {i + 1: f for i, f in enumerate(files)}
            else:
                return {get_lr_part_number(f): f for f in files}

    return None


def get_available_videos(dir_name: str, prune: bool = False) -> VideosDict:
    """Discover available videos and chapters under a game directory.

    @param dir_name: Game directory path.
    @param prune: If True, drop videos with incomplete chapter sets.
    @return: Mapping from camera key to chapter-number->filename.
    """
    # Prefer camera-specific subdirectories cam1, cam2, ... when present.
    # This avoids filename collisions when multiple cameras produce the same
    # chapter filenames. Each camX directory is treated as a separate camera and
    # its files are collected as the chapter list for that camera key.
    cam_dirs = [
        d
        for d in os.listdir(dir_name)
        if os.path.isdir(os.path.join(dir_name, d)) and re.match(r"^cam\d+$", d, re.IGNORECASE)
    ]

    def _sorted_cam_dirs(names: list[str]) -> list[str]:
        def cam_index(name: str) -> int:
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else 0

        return sorted(names, key=cam_index)

    def _collect_chapters_for_dir(d: str) -> VideoChapter:
        """Collect and order chapter files inside a directory d.

        Strategy: prefer vendor-specific patterns (GoPro, Insta360) if found;
        otherwise fall back to left*/right* patterns. Output keys are 1..N in
        the discovered order (synthetic), so downstream expects consecutive ints.
        """
        pairs = _find_vendor_chapter_pairs(d)
        if pairs:
            return _pairs_to_linear_chapter_map(pairs)
        # Plain left/right single file or parts
        left_map = _collect_lr_chapters(d, side="left", renumber=True)
        if left_map:
            return left_map
        right_map = _collect_lr_chapters(d, side="right", renumber=True)
        if right_map:
            return right_map
        return {}

    if cam_dirs:
        videos_dict: VideosDict = OrderedDict()
        for cam_name in _sorted_cam_dirs(cam_dirs):
            cam_path = os.path.join(dir_name, cam_name)
            chapter_map = _collect_chapters_for_dir(cam_path)
            if chapter_map:
                videos_dict[cam_name] = chapter_map
        if prune:
            videos_dict, discarded_videos = prune_chapters(videos=videos_dict)
            if discarded_videos:
                logger.info("Discarding videos: %s", discarded_videos)
        if videos_dict:
            return videos_dict

    # Fallback: scan the root directory as before
    # Vendor-specific files (merge by video-id)
    videos_dict: VideosDict = OrderedDict()
    for (vid, ch), f in _find_vendor_chapter_pairs(dir_name):
        videos_dict.setdefault(vid, {})[ch] = f

    # Plain left/right
    left_map = _collect_lr_chapters(dir_name, side="left", renumber=False)
    if left_map:
        videos_dict["left"] = left_map
    right_map = _collect_lr_chapters(dir_name, side="right", renumber=False)
    if right_map:
        videos_dict["right"] = right_map
    if prune:
        videos_dict, discarded_videos = prune_chapters(videos=videos_dict)
        if discarded_videos:
            logger.info("Discarding videos: %s", discarded_videos)
    return videos_dict


def detect_video_rink_masks(
    game_id: str,
    videos_dict: VideosDict,
    device: torch.device = None,
    inference_scale: float | None = None,
) -> VideosDict:
    """Run ice-rink segmentation on the first frame of each video."""
    if device is None:
        gpu_allocator = GpuAllocator(gpus=None)
        device: torch.device = (
            torch.device("cuda", gpu_allocator.allocate_fast())
            if not gpu_allocator.is_single_lowmem_gpu(low_threshold_mb=1024 * 10)
            else torch.device("cpu")
        )

    keys = []
    images: list[torch.Tensor] = []
    for key, value in videos_dict.items():
        keys.append(key)
        min_chapter_key = min(value, key=value.get)
        frame = load_first_video_frame(value[min_chapter_key])
        images.append(frame)
        value["first_frame"] = load_first_video_frame(value[min_chapter_key])

    config_file, checkpoint = get_model_config(game_id=game_id, model_name="ice_rink_segm")

    frame_ir_masks = find_ice_rink_masks(
        image=images,
        config_file=prepend_root_dir(config_file),
        checkpoint=prepend_root_dir(checkpoint),
        device=device,
        inference_scale=inference_scale,
    )
    assert len(frame_ir_masks) == len(keys)
    for i, key in enumerate(keys):
        videos_dict[key]["rink_profile"] = frame_ir_masks[i]

    return videos_dict


def get_orientation_dict(rink_mask: torch.Tensor) -> dict[str, float]:
    """Summarize rink occupancy on each edge of the mask."""
    assert rink_mask.dtype == torch.bool
    assert rink_mask.ndim == 2
    width = image_width(rink_mask)
    height = image_height(rink_mask)

    float_mask = rink_mask.to(torch.float)
    divisor = 8
    left_sum = float_mask[:, : int(width // divisor)].sum().item()
    right_sum = float_mask[:, int(width - width // divisor) :].sum().item()
    top_sum = float_mask[: int(height // divisor), :].sum().item()
    bottom_sum = float_mask[int(height // divisor) :, :].sum().item()
    return {
        "left": left_sum,
        "right": right_sum,
        "top": top_sum,
        "bottom": bottom_sum,
    }


def get_orientation(rink_mask: torch.Tensor) -> str:
    """Classify a rink mask as 'left', 'right' or 'UNKNOWN'."""
    assert rink_mask.dtype == torch.bool
    assert rink_mask.ndim == 2
    width = image_width(rink_mask)

    float_mask = rink_mask.to(torch.float)
    divisor = 8
    left_sum = float_mask[:, : int(width // divisor)].sum().item()
    right_sum = float_mask[:, int(width - width // divisor) :].sum().item()

    # show_image("mask", float_mask * 255)

    if left_sum > right_sum:
        # Most ice on the left of image, so this is the right side
        return "right"
    if right_sum > left_sum:
        # Most ice on the right of image, so this is the left side
        return "left"

    # For right end-zone, most ice will be at the bottom and left
    # height = image_height(rink_mask)
    # top_sum = float_mask[: int(height // divisor), :].sum()
    # bottom_sum = float_mask[int(height // divisor) :, :].sum()

    return "UNKNOWN"


def get_game_videos_analysis(
    game_id: str,
    videos_dict: VideosDict | None = None,
    device: torch.device = None,
    inference_scale: float | None = None,
) -> VideosDict:
    """Analyze game videos and attach orientation metadata per camera."""
    if videos_dict is None:
        videos_dict = get_available_videos(dir_name=get_game_dir(game_id=game_id))
    videos_dict = detect_video_rink_masks(
        game_id=game_id,
        videos_dict=videos_dict,
        device=device,
        inference_scale=inference_scale,
    )

    new_dict: VideosDict = {}

    video_ssm: dict[VideosDictKey, dict[str, float]] = {}

    for key, value in videos_dict.items():
        mask = value["rink_profile"]["combined_mask"]

        sum_map: dict[str, float] = get_orientation_dict(mask)
        video_ssm[key] = sum_map

        orientation = get_orientation(mask)
        if isinstance(key, str):
            if key.startswith("left"):
                assert orientation == "left"
            elif key.startswith("right"):
                assert orientation == "right"
        if orientation in videos_dict:
            if isinstance(key, str):
                assert key.startswith(orientation)
        else:
            new_dict[orientation] = value
        logger.info("%s orientation: %s", key, orientation)
    videos_dict.update(new_dict)
    return videos_dict


def extract_chapters_file_list(chapter_map: VideoChapter) -> list[str]:
    numeric_keys = sorted(key for key in chapter_map if isinstance(key, (int, float)))
    if not numeric_keys:
        return []

    start = int(numeric_keys[0])
    expected = list(range(start, start + len(numeric_keys)))
    actual = [int(key) for key in numeric_keys]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"Non-consecutive chapter numbers: present={actual}, missing={missing}")

    return [chapter_map[index] for index in expected]


def configure_game_videos(
    game_id: str,
    force: bool = False,
    write_results: bool = True,
    device: torch.device = None,
    inference_scale: float | None = None,
) -> dict[str, list[str]]:
    """Return and optionally persist left/right ordered chapter lists.

    @param game_id: Game identifier.
    @param force: If True, recompute even if config already has lists.
    @param write_results: When True, write relative paths to private config.
    @param device: Optional device for rink segmentation.
    @param inference_scale: Optional downscale factor for segmentation.
    @return: Dict with keys ``\"left\"`` and ``\"right\"`` mapping to lists of files.
    """
    dir_name = get_game_dir(game_id=game_id)
    private_config = get_game_config_private(game_id=game_id)
    if not force:
        declared = get_nested_value(private_config, "game.videos", {})
        if not isinstance(declared, dict):
            raise TypeError("game.videos must be a mapping")
        if declared.get("left") is not None or declared.get("right") is not None:
            for role in ("left", "right"):
                if not isinstance(declared.get(role), list) or not declared[role]:
                    raise ValueError("Import both left and right recordings before stitching")
            return {
                role: [os.path.join(dir_name, path) for path in declared[role]]
                for role in ("left", "right")
            }
    videos_dict = get_available_videos(dir_name=dir_name)
    if not force and ("left" in videos_dict and "right" in videos_dict):
        return {
            "left": extract_chapters_file_list(videos_dict["left"]),
            "right": extract_chapters_file_list(videos_dict["right"]),
        }
    videos_dict = get_game_videos_analysis(
        game_id=game_id,
        device=device,
        inference_scale=inference_scale,
    )
    left_list = extract_chapters_file_list(videos_dict["left"])
    right_list = extract_chapters_file_list(videos_dict["right"])
    if write_results:
        # Persist relative paths so subdirectories (e.g., cam1/, cam2/) are preserved
        rel_left = [os.path.relpath(p, start=dir_name) for p in left_list]
        rel_right = [os.path.relpath(p, start=dir_name) for p in right_list]
        set_nested_value(private_config, "game.videos.left", rel_left)
        set_nested_value(private_config, "game.videos.right", rel_right)
        save_private_config(game_id=game_id, data=private_config)
    return {
        "left": left_list,
        "right": right_list,
    }


def _main(args: argparse.Namespace):
    game_id = args.game_id
    assert game_id
    results = configure_game_videos(
        game_id=game_id,
        inference_scale=getattr(args, "ice_rink_inference_scale", None),
    )
    # videos_dict = get_game_videos_analysis(game_id=game_id)
    logger.info("Configured game videos: %s", results)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Video Orientation Analysis")
    parser.add_argument("--force", action="store_true", help="Force all recalculations")
    return parser


def main():
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()
    args = hm_opts.init(args, parser)

    _main(args)


if __name__ == "__main__":
    main()
    logger.info("Done.")
