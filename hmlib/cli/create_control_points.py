#!/usr/bin/env python3
"""
This script synchronizes two videos using audio cross-correlation, extracts the corresponding frames,
computes control points with a selectable matcher and updates a Hugin
PTO file with the newly computed control points.
"""

import argparse
import os
import re
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple, Union

import cv2
import ffmpegio
import numpy as np
import scipy.signal
import torch

from hmlib.config import get_game_config, get_game_dir
from hmlib.stitching.akaze import LensCalibrationPair, load_lens_calibration
from hmlib.stitching.configure_stitching import build_stitching_project, configure_video_stitching
from hmlib.stitching.control_points import CONTROL_POINT_MATCHERS
from hmlib.stitching.control_points import (
    calculate_control_points as calculate_stitching_control_points,
)
from hmlib.stitching.settings import (
    MAPPING_BACKENDS,
    StitchingSettings,
    read_stitching_settings,
    validate_output_scale,
)
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import time_to_frame

# Constant marker used in PTO files to denote control points.
_CONTROL_POINTS_LINE = "# control points"


def _game_dir_for_id(game_id: str) -> str:
    game_dir = get_game_dir(game_id=game_id, assert_exists=False)
    if game_dir is not None:
        return game_dir
    base_dir = os.environ.get("HM_GAME_DIR") or os.path.join(os.environ["HOME"], "Videos")
    return str(Path(base_dir) / game_id)


def load_audio_as_tensor(
    audio: Union[str, np.ndarray, torch.Tensor],
    duration_seconds: float,
    verbose: Optional[bool] = False,
) -> Tuple[torch.Tensor, float]:
    """
    Load audio from a file (or other supported source) using ffmpegio and return it as a PyTorch tensor.

    Args:
        audio: Either a file path or an array/tensor representing audio.
        duration_seconds: Duration (in seconds) to read from the audio.
        verbose: If True, prints additional debug information.

    Returns:
        A tuple (waveform, sample_rate) where waveform is a tensor of shape [channels, samples]
        and sample_rate is the number of samples per second.
    """
    sample_rate, waveform = ffmpegio.audio.read(audio, t=duration_seconds, show_log=True)
    if verbose:
        # waveform shape: [channels, samples]
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate}")
    return waveform, sample_rate


def get_video_fps_and_duration(video_path: str) -> Tuple[float, float]:
    """
    Retrieve the frames-per-second (FPS) and duration (in seconds) of a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        A tuple (fps, duration) where duration is computed as frame_count / fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        exit(1)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count / fps


def synchronize_by_audio(
    file1_path: str,
    file2_path: str,
    seconds: int = 15,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Synchronize two video files by comparing their audio tracks using cross-correlation.

    The function extracts a short audio clip from each video, computes their cross-correlation,
    and calculates the frame offset between the two videos.

    Args:
        file1_path: Path to the first video file.
        file2_path: Path to the second video file.
        seconds: Duration (in seconds) of audio to use for synchronization.
        verbose: If True, prints progress messages.

    Returns:
        A tuple (left_frame_offset, right_frame_offset) representing the number of frames to
        skip in each video so that they are synchronized. The offsets are returned as integers.
    """

    if verbose:
        print("Opening videos...")

    # Get video FPS and duration for both videos.
    video1_fps, video1_duration = get_video_fps_and_duration(file1_path)
    video2_fps, video2_duration = get_video_fps_and_duration(file2_path)

    # Ensure we do not exceed the available duration (leaving a 0.5 sec margin).
    seconds = min(seconds, min(video1_duration - 0.5, video2_duration - 0.5))

    video_1_subclip_frame_count: float = video1_fps * seconds
    video_2_subclip_frame_count: float = video2_fps * seconds

    if verbose:
        print("Loading audio...")

    # Load audio as tensor. The waveform is of shape [channels, samples].
    audio1, sample_rate1 = load_audio_as_tensor(file1_path, duration_seconds=seconds)
    audio2, sample_rate2 = load_audio_as_tensor(file2_path, duration_seconds=seconds)

    # Calculate number of audio samples per video frame.
    # Note: waveform shape is [channels, samples] so we use axis 1 for number of samples.
    audio_items_per_frame_1: float = audio1.shape[0] / video_1_subclip_frame_count
    audio_items_per_frame_2: float = audio2.shape[0] / video_2_subclip_frame_count

    # Check that the computed samples per frame match the expected value.
    assert np.isclose(sample_rate1 / video1_fps, audio_items_per_frame_1)
    assert np.isclose(sample_rate2 / video2_fps, audio_items_per_frame_2)

    if verbose:
        print("Calculating cross-correlation...")

    # Use only the first channel for correlation.
    correlation: np.ndarray = scipy.signal.correlate(audio1[:, 0], audio2[:, 0], mode="full")
    # Compute lag: subtract the length of the signal (using axis 1 length)
    lag: int = np.argmax(correlation) - audio1.shape[0] + 1

    # Convert lag (in audio samples) to frame offset.
    fps = video1_fps
    frame_offset: float = lag / audio_items_per_frame_1
    time_offset: float = frame_offset / fps

    if verbose:
        print(f"Calculated frame offset: {frame_offset}")
        print(f"Equivalent time offset: {time_offset} seconds")

    # Determine starting frame for each video.
    left_frame_offset: float = frame_offset if frame_offset > 0 else 0
    right_frame_offset: float = -frame_offset if frame_offset < 0 else 0

    return left_frame_offset, right_frame_offset


def extract_frame(video_path: str, frame_idx: Optional[float]) -> np.ndarray:
    """
    Extract a single frame from a video file using OpenCV.

    Args:
        video_path: Path to the video file.
        frame_idx: Index of the frame to extract.

    Returns:
        The extracted frame as a NumPy array (BGR format).

    Raises:
        ValueError: If the frame cannot be extracted.
    """
    if Path(video_path).suffix.lower() == ".png":
        frame = cv2.imread(video_path)
        if frame is None:
            raise ValueError(f"Could not read calibration image: {video_path}")
        return frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame {frame_idx} from {video_path}")
    return frame


def evenly_spaced_indices(n_points: int, n_samples: int) -> torch.Tensor:
    """
    Generate indices to pick n_samples evenly spaced points from a total of n_points.

    Args:
        n_points: Total number of available points.
        n_samples: Number of indices to select.

    Returns:
        A torch.Tensor of selected indices.
    """
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Select a subset of keypoints that are evenly spaced along the Y axis.

    Args:
        batch: A tensor of shape (N, 2) containing (X, Y) coordinates of keypoints.
        n_samples: Number of keypoints to select.

    Returns:
        A tensor of indices corresponding to the selected keypoints.
    """
    # Sort the keypoints based on the Y coordinate.
    _, sorted_indices = torch.sort(batch[:, 1])
    # Compute evenly spaced indices over the sorted keypoints.
    sample_indices: torch.Tensor = evenly_spaced_indices(batch.size(0), n_samples)
    # Map back to original indices.
    selected_indices: torch.Tensor = sorted_indices[sample_indices]
    return selected_indices


def calculate_control_points(
    frame0: np.ndarray,
    frame1: np.ndarray,
    max_control_points: int,
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    output_directory: Optional[str] = None,
    matcher: str = "superpoint-lightglue",
    lens_calibration: Optional[LensCalibrationPair] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute control points between two frames with the selected matcher.

    Args:
        frame0: First input frame (BGR NumPy array).
        frame1: Second input frame (BGR NumPy array).
        max_control_points: Maximum number of control point matches to return.
        device: Torch device to perform computation on (defaults to CUDA if available).
        max_num_keypoints: Maximum number of keypoints to extract.
        output_directory: Directory where visualizations (matches and keypoints) will be saved.
                          If None, no visual output is generated.

    Returns:
        A dictionary with keys "m_kpts0" and "m_kpts1" containing the matched keypoints as torch.Tensors.
    """
    return calculate_stitching_control_points(
        frame0,
        frame1,
        max_control_points=max_control_points,
        device=device,
        max_num_keypoints=max_num_keypoints,
        output_directory=output_directory,
        matcher=matcher,
        lens_calibration=lens_calibration,
    )


def load_pto_file(file_path: str) -> List[str]:
    """
    Load the contents of a Hugin PTO file.

    Args:
        file_path: Path to the PTO file.

    Returns:
        A list of strings representing the lines in the file (with trailing whitespace removed).
    """
    with open(file_path, "r") as file:
        lines: List[str] = file.readlines()
    # Remove trailing whitespace from each line.
    lines = [line.rstrip() for line in lines]
    return lines


def save_pto_file(file_path: str, data: List[str]) -> None:
    """
    Save a list of lines back into a PTO file.

    Args:
        file_path: Path to the PTO file.
        data: List of lines to write.
    """
    with open(file_path, "w") as file:
        for line in data:
            file.write(f"{line}\n")


def remove_control_points(lines: List[str]) -> Tuple[List[str], int]:
    """
    Remove existing control point lines (lines starting with "c ") from a PTO file content.

    Args:
        lines: List of strings representing the PTO file lines.

    Returns:
        A tuple (new_lines, count) where new_lines is the list without control point lines,
        and count is the number of control point lines removed.
    """
    prev_control_point_count: int = 0
    new_lines: List[str] = []
    for line in lines:
        if line.startswith(_CONTROL_POINTS_LINE):
            continue
        if line.startswith("c "):
            prev_control_point_count += 1
            continue
        new_lines.append(line)
    return new_lines, prev_control_point_count


def is_older_than(file1: str, file2: str) -> Optional[bool]:
    """
    Compare the modification times of two files.

    Args:
        file1: Path to the first file.
        file2: Path to the second file.

    Returns:
        True if file2 is older than file1, False if not, or None if there is an error.
    """
    try:
        mtime1 = os.path.getmtime(file1)
        mtime2 = os.path.getmtime(file2)
        return mtime2 < mtime1
    except OSError:
        return None


def strip(s: str) -> str:
    """
    Remove all whitespace from a string.

    Args:
        s: Input string.

    Returns:
        The string with all whitespace removed.
    """
    return re.sub(r"\s+", "", s)


def update_pto_file(pto_file: str, control_points: Dict[str, torch.Tensor]) -> None:
    """
    Update a Hugin PTO file by replacing existing control points with new ones.

    Args:
        pto_file: Path to the PTO file.
        control_points: Dictionary containing matched keypoints with keys "m_kpts0" and "m_kpts1".
    """
    pts0: torch.Tensor = control_points["m_kpts0"]
    pts1: torch.Tensor = control_points["m_kpts1"]
    assert len(pts0) == len(pts1), "The number of control points in both images must match."
    print(f"Found {len(pts0)} control points")
    assert len(pts0) > 0 and len(pts1) > 0, "No control points found."

    # Load the current PTO file and remove old control point lines.
    pto_lines: List[str] = load_pto_file(pto_file)
    pto_lines, _ = remove_control_points(pto_lines)
    pto_lines.append("")
    pto_lines.append(_CONTROL_POINTS_LINE)

    def _to_hugin_decimal(val: Union[str, float]) -> str:
        # Convert value to float and then format.
        val = float(val)
        if val == float(int(val)):
            return f"{int(val)}"
        return f"{val:.12f}"

    # Append new control point lines.
    for i in range(len(pts0)):
        point0 = [float(c) for c in pts0[i]]
        point1 = [float(c) for c in pts1[i]]
        line = (
            f"c n0 N1 x{_to_hugin_decimal(point0[0])} "
            f"y{_to_hugin_decimal(point0[1])} "
            f"X{_to_hugin_decimal(point1[0])} "
            f"Y{_to_hugin_decimal(point1[1])} t0"
        )
        pto_lines.append(line)
    save_pto_file(pto_file, pto_lines)
    print("Done updating control points in the PTO file.")


def configure_stitching(
    frame1: np.ndarray,
    frame2: np.ndarray,
    directory: str,
    force: bool = True,
    skip_if_exists: bool = False,
    fov: Optional[float] = None,
    max_control_points: int = 240,
    scale: Optional[float] = None,
    max_output_dimension: Optional[int] = None,
    device: Optional[torch.device] = None,
    control_point_matcher: Optional[str] = None,
    mapping_backend: Optional[str] = None,
    game_config: Optional[dict] = None,
    run_autooptimizer: Optional[bool] = None,
    settings: Optional[StitchingSettings] = None,
    game_id: Optional[str] = None,
) -> bool:
    """Calibrate two BGR frames using the shared project builder.

    Explicit arguments override game settings; unspecified values use the same
    defaults as the tracker. Input images stay temporary until the shared
    builder publishes the completed generation.
    """
    camera_fov = None
    if fov is not None:
        stitch_config = (game_config or {}).get("stitching")
        if stitch_config is None:
            stitch_config = {}
        inherited_fov = stitch_config.get("camera_fov")
        camera_fov = dict(inherited_fov) if inherited_fov is not None else {}
        camera_fov["horizontal_fov"] = fov
    settings = settings or read_stitching_settings(
        game_config,
        control_point_matcher=control_point_matcher,
        mapping_backend=mapping_backend,
        max_output_dimension=max_output_dimension,
        run_autooptimizer=run_autooptimizer,
        camera_fov=camera_fov,
    )
    if (
        isinstance(max_control_points, bool)
        or not isinstance(max_control_points, int)
        or max_control_points < 4
    ):
        raise ValueError("max_control_points must be an integer of at least four")
    if settings.control_point_matcher == "akaze-hamming" and max_control_points < 6:
        raise ValueError("AKAZE max_control_points must be at least six")
    settings = replace(settings, max_control_points=max_control_points)
    validate_output_scale(scale, settings.mapping_backend)
    directory = str(Path(directory).resolve())
    lens_calibration = (
        load_lens_calibration(directory)
        if settings.control_point_matcher == "akaze-hamming"
        else None
    )
    if lens_calibration is not None and settings.mapping_backend == "nona":
        raise ValueError(
            "Calibrated AKAZE requires an OpenCV mapping backend; NONA does not consume KB4 lenses"
        )
    settings = replace(
        settings,
        lens_profile_fingerprint=lens_calibration.fingerprint if lens_calibration else None,
    )
    Path(directory).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="hm-calibration-input-", dir=directory) as sampled:
        images = [str(Path(sampled) / name) for name in ("left.png", "right.png")]
        for image, frame in zip(images, (frame1, frame2)):
            if not cv2.imwrite(image, frame):
                raise OSError(f"Failed to save calibration frame: {image}")
        control_points_factory = partial(
            calculate_control_points,
            images[0],
            images[1],
            max_control_points=max_control_points,
            device=device,
            matcher=settings.control_point_matcher,
            lens_calibration=lens_calibration,
        )
        return build_stitching_project(
            project_file_path=str(Path(directory) / "hm_project.pto"),
            image_files=images,
            max_control_points=max_control_points,
            skip_if_exists=skip_if_exists,
            force=force,
            scale=scale,
            settings=settings,
            lens_calibration=lens_calibration,
            lens_calibration_resolved=True,
            control_points_factory=control_points_factory,
            game_id=game_id,
            game_config=game_config,
        )


def main() -> None:
    """
    Main entry point:
      - Parses command-line arguments.
      - Synchronizes the two videos by audio.
      - Extracts frames at the synchronization points.
      - Computes control points and runs the stitching pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize two videos using audio cross-correlation, extract sync frames, "
            "compute control points with a selectable matcher, and update a Hugin PTO file."
        )
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help="Game ID (everything being in $HOME/Videos/game-id)",
    )
    parser.add_argument("--left", default=None, help="Path to left video file")
    parser.add_argument("--right", default=None, help="Path to right video file")
    parser.add_argument(
        "--max-control-points", type=int, default=None, help="Maximum number of control points"
    )
    parser.add_argument(
        "--control-point-matcher",
        choices=CONTROL_POINT_MATCHERS,
        default=None,
        help="Feature matcher used to find control points",
    )
    parser.add_argument(
        "--mapping-backend",
        choices=MAPPING_BACKENDS,
        default=None,
        help="Backend used to generate mapping TIFFs",
    )
    parser.add_argument("--lfo", type=int, default=None, help="Left frame offset")
    parser.add_argument("--rfo", type=int, default=None, help="Right frame offset")
    parser.add_argument(
        "--synchronize-only",
        action="store_true",
        help="Only synchronize and print out the frame offsets",
    )
    parser.add_argument(
        "--scale",
        default=None,
        type=float,
        help="Scale of the final panorama (i.e. for downsizing)",
    )
    parser.add_argument(
        "--max-output-dimension",
        default=None,
        type=int,
        help="Maximum final panorama width/height; scales the Hugin canvas to fit when needed",
    )
    parser.add_argument(
        "--run-autooptimizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Hugin optimization (required for NONA)",
    )
    parser.add_argument(
        "--device", default=None, help="Torch matcher device, for example cpu or cuda:0"
    )
    parser.add_argument(
        "--stitch-frame-time", default=None, help="Calibration time (HH:MM:SS or seconds)"
    )
    parser.add_argument(
        "--calibration-frame-count",
        type=int,
        default=None,
        help="Synchronized frame pairs to sample (1–64)",
    )
    args = parser.parse_args()

    if (not args.left or not args.right) and not args.game_id:
        parser.error("Supply both --left and --right, or --game-id")
    if (args.lfo is None) != (args.rfo is None):
        parser.error("--lfo and --rfo must be supplied together")
    if any(offset is not None and offset < 0 for offset in (args.lfo, args.rfo)):
        parser.error("Frame offsets must be nonnegative")

    game_config = get_game_config(args.game_id) if args.game_id else {}
    game_dir = _game_dir_for_id(args.game_id) if args.game_id else None
    videos = game_config.get("game", {}).get("videos", {})
    for side in ("left", "right"):
        explicit = args.left if side == "left" else args.right
        if explicit is None:
            configured = videos.get(side)
            if (
                not isinstance(configured, list)
                or not configured
                or not isinstance(configured[0], str)
            ):
                parser.error(f"Game configuration must define game.videos.{side}")
            explicit = configured[0]
            if not Path(explicit).is_absolute():
                explicit = str(Path(game_dir) / explicit)
        if side == "left":
            args.left = explicit
        else:
            args.right = explicit

    image_left = Path(args.left).suffix.lower() == ".png"
    image_right = Path(args.right).suffix.lower() == ".png"
    if image_left != image_right:
        parser.error("Both inputs must be PNG images or both must be videos")
    if args.synchronize_only:
        if image_left:
            parser.error("--synchronize-only requires video inputs")
        lfo, rfo = synchronize_by_audio(args.left, args.right)
        print(f"Left frame offset: {lfo}")
        print(f"Right frame offset: {rfo}")
        return

    settings = read_stitching_settings(
        game_config,
        control_point_matcher=args.control_point_matcher,
        mapping_backend=args.mapping_backend,
        max_output_dimension=args.max_output_dimension,
        run_autooptimizer=args.run_autooptimizer,
        calibration_frame_count=args.calibration_frame_count,
        max_control_points=args.max_control_points,
    )
    validate_output_scale(args.scale, settings.mapping_backend)
    device = torch.device(args.device) if args.device is not None else None
    directory = game_dir or str(Path(args.left).resolve().parent)
    if image_left:
        if args.lfo is not None or args.stitch_frame_time is not None:
            parser.error("Frame offsets and calibration times require video inputs")
        if args.calibration_frame_count not in (None, 1):
            parser.error("Multiple calibration frames require video inputs")
        result = configure_stitching(
            extract_frame(args.left, None),
            extract_frame(args.right, None),
            directory=directory,
            max_control_points=settings.max_control_points,
            scale=args.scale,
            device=device,
            settings=settings,
            game_config=game_config,
            game_id=args.game_id,
        )
        if result is not True:
            raise RuntimeError("Stitching calibration did not produce a usable project")
    else:
        stitch_frame_time = args.stitch_frame_time
        if stitch_frame_time is None:
            stitch_config = game_config.get("stitching")
            stitch_frame_time = (
                stitch_config.get("stitch_frame_time") if stitch_config is not None else None
            )
        base_frame_offset = 0
        if stitch_frame_time is not None:
            base_frame_offset = time_to_frame(str(stitch_frame_time), BasicVideoInfo(args.left).fps)
            if base_frame_offset < 0:
                raise ValueError("Calibration time must be nonnegative")
        Path(directory).mkdir(parents=True, exist_ok=True)
        configure_video_stitching(
            dir_name=directory,
            video_left=args.left,
            video_right=args.right,
            max_control_points=settings.max_control_points,
            left_frame_offset=args.lfo,
            right_frame_offset=args.rfo,
            base_frame_offset=base_frame_offset,
            stitch_frame_time=stitch_frame_time,
            game_id=args.game_id,
            game_config=game_config,
            force=True,
            settings=settings,
            scale=args.scale,
            device=device,
        )


if __name__ == "__main__":
    main()
