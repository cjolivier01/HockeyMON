#!/usr/bin/env python3
"""
This script synchronizes two videos using audio cross-correlation, extracts the corresponding frames,
computes control points with a selectable learned matcher and updates a Hugin
PTO file with the newly computed control points.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import ffmpegio
import numpy as np
import scipy.signal
import tifffile
import torch
import yaml
from hmlib.config import get_game_dir
from hmlib.stitching.configure_stitching import (
    MAPPING_BACKENDS,
    OPENCV_MAPPING_BACKENDS,
    get_enblend_bin,
    normalize_mapping_backend,
    normalize_max_output_dimension,
)
from hmlib.stitching.control_points import (
    CONTROL_POINT_MATCHERS,
    calculate_control_points as calculate_stitching_control_points,
)
from hmlib.stitching.homography_maps import (
    create_opencv_affine_ransac_mapping_files,
    create_opencv_magsac_mapping_files,
)

# Constant marker used in PTO files to denote control points.
_CONTROL_POINTS_LINE = "# control points"


def _run_stitching_command(cmd: List[str]) -> None:
    executable = cmd[0]
    if os.path.sep in executable:
        resolved = executable if os.access(executable, os.X_OK) else None
    else:
        resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"Required Hugin executable not found: {executable}. "
            "Install/build the Jetson-native Hugin tools or put them on PATH."
        )
    subprocess.run(cmd, check=True)


def _read_pto_canvas_size(pto_file: str) -> Optional[Tuple[int, int]]:
    with open(pto_file, "r") as file:
        for line in file:
            if not line.startswith("p "):
                continue
            width = re.search(r"(?:^|\s)w(\d+)(?:\s|$)", line)
            height = re.search(r"(?:^|\s)h(\d+)(?:\s|$)", line)
            if width and height:
                return int(width.group(1)), int(height.group(1))
    return None


def _game_dir_for_id(game_id: str) -> str:
    game_dir = get_game_dir(game_id=game_id, assert_exists=False)
    if game_dir is not None:
        return game_dir
    base_dir = os.environ.get("HM_GAME_DIR") or os.path.join(os.environ["HOME"], "Videos")
    return str(Path(base_dir) / game_id)


def _tiff_tag_number(tag, default: float) -> float:
    if tag is None:
        return default
    value = tag.value
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            numerator, denominator = value
            return float(numerator) / float(denominator)
        if len(value) == 1:
            return float(value[0])
    return float(value)


def _read_mapping_canvas_size(mapping_files: List[str]) -> Optional[Tuple[int, int]]:
    placements = []
    for mapping_file in mapping_files:
        with tifffile.TiffFile(mapping_file) as tif:
            page = tif.pages[0]
            tags = page.tags
            x_resolution = _tiff_tag_number(tags.get("XResolution"), 1.0)
            y_resolution = _tiff_tag_number(tags.get("YResolution"), 1.0)
            x_position = _tiff_tag_number(tags.get("XPosition"), 0.0)
            y_position = _tiff_tag_number(tags.get("YPosition"), 0.0)
            placements.append(
                (
                    x_position * x_resolution,
                    y_position * y_resolution,
                    int(page.imagewidth),
                    int(page.imagelength),
                )
            )
    if not placements:
        return None

    min_x = min(x for x, _, _, _ in placements)
    min_y = min(y for _, y, _, _ in placements)
    width = math.ceil(max(x - min_x + w for x, _, w, _ in placements))
    height = math.ceil(max(y - min_y + h for _, y, _, h in placements))
    return int(width), int(height)


def _remove_remap_outputs(directory: str) -> None:
    for pattern in (
        "autooptimiser_out.pto",
        "mapping_*.tif",
        "mapping_*.tiff",
        "panorama.tif",
        "seam_file.png",
        "s.png",
        "rink_mask_*.png",
    ):
        for path in Path(directory).glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _remove_mapping_outputs(directory: str) -> None:
    for pattern in (
        "mapping_*.tif",
        "mapping_*.tiff",
        "panorama.tif",
        "seam_file.png",
        "s.png",
        "rink_mask_*.png",
    ):
        for path in Path(directory).glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


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
    if video_path.endswith(".png"):
        return cv2.imread(video_path)
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
    fov: float = 108,  # Default FOV (e.g., GoPro Wide)
    max_control_points: int = 240,
    scale: float = None,
    max_output_dimension: Optional[int] = None,
    device: Optional[torch.device] = None,
    control_point_matcher: str = "superpoint-lightglue",
    mapping_backend: str = "nona",
) -> bool:
    """
    Configure and run the stitching pipeline. This includes:
      - Saving input frames as images.
      - Generating a Hugin PTO project file.
      - Computing control points and updating the PTO file.
      - Running auto-optimisation and generating mapping and panorama images.

    Args:
        frame1: First input frame (BGR image as a NumPy array).
        frame2: Second input frame (BGR image as a NumPy array).
        directory: Directory where output files will be saved.
        force: If True, force re-creation of the project file.
        skip_if_exists: If True, skip creation if output files already exist and are up-to-date.
        fov: Field-of-view parameter for the project generation.
        max_control_points: Maximum number of control points to compute.
        max_output_dimension: Maximum generated panorama width/height. If set, the PTO is auto-scaled to fit.
        device: Torch device for computations.
        control_point_matcher: Learned matcher used for point correspondences.
        mapping_backend: ``nona`` or a native OpenCV mapping backend.

    Returns:
        True if the process completes successfully.
    """
    mapping_backend = normalize_mapping_backend(mapping_backend)
    if mapping_backend in OPENCV_MAPPING_BACKENDS and scale not in (None, 1.0):
        raise ValueError(
            f"The {mapping_backend} backend does not accept --scale; " "use --max-output-dimension"
        )
    max_output_dimension = normalize_max_output_dimension(max_output_dimension)

    # Define file names for saved images.
    left_image_file: str = "left.png"
    right_image_file: str = "right.png"
    f1: str = os.path.join(directory, left_image_file)
    f2: str = os.path.join(directory, right_image_file)

    # Save the frames to disk.
    cv2.imwrite(f1, frame1)
    cv2.imwrite(f2, frame2)

    # Define paths for the project file and autooptimiser output.
    project_file_path: str = os.path.join(directory, "hm_project.pto")
    pto_path: Path = Path(project_file_path)
    dir_name: str = str(pto_path.parent)
    hm_project: str = project_file_path
    autooptimiser_out: str = os.path.join(dir_name, "autooptimiser_out.pto")
    assert autooptimiser_out != hm_project, "Output project file conflicts with input project file."

    # Optionally skip processing if outputs already exist and are up-to-date.
    if skip_if_exists and (
        os.path.exists(project_file_path)
        and os.path.exists(autooptimiser_out)
        and not is_older_than(project_file_path, autooptimiser_out)
    ):
        print(f"Project file already exists (skipping project creation): {autooptimiser_out}")
        return True

    curr_dir: str = os.getcwd()
    os.chdir(dir_name)
    try:
        # Generate the initial PTO project file if it doesn't exist or if forced.
        if not os.path.exists(hm_project) or force:
            cmd = [
                "pto_gen",
                "-p",
                "0",
                "-o",
                hm_project,
                "-f",
                str(fov),
                left_image_file,
                right_image_file,
            ]
            _run_stitching_command(cmd)

        # Calculate control points using the provided frames.
        control_points: Dict[str, torch.Tensor] = calculate_control_points(
            frame1,
            frame2,
            max_control_points=max_control_points,
            device=device,
            max_num_keypoints=2048,
            output_directory=directory,
            matcher=control_point_matcher,
        )
        # Update the PTO file with the new control points.
        update_pto_file(project_file_path, control_points)

        _remove_remap_outputs(dir_name)

        def run_autooptimiser(output_scale: Optional[float]) -> None:
            cmd = [
                "autooptimiser",
                "-a",
                "-l",
                "-s",
                "-o",
                autooptimiser_out,
                hm_project,
            ]
            if output_scale and output_scale != 1.0:
                cmd += [
                    "-x",
                    str(output_scale),
                ]
            _run_stitching_command(cmd)

        output_scale = float(scale) if scale else None
        if mapping_backend == "nona":
            run_autooptimiser(output_scale)
            if max_output_dimension and max_output_dimension > 0:
                canvas_size = _read_pto_canvas_size(autooptimiser_out)
                if canvas_size:
                    canvas_width, canvas_height = canvas_size
                    longest_dimension = max(canvas_width, canvas_height)
                    if longest_dimension > max_output_dimension:
                        current_scale = output_scale if output_scale else 1.0
                        output_scale = current_scale * (
                            float(max_output_dimension) / float(longest_dimension)
                        )
                        print(
                            "Scaling Hugin canvas from "
                            f"{canvas_width}x{canvas_height} to fit max dimension "
                            f"{max_output_dimension} (autooptimiser -x {output_scale:.6f})"
                        )
                        run_autooptimiser(output_scale)
        else:
            shutil.copyfile(hm_project, autooptimiser_out)

        def run_nona() -> List[str]:
            cmd = [
                "nona",
                "-m",
                "TIFF_m",
                "-z",
                "NONE",
                "--bigtiff",
                "-c",
                "-o",
                "mapping_",
                autooptimiser_out,
            ]
            _run_stitching_command(cmd)
            files = sorted(str(path) for path in Path(dir_name).glob("mapping_????.tif"))
            if not files:
                raise FileNotFoundError(f"No Hugin mapping TIFFs were generated in {dir_name}")
            return files

        mapping_files: List[str] = []
        if mapping_backend == "opencv-magsac":
            mapping_files = create_opencv_magsac_mapping_files(
                [f1, f2],
                control_points,
                dir_name,
                max_output_dimension=max_output_dimension,
            )
        elif mapping_backend == "opencv-affine-ransac":
            mapping_files = create_opencv_affine_ransac_mapping_files(
                [f1, f2],
                control_points,
                dir_name,
                max_output_dimension=max_output_dimension,
            )
        else:
            for attempt in range(3):
                mapping_files = run_nona()
                if not max_output_dimension or max_output_dimension <= 0:
                    break

                mapping_canvas_size = _read_mapping_canvas_size(mapping_files)
                if not mapping_canvas_size:
                    break

                mapping_width, mapping_height = mapping_canvas_size
                longest_mapping_dimension = max(mapping_width, mapping_height)
                if longest_mapping_dimension <= max_output_dimension:
                    break

                if attempt == 2:
                    raise RuntimeError(
                        "Generated Hugin mapping canvas "
                        f"{mapping_width}x{mapping_height} still exceeds max dimension "
                        f"{max_output_dimension}"
                    )

                current_scale = output_scale if output_scale else 1.0
                output_scale = (
                    current_scale
                    * (float(max_output_dimension) / float(longest_mapping_dimension))
                    * 0.999
                )
                print(
                    "Generated mapping canvas "
                    f"{mapping_width}x{mapping_height} exceeds max dimension "
                    f"{max_output_dimension}; retrying autooptimiser -x {output_scale:.6f}"
                )
                _remove_mapping_outputs(dir_name)
                run_autooptimiser(output_scale)

        # Blend the mappings into a panorama using enblend.
        cmd = [
            get_enblend_bin(),
            "--save-masks=seam_file.png",
            "-o",
            os.path.join(dir_name, "panorama.tif"),
            *mapping_files,
        ]
        try:
            _run_stitching_command(cmd)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            try:
                Path(dir_name, "seam_file.png").unlink()
            except FileNotFoundError:
                pass
            print(f"Warning: failed to run enblend for seam mask generation: {exc}")
    finally:
        os.chdir(curr_dir)
    return True


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
    parser.add_argument("--right", default=None, help="Path to left video file")
    parser.add_argument(
        "--max-control-points", type=int, default=500, help="Maximum number of control points"
    )
    parser.add_argument(
        "--control-point-matcher",
        choices=CONTROL_POINT_MATCHERS,
        default="superpoint-lightglue",
        help="Feature matcher used to find control points",
    )
    parser.add_argument(
        "--mapping-backend",
        choices=MAPPING_BACKENDS,
        default="nona",
        help="Backend used to generate mapping TIFFs",
    )
    parser.add_argument("--lfo", default=None, help="Left frame offset")
    parser.add_argument("--rfo", default=None, help="Right frame offset")
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
    args = parser.parse_args()

    if (not args.left or not args.right) and not args.game_id:
        print("You must supply either left and right videos or a game-id")
        exit(1)

    if (not args.left or not args.right) and args.game_id:
        game_dir: str = _game_dir_for_id(args.game_id)
        config_file: str = os.path.join(game_dir, "config.yaml")
        if not os.path.exists(config_file):
            print(f"Could not find config file: {config_file}")
            exit(1)
        with open(config_file, "r") as file:
            config_yaml = yaml.safe_load(file)
        args.left = config_yaml["game"]["videos"]["left"][0]
        if "/" not in args.left:
            args.left = os.path.join(game_dir, args.left)
        args.right = config_yaml["game"]["videos"]["right"][0]
        if "/" not in args.right:
            args.right = os.path.join(game_dir, args.right)

    is_image = False
    if args.left.endswith(".png") and args.right.endswith(".png"):
        is_image = True

    if not is_image:
        # Determine frame offsets by synchronizing audio.
        if (args.lfo is None and args.rfo is None) or args.synchronize_only:
            lfo, rfo = synchronize_by_audio(args.left, args.right)
        else:
            lfo, rfo = args.lfo, args.rfo

        if args.synchronize_only:
            print(f"Left frame offset: {lfo}")
            print(f"Right frame offset: {rfo}")
            exit(0)

        print("Extracting frames at the sync points...")
    else:
        lfo, rfo = None, None

    # Ensure frame indices are integers.
    frame1: np.ndarray = extract_frame(args.left, lfo)
    frame2: np.ndarray = extract_frame(args.right, rfo)

    # Run the stitching pipeline which includes control point computation and PTO update.
    print(f"Running {args.control_point_matcher} to obtain control point matches...")
    configure_stitching(
        frame1,
        frame2,
        directory=str(Path(args.left).parent),
        max_control_points=args.max_control_points,
        scale=args.scale,
        max_output_dimension=args.max_output_dimension,
        control_point_matcher=args.control_point_matcher,
        mapping_backend=args.mapping_backend,
    )


if __name__ == "__main__":
    main()
