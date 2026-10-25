"""Audio-based synchronization utilities for multi-camera stitching."""

import logging
import os
import subprocess

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from hmlib.config import (
    get_game_config_private,
    get_nested_value,
    normalize_runtime_config,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.utils.audio import load_audio_as_tensor
from hmlib.video.ffmpeg import BasicVideoInfo

logger = logging.getLogger(__name__)


def synchronize_by_audio(
    file1_path: str,
    file2_path: str,
    seconds: int = 15,
    device: torch.device = None,
    verbose: bool = True,
):
    """Estimate relative frame offset between two videos from audio tracks.

    @param file1_path: First video path.
    @param file2_path: Second video path.
    @param seconds: Number of initial seconds to use for correlation.
    @param device: Optional CUDA device for GPU-based correlation.
    @param verbose: If True, print progress messages.
    @return: Tuple ``(left_frame_offset, right_frame_offset)``.
    """
    # Load the videos
    if verbose:
        print("Openning videos...")

    video1_info = BasicVideoInfo(file1_path)
    video2_info = BasicVideoInfo(file2_path)

    seconds = min(seconds, min(video1_info.duration - 0.5, video1_info.duration - 0.5))

    video_1_subclip_frame_count = video1_info.fps * seconds
    video_2_subclip_frame_count = video2_info.fps * seconds

    audio1, sample_rate1 = load_audio_as_tensor(file1_path, duration_seconds=seconds)
    audio2, sample_rate2 = load_audio_as_tensor(file2_path, duration_seconds=seconds)

    if verbose:
        print("Loading audio...")

    audio_items_per_frame_1 = audio1.shape[0] / video_1_subclip_frame_count
    audio_items_per_frame_2 = audio2.shape[0] / video_2_subclip_frame_count

    assert np.isclose(float(sample_rate1 / video1_info.fps), float(audio_items_per_frame_1))
    assert np.isclose(float(sample_rate2 / video2_info.fps), float(audio_items_per_frame_2))

    # Calculate the cross-correlation of audio1 and audio2
    if verbose:
        print("Calculating cross-correlation...")
    if device is None:
        # correlation = np.correlate(audio1[:, 0], audio2[:, 0], mode="full")
        correlation = scipy.signal.correlate(audio1[:, 0], audio2[:, 0], mode="full")
        lag = np.argmax(correlation) - len(audio1) + 1
    else:
        audio1 = torch.from_numpy(audio1[:, 0]).unsqueeze(0).unsqueeze(0).to(device)
        audio2 = torch.from_numpy(audio2[:, 0]).unsqueeze(0).unsqueeze(0).to(device)

        # Compute correlation using convolution
        # The 'groups' argument ensures a separate convolution for each batch
        correlation = F.conv1d(audio1, audio2, padding=audio2.size(-1) - 1, groups=1)

        # Remove added dimensions to get the final 1D correlation tensor
        correlation = correlation.squeeze()
        lag = int(torch.argmax(correlation)) - audio2.size(-1) + 1

    # Calculate the time offset in seconds
    fps = video1_info.fps
    frame_offset = lag / audio_items_per_frame_1
    time_offset = frame_offset / fps

    if verbose:
        print(f"Left frame offset: {frame_offset}")
        print(f"Time offset: {time_offset} seconds")

    # Adjust to the starting frame number in each video (i.e. frame_offset might be a negative number)
    left_frame_offset = max(0, frame_offset)
    right_frame_offset = -frame_offset if frame_offset < 0 else 0

    return left_frame_offset, right_frame_offset


def configure_synchronization(
    game_id: str,
    video_left: str,
    video_right: str,
    audio_sync_seconds: float = 15.0,
    force: bool = False,
    sync_method: str | None = None,
) -> dict[str, float]:
    """Load or compute audio-based frame offsets and persist them to config.

    @param game_id: Game identifier (used to locate private config).
    @param video_left: Absolute path to left video.
    @param video_right: Absolute path to right video.
    @param audio_sync_seconds: Seconds of audio used for correlation.
    @param force: If True, ignore any cached offsets and recompute.
    @return: Mapping with ``\"left\"`` and ``\"right\"`` frame offsets.
    """
    config = get_game_config_private(game_id=game_id) or {}
    normalize_runtime_config(config)
    method = sync_method or get_nested_value(config, "stitching.sync_method", "audio")
    if method not in ("audio", "imu", "auto", "manual"):
        raise ValueError("stitching.sync_method must be audio, imu, auto or manual")
    frame_offsets = get_nested_value(config, "stitching.frame_offsets", None)
    if method == "manual":
        if not isinstance(frame_offsets, dict) or any(
            isinstance(frame_offsets.get(side), bool)
            or not isinstance(frame_offsets.get(side), (int, float))
            or not np.isfinite(frame_offsets[side])
            or frame_offsets[side] < 0
            for side in ("left", "right")
        ):
            raise ValueError(
                "Manual synchronization requires nonnegative left and right frame offsets"
            )
        return {side: float(frame_offsets[side]) for side in ("left", "right")}
    if force:
        frame_offsets = {}
    if (
        force
        or not frame_offsets
        or frame_offsets.get("left") is None
        or frame_offsets.get("right") is None
    ):
        # Calculate by audio
        # game_dir = get_game_dir(game_id=game_id)
        assert "/" in video_left  # should be full path
        offsets = None
        if method in ("imu", "auto"):
            from hmlib.stitching.imu_synchronization import synchronize_by_imu

            try:
                offsets = synchronize_by_imu(video_left, video_right)
            except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
                if method == "imu":
                    raise
                logger.warning("Gyroscope synchronization unavailable (%s); using audio", error)
        if offsets is None:
            offsets = synchronize_by_audio(
                file1_path=video_left,
                file2_path=video_right,
                seconds=audio_sync_seconds,
            )
        lfo, rfo = offsets
        if frame_offsets is None:
            frame_offsets = {}
        frame_offsets["left"] = float(lfo)
        frame_offsets["right"] = float(rfo)
        set_nested_value(config, "stitching.frame_offsets", frame_offsets)
        save_private_config(game_id=game_id, data=config)
    else:
        print(
            f"Preconfigured: left frame offset: {frame_offsets['left']}, right frame offset: {frame_offsets['right']}"
        )
    # Not get form the config
    frame_offsets = get_nested_value(config, "stitching.frame_offsets")
    return frame_offsets


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()
    synchronize_by_audio(
        file1_path=f"{os.environ['HOME']}/Videos/{args.game_id}/left.mp4",
        file2_path=f"{os.environ['HOME']}/Videos/{args.game_id}/right.mp4",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
