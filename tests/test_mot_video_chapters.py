"""Decode real chapter files across playlist-global starts and runtime seeks."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import torch

from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig


@pytest.fixture
def chapters(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required for real chapter decoding")
    paths = []
    for chapter in range(2):
        path = tmp_path / f"chapter-{chapter}.mkv"
        frames = np.stack(
            [
                np.full((24, 32, 3), frame, np.uint8)
                for frame in range(chapter * 10, chapter * 10 + 10)
            ]
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "32x24",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                str(path),
            ],
            input=frames.tobytes(),
            check=True,
            capture_output=True,
        )
        paths.append(str(path))
    return paths


def loader(paths, **kwargs):
    return MOTLoadVideoWithOrig(
        path=paths,
        original_image_only=True,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
        decoder_type="cv2",
        **kwargs,
    )


def samples(dataset):
    return [(int(batch["frame_ids"][0]), int(batch["img"][0, 0, 0, 0])) for batch in dataset]


@pytest.mark.parametrize("async_mode", [False, True])
def should_seek_to_the_local_frame_in_a_later_chapter(chapters, async_mode):
    dataset = loader(chapters, start_frame_number=15, async_mode=async_mode)
    try:
        assert samples(dataset) == [(frame, frame) for frame in range(15, 20)]
    finally:
        dataset.close()


@pytest.mark.parametrize("async_mode", [False, True])
def should_keep_frame_step_and_sync_offset_separate_across_chapters(chapters, async_mode):
    dataset = loader(
        chapters, start_frame_number=2, source_frame_offset=5, frame_step=2, async_mode=async_mode
    )
    try:
        assert dataset.fps == 5
        assert samples(dataset) == [(7, 10), (8, 12), (9, 14), (10, 16), (11, 18)]
    finally:
        dataset.close()


@pytest.mark.parametrize("async_mode", [False, True])
def should_seek_between_chapters_without_resetting_global_ids(chapters, async_mode):
    dataset = loader(chapters, start_frame_number=2, async_mode=async_mode, prefetch_batches=1)
    try:
        iterator = iter(dataset)
        first = next(iterator)
        assert int(first["frame_ids"][0]) == 2
        dataset.seek(15)
        batch = next(iterator)
        assert int(batch["frame_ids"][0]) == 15
        assert int(batch["img"][0, 0, 0, 0]) == 15
        dataset.seek(3)
        batch = next(iterator)
        assert int(batch["frame_ids"][0]) == 3
        assert int(batch["img"][0, 0, 0, 0]) == 3
    finally:
        dataset.close()


def should_preserve_source_pts_for_mixed_rate_camera_frames(chapters):
    dataset = loader(
        chapters, start_frame_number=2, source_frame_offset=5, frame_step=2, async_mode=False
    )
    try:
        batches = list(dataset)
        assert [batch["pts_ns"][0] for batch in batches] == [
            1_000_000_000,
            1_200_000_000,
            1_400_000_000,
            1_600_000_000,
            1_800_000_000,
        ]
        assert [int(batch["frame_ids"][0]) for batch in batches] == [7, 8, 9, 10, 11]
    finally:
        dataset.close()


@pytest.mark.parametrize("async_mode", [False, True])
def should_sample_each_frame_in_a_mixed_rate_batch(chapters, async_mode):
    dataset = loader(
        chapters, start_frame_number=2, frame_step=2, batch_size=2, async_mode=async_mode
    )
    try:
        batches = list(dataset)
        assert [batch["frame_ids"].tolist() for batch in batches] == [
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ]
        assert [batch["img"][:, 0, 0, 0].tolist() for batch in batches] == [
            [5, 7],
            [9, 11],
            [13, 15],
            [17, 19],
        ]
        assert [batch["pts_ns"] for batch in batches] == [
            [500000000, 700000000],
            [900000000, 1100000000],
            [1300000000, 1500000000],
            [1700000000, 1900000000],
        ]
    finally:
        dataset.close()
