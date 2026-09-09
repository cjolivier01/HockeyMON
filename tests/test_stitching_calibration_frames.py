from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from hmlib.stitching import configure_stitching
from hmlib.stitching.calibration import (
    CalibrationAlignmentError,
    calibration_candidates,
    sample_frame_indices,
)
from hmlib.stitching.settings import read_stitching_settings


def _points(offset=0):
    points = torch.tensor([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=torch.float32) + offset
    return {"m_kpts0": points, "m_kpts1": points + 1}


def _images(directory, count=2):
    pairs = []
    for index in range(count):
        pair = (directory / f"left-{index}.png", directory / f"right-{index}.png")
        for path in pair:
            assert cv2.imwrite(str(path), np.full((64, 64, 3), index, np.uint8))
        pairs.append(pair)
    return pairs


def should_sample_distinct_synchronized_pairs_and_clip_short_clips():
    assert sample_frame_indices(5, 2, 4, 100, 100) == [(5, 2), (6, 3), (7, 4), (8, 5)]
    assert sample_frame_indices(5, 2, 4, 7, 100) == [(5, 2), (6, 3)]
    with pytest.raises(ValueError, match="beyond"):
        sample_frame_indices(5, 2, 4, 5, 100)


@pytest.mark.parametrize("value", [0, -1, 65, 1.5, True])
def should_reject_invalid_frame_count(value):
    with pytest.raises(ValueError, match="calibration_frame_count"):
        read_stitching_settings({"stitching": {"calibration_frame_count": value}})


def should_pool_distinct_correspondences_before_individual_candidates(tmp_path):
    pairs = _images(tmp_path)
    calls = []

    def matcher(left, right, **kwargs):
        calls.append((left, right))
        return _points(len(calls))

    candidates = list(calibration_candidates(pairs, 20, matcher, "loftr"))
    assert len(calls) == 2
    assert len(candidates) == 3
    assert len(candidates[0].points["m_kpts0"]) == 8
    assert len(candidates[1].points["m_kpts0"]) == 4
    assert candidates[0].images == candidates[1].images == pairs[0]


def should_skip_only_explicit_matching_failures(tmp_path):
    pairs = _images(tmp_path)

    def matcher(left, *args, **kwargs):
        if left == str(pairs[0][0]):
            raise CalibrationAlignmentError("no overlap")
        return _points()

    candidates = list(calibration_candidates(pairs, 20, matcher, "loftr"))
    assert [candidate.images for candidate in candidates] == [pairs[1]]

    def broken_matcher(*args, **kwargs):
        raise OSError("model file unreadable")

    with pytest.raises(OSError, match="model file unreadable"):
        list(calibration_candidates(pairs, 20, broken_matcher, "loftr"))


def should_reject_changing_camera_geometry_and_invalid_points(tmp_path):
    pairs = _images(tmp_path)
    assert cv2.imwrite(str(pairs[1][1]), np.zeros((63, 64, 3), np.uint8))
    with pytest.raises(ValueError, match="stable source dimensions"):
        list(calibration_candidates(pairs, 20, lambda *args, **kwargs: _points(), "loftr"))
    with pytest.raises(ValueError, match="outside"):
        list(calibration_candidates(pairs[:1], 20, lambda *args, **kwargs: _points(100), "loftr"))


@pytest.mark.parametrize("terminal", [False, True])
def should_retry_rejected_geometry_and_preserve_late_failures(monkeypatch, tmp_path, terminal):
    extracted = []
    attempts = []
    monkeypatch.setattr(
        configure_stitching, "BasicVideoInfo", lambda path: SimpleNamespace(frame_count=100)
    )
    monkeypatch.setattr(configure_stitching, "sync_stitch_frame_time_state", lambda **kwargs: False)
    monkeypatch.setattr(
        configure_stitching, "_save_stitched_reference_frame", lambda directory: None
    )

    def extract(video, frame_number, dest_image):
        extracted.append((video, frame_number))
        assert cv2.imwrite(dest_image, np.full((64, 64, 3), frame_number, np.uint8))

    def matcher(left, right, **kwargs):
        return _points(int(cv2.imread(left)[0, 0, 0]))

    failure = OSError("seam disk full") if terminal else CalibrationAlignmentError("bad alignment")

    def build(**kwargs):
        attempts.append(kwargs)
        # Each PTO uses stable references, never filenames in a deleted sample directory.
        assert [Path(name).name for name in kwargs["image_files"]] == ["left.png", "right.png"]
        if len(attempts) == 1:
            raise failure
        return True

    monkeypatch.setattr(configure_stitching, "extract_frame_image", extract)
    monkeypatch.setattr(configure_stitching, "calculate_control_points", matcher)
    monkeypatch.setattr(configure_stitching, "build_stitching_project", build)
    arguments = dict(
        dir_name=str(tmp_path),
        video_left="left.mp4",
        video_right="right.mp4",
        max_control_points=100,
        left_frame_offset=3,
        right_frame_offset=1,
        game_config={"stitching": {"calibration_frame_count": 2}},
        ignore_private_config=True,
    )
    if terminal:
        with pytest.raises(OSError) as caught:
            configure_stitching.configure_video_stitching(**arguments)
        assert caught.value is failure
        assert len(attempts) == 1
    else:
        assert configure_stitching.configure_video_stitching(**arguments)[1:] == (3, 1)
        assert len(attempts) == 2
        assert len(attempts[0]["control_points"]["m_kpts0"]) == 8
    assert extracted == [("left.mp4", 3), ("right.mp4", 1), ("left.mp4", 4), ("right.mp4", 2)]
    assert not list(tmp_path.glob("hm-calibration-input-*"))


def should_keep_frame_count_in_cache_provenance():
    first = read_stitching_settings({"stitching": {"calibration_frame_count": 1}})
    multiple = read_stitching_settings({"stitching": {"calibration_frame_count": 4}})
    assert first.manifest() != multiple.manifest()
