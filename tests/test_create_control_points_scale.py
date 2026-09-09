"""Public standalone calibration contracts shared with the tracker pipeline."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from hmlib.cli import create_control_points as cli
from hmlib.stitching import configure_stitching as shared


def _frame():
    return np.full((32, 48, 3), 37, np.uint8)


def _points():
    return {"m_kpts0": torch.ones((8, 2)), "m_kpts1": torch.ones((8, 2))}


def should_delegate_frame_calibration_with_settings_device_and_temporary_inputs(
    monkeypatch, tmp_path
):
    config = {
        "stitching": {
            "mapping_backend": "nona",
            "run_autooptimizer": True,
            "control_point_matcher": "loftr",
            "max_output_dimension": 2048,
            "camera_fov": {"horizontal_fov": 95, "vertical_fov": 70},
        }
    }
    original = copy.deepcopy(config)
    (tmp_path / "left.png").write_bytes(b"previous left")
    (tmp_path / "right.png").write_bytes(b"previous right")
    points = _points()
    matches, builds = [], []

    def match(*args, **kwargs):
        matches.append(kwargs)
        return points

    def build(**kwargs):
        builds.append(kwargs)
        assert kwargs["control_points"] is points
        for image in kwargs["image_files"]:
            assert Path(image).parent != tmp_path
            np.testing.assert_array_equal(cv2.imread(image), _frame())
        assert (tmp_path / "left.png").read_bytes() == b"previous left"
        assert (tmp_path / "right.png").read_bytes() == b"previous right"
        return True

    monkeypatch.setattr(cli, "calculate_control_points", match)
    monkeypatch.setattr(cli, "build_stitching_project", build)
    assert (
        cli.configure_stitching(
            _frame(),
            _frame(),
            str(tmp_path),
            game_config=config,
            device=torch.device("cpu"),
            scale=0.5,
            fov=100,
            control_point_matcher="dedode-lightglue",
        )
        is True
    )
    assert len(builds) == 1
    assert builds[0]["settings"].control_point_matcher == "dedode-lightglue"
    assert builds[0]["settings"].horizontal_fov == 100
    assert builds[0]["settings"].vertical_fov == 70
    assert builds[0]["settings"].max_output_dimension == 2048
    assert builds[0]["scale"] == 0.5
    assert matches[0]["device"] == torch.device("cpu")
    assert builds[0]["lens_calibration_resolved"] is True
    assert config == original
    assert not list(tmp_path.glob("hm-calibration-input-*"))


@pytest.mark.parametrize(
    "options,match",
    [
        ({"mapping_backend": "opencv-magsac", "scale": 0.5}, "opencv-magsac"),
        ({"mapping_backend": "opencv-affine-ransac", "scale": 0.5}, "opencv-affine-ransac"),
        ({"mapping_backend": "nona"}, "run_autooptimizer"),
        ({"max_output_dimension": 0}, "max_output_dimension"),
        ({"max_output_dimension": 65535}, "max_output_dimension"),
        ({"scale": 0}, "scale"),
        ({"scale": float("nan")}, "scale"),
        ({"scale": float("inf")}, "scale"),
        ({"max_control_points": 2}, "max_control_points"),
        ({"control_point_matcher": "akaze", "max_control_points": 4}, "at least six"),
    ],
)
def should_reject_invalid_calibration_before_writing_frames(tmp_path, options, match):
    directory = tmp_path / "new-game"
    with pytest.raises(ValueError, match=match):
        cli.configure_stitching(_frame(), _frame(), str(directory), **options)
    assert not directory.exists()


def should_preserve_references_and_cleanup_temporary_frames_on_builder_failure(
    monkeypatch, tmp_path
):
    (tmp_path / "left.png").write_bytes(b"old left")
    (tmp_path / "right.png").write_bytes(b"old right")
    failure = OSError("mapping write failed")
    monkeypatch.setattr(cli, "calculate_control_points", lambda *args, **kwargs: _points())

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(cli, "build_stitching_project", fail)
    with pytest.raises(OSError) as caught:
        cli.configure_stitching(_frame(), _frame(), str(tmp_path))
    assert caught.value is failure
    assert (tmp_path / "left.png").read_bytes() == b"old left"
    assert (tmp_path / "right.png").read_bytes() == b"old right"
    assert not list(tmp_path.glob("hm-calibration-input-*"))


def should_surface_failed_input_image_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="save calibration frame"):
        cli.configure_stitching(_frame(), _frame(), str(tmp_path))
    assert not list(tmp_path.glob("hm-calibration-input-*"))


def should_apply_game_settings_even_with_explicit_image_inputs(monkeypatch, tmp_path):
    config = {
        "stitching": {
            "control_point_matcher": "loftr",
            "mapping_backend": "nona",
            "run_autooptimizer": False,
        }
    }
    monkeypatch.setattr(cli, "get_game_config", lambda game_id: config)
    monkeypatch.setattr(cli, "_game_dir_for_id", lambda game_id: str(tmp_path))
    monkeypatch.setattr(cli, "extract_frame", lambda *args: _frame())
    calls = []
    monkeypatch.setattr(
        cli, "configure_stitching", lambda *args, **kwargs: calls.append(kwargs) or True
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_control_points",
            "--game-id",
            "demo",
            "--left",
            "a.png",
            "--right",
            "b.png",
            "--run-autooptimizer",
            "--scale",
            ".5",
            "--device",
            "cpu",
        ],
    )
    cli.main()
    assert calls[0]["directory"] == str(tmp_path)
    assert calls[0]["settings"].control_point_matcher == "loftr"
    assert calls[0]["settings"].run_autooptimizer is True
    assert calls[0]["scale"] == 0.5
    assert calls[0]["device"] == torch.device("cpu")
    assert calls[0]["game_config"] is config


def should_route_videos_to_multiframe_pipeline_with_configured_time(monkeypatch, tmp_path):
    config = {
        "game": {"videos": {"left": ["left/clip.mp4"], "right": ["right/clip.mp4"]}},
        "stitching": {
            "control_point_matcher": "loftr",
            "calibration_frame_count": 3,
            "stitch_frame_time": "00:00:02",
        },
    }
    monkeypatch.setattr(cli, "get_game_config", lambda game_id: config)
    monkeypatch.setattr(cli, "_game_dir_for_id", lambda game_id: str(tmp_path))
    monkeypatch.setattr(cli, "BasicVideoInfo", lambda video: SimpleNamespace(fps=30))
    calls = []
    monkeypatch.setattr(cli, "configure_video_stitching", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["create_control_points", "--game-id", "demo", "--lfo", "3", "--rfo", "1"]
    )
    cli.main()
    assert calls[0]["video_left"] == str(tmp_path / "left/clip.mp4")
    assert calls[0]["video_right"] == str(tmp_path / "right/clip.mp4")
    assert calls[0]["left_frame_offset"] == 3
    assert calls[0]["right_frame_offset"] == 1
    assert calls[0]["base_frame_offset"] == 60
    assert calls[0]["settings"].calibration_frame_count == 3
    assert calls[0]["game_config"] is config
    assert calls[0]["game_id"] == "demo"


@pytest.mark.parametrize("extra", [["--lfo", "2"], ["--lfo", "-1", "--rfo", "0"]])
def should_reject_partial_or_negative_offsets(monkeypatch, extra):
    monkeypatch.setattr(
        sys, "argv", ["create_control_points", "--left", "a.mp4", "--right", "b.mp4", *extra]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def should_forward_scale_and_device_to_the_shared_video_worker(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        shared, "_configure_video_stitching_locked", lambda **kwargs: calls.append(kwargs)
    )
    shared.configure_video_stitching(
        str(tmp_path),
        "a.mp4",
        "b.mp4",
        100,
        device=torch.device("cpu"),
        scale=0.5,
        game_config={"stitching": {"mapping_backend": "nona", "run_autooptimizer": True}},
    )
    assert calls[0]["scale"] == 0.5
    assert calls[0]["device"] == torch.device("cpu")
