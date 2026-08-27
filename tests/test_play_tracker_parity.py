# ruff: noqa: E402

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import pytest
import torch
from torch.testing import assert_close

from hmlib.camera.camera import HockeyMON
from hmlib.camera.play_tracker import PlayTracker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
DEVICE = torch.device("cpu")
DEFAULT_CLUSTER_CENTROIDS = [
    [300.0, 300.0],
    [600.0, 400.0],
    [900.0, 500.0],
]


def _base_game_config() -> Dict:
    return {
        "rink": {
            "camera": {
                "pan_smoothing_alpha": 0.18,
                "sticky_size_ratio_to_frame_width": 0.5,
                "sticky_translation_gaussian_mult": 1.0,
                "unsticky_translation_size_ratio": 0.2,
                "follower_box_scale_width": 1.0,
                "follower_box_scale_height": 1.0,
                "time_to_dest_speed_limit_frames": 10,
                "time_to_dest_stop_speed_threshold": 0.25,
                "resizing_stop_on_dir_change_delay": 10,
                "resizing_cancel_stop_on_opposite_dir": True,
                "resizing_stop_cancel_hysteresis_frames": 2,
                "resizing_stop_delay_cooldown_frames": 2,
                "resizing_time_to_dest_speed_limit_frames": 10,
                "resizing_time_to_dest_stop_speed_threshold": 0.25,
                "fixed_edge_scaling_factor": 1.0,
                "fixed_edge_rotation_angle": 0.0,
                "stop_on_dir_change_delay": 10,
                "cancel_stop_on_opposite_dir": True,
                "stop_cancel_hysteresis_frames": 2,
                "stop_delay_cooldown_frames": 2,
                "max_speed_ratio_x": 1.0,
                "max_speed_ratio_y": 1.0,
                "max_accel_ratio_x": 1.0,
                "max_accel_ratio_y": 1.0,
                "color": {
                    "white_balance": [1.0, 1.0, 1.0],
                    "brightness": 1.0,
                    "contrast": 1.0,
                    "gamma": 1.0,
                },
                "breakaway_detection": {
                    "min_considered_group_velocity": 1.0,
                    "group_ratio_threshold": 0.5,
                    "group_velocity_speed_ratio": 0.5,
                    "scale_speed_constraints": 1.0,
                    "nonstop_delay_count": 3,
                    "overshoot_scale_speed_ratio": 1.0,
                    "overshoot_stop_delay_count": 3,
                    "post_nonstop_stop_delay_count": 2,
                },
            },
            "tracking": {"cam_ignore_largest": False},
        },
        "game": {
            "boundaries": {
                "upper": [],
                "lower": [],
                "upper_tune_position": [],
                "lower_tune_position": [],
                "scale_width": 1.0,
                "scale_height": 1.0,
            }
        },
    }


def _build_tracker(game_cfg: Dict, overrides: Dict, cpp_playtracker: bool) -> PlayTracker:
    hockey_mon = HockeyMON(
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fps=30.0,
        device=DEVICE,
        camera_name="GoPro",
    )
    play_box = torch.tensor(
        [0.0, 0.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT)], dtype=torch.float32
    )
    return PlayTracker(
        hockey_mon=hockey_mon,
        play_box=play_box,
        device=DEVICE,
        original_clip_box=None,
        progress_bar=None,
        game_config=game_cfg,
        cam_ignore_largest=game_cfg["rink"]["tracking"]["cam_ignore_largest"],
        no_wide_start=bool(overrides.get("no_wide_start", False)),
        track_ids=overrides.get("track_ids", ""),
        debug_play_tracker=False,
        plot_moving_boxes=False,
        plot_individual_player_tracking=False,
        plot_boundaries=False,
        plot_all_detections=None,
        plot_trajectories=False,
        plot_speed=False,
        plot_jersey_numbers=False,
        plot_actions=False,
        camera_ui=0,
        camera_controller="rule",
        camera_model=None,
        camera_window=8,
        force_stitching=False,
        cluster_centroids=copy.deepcopy(
            overrides.get("cluster_centroids", DEFAULT_CLUSTER_CENTROIDS)
        ),
        cpp_boxes=True,
        cpp_playtracker=cpp_playtracker,
        plot_cluster_tracking=bool(overrides.get("plot_cluster_tracking", False)),
    )


def _make_frame(frame_id: int, track_ids: Iterable[int]) -> SimpleNamespace:
    shift = frame_id * 5
    bboxes = torch.tensor(
        [
            [200 + shift, 300, 260 + shift, 360],
            [400 + shift, 320, 470 + shift, 390],
            [800 + shift, 330, 860 + shift, 400],
            [1000 + shift, 340, 1070 + shift, 410],
        ],
        dtype=torch.float32,
    )
    ids_tensor = torch.tensor(track_ids, dtype=torch.int64)
    pred_instances = SimpleNamespace(bboxes=bboxes, scores=torch.ones(len(track_ids)))
    pred_track_instances = SimpleNamespace(bboxes=bboxes, instances_id=ids_tensor)
    return SimpleNamespace(
        frame_id=frame_id,
        pred_instances=pred_instances,
        pred_track_instances=pred_track_instances,
        metainfo={},
    )


def _make_results(num_frames: int = 3) -> Dict:
    frames = [_make_frame(idx, track_ids=(1, 2, 3, 4)) for idx in range(num_frames)]
    data_samples = SimpleNamespace(video_data_samples=frames)
    original_images = torch.zeros((num_frames, 3, 540, 960), dtype=torch.float32)
    return {
        "data_samples": data_samples,
        "original_images": original_images,
    }


def _make_results_from_box_list(per_frame_boxes: Iterable[torch.Tensor]) -> Dict:
    frames = []
    for idx, boxes in enumerate(per_frame_boxes):
        if boxes is None:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            ids_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            bboxes = torch.as_tensor(boxes, dtype=torch.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes.unsqueeze(0)
            ids_tensor = torch.arange(1, bboxes.shape[0] + 1, dtype=torch.int64)
        preds = SimpleNamespace(bboxes=bboxes, scores=torch.ones(len(ids_tensor)))
        pred_track_instances = SimpleNamespace(bboxes=bboxes, instances_id=ids_tensor)
        frames.append(
            SimpleNamespace(
                frame_id=idx,
                pred_instances=preds,
                pred_track_instances=pred_track_instances,
                metainfo={},
            )
        )

    data_samples = SimpleNamespace(video_data_samples=frames)
    original_images = torch.zeros((len(frames), 3, 540, 960), dtype=torch.float32)
    return {
        "data_samples": data_samples,
        "original_images": original_images,
    }


def _run_play_trackers(overrides: Dict | None = None) -> Tuple[Dict, Dict]:
    overrides = overrides or {}
    game_cfg = _base_game_config()
    if "cam_ignore_largest" in overrides:
        game_cfg["rink"]["tracking"]["cam_ignore_largest"] = overrides["cam_ignore_largest"]
    ratio_keys = (
        "max_speed_ratio_x",
        "max_speed_ratio_y",
        "max_accel_ratio_x",
        "max_accel_ratio_y",
    )
    for key in ratio_keys:
        if key in overrides:
            game_cfg["rink"]["camera"][key] = float(overrides[key])

    python_tracker = _build_tracker(game_cfg, overrides, cpp_playtracker=False)
    cpp_tracker = _build_tracker(game_cfg, overrides, cpp_playtracker=True)
    python_results = python_tracker.forward(_make_results())
    cpp_results = cpp_tracker.forward(_make_results())
    return python_results, cpp_results


@pytest.mark.parametrize("overrides", [{}, {"cam_ignore_largest": True}])
def should_match_camera_boxes_between_cpp_and_python(overrides):
    py_results, cpp_results = _run_play_trackers(overrides)
    assert_close(py_results["current_box"], cpp_results["current_box"], atol=1e-4, rtol=0)
    assert_close(
        py_results["current_fast_box_list"], cpp_results["current_fast_box_list"], atol=1e-4, rtol=0
    )


def should_match_when_cluster_debugging_enabled():
    py_results, cpp_results = _run_play_trackers({"plot_cluster_tracking": True})
    assert_close(py_results["current_box"], cpp_results["current_box"], atol=1e-4, rtol=0)
    assert_close(
        py_results["current_fast_box_list"], cpp_results["current_fast_box_list"], atol=1e-4, rtol=0
    )


def should_filter_invalid_tracks_before_cpp_playtracker():
    tracker = _build_tracker(_base_game_config(), {}, cpp_playtracker=True)
    pred_track_instances = SimpleNamespace(
        bboxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [4.0, 9.0, 12.0, 3.0],
                [20.0, 20.0, 40.0, 60.0],
                [50.0, 50.0, 60.0, 70.0],
            ],
            dtype=torch.float32,
        ),
        instances_id=torch.tensor([5, 6, -1, 7], dtype=torch.int64),
    )
    video_data_sample = SimpleNamespace(
        metainfo={"num_tracks": torch.tensor([3], dtype=torch.long)}
    )

    ids, bboxes, tlwh = tracker._prepare_online_tracks(
        pred_track_instances,
        video_data_sample,
        frame_id=42,
    )

    assert ids.tolist() == [5]
    assert_close(bboxes, torch.tensor([[0.0, 0.0, 10.0, 10.0]]))
    assert_close(tlwh, torch.tensor([[0.0, 0.0, 10.0, 10.0]]))


def should_match_with_custom_cluster_centroids():
    custom_centroids = [
        [100.0, 200.0],
        [500.0, 450.0],
        [900.0, 500.0],
    ]
    py_results, cpp_results = _run_play_trackers({"cluster_centroids": custom_centroids})
    assert_close(py_results["current_box"], cpp_results["current_box"], atol=1e-4, rtol=0)
    assert_close(
        py_results["current_fast_box_list"], cpp_results["current_fast_box_list"], atol=1e-4, rtol=0
    )


def should_draw_cluster_boxes_with_external_camera_boxes(monkeypatch):
    overrides = {"plot_cluster_tracking": True}
    game_cfg = _base_game_config()
    tracker = _build_tracker(game_cfg, overrides, cpp_playtracker=False)
    results = _make_results()
    camera_boxes = [
        torch.tensor([150.0, 150.0, 600.0, 450.0], dtype=torch.float32)
        for _ in results["data_samples"].video_data_samples
    ]
    results["camera_boxes"] = camera_boxes

    call_counter = {"count": 0}

    def fake_plot_alpha_rectangle(image, box, **kwargs):
        call_counter["count"] += 1
        return image

    monkeypatch.setattr(
        "hmlib.camera.play_tracker.vis.plot_alpha_rectangle", fake_plot_alpha_rectangle
    )
    tracker.forward(copy.deepcopy(results))
    assert call_counter["count"] > 0


def should_apply_no_wide_start_after_initial_empty_frame():
    overrides = {"no_wide_start": True}
    game_cfg = _base_game_config()
    tracker = _build_tracker(game_cfg, overrides, cpp_playtracker=False)
    narrow_box = torch.tensor(
        [
            [400.0, 300.0, 600.0, 500.0],
            [620.0, 320.0, 780.0, 520.0],
            [800.0, 330.0, 960.0, 540.0],
        ],
        dtype=torch.float32,
    )
    results = _make_results_from_box_list(
        [
            None,  # first frame has no detections
            narrow_box,
        ]
    )
    output = tracker.forward(results)
    play_box = tracker.play_box
    second_frame_box = output["current_box"][1]
    assert not torch.allclose(second_frame_box, play_box)


def should_match_when_speed_ratios_change():
    overrides = {
        "max_speed_ratio_x": 0.7,
        "max_speed_ratio_y": 0.9,
        "max_accel_ratio_x": 1.3,
        "max_accel_ratio_y": 0.8,
    }
    py_results, cpp_results = _run_play_trackers(overrides)
    assert_close(py_results["current_box"], cpp_results["current_box"], atol=1e-4, rtol=0)
    assert_close(
        py_results["current_fast_box_list"], cpp_results["current_fast_box_list"], atol=1e-4, rtol=0
    )
