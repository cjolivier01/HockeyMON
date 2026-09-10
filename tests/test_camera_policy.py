from __future__ import annotations

import copy
import csv

import numpy as np
import pandas as pd
import pytest
import torch

from hmlib.aspen.plugins.save_plugins import SaveCameraPlugin
from hmlib.camera.camera_gpt_dataset import (
    CameraPanZoomGPTIterableDataset,
    GameCsvPaths,
    _load_game,
)
from hmlib.camera.camera_model_dataset import CameraPanZoomDataset
from hmlib.camera.camera_policy import (
    CameraPolicyRecorder,
    camera_policy_path,
    read_camera_policy_boundaries,
)
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.utils import output_publication


def _export(directory, changes=(1, 5, 9), label=None):
    directory.mkdir(exist_ok=True)
    recorder = CameraPolicyRecorder()
    plugin = SaveCameraPlugin(write_interval=2)
    rows = []
    for batch in (range(1, 7), range(7, 13)):
        events = []
        for frame in batch:
            policy = {"speed": max(change for change in changes if change <= frame)}
            event = recorder.record(frame, policy)
            if event is not None:
                events.append(event)
            rows.append(f"{frame},1,10,5,10,20,0.9,0,1,\n")
        boxes = torch.tensor([[frame, 0, frame + 10, 5] for frame in batch], dtype=torch.float32)
        plugin.forward(
            {
                "work_dir": str(directory),
                "frame_id": batch.start,
                "frame_ids": torch.tensor(list(batch)),
                "current_box": boxes,
                "current_fast_box_list": boxes,
                "camera_policy_events": events,
                "output_label": label,
            }
        )
    plugin.finalize()
    prefix = f"{label}_" if label else ""
    (directory / f"{prefix}tracking.csv").write_text("".join(rows))
    return directory / f"{prefix}camera.csv"


def _paths(directory, suffix=""):
    return GameCsvPaths(
        game_id="game",
        tracking_csv=str(directory / f"tracking{suffix}.csv"),
        camera_csv=str(directory / f"camera{suffix}.csv"),
        camera_fast_csv=str(directory / f"camera_fast{suffix}.csv"),
    )


def should_export_all_rows_and_keep_both_training_models_inside_policy_runs(tmp_path):
    camera = _export(tmp_path)
    assert read_camera_policy_boundaries(camera, range(1, 13)) == {1, 5, 9}
    for name in ("camera.csv", "camera_fast.csv", "tracking.csv"):
        assert pd.read_csv(tmp_path / name, header=None)[0].tolist() == list(range(1, 13))
    norm = CameraNorm(scale_x=100, scale_y=50, max_players=22)
    loaded = _load_game(_paths(tmp_path), norm, "slow_tlwh", False, False)
    assert loaded.frame_runs == [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))]
    dataset = CameraPanZoomGPTIterableDataset(
        [_paths(tmp_path)], norm, seq_len=4, target_mode="slow_tlwh", feature_mode="base_prev_y"
    )
    samples = iter(dataset)
    for _ in range(20):
        sample = next(samples)
        first_frame = round(float(sample["y"][0, 0]) * 100)
        assert first_frame in {1, 5, 9}
        assert torch.equal(sample["prev0"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    model_dataset = CameraPanZoomDataset(str(tmp_path / "tracking.csv"), str(camera), window=2)
    assert [model_dataset.frames[index] for index in model_dataset.valid_indices] == [
        3,
        4,
        7,
        8,
        11,
        12,
    ]


def should_clear_transformer_previous_camera_state_at_policy_boundary(tmp_path, monkeypatch):
    camera = _export(tmp_path)
    dataset = CameraPanZoomDataset(str(tmp_path / "tracking.csv"), str(camera), window=2)
    import hmlib.camera.camera_model_dataset as module

    previous = []
    build = module.build_frame_features

    def capture(**kwargs):
        previous.append(kwargs["prev_cam_center"])
        return build(**kwargs)

    monkeypatch.setattr(module, "build_frame_features", capture)
    dataset[2]  # Features at frames 5/6 predict frame 7.
    assert previous[0] is None and previous[1] is not None


def should_preserve_label_and_generation_when_publishing_policy_companion(tmp_path):
    work, game = tmp_path / "work", tmp_path / "game"
    _export(work, label="experiment")
    source_files = {path.name: path for path in work.glob("*.csv")}
    published = output_publication.publish_artifacts(source_files, game, suffix=7)
    camera = published.files["experiment_camera.csv"]
    assert camera.name == "experiment_camera-7.csv"
    assert camera_policy_path(camera).name == "experiment_camera_policy-7.csv"
    assert read_camera_policy_boundaries(camera, range(1, 13)) == {1, 5, 9}


def should_keep_legacy_csvs_usable_without_policy_companion(tmp_path):
    camera = _export(tmp_path)
    camera_policy_path(camera).unlink()
    assert read_camera_policy_boundaries(camera, range(1, 13)) == set()
    loaded = _load_game(_paths(tmp_path), CameraNorm(100, 50, 22), "slow_tlwh", False, False)
    assert loaded.frame_runs == [list(range(1, 13))]
    assert len(CameraPanZoomDataset(str(tmp_path / "tracking.csv"), str(camera), window=2)) == 10


def should_split_fast_camera_only_training_on_its_own_paired_policy(tmp_path):
    _export(tmp_path)
    fast = tmp_path / "camera_fast.csv"
    assert read_camera_policy_boundaries(fast, range(1, 13)) == {1, 5, 9}
    paths = GameCsvPaths("fast", str(tmp_path / "tracking.csv"), str(fast))
    loaded = _load_game(paths, CameraNorm(100, 50, 22), "slow_tlwh", False, False)
    assert loaded.frame_runs == [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))]
    assert len(CameraPanZoomDataset(paths.tracking_csv, str(fast), window=2)) == 6


def should_use_fast_boundaries_when_joint_training_has_no_slow_companion(tmp_path):
    camera = _export(tmp_path)
    camera_policy_path(camera).unlink()
    loaded = _load_game(_paths(tmp_path), CameraNorm(100, 50, 22), "slow_fast_tlwh", False, False)
    assert loaded.frame_runs == [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))]


def should_reject_malformed_fast_provenance_in_joint_training(tmp_path):
    _export(tmp_path)
    camera_policy_path(tmp_path / "camera_fast.csv").write_text("bad event\n")
    with pytest.raises(ValueError, match="policy"):
        _load_game(_paths(tmp_path), CameraNorm(100, 50, 22), "slow_fast_tlwh", False, False)


def should_pair_custom_slow_and_fast_output_filenames(tmp_path):
    plugin = SaveCameraPlugin(output_filename="slow.csv", fast_output_filename="quick.csv")
    event = CameraPolicyRecorder().record(1, {"speed": 1})
    plugin.forward(
        {
            "work_dir": str(tmp_path),
            "frame_id": 1,
            "current_box": np.array([[0, 0, 10, 5]]),
            "current_fast_box_list": np.array([[0, 0, 10, 5]]),
            "camera_policy_events": [event],
        }
    )
    plugin.finalize()
    for filename in ("slow.csv", "quick.csv"):
        assert read_camera_policy_boundaries(tmp_path / filename, [1]) == {1}


def should_freeze_startup_changes_and_reset_without_noop_boundaries():
    recorder = CameraPolicyRecorder()
    policy = {"speed": 1, "controls": {"delay": 3}}
    startup = recorder.record(10, policy)
    assert startup["kind"] == "startup"
    assert recorder.record(11, copy.deepcopy(policy)) is None
    policy["speed"] = 1.0
    assert recorder.record(12, policy) is None
    policy["controls"]["delay"] = 7
    change = recorder.record(13, policy)
    assert startup["policy"]["controls"]["delay"] == 3
    assert change["kind"] == "change" and change["frame"] == 13
    reset = recorder.record(14, startup["policy"])
    assert reset["kind"] == "change" and reset["frame"] == 14
    assert recorder.record(15, startup["policy"]) is None


@pytest.mark.parametrize("content", ["", "bad row\n", "1,{}\n", '1,"{""schema"":""wrong""}"\n'])
def should_fail_training_on_malformed_policy_companions(tmp_path, content):
    camera = _export(tmp_path)
    camera_policy_path(camera).write_text(content)
    with pytest.raises(ValueError, match="policy"):
        _load_game(_paths(tmp_path), CameraNorm(100, 50, 22), "slow_tlwh", False, False)
    with pytest.raises(ValueError, match="policy"):
        CameraPanZoomDataset(str(tmp_path / "tracking.csv"), str(camera))


def should_reject_policy_events_for_unexported_frames(tmp_path):
    camera = _export(tmp_path)
    with camera_policy_path(camera).open(newline="") as stream:
        events = list(csv.reader(stream))
    events[-1][0] = "99"
    with camera_policy_path(camera).open("w", newline="") as stream:
        csv.writer(stream).writerows(events)
    with pytest.raises(ValueError, match="matching camera frame"):
        read_camera_policy_boundaries(camera, range(1, 13))


def should_propagate_policy_sync_failure_and_attempt_camera_finalization(tmp_path, monkeypatch):
    import hmlib.datasets.dataframe as module

    plugin = SaveCameraPlugin()
    event = CameraPolicyRecorder().record(1, {"speed": 1})
    original_sync = module.os.fsync
    monkeypatch.setattr(
        module.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("policy disk full"))
    )
    with pytest.raises(OSError, match="policy disk full"):
        plugin.forward(
            {
                "work_dir": str(tmp_path),
                "frame_id": 1,
                "current_box": np.array([[0, 0, 10, 5]]),
                "camera_policy_events": [event],
            }
        )
    monkeypatch.setattr(module.os, "fsync", original_sync)
    with pytest.raises(RuntimeError, match="previously failed"):
        plugin.finalize()
    assert (tmp_path / "camera.csv").exists()


@pytest.mark.parametrize("event_frame", [0, 2])
def should_reject_misaligned_startup_before_writing_camera_rows(tmp_path, event_frame):
    plugin = SaveCameraPlugin()
    event = CameraPolicyRecorder().record(event_frame, {"speed": 1})
    with pytest.raises(ValueError, match="policy"):
        plugin.forward(
            {
                "work_dir": str(tmp_path),
                "frame_id": 1,
                "current_box": np.array([[0, 0, 10, 5]]),
                "camera_policy_events": [event],
            }
        )
    assert not (tmp_path / "camera.csv").exists()


def should_use_explicit_source_frame_ids_instead_of_batch_arithmetic(tmp_path):
    recorder = CameraPolicyRecorder()
    plugin = SaveCameraPlugin()
    plugin.forward(
        {
            "work_dir": str(tmp_path),
            "frame_id": 0,
            "frame_ids": torch.tensor([100, 102]),
            "current_box": np.array([[0, 0, 10, 5], [1, 0, 11, 5]]),
            "camera_policy_events": [recorder.record(100, {"speed": 1})],
        }
    )
    plugin.finalize()
    assert pd.read_csv(tmp_path / "camera.csv", header=None)[0].tolist() == [100, 102]
    assert read_camera_policy_boundaries(tmp_path / "camera.csv", [100, 102]) == {100}


def should_emit_policy_changes_before_their_exact_frames_in_real_tracker_batches(monkeypatch):
    from test_play_tracker_parity import _base_game_config, _build_tracker, _make_results

    tracker = _build_tracker(_base_game_config(), {}, cpp_playtracker=False)
    changes = iter([1.0, 2.0, 2.0])

    def apply():
        tracker._game_config["rink"]["camera"]["max_speed_ratio_x"] = next(changes)

    monkeypatch.setattr(tracker, "_apply_ui_controls", apply)
    results = tracker.forward(_make_results(3))
    events = results["camera_policy_events"]
    assert [event["frame"] for event in events] == [0, 1]
    assert [event["policy"]["camera"]["max_speed_ratio_x"] for event in events] == [1, 2]
    assert results["frame_ids"].tolist() == [0, 1, 2]


def should_capture_applied_geometry_when_affected_batches_arrive(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_play_tracker_parity import _base_game_config, _build_tracker, _make_results

    tracker = _build_tracker(_base_game_config(), {}, cpp_playtracker=False)
    tracker._stitch_rotation_controller = SimpleNamespace(post_stitch_rotate_degrees=0.0)
    requests = iter([0.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        tracker,
        "_apply_ui_controls",
        lambda: tracker._set_stitch_rotation_degrees(next(requests)),
    )
    calculate = tracker.get_cluster_boxes
    applied = []

    def advance(*args, **kwargs):
        applied.append(tracker._camera_policy_recorder._previous["post_stitch_rotate_degrees"])
        return calculate(*args, **kwargs)

    monkeypatch.setattr(tracker, "get_cluster_boxes", advance)
    saver = SaveCameraPlugin()
    events = []
    for start, rotation in ((0, 0.0), (3, 4.0), (6, 0.0)):
        inputs = _make_results(3)
        for frame in inputs["data_samples"].video_data_samples:
            frame.frame_id += start
        # This is the rotation actually applied by the upstream stitch operation.
        inputs["camera_input_geometry"] = {"post_stitch_rotate_degrees": rotation}
        output = tracker.forward(inputs)
        events.extend(output["camera_policy_events"])
        saver.forward({**output, "work_dir": str(tmp_path), "frame_id": start})
    saver.finalize()
    assert applied == [0, 0, 0, 4, 4, 4, 0, 0, 0]
    assert [event["frame"] for event in events] == [0, 3, 6]
    assert events[1]["policy"]["canvas_wh"] == [960, 540]
    assert events[1]["policy"]["play_box"] == [0, 0, 1920, 1080]
    assert read_camera_policy_boundaries(tmp_path / "camera.csv", range(9)) == {0, 3, 6}
    assert pd.read_csv(tmp_path / "camera.csv", header=None)[0].tolist() == list(range(9))


def should_pin_legacy_geometry_instead_of_reading_later_controller_requests():
    from types import SimpleNamespace
    from test_play_tracker_parity import _base_game_config, _build_tracker

    tracker = _build_tracker(_base_game_config(), {}, cpp_playtracker=False)
    assert (
        tracker._record_camera_policy(0, (1920, 1080))["policy"]["post_stitch_rotate_degrees"] == 0
    )
    tracker._stitch_rotation_controller = SimpleNamespace(post_stitch_rotate_degrees=4.0)
    assert tracker._record_camera_policy(1, (1920, 1080)) is None


def should_attach_rendered_rotation_even_if_request_changes_during_stitching(monkeypatch):
    from types import SimpleNamespace
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    plugin = StitchingPlugin(post_stitch_rotate_degrees=2.0)
    monkeypatch.setattr(plugin, "_ensure_initialized", lambda context: None)
    monkeypatch.setattr(plugin, "_create_stitcher", lambda **kwargs: None)
    plugin._stitcher = SimpleNamespace(forward=lambda inputs: inputs[0].clone())
    rotate = plugin._rotate_tensor_keep_size
    angles = []

    def render(image, degrees):
        angles.append(degrees)
        result = rotate(image, degrees)
        plugin.set_post_stitch_rotate_degrees(9.0)
        return result

    monkeypatch.setattr(plugin, "_rotate_tensor_keep_size", render)
    view = {"img": torch.rand(1, 8, 12, 3), "frame_ids": torch.tensor([0])}
    context = {"stitch_inputs": {"left": view, "right": view}}
    first = plugin.forward(context)
    second = plugin.forward(context)
    assert angles == [2, 9]
    assert first["camera_input_geometry"] == {"post_stitch_rotate_degrees": 2}
    assert second["camera_input_geometry"] == {"post_stitch_rotate_degrees": 9}


def should_ignore_color_changes_and_repeat_controls_but_capture_applied_target_changes():
    from test_play_tracker_parity import _base_game_config, _build_tracker

    tracker = _build_tracker(_base_game_config(), {}, cpp_playtracker=False)
    assert tracker._record_camera_policy(0, (1920, 1080)) is not None
    tracker._game_config["rink"]["camera"]["color"]["brightness"] = 2
    assert tracker._record_camera_policy(1, (1920, 1080)) is None
    tracker._applied_camera_targets = {"fast": True, "follower": False}
    assert tracker._record_camera_policy(2, (1920, 1080)) is not None
    assert tracker._record_camera_policy(3, (1920, 1080)) is None
    assert tracker._record_camera_policy(4, (1280, 720)) is not None
