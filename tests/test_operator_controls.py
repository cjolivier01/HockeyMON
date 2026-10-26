import copy
import json
from types import SimpleNamespace

import pytest

from hmlib.camera.hm_ui_bridge import HmUiProcess
from hmlib.camera.operator_controls import OperatorControls


def make_tracker(tmp_path, native=True):
    config = {
        "rink": {
            "tracking": {
                "cam_ignore_largest": True,
                "cam_ignore_largest_count": 2,
                "cam_ignore_oversized": False,
                "cam_oversized_percent": 123.456,
            }
        }
    }
    process = HmUiProcess(tmpdir=tmp_path)
    process.ensure_started = lambda: None
    calls = []
    tracker = SimpleNamespace(
        _hm_ui_process=process,
        _game_config=config,
        _camera_controller="rule",
        _system_game_config=copy.deepcopy(config),
        _cam_ignore_largest=True,
        _cam_ignore_largest_count=2,
        _cam_ignore_oversized=False,
        _cam_oversized_percent=123.456,
        _stitch_rotation_controller=None,
        _plot_individual_player_tracking=False,
        _plot_moving_boxes=False,
        _plot_tracking_circles=True,
        _plot_trajectories=False,
        _plot_speed=False,
        _plot_boundaries=False,
        _playtracker=(
            SimpleNamespace(set_player_size_filter=lambda *v: calls.append(v)) if native else None
        ),
    )

    def set_value(path, value):
        parent = config
        for key in path[:-1]:
            parent = parent.setdefault(key, {})
        parent[path[-1]] = value

    tracker._set_ui_config_value = set_value
    return tracker, calls


@pytest.mark.parametrize("native", [False, True])
def should_change_filters_live_and_restore_exact_config(tmp_path, native):
    tracker, calls = make_tracker(tmp_path, native)
    controls = OperatorControls(tracker)
    before = copy.deepcopy(tracker._game_config)
    controls.apply()
    assert tracker._game_config == before
    assert not calls
    controls.process.set_value("Player Filtering", "Ignore_Largest_Count", 0)
    controls.apply()
    assert not tracker._cam_ignore_largest
    assert tracker._cam_oversized_percent == 123.456
    if native:
        assert calls == [(0, False, 123.456)]
    controls.process.set_value("Player Filtering", "Ignore_Oversized_Players", 1)
    controls.process.set_value("Player Filtering", "Oversized_Player_Percent_x10", 1750)
    controls.apply()
    assert tracker._cam_ignore_oversized
    assert tracker._cam_oversized_percent == 175
    controls.restore(before)
    assert tracker._cam_ignore_largest_count == 2
    assert tracker._cam_oversized_percent == 123.456
    if native:
        assert calls[-1] == (2, False, 123.456)


def should_apply_overlay_to_runtime_and_persist_only_changed_field(tmp_path):
    tracker, _ = make_tracker(tmp_path)
    controls = OperatorControls(tracker)
    controls.process.set_value("Program Overlays", "Player_Boxes", 1)
    controls.apply()
    assert tracker._plot_individual_player_tracking
    assert tracker._game_config["plot"] == {"plot_individual_player_tracking": True}


def should_route_live_overlays_to_the_graph_owner_and_restore_without_double_drawing(tmp_path):
    import torch

    from hmlib.aspen.plugins.overlay_plugin import OverlayPlugin
    from hmlib.utils.gpu import unwrap_tensor

    tracker, _ = make_tracker(tmp_path)
    tracker._game_config["plot"] = {
        "plot_individual_player_tracking": True,
        "plot_tracking_circles": False,
    }
    tracker._game_config["aspen"] = {
        "plugins": {
            "overlays": {
                "class": "hmlib.aspen.plugins.overlay_plugin.OverlayPlugin",
                "params": {"plot_tracking": True, "draw_tracking_circles": False},
            }
        }
    }
    controls = OperatorControls(tracker)
    assert controls.process.get_value("Program Overlays", "Player_Boxes", poll=False) == 1
    assert controls.process.get_value("Program Overlays", "Tracking_Circles", poll=False) == 0
    owner = OverlayPlugin(plot_tracking=True, draw_tracking_circles=False, print_available=False)
    tracks = SimpleNamespace(
        instances_id=torch.tensor([1]),
        bboxes=torch.tensor([[20.0, 20.0, 60.0, 80.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
    )
    sample = SimpleNamespace(
        video_data_samples=[SimpleNamespace(pred_track_instances=tracks, frame_id=1)]
    )
    original = torch.full((1, 100, 120, 3), 200, dtype=torch.uint8)
    context = {
        "img": original,
        "data_samples": sample,
        "shared": {"overlay_controls": controls.overlay_values},
    }
    rendered = unwrap_tensor(owner.forward(context)["img"])
    assert torch.any(rendered != original)
    assert not tracker._plot_individual_player_tracking
    assert not tracker._plot_trajectories
    assert not tracker._plot_tracking_circles
    before = copy.deepcopy(tracker._game_config)
    controls.process.set_value("Program Overlays", "Player_Boxes", 0)
    controls.apply()
    assert owner.forward(context) == {}
    controls.process.set_value("Program Overlays", "Player_Trajectories", 1)
    controls.process.set_value("Program Overlays", "Tracking_Circles", 1)
    controls.apply()
    owner.forward(context)
    assert owner._plot_trajectories and owner._draw_tracking_circles
    assert not tracker._plot_individual_player_tracking and not tracker._plot_trajectories
    controls.restore(before)
    owner.forward(context)
    assert (
        owner._plot_tracking and not owner._plot_trajectories and not owner._draw_tracking_circles
    )
    assert not tracker._plot_individual_player_tracking and not tracker._plot_tracking_circles


def should_process_controls_during_pause_and_step_exactly_one_batch(tmp_path):
    process = HmUiProcess(tmpdir=tmp_path)
    process.enable_playback()
    assert process.apply_playback_action("pause")
    polls = []

    def poll():
        polls.append(True)
        if len(polls) == 2:
            process.apply_playback_action("step")

    process.wait_for_playback(poll)
    assert process.paused
    assert process._step_batches == 0
    assert len(polls) == 2
    process.wait_for_playback(lambda: process.apply_playback_action("resume"))
    assert not process.paused
    process.apply_playback_action("stop")
    process.apply_playback_action("resume")
    assert process.stop_requested
    assert json.loads(process.runtime_path.read_text())["state"] == "stopping"


def should_keep_attached_desktop_files_and_publish_finished_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_UI_SESSION_DIR", str(tmp_path))
    process = HmUiProcess()
    process.add_window("Test")
    process.enable_playback()
    process.publish_runtime(frame=42, fps=30, force=True)
    process.close()
    assert process.spec_path.exists()
    status = json.loads(process.runtime_path.read_text())
    assert status["state"] == "finished"
    assert status["frame"] == 42
    assert status["capabilities"] == []


def should_apply_supervisor_stop_without_overwriting_rust_state(tmp_path):
    process = HmUiProcess(tmpdir=tmp_path)
    process.enable_playback()
    state = {"windows": {"untouched": {"value": 123}}, "actions": []}
    process.state_path.write_text(json.dumps(state))
    requests = tmp_path / "operator-actions"
    requests.mkdir()
    (requests / "stop-one.json").write_text(json.dumps({"kind": "stop"}))
    process.poll()
    assert process.stop_requested
    assert json.loads(process.state_path.read_text()) == state
    assert json.loads((tmp_path / "operator-results/stop-one.json").read_text())["ok"]
    (requests / "stop-one.json").write_text(json.dumps({"kind": "rate:4"}))
    process.poll()
    assert process.playback_rate == 1
    (requests / "invalid.json").write_text(json.dumps({"kind": "rate:-1"}))
    process.poll()
    assert not json.loads((tmp_path / "operator-results/invalid.json").read_text())["ok"]


def should_pace_playback_at_selected_source_rate_and_allow_unbounded_render(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("hmlib.camera.hm_ui_bridge.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "hmlib.camera.hm_ui_bridge.time.sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )
    process = HmUiProcess(tmpdir=tmp_path)
    process.enable_playback()
    process.publish_runtime(frame=1, fps=10)
    process.wait_for_playback(lambda: None)
    process.publish_runtime(frame=2, fps=10)
    process.wait_for_playback(lambda: None)
    assert now[0] == pytest.approx(0.1)
    process.apply_playback_action("rate:2")
    process.wait_for_playback(lambda: None)
    process.publish_runtime(frame=3, fps=10)
    process.wait_for_playback(lambda: None)
    assert now[0] == pytest.approx(0.15)
    process.apply_playback_action("rate:0")
    process.publish_runtime(frame=4, fps=10)
    process.wait_for_playback(lambda: None)
    assert now[0] == pytest.approx(0.15)


def should_toggle_roles_without_removing_published_tracks(tmp_path):
    tracker, _ = make_tracker(tmp_path)
    controls = OperatorControls(tracker)
    controls.process.set_value("Player Filtering", "Ignore_Referees", 1)
    controls.apply()
    assert tracker._game_config["rink"]["tracking"]["cam_ignore_referees"] is True
    assert tracker._game_config["rink"]["tracking"]["cam_ignore_goalies"] is False
    controls.process.set_value("Player Filtering", "Ignore_Goalies", 1)
    controls.apply()
    assert tracker._game_config["rink"]["tracking"]["cam_ignore_goalies"] is True


def should_require_restart_for_learned_camera_role_changes(tmp_path):
    tracker, _ = make_tracker(tmp_path)
    tracker._camera_controller = "transformer"
    controls = OperatorControls(tracker)
    controls.process.set_value("Player Filtering", "Ignore_Referees", 1)
    with pytest.raises(ValueError, match="require a restart"):
        controls.apply()
    assert not tracker._game_config["rink"]["tracking"].get("cam_ignore_referees", False)
