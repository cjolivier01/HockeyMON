"""Exercise the real supervisor/frame-owner mailbox used by shared hstream-ui."""

import base64
import json
import time

import pytest
import yaml
from PIL import Image

from hmlib.camera.hm_ui_bridge import HmUiProcess
from hmlib.ui.hstream_runtime import RuntimeProtocol, read_document, write_document


@pytest.fixture
def runtime(tmp_path):
    ui = HmUiProcess(title="runtime test", tmpdir=tmp_path, preview_names=("Final", "Stitched"))
    ui.ensure_started = lambda: None
    ui.add_window("Tracker Controls")
    for name, maximum, value in (
        ("Apply_To_Fast_Box", 1, 0),
        ("Apply_To_Follower_Box", 1, 1),
        ("Max_Speed_X_x10", 2000, 500),
        ("Link_Fixed_Edge_Rotation_Left_Right", 1, 1),
        ("Left_Fixed_Edge_Rotation_Angle_x10", 900, 100),
        ("Right_Fixed_Edge_Rotation_Angle_x10", 900, 100),
    ):
        ui.add_slider("Tracker Controls", name, maximum, value)
    ui.add_window("Tracker Controls (Color)")
    ui.add_slider("Tracker Controls (Color)", "HStream_Exposure_x100", 130, 0)
    messages = []
    protocol = RuntimeProtocol(
        tmp_path,
        messages.append,
        graph={
            "plugins": {
                "camera": {
                    "class": "hmlib.aspen.plugins.PlayTrackerPlugin",
                    "params": {"enabled": True},
                    "depends": [],
                }
            }
        },
    )
    yield ui, protocol, messages
    ui.close()


def should_acknowledge_only_after_frame_owner_and_restore_values_on_seek(runtime):
    ui, protocol, messages = runtime
    command = "@set-property playcropper0 exposure=0.37"
    assert protocol.handle_line(command)
    protocol.poll()
    assert not any(line.startswith("runtime property") for line in messages)
    applied = []
    ui.apply_runtime_requests(lambda: applied.append(ui.control_values()) or True)
    protocol.poll()
    assert applied[0]["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37
    assert messages[-1] == "runtime property playcropper0 exposure=0.37"
    assert protocol.replay_values["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37
    ui.set_value("Tracker Controls (Color)", "HStream_Exposure_x100", 0)
    replay = {**protocol.replay_state, "preview": "Stitched", "overlays": [True, False, True]}
    write_document(ui._tmpdir / "replay-controls.json", replay)
    ui._replay_controls_restored = False
    ui.apply_runtime_requests(lambda: True)
    assert ui.control_values()["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37
    state = json.loads(ui.state_path.read_text())
    assert state["windows"]["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37
    assert state["selected_preview"] == "Stitched"
    ui.poll()
    assert ui._selected_preview_name == "Stitched"
    assert ui.preview_overlay_layers == (True, False, True)
    assert not (ui._tmpdir / "replay-controls.json").exists()


def should_validate_sparse_tuning_before_mutating_any_owner(runtime, tmp_path):
    ui, protocol, messages = runtime
    config = tmp_path / "tuning.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "play-tracker": {
                    "hstream-runtime-tuning": {"max-speed-x": 72.5},
                    "hstream-apply-to-fast-box": True,
                    "hstream-apply-to-follower-box": False,
                }
            }
        )
    )
    assert protocol.handle_line(f"@set-property dsplaytracker0 runtime-tuning-config-file={config}")
    ui.apply_runtime_requests(lambda: True)
    protocol.poll()
    controls = ui.control_values()["Tracker Controls"]
    assert controls["Max_Speed_X_x10"] == 725
    assert controls["Apply_To_Fast_Box"] == 1
    assert controls["Apply_To_Follower_Box"] == 0
    before = ui.control_values()
    assert not protocol.handle_line(
        "@set-properties playcropper0 exposure=0.31;dsplaytracker0 fixed-edge-rotation-angle=-1"
    )
    ui.apply_runtime_requests(lambda: pytest.fail("Invalid batch reached the frame owner"))
    assert before == ui.control_values()
    assert "outside its supported range" in messages[-1]


def should_rollback_a_rejected_owner_batch_and_fail_a_broken_rollback(runtime):
    ui, protocol, messages = runtime
    assert protocol.handle_line("@set-property playcropper0 exposure=0.5")
    applied = []

    def reject_new():
        value = ui.control_values()["Tracker Controls (Color)"]["HStream_Exposure_x100"]
        applied.append(value)
        return value == 0

    ui.apply_runtime_requests(reject_new)
    protocol.poll()
    assert applied == [50, 0]
    assert "rejected" in messages[-1]
    assert protocol.replay_values == {}
    assert protocol.handle_line("@set-property playcropper0 exposure=0.5")
    with pytest.raises(RuntimeError, match="rollback failed"):
        ui.apply_runtime_requests(lambda: False)


def should_reject_expired_stale_and_missing_identity_requests(runtime):
    ui, protocol, messages = runtime
    for field, value in (("deadline", time.monotonic() - 1), ("session_id", "old")):
        assert protocol.handle_line("@set-property playcropper0 exposure=0.5")
        request = next((ui._tmpdir / "runtime-requests").glob("*.json"))
        body = read_document(request)
        body[field] = value
        write_document(request, body)
        ui.apply_runtime_requests(lambda: pytest.fail("Invalid request reached owner"))
        protocol.poll()
        assert messages[-1].startswith("runtime command failed:")
    assert protocol.handle_line("@set-property playcropper0 exposure=0.5")
    request = next((ui._tmpdir / "runtime-requests").glob("*.json"))
    write_document(ui._tmpdir / "runtime-results" / request.name, {"ok": True})
    protocol.poll()
    assert "stale runtime acknowledgement" in messages[-1]


def should_activate_preview_after_owner_receipt_and_encode_png(runtime, tmp_path):
    ui, protocol, messages = runtime
    ui.publish_runtime(frame=20, fps=10)
    Image.new("RGB", (32, 20), "red").save(tmp_path / "preview-stitched.jpg")
    write_document(tmp_path / "preview-stitched.json", {"width": 320, "height": 200})
    assert protocol.handle_line("@set-preview-active stitched 1")
    protocol.poll()
    assert not any("channel=stitched status=ready" in line for line in messages)
    ui.apply_runtime_requests(lambda: True)
    protocol.poll()
    assert ui._selected_preview_name == "Stitched"
    assert any("channel=stitched status=activated generation=1" in line for line in messages)
    assert any("channel=stitched status=ready generation=1" in line for line in messages)
    destination = tmp_path / "captured.png"
    assert protocol.handle_line(f"@capture-preview-frame stitched {destination}")
    with Image.open(destination) as captured:
        assert captured.format == "PNG"
        assert captured.size == (32, 20)


def should_report_native_seek_errors_and_absolute_frame_progress(runtime):
    ui, protocol, messages = runtime
    assert not protocol.handle_line("@seek 123 7")
    assert any(line.startswith("HSTREAM_SEEK status=failed generation=7 ") for line in messages)
    protocol.start_ns = 2_000_000_000
    ui.publish_runtime(frame=50, fps=10)
    protocol.poll()
    assert any("processed_ns=3000000000" in line for line in messages)
    protocol.seek_completed(3_000_000_000, 7)
    assert messages[-1] == "HSTREAM_SEEK status=ok generation=7 position_ns=3000000000"


def should_inspect_real_graph_and_acknowledge_live_property_edit(runtime):
    ui, protocol, messages = runtime
    protocol._spec()
    assert protocol.handle_line("@inspect-pipeline 1 0 1")
    graph = json.loads(messages[-1].partition(" ")[2])
    assert graph["nodes"][0]["factory"].endswith("PlayTrackerPlugin")

    def token(value):
        return base64.b64encode(value.encode()).decode()

    assert protocol.handle_line(f"@inspect-properties 2 0 1 0 {token('camera')}")
    properties = json.loads(messages[-1].partition(" ")[2])["properties"]
    live = next(p for p in properties if p["name"].endswith("HStream_Exposure_x100"))
    assert live["editable"] is True
    assert next(p for p in properties if p["name"] == "enabled")["editable"] is False
    assert protocol.handle_line(
        f"@inspect-set-property 3 0 1 0 {token('camera')} {token(live['name'])} {token('37')}"
    )
    assert not any('"kind":"set-result"' in message for message in messages)
    ui.apply_runtime_requests(lambda: True)
    protocol.poll()
    assert json.loads(messages[-1].partition(" ")[2])["status"] == "ok"
    assert ui.control_values()["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37


def should_reject_non_regular_protocol_documents(tmp_path):
    source = tmp_path / "source.json"
    write_document(source, {"ok": True})
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(OSError):
        read_document(link)
    with pytest.raises(ValueError):
        read_document(tmp_path)


def should_apply_native_output_epoch_and_controls_as_one_transaction(
    runtime, monkeypatch, tmp_path
):
    from hmlib import config
    from hmlib.camera.play_tracker import PlayTracker
    from hmlib.stitching import native_calibration

    ui, protocol, messages = runtime
    monkeypatch.setattr(config, "get_game_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        native_calibration,
        "validate_live_output_epoch",
        lambda game, serialized: {
            "post_stitch_rotate_degrees": 2.0,
            "authorization_id": "auth",
            "output_generation": "output-2",
            "scoreboard_property_value": "0,0,0,0,0,0,0,0",
        },
    )

    class Controller:
        rotation = 0.0

        def set_post_stitch_rotate_degrees(self, degrees):
            self.rotation = degrees

        def get_post_stitch_rotate_degrees(self):
            return self.rotation

    owner = PlayTracker.__new__(PlayTracker)
    owner._game_id = "test"
    owner._game_config = {
        "stitching": {"post_stitch_rotate_degrees": 0},
        "rink": {"scoreboard": {"perspective_polygon": [[1, 1], [4, 1], [4, 4], [1, 4]]}},
    }
    owner._rink_map = None
    owner._hm_ui_process = ui
    owner._stitch_rotation_controller = Controller()
    owner._apply_current_ui_control_values = lambda: True
    assert protocol.handle_line(
        "@set-properties playcropper0 exposure=0.37;hmstitcher0 stitched-output-epoch=4:auth2"
    )
    ui.apply_runtime_requests(lambda: True, owner._apply_shared_runtime_operation)
    protocol.poll()
    assert owner._stitch_rotation_controller.rotation == 2
    assert owner._game_config["hm_shared_runtime"] == {
        "post_stitch_rotate_degrees": 2,
        "authorization_id": "auth",
        "output_generation": "output-2",
    }
    assert owner._game_config["rink"]["scoreboard"]["perspective_polygon"] == [[0, 0]] * 4
    assert ui.control_values()["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37
    assert messages[-1] == "runtime property hmstitcher0 stitched-output-epoch=4:auth2"


def should_report_actual_precision_and_reject_lost_precision():
    import torch

    from hmlib.ui.hstream_runtime import publish_precision_status

    environment = {
        "HM_SHARED_HIGH_BIT_DEPTH_MODE": "auto",
        "HM_SHARED_HIGH_BIT_DEPTH": "1",
        "HM_SHARED_SOURCE_BIT_DEPTHS": "[10, null]",
    }
    messages = []
    publish_precision_status(torch.zeros((1, 2, 3, 3)), messages.append, environment=environment)
    assert messages == [
        "HSTREAM_HIGH_BIT_DEPTH mode=auto enabled=1 sources=2 unknown=1 minimum-source-bit-depth=10"
    ]
    with pytest.raises(ValueError, match="lost source precision"):
        publish_precision_status(
            torch.zeros((1, 2, 3, 3), dtype=torch.uint8), messages.append, environment=environment
        )


def should_notify_local_audio_owner_before_native_pause_acknowledgements(runtime):
    _ui, protocol, messages = runtime
    protocol._spec()
    messages.clear()
    notifications = []
    protocol.process_paused = lambda paused: notifications.append((paused, len(messages)))
    assert protocol.handle_line("@process-pause 1")
    assert protocol.handle_line("@process-resume 1")
    assert notifications == [(True, 0), (False, 1)]
    assert not protocol.handle_line("@process-pause 1")
    assert not protocol.handle_line("@process-resume 2")
    assert len(notifications) == 2


def should_pace_source_frames_and_service_controls_during_wait(runtime, monkeypatch):
    ui, protocol, _messages = runtime
    monkeypatch.setenv("HM_SHARED_RENDER_REALTIME", "1")
    ui.enable_playback()
    clock = [100.0]
    monkeypatch.setattr("hmlib.camera.hm_ui_bridge.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "hmlib.camera.hm_ui_bridge.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    polls = []
    ui.wait_for_shared_frame(30, 30, lambda: polls.append(clock[0]))
    ui.wait_for_shared_frame(33, 30, lambda: polls.append(clock[0]))
    assert clock[0] == pytest.approx(100.1)
    assert len(polls) >= 5
    # The resume handshake must reset source/wall alignment after SIGSTOP.
    protocol._spec()
    assert protocol.handle_line("@process-pause 1")
    clock[0] += 10
    assert protocol.handle_line("@process-resume 1")
    ui.apply_runtime_requests(lambda: True)
    ui.wait_for_shared_frame(34, 30, lambda: None)
    ui.wait_for_shared_frame(37, 30, lambda: None)
    assert clock[0] == pytest.approx(110.2)
    monkeypatch.setenv("HM_SHARED_RENDER_REALTIME", "0")
    ui.wait_for_shared_frame(67, 30, lambda: None)
    assert clock[0] == pytest.approx(110.2)


def should_resume_during_model_startup_before_controls_exist(tmp_path):
    messages = []
    protocol = RuntimeProtocol(tmp_path, messages.append)
    assert protocol.handle_line("@process-pause 1")
    assert protocol.handle_line("@process-resume 1")
    assert messages[-1] == "HSTREAM_PROCESS_PAUSE status=resumed generation=1"


def should_report_recoverable_operation_errors_without_stopping_pipeline(runtime):
    ui, protocol, messages = runtime
    protocol._queue({"kind": "operation", "operation": {"kind": "scoreboard"}}, [])

    def reject(_operation):
        raise RuntimeError("Native canvas changed; reopen the selector")

    ui.apply_runtime_requests(lambda: True, reject)
    protocol.poll()
    assert "Native canvas changed" in messages[-1]
    assert protocol.handle_line("@set-property playcropper0 exposure=0.31")
    ui.apply_runtime_requests(lambda: True)
    protocol.poll()
    assert messages[-1] == "runtime property playcropper0 exposure=0.31"
