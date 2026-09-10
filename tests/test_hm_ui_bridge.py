import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

_HAS_IMAGE_DEPS = (
    importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None
)
pytestmark = pytest.mark.skipif(
    not _HAS_IMAGE_DEPS,
    reason="cv2/numpy are not available",
)

if _HAS_IMAGE_DEPS:
    import cv2
    import numpy as np

    from hmlib.camera.hm_ui_bridge import HmUiProcess


def should_update_controls_from_hm_ui_state(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Max_Speed_X_x10", 2000, 500)

    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {"Tracker Controls": {"Max_Speed_X_x10": 725}},
        "last_action": None,
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ui.get_value("Tracker Controls", "Max_Speed_X_x10") == 725
    assert ui.last_poll_values_changed is True


def should_write_programmatic_updates_to_state_and_spec(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Brightness_Multiplier_x100", 300, 100)
    ui.set_value("Tracker Controls", "Brightness_Multiplier_x100", 145, notify=False)

    state = json.loads(ui.state_path.read_text(encoding="utf-8"))
    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    assert state["windows"]["Tracker Controls"]["Brightness_Multiplier_x100"] == 145
    control = spec["windows"][0]["controls"][0]
    assert control["name"] == "Brightness_Multiplier_x100"
    assert control["value"] == 145
    assert control["default_value"] == 145
    assert control["value_revision"] == 1


def should_not_change_defaults_for_runtime_notifications(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Brightness_Multiplier_x100", 300, 100)
    ui.set_value("Tracker Controls", "Brightness_Multiplier_x100", 145, notify=True)

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    control = spec["windows"][0]["controls"][0]
    assert control["value"] == 145
    assert control["default_value"] == 100


def should_publish_action_snapshot_with_a_newer_control_revision(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Left_Fixed_Edge_Rotation_Angle_x10", 900, 100)
    ui.set_value("Tracker Controls", "Left_Fixed_Edge_Rotation_Angle_x10", 255)
    ui.apply_control_values(
        {"Tracker Controls": {"Left_Fixed_Edge_Rotation_Angle_x10": 125}},
        publish=True,
    )

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    control = spec["windows"][0]["controls"][0]
    assert control["value"] == 125
    assert control["value_revision"] == 2


def should_keep_newer_python_value_and_accept_equal_revision_rust_edit(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None
    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Right_Fixed_Edge_Rotation_Angle_x10", 900, 100)
    ui.set_value("Tracker Controls", "Right_Fixed_Edge_Rotation_Angle_x10", 255)

    payload = {
        "windows": {"Tracker Controls": {"Right_Fixed_Edge_Rotation_Angle_x10": 100}},
        "control_revisions": {"Tracker Controls": {"Right_Fixed_Edge_Rotation_Angle_x10": 0}},
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")
    ui._last_state_mtime_ns = None
    ui.poll()
    assert ui.get_value("Tracker Controls", "Right_Fixed_Edge_Rotation_Angle_x10") == 255

    payload["windows"]["Tracker Controls"]["Right_Fixed_Edge_Rotation_Angle_x10"] = 355
    payload["control_revisions"]["Tracker Controls"]["Right_Fixed_Edge_Rotation_Angle_x10"] = 1
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")
    ui._last_state_mtime_ns = None
    ui.poll()
    assert ui.get_value("Tracker Controls", "Right_Fixed_Edge_Rotation_Angle_x10") == 355


def should_publish_grouping_and_system_defaults(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Overshoot_Stop_Delay_Frames", 60, 8)
    ui.add_slider("Tracker Controls", "Stitch_Rotate_Degrees", 180, 90)
    ui.add_slider("Tracker Controls", "Left_Fixed_Edge_Rotation_Angle_x10", 900, 250)
    ui.set_system_defaults({"Tracker Controls": {"Overshoot_Stop_Delay_Frames": 3}})

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    controls = {control["name"]: control for control in spec["windows"][0]["controls"]}
    assert controls["Overshoot_Stop_Delay_Frames"]["view"] == "Final"
    assert controls["Overshoot_Stop_Delay_Frames"]["group"] == "Breakaway Tracking"
    assert controls["Overshoot_Stop_Delay_Frames"]["default_value"] == 8
    assert controls["Overshoot_Stop_Delay_Frames"]["system_default_value"] == 3
    assert controls["Stitch_Rotate_Degrees"]["view"] == "Stitched"
    assert controls["Stitch_Rotate_Degrees"]["group"] == "Alignment"
    assert controls["Left_Fixed_Edge_Rotation_Angle_x10"]["view"] == "Final"
    assert controls["Left_Fixed_Edge_Rotation_Angle_x10"]["group"] == "Perspective Rotation"


def should_consume_hm_ui_action_snapshots_once_and_acknowledge_them(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None

    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Apply_To_Follower_Box", 1, 1)

    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
        "actions": [
            {
                "seq": 1,
                "kind": "reset-system",
                "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 0}},
            },
            {
                "seq": 2,
                "kind": "save",
                "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
            },
        ],
        "last_action": {
            "seq": 2,
            "kind": "save",
            "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
        },
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")

    events = ui.consume_action_events()
    assert [(event.seq, event.kind) for event in events] == [
        (1, "reset-system"),
        (2, "save"),
    ]
    assert events[0].values == {"Tracker Controls": {"Apply_To_Follower_Box": 0}}
    assert events[1].values == {"Tracker Controls": {"Apply_To_Follower_Box": 1}}
    assert not ui.action_ack_path.exists()
    ui.acknowledge_action_events(events[-1].seq)
    assert json.loads(ui.action_ack_path.read_text(encoding="utf-8")) == {"seq": 2}
    assert ui.last_poll_values_changed is False
    assert ui.consume_action_events() == []


def should_not_overwrite_pending_actions_with_programmatic_values(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None
    ui.add_window("Tracker Controls")
    ui.add_slider("Tracker Controls", "Apply_To_Follower_Box", 1, 1)
    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
        "actions": [
            {
                "seq": 1,
                "kind": "save",
                "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
            }
        ],
        "last_action": {
            "seq": 1,
            "kind": "save",
            "windows": {"Tracker Controls": {"Apply_To_Follower_Box": 1}},
        },
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")

    ui._process = object()
    ui.set_value("Tracker Controls", "Apply_To_Follower_Box", 0)
    ui._process = None

    state = json.loads(ui.state_path.read_text(encoding="utf-8"))
    assert state["actions"] == payload["actions"]


def should_follow_selected_preview_without_marking_controls_changed(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None
    ui.add_window("Tracker Controls")

    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {"Tracker Controls": {}},
        "selected_preview": "Final",
        "last_action": None,
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ui.poll() is False
    assert ui.last_poll_values_changed is False
    assert ui._selected_preview_name == "Final"


def should_publish_preview_frame(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 200

    ui.publish_preview(frame, min_interval_seconds=0)
    assert ui.flush_previews()

    assert ui.preview_path.exists()
    decoded = cv2.imread(str(ui.preview_path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (24, 32)


def should_publish_named_preview_frames_independently(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    ui.ensure_started = lambda: None
    ui.add_window("Stitch Alignment")
    stitched = np.zeros((24, 32, 3), dtype=np.uint8)
    final = np.zeros((18, 30, 3), dtype=np.uint8)

    ui.publish_preview(stitched, name="Stitched", min_interval_seconds=0)
    ui.publish_preview(final, name="Final", min_interval_seconds=0)
    assert ui.flush_previews()

    spec = json.loads(ui.spec_path.read_text(encoding="utf-8"))
    assert [preview["name"] for preview in spec["previews"]] == ["Stitched", "Final"]
    assert cv2.imread(str(ui.preview_paths["Stitched"])).shape[:2] == (24, 32)
    assert cv2.imread(str(ui.preview_paths["Final"])).shape[:2] == (18, 30)
    metadata = {
        preview["name"]: json.loads(Path(preview["metadata_path"]).read_text(encoding="utf-8"))
        for preview in spec["previews"]
    }
    assert metadata == {
        "Stitched": {"width": 32, "height": 24},
        "Final": {"width": 30, "height": 18},
    }


@pytest.mark.parametrize("channels_first", [False, True])
def should_report_source_resolution_before_preview_scaling_and_after_resize(
    tmp_path, channels_first
):
    import torch

    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    for height, width in [(240, 640), (180, 480)]:
        frame = np.zeros((2, height, width, 3), dtype=np.uint8)
        if channels_first:
            frame = torch.from_numpy(frame).permute(0, 3, 1, 2)
        ui.publish_preview(frame, show_scaled=0.5, max_width=128, min_interval_seconds=0)
        assert ui.flush_previews()
        assert cv2.imread(str(ui.preview_path)).shape[1] == 128
        metadata = json.loads(ui.preview_path.with_suffix(".json").read_text())
        assert metadata == {"width": width, "height": height}


def should_throttle_inactive_preview_until_it_is_selected(tmp_path, monkeypatch):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    encoded = []

    def encode(name, _job):
        encoded.append(name)

    monkeypatch.setattr(ui, "_encode_preview", encode)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    ui.publish_preview(frame, name="Final")
    assert ui.flush_previews()
    ui.publish_preview(frame, name="Final")
    assert ui.flush_previews()
    assert encoded == ["Final"]

    payload = {
        "version": 1,
        "updated_ms": int(time.time() * 1000),
        "windows": {},
        "selected_preview": "Final",
        "last_action": None,
    }
    ui.state_path.write_text(json.dumps(payload), encoding="utf-8")
    assert ui.poll() is False

    ui.publish_preview(frame, name="Final")
    assert ui.flush_previews()
    assert encoded == ["Final", "Final"]


def should_publish_latest_preview_frame_from_batch(tmp_path):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    frame = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    frame[0, :, :, 1] = 200
    frame[1, :, :, 2] = 220

    ui.publish_preview(frame, min_interval_seconds=0)
    assert ui.flush_previews()

    decoded = cv2.imread(str(ui.preview_path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (24, 32)
    assert decoded[:, :, 2].mean() > decoded[:, :, 1].mean()


def should_keep_only_latest_pending_preview_per_stream(tmp_path, monkeypatch):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    encoded = []
    gate = threading.Event()

    def encode(name, job):
        gate.wait(timeout=2.0)
        encoded.append((name, int(job.img[0, 0, 0])))

    monkeypatch.setattr(ui, "_encode_preview", encode)
    first = np.full((2, 2, 3), 1, dtype=np.uint8)
    second = np.full((2, 2, 3), 2, dtype=np.uint8)
    latest = np.full((2, 2, 3), 3, dtype=np.uint8)

    ui.publish_preview(first, min_interval_seconds=0)
    ui.publish_preview(second, min_interval_seconds=0)
    ui.publish_preview(latest, min_interval_seconds=0)
    gate.set()

    assert ui.flush_previews()
    assert encoded[-1] == ("Stitched", 3)
    assert len(encoded) <= 2


def should_preserve_pending_preview_order_when_replacing_frames(tmp_path, monkeypatch):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    encoded = []
    started = threading.Event()
    gate = threading.Event()

    def encode(name, job):
        started.set()
        gate.wait(timeout=2.0)
        encoded.append((name, int(job.img[0, 0, 0])))

    monkeypatch.setattr(ui, "_encode_preview", encode)
    ui.publish_preview(np.full((2, 2, 3), 1, dtype=np.uint8), min_interval_seconds=0)
    assert started.wait(timeout=2.0)
    ui.publish_preview(
        np.full((2, 2, 3), 2, dtype=np.uint8),
        name="Final",
        min_interval_seconds=0,
    )
    ui.publish_preview(np.full((2, 2, 3), 3, dtype=np.uint8), min_interval_seconds=0)
    ui.publish_preview(
        np.full((2, 2, 3), 4, dtype=np.uint8),
        name="Final",
        min_interval_seconds=0,
    )

    with ui._preview_condition:
        assert list(ui._pending_preview_jobs) == ["Final", "Stitched"]
    gate.set()
    assert ui.flush_previews()
    assert encoded == [("Stitched", 1), ("Final", 4), ("Stitched", 3)]


def should_keep_preview_worker_alive_after_opencv_error(tmp_path, monkeypatch):
    ui = HmUiProcess(title="test", tmpdir=tmp_path)
    encoded = []

    def encode(name, job):
        if not encoded:
            encoded.append("failed")
            raise cv2.error("invalid preview frame")
        encoded.append((name, int(job.img[0, 0, 0])))

    monkeypatch.setattr(ui, "_encode_preview", encode)
    failed = np.full((2, 2, 3), 1, dtype=np.uint8)
    recovered = np.full((2, 2, 3), 2, dtype=np.uint8)

    ui.publish_preview(failed, min_interval_seconds=0)
    assert ui.flush_previews()
    ui.publish_preview(recovered, min_interval_seconds=0)
    assert ui.flush_previews()

    assert encoded == ["failed", ("Stitched", 2)]
    assert ui._preview_worker.is_alive()
