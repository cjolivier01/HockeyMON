"""Typed web controls use real frame-owner application and route identities."""

import base64
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from hmlib.camera.hm_ui_bridge import HmUiProcess
from hmlib.ui import hstream_web_preview as preview
from hmlib.ui.hstream_runtime import read_document, write_document
from hmlib.ui.hstream_web import canonical_bytes
from hmlib.ui.hstream_web_control import ACTIONS, decode_request
from hmlib.ui.hstream_web_runtime import WebRuntimeProtocol


@pytest.fixture
def owner(tmp_path, monkeypatch):
    session = tmp_path / "work/hockeymon-session"
    session.mkdir(parents=True)
    request = {"run_id": str(uuid4()), "request_digest": "a" * 64, "launch": {"preview": "program"}}
    request["sources"] = {
        role: [{"content": {"size": 1, "sha256": "b" * 64}}] for role in ("left", "right")
    }
    monkeypatch.setattr(preview, "worker_binding", lambda: (tmp_path, request))
    monkeypatch.setenv("HM_WEB_PROGRAM_REQUEST", str(tmp_path / "hockeymon-program-request.json"))
    monkeypatch.setenv("HM_UI_SESSION_DIR", str(session))
    monkeypatch.setenv("HM_SHARED_UI_BACKEND", "1")
    ui = HmUiProcess(preview_names=("Final", "Stitched", "Camera 1", "Camera 2"))
    ui.ensure_started = lambda: None
    ui.add_window("Tracker Controls (Color)")
    ui.add_slider("Tracker Controls (Color)", "HStream_Exposure_x100", 130, 0)
    ui.enable_playback()
    ui.publish_runtime(frame=1, fps=15, force=True)
    binding = {key: request[key] for key in ("run_id", "request_digest")}
    messages, seeks, stops = [], [], []
    protocol = WebRuntimeProtocol(
        session,
        messages.append,
        web_binding=binding,
        allow_seek=True,
        realtime=True,
        seek=lambda target, generation: seeks.append((target, generation)),
        stop_applied=lambda: stops.append(True),
        total_ns=10_000_000_000,
    )
    result = SimpleNamespace(
        ui=ui,
        protocol=protocol,
        session=session,
        binding=binding,
        messages=messages,
        seeks=seeks,
        stops=stops,
        request_id=0,
    )
    yield result
    result.ui.close()


def events(owner, prefix="HM_WEB_PROGRAM_CONTROL "):
    return [json.loads(line[len(prefix) :]) for line in owner.messages if line.startswith(prefix)]


def command(owner, action, payload=None, **overrides):
    owner.request_id += 1
    document = {
        "version": 1,
        **owner.binding,
        "session_id": owner.ui.session_id,
        "request_id": owner.request_id,
        "action": action,
        "payload": {} if payload is None else payload,
        **overrides,
    }
    return "@web-control " + base64.urlsafe_b64encode(canonical_bytes(document)).decode().rstrip(
        "="
    )


def apply(owner, operation=None):
    owner.ui.apply_runtime_requests(lambda: True, operation)
    owner.protocol.poll()


def should_require_explicit_ready_and_match_runtime_owner(owner):
    assert not owner.protocol.handle_line(command(owner, "pause"))
    assert events(owner)[-1]["status"] == "failed"
    state = read_document(owner.session / "runtime.json")
    write_document(owner.session / "runtime.json", {**state, "session_id": "b" * 32})
    owner.protocol.poll()
    assert not events(owner, "HM_WEB_PROGRAM_READY ")
    write_document(owner.session / "runtime.json", state)
    owner.protocol.poll()
    ready = events(owner, "HM_WEB_PROGRAM_READY ")
    assert len(ready) == 1
    assert set(ready[0]) == {
        "version",
        "run_id",
        "request_digest",
        "session_id",
        "actions",
        "owner",
    }
    assert ready[0]["actions"] == list(ACTIONS)
    assert ready[0]["owner"] == preview.read_owner(owner.session, expected=owner.binding)
    owner.protocol.poll()
    assert len(events(owner, "HM_WEB_PROGRAM_READY ")) == 1


@pytest.mark.parametrize(
    "action,payload", [("pause", {}), ("resume", {}), ("step", {}), ("rate", {"rate": 0.5})]
)
def should_acknowledge_playback_only_after_owner_application(owner, action, payload):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, action, payload))
    owner.protocol.poll()
    assert events(owner) == []
    apply(owner)
    ack = events(owner)[-1]
    assert set(ack) == {
        "version",
        "run_id",
        "request_digest",
        "session_id",
        "request_id",
        "action",
        "status",
        "result",
        "error",
    }
    assert ack["status"] == "applied" and ack["error"] is None and ack["result"] == payload
    assert ack["session_id"] == owner.ui.session_id
    if action in {"pause", "resume", "step"}:
        assert owner.protocol.replay_state["paused"] == (action != "resume")


def should_bind_route_and_overlay_acknowledgments_to_resulting_owner(owner):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "preview", {"channel": "stitched"}))
    assert events(owner) == []
    apply(owner)
    selected = preview.read_owner(owner.session, expected=owner.binding)
    assert selected["generation"] == 2 and selected["channel"] == "stitched"
    assert events(owner)[-1]["result"] == {"channel": "stitched", "owner": selected}
    layers = {"players": True, "play": False, "rink": True}
    assert owner.protocol.handle_line(command(owner, "preview-overlays", layers))
    apply(
        owner,
        lambda operation: setattr(owner.ui, "preview_overlay_layers", tuple(operation["layers"])),
    )
    assert events(owner)[-1]["result"] == {**layers, "owner": selected}


def should_not_accept_mismatched_or_replayed_or_overlapping_requests(owner):
    owner.protocol.poll()
    first = command(owner, "pause")
    assert owner.protocol.handle_line(first)
    assert not owner.protocol.handle_line(command(owner, "resume"))
    assert "pending" in events(owner)[-1]["error"]
    apply(owner)
    assert events(owner)[-1]["status"] == "applied" and owner.ui.paused
    assert not owner.protocol.handle_line(first)
    for fields in (
        {"session_id": "b" * 32},
        {"run_id": str(uuid4())},
        {"request_digest": "b" * 64},
    ):
        assert not owner.protocol.handle_line(command(owner, "resume", **fields))
    apply(owner)
    assert owner.ui.paused
    assert len([item for item in events(owner) if item["status"] == "applied"]) == 1


def should_reject_legacy_commands_in_web_runtime(owner):
    owner.protocol.poll()
    for line in (
        "q",
        "@set-preview-active stitched 2",
        "@capture-preview-frame program /tmp/leak.jpg",
        "@set-property dsplaytracker0 runtime-tuning-config-file=/etc/passwd",
    ):
        assert not owner.protocol.handle_line(line)
    assert not list((owner.session / "runtime-requests").glob("*.json"))
    assert not events(owner)


@pytest.mark.parametrize(
    "action,payload",
    [
        ("pause", {"extra": True}),
        ("rate", {"rate": True}),
        ("rate", {"rate": 1.1}),
        ("seek", {"position_ns": -1}),
        ("seek", {"position_ns": 2**53}),
        ("preview", {"channel": "main"}),
        ("preview-overlays", {"players": 1, "play": False, "rink": False}),
        (
            "properties",
            {
                "assignments": [
                    {
                        "element": "dsplaytracker0",
                        "property": "runtime-tuning-config-file",
                        "value": "/etc/passwd",
                    }
                ]
            },
        ),
        (
            "properties",
            {"assignments": [{"element": "playcropper0", "property": "exposure", "value": "0.5"}]},
        ),
        (
            "properties",
            {"assignments": [{"element": "playcropper0", "property": "exposure", "value": 0.001}]},
        ),
        ("unknown", {}),
    ],
)
def should_fail_invalid_typed_controls_without_mutating(owner, action, payload):
    owner.protocol.poll()
    assert not owner.protocol.handle_line(command(owner, action, payload))
    assert events(owner)[-1]["status"] == "failed"
    assert events(owner)[-1]["result"] is None
    assert not list((owner.session / "runtime-requests").glob("*.json"))


def should_apply_properties_through_the_real_control_transaction(owner):
    owner.protocol.poll()
    payload = {"assignments": [{"element": "playcropper0", "property": "exposure", "value": 0.37}]}
    assert owner.protocol.handle_line(command(owner, "properties", payload))
    assert owner.ui.control_values()["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 0
    apply(owner)
    assert owner.ui.control_values()["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37
    assert events(owner)[-1]["result"] == payload
    assert owner.protocol.replay_values["Tracker Controls (Color)"]["HStream_Exposure_x100"] == 37


def should_report_actual_frame_owner_rejection_and_stale_receipt(owner):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "preview", {"channel": "source1"}))
    owner.ui.preview_paths.pop("Camera 2")
    apply(owner)
    assert events(owner)[-1]["status"] == "failed"
    assert owner.protocol._active_channel == "program"
    assert owner.protocol.handle_line(command(owner, "pause"))
    path = next((owner.session / "runtime-requests").glob("*.json"))
    write_document(
        owner.session / "runtime-results" / path.name, {"ok": True, "session_id": "b" * 32}
    )
    owner.protocol.poll()
    assert events(owner)[-1]["status"] == "failed"
    assert owner.protocol.replay_state["paused"] is False


def should_never_treat_a_negative_receipt_with_null_error_as_applied(owner):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "pause"))
    path = next((owner.session / "runtime-requests").glob("*.json"))
    write_document(owner.session / "runtime-results" / path.name, {"ok": False, "error": None})
    owner.protocol.poll()
    assert events(owner)[-1]["status"] == "failed"
    assert events(owner)[-1]["result"] is None


def should_bound_integer_properties_before_floating_conversion(owner):
    owner.protocol.poll()
    assert not owner.protocol.handle_line(
        command(
            owner,
            "properties",
            {
                "assignments": [
                    {"element": "playcropper0", "property": "exposure", "value": 10**1000}
                ]
            },
        )
    )
    assert events(owner)[-1]["status"] == "failed"


def should_retire_stop_before_terminal_ack_and_reject_later_controls(owner):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "stop"))
    assert owner.stops == []
    apply(owner)
    assert owner.stops == [True] and owner.ui.stop_requested
    record = preview.read_owner(owner.session, expected={**owner.binding, "status": "retired"})
    assert events(owner)[-1]["result"] == {"owner": record}
    assert not owner.protocol.handle_line(command(owner, "resume"))


def should_prevent_inflight_preview_from_reappearing_after_stop(owner, monkeypatch):
    owner.protocol.poll()
    entered, release = threading.Event(), threading.Event()
    encode = cv2.imencode

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return encode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", blocked)
    owner.ui.publish_preview(np.zeros((12, 16, 3), np.uint8), name="Final")
    try:
        assert entered.wait(3)
        assert owner.protocol.handle_line(command(owner, "stop"))
        apply(owner)
        assert events(owner)[-1]["result"]["owner"]["status"] == "retired"
    finally:
        release.set()
    assert owner.ui.flush_previews()
    assert not (owner.session / preview.MANIFEST).exists()
    assert not list(owner.session.glob("preview-web-*.jpg"))


def should_retain_applied_rate_in_seek_replay_and_omit_unpaced_rate(owner):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "rate", {"rate": 2}))
    apply(owner)
    replay = owner.protocol.replay_state
    assert replay["rate"] == 2
    owner.ui.playback_rate = 1
    owner.ui._replay_controls_restored = False
    write_document(owner.session / "replay-controls.json", replay)
    apply(owner)
    assert owner.ui.playback_rate == 2
    owner.protocol._realtime = False
    owner.protocol._ready_session = None
    owner.protocol.poll()
    assert "rate" not in events(owner, "HM_WEB_PROGRAM_READY ")[-1]["actions"]
    assert not owner.protocol.handle_line(command(owner, "rate", {"rate": 0.5}))


def should_seek_only_acknowledge_a_ready_replacement_session(owner):
    owner.protocol.poll()
    original_session = owner.ui.session_id
    assert owner.protocol.handle_line(command(owner, "seek", {"position_ns": 2_000_000_000}))
    assert owner.seeks == [(2_000_000_000, 1)] and events(owner) == []
    assert owner.protocol.seek_completed(2_000_000_000, 1) is False
    owner.ui.close()
    owner.protocol.reset_child(2_000_000_000)
    assert events(owner) == []
    replacement = HmUiProcess(preview_names=("Final", "Stitched"))
    replacement.ensure_started = lambda: None
    replacement.add_window("Tracker Controls")
    replacement.enable_playback()
    owner.ui = replacement
    owner.protocol.poll()
    assert len(events(owner, "HM_WEB_PROGRAM_READY ")) == 1
    assert owner.protocol.seek_completed(2_000_000_000, 1) is False
    replacement.publish_runtime(frame=30, fps=15, force=True)
    assert owner.protocol.seek_completed(2_000_000_000, 1) is True
    ack = events(owner)[-1]
    assert ack["session_id"] == original_session
    assert ack["result"]["owner"]["session_id"] == replacement.session_id
    assert ack["result"]["position_ns"] == 2_000_000_000
    assert len(events(owner, "HM_WEB_PROGRAM_READY ")) == 2


def should_fail_pending_request_on_timeout_replacement_or_stop(owner):
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "pause"))
    for pending in owner.protocol._pending.values():
        pending.deadline = 0
    owner.protocol.poll()
    assert "timed out" in events(owner)[-1]["error"]
    assert owner.protocol.handle_line(command(owner, "pause"))
    owner.protocol.reset_child()
    assert "replaced" in events(owner)[-1]["error"]
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "seek", {"position_ns": 100})) is False


def should_fail_interrupted_seek_and_omit_disallowed_seek(owner):
    owner.protocol._allow_seek = False
    owner.protocol.poll()
    assert "seek" not in events(owner, "HM_WEB_PROGRAM_READY ")[0]["actions"]
    assert not owner.protocol.handle_line(command(owner, "seek", {"position_ns": 100}))
    owner.protocol._allow_seek = True
    owner.protocol._ready_session = None
    owner.protocol.poll()
    assert owner.protocol.handle_line(command(owner, "seek", {"position_ns": 100}))
    owner.protocol.stopped("replacement failed")
    assert events(owner)[-1]["error"] == "replacement failed"


def should_reject_noncanonical_or_oversized_control_envelopes(owner):
    line = command(owner, "pause")
    assert decode_request(line)["action"] == "pause"
    value = decode_request(line)
    for bad in (
        line + "=",
        "@web-control " + "x" * 24000,
        "@web-control " + base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("="),
    ):
        assert not owner.protocol.handle_line(bad)
    assert events(owner) == []
