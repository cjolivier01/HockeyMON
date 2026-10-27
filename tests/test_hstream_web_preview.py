"""Exact JPEG publication through the real frame-owner/runtime mailbox."""

import hashlib
import json
import os
import threading
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from hmlib.camera.hm_ui_bridge import HmUiProcess
from hmlib.ui import hstream_web_preview as preview
from hmlib.ui.hstream_runtime import RuntimeProtocol, write_document


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
    ui.add_window("Tracker Controls")
    ui.publish_runtime(frame=1, fps=15)
    binding = {key: request[key] for key in ("run_id", "request_digest")}
    messages = []
    protocol = RuntimeProtocol(session, messages.append, web_binding=binding)
    result = SimpleNamespace(
        ui=ui, protocol=protocol, session=session, binding=binding, messages=messages
    )
    yield result
    result.ui.close()


def publish(owner, *, name="Final", **kwargs):
    owner.ui.publish_preview(
        np.full((36, 64, 3), 100, np.uint8), name=name, min_interval_seconds=0, **kwargs
    )
    assert owner.ui.flush_previews()


def read(owner, *, channel="program", generation=1):
    return preview.read_snapshot(
        owner.session,
        expected={
            **owner.binding,
            "session_id": owner.ui.session_id,
            "channel": channel,
            "generation": generation,
        },
    )


def should_bind_exact_selected_jpeg_and_preserve_desktop_dimensions(owner):
    publish(
        owner, metadata={"source_frame_id": 9, "pts_ns": 300000000}, snapshot=True, max_width=32
    )
    document, encoded = read(owner)
    assert document["source"] == {"width": 64, "height": 36}
    assert document["image"]["width"] == 32
    assert document["image"]["height"] == 18
    assert document["image"]["size"] == len(encoded)
    assert document["image"]["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert document["frame"] == {"source_frame_id": 9, "pts_ns": None}
    assert cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR).shape == (18, 32, 3)
    assert json.loads((owner.session / "preview-final.json").read_text()) == document["source"]
    assert (owner.session / "preview-final.jpg").read_bytes() == encoded
    owner.protocol.poll()
    assert any("channel=program status=ready generation=1" in message for message in owner.messages)


def should_publish_only_current_route_after_inflight_encoder_finishes(owner, monkeypatch):
    publish(owner)
    entered, release = threading.Event(), threading.Event()
    encode = cv2.imencode

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return encode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", blocked)
    owner.ui.publish_preview(np.zeros((36, 64, 3), np.uint8), name="Final", min_interval_seconds=0)
    try:
        assert entered.wait(3)
        assert owner.protocol.handle_line("@set-preview-active stitched 2")
        owner.ui.apply_runtime_requests(lambda: True)
        owner.protocol.poll()
        assert not (owner.session / preview.MANIFEST).exists()
        assert not any("channel=stitched status=ready" in message for message in owner.messages)
    finally:
        release.set()
    assert owner.ui.flush_previews()
    assert not (owner.session / preview.MANIFEST).exists()
    publish(owner, name="Stitched")
    document, _ = read(owner, channel="stitched", generation=2)
    assert document["sequence"] == 3
    owner.protocol.poll()
    assert any(
        "channel=stitched status=ready generation=2" in message for message in owner.messages
    )


def should_capture_frame_identity_before_async_encoding(owner, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    encode = cv2.imencode

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return encode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", blocked)
    metadata = {"source_frame_id": 17, "decoded_pts_ns": 333333333}
    owner.ui.publish_preview(np.zeros((36, 64, 3), np.uint8), name="Final", metadata=metadata)
    try:
        assert entered.wait(3)
        metadata.update(source_frame_id=91, decoded_pts_ns=999999999)
    finally:
        release.set()
    assert owner.ui.flush_previews()
    document, _ = read(owner)
    assert document["frame"] == {"source_frame_id": 17, "pts_ns": 333333333}


def should_bind_lossless_chapter_identity_before_encoding_without_changing_v1(owner):
    from fractions import Fraction

    from hmlib.video.frame_timing import DecodedTimestamp

    timing = DecodedTimestamp(9009, Fraction(1, 90000)).document(0)
    metadata = {"decoded_source_role": "left", "decoded_timing": timing, "source_frame_id": 8}
    bound = owner.ui._web_preview.bind_frame_metadata("Final", metadata)
    timing["pts"] = "0"
    assert bound["decoded_source"] == {
        "role": "left",
        "chapter_index": 0,
        "content": {"size": 1, "sha256": "b" * 64},
        "pts": "9009",
        "time_base_num": 1,
        "time_base_den": 90000,
    }
    publish(owner, metadata=bound)
    document, _ = read(owner)
    assert document["version"] == 1
    assert document["frame"] == {"source_frame_id": 8, "pts_ns": 100100000}
    with pytest.raises(ValueError, match="source"):
        owner.ui._web_preview.bind_frame_metadata("Camera 2", bound)
    with pytest.raises(ValueError, match="chapter"):
        owner.ui._web_preview.bind_frame_metadata(
            "Final", {**bound, "decoded_timing": {**bound["decoded_timing"], "chapter_index": 1}}
        )


def should_retire_same_route_old_generation_and_none_without_claiming_pixels(owner):
    publish(owner)
    assert owner.protocol.handle_line("@set-preview-active program 2")
    owner.ui.apply_runtime_requests(lambda: True)
    owner.protocol.poll()
    assert not (owner.session / preview.MANIFEST).exists()
    publish(owner)
    read(owner, generation=2)
    assert owner.protocol.handle_line("@set-preview-active none 3")
    owner.ui.apply_runtime_requests(lambda: True)
    owner.protocol.poll()
    publish(owner)
    assert not (owner.session / preview.MANIFEST).exists()
    assert not list(owner.session.glob("preview-web-*.jpg"))
    assert not any("channel=none status=ready" in message for message in owner.messages)


def should_bound_retained_snapshots_and_remove_the_manifest_first(owner, monkeypatch):
    for _ in range(6):
        publish(owner)
        assert len(list(owner.session.glob("preview-web-*.jpg"))) <= 2
    document, _ = read(owner)
    assert document["sequence"] == 6
    from pathlib import Path

    unlink = Path.unlink

    def checked(path, **kwargs):
        if path.name.startswith("preview-web-"):
            assert not (owner.session / preview.MANIFEST).exists()
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", checked)
    owner.ui._select_preview("none", 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", str(uuid4())),
        ("request_digest", "b" * 64),
        ("session_id", "c" * 32),
        ("channel", "stitched"),
        ("generation", 2),
        ("sequence", True),
    ],
)
def should_reject_wrong_snapshot_identity_before_readiness(owner, field, value):
    publish(owner)
    path = owner.session / preview.MANIFEST
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read(owner)
    owner.protocol.poll()
    assert not any("message=frame-ready" in message for message in owner.messages)


@pytest.mark.parametrize("kind", ["missing", "replaced", "symlink", "fifo", "oversized"])
def should_retry_without_painting_missing_or_changed_jpeg(owner, kind):
    publish(owner)
    document, _ = read(owner)
    image = owner.session / document["image"]["path"]
    image.unlink()
    if kind == "replaced":
        image.write_bytes(b"0" * document["image"]["size"])
    elif kind == "symlink":
        image.symlink_to(owner.session / "preview-final.jpg")
    elif kind == "fifo":
        os.mkfifo(image)
    elif kind == "oversized":
        image.write_bytes(b"0" * (preview.MAX_IMAGE_BYTES + 1))
    with pytest.raises((ValueError, OSError)):
        read(owner)
    owner.protocol.poll()
    assert not any("message=frame-ready" in message for message in owner.messages)


def should_fence_seek_replacement_and_restore_acknowledged_generation(owner):
    assert owner.protocol.handle_line("@set-preview-active stitched 7")
    owner.ui.apply_runtime_requests(lambda: True)
    owner.protocol.poll()
    publish(owner, name="Stitched")
    old_session = owner.ui.session_id
    old_ui = owner.ui
    owner.protocol.reset_child(1000000000)
    assert not (owner.session / preview.MANIFEST).exists()
    owner.ui.close()
    write_document(owner.session / "replay-controls.json", owner.protocol.replay_state)
    owner.ui = HmUiProcess(preview_names=("Final", "Stitched", "Camera 1", "Camera 2"))
    owner.ui.ensure_started = lambda: None
    owner.ui.add_window("Tracker Controls")
    owner.ui.apply_runtime_requests(lambda: True)
    owner.ui.publish_runtime(frame=31, fps=15)
    publish(owner, name="Stitched")
    document, _ = read(owner, channel="stitched", generation=7)
    assert document["session_id"] != old_session and document["sequence"] == 1
    old_ui.close()
    assert read(owner, channel="stitched", generation=7)[0] == document
    owner.protocol.poll()
    assert any(
        "channel=stitched status=ready generation=7" in message for message in owner.messages
    )


def should_enforce_shared_desktop_decoded_pixel_bound(owner):
    publish(owner)
    document, _ = read(owner)
    document["image"].update(width=4096, height=1025)
    with pytest.raises(ValueError, match="pixel bound"):
        preview.validate_manifest(document)


@pytest.mark.parametrize("state", ["missing", "retired", "none", "old-session"])
def should_require_the_applied_live_owner_before_returning_pixels(owner, state):
    publish(owner)
    path = owner.session / preview.OWNER
    document = json.loads(path.read_text())
    if state == "missing":
        path.unlink()
    else:
        if state == "retired":
            document["status"] = "retired"
        elif state == "none":
            document["channel"] = "none"
        else:
            document["session_id"] = "b" * 32
        path.write_text(json.dumps(document))
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            read(owner)
    finally:
        document.update(status="selected", channel="program", session_id=owner.ui.session_id)
        path.write_text(json.dumps(document))


def should_recheck_owner_after_reading_pinned_jpeg_bytes(owner, monkeypatch):
    publish(owner)
    original = preview.PinnedFile.read

    def retire_after_read(pin, maximum):
        data = original(pin, maximum)
        if pin.relative.endswith(".jpg"):
            path = owner.session / preview.OWNER
            owner_document = json.loads(path.read_text())
            path.write_text(json.dumps({**owner_document, "status": "retired"}))
        return data

    monkeypatch.setattr(preview.PinnedFile, "read", retire_after_read)
    with pytest.raises(ValueError, match="owner changed"):
        read(owner)


def should_retire_owner_in_supervisor_with_last_route_identity(owner):
    publish(owner)
    before = preview.read_owner(owner.session, expected=owner.binding)
    owner.protocol.stopped()
    after = preview.read_owner(owner.session, expected=owner.binding)
    assert after == {**before, "status": "retired"}
    assert not (owner.session / preview.MANIFEST).exists()


def should_map_sampled_source_indices_without_guessing_decoded_pts():
    from hmlib.ui.preview_overlays import source_metadata

    context = {
        "source_frame_ids": [10, 12],
        "frame_ids": [7, 8],
        "pts_ns": [1000000000, 1200000000],
    }
    assert source_metadata(context, 0) == {"source_frame_id": 10}
    assert source_metadata(context) == {"source_frame_id": 12}
    assert source_metadata({"frame_ids": [7, 8], "fps": 5}) == {}
    context["decoded_pts_ns"] = [1002000000, 1203000000]
    assert source_metadata(context) == {"source_frame_id": 12, "decoded_pts_ns": 1203000000}


def should_publish_actual_pts_without_claiming_a_source_ordinal_after_seek(owner):
    from hmlib.ui.preview_overlays import source_metadata

    captured = source_metadata(
        {
            "source_frame_ids": [None],
            "frame_ids": [12],
            "pts_ns": [1200000000],
            "decoded_pts_ns": [1200000000],
            "decoded_source_role": "left",
            "decoded_timing": [
                {"chapter_index": 0, "pts": "12000", "time_base_num": 1, "time_base_den": 10000}
            ],
        }
    )
    publish(owner, metadata=captured)
    document, _ = read(owner)
    assert document["frame"] == {"source_frame_id": None, "pts_ns": 1200000000}


@pytest.mark.parametrize("bad", [[True], [1.0], [-1], [], [2**63]])
def should_reject_malformed_source_indices(bad):
    from hmlib.ui.preview_overlays import source_metadata

    with pytest.raises(ValueError):
        source_metadata({"source_frame_ids": bad})
