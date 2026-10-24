"""Playback replacement and real subprocess-tree ownership contracts."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hmlib.cli.hstream_backend import Supervisor
from hmlib.ui import process_guard
from hmlib.ui.hstream_launch import LaunchPlan, parser
from hmlib.ui.process_guard import guarded_command


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for subprocess state")


@pytest.fixture
def process_tree(tmp_path):
    script = tmp_path / "tree.py"
    script.write_text("""
import json, os, signal, subprocess, sys, time
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
if mode == 'grandchild':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / 'grandchild').write_text(str(os.getpid()))
    while True: time.sleep(.02)
child = subprocess.Popen([sys.executable, __file__, str(root), 'grandchild'])
while not (root / 'grandchild').exists(): time.sleep(.01)
(root / 'worker').write_text(str(os.getpid()))
if mode == 'crash': os._exit(7)
if mode == 'leak': sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / 'ready').write_text('ready')
while True: time.sleep(.02)
""")
    return script


def assert_reaped(root):
    for name in ("worker", "grandchild"):
        pid = int((root / name).read_text())
        assert not Path(f"/proc/{pid}").exists(), f"Leaked {name} PID {pid}"


@pytest.mark.parametrize("mode, expected", [("leak", 0), ("crash", 7), ("stubborn", 143)])
def should_reap_workers_and_descendants_on_exit_crash_and_forced_stop(
    tmp_path, process_tree, mode, expected
):
    process = subprocess.Popen(
        guarded_command([sys.executable, str(process_tree), str(tmp_path), mode]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for(lambda: (tmp_path / "worker").exists())
        assert os.getpgid(process.pid) == os.getpgrp()
        if mode == "stubborn":
            wait_for(lambda: (tmp_path / "ready").exists())
            process.terminate()
        output, _ = process.communicate(timeout=8)
        assert process.returncode == expected, output.decode()
        assert_reaped(tmp_path)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=8)


def should_reap_worker_tree_when_its_supervisor_is_killed(tmp_path, process_tree):
    launcher = tmp_path / "launcher.py"
    launcher.write_text("""
import os, subprocess, sys, time
from pathlib import Path
command = [sys.executable, sys.argv[3], '--parent-pid', str(os.getpid()), '--',
           sys.executable, sys.argv[1], sys.argv[2], 'stubborn']
child = subprocess.Popen(command)
(Path(sys.argv[2]) / 'guard').write_text(str(child.pid))
while True: time.sleep(.02)
""")
    parent = subprocess.Popen(
        [sys.executable, str(launcher), str(process_tree), str(tmp_path), process_guard.__file__]
    )
    try:
        wait_for(lambda: (tmp_path / "ready").exists())
        parent.kill()
        parent.wait(timeout=2)
        worker = int((tmp_path / "worker").read_text())
        grandchild = int((tmp_path / "grandchild").read_text())
        wait_for(
            lambda: not Path(f"/proc/{worker}").exists()
            and not Path(f"/proc/{grandchild}").exists()
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    from hmlib.cli import hstream_backend
    from hmlib.ui import hstream_launch

    game = tmp_path / "game"
    game.mkdir()
    plan = LaunchPlan(
        game=game,
        config={"hm_shared_ui": {"telemetry": False}, "stitching": {}},
        arguments=["--game-id=game"],
        explicit_arguments=["--start-frame=30", "--max-frames=120"],
        cwd=tmp_path,
        sources={"left": [game / "left.mp4"], "right": [game / "right.mp4"]},
        sinks={"FAKE"},
        start_seconds=0,
        start_frame=30,
        limit_frames=120,
        graph={"plugins": {}},
    )
    monkeypatch.setattr(hstream_launch, "make_plan", lambda *args, **kw: copy.deepcopy(plan))
    monkeypatch.setattr(hstream_backend, "_media_info", lambda paths: (20, 30, False, [10], [20]))
    with open(os.devnull) as input_file:
        monkeypatch.setattr(sys, "stdin", input_file)
        instance = Supervisor(parser().parse_args(["-g=game", "--enable-sinks=FAKE"]), [], tmp_path)
        try:
            yield instance
        finally:
            instance._terminate_child()
            instance.protocol.stopped()
            instance.selector.close()
            for kind, handler in instance._old_handlers.items():
                signal.signal(kind, handler)


def should_keep_resolved_frame_origin_precision_and_effective_config_during_native_seek(
    supervisor, monkeypatch
):
    from hmlib.stitching import native_calibration

    captured = []

    def export(game, output, *, effective_config):
        captured.append(yaml.safe_load(effective_config.read_text()))
        return SimpleNamespace(directory=output, frame_offsets=(0, 0))

    monkeypatch.setattr(native_calibration, "prepare_native_calibration", export)
    assert supervisor.plan.start_seconds == 1
    assert supervisor.duration == 4
    assert supervisor.environment["HSTREAM_UI_BACKEND_SESSION"] == str(supervisor.session)
    supervisor._refresh_calibration()
    command = supervisor.plan.write(supervisor.session, seek_seconds=2)
    assert "--start-time=3.000000000" in command
    assert "--max-time=2.000000000" in command
    assert not any(argument.startswith("--start-frame") for argument in command)
    assert not any(argument.startswith("--max-frames") for argument in command)
    assert captured[0]["stitching"]["dtype"] == "float32"
    assert supervisor.plan.start_seconds == 1
    assert supervisor.plan.limit_seconds == 4


def should_clear_previous_worker_artifacts_and_replay_acknowledged_controls(
    supervisor, monkeypatch
):
    root = supervisor.session
    for name in (
        "spec.json",
        "runtime.json",
        "state.json",
        "action-ack.json",
        "publication.json",
        "preview-program.json",
        "preview-program.jpg",
    ):
        (root / name).write_text("stale")
    mailboxes = [
        root / name
        for name in ("operator-actions", "operator-results", "runtime-requests", "runtime-results")
    ]
    for directory in mailboxes:
        directory.mkdir(exist_ok=True)
        (directory / "stop.json").write_text("stale")
    protocol = supervisor.protocol
    protocol._replay_values = {"Program": {"HStream_Exposure_x100": 80}}
    protocol._active_channel = "stitched"
    protocol._overlay_layers = (True, False, True)
    protocol._session_id = "old-child"
    calls = []
    monkeypatch.setattr(supervisor, "_spawn", lambda command, **kw: calls.append(command))
    supervisor._python(2_000_000_000)
    assert not list(root.glob("preview-*"))
    assert all(not list(directory.glob("*.json")) for directory in mailboxes)
    assert not (root / "runtime.json").exists()
    assert protocol._session_id == ""
    assert json.loads((root / "replay-controls.json").read_text()) == {
        "values": {"Program": {"HStream_Exposure_x100": 80}},
        "preview": "Stitched",
        "overlays": [True, False, True],
    }
    assert "hmlib.cli.hstream_worker" in calls[0]


def should_pause_audio_until_seek_replacement_is_ready(supervisor, monkeypatch):
    pauses = []
    supervisor.audio = SimpleNamespace(set_paused=pauses.append)
    supervisor.phase = "python"
    monkeypatch.setattr(supervisor, "_request_stop", lambda: None)
    supervisor.seek(1_000_000_000, 1)
    assert pauses == [True]
    # Desktop resume during startup must not restart old-position audio.
    supervisor.pause_audio(False)
    assert pauses[-1]
    supervisor.audio_seeking = False
    supervisor.pause_audio(supervisor.audio_paused)
    assert not pauses[-1]


def should_clean_all_resources_and_stale_publication_when_startup_fails(supervisor, monkeypatch):
    cleaned = []
    receipt = supervisor.session / "publication.json"
    receipt.write_text('[{"path":"stale.mp4"}]')

    def fail():
        raise RuntimeError("startup failed")

    monkeypatch.setattr(supervisor, "_prepare", fail)
    monkeypatch.setattr(supervisor, "_terminate_child", lambda: cleaned.append("child"))
    monkeypatch.setattr(supervisor.protocol, "stopped", lambda: cleaned.append("protocol"))
    supervisor.audio = SimpleNamespace(close=lambda: cleaned.append("audio"))
    with pytest.raises(RuntimeError, match="startup failed"):
        supervisor.run()
    assert not receipt.exists()
    assert cleaned == ["child", "audio", "protocol"]


def should_reap_actual_worker_descendants_when_supervised_stop_times_out(supervisor, process_tree):
    supervisor._spawn(
        [sys.executable, str(process_tree), str(supervisor.session), "stubborn"], phase="python"
    )
    wait_for(lambda: (supervisor.session / "ready").exists())
    supervisor.stop_deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="did not complete its stop request"):
        supervisor._pump()
    assert supervisor.child.poll() is not None
    assert supervisor.child.stdin.closed and supervisor.child.stdout.closed
    assert_reaped(supervisor.session)


def should_finish_other_cleanup_even_when_worker_cleanup_fails(supervisor, monkeypatch):
    cleaned = []

    def startup():
        raise RuntimeError("startup failed")

    def child_cleanup():
        cleaned.append("child")
        raise RuntimeError("cleanup failed")

    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "_prepare", startup)
        patch.setattr(supervisor, "_terminate_child", child_cleanup)
        patch.setattr(supervisor.protocol, "stopped", lambda: cleaned.append("protocol"))
        supervisor.audio = SimpleNamespace(close=lambda: cleaned.append("audio"))
        with pytest.raises(RuntimeError, match="cleanup failed"):
            supervisor.run()
    assert cleaned == ["child", "audio", "protocol"]
