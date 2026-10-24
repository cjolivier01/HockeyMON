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


def should_bind_native_parent_identity_to_the_immediate_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("HSTREAM_UI_PARENT_PID", "99999999")
    result = subprocess.run(
        guarded_command(
            [
                sys.executable,
                "-c",
                "import os; assert int(os.environ['HSTREAM_UI_PARENT_PID']) == os.getppid()",
            ]
        ),
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    from hmlib.cli import hstream_backend
    from hmlib.ui import hstream_launch

    monkeypatch.setenv("HM_OUTPUT_WORK_DIR", str(tmp_path / "work"))

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

    def export(game, output, *, effective_config, max_output_width):
        captured.append(yaml.safe_load(effective_config.read_text()))
        assert max_output_width == 0
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


def should_keep_default_worker_files_in_durable_output_storage(supervisor, monkeypatch):
    work = supervisor.working_directory
    assert work.is_dir()
    assert work.parent == supervisor.session / "work" / "game" / "runs"
    assert supervisor.worker_cwd == work
    assert supervisor.environment["HM_SHARED_WORK_DIR"] == str(work)
    (work / "incomplete.mkv").write_bytes(b"recoverable working recording")

    def fail():
        raise RuntimeError("native startup failed")

    monkeypatch.setattr(supervisor, "_prepare", fail)
    with pytest.raises(RuntimeError, match="native startup failed"):
        supervisor.run()
    assert (work / "incomplete.mkv").read_bytes() == b"recoverable working recording"


def should_preserve_initial_argument_cwd_while_using_durable_working_outputs(
    supervisor, monkeypatch
):
    monkeypatch.setenv("HM_UI_INITIAL_LAUNCH", str(supervisor.session / "initial.json"))
    instance = Supervisor(supervisor.args, [], supervisor.session)
    try:
        assert instance.worker_cwd == instance.plan.cwd
        assert instance.environment["HM_SHARED_WORK_DIR"] == str(instance.working_directory)
        assert instance.working_directory != supervisor.working_directory
    finally:
        instance.selector.close()
        for kind, handler in instance._old_handlers.items():
            signal.signal(kind, handler)


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


@pytest.mark.parametrize("command", ["q", "quit"])
def should_stop_native_preparation_without_starting_a_python_worker(
    supervisor, monkeypatch, capfd, command
):
    marker = supervisor.session / "native-stopped"
    native = supervisor.session / "native-runner"
    native.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\nfor line in sys.stdin:\n    if line.strip() == 'q':\n        Path({str(marker)!r}).write_text('stopped')\n        break\n"
    )
    native.chmod(0o755)
    supervisor.native = str(native)
    supervisor.args.config = ["native.yaml"]
    spawn = supervisor._spawn

    def start_and_stop(*args, **kwargs):
        spawn(*args, **kwargs)
        supervisor._line(command)

    monkeypatch.setattr(supervisor, "_spawn", start_and_stop)
    monkeypatch.setattr(
        supervisor,
        "_python",
        lambda: pytest.fail("Python must not start after stopping calibration"),
    )
    assert supervisor.run() == 0
    assert marker.read_text() == "stopped"
    assert supervisor.snapshot is None
    output = capfd.readouterr().out
    assert output.count("HM_BACKEND_LAUNCH ") == 1
    receipt = json.loads(
        next(
            line.split(" ", 1)[1]
            for line in output.splitlines()
            if line.startswith("HM_BACKEND_PUBLICATION ")
        )
    )
    assert receipt["status"] == "cancelled" and receipt["stage"] == "preparing"
    assert receipt["artifacts"] == []


@pytest.mark.parametrize("game_id", ["rink-4k", "stitched-rink"])
def should_report_producer_output_kinds_without_interpreting_filenames(
    supervisor, monkeypatch, capsys, game_id
):
    supervisor.plan.game = supervisor.session / game_id
    supervisor.plan.game.mkdir()
    (supervisor.plan.game / "stitched-4k-custom-1.mkv").write_bytes(b"published media")
    records = [
        {
            "name": "stitched-4k-custom.mkv",
            "path": str(supervisor.plan.game / "stitched-4k-custom-1.mkv"),
            "version": 1,
            "kind": "program",
        }
    ]
    monkeypatch.setattr(supervisor, "_prepare", lambda: None)
    monkeypatch.setattr(supervisor, "_prepare_audio", lambda: None)
    monkeypatch.setattr(
        supervisor,
        "_python",
        lambda: (supervisor.session / "publication.json").write_text(json.dumps(records)),
    )
    monkeypatch.setattr(supervisor, "_pump", lambda: 0)
    assert supervisor.run() == 0
    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("HM_BACKEND_PUBLICATION ")
    )
    assert json.loads(line.split(" ", 1)[1])["artifacts"] == records


@pytest.mark.parametrize("cap", [None, 0, 4096, 16000])
@pytest.mark.parametrize("refresh", [False, True])
def should_export_the_exact_native_canvas_cap(supervisor, monkeypatch, cap, refresh):
    from hmlib.stitching import native_calibration
    from hmlib.ui import hstream_launch

    supervisor.plan.config["stitching"]["max_output_width"] = cap
    monkeypatch.setattr(
        hstream_launch, "make_plan", lambda *a, **kw: copy.deepcopy(supervisor.plan)
    )
    captured = []

    def export(game, output, *, effective_config, max_output_width):
        captured.append(max_output_width)
        assert yaml.safe_load(effective_config.read_text())["stitching"]["max_output_width"] == cap
        return SimpleNamespace(directory=output, frame_offsets=(300, 0))

    monkeypatch.setattr(native_calibration, "prepare_native_calibration", export)
    supervisor.native = "/unused/native"
    supervisor.args.config = ["native.yaml"]
    monkeypatch.setattr(supervisor, "_spawn", lambda *a, **kw: None)
    monkeypatch.setattr(supervisor, "_pump", lambda: 0)
    (supervisor._refresh_calibration if refresh else supervisor._prepare)()
    assert captured == [cap or 0]
    assert supervisor.protocol.frame_offset == 300


@pytest.mark.parametrize("start", [1.0, 120.0])
@pytest.mark.parametrize("reference", [None, "00:00:02"])
def should_calibrate_at_the_resolved_python_playback_origin(
    supervisor, monkeypatch, start, reference
):
    from hmlib.stitching import native_calibration

    supervisor.plan.start_seconds = start
    supervisor.arguments = ["--start-time=00:00:00"]
    if reference:
        supervisor.plan.initial_settings = {"stitching": {"stitch_frame_time": reference}}
    supervisor.native = "/unused/native"
    supervisor.args.config = ["native.yaml"]
    captured = []
    monkeypatch.setattr(supervisor, "_spawn", lambda command, **kw: captured.append(command))
    monkeypatch.setattr(supervisor, "_pump", lambda: 0)
    monkeypatch.setattr(
        native_calibration,
        "prepare_native_calibration",
        lambda *a, **kw: SimpleNamespace(directory=supervisor.session, frame_offsets=(0, 0)),
    )
    supervisor._prepare()
    command = captured[0]
    assert [token for token in command if token.startswith("--start-time=")] == [
        f"--start-time={start:.9f}"
    ]
    if reference:
        assert "--stitch-frame-time=" + reference in command


@pytest.mark.parametrize("cap", [-1, True, 1.5, "4096", []])
def should_reject_invalid_native_canvas_caps(cap):
    from hmlib.cli.hstream_backend import _native_max_output_width

    with pytest.raises(ValueError, match="nonnegative integer"):
        _native_max_output_width({"stitching": {"max_output_width": cap}})


def completed_worker(supervisor, monkeypatch, records):
    monkeypatch.setattr(supervisor, "_prepare", lambda: None)
    monkeypatch.setattr(supervisor, "_prepare_audio", lambda: None)
    monkeypatch.setattr(
        supervisor,
        "_python",
        lambda: (supervisor.session / "publication.json").write_text(json.dumps(records)),
    )
    monkeypatch.setattr(supervisor, "_pump", lambda: 0)


def should_remove_only_owned_working_storage_after_all_publications_exist(
    supervisor, monkeypatch, capsys
):
    supervisor.plan.sinks = {
        "ENCODE_FILE",
        "ENCODE_STITCHED_FILE",
        "ENCODE_PROGRAM_4K_FILE",
        "RTMP",
    }
    supervisor.environment["HM_SHARED_TELEMETRY"] = "1"
    outside = supervisor.plan.cwd / "original-user-file"
    outside.write_bytes(b"retain original working directory")
    (supervisor.working_directory / "outside-link").symlink_to(
        supervisor.plan.game, target_is_directory=True
    )
    (supervisor.working_directory / "working.mkv").write_bytes(b"copied recording")
    records = []
    for kind, suffix in [
        ("program", "mkv"),
        ("stitched", "mkv"),
        ("program-4k", "mkv"),
        ("telemetry", "db"),
        (None, "png"),
    ]:
        path = supervisor.plan.game / f"artifact-{kind}.{suffix}"
        path.write_bytes(b"published companion")
        records.append({"path": str(path), **({"kind": kind} if kind else {})})
    completed_worker(supervisor, monkeypatch, records)
    assert supervisor.run() == 0
    assert not supervisor.working_directory.exists()
    assert outside.is_file()
    assert all(Path(record["path"]).is_file() for record in records)
    assert '"status": "complete"' in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "empty", "working", "auxiliary"])
def should_preserve_working_recordings_when_publication_validation_fails(
    supervisor, monkeypatch, capsys, problem
):
    supervisor.plan.sinks = {"ENCODE_FILE"}
    work = supervisor.working_directory / "recoverable.mkv"
    work.write_bytes(b"working recording")
    output = supervisor.plan.game / "recording.mkv"
    output.write_bytes(b"published recording")
    records = [{"path": str(output), "kind": "program"}]
    if problem == "missing":
        records.clear()
    elif problem == "empty":
        output.write_bytes(b"")
    elif problem == "working":
        records[0]["path"] = str(work)
    else:
        records.append({"path": str(supervisor.plan.game / "missing-mask.png")})
    completed_worker(supervisor, monkeypatch, records)
    with pytest.raises((ValueError, FileNotFoundError)):
        supervisor.run()
    assert work.is_file()
    assert '"status": "complete"' not in capsys.readouterr().out


def should_fail_visibly_without_success_receipt_when_workdir_cleanup_fails(
    supervisor, monkeypatch, capsys
):
    from hmlib.cli import hstream_backend

    completed_worker(supervisor, monkeypatch, [])

    def fail(path):
        raise PermissionError("cannot remove completed working directory")

    monkeypatch.setattr(hstream_backend.shutil, "rmtree", fail)
    with pytest.raises(PermissionError, match="cannot remove"):
        supervisor.run()
    assert supervisor.working_directory.is_dir()
    assert '"status": "complete"' not in capsys.readouterr().out


def should_allow_network_only_completion_without_file_artifacts(supervisor, monkeypatch):
    supervisor.plan.sinks = {"RTMP", "RTSP"}
    completed_worker(supervisor, monkeypatch, [])
    assert supervisor.run() == 0
    assert not supervisor.working_directory.exists()


def should_feed_the_synchronized_runtime_clock_to_the_audio_monitor(supervisor):
    updates = []
    supervisor.protocol.frame_offset = 300
    supervisor.audio = SimpleNamespace(update=lambda value, **kwargs: updates.append(value))
    (supervisor.session / "runtime.json").write_text(
        json.dumps({"frame": 390, "fps": 30, "state": "running"})
    )
    supervisor._spawn([sys.executable, "-c", "import time; time.sleep(.1)"], phase="python")
    assert supervisor._pump() == 0
    assert updates and all(value == 3 for value in updates)


def should_keep_audio_start_seconds_independent_of_selected_camera_rate(monkeypatch):
    from hmlib.cli.hmtrack import _resolve_mux_audio_source
    from hmlib.video import ffmpeg

    monkeypatch.setattr(ffmpeg, "BasicVideoInfo", lambda _: SimpleNamespace(fps=60))
    monkeypatch.setattr(
        "hmlib.ui.output_audio.prepare_chapter_audio",
        lambda paths, work_dir, **kw: (str(paths[0]), None),
    )
    source, temporary, offset = _resolve_mux_audio_source(
        ["left.mp4", "right.mp4"], frame_offsets=(60, 120), start_frame=90, start_seconds=3
    )
    assert source == "left.mp4" and temporary is None
    assert offset == -4


@pytest.mark.parametrize(
    "rates, expected", [((60, 30), 30), ((30, 60), 30), ((59.94, 29.97), 29.97), ((25, 25), 25)]
)
def should_resolve_only_representable_shared_camera_cadences(rates, expected):
    from hmlib.cli.hmtrack import _shared_playback_fps

    assert _shared_playback_fps(*rates) == expected


@pytest.mark.parametrize("rates", [(60, 24), (60, 29.97), (0, 30), (float("nan"), 30)])
def should_refuse_shared_camera_cadences_that_cannot_stay_synchronized(rates):
    from hmlib.cli.hmtrack import _shared_playback_fps

    with pytest.raises(ValueError, match="frame rates"):
        _shared_playback_fps(*rates)


def should_resolve_frame_bounds_in_delivered_time_for_mixed_camera_rates(supervisor, monkeypatch):
    from hmlib.cli import hstream_backend

    monkeypatch.setattr(
        hstream_backend,
        "_media_info",
        lambda paths: (20, 60 if paths[0].name == "left.mp4" else 30, False, [8], [20]),
    )
    instance = Supervisor(supervisor.args, [], supervisor.session)
    try:
        assert instance.fps == 30
        assert instance.plan.start_seconds == 1
        assert instance.duration == 4
    finally:
        instance.selector.close()
        for kind, handler in instance._old_handlers.items():
            signal.signal(kind, handler)


def should_prepare_shared_models_before_single_camera_processing(supervisor, monkeypatch):
    from hmlib.ui import shared_analytics

    supervisor.plan.sources = {"source": [supervisor.plan.game / "one.mp4"]}
    supervisor.plan.config.update(
        hm_ui={"shared_analytics": {"pose": "coco"}},
        pipeline={"player-analytics": {"pose": {"enable": True, "model": "coco"}}},
    )
    application = supervisor.session / "ds_hockey_app_config.yaml"
    application.write_text("tracker:\n  ll-config-file: /native/tracker.yaml\n  gpu-id: 0\n")
    supervisor.args.config = [str(application)]
    supervisor.native = "/matching/hstream-cli"
    supervisor.environment["HM_HSTREAM_MODEL_PREPARE"] = "/bin/true"
    calls = []
    accepted = []

    def spawn(command, *, phase):
        calls.append((command, phase))
        input_config = yaml.safe_load((supervisor.session / "models-input.yaml").read_text())
        assert input_config["pipeline"]["tracker"]["ll-config-file"] == "/native/tracker.yaml"
        assert input_config["pipeline"]["tracker"]["reid-enable"] is False
        assert input_config["pipeline"]["player-analytics"]["pose"]["enable"] is True
        assert input_config["pipeline"]["player-analytics"]["jersey"]["enable"] is False
        output = supervisor.session / "models-resolved.yaml"
        input_config["pipeline"]["player-analytics"]["pose"]["bundle"] = "/prepared/exact-pose"
        output.write_text(yaml.safe_dump(input_config))
        (supervisor.session / "models-receipt.json").write_text(
            json.dumps({"version": 1, "status": "complete", "output": str(output)})
        )

    monkeypatch.setattr(supervisor, "_spawn", spawn)
    monkeypatch.setattr(supervisor, "_pump", lambda: 0)
    monkeypatch.setattr(
        shared_analytics,
        "apply_prepared_shared_analytics",
        lambda config, prepared: accepted.append(prepared),
    )
    supervisor._prepare()
    assert len(calls) == 1 and calls[0][1] == "models"
    assert accepted == [supervisor.prepared_analytics]
    assert accepted[0]["pipeline"]["player-analytics"]["pose"]["bundle"] == "/prepared/exact-pose"


def should_not_start_calibration_after_model_preparation_is_cancelled(supervisor, monkeypatch):
    def prepare():
        supervisor.stop_requested = True

    monkeypatch.setattr(supervisor, "_prepare_models", prepare)
    monkeypatch.setattr(
        supervisor, "_native_runner", lambda: pytest.fail("Calibration must not start after Stop")
    )
    supervisor._prepare()
    assert supervisor.child is None


def should_reapply_accepted_models_when_native_edits_refresh(supervisor, monkeypatch):
    from hmlib.stitching import native_calibration
    from hmlib.ui import shared_analytics

    supervisor.prepared_analytics = {
        "pipeline": {"player-analytics": {"pose": {"bundle": "/accepted"}}}
    }
    accepted = []
    monkeypatch.setattr(
        shared_analytics,
        "apply_prepared_shared_analytics",
        lambda config, prepared: accepted.append(prepared),
    )
    monkeypatch.setattr(
        native_calibration,
        "prepare_native_calibration",
        lambda game, output, **kwargs: SimpleNamespace(directory=output, frame_offsets=(0, 0)),
    )
    supervisor._refresh_calibration()
    assert accepted == [supervisor.prepared_analytics]
