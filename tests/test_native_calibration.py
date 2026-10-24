import fcntl
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from hmlib.stitching import native_calibration as native


@pytest.fixture
def snapshot(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    videos = {}
    for role in ("left", "right"):
        source = game / f"{role}.mp4"
        source.write_bytes(b"source recording")
        videos[role] = [str(source)]
    directory = tmp_path / "snapshot"
    directory.mkdir()
    config = {"game": {"videos": videos}}
    manifest = {
        "version": 1,
        "game_directory": str(game),
        "generation": "native-hugin-generation",
        "output_generation": "native-output-generation",
        "invalidation_id": "owner",
        "frame_offsets": {"left": 0, "right": 0.885511},
        "canvas": {"width": 4, "height": 3},
        "post_stitch_rotate_degrees": 0,
        "configuration": config,
        "sources": {role: [native._identity(Path(videos[role][0]))] for role in videos},
        "artifacts": {},
    }
    for name in native._REQUIRED_ARTIFACTS:
        data = yaml.safe_dump(config).encode() if name == "config.yaml" else b"artifact"
        (directory / name).write_bytes(data)
        manifest["artifacts"][name] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (directory / "native-calibration.yaml").write_text(yaml.safe_dump(manifest))
    return directory, manifest


def should_preserve_native_fractional_sync_and_source_order(snapshot):
    directory, _ = snapshot
    result = native.load_native_calibration(directory)
    assert result.frame_offsets == (0.0, 0.885511)
    assert result.canvas_size == (4, 3)
    assert result.directory == directory
    swapped = {
        "left": result.config["game"]["videos"]["right"],
        "right": result.config["game"]["videos"]["left"],
    }
    with pytest.raises(ValueError, match="different or changed left"):
        result.verify_sources(result.game_directory, swapped)


def should_use_selected_game_for_explicit_native_inputs_in_distinct_directories(snapshot, tmp_path):
    from dataclasses import replace

    from hmlib.cli.hmtrack import _stitching_input_directory

    directory, _ = snapshot
    native_snapshot = native.load_native_calibration(directory)
    videos = {}
    for role in ("left", "right"):
        folder = tmp_path / (role + "-recordings")
        folder.mkdir()
        path = folder / "source.mp4"
        path.write_bytes(role.encode())
        videos[role] = [str(path)]
    native_snapshot = replace(
        native_snapshot,
        sources={role: [native._identity(Path(paths[0]))] for role, paths in videos.items()},
    )
    inputs = [videos[role][0] for role in ("left", "right")]
    game = _stitching_input_directory(inputs, native_snapshot.game_directory, native_snapshot)
    assert Path(game) == native_snapshot.game_directory
    native_snapshot.verify_sources(Path(game), videos)
    wrong_game = _stitching_input_directory(inputs, tmp_path, native_snapshot)
    with pytest.raises(ValueError, match="different game"):
        native_snapshot.verify_sources(Path(wrong_game), videos)
    Path(inputs[0]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed left"):
        native_snapshot.verify_sources(Path(game), videos)


def should_reject_calibration_for_a_different_game(snapshot, tmp_path):
    directory, _ = snapshot
    result = native.load_native_calibration(directory)
    with pytest.raises(ValueError, match="different game"):
        result.verify_sources(tmp_path, result.config["game"]["videos"])


def should_reject_replaced_source_recording(snapshot):
    directory, manifest = snapshot
    Path(manifest["sources"]["right"][0]["path"]).write_bytes(b"replacement recording")
    with pytest.raises(ValueError, match="different or changed right"):
        native.load_native_calibration(directory)


def should_reject_corrupted_mapping_even_at_the_same_size(snapshot):
    directory, _ = snapshot
    (directory / "mapping_0000_x.tif").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum changed"):
        native.load_native_calibration(directory)


def should_validate_snapshots_without_python_311_file_digest(snapshot, monkeypatch):
    directory, _ = snapshot
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert native.load_native_calibration(directory).directory == directory


def should_reject_artifact_symlinks(snapshot, tmp_path):
    directory, _ = snapshot
    path = directory / "mapping_0000_x.tif"
    other = tmp_path / "external"
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match="not a regular file"):
        native.load_native_calibration(directory)


@pytest.mark.parametrize("change", ["version", "offset", "missing", "escape", "config"])
def should_reject_invalid_snapshot_contract(snapshot, change):
    directory, manifest = snapshot
    if change == "version":
        manifest["version"] = 2
    elif change == "offset":
        manifest["frame_offsets"]["left"] = float("nan")
    elif change == "missing":
        del manifest["artifacts"]["rink_mask_0.png"]
    elif change == "escape":
        manifest["artifacts"]["../outside"] = {"size": 1, "sha256": "invalid"}
    else:
        manifest["configuration"]["game"]["changed"] = True
    (directory / "native-calibration.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError):
        native.load_native_calibration(directory)


def should_fail_when_explicit_native_exporter_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_HSTREAM_CALIBRATION_EXPORT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="not executable"):
        native.prepare_native_calibration(tmp_path, tmp_path)


def should_find_packaged_exporter_beside_the_native_wrapper(tmp_path, monkeypatch):
    prefix = tmp_path / "installation"
    (prefix / "bin").mkdir(parents=True)
    runner = prefix / "run.sh"
    runner.write_text("#!/bin/sh\n")
    exporter = prefix / "bin" / "hstream-calibration-export"
    exporter.write_text("#!/bin/sh\n")
    exporter.chmod(0o755)
    monkeypatch.delenv("HM_HSTREAM_CALIBRATION_EXPORT", raising=False)
    monkeypatch.setenv("HSTREAM_NATIVE_RUNNER", str(runner))
    assert native._exporter_path(None) == exporter


def should_reuse_native_mask_pixels_without_modifying_game(snapshot):
    import cv2
    import numpy as np

    directory, manifest = snapshot
    mask = np.array([[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0]], dtype=np.uint8)
    cv2.imwrite(str(directory / "rink_mask_0.png"), mask)
    data = (directory / "rink_mask_0.png").read_bytes()
    manifest["artifacts"]["rink_mask_0.png"] = {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (directory / "native-calibration.yaml").write_text(yaml.safe_dump(manifest))
    result = native.load_native_calibration(directory)
    profile = result.rink_profile((3, 4))
    assert np.array_equal(profile["combined_mask"].numpy(), mask != 0)
    assert profile["combined_bbox"] == [1, 1, 2, 2]
    assert sorted(p.name for p in result.game_directory.iterdir()) == ["left.mp4", "right.mp4"]
    with pytest.raises(ValueError, match="dimensions"):
        result.rink_profile((4, 3))


@pytest.fixture
def blocked_exporter(tmp_path, monkeypatch):
    game, session = tmp_path / "game", tmp_path / "session"
    game.mkdir()
    session.mkdir()
    (game / "config.yaml").write_text("rink: {scoreboard: {perspective_polygon: null}}")
    helper = tmp_path / "exporter.py"
    helper.write_text(f"#!{sys.executable}\n" """
import fcntl, json, os, pathlib, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[1] == '--descendant':
    while True: time.sleep(.02)
game = pathlib.Path(sys.argv[sys.argv.index('--game-dir') + 1])
with (game / 'artifact.lock').open('w') as artifact:
    fcntl.flock(artifact, fcntl.LOCK_EX)
    child = subprocess.Popen([sys.executable, __file__, '--descendant'])
    (game / 'helper.json').write_text(json.dumps({
        'guard': os.getppid(), 'helper': os.getpid(), 'descendant': child.pid}))
    with (game / 'config.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    while True: time.sleep(.02)
""")
    helper.chmod(0o755)
    monkeypatch.setenv("HM_HSTREAM_CALIBRATION_EXPORT", str(helper))
    with (game / "config.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield game, session, helper


def _wait_for_helper(game):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if (game / "helper.json").exists():
            return json.loads((game / "helper.json").read_text())
        time.sleep(0.02)
    raise AssertionError("Owned helper did not reach its lock wait")


@pytest.mark.parametrize("stage", ["canvas", "proposal"])
def should_cancel_and_reap_selector_preparation_before_close_returns(
    blocked_exporter, monkeypatch, stage
):
    from hmlib.ui import hstream_scoreboard, process_guard

    game, session, helper = blocked_exporter
    if stage == "proposal":
        image = session / "canvas.png"
        image.write_bytes(b"pinned")
        monkeypatch.setattr(
            native,
            "read_scoreboard_canvas",
            lambda *args, **kwargs: {"image_path": str(image), "width": 64, "height": 40},
        )
        monkeypatch.setattr(
            hstream_scoreboard,
            "run_cancellable",
            lambda command, **kwargs: process_guard.run_cancellable(
                [str(helper), "--game-dir", str(game)], **kwargs
            ),
        )
    selector = hstream_scoreboard.ScoreboardSession(game, session, "blocked", lambda line: None)
    selector.start()
    try:
        pids = _wait_for_helper(game)
        started = time.monotonic()
        selector.close()
        assert time.monotonic() - started < 5
        assert not selector._thread.is_alive()
        assert all(not Path(f"/proc/{pid}").exists() for pid in pids.values()), pids
        assert not list(session.glob("*.png"))
        # Closing must release the helper's artifact lock even while another
        # process continues holding the config lock it was waiting to acquire.
        with (game / "artifact.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        selector.close()


def should_reap_blocked_selector_helper_when_its_supervisor_dies(blocked_exporter):
    game, session, _helper = blocked_exporter
    script = """
import pathlib, sys, time
from hmlib.ui.hstream_scoreboard import ScoreboardSession
selector = ScoreboardSession(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), 'death', print)
selector.start()
while True: time.sleep(.02)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(game), str(session)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        pids = _wait_for_helper(game)
        process.kill()
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        children = [pids["helper"], pids["descendant"]]
        while any(Path(f"/proc/{pid}").exists() for pid in children):
            assert time.monotonic() < deadline, pids
            time.sleep(0.02)
        with (game / "artifact.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


def should_capture_guarded_helper_results_and_cancel_before_start(monkeypatch):
    from hmlib.ui import process_guard

    cancelled = threading.Event()
    result = process_guard.run_cancellable(
        [
            sys.executable,
            "-c",
            "import sys; print('metadata'); sys.stderr.write('details'); exit(3)",
        ],
        cancelled=cancelled,
    )
    assert (result.returncode, result.stdout, result.stderr) == (3, "metadata\n", "details")
    cancelled.set()
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("Cancelled helper started")
    )
    with pytest.raises(process_guard.ProcessCancelledError, match="before startup"):
        process_guard.run_cancellable(["unused"], cancelled=cancelled)


def should_fail_selector_close_when_guard_cleanup_has_not_completed(tmp_path, monkeypatch):
    from hmlib.ui.hstream_scoreboard import ScoreboardSession
    from hmlib.ui.process_guard import ProcessCleanupError

    def incomplete(*args, **kwargs):
        raise ProcessCleanupError("Helper still owns its lock")

    monkeypatch.setattr(native, "read_scoreboard_canvas", incomplete)
    messages = []
    selector = ScoreboardSession(tmp_path, tmp_path, "failed-cleanup", messages.append)
    selector.start()
    with pytest.raises(ProcessCleanupError, match="still owns its lock"):
        selector.close()
    assert not selector._thread.is_alive()
    assert messages == [
        "runtime command failed: scoreboard subprocess ownership: Helper still owns its lock"
    ]
