"""Projection review captures isolate sources and only reuse intact results."""

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
import yaml

from scripts import capture_stitching_projection_frames as capture
from stitching_fixtures import write_generation


@pytest.fixture
def matrix():
    return {
        "version": 1,
        "defaults": {"auto_fov": True},
        "projections": [
            {"name": "rectilinear", "variants": [{"label": "first"}, {"label": "second"}]}
        ],
    }


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    (path / "config.yaml").write_text("stitching: {}\n")
    for side in ("left", "right"):
        assert cv2.imwrite(str(path / f"{side}.png"), np.zeros((3, 4, 3), np.uint8))
    return path


@pytest.fixture
def successful_runner():
    calls = []

    def run(plan_path, log_path, timeout):
        calls.append(plan_path)
        game = plan_path.parent / "game"
        game.mkdir()
        write_generation(game)
        assert cv2.imwrite(str(game / "s.png"), np.zeros((3, 4, 3), np.uint8))
        (game / ".stitching_artifacts.json").write_text("{}")
        log_path.write_text("Completed fixture generation\n")
        return 0

    return calls, run


def launch(matrix, source, output, runner, **kwargs):
    return capture.capture(
        matrix,
        capture.expand_cases(matrix, capture.read_yaml(source / "config.yaml")),
        source,
        output,
        mode="images",
        runner=runner,
        **kwargs,
    )


def should_expand_all_58_variants_in_22_projections():
    cases = capture.expand_cases(
        capture.read_yaml(
            (capture.ROOT / "hmlib/config/stitching_projection_frames.yaml").resolve()
        )
    )
    assert len(cases) == 58
    assert len({case["effective_config"]["stitching"]["projection"] for case in cases}) == 22


@pytest.mark.parametrize("value", [False, "", -1, 2.5])
def should_reject_invalid_width_instead_of_disabling_cap(matrix, value):
    matrix["defaults"]["max_output_width"] = value
    with pytest.raises(ValueError):
        capture.expand_cases(matrix)


def should_apply_camera_presets_and_variant_fov_overrides(matrix):
    matrix["defaults"]["camera_config"] = "gopro-hero-11"
    matrix["projections"][0]["variants"][1].update(
        camera_horizontal_fov=120, camera_vertical_fov=80
    )
    source = {"stitching": {"camera_fov": {"horizontal_fov": 60}}}
    original = copy.deepcopy(source)
    cases = capture.expand_cases(matrix, source)
    assert cases[0]["effective_config"]["stitching"]["camera_fov"] == {
        "horizontal_fov": 108,
        "vertical_fov": 90,
    }
    assert cases[1]["effective_config"]["stitching"]["camera_fov"] == {
        "horizontal_fov": 120,
        "vertical_fov": 80,
    }
    assert source == original
    matrix["defaults"]["camera_horizontal_fov"] = 500
    with pytest.raises(ValueError, match="Camera FOV"):
        capture.expand_cases(matrix)


@pytest.mark.parametrize(
    "variant",
    [{"label": "../escape"}, {"parameters": [1]}, {"auto_fov": False, "horizontal_fov": 190}],
)
def should_reject_unsafe_or_invalid_cases(matrix, variant):
    matrix["projections"][0]["variants"][0].update(variant)
    with pytest.raises(ValueError):
        capture.expand_cases(matrix)


def should_reject_duplicate_case_names(matrix):
    matrix["projections"][0]["variants"][1]["label"] = "first"
    with pytest.raises(ValueError, match="Duplicate"):
        capture.expand_cases(matrix)


def should_reuse_only_unchanged_complete_captures(matrix, source, tmp_path, successful_runner):
    calls, runner = successful_runner
    output = tmp_path / "captures"
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert launch(matrix, source, output, runner) == 0
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 2
    first = output / "001__rectilinear--first"
    # A valid but changed preview must also invalidate a saved success.
    assert cv2.imwrite(str(first / "game/s.png"), np.ones((3, 4, 3), np.uint8))
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 3
    (first / "game/mapping_0000_x.tif").unlink()
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 4
    (first / "effective-config.yaml").write_text("changed: true\n")
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 5
    (first / "effective-config.yaml").write_text("stitching: [broken\n")
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 6
    (first / "plan.json").write_text("{}")
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 7
    assert launch(matrix, source, output, runner, force=True, start_at=2, limit=1) == 0
    assert len(calls) == 8
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    # Content changes survive unchanged file size and mtime.
    image = source / "left.png"
    stamp = image.stat()
    data = bytearray(image.read_bytes())
    data[-1] ^= 1
    image.write_bytes(data)
    os.utime(image, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 10


def should_record_failures_continue_and_retry_failed_force(
    matrix, source, tmp_path, successful_runner
):
    calls, runner = successful_runner
    output = tmp_path / "captures"
    assert launch(matrix, source, output, runner) == 0

    def fail(plan, log, timeout):
        if "first" in str(plan):
            return 7
        return runner(plan, log, timeout)

    assert launch(matrix, source, output, fail, force=True) == 1
    rows = json.loads((output / "manifest.json").read_text())
    assert rows["001__rectilinear--first"]["return_code"] == 7
    assert rows["001__rectilinear--first"]["outcome"] == "fail"
    assert rows["002__rectilinear--second"]["outcome"] == "pass"
    assert launch(matrix, source, output, runner) == 0
    assert len(calls) == 4


@pytest.mark.parametrize(
    "failure", [subprocess.TimeoutExpired("worker", 1), OSError("cannot spawn")]
)
def should_record_timeout_or_setup_error_and_continue(
    matrix, source, tmp_path, successful_runner, failure
):
    _, runner = successful_runner

    def run(plan, log, timeout):
        if "first" in str(plan):
            raise failure
        return runner(plan, log, timeout)

    output = tmp_path / "captures"
    assert launch(matrix, source, output, run) == 1
    rows = json.loads((output / "manifest.json").read_text())
    assert rows["001__rectilinear--first"]["error"] == str(failure)
    assert rows["002__rectilinear--second"]["outcome"] == "pass"


def should_refuse_unowned_or_overlapping_outputs(source, tmp_path):
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    sentinel = unowned / "keep"
    sentinel.write_text("keep")
    linked = tmp_path / "link"
    linked.symlink_to(unowned, target_is_directory=True)
    for output in (source, source / "nested", source.parent, unowned, linked):
        with pytest.raises(ValueError):
            capture.prepare_output(output, source)
    assert sentinel.read_text() == "keep"


def should_refuse_symlink_case_marker_without_deleting_files(
    matrix, source, tmp_path, successful_runner
):
    _, runner = successful_runner
    output = tmp_path / "captures"
    assert launch(matrix, source, output, runner) == 0
    first = output / "001__rectilinear--first"
    marker = first / capture.MARKER
    marker.unlink()
    marker.symlink_to(output / capture.MARKER)
    assert launch(matrix, source, output, runner, force=True) == 1
    assert (first / "game/s.png").exists()


def should_resolve_cli_paths_from_cwd_and_yaml_paths_from_config_directory(
    matrix, source, tmp_path, monkeypatch
):
    config_dir = tmp_path / "matrices"
    config_dir.mkdir()
    matrix["source_game_dir"] = "../source"
    config = config_dir / "matrix.yaml"
    config.write_text(yaml.safe_dump(matrix))
    monkeypatch.chdir(tmp_path)
    assert (
        capture.main(
            [
                "--config",
                str(config),
                "--source-game-dir",
                "source",
                "--dry-run",
                "--start-at",
                "2",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert capture.main(["--config", str(config), "--dry-run"]) == 0
    assert not (tmp_path / "captures").exists()


def should_run_direct_script_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(capture.ROOT / "scripts/capture_stitching_projection_frames.py"),
            "--dry-run",
            "--limit",
            "1",
        ],
        env=environment,
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run: 1 of 58 cases" in result.stdout


@pytest.mark.parametrize("leader_exit", [False, True])
def should_kill_children_on_timeout_or_worker_failure(tmp_path, monkeypatch, leader_exit):
    pidfile = tmp_path / "child.pid"
    ready = tmp_path / "ready"
    child_code = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(os.environ['CAPTURE_TEST_READY']).touch(); time.sleep(60)"
    leader_code = "import os,subprocess,sys,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c',os.environ['CAPTURE_TEST_CHILD']]); Path(os.environ['CAPTURE_TEST_PID']).write_text(str(child.pid)); time.sleep(60)"
    if leader_exit:
        leader_code = leader_code.replace("time.sleep(60)", "time.sleep(0.05); os._exit(7)")
    monkeypatch.setenv("CAPTURE_TEST_READY", str(ready))
    monkeypatch.setenv("CAPTURE_TEST_PID", str(pidfile))
    monkeypatch.setenv("CAPTURE_TEST_CHILD", child_code)
    real_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        process = real_popen([sys.executable, "-c", leader_code], **kwargs)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        return process

    monkeypatch.setattr(capture.subprocess, "Popen", spawn)
    child_pid = None
    try:
        if leader_exit:
            assert capture.run_subprocess(tmp_path / "plan.json", tmp_path / "log", 5) == 7
        else:
            with pytest.raises(subprocess.TimeoutExpired):
                capture.run_subprocess(tmp_path / "plan.json", tmp_path / "log", 0.1)
        child_pid = int(pidfile.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Timed-out capture left its SIGTERM-ignoring child alive")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                assert not Path(f"/proc/{child_pid}").exists()


def should_fsync_manifest_and_parent_directory(tmp_path, monkeypatch):
    kinds = []
    real_fsync = os.fsync

    def sync(descriptor):
        kinds.append(Path(f"/proc/self/fd/{descriptor}").resolve().is_dir())
        real_fsync(descriptor)

    monkeypatch.setattr(capture.os, "fsync", sync)
    capture.atomic_json(tmp_path / "manifest.json", {"ok": True})
    assert kinds == [False, True]
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"ok": True}


@pytest.mark.parametrize("configured", [False, True])
def should_resolve_video_offsets_without_writing_source_game(
    source, tmp_path, monkeypatch, configured
):
    from types import SimpleNamespace
    import hmlib.stitching.configure_stitching as builder
    import hmlib.stitching.synchronize as synchronization
    import hmlib.video.ffmpeg as ffmpeg

    config = (
        {"stitching": {"frame_offsets": {"left": 3, "right": 7}}}
        if configured
        else {"stitching": {}}
    )
    (source / "left_calibration.json").write_text('{"source-profile": true}')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    case = tmp_path / "case"
    case.mkdir()
    plan = case / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "effective_config": {
                    "stitching": {**config["stitching"], "stitch_frame_time": "00:00:01"}
                },
                "inputs": [str(source / "left.mp4"), str(source / "right.mp4")],
                "source_game_dir": str(source),
                "source_mode": "videos",
            }
        )
    )
    audio_calls, captures = [], []

    def audio(*paths):
        audio_calls.append(paths)
        return 3, 7

    def calibrate(directory, *args, **kwargs):
        captures.append((directory, args, kwargs))
        game = Path(directory)
        assert game == case / "game"
        assert (game / "left_calibration.json").read_bytes() == before["left_calibration.json"]
        write_generation(game)
        assert cv2.imwrite(str(game / "s.png"), np.zeros((3, 4, 3), np.uint8))
        (game / ".stitching_artifacts.json").write_text("{}")

    monkeypatch.setattr(synchronization, "synchronize_by_audio", audio)
    monkeypatch.setattr(ffmpeg, "BasicVideoInfo", lambda path: SimpleNamespace(fps=30))
    monkeypatch.setattr(builder, "configure_video_stitching", calibrate)
    assert capture.run_worker(plan) == 0
    assert len(audio_calls) == int(not configured)
    args = captures[0][2]
    assert args["ignore_private_config"] is True
    assert args["left_frame_offset"] == 3
    assert args["right_frame_offset"] == 7
    assert args["base_frame_offset"] == 30
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
