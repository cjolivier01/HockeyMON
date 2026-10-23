"""Desktop IPC and process ownership, using real isolated child processes."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from hmlib.ui.workspace import Workspace
from hmlib.ui.workspace.files import read_config, write_config
from hmlib.ui.workspace.media import trajectory


@pytest.fixture
def workspace(tmp_path):
    with Workspace(tmp_path / "games", tmp_path / "session") as instance:
        instance.create_game("sample")
        yield instance


def wait_job(workspace, job_id, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        workspace.poll()
        job = workspace.jobs.get(job_id)
        if job["status"] not in {"running", "paused", "stopping"}:
            return job
        time.sleep(0.025)
    raise AssertionError(f"Job did not finish: {workspace.jobs.get(job_id)}")


def should_return_durable_ipc_results_without_replaying_mutations(workspace):
    command = workspace.commands / "001.json"
    command.write_text(
        json.dumps({"id": "001", "operation": "games.create", "args": {"game_id": "second"}})
    )
    workspace.poll()
    first = (workspace.results / command.name).read_bytes()
    assert json.loads(first)["ok"]
    workspace.poll()
    assert (workspace.results / command.name).read_bytes() == first
    assert workspace.current_game == workspace.root_dir / "second"
    (workspace.commands / "002.json").write_text(
        json.dumps({"id": "002", "operation": "games.create", "args": {"game_id": "../escape"}})
    )
    workspace.poll()
    failure = json.loads((workspace.results / "002.json").read_text())
    assert not failure["ok"]
    assert "Invalid game ID" in failure["error"]


def should_import_video_aliases_without_changing_sources(workspace, tmp_path):
    source = tmp_path / "unusual name's $(touch nope).MP4"
    source.write_bytes(b"video content")
    result = workspace.dispatch("games.import_videos", {"role": "left", "paths": [str(source)]})
    assert result["paths"] == [f"cam1/{source.name}"]
    imported = workspace.game() / result["paths"][0]
    assert imported.is_symlink()
    assert imported.read_bytes() == b"video content"
    workspace.dispatch("games.import_videos", {"role": "left", "paths": [str(source)]})
    assert len(read_config(workspace.game())["game"]["videos"]["left"]) == 1
    assert not (workspace.game() / "nope").exists()


def should_launch_isolated_children_with_live_ipc_and_reject_conflicting_work(workspace):
    code = "import os,time; print(os.environ['HM_UI_SESSION_DIR'], flush=True); time.sleep(60)"
    job = workspace.start_command([sys.executable, "-u", "-c", code], "sample", "track")
    with pytest.raises(RuntimeError, match="Stop the active pipeline"):
        workspace.start_command([sys.executable, "-c", "pass"], "sample", "stitch")
    with pytest.raises(RuntimeError, match="Stop job"):
        workspace.create_experiment({"kind": "camera"})
    workspace.dispatch("jobs.pause", {"job_id": job["id"]})
    assert workspace.jobs.get(job["id"])["status"] == "paused"
    workspace.dispatch("jobs.resume", {"job_id": job["id"]})
    workspace.dispatch("jobs.stop", {"job_id": job["id"]})
    assert wait_job(workspace, job["id"])["status"] == "cancelled"
    completed = workspace.start_command([sys.executable, "-c", "print('done')"], "sample", "track")
    assert wait_job(workspace, completed["id"])["status"] == "completed"
    assert workspace.dispatch("jobs.log", {"job_id": completed["id"]})["text"].strip() == "done"


def should_stop_descendants_and_not_launch_queued_commands_on_close(tmp_path):
    workspace = Workspace(tmp_path / "games", tmp_path / "session")
    workspace.create_game("sample")
    child_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    workspace.start_command([sys.executable, "-c", code], "sample")
    end = time.monotonic() + 5
    while not child_file.exists() and time.monotonic() < end:
        time.sleep(0.02)
    assert child_file.exists()
    child_pid = int(child_file.read_text())
    (workspace.commands / "pending.json").write_text(
        json.dumps({"id": "pending", "operation": "games.create", "args": {"game_id": "unwanted"}})
    )
    workspace.close()
    assert not (workspace.root_dir / "unwanted").exists()
    state = Path(f"/proc/{child_pid}/stat")
    if state.exists():
        assert state.read_text().split()[2] == "Z"


def should_preserve_exact_script_arguments_and_never_overwrite_export(workspace, tmp_path):
    destination = tmp_path / "render.sh"
    result = workspace.export_job(
        {
            "path": str(destination),
            "kind": "track",
            "arguments": ["--label", "x' $(touch nope)", "-t", "3"],
        }
    )
    subprocess.run(["bash", "-n", str(destination)], check=True)
    assert "--camera-ui=0" in result["command"]
    assert "x' $(touch nope)" in result["command"]
    assert os.access(destination, os.X_OK)
    with pytest.raises(FileExistsError):
        workspace.export_job({"path": str(destination)})


def should_isolate_experiments_and_refuse_promoting_over_new_settings(workspace):
    game = workspace.game()
    (game / "left.mp4").write_bytes(b"immutable source")
    (game / "mapping_0000.tif").write_bytes(b"old calibration")
    candidate = workspace.create_experiment(
        {
            "name": "fast camera",
            "kind": "camera",
            "changes": {"rink.camera.zoom_in_aggressiveness": 90},
        }
    )
    candidate_game = Path(candidate["game_directory"])
    assert (candidate_game / "left.mp4").is_symlink()
    (candidate_game / "mapping_0000.tif").write_bytes(b"new calibration")
    assert (game / "mapping_0000.tif").read_bytes() == b"old calibration"
    assert read_config(candidate_game)["rink"]["camera"]["zoom_in_aggressiveness"] == 90
    directory, record = workspace._experiment(game, candidate["id"])
    record["status"] = "completed"
    (directory / "experiment.json").write_text(json.dumps(record))
    write_config(game, {"rink": {"camera": {"zoom_in_aggressiveness": 30}}})
    with pytest.raises(RuntimeError, match="configuration changed"):
        workspace.promote_experiment({"experiment_id": candidate["id"]})
    assert read_config(game)["rink"]["camera"]["zoom_in_aggressiveness"] == 30


def should_promote_completed_camera_settings(workspace):
    experiment = workspace.create_experiment(
        {"kind": "camera", "changes": {"rink.camera.zoom_in_aggressiveness": 75}}
    )
    directory, record = workspace._experiment(workspace.game(), experiment["id"])
    record["status"] = "completed"
    (directory / "experiment.json").write_text(json.dumps(record))
    workspace.promote_experiment({"experiment_id": experiment["id"]})
    assert (
        yaml.safe_load((workspace.game() / "config.yaml").read_text())["rink"]["camera"][
            "zoom_in_aggressiveness"
        ]
        == 75
    )


def should_pair_archive_and_telemetry_versions_in_catalog(workspace):
    game = workspace.game()
    (game / "sample-tracking_output-with-audio-3.mp4").write_bytes(b"video")
    (game / "sample_telemetry-3.db").write_bytes(b"database")
    (game / "sample_telemetry-2.db").write_bytes(b"older")
    result = workspace.catalog(game)
    assert result["archives"][0]["version"] == 3
    assert result["archives"][0]["recordings"] == [str(game / "sample_telemetry-3.db")]


def should_bound_camera_trajectory_reads_and_preserve_segment_boundaries(tmp_path):
    database = tmp_path / "telemetry.db"
    with sqlite3.connect(database) as connection:
        schema = Path(__file__).parents[1] / "hmlib" / "telemetry" / "schema.sql"
        connection.executescript(schema.read_text())
        connection.execute(
            "INSERT INTO runs VALUES('run','sample','2026-10-22','test','{}','{}',1,'completed',20)"
        )
        connection.execute(
            "INSERT INTO geometries VALUES('run',1,1000,500,'stitched','revision',NULL,NULL,NULL,'{}')"
        )
        for index in range(1, 21):
            connection.execute(
                "INSERT INTO frames VALUES('run',?,1,?,NULL,NULL,?,NULL,0,?,1,0,0)",
                (index, index, index * 100000000, index // 10),
            )
            connection.execute(
                "INSERT INTO cameras VALUES('run',?,'program',?,100,200,100)", (index, index * 10)
            )
    result = trajectory(database, max_points=5)
    points = result["series"]["program"]
    assert len(points) <= 5
    assert points[0]["time_seconds"] == 0
    assert points[-1]["time_seconds"] == pytest.approx(1.9)
    assert points[0]["x"] == pytest.approx(0.11)
    assert points[0]["segment"] != points[-1]["segment"]


def should_retire_even_paused_children_when_the_desktop_disappears(tmp_path):
    game_root = tmp_path / "games"
    session = tmp_path / "session"
    code = (
        "import json,sys,time; from hmlib.ui.workspace import Workspace; "
        f"w=Workspace({str(game_root)!r},{str(session)!r}); w.create_game('sample'); "
        "j=w.start_command([sys.executable,'-c','import time; time.sleep(60)'],'sample'); "
        "w.jobs.pause(j['id'],True); print(json.dumps(j),flush=True); time.sleep(60)"
    )
    owner = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        record = json.loads(owner.stdout.readline())
        owner.kill()
        owner.wait(timeout=5)
        child_status = Path(f"/proc/{record['child_pid']}/stat")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not child_status.exists() or child_status.read_text().split()[2] == "Z":
                break
            time.sleep(0.025)
        else:
            raise AssertionError("Paused pipeline survived loss of its desktop owner")
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def should_concatenate_imported_chapters_without_replacing_originals(workspace, tmp_path):
    sources = []
    for index, color in enumerate(("red", "blue")):
        source = tmp_path / f"chapter's {index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=160x96:r=10:d=0.5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
        )
        sources.append(source)
    workspace.import_videos({"role": "source", "paths": [str(path) for path in sources]})
    job = workspace.dispatch("games.concatenate", {"role": "source"})
    record = wait_job(workspace, job["id"], timeout=30)
    assert record["status"] == "completed", workspace.jobs.log(job["id"])
    joined = Path(record["result"]["path"])
    assert workspace.media.info(joined)["duration_ms"] == pytest.approx(1000, abs=100)
    assert all(source.is_file() for source in sources)
    assert read_config(workspace.game())["game"]["videos"]["source"] == [
        str(joined.relative_to(workspace.game()))
    ]


def should_report_invalid_external_config_without_stopping_ipc(workspace):
    (workspace.game() / "config.yaml").write_text("game: [broken")
    workspace.poll()
    snapshot = json.loads((workspace.session_dir / "workspace.json").read_text())
    assert snapshot["errors"] and "Catalog:" in snapshot["errors"][0]
    assert snapshot["current_game"] == str(workspace.game())


def should_promote_derived_stitch_geometry_and_invalidate_old_rink_map(workspace):
    from stitching_fixtures import write_generation

    game = workspace.game()
    write_generation(game)
    (game / ".stitching_artifacts.json").write_text('{"generation":"baseline"}')
    write_config(
        game,
        {
            "stitching": {"camera_automation": {"fov": False, "lens": False}},
            "rink": {
                "workspace_map": {"geometry": "old"},
                "scoreboard": {"perspective_polygon": [[1, 2]]},
                "ice_contours_mask_file_prefix": "old-mask",
            },
        },
    )
    trial = workspace.create_experiment(
        {"kind": "stitch", "changes": {"stitching.max_control_points": 200}}
    )
    candidate = Path(trial["game_directory"])
    config = read_config(candidate)
    config["stitching"].update(control_points=[[1, 2, 3, 4]], frame_offsets={"left": 0, "right": 3})
    config["rink"] = {"ice_contours_geometry_revision": "new-geometry"}
    write_config(candidate, config)
    for name in ("hm_project.pto", "autooptimiser_out.pto"):
        (candidate / name).write_text(f'p f2 w4 h3 v180\ni w4 h3 n"{candidate}/left.png"\n')
    directory, record = workspace._experiment(game, trial["id"])
    record["status"] = "completed"
    (directory / "experiment.json").write_text(json.dumps(record))
    workspace.promote_experiment({"experiment_id": trial["id"]})
    promoted = read_config(game)
    assert promoted["stitching"]["control_points"] == [[1, 2, 3, 4]]
    assert promoted["stitching"]["frame_offsets"]["right"] == 3
    assert promoted["rink"] == {"scoreboard": {}, "ice_contours_geometry_revision": "new-geometry"}
    assert str(candidate) not in (game / "hm_project.pto").read_text()
    assert str(game / "left.png") in (game / "hm_project.pto").read_text()


def _cooperative_pipeline(workspace, tmp_path):
    finalized = tmp_path / "finalized.txt"
    code = (
        "import json,os,pathlib,sys,time; "
        "live=pathlib.Path(os.environ['HM_UI_SESSION_DIR']); "
        "(live/'runtime.json').write_text(json.dumps({'capabilities':['stop']})); "
        "commands=live/'operator-actions'; commands.mkdir(exist_ok=True); "
        f"finalized=pathlib.Path({str(finalized)!r}); "
        "\nif '--start-time' in sys.argv: finalized.write_text(sys.argv[-1]); sys.exit(0)"
        "\nwhile not list(commands.glob('*.json')): time.sleep(.01)"
        "\ntime.sleep(.15); finalized.write_text('drained and published')"
    )
    job = workspace.start_command([sys.executable, "-c", code], "sample")
    deadline = time.monotonic() + 5
    while not (workspace.session_dir / "live" / "runtime.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return job, finalized


def should_drain_pipeline_before_marking_operator_stop_complete(workspace, tmp_path):
    job, finalized = _cooperative_pipeline(workspace, tmp_path)
    result = workspace.dispatch("jobs.stop", {"job_id": job["id"]})
    assert result["graceful_stop"]
    assert result["status"] == "stopping"
    completed = wait_job(workspace, job["id"])
    assert completed["status"] == "completed"
    assert completed["stopped"] and not completed.get("forced_stop")
    assert finalized.read_text() == "drained and published"


def should_relaunch_seek_only_after_the_prior_run_finishes(workspace, tmp_path):
    job, finalized = _cooperative_pipeline(workspace, tmp_path)
    result = workspace.dispatch("jobs.seek", {"job_id": job["id"], "seconds": 65.25})
    assert result["waiting_for"] == job["id"]
    previous = wait_job(workspace, job["id"])
    assert previous["status"] == "completed"
    next_job = previous["seek_job_id"]
    assert wait_job(workspace, next_job)["status"] == "completed"
    assert finalized.read_text() == "00:01:05.250"
    assert workspace.jobs.get(next_job)["previous_job"] == job["id"]


def should_expose_forced_stop_when_pipeline_does_not_acknowledge(workspace):
    job = workspace.start_command([sys.executable, "-c", "import time; time.sleep(60)"], "sample")
    workspace.jobs.request_stop(job["id"], grace_seconds=0.05)
    record = wait_job(workspace, job["id"])
    assert record["status"] == "cancelled"
    assert record["forced_stop"]
    assert "incomplete outputs" in record["error"]


def should_remove_selected_source_reference_and_switch_game_root(workspace, tmp_path):
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original recording")
    workspace.import_videos({"role": "left", "paths": [str(source)]})
    selected = workspace.catalog(workspace.game())["videos"][0]
    result = workspace.dispatch("games.remove_video", {"role": "left", "path": selected["path"]})
    assert result["paths"] == []
    assert Path(selected["path"]).is_symlink()
    assert source.read_bytes() == b"original recording"
    second = tmp_path / "second-root"
    (second / "another-game").mkdir(parents=True)
    workspace.dispatch("workspace.set_root", {"path": str(second)})
    assert workspace.snapshot()["current_game"] is None
    assert workspace.list_games()[0]["id"] == "another-game"


def should_restart_a_completed_stitch_stage_with_identical_arguments(workspace, tmp_path):
    output = tmp_path / "runs.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; p=Path({str(output)!r}); p.open('a').write('run\\n')",
    ]
    first = workspace.start_command(command, "sample", "stitch")
    assert wait_job(workspace, first["id"])["status"] == "completed"
    second = workspace.dispatch("jobs.restart", {"job_id": first["id"]})
    assert wait_job(workspace, second["id"])["status"] == "completed"
    assert second["command"] == command
    assert second["previous_job"] == first["id"]
    assert output.read_text() == "run\nrun\n"


def should_retire_only_managed_calibration_and_keep_recovery_copy(workspace):
    game = workspace.game()
    write_config(
        game,
        {
            "stitching": {"frame_offsets": {"left": 3}, "stitch_frame_time": "00:00:04"},
            "rink": {"workspace_map": {}, "ice_contours_mask_count": 1},
        },
    )
    for name in (
        ".stitching_artifacts.json",
        "hm_project.pto",
        "mapping_0000.tif",
        "rink_mask_0.png",
        "left.png",
        "left.mp4",
        "sample-stitched_output-1.mkv",
        "rink_mask_0-1.png",
        "operator-notes.pto",
    ):
        (game / name).write_bytes(name.encode())
    result = workspace.dispatch("games.clean_stitching", {})
    assert not (game / "hm_project.pto").exists()
    assert (Path(result["recovery_directory"]) / "hm_project.pto").read_bytes() == b"hm_project.pto"
    for name in (
        "left.mp4",
        "sample-stitched_output-1.mkv",
        "rink_mask_0-1.png",
        "operator-notes.pto",
    ):
        assert (game / name).read_bytes() == name.encode()
    assert read_config(game)["stitching"] == {"stitch_frame_time": "00:00:04"}
    assert read_config(game)["rink"] == {}
