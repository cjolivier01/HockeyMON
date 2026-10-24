"""Freeze real telemetry and media as an isolated camera replay candidate."""

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from hmlib.telemetry.database import create_database, read_database
from hmlib.ui.workspace import Workspace
from hmlib.ui.workspace.files import read_config, write_config, write_json


@pytest.fixture
def replay_workspace(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("Replay integration requires FFmpeg and FFprobe")
    with Workspace(tmp_path / "games", tmp_path / "session") as workspace:
        workspace.create_game("sample")
        game = workspace.game()
        source = game / "source.mp4"
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
                "testsrc2=size=64x48:rate=10:duration=4",
                "-c:v",
                "mpeg4",
                str(source),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        write_config(
            game,
            {
                "game": {"videos": {"source": [source.name]}},
                "rink": {"camera": {"zoom_in_aggressiveness": 99}},
            },
        )
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[8:40, 6:58] = 255
        ok, png = cv2.imencode(".png", mask)
        assert ok
        blob = png.tobytes()
        # Today's mask intentionally differs from the recording's frozen mask.
        cv2.imwrite(str(game / "rink_mask_0.png"), np.ones_like(mask) * 255)
        recorded = {
            "source_video_paths": [str(source)],
            "game_config": {
                "game": {"videos": {"source": [str(source)]}},
                "rink": {"camera": {"zoom_in_aggressiveness": 10, "pan_reactivity": 13}},
                "initial_args": {"work_dir": "/old-run"},
                "video_out": {
                    "output_video_path": "/old-run/tracking_output.mkv",
                    "save_frame_dir": "/old-run/frames",
                },
                "aspen": {
                    "video_out": {"output_video_path": "/legacy-run/tracking_output.mkv"},
                    "plugins": {
                        "video_out": {
                            "class": "hmlib.aspen.plugins.video_out_plugin.VideoOutPlugin",
                            "params": {"output_video_path": "/resolved-run/tracking_output.mkv"},
                        }
                    },
                },
            },
        }
        connection = create_database(game / "sample_telemetry-1.db")
        try:
            with connection:
                for run in ("selected", "other"):
                    connection.execute(
                        "INSERT INTO runs VALUES(?, 'sample','2026-10-22','hm','{}',?,1,'end-of-stream',40)",
                        (run, yaml.safe_dump(recorded)),
                    )
                    connection.execute(
                        "INSERT INTO geometries VALUES(?,1,64,48,'original_stitched_pixels','revision','png',?,?, '[[1,0,0],[0,1,0]]')",
                        (run, blob, hashlib.sha256(blob).hexdigest()),
                    )
                    connection.execute(
                        "INSERT INTO config_events VALUES(?,1,3,'camera-policy','updated',?, '', '')",
                        (run, '{"camera":{"zoom_in_aggressiveness":20}}'),
                    )
                    for frame in range(1, 41):
                        connection.execute(
                            "INSERT INTO frames VALUES(?,?,0,?,0,?,?,NULL,?,0,1,0,1)",
                            (
                                run,
                                frame,
                                frame,
                                frame,
                                1_000_000_000 + (frame - 1) * 100_000_000,
                                int(frame > 20),
                            ),
                        )
                        connection.execute(
                            "INSERT INTO tracks VALUES(?,?,0,'50',20,10,4,8,.9,0,'{}')",
                            (run, frame),
                        )
                        connection.execute(
                            "INSERT INTO cameras VALUES(?,?,'program',10,5,40,30)", (run, frame)
                        )
        finally:
            connection.close()
        yield workspace


def replay_args(workspace, **changes):
    return {
        "kind": "camera",
        "changes": {"rink.camera.zoom_in_aggressiveness": 32},
        "replay": {
            "recording_path": str(workspace.game() / "sample_telemetry-1.db"),
            "run_id": "selected",
            "start_ms": 300,
            "end_ms": 600,
            **changes,
        },
    }


def should_freeze_selected_run_interval_config_and_mask_without_modifying_inputs(replay_workspace):
    workspace = replay_workspace
    game = workspace.game()
    originals = {path.name: path.read_bytes() for path in game.iterdir() if path.is_file()}
    experiment = workspace.create_experiment(replay_args(workspace))
    replay = experiment["replay"]
    assert replay["run_id"] == "selected"
    assert replay["start_frame"] == 3
    assert replay["max_frames"] == 4
    assert replay["start_ms"] == 300
    assert replay["end_ms"] == 600
    with read_database(replay["input_tracking_data"]) as connection:
        assert list(map(tuple, connection.execute("SELECT run_id,sample_count FROM runs"))) == [
            ("selected", 4)
        ]
        assert [
            row[0]
            for row in connection.execute("SELECT source_frame FROM frames ORDER BY sample_id")
        ] == [4, 5, 6, 7]
        assert connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 4
        assert connection.execute("PRAGMA foreign_key_check").fetchone() is None
    candidate = Path(experiment["game_directory"])
    config = read_config(candidate)
    assert config["game"]["videos"] == {"source": [str(game / "source.mp4")]}
    assert config["rink"]["camera"] == {"pan_reactivity": 13, "zoom_in_aggressiveness": 32}
    assert config["hm_ui"]["run"]["detector"] == "saved"
    assert "initial_args" not in config
    assert "output_video_path" not in config["video_out"]
    assert "output_video_path" not in config["aspen"]["video_out"]
    assert (
        config["aspen"]["plugins"]["video_out"]["params"]["output_video_path"]
        == "GLOBAL.video_out.output_video_path"
    )
    mask = cv2.imread(str(candidate / "rink_mask_0.png"), cv2.IMREAD_GRAYSCALE)
    assert np.count_nonzero(mask) == 32 * 52
    assert config["rink"]["ice_contours_combined_bbox"] == [6, 8, 58, 40]
    for name, contents in originals.items():
        assert (game / name).read_bytes() == contents


def should_keep_selected_baseline_and_filter_before_trajectory_sampling(replay_workspace):
    workspace = replay_workspace
    experiment = workspace.create_experiment(replay_args(workspace))
    # A later publication must never replace an experiment's selected baseline.
    shutil.copy2(
        workspace.game() / "sample_telemetry-1.db", workspace.game() / "sample_telemetry-2.db"
    )
    result = workspace.experiment_telemetry({"experiment_id": experiment["id"], "max_points": 2})
    assert result["baseline"]["run_id"] == "selected"
    assert result["baseline"]["database"].endswith("sample_telemetry-1.db")
    assert [point["time_seconds"] for point in result["baseline"]["series"]["program"]] == [
        0.3,
        0.6,
    ]
    assert result["trial"] is None


def should_rebase_resumed_stitched_archive_frames_to_its_segment_origin(replay_workspace):
    workspace = replay_workspace
    game = workspace.game()
    archive = game / "sample-stitched_output-part2-1.mp4"
    shutil.copy2(game / "source.mp4", archive)
    write_json(
        game / "recording_segments-1.json",
        {
            "segments": [
                {
                    "filename": "stitched_output-part2.mp4",
                    "timeline_offset_seconds": 2,
                    "duration_seconds": 4,
                }
            ]
        },
    )
    experiment = workspace.create_experiment(
        replay_args(workspace, source_path=str(archive), start_ms=2200, end_ms=2500)
    )
    assert experiment["replay"]["start_frame"] == 2
    assert experiment["replay"]["max_frames"] == 4
    with read_database(experiment["replay"]["input_tracking_data"]) as connection:
        assert [
            row[0]
            for row in connection.execute("SELECT source_frame FROM frames ORDER BY sample_id")
        ] == [3, 4, 5, 6]


@pytest.mark.parametrize(
    "name", ["sample-stitched_output-2.mp4", "sample-tracking_output-1.mp4", "unrecorded.mp4"]
)
def should_reject_replay_media_unrelated_to_the_selected_recording(replay_workspace, name):
    workspace = replay_workspace
    game = workspace.game()
    source = game / name
    shutil.copy2(game / "source.mp4", source)
    # Even an imported original with identical dimensions/FPS must be one of
    # the selected run's recorded sources; today's configuration cannot vouch for it.
    config = read_config(game)
    config["game"]["videos"]["source"].append(name)
    write_config(game, config)
    with pytest.raises(ValueError, match="paired with the selected telemetry"):
        workspace.create_experiment(replay_args(workspace, source_path=str(source)))


def should_accept_explicit_recorded_original_for_replay(replay_workspace):
    workspace = replay_workspace
    path = workspace.game() / "source.mp4"
    result = workspace.create_experiment(replay_args(workspace, source_path=str(path)))
    assert read_config(Path(result["game_directory"]))["game"]["videos"]["source"] == [str(path)]


def should_reject_raw_camera_replay_even_when_dimensions_match_stitched_geometry(replay_workspace):
    workspace = replay_workspace
    game = workspace.game()
    for name in ("left.mp4", "right.mp4"):
        shutil.copy2(game / "source.mp4", game / name)
    with sqlite3.connect(game / "sample_telemetry-1.db") as connection:
        raw = connection.execute(
            "SELECT effective_config FROM runs WHERE run_id='selected'"
        ).fetchone()[0]
        config = yaml.safe_load(raw)
        config["source_video_paths"] = [str(game / name) for name in ("left.mp4", "right.mp4")]
        config["game_config"]["stitching"] = {"enabled": True}
        connection.execute(
            "UPDATE runs SET effective_config=? WHERE run_id='selected'", (json.dumps(config),)
        )
    with pytest.raises(ValueError, match="paired with the selected telemetry"):
        workspace.create_experiment(replay_args(workspace, source_path=str(game / "right.mp4")))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_id": "missing"}, "does not belong"),
        ({"run_id": None}, "Select a completed"),
        ({"start_ms": 1900, "end_ms": 2100}, "one geometry"),
        ({"start_ms": 5000, "end_ms": 6000}, "no recorded frames"),
    ],
)
def should_reject_ambiguous_or_discontinuous_replay_ranges(replay_workspace, changes, message):
    with pytest.raises(ValueError, match=message):
        replay_workspace.create_experiment(replay_args(replay_workspace, **changes))


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("UPDATE geometries SET width=128", "dimensions differ"),
        ("UPDATE geometries SET mask_sha256='changed'", "checksum"),
        ("UPDATE frames SET pts_ns=1300000000 WHERE sample_id=5", "increasing recorded timeline"),
    ],
)
def should_reject_invalid_recorded_geometry_and_timestamps(replay_workspace, sql, message):
    with sqlite3.connect(replay_workspace.game() / "sample_telemetry-1.db") as connection:
        connection.execute(sql)
    with pytest.raises(ValueError, match=message):
        replay_workspace.create_experiment(replay_args(replay_workspace))


def should_refuse_replay_range_override_and_changed_source_before_launch(replay_workspace):
    workspace = replay_workspace
    experiment = workspace.create_experiment(replay_args(workspace))
    with pytest.raises(ValueError, match="owned by the experiment"):
        workspace.run_experiment(
            {"experiment_id": experiment["id"], "arguments": ["--start-frame=20"]}
        )
    (workspace.game() / "source.mp4").touch()
    with pytest.raises(RuntimeError, match="source changed"):
        workspace.run_experiment({"experiment_id": experiment["id"]})
    assert not workspace.jobs.records


def should_replay_analytics_recording_without_retaining_live_model_branches(replay_workspace):
    from hmlib.cli.hmtrack import _enable_load_tracking_plugin
    from hmlib.config import load_yaml_files_ordered
    from hmlib.ui.analytics_graph import graph_paths
    from hmlib.ui.runtime_setup import (
        apply_engine_preferences,
        launch_arguments,
        launch_configuration,
    )

    workspace = replay_workspace
    database = workspace.game() / "sample_telemetry-1.db"
    with sqlite3.connect(database) as connection:
        original = yaml.safe_load(
            connection.execute(
                "SELECT effective_config FROM runs WHERE run_id='selected'"
            ).fetchone()[0]
        )
        original["game_config"] = load_yaml_files_ordered(
            [str(path) for path in graph_paths("pose-jersey-actions")],
            base=original["game_config"],
        )
        original["game_config"]["hm_ui"] = {"run": {"analytics": "pose-jersey-actions"}}
        connection.execute(
            "UPDATE runs SET effective_config=? WHERE run_id='selected'",
            (yaml.safe_dump(original),),
        )
    experiment = workspace.create_experiment(replay_args(workspace))
    candidate = Path(experiment["game_directory"])
    config = launch_configuration(candidate, [])
    arguments = launch_arguments(candidate, "track", config=config)
    paths = [arguments[i + 1] for i, arg in enumerate(arguments) if arg == "--config"]
    config = load_yaml_files_ordered(paths, base=config)
    apply_engine_preferences(config)
    _enable_load_tracking_plugin(config, prefer_detector=False, disable_detector=True)
    nodes = config["aspen"]["plugins"]
    assert all(
        name not in nodes or not nodes[name].get("enabled", True)
        for name in (
            "pose_factory",
            "pose",
            "jersey_numbers",
            "actions",
            "detector_factory",
            "detector",
        )
    )
    assert nodes["load_tracking"]["params"]["load_jerseys"] is True
    assert nodes["load_tracking"]["params"]["load_actions"] is True
    assert nodes["overlays"].get("enabled", True)
    assert nodes["camera_controller"].get("enabled", True)
    with read_database(experiment["replay"]["input_tracking_data"]) as connection:
        archived = yaml.safe_load(
            connection.execute("SELECT effective_config FROM runs").fetchone()[0]
        )
    assert "actions" in archived["game_config"]["aspen"]["plugins"]
