"""Real SQLite track capture, archive binding and output projection."""

import sqlite3
from pathlib import Path

import pytest

from hmlib.ui.workspace import Workspace
from hmlib.ui.workspace.files import write_json


@pytest.fixture
def recorded_workspace(tmp_path):
    with Workspace(tmp_path / "games", tmp_path / "session") as workspace:
        workspace.create_game("sample")
        game = workspace.game()
        for basename in (
            "sample-tracking_output-3.mkv",
            "sample-tracking_output-part2-3.mkv",
            "sample-stitched_output-3.mkv",
            "sample-tracking_output-4.mkv",
        ):
            (game / basename).write_bytes(b"archive")
        write_json(
            game / "recording_segments-3.json",
            {
                "segments": [
                    {
                        "filename": "tracking_output-part2.mkv",
                        "timeline_offset_seconds": 0.4,
                        "duration_seconds": 0.4,
                    }
                ]
            },
        )
        with sqlite3.connect(game / "sample_telemetry-3.db") as connection:
            connection.executescript(
                (Path(__file__).parents[1] / "hmlib/telemetry/schema.sql").read_text()
            )
            connection.execute(
                "INSERT INTO runs VALUES('run','sample','2026-10-22','hm','{}','{}',1,'completed',8)"
            )
            connection.execute(
                "INSERT INTO geometries VALUES('run',1,1000,500,'original_stitched_pixels','revision',NULL,NULL,NULL,'{}')"
            )
            for index in range(1, 9):
                connection.execute(
                    "INSERT INTO frames VALUES('run',?,0,?,0,?, ?,NULL,?,0,1,0,1)",
                    (
                        index,
                        index,
                        index,
                        1_000_000_000 + (index - 1) * 100_000_000,
                        int(index > 4),
                    ),
                )
                connection.execute(
                    "INSERT INTO tracks VALUES('run',?,0,'50',200,100,40,80,.9,0,'{}')", (index,)
                )
                connection.execute(
                    "INSERT INTO cameras VALUES('run',?,'program',100,50,400,300)", (index,)
                )
        yield workspace


def should_capture_selected_track_in_its_recorded_segment(recorded_workspace):
    workspace = recorded_workspace
    archive = workspace.game() / "sample-tracking_output-3.mkv"
    args = {"archive_path": str(archive), "time_ms": 0}
    choices = workspace.dispatch("highlights.track_choices", args)
    choice = choices["choices"][0]
    assert choice["tracking_id"] == "50"
    assert choice["box"]["x"] == pytest.approx(0.25)
    capture = workspace.dispatch(
        "highlights.capture_track", {**args, "start_ms": 0, "end_ms": 700, "choice": choice}
    )
    assert len(capture["positions"]) == 4
    assert capture["positions"][-1]["time_ms"] == 300
    assert capture["positions"][0]["x"] == pytest.approx(0.3)
    assert capture["positions"][0]["y"] == pytest.approx(1 / 6)


def should_bind_resumed_archive_and_source_origin_to_matching_pts(recorded_workspace):
    workspace = recorded_workspace
    args = {
        "archive_path": str(workspace.game() / "sample-tracking_output-part2-3.mkv"),
        "time_ms": 5000,
        "archive_offset_ms": 5000,
    }
    choices = workspace.dispatch("highlights.track_choices", args)
    choice = choices["choices"][0]
    assert choice["seek_epoch"] == 1
    capture = workspace.dispatch(
        "highlights.capture_track", {**args, "start_ms": 5000, "end_ms": 5300, "choice": choice}
    )
    assert [position["time_ms"] for position in capture["positions"]] == [5000, 5100, 5200, 5300]
    with pytest.raises(ValueError, match="publication version"):
        workspace.dispatch(
            "highlights.track_choices",
            {"archive_path": str(workspace.game() / "sample-tracking_output-4.mkv"), "time_ms": 0},
        )


def should_capture_stitched_coordinates_without_program_crop(recorded_workspace):
    workspace = recorded_workspace
    args = {"archive_path": str(workspace.game() / "sample-stitched_output-3.mkv"), "time_ms": 0}
    choice = workspace.dispatch("highlights.track_choices", args)["choices"][0]
    capture = workspace.dispatch(
        "highlights.capture_track", {**args, "start_ms": 0, "end_ms": 300, "choice": choice}
    )
    assert capture["positions"][0]["x"] == pytest.approx(0.22)
    assert capture["positions"][0]["y"] == pytest.approx(0.2)
