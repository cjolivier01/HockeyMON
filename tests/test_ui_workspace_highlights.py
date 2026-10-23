"""Render real clips, title cards, moving cues and audio with FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hmlib.ui.workspace.highlights import normalize_plan, probe_media, render_plan
from hmlib.ui.workspace.media import Media


@pytest.fixture
def archive(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required")
    path = tmp_path / "game-tracking_output-1.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=320x180:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def plan_for(archive: Path) -> dict:
    return {
        "schema": 2,
        "base_name": "highlights",
        "archive_path": str(archive),
        "width": 320,
        "height": 180,
        "fps": 10,
        "intervals": [
            {"is_card": True, "card": {"heading": "Game 1", "duration_ms": 400}},
            {
                "start_ms": 500,
                "end_ms": 1100,
                "annotations": [
                    {
                        "kind": "arrow",
                        "blink": False,
                        "motion": "keyframes",
                        "positions": [
                            {"time_ms": 500, "x": 0.2, "y": 0.4},
                            {"time_ms": 1100, "x": 0.8, "y": 0.4},
                        ],
                    },
                    {"kind": "text", "text": "GOAL: '1'", "x": 0.5, "y": 0.2, "blink": False},
                ],
            },
        ],
    }


def should_render_cards_clips_moving_annotations_and_audio(archive, tmp_path):
    plan = plan_for(archive)
    output = tmp_path / "preview" / "highlight.mp4"
    result = render_plan(plan, tmp_path, output, preview=True)
    media = probe_media(output)
    assert result["preview"]
    assert media["duration"] == pytest.approx(1, abs=0.15)
    assert (media["width"], media["height"]) == (320, 180)
    assert media["audio"]
    frame = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            "0.6",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert len(frame) == 320 * 180 * 3
    # Blue source alone has almost no red/green. Visible yellow cues add both.
    pixels = zip(frame[0::3], frame[1::3], frame[2::3], strict=True)
    assert sum(r > 120 and g > 120 and b < 150 for r, g, b in pixels) > 20


def should_publish_numbered_highlights_without_replacing_existing_files(archive, tmp_path):
    original = tmp_path / f"{tmp_path.name}-highlights-1.mp4"
    original.write_bytes(b"previous publication")
    plan = plan_for(archive)
    plan["intervals"] = [plan["intervals"][0]]
    result = render_plan(plan, tmp_path, tmp_path / "render" / "reel.mp4", preview=False)
    assert result["version"] == 2
    assert Path(result["path"]).is_file()
    assert original.read_bytes() == b"previous publication"


def should_render_full_sized_captured_tracks_without_expression_or_argument_overflow(
    archive, tmp_path
):
    plan = plan_for(archive)
    plan["intervals"] = [plan["intervals"][1]]
    cue = plan["intervals"][0]["annotations"][0]
    cue["motion"] = "track"
    cue["positions"] = [
        {"time_ms": 500 + index * 600 / 3999, "x": 0.2 + index * 0.6 / 3999, "y": 0.4}
        for index in range(4000)
    ]
    output = tmp_path / "long-track" / "highlight.mp4"
    result = render_plan(plan, tmp_path, output, preview=True)
    assert result["duration"] == pytest.approx(0.6, abs=0.15)
    assert probe_media(output)["audio"]


def should_reject_unbound_motion_and_clips_outside_archive(archive, tmp_path):
    plan = plan_for(archive)
    plan["intervals"][1]["annotations"][0]["positions"] = []
    with pytest.raises(ValueError, match="require captured"):
        normalize_plan(plan, tmp_path)
    plan = plan_for(archive)
    plan["intervals"][1]["end_ms"] = 5000
    with pytest.raises(ValueError, match="beyond"):
        render_plan(plan, tmp_path, tmp_path / "bad" / "reel.mp4", preview=False)
    assert not list(tmp_path.glob(f"{tmp_path.name}-highlights-*.mp4"))


def should_cache_bounded_media_frames_by_identity_and_timestamp(archive, tmp_path):
    media = Media(tmp_path / "cache")
    info = media.info(archive)
    assert info["fps"] == pytest.approx(10)
    assert info["duration_ms"] == pytest.approx(2000, abs=100)
    frame = media.frame(archive, 500, 160)
    assert (frame["width"], frame["height"]) == (160, 90)
    assert Path(frame["path"]).is_file()
    before = Path(frame["path"]).stat().st_mtime_ns
    assert media.frame(archive, 500, 160) == frame
    assert Path(frame["path"]).stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="Preview width"):
        media.frame(archive, 0, 4000)
