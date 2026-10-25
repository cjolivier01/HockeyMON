"""Bounded, identity-keyed desktop frame inspection and telemetry plots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from .files import read_json, write_json
from .highlights import number


class Media:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(path: Path) -> str:
        info = path.stat()
        return hashlib.sha256(
            json.dumps(
                [str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
            ).encode()
        ).hexdigest()

    def info(self, path: Path) -> dict:
        key = self.identity(path)
        cached = self.directory / f"{key}.json"
        if cached.exists():
            return read_json(cached)
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), None)
        if video is None:
            raise ValueError(f"No video stream in {path}")
        duration = data.get("format", {}).get("duration", video.get("duration"))
        if duration is None:
            raise ValueError(f"Cannot determine the media duration: {path}")
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
        if not rate or rate == "0/0":
            raise ValueError(f"Cannot determine the media frame rate: {path}")
        value = {
            "path": str(path),
            "width": int(video["width"]),
            "height": int(video["height"]),
            "duration_ms": number(float(duration) * 1000, "Media duration", 1, 1e12),
            "fps": number(float(Fraction(rate)), "Media frame rate", 0.001, 1000),
            "audio": any(item.get("codec_type") == "audio" for item in streams),
            "identity": key,
        }
        write_json(cached, value)
        return value

    def frame(self, path: Path, time_ms: float = 0, width: int = 1280) -> dict:
        width = int(number(width, "Preview width", 16, 1920))
        info = self.info(path)
        timestamp = number(time_ms, "Frame time", 0, info["duration_ms"])
        timestamp = min(timestamp, max(0, info["duration_ms"] - 1000 / info["fps"]))
        digest = hashlib.sha256(f"{info['identity']}:{timestamp:.3f}:{width}".encode()).hexdigest()
        target = self.directory / f"{digest}.jpg"
        if not target.exists():
            temporary = self.directory / f"{digest}.partial.jpg"
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        str(timestamp / 1000),
                        "-i",
                        str(path),
                        "-frames:v",
                        "1",
                        "-vf",
                        f"scale={min(width, info['width'])}:-2",
                        "-q:v",
                        "3",
                        str(temporary),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=45,
                )
                if not temporary.is_file() or not temporary.stat().st_size:
                    raise RuntimeError("The selected video did not yield a preview frame")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        from PIL import Image

        with Image.open(target) as frame:
            frame_width, frame_height = frame.size
        return {
            **info,
            "path": str(target),
            "source": str(path),
            "time_ms": timestamp,
            "width": frame_width,
            "height": frame_height,
        }


def trajectory(
    database: Path,
    *,
    max_points: int = 2000,
    run_id: str | None = None,
    start_seconds: float = 0,
    end_seconds: float | None = None,
) -> dict:
    max_points = int(number(max_points, "Trajectory point limit", 2, 10000))
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0] != 1213027156
            or connection.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise ValueError("Unsupported HockeyMON telemetry database")
        if run_id:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id=? AND completed=1", (run_id,)
            ).fetchone()
        else:
            run = connection.execute(
                "SELECT * FROM runs WHERE completed=1 ORDER BY started_utc DESC LIMIT 1"
            ).fetchone()
        if run is None:
            raise ValueError("Telemetry database has no selected completed run")
        first_pts = connection.execute(
            "SELECT MIN(pts_ns) FROM frames WHERE run_id=?", (run["run_id"],)
        ).fetchone()[0]
        if first_pts is None:
            raise ValueError("Telemetry run has no timestamped frames")
        start_pts = round(first_pts + number(start_seconds, "Trajectory start", 0, 1e12) * 1e9)
        end_pts = (
            round(first_pts + number(end_seconds, "Trajectory end", start_seconds, 1e12) * 1e9)
            if end_seconds is not None
            else 2**63 - 1
        )
        roles = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT role FROM cameras WHERE run_id=? ORDER BY role", (run["run_id"],)
            )
        ]
        series: dict[str, list[dict[str, Any]]] = {}
        for role in roles:
            count = connection.execute(
                "SELECT COUNT(*) FROM cameras c JOIN frames f USING(run_id,sample_id) WHERE c.run_id=? AND c.role=? AND f.pts_ns BETWEEN ? AND ?",
                (run["run_id"], role, start_pts, end_pts),
            ).fetchone()[0]
            stride = max(1, (count + max_points - 3) // (max_points - 1))
            query = """
                SELECT f.pts_ns,f.geometry_id,f.source_id,f.seek_epoch,f.reset_epoch,
                       c.left,c.top,c.width,c.height,g.width AS canvas_width,g.height AS canvas_height
                FROM cameras c JOIN frames f USING(run_id,sample_id)
                JOIN geometries g ON g.run_id=f.run_id AND g.geometry_id=f.geometry_id
                WHERE c.run_id=? AND c.role=? AND f.pts_ns BETWEEN ? AND ? ORDER BY f.sample_id
            """
            points = []
            for index, row in enumerate(
                connection.execute(query, (run["run_id"], role, start_pts, end_pts))
            ):
                if index % stride and index + 1 != count:
                    continue
                points.append(
                    {
                        "time_seconds": (row["pts_ns"] - first_pts) / 1e9,
                        "x": (row["left"] + row["width"] / 2) / row["canvas_width"],
                        "y": (row["top"] + row["height"] / 2) / row["canvas_height"],
                        "width": row["width"] / row["canvas_width"],
                        "height": row["height"] / row["canvas_height"],
                        "segment": [
                            row[key]
                            for key in ("geometry_id", "source_id", "seek_epoch", "reset_epoch")
                        ],
                    }
                )
            series[role] = points
        if not series:
            raise ValueError("Telemetry run contains no camera trajectories")
        return {
            "database": str(database),
            "run_id": run["run_id"],
            "producer": run["producer"],
            "series": series,
        }
    finally:
        connection.close()
