"""Capture highlight positions from the selected archive's recorded tracks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from hmlib.telemetry.database import read_database

from .highlights import number

_SEGMENT = ("geometry_id", "seek_epoch", "reset_epoch", "source_id")
_ROWS = """
 SELECT f.pts_ns,f.sample_id,f.geometry_id,f.seek_epoch,f.reset_epoch,f.source_id,
        t.tracking_id,t.left,t.top,t.width,t.height,
        g.width AS canvas_width,g.height AS canvas_height,
        c.left AS camera_left,c.top AS camera_top,c.width AS camera_width,c.height AS camera_height,
        r.edge_rotation_left,r.edge_rotation_right
 FROM frames f JOIN tracks t USING(run_id,sample_id)
 JOIN geometries g USING(run_id,geometry_id)
 LEFT JOIN cameras c ON c.run_id=f.run_id AND c.sample_id=f.sample_id AND c.role='program'
 LEFT JOIN replay_frames r USING(run_id,sample_id)
 WHERE f.run_id=? AND f.pts_ns BETWEEN ? AND ?
"""


class TrackCapture:
    def __init__(self, connection, run, route: str):
        self.connection = connection
        self.run = run
        self.route = route
        if route not in {"program", "program-full", "stitched"}:
            raise ValueError("Track geometry must be program, program-full, or stitched")
        snapshot = yaml.safe_load(run["effective_config"]) or {}
        self.config = snapshot.get("game_config", snapshot)
        self.camera = self.config.get("rink", {}).get("camera", {})
        self.events = [
            (row["sample_boundary"], json.loads(row["value"]).get("camera", {}))
            for row in connection.execute(
                "SELECT sample_boundary,value FROM config_events WHERE run_id=? AND kind='camera-policy' ORDER BY sample_boundary,event_id",
                (run["run_id"],),
            )
        ]
        self.gaussians = {}
        self.suppress_rotation = False
        if run["producer"] == "hm" and route != "stitched":
            from hmlib.stitching.settings import read_stitching_settings

            settings = read_stitching_settings(self.config)
            self.suppress_rotation = settings.mapping_backend == "nona" and any(
                settings.framing.rotation_degrees[1:]
            )

    def point(self, row, x: float, y: float) -> tuple[float, float]:
        width, height = row["canvas_width"], row["canvas_height"]
        if self.route == "stitched":
            return x / width, y / height
        left, top, cw, ch = (row[f"camera_{key}"] for key in ("left", "top", "width", "height"))
        if left is None or cw <= 0 or ch <= 0:
            raise ValueError("Program track capture requires recorded camera boxes")
        cx, cy = left + cw / 2, top + ch / 2
        camera = self.camera
        for boundary, value in self.events:
            if boundary > row["sample_id"]:
                break
            camera = {**camera, **value}
        if row["edge_rotation_left"] is not None:
            angle = (
                row["edge_rotation_left"] * (1 - cx / (width / 2))
                if cx < width / 2
                else row["edge_rotation_right"] * ((width / 2 - cx) / (width / 2))
            )
        elif self.run["producer"] == "hm":
            angle = camera.get("fixed_edge_rotation_angle", 0)
            if isinstance(angle, (list, tuple)):
                angle = angle[0 if cx < width // 2 else 1]
            if angle and not self.suppress_rotation:
                from hmlib.utils.distributions import ImageHorizontalGaussianDistribution

                if width not in self.gaussians:
                    self.gaussians[width] = ImageHorizontalGaussianDistribution(int(width))
                weight = float(
                    self.gaussians[width].get_gaussian_y_from_image_x_position(int(cx), wide=True)
                )
                # Python's image rotation uses visual counterclockwise angles.
                angle = -float(angle) * weight * (-1 if cx < width / 2 else 1)
            else:
                angle = 0
            cx, cy = int(cx), int(cy)
        else:
            raise ValueError(
                "This recording has no saved Program rotation; select stitched geometry"
            )
        radians = math.radians(angle)
        rx = cx + (x - cx) * math.cos(radians) - (y - cy) * math.sin(radians)
        ry = cy + (x - cx) * math.sin(radians) + (y - cy) * math.cos(radians)
        if self.route == "program-full" or not camera.get("crop_image", True):
            return rx / width, ry / height
        if self.run["producer"] == "hm":
            right, bottom = min(int(left + cw), width), min(int(top + ch), height)
            left, top = int(left), int(top)
            cw, ch = right - left, bottom - top
        if cw <= 0 or ch <= 0:
            raise ValueError("Recorded camera crop has no visible area")
        return (rx - left) / cw, (ry - top) / ch


def capture_tracks(
    database: Path, game_id: str, args: dict, *, route: str, segment_offset_ms: float, capture: bool
) -> dict:
    with read_database(database) as connection:
        runs = list(
            connection.execute(
                "SELECT * FROM runs WHERE game_id=? AND completed=1 ORDER BY started_utc DESC",
                (game_id,),
            )
        )
        if not runs:
            raise ValueError("The matching telemetry contains no completed run for this game")
        selected = args.get("run_id")
        run = (
            next((item for item in runs if item["run_id"] == selected), None)
            if selected
            else runs[0]
        )
        if run is None:
            raise ValueError("The selected run does not belong to this archive's telemetry")
        first = connection.execute(
            "SELECT MIN(pts_ns) FROM frames WHERE run_id=?", (run["run_id"],)
        ).fetchone()[0]
        if first is None:
            raise ValueError("The selected recording has no timestamped frames")
        offset = (
            first / 1e6
            + segment_offset_ms
            - number(args.get("archive_offset_ms", 0), "Archive origin", 0, 1e12)
        )
        result = {
            "database": str(database),
            "run_id": run["run_id"],
            "route": route,
            "telemetry_offset_ms": offset,
            "runs": [
                {key: item[key] for key in ("run_id", "producer", "started_utc")} for item in runs
            ],
        }
        projection = TrackCapture(connection, run, route)
        if capture:
            start = number(args["start_ms"], "Cue start", 0, 1e12)
            end = number(args["end_ms"], "Cue end", start + 1, start + 120000)
            choice = args["choice"]
            conditions = " AND t.tracking_id=?" + "".join(f" AND f.{key}=?" for key in _SEGMENT)
            rows = connection.execute(
                _ROWS + conditions + " ORDER BY f.pts_ns,f.sample_id LIMIT 4001",
                (
                    run["run_id"],
                    round((start + offset) * 1e6),
                    round((end + offset) * 1e6),
                    str(choice["tracking_id"]),
                    *(int(choice[key]) for key in _SEGMENT),
                ),
            )
            positions = []
            for row in rows:
                timestamp = row["pts_ns"] / 1e6 - offset
                if positions and timestamp <= positions[-1]["time_ms"]:
                    raise ValueError("The selected track has ambiguous timestamps")
                x, y = projection.point(row, row["left"] + row["width"] / 2, row["top"])
                if not math.isfinite(x) or not math.isfinite(y) or abs(x) > 2 or abs(y) > 2:
                    raise ValueError("The selected track leaves the supported annotation canvas")
                positions.append({"time_ms": timestamp, "x": x, "y": y})
            if not positions or len(positions) > 4000:
                raise ValueError(
                    "The selected track needs between one and 4000 positions in this cue"
                )
            return {
                **result,
                "motion": "track",
                "positions": positions,
                "tracking_id": str(choice["tracking_id"]),
                **{key: int(choice[key]) for key in _SEGMENT},
                "start_ms": start,
                "end_ms": end,
            }
        timestamp = number(args["time_ms"], "Track selection time", 0, 1e12)
        pts = timestamp + offset
        rows = connection.execute(
            _ROWS + " ORDER BY ABS(f.pts_ns-?),f.sample_id,t.ordinal LIMIT 2048",
            (run["run_id"], round((pts - 100) * 1e6), round((pts + 100) * 1e6), round(pts * 1e6)),
        )
        choices, seen = [], set()
        for row in rows:
            identity = (row["tracking_id"], *(row[key] for key in _SEGMENT))
            if identity in seen:
                continue
            seen.add(identity)
            points = [
                projection.point(row, x, y)
                for x in (row["left"], row["left"] + row["width"])
                for y in (row["top"], row["top"] + row["height"])
            ]
            xs, ys = zip(*points)
            if not all(math.isfinite(value) for value in (*xs, *ys)):
                raise ValueError("The recorded track has nonfinite projected coordinates")
            choices.append(
                {
                    "tracking_id": row["tracking_id"],
                    **{key: row[key] for key in _SEGMENT},
                    "time_ms": row["pts_ns"] / 1e6 - offset,
                    "box": {
                        "x": min(xs),
                        "y": min(ys),
                        "width": max(xs) - min(xs),
                        "height": max(ys) - min(ys),
                    },
                }
            )
        if not choices:
            raise ValueError("No recorded tracks within 100 ms of this frame")
        return {**result, "choices": choices}
