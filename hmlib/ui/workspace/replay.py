"""Freeze a selected telemetry interval for inference-free camera experiments."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import yaml

from hmlib.telemetry.database import TABLES, create_database, read_database

from .files import atomic_bytes, set_path
from .highlights import number
from .media import Media


def replay_sources(database: Path, game: Path, catalog: dict) -> dict:
    with read_database(database) as connection:
        runs = []
        for run in connection.execute(
            "SELECT * FROM runs WHERE game_id=? AND completed=1 ORDER BY started_utc DESC",
            (game.name,),
        ):
            first = connection.execute(
                "SELECT MIN(pts_ns) FROM frames WHERE run_id=?", (run["run_id"],)
            ).fetchone()[0]
            if first is None:
                continue
            snapshot = yaml.safe_load(run["effective_config"]) or {}
            segments = []
            for segment in connection.execute(
                "SELECT geometry_id,seek_epoch,reset_epoch,source_id,MIN(pts_ns) AS first,MAX(pts_ns) AS last,COUNT(*) AS samples FROM frames WHERE run_id=? GROUP BY geometry_id,seek_epoch,reset_epoch,source_id ORDER BY first",
                (run["run_id"],),
            ):
                segments.append(
                    {
                        **{
                            key: segment[key]
                            for key in ("geometry_id", "seek_epoch", "reset_epoch", "source_id")
                        },
                        "start_ms": (segment["first"] - first) / 1e6,
                        "end_ms": (segment["last"] - first) / 1e6,
                        "sample_count": segment["samples"],
                    }
                )
            runs.append(
                {
                    "run_id": run["run_id"],
                    "producer": run["producer"],
                    "start_ms": 0,
                    "end_ms": max(item["end_ms"] for item in segments),
                    "sample_count": sum(item["sample_count"] for item in segments),
                    "source_paths": snapshot.get("source_video_paths", []),
                    "segments": segments,
                }
            )
    return {
        "recording_path": str(database),
        "runs": runs,
        "sources": [*catalog["videos"], *catalog["archives"]],
    }


def prepare_replay(
    database: Path, game: Path, directory: Path, args: dict, catalog: dict, media: Media
) -> tuple[dict, dict]:
    with read_database(database) as source:
        runs = list(
            source.execute(
                "SELECT * FROM runs WHERE game_id=? AND completed=1 ORDER BY started_utc DESC",
                (game.name,),
            )
        )
        selected = args.get("run_id")
        if not selected and len(runs) != 1:
            raise ValueError("Select a completed telemetry run for the experiment")
        run = next((row for row in runs if not selected or row["run_id"] == selected), None)
        if run is None:
            raise ValueError("The selected replay run does not belong to this game")
        first = source.execute(
            "SELECT MIN(pts_ns) FROM frames WHERE run_id=?", (run["run_id"],)
        ).fetchone()[0]
        if first is None:
            raise ValueError("Camera replay requires timestamped source frames")
        start = number(args.get("start_ms", 0), "Replay start", 0, 1e12)
        end = (
            number(args["end_ms"], "Replay end", start + 1, 1e12)
            if args.get("end_ms") is not None
            else 1e12
        )
        frames = list(
            source.execute(
                "SELECT * FROM frames WHERE run_id=? AND pts_ns BETWEEN ? AND ? ORDER BY sample_id",
                (
                    run["run_id"],
                    round(first + start * 1e6),
                    min(2**63 - 1, round(first + end * 1e6)),
                ),
            )
        )
        if not frames:
            raise ValueError("The replay interval has no recorded frames")
        if any(
            right["pts_ns"] <= left["pts_ns"] or right["sample_id"] != left["sample_id"] + 1
            for left, right in itertools.pairwise(frames)
        ):
            raise ValueError("Replay requires a continuous, increasing recorded timeline")
        if (
            len(
                {
                    tuple(
                        row[key]
                        for key in ("geometry_id", "seek_epoch", "reset_epoch", "source_id")
                    )
                    for row in frames
                }
            )
            != 1
        ):
            raise ValueError(
                "Choose a replay range within one geometry, source, seek and reset segment"
            )
        ids = [row["source_frame"] for row in frames]
        if any(right != left + 1 for left, right in itertools.pairwise(ids)):
            raise ValueError("Replay requires consecutive source frames")
        geometry = source.execute(
            "SELECT * FROM geometries WHERE run_id=? AND geometry_id=?",
            (run["run_id"], frames[0]["geometry_id"]),
        ).fetchone()
        if geometry is None or geometry["coordinate_space"] != "original_stitched_pixels":
            raise ValueError("Replay requires the original recorded tracking geometry")
        snapshot = yaml.safe_load(run["effective_config"]) or {}
        config = copy.deepcopy(snapshot.get("game_config", snapshot))
        # Telemetry stores the resolved runtime configuration, including paths
        # and constructor arguments that belonged to the original job. A new
        # candidate must let its CLI choose and publish its own output paths.
        config.pop("initial_args", None)
        for block in (config.get("video_out", {}), config.get("aspen", {}).get("video_out", {})):
            for key in ("output_video_path", "save_frame_dir"):
                block.pop(key, None)
        for plugin in config.get("aspen", {}).get("plugins", {}).values():
            if str(plugin.get("class", "")).endswith(".VideoOutPlugin"):
                params = plugin.get("params") or {}
                params.pop("output_video_path", None)
                params.pop("save_frame_dir", None)
        for event in source.execute(
            "SELECT value FROM config_events WHERE run_id=? AND kind='camera-policy' AND sample_boundary<=? ORDER BY sample_boundary,event_id",
            (run["run_id"], frames[0]["sample_id"]),
        ):
            for key, value in json.loads(event["value"]).get("camera", {}).items():
                set_path(config, "rink.camera." + key, value)
        original = snapshot.get("source_video_paths", [])
        permitted = {
            Path(item["path"]).resolve() for item in (*catalog["videos"], *catalog["archives"])
        }
        permitted.update(Path(value).expanduser().resolve() for value in original)
        requested = args.get("source_path")
        if requested:
            source_path = Path(requested).expanduser().resolve(strict=True)
            if source_path not in permitted:
                raise ValueError("Replay source must be a recorded input or game archive")
        elif len(original) == 1 and Path(original[0]).is_file():
            source_path = Path(original[0]).resolve()
        else:
            candidates = [
                Path(item["path"])
                for item in catalog["archives"]
                if item["kind"] == "stitched" and str(database) in item["recordings"]
            ]
            if not candidates:
                raise ValueError(
                    "Select a full-resolution Stitched archive for this telemetry recording"
                )
            source_path = candidates[-1]
        info = media.info(source_path)
        if (info["width"], info["height"]) != (geometry["width"], geometry["height"]):
            raise ValueError(
                "Replay source dimensions differ from the recorded tracking geometry; select the original or full Stitched archive"
            )
        if len(frames) > 1:
            recorded_fps = (len(frames) - 1) * 1e9 / (frames[-1]["pts_ns"] - frames[0]["pts_ns"])
            if abs(recorded_fps - info["fps"]) > 0.01:
                raise ValueError("Replay source frame rate differs from the selected telemetry")
        archive = next(
            (item for item in catalog["archives"] if Path(item["path"]).resolve() == source_path),
            None,
        )
        if archive:
            source_origin = archive.get("timeline_offset_seconds", 0)
            mapped = [
                round(((row["pts_ns"] - first) / 1e9 - source_origin) * info["fps"]) + 1
                for row in frames
            ]
        else:
            mapped = ids
        if mapped[0] < 1 or mapped[-1] > math.ceil(info["duration_ms"] * info["fps"] / 1000):
            raise ValueError("The replay interval is outside the selected source video")
        frozen = directory / "input-telemetry.db"
        frame_mapping = dict(zip(ids, mapped))
        target = create_database(frozen)
        try:
            with target:
                for table in TABLES:
                    columns = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
                    query, params = f"SELECT * FROM {table} WHERE run_id=?", [run["run_id"]]
                    if "sample_id" in columns:
                        query += " AND sample_id BETWEEN ? AND ?"
                        params += [frames[0]["sample_id"], frames[-1]["sample_id"]]
                    for row in source.execute(query, params):
                        values = list(row)
                        if table == "frames":
                            values[columns.index("source_frame")] = frame_mapping[
                                row["source_frame"]
                            ]
                        if table == "runs":
                            values[columns.index("sample_count")] = len(frames)
                        target.execute(
                            f"INSERT INTO {table} VALUES({','.join('?' for _ in columns)})", values
                        )
        finally:
            target.close()
        set_path(config, "game.videos", {"source": [str(source_path)]})
        set_path(config, "stitching.enabled", False)
        for key, value in {
            "detector": "saved",
            "precision": "saved",
            "pose_precision": "saved",
            "analytics": "tracking",
            "save_video": True,
            "batch_size": 1,
        }.items():
            set_path(config, "hm_ui.run." + key, value)
        replay = {
            "recording_path": str(database),
            "recording_identity": Media.identity(database),
            "run_id": run["run_id"],
            "source_path": str(source_path),
            "source_identity": info["identity"],
            "input_tracking_data": str(frozen),
            "start_frame": mapped[0] - 1,
            "max_frames": len(frames),
            "start_ms": (frames[0]["pts_ns"] - first) / 1e6,
            "end_ms": (frames[-1]["pts_ns"] - first) / 1e6,
            "geometry_id": geometry["geometry_id"],
            "dimensions": [geometry["width"], geometry["height"]],
        }
        return replay, config


def restore_replay_mask(replay: dict, candidate: Path, config: dict) -> None:
    """Restore the archived rink in tracking pixels after copying game artifacts."""
    import cv2
    import numpy as np

    with read_database(replay["input_tracking_data"]) as connection:
        geometry = connection.execute(
            "SELECT * FROM geometries WHERE run_id=? AND geometry_id=?",
            (replay["run_id"], replay["geometry_id"]),
        ).fetchone()
    blob = geometry["mask"]
    if not blob or geometry["mask_codec"] != "png":
        raise ValueError("Camera replay requires an archived PNG rink mask")
    if hashlib.sha256(blob).hexdigest() != geometry["mask_sha256"]:
        raise ValueError("The recorded replay rink mask checksum is invalid")
    mask = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("The recorded replay rink mask cannot be decoded")
    affine = np.asarray(json.loads(geometry["mask_to_tracking"]), dtype=np.float64)
    if (
        affine.shape != (2, 3)
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:, :2])) < 1e-12
    ):
        raise ValueError("The recorded replay mask-to-tracking transform is invalid")
    mask = cv2.warpAffine(
        mask,
        affine,
        (geometry["width"], geometry["height"]),
        flags=cv2.INTER_NEAREST,
    )
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("The recorded replay rink mask is empty in tracking coordinates")
    encoded, png = cv2.imencode(".png", (mask != 0).astype(np.uint8) * 255)
    if not encoded:
        raise ValueError("The recorded replay rink mask cannot be written")
    atomic_bytes(candidate / "rink_mask_0.png", png.tobytes())
    set_path(config, "rink.ice_contours_mask_count", 1)
    set_path(config, "rink.ice_contours_mask_centroid", [float(xs.mean()), float(ys.mean())])
    set_path(
        config,
        "rink.ice_contours_combined_bbox",
        [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
    )
    # The selected video already occupies the tracking raster. The non-stitching
    # reader loads its frozen bare mask; it must not regenerate a timed sample.
    set_path(config, "rink.ice_contours_geometry_revision", None)
    set_path(config, "stitching.rink_mask_frame_time", "auto")
