"""DriveGPT training directly from one or many SQLite recording databases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hmlib.camera.rink_context import mask_to_grid
from hmlib.telemetry.database import discover_runs, read_database


def discover_database_games(inputs):
    from hmlib.camera.camera_gpt_dataset import GameCsvPaths

    games, identities = [], {}
    for run in discover_runs(inputs):
        identities[run["run_id"]] = {
            "game_id": run["game_id"],
            "sha256": run["sha256"],
        }
        with read_database(run["database"]) as connection:
            geometries = connection.execute(
                "SELECT DISTINCT geometry_id FROM frames WHERE run_id=? ORDER BY geometry_id",
                (run["run_id"],),
            )
            for geometry in geometries:
                gid = geometry[0]
                games.append(
                    GameCsvPaths(
                        game_id=f"{run['game_id']}@{run['run_id']}/{gid}",
                        tracking_csv="",
                        camera_csv="",
                        database_path=run["database"],
                        run_id=run["run_id"],
                        geometry_id=gid,
                        source_game_id=run["game_id"],
                    )
                )
    return games, {"schema": "hockey-telemetry-selection-v1", "runs": identities}


def geometry(paths):
    with read_database(paths.database_path) as connection:
        row = connection.execute(
            "SELECT * FROM geometries WHERE run_id=? AND geometry_id=?",
            (paths.run_id, paths.geometry_id),
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing geometry for {paths.game_id}")
    if row["coordinate_space"] != "original_stitched_pixels":
        raise ValueError(f"Unsupported rink coordinates for {paths.game_id}")
    return row


def scan_database_max_xy(paths):
    row = geometry(paths)
    return float(row["width"]), float(row["height"])


def load_database_rink(paths, norm, height=32, width=64, *, rink_input="grid"):
    row = geometry(paths)
    if row["width"] > norm.scale_x + 1e-3 or row["height"] > norm.scale_y + 1e-3:
        raise ValueError(f"Recording canvas exceeds model normalization: {paths.game_id}")
    if row["mask_codec"] != "png" or not row["mask"]:
        raise ValueError(f"Recording has no archived rink mask: {paths.game_id}")
    if hashlib.sha256(row["mask"]).hexdigest() != row["mask_sha256"]:
        raise ValueError(f"Archived rink mask checksum mismatch: {paths.game_id}")
    mask = cv2.imdecode(np.frombuffer(row["mask"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask):
        raise ValueError(f"Invalid archived rink mask: {paths.game_id}")
    affine = np.asarray(json.loads(row["mask_to_tracking"]), dtype=np.float64)
    if (
        affine.shape != (2, 3)
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:, :2])) < 1e-12
    ):
        raise ValueError(f"Invalid rink-to-tracking transform: {paths.game_id}")
    if rink_input == "grid":
        result = mask_to_grid(
            mask,
            norm,
            height,
            width,
            mask_to_tracking=affine,
            frame_size=(row["width"], row["height"]),
        )
        if not np.any(result):
            raise ValueError(f"Rink mask has no visible occupancy: {paths.game_id}")
        return result
    if not np.array_equal(affine, np.eye(3)[:2]):
        raise ValueError("Legacy rink features require an identity mask transform; use grid input")
    ys, xs = np.nonzero(mask)
    return np.asarray(
        [
            xs.min() / norm.scale_x,
            ys.min() / norm.scale_y,
            xs.max() / norm.scale_x,
            ys.max() / norm.scale_y,
            xs.mean() / norm.scale_x,
            ys.mean() / norm.scale_y,
            len(xs) / mask.size,
        ],
        dtype=np.float32,
    )


def load_database_frames(paths):
    """Return ordered training rows and discontinuities, including empty-player frames."""
    with read_database(paths.database_path) as connection:
        run = connection.execute(
            "SELECT completed FROM runs WHERE run_id=?", (paths.run_id,)
        ).fetchone()
        if run is None or not run[0]:
            raise ValueError(f"Recording is no longer complete: {paths.run_id}")
        params = (paths.run_id, paths.geometry_id)
        frames = connection.execute(
            "SELECT sample_id,source_id,seek_epoch,reset_epoch,pts_ns FROM frames "
            "WHERE run_id=? AND geometry_id=? ORDER BY sample_id",
            params,
        ).fetchall()
        boundaries = set()
        previous = None
        for frame in frames:
            if previous is not None and (
                tuple(frame[1:4]) != tuple(previous[1:4])
                or frame[4] is None
                or previous[4] is None
                or frame[4] <= previous[4]
            ):
                boundaries.add(frame[0])
            previous = frame
        # Events may lie in an intentional sample-ID gap. Apply each to the first
        # sample at or after its effective boundary.
        events = connection.execute(
            "SELECT sample_boundary FROM config_events WHERE run_id=? ORDER BY sample_boundary",
            (paths.run_id,),
        ).fetchall()
        ids = np.asarray([frame[0] for frame in frames], dtype=np.int64)
        for event in events:
            index = np.searchsorted(ids, event[0])
            if index < len(ids):
                boundaries.add(int(ids[index]))
        tracks = pd.read_sql_query(
            "SELECT t.sample_id AS Frame,t.tracking_id AS ID,t.left AS BBox_X,t.top AS BBox_Y,"
            "t.width AS BBox_W,t.height AS BBox_H,t.score AS Scores,t.class_id AS Labels "
            "FROM tracks t JOIN frames f USING(run_id,sample_id) "
            "WHERE t.run_id=? AND f.geometry_id=? ORDER BY t.sample_id,t.ordinal",
            connection,
            params=params,
        )
        cameras = []
        for role in ("program", "fast"):
            cameras.append(
                pd.read_sql_query(
                    "SELECT c.sample_id AS Frame,c.left AS BBox_X,c.top AS BBox_Y,c.width AS BBox_W,c.height AS BBox_H "
                    "FROM cameras c JOIN frames f USING(run_id,sample_id) "
                    "WHERE c.run_id=? AND f.geometry_id=? AND c.role=? ORDER BY c.sample_id",
                    connection,
                    params=(*params, role),
                )
            )
    return tracks, cameras[0], cameras[1], set(ids.tolist()), boundaries


def split_database_games(games, fraction, seed, validation_games=None):
    """Keep every processing run/revision of the same source game on one side."""
    import random

    ids = sorted({game.source_game_id for game in games})
    if validation_games is None:
        if not 0 <= fraction < 1:
            raise ValueError("Validation fraction must be in [0, 1)")
        random.Random(seed).shuffle(ids)
        count = max(1, round(len(ids) * fraction)) if fraction else 0
        validation_games = set(ids[:count])
    else:
        validation_games = set(validation_games)
        if validation_games - set(ids):
            raise ValueError("Unknown validation game IDs")
    train = [game for game in games if game.source_game_id not in validation_games]
    val = [game for game in games if game.source_game_id in validation_games]
    if not train:
        raise ValueError("Database split contains no training games")
    return train, val


def database_config_split(
    config, config_path, root_override, min_train_frames, min_val_frames, require_rink_grid
):
    import fnmatch

    from hmlib.camera.camera_gpt_dataset import _contiguous_frame_runs

    if set(config) - {
        "schema",
        "root",
        "databases",
        "catalog",
        "recordings",
        "include",
        "exclude",
        "split",
    }:
        raise ValueError("Unknown database dataset configuration keys")
    root = Path(root_override or config.get("root", ".")).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    inputs = config.get("databases")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(isinstance(value, str) for value in inputs)
    ):
        raise ValueError("Dataset databases must be a nonempty list of paths/globs")
    games, identity = discover_database_games([str(root / value) for value in inputs])
    if "catalog" in config:
        catalog = json.loads((root / config["catalog"]).read_text())
        if catalog.get("schema") != "hockey-drivegpt-catalog-v2":
            raise ValueError("Unsupported database publication catalog")
        expected = {
            run["run_id"]: {"game_id": run["game_id"], "sha256": run["sha256"]}
            for run in catalog["runs"]
        }
        if expected != identity["runs"]:
            raise ValueError("Database content differs from the published catalog")
    if "recordings" in config:
        selected = config["recordings"]
        if (
            not isinstance(selected, list)
            or not selected
            or any(
                not isinstance(item, dict)
                or set(item) != {"run_id", "geometry_id"}
                or not isinstance(item["run_id"], str)
                or type(item["geometry_id"]) is not int
                or item["geometry_id"] <= 0
                for item in selected
            )
        ):
            raise ValueError("Dataset recordings must list run_id/geometry_id selections")
        keys = {(item["run_id"], item["geometry_id"]) for item in selected}
        available = {(game.run_id, game.geometry_id) for game in games}
        if len(keys) != len(selected) or keys - available:
            raise ValueError("Unknown or repeated recording geometry selection")
        games = [game for game in games if (game.run_id, game.geometry_id) in keys]
    include, exclude = config.get("include", ["*"]), config.get("exclude", [])
    if not all(
        isinstance(patterns, list) and all(isinstance(p, str) for p in patterns)
        for patterns in (include, exclude)
    ):
        raise ValueError("Dataset include/exclude must contain game ID patterns")
    games = [
        game
        for game in games
        if any(fnmatch.fnmatchcase(game.source_game_id, p) for p in include)
        and not any(fnmatch.fnmatchcase(game.source_game_id, p) for p in exclude)
    ]
    split = config.get("split", {})
    if not isinstance(split, dict) or set(split) - {
        "seed",
        "validation_fraction",
        "validation_games",
    }:
        raise ValueError("Invalid database dataset split options")
    train, val = split_database_games(
        games,
        float(split.get("validation_fraction", 0.1)),
        int(split.get("seed", 0)),
        split.get("validation_games"),
    )
    for selection, required in ((train, min_train_frames), (val, min_val_frames)):
        for game in selection:
            _, slow, fast, ids, boundaries = load_database_frames(game)
            aligned = sorted(ids & set(slow.Frame) & set(fast.Frame))
            longest = max(
                (len(run) for run in _contiguous_frame_runs(aligned, boundaries)), default=0
            )
            if longest < required:
                raise ValueError(f"{game.game_id} has no contiguous passage of {required} frames")
            if require_rink_grid and not geometry(game)["mask"]:
                raise ValueError(f"{game.game_id} has no archived rink mask")
    selected_runs = {game.run_id for game in games}
    identity["runs"] = {
        key: value for key, value in identity["runs"].items() if key in selected_runs
    }
    identity["train_games"] = [game.game_id for game in train]
    identity["validation_games"] = [game.game_id for game in val]
    return train, val, identity


def publish_database_dataset(inputs, destination, min_frames=32):
    """Publish closed SQLite snapshots, preserving source database generation names."""
    import os
    import shutil
    import sqlite3
    import tempfile

    import yaml

    from hmlib.camera.camera_gpt_dataset import _contiguous_frame_runs
    from hmlib.telemetry.database import database_files, discover_runs

    files = database_files(inputs)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        originals = discover_runs(files)
        copies = []
        for index, source in enumerate(files):
            copied = stage / "databases" / str(index) / source.name
            copied.parent.mkdir(parents=True)
            with read_database(source) as connection:
                target = sqlite3.connect(copied)
                try:
                    connection.backup(target)
                finally:
                    target.close()
            copies.append(copied)
        copied_runs = discover_runs(copies)
        if {run["run_id"]: run["sha256"] for run in originals} != {
            run["run_id"]: run["sha256"] for run in copied_runs
        }:
            raise ValueError("Recording content changed while publishing dataset")
        games, _ = discover_database_games(copies)
        coverage = []
        for game in games:
            _, slow, fast, ids, boundaries = load_database_frames(game)
            aligned = sorted(ids & set(slow.Frame) & set(fast.Frame))
            longest = max(
                (len(run) for run in _contiguous_frame_runs(aligned, boundaries)), default=0
            )
            coverage.append(
                {
                    "game_id": game.source_game_id,
                    "run_id": game.run_id,
                    "geometry_id": game.geometry_id,
                    "longest_run": longest,
                }
            )
        usable = [game for game in coverage if game["longest_run"] >= min_frames]
        if not usable:
            raise ValueError("No recording has a sufficiently long contiguous training passage")
        catalog = {
            "schema": "hockey-drivegpt-catalog-v2",
            "games": usable,
            "rejected": [
                {**game, "reason": f"No contiguous passage of {min_frames} frames"}
                for game in coverage
                if game not in usable
            ],
            "runs": [
                {**run, "database": str(Path(run["database"]).relative_to(stage))}
                for run in copied_runs
            ],
        }
        # Configuration payloads belong inside the database, not the JSON catalog.
        for run in catalog["runs"]:
            run.pop("source_config")
            run.pop("effective_config")
        (stage / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        config = {
            "schema": "hockey-drivegpt-dataset-v2",
            "root": ".",
            "databases": [str(path.relative_to(stage)) for path in copies],
            "catalog": "catalog.json",
            "recordings": [
                {"run_id": game["run_id"], "geometry_id": game["geometry_id"]} for game in usable
            ],
            "include": ["*"],
            "exclude": [],
            "split": {
                "seed": 0,
                "validation_fraction": 0.1 if len({game["game_id"] for game in usable}) > 1 else 0,
            },
        }
        (stage / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (stage / "README.md").write_text(
            "# DriveGPT recording databases\n\nTrain using `--dataset-config=dataset.yaml`. "
            "Each database includes its masks, configuration, detections, tracks and camera outputs. "
            "Repeated recording GUIDs are deduplicated; all runs of a source game share a training/validation split.\n"
        )
        os.replace(stage, destination)
        return catalog
    except BaseException:
        shutil.rmtree(stage)
        raise


def has_database_passage(paths, minimum, target_mode):
    """Check frame/camera indexes without loading tracks or mask BLOBs."""
    from bisect import bisect_right

    with read_database(paths.database_path) as connection:
        run = connection.execute(
            "SELECT completed FROM runs WHERE run_id=?", (paths.run_id,)
        ).fetchone()
        if run is None or not run[0]:
            raise ValueError(f"Recording is incomplete: {paths.run_id}")
        boundaries = [
            row[0]
            for row in connection.execute(
                "SELECT sample_boundary FROM config_events WHERE run_id=? ORDER BY sample_boundary",
                (paths.run_id,),
            )
        ]
        sql = (
            "SELECT sample_id,source_id,seek_epoch,reset_epoch,pts_ns FROM frames f "
            "WHERE run_id=? AND geometry_id=? AND EXISTS "
            "(SELECT 1 FROM cameras c WHERE c.run_id=f.run_id AND c.sample_id=f.sample_id AND role='program') "
        )
        if target_mode == "slow_fast_tlwh":
            sql += "AND EXISTS (SELECT 1 FROM cameras c WHERE c.run_id=f.run_id AND c.sample_id=f.sample_id AND role='fast') "
        previous, length = None, 0
        for row in connection.execute(
            sql + "ORDER BY sample_id", (paths.run_id, paths.geometry_id)
        ):
            continuous = (
                previous is not None
                and row[0] == previous[0] + 1
                and tuple(row[1:4]) == tuple(previous[1:4])
                and row[4] is not None
                and previous[4] is not None
                and row[4] > previous[4]
                and bisect_right(boundaries, row[0]) == bisect_right(boundaries, previous[0])
            )
            length = length + 1 if continuous else 1
            if length >= minimum:
                return True
            previous = row
    return False


def usable_database_games(games, minimum, target_mode):
    import logging

    usable = []
    for game in games:
        if not game.database_path or has_database_passage(game, minimum, target_mode):
            usable.append(game)
        else:
            logging.getLogger(__name__).warning(
                "Excluding %s: no contiguous passage of %d frames", game.game_id, minimum
            )
    return usable
