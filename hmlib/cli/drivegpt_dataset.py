"""Publish one validated CSV generation per game, with a provenance catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hmlib.camera.camera_gpt_dataset import (
    _contiguous_frame_runs,
    _hstream_manifest_allows_generation,
)
from hmlib.camera.camera_policy import camera_policy_path, read_camera_policy_boundaries
from hmlib.camera.rink_context import read_rink_context, rink_context_path

CATALOG_SCHEMA = "hockey-drivegpt-catalog-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path: Path, *, tracking: bool) -> tuple[dict, set[int]]:
    """Validate the numeric frame/box columns without loading object metadata."""
    columns = [0, 2, 3, 4, 5] if tracking else list(range(5))
    frames: set[int] = set()
    rows = 0
    max_xy = np.zeros(2)
    for chunk in pd.read_csv(path, header=None, usecols=columns, chunksize=250_000):
        values = chunk.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite frame/box values in {path}")
        ids = values[:, 0]
        if (ids < 0).any() or (ids != np.floor(ids)).any():
            raise ValueError(f"Invalid frame IDs in {path}")
        if (values[:, 3:5] <= 0).any():
            raise ValueError(f"Nonpositive box dimensions in {path}")
        frames.update(ids.astype(np.int64).tolist())
        rows += len(values)
        max_xy = np.maximum(max_xy, (values[:, 1:3] + values[:, 3:5]).max(axis=0))
    if not frames:
        raise ValueError(f"Empty CSV: {path}")
    if not tracking and len(frames) != rows:
        raise ValueError(f"Duplicate camera frame IDs in {path}")
    return {
        "rows": rows,
        "frames": len(frames),
        "first_frame": min(frames),
        "last_frame": max(frames),
        "max_xy": max_xy.tolist(),
    }, frames


def inspect_generation(directory: Path, suffix: str, min_frames: int) -> dict:
    part = f"-{suffix}" if suffix else ""
    paths = {
        role: directory / f"{role}{part}.csv" for role in ("tracking", "camera", "camera_fast")
    }
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise ValueError(f"Missing matched files: {', '.join(missing)}")
    if not _hstream_manifest_allows_generation(directory, part, *paths.values()):
        raise ValueError("Telemetry manifest marks this generation incomplete or ineligible")
    before = {role: (p.stat().st_size, p.stat().st_mtime_ns) for role, p in paths.items()}
    stats, frames = {}, {}
    for role, path in paths.items():
        stats[role], frames[role] = inspect_csv(path, tracking=role == "tracking")
    aligned = sorted(set.intersection(*frames.values()))
    boundaries = read_camera_policy_boundaries(paths["camera"], frames["camera"])
    boundaries.update(read_camera_policy_boundaries(paths["camera_fast"], frames["camera_fast"]))
    runs = _contiguous_frame_runs(aligned, boundaries)
    longest = max((len(run) for run in runs), default=0)
    if longest < min_frames:
        raise ValueError(f"Longest aligned contiguous run is {longest}; need {min_frames} frames")
    for role, path in paths.items():
        if before[role] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"Source changed during inspection: {path}")
    companions = [camera_policy_path(paths["camera"]), camera_policy_path(paths["camera_fast"])]
    companions += [
        directory / f"{name}{part}.{extension}"
        for name, extension in (
            ("rink_context", "json"),
            ("hstream_telemetry", "json"),
            ("hstream_frame_index", "csv"),
            ("hstream_config_events", "csv"),
        )
    ]
    companions += [
        directory / name
        for name in (
            "config.yaml",
            "play_tracker_source.yaml",
            "play_tracker_effective.yaml",
        )
    ]
    companions += sorted(directory.glob("rink_mask_*.png"))
    if rink_context_path(str(paths["tracking"])).is_file():
        context = read_rink_context(str(paths["tracking"]))
        companions += [directory / binding["file"] for binding in context["masks"]]
    companions = sorted(set(companions))
    return {
        "generation": suffix or "bare",
        "generation_number": int(suffix or 0),
        "source_files": {role: str(path) for role, path in paths.items()},
        "source_stats": {role: list(stat) for role, stat in before.items()},
        "companions": [str(p) for p in companions if p.is_file()],
        "csv_stats": stats,
        "aligned_frames": len(aligned),
        "alignment_fraction": len(aligned) / max(len(f) for f in frames.values()),
        "contiguous_runs": len(runs),
        "longest_run": longest,
        "modified_ns": max(stat[1] for stat in before.values()),
    }


def choose_generation(directory: Path, min_frames: int = 32) -> tuple[dict | None, list[dict]]:
    candidates, rejected = [], []
    for path in sorted(directory.glob("tracking*.csv")):
        match = re.fullmatch(r"tracking(?:-(\d+))?\.csv", path.name)
        if not match:
            continue
        suffix = match[1] or ""
        try:
            candidates.append(inspect_generation(directory, suffix, min_frames))
        except (ValueError, OSError, pd.errors.ParserError) as error:
            rejected.append({"generation": suffix or "bare", "reason": str(error)})
    if not candidates:
        return None, rejected
    candidates.sort(
        key=lambda c: (c["aligned_frames"], c["modified_ns"], c["generation_number"]), reverse=True
    )
    selected = candidates[0]
    for candidate in candidates[1:]:
        rejected.append(
            {
                "generation": candidate["generation"],
                "aligned_frames": candidate["aligned_frames"],
                "reason": "A more complete or equally complete newer generation was selected",
            }
        )
    return selected, rejected


def publish_dataset(source: Path, destination: Path, min_frames: int = 32) -> dict:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if any(source.rglob("*.db")) or any(source.rglob("*.sqlite")):
        from hmlib.camera.camera_database import publish_database_dataset

        return publish_database_dataset([source], destination, min_frames)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    catalog = {
        "schema": CATALOG_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "selection": "most aligned frames; newest file modification time breaks ties",
        "required_csvs": ["tracking", "camera", "camera_fast"],
        "pose_included": False,
        "min_contiguous_frames": min_frames,
        "games": [],
        "rejected": [],
    }
    identities: dict[str, str] = {}
    try:
        directories = []
        for parent, dirs, files in os.walk(source):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            if any(re.fullmatch(r"tracking(?:-\d+)?\.csv", f) for f in files):
                directories.append(Path(parent))
        for directory in sorted(directories):
            game_id = directory.relative_to(source).as_posix()
            selected, rejected = choose_generation(directory, min_frames)
            catalog["rejected"].extend({"game_id": game_id, **r} for r in rejected)
            if selected is None:
                print(f"Excluded {game_id}: no complete usable generation", flush=True)
                continue
            game_destination = stage / "games" / game_id
            game_destination.mkdir(parents=True)
            files = {}
            all_sources = dict(selected["source_files"])
            all_sources.update({Path(p).name: p for p in selected.pop("companions")})
            for role, source_file in all_sources.items():
                original = Path(source_file)
                before = original.stat()
                if role in selected["source_stats"] and selected["source_stats"][role] != [
                    before.st_size,
                    before.st_mtime_ns,
                ]:
                    raise RuntimeError(f"Source changed before copy: {original}")
                copied = game_destination / original.name
                shutil.copy2(original, copied)
                digest = sha256_file(copied)
                if digest != sha256_file(original):
                    raise RuntimeError(f"Copy verification failed: {original}")
                after = original.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed during copy: {original}")
                files[role] = {
                    "path": copied.relative_to(stage).as_posix(),
                    "source": str(original),
                    "bytes": before.st_size,
                    "modified_ns": before.st_mtime_ns,
                    "sha256": digest,
                }
            identity = files["tracking"]["sha256"]
            duplicate_of = identities.get(identity)
            identities.setdefault(identity, game_id)
            entry = {
                "game_id": game_id,
                "directory": game_destination.relative_to(stage).as_posix(),
                "source_directory": str(directory),
                "duplicate_of": duplicate_of,
                **{k: v for k, v in selected.items() if k not in {"source_files", "source_stats"}},
                "files": files,
            }
            catalog["games"].append(entry)
            (game_destination / "provenance.json").write_text(json.dumps(entry, indent=2) + "\n")
            print(
                f"Included {game_id}: generation={entry['generation']} aligned={entry['aligned_frames']:,} duplicate_of={duplicate_of}",
                flush=True,
            )
        if not catalog["games"]:
            raise ValueError("No complete usable games found")
        catalog["total_aligned_frames"] = sum(g["aligned_frames"] for g in catalog["games"])
        (stage / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        dataset_config = {
            "schema": "hockey-drivegpt-dataset-v1",
            "root": ".",
            "catalog": "catalog.json",
            "include": ["*"],
            "exclude": [g["game_id"] for g in catalog["games"] if g["duplicate_of"]],
            "split": {"seed": 0, "validation_fraction": 0.1},
        }
        (stage / "dataset.yaml").write_text(yaml.safe_dump(dataset_config, sort_keys=False))
        (stage / "games.lst").write_text(
            "".join(g["directory"] + "\n" for g in catalog["games"] if not g["duplicate_of"])
        )
        (stage / "README.md").write_text(
            "# HockeyDriveGPT\n\nOne complete tracking/slow/fast CSV generation per source game directory. "
            "Files are verified copies; pose data is omitted. Game paths preserve their path beneath "
            f"`{source}`.\n\n`catalog.json` records hashes, source paths, frame coverage, selected "
            "generations, duplicate tracking identities, and rejected candidates. Each game also has "
            "`provenance.json`. `dataset.yaml` controls inclusion and the deterministic game-level split. "
            "Exact duplicate tracking exports are retained for provenance and excluded from training.\n"
        )
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        action="append",
        default=[],
        help="Database file/directory/glob; repeat to combine inputs",
    )
    parser.add_argument("--source", type=Path, default=Path.home() / "Videos")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=32)
    args = parser.parse_args()
    if args.min_frames < 2:
        parser.error("--min-frames must be at least 2")
    if args.database:
        from hmlib.camera.camera_database import publish_database_dataset

        catalog = publish_database_dataset(args.database, args.out, args.min_frames)
    else:
        catalog = publish_dataset(args.source, args.out, args.min_frames)
    print(f"Published {len(catalog['games'])} games to {args.out}")


if __name__ == "__main__":
    main()
