"""Versioned SQLite telemetry, stable run identity, and transactional merging.

Connections are opened inside workers, never pickled or shared across processes.
All payload tables are namespaced by run_id; filenames have no identity semantics.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

APPLICATION_ID = 1213027156
SCHEMA_VERSION = 1
TABLES = (
    "runs",
    "geometries",
    "rink_inputs",
    "frames",
    "detections",
    "tracks",
    "cameras",
    "replay_frames",
    "replay_tracks",
    "checkpoints",
    "config_events",
)


def validate_schema(connection: sqlite3.Connection) -> None:
    if (
        connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
        or connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported hockey telemetry database schema")
    actual = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if actual != set(TABLES):
        raise ValueError(f"Unexpected telemetry tables: {actual.symmetric_difference(TABLES)}")


@contextmanager
def read_database(path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(path).expanduser().resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        validate_schema(connection)
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def create_database(path: str | Path) -> sqlite3.Connection:
    """Create exclusively; callers own the returned connection."""
    path = Path(path)
    with path.open("xb"):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.executescript(Path(__file__).with_name("schema.sql").read_text())
        return connection
    except BaseException:
        connection.close()
        raise


def primary_key(connection: sqlite3.Connection, table: str) -> list[str]:
    if table not in TABLES:
        raise ValueError(f"Unknown telemetry table: {table}")
    columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]


def run_fingerprint(connection: sqlite3.Connection, run_id: str) -> str:
    """Hash logical content in stable key order, independent of database packing."""
    digest = hashlib.sha256()
    for table in TABLES:
        digest.update(table.encode() + b"\0")
        order = ",".join(primary_key(connection, table))
        for row in connection.execute(
            f"SELECT * FROM {table} WHERE run_id=? ORDER BY {order}", (run_id,)
        ):
            # BLOB hashes avoid constructing huge JSON strings for full-resolution masks.
            values = [
                (
                    {"blob_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                    if isinstance(value, bytes)
                    else value
                )
                for value in row
            ]
            digest.update(json.dumps(values, separators=(",", ":"), allow_nan=False).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def database_files(inputs: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        expanded = Path(item).expanduser()
        matches = sorted(glob.glob(str(expanded)))
        if not matches:
            raise FileNotFoundError(f"No database input matches {item}")
        for match in matches:
            path = Path(match)
            if path.is_dir():
                found = list(path.rglob("*.db")) + list(path.rglob("*.sqlite"))
                if not found:
                    raise ValueError(f"No databases in {path}")
                files.update(p.resolve() for p in found)
            elif path.is_file():
                files.add(path.resolve())
            else:
                raise ValueError(f"Not a database file: {path}")
    if not files:
        raise ValueError("No database inputs supplied")
    return sorted(files)


def completed_runs(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = connection.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
    if not rows:
        raise ValueError("Telemetry database contains no recordings")
    for row in rows:
        if not row["completed"] or row["outcome"] not in {"end-of-stream", "intentional-stop"}:
            raise ValueError(f"Recording {row['run_id']} is incomplete or failed")
        count = connection.execute(
            "SELECT count(*) FROM frames WHERE run_id=?", (row["run_id"],)
        ).fetchone()[0]
        if count <= 0 or count != row["sample_count"]:
            raise ValueError(f"Recording {row['run_id']} sample count is inconsistent")
    return rows


def discover_runs(inputs: Iterable[str | Path]) -> list[dict]:
    """Deduplicate identical GUIDs and reject conflicting copies."""
    runs: dict[str, dict] = {}
    for path in database_files(inputs):
        with read_database(path) as connection:
            for row in completed_runs(connection):
                run_id = row["run_id"]
                fingerprint = run_fingerprint(connection, run_id)
                if run_id in runs:
                    if runs[run_id]["sha256"] != fingerprint:
                        raise ValueError(f"Conflicting content for recording GUID {run_id}: {path}")
                    continue
                runs[run_id] = {**dict(row), "database": str(path), "sha256": fingerprint}
    return list(runs.values())


def merge_databases(destination: str | Path, inputs: Iterable[str | Path]) -> dict:
    """Merge all inputs atomically; never overwrite differing content under an existing GUID."""
    paths = database_files(inputs)
    destination = Path(destination).expanduser().resolve()
    if destination in paths:
        raise ValueError("Merge destination must not also be an input")
    output = sqlite3.connect(destination) if destination.exists() else create_database(destination)
    output.row_factory = sqlite3.Row
    added, skipped = [], []
    try:
        validate_schema(output)
        output.execute("PRAGMA foreign_keys=ON")
        output.execute("BEGIN IMMEDIATE")
        for path in paths:
            with read_database(path) as source:
                if source.execute("PRAGMA foreign_key_check").fetchone():
                    raise ValueError(f"Broken telemetry references in {path}")
                for run in completed_runs(source):
                    run_id = run["run_id"]
                    existing = output.execute(
                        "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
                    ).fetchone()
                    if existing:
                        if run_fingerprint(output, run_id) != run_fingerprint(source, run_id):
                            raise ValueError(f"Conflicting content for recording GUID {run_id}")
                        skipped.append(run_id)
                        continue
                    for table in TABLES:
                        rows = source.execute(f"SELECT * FROM {table} WHERE run_id=?", (run_id,))
                        placeholders = ",".join("?" for _ in rows.description)
                        while batch := rows.fetchmany(2048):
                            output.executemany(f"INSERT INTO {table} VALUES({placeholders})", batch)
                    added.append(run_id)
        if output.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Merge produced invalid references")
        output.commit()
    except BaseException:
        output.rollback()
        raise
    finally:
        output.close()
    return {"database": str(destination), "added": added, "already_present": skipped}
