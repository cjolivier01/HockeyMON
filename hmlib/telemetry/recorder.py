"""Lossless HM recording on a bounded metadata writer queue.

Only small observation arrays cross from CUDA to CPU at the existing save stages.
The immutable CPU calibration mask is shared until PNG encoding on the writer;
video surfaces are never read back or retained here.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from hmlib.telemetry.database import create_database, read_database


@dataclass(frozen=True)
class RecordingArtifacts:
    telemetry_path: Path | None
    supplementary_paths: tuple[Path, ...] = ()


def configuration_snapshot(value, active=None, path="root"):
    """Archive resolved settings without traversing runtime models or video tensors."""
    import math

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    active = {} if active is None else active
    if id(value) in active:
        return {"reference": active[id(value)]}
    if isinstance(value, (dict, list, tuple)):
        active[id(value)] = path
        try:
            if isinstance(value, dict):
                return {
                    str(k): configuration_snapshot(v, active, f"{path}.{k}")
                    for k, v in value.items()
                }
            return [configuration_snapshot(v, active, f"{path}[{i}]") for i, v in enumerate(value)]
        finally:
            del active[id(value)]
    return {"runtime_type": f"{type(value).__module__}.{type(value).__name__}"}


class TelemetryRecorder:
    def __init__(self, directory, game_id, config, stages, *, capacity=64):
        if not game_id:
            raise ValueError("Telemetry requires a source game ID")
        if capacity < 1:
            raise ValueError("Telemetry queue capacity must be positive")
        self.stages = frozenset(stages)
        self._capacity = capacity
        self.run_id = str(uuid.uuid4())
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Reserve a fresh working generation; previous runs remain recoverable.
        index = 0
        while True:
            self.path = directory / f"hm_telemetry{'-' + str(index) if index else ''}.db"
            try:
                connection = create_database(self.path)
                break
            except FileExistsError:
                index += 1
        configuration = yaml.safe_dump(configuration_snapshot(config), sort_keys=True)
        try:
            connection.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?,0,'incomplete',0)",
                (
                    self.run_id,
                    str(game_id),
                    datetime.now(timezone.utc).isoformat(),
                    "hm",
                    configuration,
                    configuration,
                ),
            )
            connection.execute(
                "INSERT INTO config_events VALUES(?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    1,
                    1,
                    "run-configuration",
                    "startup",
                    "",
                    "run-config.yaml",
                    configuration,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        self._queue = queue.Queue(maxsize=capacity)
        self._error = None
        self._closed = False
        self._batches = 0
        self._thread = threading.Thread(target=self._write, name="hm-telemetry", daemon=True)
        self._thread.start()

    def new_batch(self):
        from hmlib.telemetry.capture import TelemetryBatch

        self.check()
        if self._closed:
            raise RuntimeError("Telemetry recorder is closed")
        batch = TelemetryBatch(self, self._batches)
        self._batches += 1
        return batch

    def check(self):
        if self._error is not None:
            raise RuntimeError(f"Telemetry writer failed: {self._path_error()}") from self._error

    def _path_error(self):
        return f"{self.path}: {self._error}"

    def submit(self, batch):
        while True:
            self.check()
            try:
                self._queue.put(batch, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self):
        """Drain and close, leaving completion to the outer pipeline owner."""
        if not self._closed:
            self._closed = True
            try:
                self.submit(None)
            finally:
                self._thread.join()
        self.check()

    def _write(self):
        import sqlite3

        connection = None
        try:
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            self._sample_id = self._geometry_id = 0
            self._event_id = 1
            self._geometry_key = None
            self._mask_reference = None
            self._last_frame = None
            self._seek_epoch = 0
            pending = {}
            expected = 0
            while True:
                item = self._queue.get()
                if item is None:
                    if pending or expected != self._batches:
                        raise ValueError("Telemetry is missing one or more capture stages/batches")
                    break
                if item.index < expected or item.index in pending:
                    raise ValueError("Duplicate telemetry batch")
                pending[item.index] = item
                if len(pending) > self._capacity:
                    raise ValueError(
                        "Telemetry reorder capacity exceeded; a capture batch is stalled"
                    )
                # Aspen can finish different branches/batches out of order.
                while expected in pending:
                    self._write_batch(connection, pending.pop(expected))
                    expected += 1
                    if expected % 120 == 0:
                        connection.commit()
            connection.execute(
                "UPDATE runs SET sample_count=? WHERE run_id=?", (self._sample_id, self.run_id)
            )
            connection.commit()
        except BaseException as error:
            self._error = error
        finally:
            if connection is not None:
                connection.close()

    def _write_batch(self, connection, batch):
        stages = batch.stages
        reference = stages.get("cameras") or stages.get("tracks") or stages["detections"]
        ids = reference["ids"]
        if any(stage["ids"] != ids for stage in stages.values()):
            raise ValueError("Telemetry stages disagree on source frame IDs")
        geometry = reference["geometry"]
        # Prefer a mask from a stage after rink configuration.
        for kind in ("tracks", "cameras"):
            if kind in stages and stages[kind]["geometry"][3] is not None:
                geometry = stages[kind]["geometry"]
        width, height, revision, mask = geometry
        if any(stage["geometry"][:2] != (width, height) for stage in stages.values()):
            raise ValueError("Telemetry stages disagree on canvas dimensions")
        pointer = None if mask is None else mask.__array_interface__["data"][0]
        key = (width, height, revision, pointer)
        if key != self._geometry_key:
            png = None
            if mask is not None:
                if mask.shape != (height, width):
                    raise ValueError("Rink mask dimensions differ from the telemetry canvas")
                ok, encoded = cv2.imencode(".png", (mask != 0).astype(np.uint8) * 255)
                if not ok:
                    raise ValueError("Cannot encode telemetry rink mask")
                png = encoded.tobytes()
            self._geometry_id += 1
            connection.execute(
                "INSERT INTO geometries VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    self._geometry_id,
                    width,
                    height,
                    "original_stitched_pixels",
                    revision,
                    "png" if png else None,
                    png,
                    hashlib.sha256(png).hexdigest() if png else None,
                    "[[1,0,0],[0,1,0]]",
                ),
            )
            self._geometry_key = key
            self._mask_reference = mask  # Keep pointer identity valid between revisions.
        events = stages.get("cameras", {}).get("events", [])
        event_frames = [event["frame"] for event in events]
        if len(set(event_frames)) != len(event_frames) or any(f not in ids for f in event_frames):
            raise ValueError("Camera policy events do not match the telemetry batch")
        for index, frame in enumerate(ids):
            self._sample_id += 1
            if self._last_frame is not None and frame != self._last_frame + 1:
                self._seek_epoch += 1
            self._last_frame = frame
            detections = stages.get("detections", {}).get("rows", [[] for _ in ids])[index]
            tracks = stages.get("tracks", {}).get("rows", [[] for _ in ids])[index]
            connection.execute(
                "INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    self._sample_id,
                    0,
                    frame,
                    0,
                    frame,
                    reference["pts"][index],
                    None,
                    self._seek_epoch,
                    0,
                    self._geometry_id,
                    len(detections),
                    len(tracks),
                ),
            )
            for table, rows in (("detections", detections), ("tracks", tracks)):
                for ordinal, row in enumerate(rows):
                    values = (self.run_id, self._sample_id, ordinal, *row)
                    connection.execute(
                        f"INSERT INTO {table} VALUES({','.join('?' for _ in values)})", values
                    )
            for role, boxes in stages.get("cameras", {}).get("boxes", {}).items():
                connection.execute(
                    "INSERT INTO cameras VALUES(?,?,?,?,?,?,?)",
                    (self.run_id, self._sample_id, role, *boxes[index]),
                )
            for event in events:
                if event["frame"] == frame:
                    self._event_id += 1
                    connection.execute(
                        "INSERT INTO config_events VALUES(?,?,?,?,?,?,?,?)",
                        (
                            self.run_id,
                            self._event_id,
                            self._sample_id,
                            "camera-policy",
                            event["kind"],
                            json.dumps(event["policy"], allow_nan=False),
                            "",
                            "",
                        ),
                    )


def complete_recording(path, *, outcome="end-of-stream"):
    """Only called after every pipeline resource has finalized successfully."""
    import sqlite3

    if outcome not in {"end-of-stream", "intentional-stop"}:
        raise ValueError("Invalid successful telemetry outcome")
    with read_database(path) as source:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Telemetry database failed integrity validation")
        if source.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Telemetry database contains broken references")
        count = source.execute("SELECT count(*) FROM frames").fetchone()[0]
        run = source.execute("SELECT sample_count,completed FROM runs").fetchone()
        if not count or count != run[0] or run[1]:
            raise ValueError("Telemetry recording is empty, incomplete, or already finalized")
    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE runs SET completed=1,outcome=?", (outcome,))
        connection.commit()
    finally:
        connection.close()
