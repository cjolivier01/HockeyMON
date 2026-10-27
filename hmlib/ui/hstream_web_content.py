"""CPU inspection of closed Program artifacts, isolated from the web database.

This file is mirrored by HockeyMON's hmlib/ui/hstream_web_content.py. Protocol
v1 describes actual file bytes; request bindings never stand in for media probes.
SQLite here is an untrusted recording format, not the service's Django database.
"""

from __future__ import annotations

import array
import bisect
import csv
import hashlib
import io
import json
import math
import os
import re
import resource
import selectors
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from functools import partial
from pathlib import Path
from uuid import UUID

FEATURE = "pinned-output-content-v1"
MANIFEST_NAME = "hockeymon-output-content.json"
MAX_DOCUMENT = 4 * 1024 * 1024
MAX_METADATA = 32 * 1024 * 1024
MAX_DATABASE = 2 * 1024**3
MAX_MEDIA = 64 * 1024**3
MAX_FILES = 32
MAX_ROWS = 20_000_000
MAX_DIMENSION = 32768
MAX_PIXELS = 256 * 1024 * 1024
TIMEOUT = 120
TELEMETRY_SCHEMA = """-- Hockey telemetry database v1. Keep identical to hmlib/telemetry/schema.sql.
PRAGMA application_id = 1213027156;
PRAGMA user_version = 1;
PRAGMA foreign_keys = ON;
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, started_utc TEXT NOT NULL,
  producer TEXT NOT NULL, source_config TEXT NOT NULL, effective_config TEXT NOT NULL,
  completed INTEGER NOT NULL DEFAULT 0 CHECK(completed IN (0,1)),
  outcome TEXT NOT NULL DEFAULT 'incomplete', sample_count INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;
CREATE TABLE geometries (
  run_id TEXT NOT NULL REFERENCES runs(run_id), geometry_id INTEGER NOT NULL,
  width INTEGER NOT NULL CHECK(width>0), height INTEGER NOT NULL CHECK(height>0),
  coordinate_space TEXT NOT NULL, revision TEXT NOT NULL, mask_codec TEXT,
  mask BLOB, mask_sha256 TEXT, mask_to_tracking TEXT NOT NULL,
  PRIMARY KEY(run_id,geometry_id)
) WITHOUT ROWID;
CREATE TABLE rink_inputs (
  run_id TEXT NOT NULL, geometry_id INTEGER NOT NULL, input_id TEXT NOT NULL,
  format TEXT NOT NULL, metadata_json TEXT NOT NULL, data BLOB NOT NULL,
  PRIMARY KEY(run_id,geometry_id,input_id),
  FOREIGN KEY(run_id,geometry_id) REFERENCES geometries(run_id,geometry_id)
) WITHOUT ROWID;
CREATE TABLE frames (
  run_id TEXT NOT NULL REFERENCES runs(run_id), sample_id INTEGER NOT NULL CHECK(sample_id>0),
  source_id INTEGER NOT NULL, source_frame INTEGER NOT NULL,
  decoded_source_id INTEGER, decoded_sequence INTEGER, pts_ns INTEGER, ntp_ns INTEGER,
  seek_epoch INTEGER NOT NULL, reset_epoch INTEGER NOT NULL,
  geometry_id INTEGER NOT NULL, detection_count INTEGER NOT NULL, track_count INTEGER NOT NULL,
  PRIMARY KEY(run_id,sample_id),
  FOREIGN KEY(run_id,geometry_id) REFERENCES geometries(run_id,geometry_id)
) WITHOUT ROWID;
CREATE INDEX frames_time ON frames(run_id,pts_ns,sample_id);
CREATE TABLE detections (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
  left REAL NOT NULL, top REAL NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
  score REAL NOT NULL, class_id INTEGER NOT NULL,
  PRIMARY KEY(run_id,sample_id,ordinal),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE tracks (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
  tracking_id TEXT NOT NULL, left REAL NOT NULL, top REAL NOT NULL,
  width REAL NOT NULL, height REAL NOT NULL, score REAL NOT NULL, class_id INTEGER NOT NULL,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(run_id,sample_id,ordinal),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE cameras (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, role TEXT NOT NULL,
  left REAL NOT NULL, top REAL NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
  PRIMARY KEY(run_id,sample_id,role),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE replay_frames (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL,
  arena_left REAL NOT NULL, arena_top REAL NOT NULL, arena_right REAL NOT NULL, arena_bottom REAL NOT NULL,
  stepped INTEGER NOT NULL, has_received_tracks INTEGER NOT NULL,
  edge_rotation_left REAL NOT NULL, edge_rotation_right REAL NOT NULL,
  PRIMARY KEY(run_id,sample_id),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE replay_tracks (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, tracking_id TEXT NOT NULL,
  left REAL NOT NULL, top REAL NOT NULL, right REAL NOT NULL, bottom REAL NOT NULL,
  PRIMARY KEY(run_id,sample_id,ordinal),
  FOREIGN KEY(run_id,sample_id) REFERENCES replay_frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE checkpoints (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, schema_version INTEGER NOT NULL,
  state TEXT NOT NULL, base_state TEXT NOT NULL,
  PRIMARY KEY(run_id,sample_id),
  FOREIGN KEY(run_id,sample_id) REFERENCES replay_frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE config_events (
  run_id TEXT NOT NULL REFERENCES runs(run_id), event_id INTEGER NOT NULL,
  sample_boundary INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
  artifact_name TEXT NOT NULL, artifact_contents TEXT NOT NULL,
  PRIMARY KEY(run_id,event_id)
) WITHOUT ROWID;
CREATE INDEX config_boundaries ON config_events(run_id,sample_boundary);
"""

TABLE_COLUMNS = {
    "runs": "run_id game_id started_utc producer source_config effective_config completed outcome sample_count",
    "geometries": "run_id geometry_id width height coordinate_space revision mask_codec mask mask_sha256 mask_to_tracking",
    "rink_inputs": "run_id geometry_id input_id format metadata_json data",
    "frames": "run_id sample_id source_id source_frame decoded_source_id decoded_sequence pts_ns ntp_ns seek_epoch reset_epoch geometry_id detection_count track_count",
    "detections": "run_id sample_id ordinal left top width height score class_id",
    "tracks": "run_id sample_id ordinal tracking_id left top width height score class_id attributes_json",
    "cameras": "run_id sample_id role left top width height",
    "replay_frames": "run_id sample_id arena_left arena_top arena_right arena_bottom stepped has_received_tracks edge_rotation_left edge_rotation_right",
    "replay_tracks": "run_id sample_id ordinal tracking_id left top right bottom",
    "checkpoints": "run_id sample_id schema_version state base_state",
    "config_events": "run_id event_id sample_boundary kind key value artifact_name artifact_contents",
}


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate Program JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite Program JSON value: " + value)

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    # Also rejects finite-looking exponents that overflow Python's float.
    canonical(value)
    return value


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def binding(request):
    return {
        key: request[key]
        for key in (
            "run_id",
            "request_digest",
            "deployment_id",
            "runtime",
            "sources",
            "calibration",
            "launch",
        )
    }


def manifest_filename(version):
    return f"hockeymon-output-content-{version}.json"


def run_probe(request, entries, pins, *, configuration):
    """Inspect only inherited pinned descriptors, with a contained CPU child."""
    if not 1 <= len(entries) <= MAX_FILES:
        raise ValueError("Program output count exceeds the content validation limit")
    fds = {entry["path"]: pins[entry["path"]].fd for entry in entries}
    spec = canonical(
        {
            "request": request,
            "entries": entries,
            "fds": fds,
            "configuration_fd": configuration.fd,
        }
    )
    if len(spec) > MAX_DOCUMENT:
        raise ValueError("Program content request exceeds its byte limit")
    with tempfile.TemporaryDirectory(prefix="hm-program-content-", dir="/tmp") as temporary:
        process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(*fds.values(), configuration.fd),
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TMPDIR": temporary},
        )
        try:
            stdout, stderr = process.communicate(spec, timeout=TIMEOUT)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            # Includes descendants left behind by a failed external probe.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode or len(stdout) > MAX_DOCUMENT:
            raise ValueError(
                "Program content validation failed: " + stderr.decode(errors="replace")[:512]
            )
        return strict_json(stdout)


def read_fd(fd, maximum):
    size = os.fstat(fd).st_size
    if not 0 <= size <= maximum:
        raise ValueError("Program artifact exceeds its content byte limit")
    data = bytearray()
    while len(data) < size:
        chunk = os.pread(fd, min(1024 * 1024, size - len(data)), len(data))
        if not chunk:
            raise ValueError("Program artifact was truncated")
        data.extend(chunk)
    return bytes(data)


def content(fd, maximum):
    size = os.fstat(fd).st_size
    if not 0 <= size <= maximum:
        raise ValueError("Program artifact exceeds its content byte limit")
    result = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise ValueError("Program artifact was truncated")
        result.update(chunk)
        offset += len(chunk)
    return {"size": size, "sha256": result.hexdigest()}


def number(value, *, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError("Invalid Program numeric metadata")
    return result


def dimensions(width, height):
    if (
        type(width) is not int
        or type(height) is not int
        or not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION)
        or width * height > MAX_PIXELS
    ):
        raise ValueError("Program geometry exceeds the content limit")


def tool(arguments, fd):
    """Bound diagnostics and retire the actual probe, even for malformed media."""
    process = subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(fd,),
    )
    output = {process.stdout: bytearray(), process.stderr: bytearray()}
    deadline = time.monotonic() + 45
    try:
        with selectors.DefaultSelector() as selector:
            for stream in output:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() > deadline:
                    raise ValueError("Program media probe exceeded its deadline")
                for key, _ in selector.select(0.1):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        output[key.fileobj].extend(chunk)
                        if len(output[key.fileobj]) > 128 * 1024:
                            raise ValueError("Program media probe exceeded its output limit")
        if process.wait(timeout=1):
            raise ValueError("Program media could not be probed/decoded")
        return bytes(output[process.stdout])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def media(fd, request, recipe, kind):
    source = f"/proc/self/fd/{fd}"
    common = [
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        "mov,matroska,webm",
        "-probesize",
        "16777216",
        "-analyzeduration",
        "5000000",
    ]
    document = strict_json(
        tool(
            [
                "/usr/bin/ffprobe",
                *common,
                "-count_packets",
                "-show_entries",
                "stream=index,codec_type,codec_name,width,height,avg_frame_rate,start_time,duration,nb_read_packets:format=format_name,duration,size",
                "-of",
                "json",
                source,
            ],
            fd,
        )
    )
    streams = document.get("streams", [])
    videos = [row for row in streams if row.get("codec_type") == "video"]
    if (
        len(videos) != 1
        or len(streams) > 2
        or any(row.get("codec_type") not in {"video", "audio"} for row in streams)
    ):
        raise ValueError("Program archive requires one video and at most one audio stream")
    video = videos[0]
    if video.get("codec_name") not in {"h264", "hevc", "mpeg4"}:
        raise ValueError("Unsupported Program archive video codec")
    dimensions(video.get("width"), video.get("height"))
    # ApplyCameraPlugin produces the fixed UHD canvas and its archive writer
    # applies no resize. This logical role cannot adopt an arbitrary Program.
    if kind == "program-4k" and (video["width"], video["height"]) != (3840, 2160):
        raise ValueError("Program 4K archive must have the producer's 3840x2160 canvas")
    fps = Fraction(video["avg_frame_rate"])
    if not 0 < fps <= 240 or int(video["nb_read_packets"]) < 1:
        raise ValueError("Program archive has no valid packet timeline")
    duration = number(document["format"]["duration"], positive=True)
    if abs(int(video["nb_read_packets"]) / float(fps) - duration) > max(0.25, 2 / float(fps)):
        raise ValueError("Program archive packet count disagrees with its timeline")
    if duration > 100_000_000 or abs(float(video.get("start_time", 0))) > 1:
        raise ValueError("Invalid Program archive timeline")
    limit = request["launch"]["duration_seconds"]
    if limit is not None and duration > limit + max(0.25, 2 / float(fps)):
        raise ValueError("Program archive exceeds its admitted playback duration")
    if int(document["format"]["size"]) != os.fstat(fd).st_size:
        raise ValueError("Program media size does not match its pinned file")
    configured_fps = recipe.get("video_out", {}).get("output_fps")
    if configured_fps is None:
        configured_fps = recipe.get("camera", {}).get("output-fps")
    if (
        configured_fps not in (None, 0)
        and abs(float(fps) - number(configured_fps, positive=True)) > 0.001
    ):
        raise ValueError("Program archive cadence differs from admitted configuration")
    # Final Program scaling is explicit in video_out. Auto/aspect-derived and
    # stitched encoder-limited geometry cannot be inferred from this recipe.
    if kind == "program":
        settings = recipe.get("video_out", {})
        for key, actual in (
            ("output_width", video["width"]),
            ("output_height", video["height"]),
        ):
            expected = settings.get(key)
            if expected is not None and str(expected).strip().lower() not in {
                "",
                "auto",
            }:
                aliases = {
                    "output_width": {"4k": 4096, "8k": 8192},
                    "output_height": {"4k": 2160, "8k": 4320},
                }
                alias = aliases[key].get(str(expected).strip().lower())
                if alias is None:
                    expected = int(number(expected, positive=True))
                else:
                    expected = alias
                expected += expected % 2  # PyVideoOutput's YUV420 canvas rounding.
                if actual != expected:
                    raise ValueError("Program archive geometry differs from admitted configuration")
    for row in streams:
        if row["codec_type"] == "audio" and row.get("codec_name") not in {
            "aac",
            "opus",
        }:
            raise ValueError("Unsupported Program archive audio codec")
    tool(
        [
            "/usr/bin/ffmpeg",
            *common,
            "-xerror",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        fd,
    )
    return {"schema": "archive-v1", "streams": streams, "format": document["format"]}


def png(data):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode not in {"1", "L", "P", "RGB", "RGBA"}:
            raise ValueError("Unsupported Program mask format")
        dimensions(*image.size)
        image.load()
        return {"schema": "mask-png-v1", "width": image.width, "height": image.height}


def telemetry(fd, request, context):
    # SQLite resolves /proc/self/fd symlinks internally. Copy the pinned bytes
    # into private bounded storage rather than reopening the tenant pathname.
    if os.pread(fd, 20, 0)[:16] != b"SQLite format 3\0" or os.pread(fd, 2, 18) != b"\x01\x01":
        raise ValueError("Telemetry must be a closed rollback-journal SQLite database")
    with tempfile.TemporaryDirectory(prefix="telemetry-") as directory:
        path = Path(directory) / "recording.db"
        with path.open("xb") as output:
            size = os.fstat(fd).st_size
            if size > MAX_DATABASE:
                raise ValueError("Telemetry exceeds its content byte limit")
            offset = 0
            while offset < size:
                block = os.pread(fd, min(1024 * 1024, size - offset), offset)
                if not block:
                    raise ValueError("Telemetry was truncated")
                output.write(block)
                offset += len(block)
        connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-8192")
            deadline = time.monotonic() + 30
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            if (
                connection.execute("PRAGMA application_id").fetchone()[0] != 1213027156
                or connection.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise ValueError("Unsupported telemetry schema/version")
            objects = connection.execute(
                "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if set(objects) != {
                *(("table", name) for name in TABLE_COLUMNS),
                ("index", "frames_time"),
                ("index", "config_boundaries"),
            }:
                raise ValueError("Unexpected telemetry schema objects")
            expected_schema = sqlite3.connect(":memory:")
            try:
                expected_schema.executescript(TELEMETRY_SCHEMA)
                expected = expected_schema.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
                ).fetchall()
            finally:
                expected_schema.close()
            actual = connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
            ).fetchall()
            normalize = lambda rows: [
                (
                    kind,
                    name,
                    table,
                    None if sql is None else re.sub(r"\s+", " ", sql).strip(),
                )
                for kind, name, table, sql in rows
            ]
            if normalize(actual) != normalize(expected):
                raise ValueError("Telemetry schema definitions differ from producer v1")
            counts = {}
            for table, columns in TABLE_COLUMNS.items():
                actual = connection.execute(f"PRAGMA table_info({table})").fetchall()
                if [row[1] for row in actual] != columns.split():
                    raise ValueError("Unexpected telemetry columns")
                # SQL length inspection keeps oversized cells out of Python on
                # supported Python 3.10, which has no Connection.setlimit API.
                oversized = " OR ".join(
                    f"(typeof([{column}]) IN ('text','blob') AND length([{column}])>{MAX_METADATA})"
                    for column in columns.split()
                )
                if connection.execute(
                    f"SELECT 1 FROM {table} WHERE {oversized} LIMIT 1"
                ).fetchone():
                    raise ValueError("Telemetry cell exceeds its byte limit")
                counts[table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                if counts[table] > MAX_ROWS:
                    raise ValueError("Telemetry exceeds its row limit")
            if (
                connection.execute("PRAGMA integrity_check(1)").fetchone() != ("ok",)
                or connection.execute("PRAGMA foreign_key_check").fetchone()
            ):
                raise ValueError("Telemetry integrity or references are invalid")
            runs = connection.execute(
                "SELECT run_id,producer,completed,outcome,sample_count,length(source_config),length(effective_config) FROM runs"
            ).fetchall()
            if len(runs) != 1:
                raise ValueError("Telemetry must contain exactly one recording")
            run, producer, completed, outcome, samples, source_size, effective_size = runs[0]
            if (
                str(UUID(run)) != run
                or producer != "hm"
                or completed != 1
                or outcome not in {"end-of-stream", "intentional-stop"}
                or not 0 < samples == counts["frames"]
                or max(source_size, effective_size) > MAX_DOCUMENT
            ):
                raise ValueError("Telemetry recording is incomplete or inconsistent")
            expected = canonical(binding(request)).decode()
            events = connection.execute(
                "SELECT sample_boundary,key,value,artifact_name,artifact_contents FROM config_events WHERE kind='web-program-binding'"
            ).fetchall()
            if events != [(1, "request-v1", expected, "", "")]:
                raise ValueError(
                    "Telemetry does not belong to the admitted source/output generation"
                )
            import yaml

            configs = connection.execute(
                "SELECT source_config,effective_config FROM runs"
            ).fetchone()
            if any(not isinstance(yaml.safe_load(value), dict) for value in configs):
                raise ValueError("Invalid telemetry configuration archive")
            if connection.execute(
                "SELECT sample_boundary,key,value,artifact_name,artifact_contents FROM config_events WHERE kind='run-configuration'"
            ).fetchall() != [(1, "startup", "", "run-config.yaml", configs[0])]:
                raise ValueError("Telemetry startup configuration companion is inconsistent")
            previous = None
            for row in connection.execute(
                "SELECT sample_id,source_id,source_frame,pts_ns,seek_epoch,reset_epoch,detection_count,track_count FROM frames ORDER BY sample_id"
            ):
                sample, source, frame, pts, seek, reset, detections, tracks = row
                if any(
                    type(value) is not int or value < 0
                    for value in (
                        sample,
                        source,
                        frame,
                        seek,
                        reset,
                        detections,
                        tracks,
                    )
                ) or (pts is not None and (type(pts) is not int or pts < 0)):
                    raise ValueError("Invalid telemetry frame timeline/counts")
                if sample != (1 if previous is None else previous[0] + 1):
                    raise ValueError("Nonsequential telemetry samples")
                previous = (sample, seek, reset, pts)
            for table, count in (
                ("detections", "detection_count"),
                ("tracks", "track_count"),
            ):
                if connection.execute(
                    f"SELECT 1 FROM frames f LEFT JOIN (SELECT run_id,sample_id,count(*) n FROM {table} GROUP BY run_id,sample_id) o USING(run_id,sample_id) WHERE f.{count} != coalesce(o.n,0) LIMIT 1"
                ).fetchone():
                    raise ValueError("Telemetry observation counts disagree with frames")
            for table in ("detections", "tracks", "cameras"):
                for row in connection.execute(f"SELECT left,top,width,height FROM {table}"):
                    if (
                        any(
                            type(value) not in {float, int} or not math.isfinite(value)
                            for value in row
                        )
                        or row[2] <= 0
                        or row[3] <= 0
                    ):
                        raise ValueError("Invalid telemetry observation geometry")
            for table in ("detections", "tracks"):
                for score, label in connection.execute(f"SELECT score,class_id FROM {table}"):
                    if (
                        type(score) not in {float, int}
                        or not math.isfinite(score)
                        or type(label) is not int
                        or label < 0
                    ):
                        raise ValueError("Invalid telemetry observation score/class")
            for tracking, attributes in connection.execute(
                "SELECT tracking_id,attributes_json FROM tracks"
            ):
                if (
                    not re.fullmatch(r"0|[1-9][0-9]{0,19}", tracking)
                    or int(tracking) > 2**64 - 1
                    or not isinstance(strict_json(attributes), dict)
                ):
                    raise ValueError("Invalid telemetry track identity/attributes")
            for table in ("detections", "tracks", "replay_tracks"):
                if connection.execute(
                    f"SELECT 1 FROM {table} GROUP BY run_id,sample_id HAVING sum(typeof(ordinal)!='integer')>0 OR min(ordinal)!=0 OR max(ordinal)!=count(*)-1 LIMIT 1"
                ).fetchone():
                    raise ValueError("Nonsequential telemetry observation ordinals")
            for decoded, sequence, ntp in connection.execute(
                "SELECT decoded_source_id,decoded_sequence,ntp_ns FROM frames"
            ):
                if any(
                    value is not None and (type(value) is not int or value < 0)
                    for value in (decoded, sequence, ntp)
                ):
                    raise ValueError("Invalid telemetry decoded identity/timestamp")
            for (role,) in connection.execute("SELECT DISTINCT role FROM cameras"):
                if role not in {"program", "fast", "program-native"}:
                    raise ValueError("Unsupported telemetry camera role")
            for row in connection.execute(
                "SELECT arena_left,arena_top,arena_right,arena_bottom,stepped,has_received_tracks,edge_rotation_left,edge_rotation_right FROM replay_frames"
            ):
                if (
                    any(
                        type(value) not in {float, int} or not math.isfinite(value) for value in row
                    )
                    or row[2] <= row[0]
                    or row[3] <= row[1]
                    or any(type(value) is not int or value not in {0, 1} for value in row[4:6])
                ):
                    raise ValueError("Invalid telemetry replay geometry/state")
            for tracking, left, top, right, bottom in connection.execute(
                "SELECT tracking_id,left,top,right,bottom FROM replay_tracks"
            ):
                if (
                    not re.fullmatch(r"0|[1-9][0-9]{0,19}", tracking)
                    or int(tracking) > 2**64 - 1
                    or any(
                        type(value) not in {float, int} or not math.isfinite(value)
                        for value in (left, top, right, bottom)
                    )
                    or right <= left
                    or bottom <= top
                ):
                    raise ValueError("Invalid telemetry replay track")
            for version, state, base in connection.execute(
                "SELECT schema_version,state,base_state FROM checkpoints"
            ):
                if version != 1 or not all(
                    isinstance(strict_json(value), dict) for value in (state, base)
                ):
                    raise ValueError("Unsupported telemetry replay checkpoint")
            # This producer emits no rink_inputs. Future payload codecs require
            # their own validator before they can cross the web boundary.
            if counts["rink_inputs"]:
                raise ValueError("Unsupported telemetry rink input payload")
            previous_event = 0
            for event, boundary, kind, key, value, name, contents in connection.execute(
                "SELECT event_id,sample_boundary,kind,key,value,artifact_name,artifact_contents FROM config_events ORDER BY event_id"
            ):
                if (
                    type(event) is not int
                    or event != previous_event + 1
                    or type(boundary) is not int
                    or not 1 <= boundary <= samples
                ):
                    raise ValueError("Invalid telemetry configuration event order")
                if kind == "camera-policy":
                    if not key or name or contents or not isinstance(strict_json(value), dict):
                        raise ValueError("Invalid telemetry camera policy event")
                elif kind not in {"run-configuration", "web-program-binding"}:
                    raise ValueError("Unsupported telemetry configuration event")
                previous_event = event
            geometry = []
            for (
                width,
                height,
                space,
                codec,
                mask,
                mask_sha,
                transform,
            ) in connection.execute(
                "SELECT width,height,coordinate_space,mask_codec,mask,mask_sha256,mask_to_tracking FROM geometries"
            ):
                dimensions(width, height)
                if space != "original_stitched_pixels" or strict_json(transform) != [
                    [1, 0, 0],
                    [0, 1, 0],
                ]:
                    raise ValueError("Unsupported telemetry coordinate geometry")
                if mask is not None:
                    if (
                        codec != "png"
                        or hashlib.sha256(mask).hexdigest() != mask_sha
                        or png(mask) != {"schema": "mask-png-v1", "width": width, "height": height}
                    ):
                        raise ValueError("Telemetry mask content does not match its geometry")
                elif codec is not None or mask_sha is not None:
                    raise ValueError("Telemetry mask metadata has no pixels")
                geometry.append([width, height])
            if not geometry:
                raise ValueError("Telemetry has no geometry")
            context["frames"] = array.array(
                "q",
                (
                    row[0]
                    for row in connection.execute(
                        "SELECT DISTINCT source_frame FROM frames ORDER BY source_frame"
                    )
                ),
            )
            context["geometries"] = geometry
            return {
                "schema": "telemetry-sqlite-v1",
                "recording_id": run,
                "outcome": outcome,
                "sample_count": samples,
                "counts": counts,
                "geometries": geometry,
            }
        finally:
            connection.close()


def companion(fd, name, context):
    data = read_fd(fd, MAX_METADATA)
    if re.fullmatch(r"(?:.*[-_])?rink_mask_[0-9]+\.png", name):
        result = png(data)
        if (
            context.get("geometries")
            and [result["width"], result["height"]] not in context["geometries"]
        ):
            raise ValueError("Program mask companion belongs to another canvas")
        return result
    if name.endswith("recording_segments.json"):
        # Browser controls cannot toggle recording. Segmented archives need a
        # separate admission contract (and repeated output roles) before adoption.
        raise ValueError("Segmented Program recordings are not admitted")
    match = re.fullmatch(
        r"(?:.*[-_])?(tracking|detections|camera|camera_fast|pose|poses|actions|camera_policy|camera_fast_policy)\.csv",
        name,
    )
    if not match:
        raise ValueError("Unsupported Program metadata schema")
    kind = match[1]
    fields = {
        "tracking": 13,
        "detections": 7,
        "camera": 5,
        "camera_fast": 5,
        "pose": 2,
        "poses": 2,
        "actions": 2,
        "camera_policy": 2,
        "camera_fast_policy": 2,
    }[kind]
    csv.field_size_limit(1024 * 1024)
    count, first, last = 0, None, None
    camera_frames = (
        array.array("Q")
        if kind in {"camera", "camera_fast", "camera_policy", "camera_fast_policy"}
        else None
    )
    for row in csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True):
        if len(row) != fields or not re.fullmatch(r"[0-9]+", row[0]):
            raise ValueError("Invalid Program companion row schema")
        frame = int(row[0])
        if camera_frames is not None:
            if frame > 2**64 - 1:
                raise ValueError("Program camera companion frame exceeds its integer range")
            if last != frame:
                camera_frames.append(frame)
        frames = context.get("frames")
        if frames is not None:
            index = bisect.bisect_left(frames, frame)
            if index == len(frames) or frames[index] != frame:
                raise ValueError("Program companion refers to an absent telemetry frame")
        if last is not None and frame < last:
            raise ValueError("Program companion frame order is invalid")
        if fields == 2:
            value = strict_json(row[1])
            if kind in {"pose", "poses"} and (
                not isinstance(value, dict)
                or not isinstance(value.get("predictions"), list)
                or any(not isinstance(item, dict) for item in value["predictions"])
            ):
                raise ValueError("Invalid pose companion schema")
            if kind == "actions" and (
                not isinstance(value, list) or any(not isinstance(item, dict) for item in value)
            ):
                raise ValueError("Invalid actions companion schema")
            if "policy" in kind and (
                not isinstance(value, dict)
                or value.get("schema") != "hm-camera-policy-v1"
                or value.get("kind") != ("startup" if count == 0 else "change")
                or not isinstance(value.get("policy"), dict)
                or (last is not None and frame <= last)
            ):
                raise ValueError("Invalid camera policy companion schema")
        else:
            numeric = row[1:9] if kind == "tracking" else row[1:]
            if any(not math.isfinite(float(value)) for value in numeric):
                raise ValueError("Nonfinite Program companion values")
            if kind == "tracking":
                if (
                    not re.fullmatch(r"[0-9]+", row[1])
                    or int(row[1]) > 2**64 - 1
                    or float(row[4]) <= 0
                    or float(row[5]) <= 0
                    or not float(row[7]).is_integer()
                    or float(row[7]) < 0
                    or not isinstance(strict_json(row[9]), dict)
                    or (row[11] and not math.isfinite(float(row[11])))
                    or (row[12] and not re.fullmatch(r"-?[0-9]+", row[12]))
                ):
                    raise ValueError("Invalid tracking companion observation")
            elif kind == "detections":
                if (
                    float(row[3]) <= float(row[1])
                    or float(row[4]) <= float(row[2])
                    or not float(row[6]).is_integer()
                    or float(row[6]) < 0
                ):
                    raise ValueError("Invalid detection companion observation")
            elif float(row[3]) <= 0 or float(row[4]) <= 0:
                raise ValueError("Invalid camera companion observation")
        count += 1
        if count > MAX_ROWS:
            raise ValueError("Program companion exceeds its row limit")
        first = frame if first is None else first
        last = frame
    if camera_frames is not None:
        context.setdefault("camera_csv_frames", {})[name] = camera_frames
    return {
        "schema": kind + "-csv-v1",
        "rows": count,
        "first_frame": first,
        "last_frame": last,
    }


def describe(spec):
    request, entries, fds = spec["request"], spec["entries"], spec["fds"]
    versions = {entry["version"] for entry in entries}
    if (
        not 1 <= len(entries) <= MAX_FILES
        or len(versions) != 1
        or any(type(v) is not int or v < 1 for v in versions)
    ):
        raise ValueError("Program companions must share one positive publication version")
    if len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("Program publication repeats a companion logical name")
    for entry in entries:
        match = re.search(r"-([0-9]+)$", Path(entry["path"]).stem)
        if match is None or int(match[1]) != entry["version"]:
            raise ValueError("Program companion filename and publication version differ")
    config_fd = spec["configuration_fd"]
    if content(config_fd, MAX_DOCUMENT) != request["game_config"]:
        raise ValueError("Program admitted configuration bytes changed")
    import yaml

    recipe = yaml.safe_load(read_fd(config_fd, MAX_DOCUMENT))
    if not isinstance(recipe, dict):
        # A malformed serialized artifact is invalid input at this boundary.
        raise ValueError("Invalid Program configuration")  # noqa: TRY004
    artifacts, context = [], {}
    for entry in sorted(entries, key=lambda row: (row["kind"] != "telemetry", row["path"])):
        fd, kind = fds[entry["path"]], entry["kind"]
        if kind in {"program", "stitched", "program-4k"}:
            maximum, inspect = MAX_MEDIA, partial(media, fd, request, recipe, kind)
        elif kind == "telemetry":
            maximum, inspect = MAX_DATABASE, partial(telemetry, fd, request, context)
        elif kind == "metadata":
            maximum, inspect = MAX_METADATA, partial(companion, fd, entry["name"], context)
        else:
            raise ValueError("Unsupported Program output role")
        actual = content(fd, maximum)
        if actual["size"] != int(entry["size"]):
            raise ValueError("Program content differs from its receipt size")
        artifacts.append(
            {
                **{key: entry[key] for key in ("path", "name", "kind", "version")},
                "content": actual,
                "format": inspect(),
            }
        )
    videos = [
        row["format"] for row in artifacts if row["kind"] in {"program", "stitched", "program-4k"}
    ]
    camera_frames = context.get("camera_csv_frames", {})
    for row in artifacts:
        if row["format"]["schema"] not in {"camera_policy-csv-v1", "camera_fast_policy-csv-v1"}:
            continue
        # Match read_camera_policy_boundaries against the paired camera CSV,
        # independently of membership in the telemetry recording's frame set.
        camera = camera_frames.get(row["name"].replace("_policy.csv", ".csv"))
        policy = camera_frames[row["name"]]
        if camera is None:
            raise ValueError("Camera policy companion has no matching camera CSV")
        if not camera or not policy or policy[0] != camera[0]:
            raise ValueError("Camera policy startup does not match its camera CSV")
        for boundary in policy:
            index = bisect.bisect_left(camera, boundary)
            if index == len(camera) or camera[index] != boundary:
                raise ValueError("Camera policy event has no matching camera CSV frame")
    if videos:
        durations = [float(row["format"]["duration"]) for row in videos]
        if max(durations) - min(durations) > 0.25:
            raise ValueError("Program archive companions have different timelines")
    return {
        "protocol": "hstream-web-output-content",
        "version": 1,
        **binding(request),
        "publication_version": next(iter(versions)),
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
    }


def main():
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_DATABASE, MAX_DATABASE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (96, 96))
    data = sys.stdin.buffer.read(MAX_DOCUMENT + 1)
    if len(data) > MAX_DOCUMENT:
        raise ValueError("Program probe request exceeds its byte limit")
    spec = strict_json(data)
    result = canonical(describe(spec))
    if len(result) > MAX_DOCUMENT:
        raise ValueError("Program manifest exceeds its byte limit")
    sys.stdout.buffer.write(result)


if __name__ == "__main__":
    try:
        main()
    # Bound malformed-input diagnostics at the process boundary.
    except Exception as error:  # noqa: BLE001
        sys.stderr.write(type(error).__name__ + ": " + str(error)[:512])
        raise SystemExit(1) from None
