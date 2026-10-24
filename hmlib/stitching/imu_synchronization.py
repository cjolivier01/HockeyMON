"""Bounded GoPro/Insta360 gyro extraction and orientation-independent sync.

The parser and normalized magnitude correlation follow HStream's native IMU
contract. Metadata is demuxed without decoding video or loading a full recording.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np

_MAX_SAMPLES = 200000
_DURATION = 35.0
_INSTA_MARKER = b"8db42d694ccc418790edff439fe026bf"


def parse_gpmf(data: bytes, seconds: float, duration: float) -> np.ndarray:
    if (
        len(data) > 4 * 1024 * 1024
        or not math.isfinite(seconds)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("Invalid GPMF packet or timing")
    result = []

    def parse(payload: bytes, depth: int):
        if depth > 8:
            raise ValueError("GPMF nesting is too deep")
        position = 0
        scale = None
        while position < len(payload):
            if len(payload) - position < 8:
                raise ValueError("Truncated GPMF header")
            key, kind, width, count = struct.unpack_from(">4sBBH", payload, position)
            length = width * count
            padded = (length + 3) & ~3
            if position + 8 + padded > len(payload):
                raise ValueError("Truncated GPMF payload")
            value = payload[position + 8 : position + 8 + length]
            if kind == 0:
                parse(value, depth + 1)
            elif key == b"SCAL":
                scale = (kind, width, count, value)
            elif key == b"GYRO" and not result:
                if kind != ord("s") or width != 6 or count == 0:
                    raise ValueError("Unsupported GPMF gyro format: expected signed 16-bit triples")
                factors = np.ones(3)
                if scale is not None:
                    scalar_kind, scalar_width, scalar_count, scalar_data = scale
                    if (
                        scalar_kind not in (ord("l"), ord("s"))
                        or scalar_count not in (1, 3)
                        or scalar_width != (4 if scalar_kind == ord("l") else 2)
                    ):
                        raise ValueError("Unsupported GPMF gyro scale")
                    raw = np.frombuffer(scalar_data, dtype=">i4" if scalar_width == 4 else ">i2")
                    factors = raw[np.arange(3) % scalar_count]
                    if (factors == 0).any():
                        raise ValueError("Zero GPMF gyro scale")
                values = np.frombuffer(value, dtype=">i2").reshape(count, 3) / factors
                for index, axes in enumerate(values):
                    result.append([seconds + duration * index / count, *axes.tolist()])
            position += 8 + padded

    parse(data, 0)
    return np.asarray(result, dtype=np.float64).reshape(-1, 4)


def _insta_record(video: Path, wanted: int, maximum: int) -> tuple[bytes, int] | None:
    with video.open("rb") as stream:
        stream.seek(0, 2)
        file_size = stream.tell()
        if file_size < 78:
            return None
        end = file_size

        def footer(position):
            stream.seek(position - 78)
            data = stream.read(78)
            return data if len(data) == 78 and data[46:] == _INSTA_MARKER else None

        trailer = footer(end)
        if trailer is None:
            position = 0
            for _ in range(4096):
                if position + 8 > file_size:
                    break
                stream.seek(position)
                header = stream.read(8)
                length, kind = struct.unpack(">I4s", header)
                header_size = 8
                if length == 1:
                    extended = stream.read(8)
                    if len(extended) != 8:
                        raise ValueError("Truncated MP4 extended box")
                    length = int.from_bytes(extended, "big")
                    header_size = 16
                elif length == 0:
                    length = file_size - position
                if not header_size <= length <= file_size - position:
                    break
                if kind == b"inst" and length >= header_size + 78:
                    end = position + length
                    trailer = footer(end)
                    break
                position += length
            if trailer is None:
                return None
        trailer_size = int.from_bytes(trailer[38:42], "little")
        if not 78 <= trailer_size <= end:
            raise ValueError("Invalid Insta360 trailer size")
        start, position = end - trailer_size, end - 72
        visited = set()
        directory_bytes = 0
        for _ in range(4096):
            if position - start < 6:
                return None
            if position in visited:
                raise ValueError("Insta360 directory cycle")
            visited.add(position)
            stream.seek(position - 6)
            kind, length = struct.unpack("<HI", stream.read(6))
            if length > position - start - 6:
                raise ValueError("Invalid Insta360 record size")
            position -= 6 + length
            if kind == 0 and length % 10 == 0 and length <= 40960:
                directory_bytes += length
                if directory_bytes > 4 * 1024 * 1024:
                    raise ValueError("Insta360 directory read limit exceeded")
                stream.seek(position)
                directory = stream.read(length)
                for offset in range(0, length, 10):
                    if directory[offset] != wanted:
                        continue
                    size, location = struct.unpack_from("<II", directory, offset + 2)
                    if location + size + 6 > trailer_size:
                        raise ValueError("Invalid Insta360 directory entry")
                    position = start + location + size + 6
                    break
                continue
            if kind >> 8 == wanted:
                stream.seek(position)
                data = stream.read(min(length, maximum))
                if len(data) != min(length, maximum):
                    raise ValueError("Truncated Insta360 metadata")
                return data, length
        raise ValueError("Insta360 metadata record limit exceeded")


def _insta_clock(data: bytes) -> tuple[float, float, float]:
    position = 0

    def varint():
        nonlocal position
        result = 0
        for shift in range(0, 64, 7):
            if position >= len(data):
                break
            value = data[position]
            position += 1
            if shift == 63 and value > 1:
                break
            result |= (value & 127) << shift
            if not value & 128:
                return result
        raise ValueError("Invalid Insta360 metadata varint")

    first, raw, has_offset, offset = None, False, False, 0.0
    while position < len(data):
        key = varint()
        if key < 8:
            raise ValueError("Invalid Insta360 clock field")
        wire, field = key & 7, key >> 3
        if wire == 0:
            value = varint()
            if field == 24:
                first = value
            elif field == 62:
                raw = value != 0
            elif field == 29:
                has_offset = value != 0
            continue
        if wire == 1:
            length = 8
        elif wire == 2:
            length = varint()
        elif wire == 5:
            length = 4
        else:
            raise ValueError("Invalid Insta360 clock wire type")
        if position + length > len(data):
            raise ValueError("Truncated Insta360 clock metadata")
        if field == 28 and wire == 1:
            offset = struct.unpack_from("<d", data, position)[0]
        position += length
    if first is None or first > 2**63 - 1 or not math.isfinite(offset):
        raise ValueError("Missing or invalid Insta360 clock origin")
    return first / 1000.0, 0.001 if raw else 1.0, offset / 1000.0 if has_offset else 0.0


def _validate(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not 100 <= len(values) <= _MAX_SAMPLES:
        raise ValueError("IMU synchronization requires 100 to 200000 gyro samples per camera")
    if not np.isfinite(values).all() or (np.diff(values[:, 0]) <= 0).any():
        raise ValueError("Non-finite or non-increasing gyroscope samples")
    if (np.diff(values[:, 0]) > 0.1).any():
        raise ValueError("Gyroscope timestamps contain gaps larger than 100ms")
    return values


def read_gyro(video: str | Path) -> tuple[np.ndarray, float]:
    import av

    video = Path(video)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_tag_string,avg_frame_rate,start_time",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    picture = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if picture is None:
        raise ValueError("IMU synchronization requires a video stream")
    numerator, denominator = map(float, picture["avg_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else 0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("IMU synchronization requires a positive video frame rate")
    record = _insta_record(video, 1, 4 * 1024 * 1024)
    if record is not None:
        if len(record[0]) != record[1]:
            raise ValueError("Insta360 clock metadata is too large")
        origin, scale, offset = _insta_clock(record[0])
        stride = 20 if scale == 0.001 else 56
        record = _insta_record(video, 3, stride * _MAX_SAMPLES)
        if record is None or record[1] % stride:
            raise ValueError("Missing or invalid Insta360 gyro record")
        samples = []
        for position in range(0, len(record[0]) - stride + 1, stride):
            value = record[0][position : position + stride]
            seconds = (int.from_bytes(value[:8], "little") / 1000.0 - origin) * scale - offset
            if seconds > _DURATION:
                break
            axes = (
                [number - 32768 for number in struct.unpack_from("<HHH", value, 14)]
                if stride == 20
                else list(struct.unpack_from("<ddd", value, 32))
            )
            samples.append([seconds, *axes])
        return _validate(np.asarray(samples)), fps
    metadata = next(
        (stream for stream in streams if stream.get("codec_tag_string") == "gpmd"), None
    )
    if metadata is None:
        raise ValueError(f"No supported GoPro GPMF or Insta360 gyro telemetry in {video}")
    origin = float(picture.get("start_time", 0))
    samples, count = [], 0
    with av.open(str(video), timeout=(30, 30)) as container:
        stream = container.streams[metadata["index"]]
        for packet in container.demux(stream):
            if not packet.size:
                continue
            if packet.pts is None or packet.duration <= 0:
                raise ValueError("GPMF packet has no timestamp or duration")
            seconds = float(packet.pts * packet.time_base) - origin
            if seconds > _DURATION:
                break
            parsed = parse_gpmf(bytes(packet), seconds, float(packet.duration * packet.time_base))
            count += len(parsed)
            if count > _MAX_SAMPLES:
                raise ValueError("Too many GPMF samples in synchronization window")
            samples.append(parsed)
    return _validate(np.concatenate(samples) if samples else np.empty((0, 4))), fps


def estimate_offset(left: np.ndarray, right: np.ndarray) -> float:
    """Return left-minus-right skip seconds; reject weak or ambiguous movement."""
    left, right = _validate(left), _validate(right)
    starts = [max(0, values[0, 0]) for values in (left, right)]
    duration = min(30.0, left[-1, 0] - starts[0], right[-1, 0] - starts[1])
    if duration < 0.5:
        raise ValueError("Insufficient gyro duration for synchronization")
    count = int(duration * 200)
    signals = [
        np.interp(
            start + np.arange(count) / 200, values[:, 0], np.linalg.norm(values[:, 1:], axis=1)
        )
        for values, start in zip((left, right), starts)
    ]
    maximum = min(count // 2, int(1000 + abs(starts[0] - starts[1]) * 200))
    scores = []
    for lag in range(-maximum, maximum + 1):
        offset = starts[0] - starts[1] - lag / 200
        if abs(offset) > 5:
            continue
        begin, end = max(0, -lag), min(count, count - lag)
        a, b = signals[0][begin:end], signals[1][begin + lag : end + lag]
        centered_a, centered_b = a - a.mean(), b - b.mean()
        va, vb = centered_a @ centered_a, centered_b @ centered_b
        if va <= 1e-12 * max(1, a @ a) or vb <= 1e-12 * max(1, b @ b):
            continue
        scores.append((lag, float((centered_a @ centered_b) / math.sqrt(va * vb))))
    if not scores or max(score for _, score in scores) < 0.7:
        raise ValueError(
            "IMU correlation is weak or constant; use audio or record shared rig movement"
        )
    best_lag, best = max(scores, key=lambda item: item[1])
    if any(abs(lag - best_lag) > 20 and score > best - 0.05 for lag, score in scores):
        raise ValueError("IMU correlation has ambiguous peaks; use audio synchronization")
    return starts[0] - starts[1] - best_lag / 200


def synchronize_by_imu(left: str, right: str) -> tuple[float, float]:
    a, fps_a = read_gyro(left)
    b, fps_b = read_gyro(right)
    offset = estimate_offset(a, b)
    return max(0, offset) * fps_a, max(0, -offset) * fps_b
