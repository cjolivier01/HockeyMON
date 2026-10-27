"""Frame-aligned decoder timestamps without reading or hashing video pixels."""

from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

MAX_TIMING_RECORDS = 256
MAX_LOG_LINE_BYTES = 64 * 1024
_TIMEBASE = re.compile(r"config in time_base: ([0-9]+)/([0-9]+),")
_FRAME = re.compile(r"\bn:\s*([0-9]+)\s+pts:\s*(-?[0-9]+|NOPTS)\s+pts_time:")


@dataclass(frozen=True)
class DecodedTimestamp:
    pts: int | None
    time_base: Fraction

    def __post_init__(self):
        if self.pts is not None and (type(self.pts) is not int or not -(2**63) < self.pts < 2**63):
            raise ValueError("Decoded PTS must be a signed 64-bit integer or null")
        if not isinstance(self.time_base, Fraction) or self.time_base <= 0:
            raise ValueError("Decoded timestamps require a positive rational timebase")

    @property
    def ns(self) -> int | None:
        if self.pts is None:
            return None
        return round(self.pts * self.time_base * 1_000_000_000)

    def document(self, chapter_index: int) -> dict:
        return {
            "chapter_index": chapter_index,
            "pts": str(self.pts) if self.pts is not None else None,
            "time_base_num": self.time_base.numerator,
            "time_base_den": self.time_base.denominator,
        }


def timestamp_from_document(document: dict) -> DecodedTimestamp:
    if not isinstance(document, dict) or set(document) != {
        "chapter_index",
        "pts",
        "time_base_num",
        "time_base_den",
    }:
        raise ValueError("Invalid decoded timing fields")
    if type(document["chapter_index"]) is not int or not 0 <= document["chapter_index"] < 2**31:
        raise ValueError("Invalid decoded chapter index")
    for key in ("time_base_num", "time_base_den"):
        if type(document[key]) is not int or not 0 < document[key] < 2**31:
            raise ValueError("Invalid decoded rational timebase")
    pts = document["pts"]
    if pts is not None and (
        not isinstance(pts, str)
        or not re.fullmatch(r"-?(0|[1-9][0-9]{0,18})", pts)
        or str(int(pts)) != pts
    ):
        raise ValueError("Decoded PTS must use a canonical signed decimal string")
    return DecodedTimestamp(
        int(pts) if pts is not None else None,
        Fraction(document["time_base_num"], document["time_base_den"]),
    )


def timing_ns(document: dict | None) -> int | None:
    """The v1 preview's nullable nonnegative JS-safe projection of raw source PTS.

    This is chapter-local source time, never a normalized playback/audio clock.
    Keep the rational document alongside it for a future lossless transport.
    """
    if document is None:
        return None
    value = timestamp_from_document(document).ns
    return value if value is not None and 0 <= value < 2**53 else None


def read_ffmpeg_timing(pipe, timing: FFmpegFrameTiming, diagnostic) -> None:
    """Drain bounded log records without retaining per-frame text diagnostics."""
    try:
        for raw_line in iter(lambda: pipe.readline(MAX_LOG_LINE_BYTES + 1), b""):
            if len(raw_line) > MAX_LOG_LINE_BYTES:
                raise ValueError("Decoder diagnostic line exceeded its bound")
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line and not timing.feed(line):
                diagnostic(line)
    except (OSError, ValueError) as error:
        timing.finish(str(error))
    finally:
        timing.finish()
        pipe.close()


class FFmpegFrameTiming:
    """Consume checksum-free showinfo from the same decoder as stdout frames.

    The reader takes one record after each complete raw frame. The fixed queue
    limit fails visibly instead of dropping timestamps or buffering a recording.
    No scalar timestamp is inferred from FPS, DTS or the requested seek position.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._records = deque()
        self._time_base = None
        self._next_ordinal = 0
        self._ended = False
        self._error = None

    def feed(self, line: str) -> bool:
        if "Parsed_showinfo_" not in line:
            return False
        with self._condition:
            try:
                if self._error is not None:
                    return True
                match = _TIMEBASE.search(line)
                if match is not None:
                    time_base = Fraction(int(match[1]), int(match[2]))
                    if time_base <= 0 or self._time_base not in {None, time_base}:
                        raise ValueError("Decoder timebase changed within one frame stream")
                    self._time_base = time_base
                elif (match := _FRAME.search(line)) is not None:
                    if int(match[1]) != self._next_ordinal or self._time_base is None:
                        raise ValueError("Decoder timestamp ordinal or timebase is missing")
                    if len(self._records) >= MAX_TIMING_RECORDS:
                        raise ValueError("Decoder timestamp queue exceeded its bound")
                    pts = None if match[2] == "NOPTS" else int(match[2])
                    self._records.append(DecodedTimestamp(pts, self._time_base))
                    self._next_ordinal += 1
                elif re.search(r"\bn:\s*", line):
                    raise ValueError("Malformed decoder timestamp record")
            except (ValueError, ZeroDivisionError) as error:
                self._error = str(error)
            self._condition.notify_all()
        return True

    def finish(self, error: str | None = None) -> None:
        with self._condition:
            self._ended = True
            if error is not None:
                self._error = error
            self._condition.notify_all()

    def take(self) -> DecodedTimestamp:
        with self._condition:
            # showinfo runs before stdout publication. Once its complete raw
            # frame was read, only the bounded stderr reader remains to run.
            if not self._condition.wait_for(
                lambda: bool(self._records) or self._error is not None or self._ended,
                timeout=5,
            ):
                raise RuntimeError("Decoder timestamp did not accompany its raw frame")
            if self._error is not None:
                raise RuntimeError(self._error)
            if not self._records:
                raise RuntimeError("Decoder timestamp stream ended before its raw frame")
            return self._records.popleft()

    def assert_drained(self) -> None:
        with self._condition:
            if self._error is not None or self._records:
                raise RuntimeError(self._error or "Decoder timestamps outnumber its raw frames")
