"""Local audio monitoring driven by the Python pipeline's playback position."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path


class AudioMonitor:
    """Rebase local playback on seeks, mute changes, stalls and processing drift.

    This player never feeds archive or network audio. It stays in the UI's
    process group, so the desktop's process pause suspends it with the worker.
    """

    def __init__(
        self,
        source: Path | list[Path],
        *,
        durations: list[float] | None = None,
        stream=0,
        offset=0.0,
        muted=False,
    ):
        self.sources = source if isinstance(source, list) else [source]
        self.durations = durations or [float("inf")]
        if len(self.sources) != len(self.durations) or any(value <= 0 for value in self.durations):
            raise ValueError("Audio chapters require matching positive durations")
        self.stream, self.offset = stream, offset
        self.source = None
        self.muted, self.paused = muted, False
        self.position = None
        self.updated = time.monotonic()
        self.origin = self.updated
        self.origin_position = 0.0
        self.player = None
        self.errors = tempfile.TemporaryFile()  # noqa: SIM115 - closed with the monitor

    def set_muted(self, muted: bool):
        self.muted = muted
        if muted:
            self._stop()

    def set_paused(self, paused: bool):
        self.paused = paused
        if paused:
            self._stop()
        else:
            self.updated = time.monotonic()

    def update(self, position: float, *, playing: bool):
        now = time.monotonic()
        local = max(0.0, self.offset + position)
        source = self.sources[-1]
        for source, duration in zip(self.sources, self.durations):
            if local < duration:
                break
            local -= duration
        if source != self.source:
            self._stop()
            self.source = source
        advanced = self.position != position
        if advanced:
            self.position, self.updated = position, now
        if self.player is not None and self.player.poll() is not None:
            code = self.player.returncode
            self.player = None
            if code:
                self.errors.seek(0)
                detail = self.errors.read(16384).decode(errors="replace")
                raise RuntimeError(f"Local audio playback failed ({code}): {detail}")
        if self.muted or self.paused or not playing or now - self.updated > 0.5:
            self._stop()
            return
        if self.player is not None:
            predicted = self.origin_position + now - self.origin
            if abs(predicted - position) > 0.75:
                self._stop()
        if self.player is None and advanced:
            self.errors.seek(0)
            self.errors.truncate()
            from hmlib.ui.process_guard import guarded_command

            self.player = subprocess.Popen(
                guarded_command(
                    [
                        "ffplay",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nodisp",
                        "-autoexit",
                        "-ss",
                        str(local),
                        "-ast",
                        str(self.stream),
                        str(self.source),
                    ]
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self.errors,
            )
            self.origin, self.origin_position = now, position

    def _stop(self):
        if self.player is not None:
            if self.player.poll() is None:
                self.player.terminate()
                try:
                    self.player.wait(timeout=6)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        "Audio guard is still reaping descendants after shutdown timeout"
                    ) from error
            self.player = None

    def close(self):
        try:
            self._stop()
        finally:
            self.errors.close()
