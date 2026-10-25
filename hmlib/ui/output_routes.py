"""Frame-accurate primary/spare RTMP and local multi-mount RTSP routes."""

from __future__ import annotations

import json
import os
import re
import selectors
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from hmlib.ui.headless_preview import (
    FFmpegLivePublisher,
    resolve_youtube_stream_url,
    validate_youtube_stream_url,
)
from hmlib.ui.output_cadence import OutputCadence, recording_fps
from hmlib.ui.output_control import ARCHIVE_KEYS, NETWORK_KEYS, public_config, validate_changes
from hmlib.ui.workspace.files import read_json, write_json
from hmlib.utils.gpu import unwrap_tensor
from hmlib.utils.image import make_channels_last


class RtspServer:
    def __init__(self, host: str, port: int, routes: list[dict]) -> None:
        if not host or not 1 <= port <= 65535:
            raise ValueError("RTSP requires a host and port between 1 and 65535")
        # Reserve unique input ports until the helper has bound its listening socket.
        reservations = []
        try:
            for route in routes:
                reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                reservation.bind(("127.0.0.1", 0))
                route["udp_port"] = reservation.getsockname()[1]
                reservations.append(reservation)
                if route.get("audio"):
                    audio_reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    audio_reservation.bind(("127.0.0.1", 0))
                    route["audio_udp_port"] = audio_reservation.getsockname()[1]
                    reservations.append(audio_reservation)
            interpreter = os.environ.get("HM_GI_PYTHON")
            if not interpreter:
                interpreter = (
                    "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else sys.executable
                )
            self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed with the server lifecycle
            try:
                self._process = subprocess.Popen(
                    [
                        interpreter,
                        str(Path(__file__).with_name("rtsp_server.py")),
                        "--config",
                        json.dumps({"host": host, "port": port, "routes": routes}),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=self._log,
                )
            except OSError:
                self._log.close()
                raise
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                ready = selector.select(timeout=10)
            if not ready or self._process.stdout.readline().strip() != b"READY":
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
                self._log.seek(0)
                details = self._log.read().decode(errors="replace")
                self._process.stdout.close()
                self._log.close()
                raise RuntimeError(f"Local RTSP server could not start. {details}")
        finally:
            for reservation in reservations:
                reservation.close()

    def check(self) -> None:
        if self._process.poll() is not None:
            self._log.seek(0)
            raise RuntimeError(
                "Local RTSP server stopped: " + self._log.read().decode(errors="replace")
            )

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._process.stdout.close()
        self._log.close()


class OutputRoutes:
    """Publish actual frames before desktop-only drawing or preview resizing."""

    def __init__(
        self,
        config: dict,
        *,
        fps: float,
        profiler=None,
        audio: dict | None = None,
        session_dir: str | Path | None = None,
    ) -> None:
        self.publishers: list[tuple[str, FFmpegLivePublisher]] = []
        self.server: RtspServer | None = None
        if fps <= 0:
            raise ValueError("Output routes require a positive frame rate")
        self.config = dict(config)
        self.source_fps = fps
        self.fps = recording_fps({"video_out": config}, fps)
        self.cadence = OutputCadence(self.source_fps, self.fps)
        self.profiler = profiler
        self.audio = dict(audio or {})
        self.frames_written = 0
        self.session_dir = Path(session_dir) if session_dir else None
        self._configure(config, self.audio)
        self._publish_state()

    def _configure(self, config: dict, audio: dict) -> None:
        targets = []
        if config.get("show_youtube", False):
            primary = resolve_youtube_stream_url(
                config.get("youtube_stream_url"), config.get("youtube_stream_key")
            )
            validate_youtube_stream_url(primary)
            targets.append(("program", primary, "Primary RTMP", audio))
        if config.get("spare_rtmp_enabled", False):
            spare = str(config.get("spare_rtmp_url", ""))
            parsed = urlsplit(spare)
            if (
                parsed.scheme not in {"rtmp", "rtmps"}
                or not parsed.hostname
                or not parsed.path.strip("/")
            ):
                raise ValueError("Spare RTMP requires a full rtmp:// or rtmps:// ingest URL")
            targets.append(("program", spare, "Spare RTMP", audio))
        if config.get("rtsp_enabled", False):
            definitions = config.get("rtsp_mounts")
            if definitions is None:
                definitions = [
                    {"mount": config.get("rtsp_program_mount", "/program"), "source": "program"}
                ]
                if config.get("rtsp_stitched_mount"):
                    definitions.append(
                        {"mount": config["rtsp_stitched_mount"], "source": "stitched"}
                    )
            if not isinstance(definitions, list) or not 1 <= len(definitions) <= 16:
                raise ValueError("RTSP requires between one and sixteen mount definitions")
            routes = []
            for definition in definitions:
                if not isinstance(definition, dict):
                    raise TypeError("Each RTSP mount must contain mount and source fields")
                mount = definition.get("mount", "")
                source = definition.get("source", "program")
                if not isinstance(mount, str) or not re.fullmatch(r"/[A-Za-z0-9_/-]{1,100}", mount):
                    raise ValueError(
                        "RTSP mount must start with / and contain letters, numbers, dashes, underscores or slashes"
                    )
                if source not in {"program", "stitched"}:
                    raise ValueError("RTSP source must be program or stitched")
                if any(route["mount"] == mount for route in routes):
                    raise ValueError("RTSP mount paths must be unique")
                routes.append({"mount": mount, "source": source, "audio": bool(audio.get("file"))})
            self.server = RtspServer(
                str(config.get("rtsp_host", "127.0.0.1")),
                int(config.get("rtsp_port", 8554)),
                routes,
            )
            for route in routes:
                targets.append(
                    (
                        route["source"],
                        f"rtp://127.0.0.1:{route['udp_port']}?pkt_size=1200",
                        "RTSP " + route["mount"],
                        (
                            {
                                **audio,
                                "rtp_url": f"rtp://127.0.0.1:{route['audio_udp_port']}?pkt_size=1200",
                            }
                            if route.get("audio")
                            else {}
                        ),
                    )
                )
        for source, url, label, route_audio in targets:
            self.publishers.append(
                (
                    source,
                    FFmpegLivePublisher(
                        url, label=label, fps=self.fps, profiler=self.profiler, audio=route_audio
                    ),
                )
            )

    def _publish_state(self, **extra) -> None:
        if self.session_dir is not None:
            write_json(
                self.session_dir / "output-state.json",
                {
                    "state": "running",
                    "config": public_config(self.config),
                    "frames": self.frames_written,
                    "updated_at": time.time(),
                    "capabilities": ["network-routes"],
                    **extra,
                },
            )

    def _poll_actions(self, context: dict) -> None:
        if self.session_dir is None:
            return
        directory = self.session_dir / "output-actions"
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json"))[:8]:
            result_path = self.session_dir / "output-results" / path.name
            if result_path.exists():
                path.unlink()
                continue
            try:
                if path.is_symlink():
                    raise ValueError("Output request must be a regular JSON file")
                request = read_json(path, maximum=65536)
                changes = validate_changes(request["changes"])
                updated = {**self.config, **changes}
                shared = context.get("shared", {})
                archive_changes = {
                    key: value for key, value in changes.items() if key in ARCHIVE_KEYS
                }
                if set(archive_changes) - set(shared.get("recording_capabilities", [])):
                    raise ValueError("The requested recording is not available in this pipeline")
                # Validate complete route configuration before ending publishers.
                if updated.get("show_youtube"):
                    validate_youtube_stream_url(
                        resolve_youtube_stream_url(
                            updated.get("youtube_stream_url"), updated.get("youtube_stream_key")
                        )
                    )
                if updated.get("spare_rtmp_enabled") and not updated.get("spare_rtmp_url"):
                    raise ValueError("Spare RTMP requires an ingest URL")
                if any(updated.get(key) != self.config.get(key) for key in NETWORK_KEYS):
                    self.close()
                    audio = {
                        **self.audio,
                        "offset_seconds": float(self.audio.get("offset_seconds", 0))
                        - self.frames_written / self.source_fps,
                    }
                    try:
                        self._configure(updated, audio)
                    except (RuntimeError, ValueError, TypeError, OSError):
                        self.close()
                        self._configure(self.config, audio)
                        raise
                self.config = updated
                if archive_changes:
                    shared["game_config"].setdefault("video_out", {}).update(archive_changes)
                result = {"ok": True, "config": public_config(self.config)}
            except (RuntimeError, ValueError, TypeError, OSError, KeyError) as error:
                result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            write_json(result_path, result)
            self._publish_state(request_id=path.stem, **result)
            path.unlink()

    def write(self, context: dict) -> None:
        self._poll_actions(context)
        if self.server is not None:
            self.server.check()
        program = unwrap_tensor(context.get("img"))
        if program is None:
            return
        count = int(program.shape[0]) if program.ndim == 4 else 1
        indices = self.cadence.select(count)
        for source, publisher in self.publishers:
            key = "img" if source == "program" else "original_images"
            if context.get(key) is None:
                raise ValueError(f"The {source} output route requires pipeline field {key}")
            frames = make_channels_last(unwrap_tensor(context[key]))
            if frames.ndim == 3:
                frames = frames[None]
            if len(frames) != count:
                raise ValueError("Output route frame batches must share the source timeline")
            for index in indices:
                publisher.write_frame(frames[index])
        self.frames_written += count

    def close(self) -> None:
        errors = []
        for _, publisher in self.publishers:
            try:
                publisher.close()
            except (RuntimeError, OSError) as error:
                errors.append(error)
        self.publishers.clear()
        if self.server is not None:
            self.server.close()
            self.server = None
        if errors:
            raise RuntimeError("Output route shutdown failed: " + "; ".join(map(str, errors)))
