"""Optional vertical-post leveling inside an in-progress NONA calibration."""

from __future__ import annotations

import ipaddress
import io
import json
import logging
import os
import secrets
import socket
import stat
import threading
import traceback
import webbrowser
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import urlparse

from PIL import Image

from hmlib.stitching.calibration_leveling_page import PAGE
from hmlib.stitching.projections import apply_projection_framing, cap_projection_canvas
from hmlib.stitching.rink_leveling import (
    estimate_leveling,
    format_points,
    parse_rays,
    prepare_project,
)
from hmlib.stitching.settings import StitchingSettings

logger = logging.getLogger(__name__)

_MAXIMUM_PROJECT_BYTES = 16 * 1024 * 1024
_MAXIMUM_IMAGE_BYTES = 128 * 1024 * 1024
_PREVIEW_MAXIMUM_DIMENSION = 1600


class CommandRunner(Protocol):
    """Stitching command runner shared with the final calibration pipeline."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        ...


class CalibrationLevelingCancelled(RuntimeError):
    """Raised when the operator cancels the complete calibration."""


@dataclass(frozen=True)
class CalibrationLevelingResult:
    """Distinct Use, Skip and Cancel outcomes from the selector."""

    use_angles: bool
    rotation_degrees: tuple[float, float, float]
    cancel_calibration: bool = False

    def __post_init__(self) -> None:
        if self.use_angles and self.cancel_calibration:
            raise ValueError("A leveling result cannot both use angles and cancel calibration")
        object.__setattr__(self, "rotation_degrees", _coerce_rotation(self.rotation_degrees))


def _coerce_rotation(raw: Any) -> tuple[float, float, float]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or any(isinstance(value, bool) for value in raw)
    ):
        raise ValueError("Rotation must contain yaw, pitch and roll")
    try:
        rotation = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rotation must contain finite angles") from exc
    if any(not -180 <= value <= 180 for value in rotation):
        raise ValueError("Rotation angles must be between -180 and 180 degrees")
    return rotation  # type: ignore[return-value]


def _read_bounded(path: Path, maximum: int, description: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError(f"Invalid or oversized {description}: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OSError(f"The {description} changed while it was being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OSError(f"The {description} changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


class CalibrationLevelingSession:
    """Immutable alignment/framing snapshot used by selector requests."""

    def __init__(
        self,
        aligned_project: str | Path,
        framed_project: str | Path,
        image_files: Sequence[str | Path],
        settings: StitchingSettings,
        run: CommandRunner,
        resolve_binary: Callable[[str], str],
    ) -> None:
        self.aligned_project = Path(aligned_project).resolve(strict=True)
        self.framed_project = Path(framed_project).resolve(strict=True)
        if len(image_files) != 2:
            raise ValueError("Rink leveling requires exactly two camera images")
        self.image_files = tuple(Path(path).resolve(strict=True) for path in image_files)
        self.settings = settings
        self.run = run
        self.resolve_binary = resolve_binary
        self.published_rotation = _coerce_rotation(settings.framing.rotation_degrees)
        self.directory = self.framed_project.parent
        self._tool_lock = threading.Lock()
        self._sphere = self.directory / ".rink-leveling-sphere.pto"
        self._preview_framed = self.directory / ".rink-leveling-preview-framed.pto"
        self._preview_project = self.directory / ".rink-leveling-preview.pto"
        self._preview_image = self.directory / ".rink-leveling-preview.png"

        framed = _read_bounded(
            self.framed_project, _MAXIMUM_PROJECT_BYTES, "framed stitching project"
        ).decode("utf-8")
        aligned = _read_bounded(
            self.aligned_project, _MAXIMUM_PROJECT_BYTES, "aligned stitching project"
        ).decode("utf-8")
        self.prepared = prepare_project(framed)
        if prepare_project(aligned).image_sizes != self.prepared.image_sizes:
            raise ValueError("The aligned and framed stitching projects use different sources")
        if len(self.prepared.image_sizes) != 2:
            raise ValueError("The stitching project does not contain exactly two camera images")
        self._sphere.write_text(self.prepared.pto, encoding="utf-8")

        self.source_images: list[bytes] = []
        for index, (path, expected_size) in enumerate(
            zip(self.image_files, self.prepared.image_sizes, strict=True)
        ):
            payload = _read_bounded(path, _MAXIMUM_IMAGE_BYTES, f"camera {index + 1} image")
            if (
                max(expected_size) > 65535
                or expected_size[0] * expected_size[1] > 256 * 1024 * 1024
            ):
                raise ValueError(f"Camera {index + 1} image dimensions exceed selector limits")
            with Image.open(io.BytesIO(payload)) as image:
                if image.size != expected_size:
                    raise ValueError(
                        f"Camera {index + 1} image dimensions do not match the stitching project"
                    )
                image.verify()
            self.source_images.append(payload)

    def close(self) -> None:
        for path in (
            self._sphere,
            self._preview_framed,
            self._preview_project,
            self._preview_image,
        ):
            path.unlink(missing_ok=True)

    def estimate(self, posts: list[dict], requested_rotation: Any) -> dict[str, Any]:
        rotation = _coerce_rotation(requested_rotation)
        if rotation[0] != self.published_rotation[0]:
            raise ValueError("Yaw cannot be changed by rink leveling")
        points = format_points(posts, self.prepared.image_sizes)
        cameras = {post["image_index"] for post in posts}
        if cameras != {0, 1}:
            raise ValueError("Select at least one complete vertical post in each camera")
        with self._tool_lock:
            output = self.run(
                [self.resolve_binary("pano_trafo"), str(self._sphere)],
                input_text=points,
                timeout_seconds=60,
            )
        estimate = estimate_leveling(
            parse_rays(output, len(posts)), self.published_rotation, self.published_rotation[0]
        )
        return asdict(estimate)

    def preview(self, requested_rotation: Any) -> bytes:
        rotation = _coerce_rotation(requested_rotation)
        if rotation[0] != self.published_rotation[0]:
            raise ValueError("Yaw cannot be changed by rink leveling")
        preview_settings = replace(
            self.settings,
            framing=replace(self.settings.framing, rotation_degrees=rotation),
        )
        bounded_settings = replace(
            preview_settings,
            max_output_dimension=_PREVIEW_MAXIMUM_DIMENSION,
            max_output_width=_PREVIEW_MAXIMUM_DIMENSION,
        )
        with self._tool_lock:
            for path in (self._preview_framed, self._preview_project, self._preview_image):
                path.unlink(missing_ok=True)
            self._preview_framed.write_bytes(
                _read_bounded(
                    self.aligned_project,
                    _MAXIMUM_PROJECT_BYTES,
                    "aligned stitching project",
                )
            )
            apply_projection_framing(
                self._preview_framed,
                preview_settings,
                self.run,
                self.resolve_binary("pano_modify"),
            )
            self._preview_project.write_bytes(
                _read_bounded(
                    self._preview_framed,
                    _MAXIMUM_PROJECT_BYTES,
                    "framed preview project",
                )
            )
            cap_projection_canvas(
                self._preview_project,
                bounded_settings,
                self.run,
                self.resolve_binary("pano_modify"),
            )
            self.run(
                [
                    self.resolve_binary("nona"),
                    "-m",
                    "PNG",
                    "--ignore-exposure",
                    "--seam=blend",
                    "-o",
                    str(self._preview_image),
                    str(self._preview_project),
                ],
                timeout_seconds=60,
            )
            payload = _read_bounded(
                self._preview_image, _MAXIMUM_IMAGE_BYTES, "rink leveling preview"
            )
            with Image.open(io.BytesIO(payload)) as image:
                if max(image.size) > _PREVIEW_MAXIMUM_DIMENSION:
                    raise ValueError("The rink leveling preview exceeded its size limit")
                image.verify()
            return payload


def _has_local_display() -> bool:
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _iter_local_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            address = result[4][0]
            parsed = ipaddress.ip_address(address)
            if not parsed.is_loopback and address != "0.0.0.0":
                addresses.add(address)
    except (OSError, ValueError) as exc:
        logger.warning("Could not enumerate local selector addresses: %s", exc)
    return sorted(addresses)


class CalibrationLevelingSelector:
    """Authenticated local web selector for posts, estimates and previews."""

    def __init__(
        self,
        session: CalibrationLevelingSession,
        *,
        game_id: str,
        bind_host: str = "0.0.0.0",
        port: int = 0,
        open_browser: bool | None = None,
    ) -> None:
        self.session = session
        self.game_id = game_id
        self.initial_rotation = session.published_rotation
        self.bind_host = bind_host
        self.port = port
        self.open_browser = _has_local_display() if open_browser is None else open_browser
        self.result = CalibrationLevelingResult(False, self.initial_rotation, True)
        self._token = secrets.token_urlsafe(32)
        self._completion = threading.Event()
        self._completed = False
        self._state_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._latest_preview: tuple[float, float, float] | None = None
        self._preview_bytes: bytes | None = None
        self.access_urls: list[str] = []
        self._allowed_hosts: set[str] = set()

    def run(self) -> CalibrationLevelingResult:
        self._start_server()
        print("\nRink post selector is ready.", flush=True)
        print(
            "Select vertical posts, preview the result, or skip leveling at one of these links:",
            flush=True,
        )
        for url in self.access_urls:
            print(f"  {url}", flush=True)
        print("", flush=True)
        if self.open_browser:
            browser_url = next(
                (url for url in self.access_urls if "localhost" in url), self.access_urls[0]
            )

            def open_selector() -> None:
                try:
                    if not webbrowser.open(browser_url, new=1, autoraise=True):
                        logger.warning("Could not launch a browser; open the selector URL manually")
                except (OSError, webbrowser.Error) as exc:
                    logger.warning("Could not launch the rink post selector: %s", exc)

            threading.Thread(target=open_selector, daemon=True).start()
        try:
            self._completion.wait()
            with self._operation_lock:
                return self.result
        finally:
            self.close()

    def close(self) -> None:
        with self._state_lock:
            if not self._completed:
                self.result = CalibrationLevelingResult(False, self.initial_rotation, True)
                self._completed = True
                self._completion.set()
        server, server_thread = self._server, self._server_thread
        self._server = None
        self._server_thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=2)

    def _start_server(self) -> None:
        server = ThreadingHTTPServer((self.bind_host, self.port), self._build_handler())
        server.daemon_threads = True
        server.allow_reuse_address = True
        self._server = server
        self.port = int(server.server_address[1])
        hosts = (
            [socket.gethostname(), "localhost", "127.0.0.1", *_iter_local_ipv4_addresses()]
            if self.bind_host in ("", "0.0.0.0")
            else [self.bind_host]
        )
        hosts = list(dict.fromkeys(host for host in hosts if host))
        self._allowed_hosts = {f"{host}:{self.port}" for host in hosts}
        self.access_urls = [f"http://{host}:{self.port}/#{self._token}" for host in hosts]
        self._server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._server_thread.start()

    def _require_active(self) -> None:
        with self._state_lock:
            if self._completed:
                raise ValueError("This rink leveling session is already complete")

    def _rotation(self, raw: Any) -> tuple[float, float, float]:
        selected = _coerce_rotation(raw)
        return self.initial_rotation[0], selected[1], selected[2]

    def _complete(self, action: Any, raw_rotation: Any) -> None:
        with self._state_lock:
            if self._completed:
                return
            if action == "skip":
                self.result = CalibrationLevelingResult(False, self.initial_rotation)
            elif action == "cancel":
                self.result = CalibrationLevelingResult(False, self.initial_rotation, True)
            elif action == "use":
                selected = self._rotation(raw_rotation)
                if selected != self._latest_preview:
                    raise ValueError("Preview the displayed angles before using them")
                self.result = CalibrationLevelingResult(True, selected)
            else:
                raise ValueError("Unsupported completion action")
            self._completed = True

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        return handler.headers.get("Host", "") in self._allowed_hosts

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        selector = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "HmCalibrationLeveling/1.0"

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if not selector._authorized(self):
                    self._send_json({"error": "Invalid selector host"}, HTTPStatus.FORBIDDEN)
                    return
                if path in ("/", "/index.html"):
                    self._send_bytes(
                        selector._build_page().encode("utf-8"), "text/html; charset=utf-8"
                    )
                    return
                if not secrets.compare_digest(
                    self.headers.get("X-Editor-Token", ""), selector._token
                ):
                    self._send_json({"error": "Invalid selector session"}, HTTPStatus.FORBIDDEN)
                    return
                image = path.removeprefix("/image/")
                if image in ("0", "1") and path == f"/image/{image}":
                    self._send_bytes(
                        selector.session.source_images[int(image)], "application/octet-stream"
                    )
                    return
                if path == "/preview.png" and selector._preview_bytes is not None:
                    self._send_bytes(selector._preview_bytes, "image/png")
                    return
                if path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self._send_json({"error": "Unknown selector resource"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                if not selector._authorized(self) or not secrets.compare_digest(
                    self.headers.get("X-Editor-Token", ""), selector._token
                ):
                    self._send_json({"error": "Invalid selector session"}, HTTPStatus.FORBIDDEN)
                    return
                path = urlparse(self.path).path
                try:
                    payload = self._read_json()
                    if path == "/api/estimate":
                        with selector._operation_lock:
                            selector._require_active()
                            estimate = selector.session.estimate(
                                payload.get("posts", []), payload.get("rotation")
                            )
                            with selector._state_lock:
                                selector._latest_preview = None
                        self._send_json(estimate)
                        return
                    if path == "/api/preview":
                        rotation = selector._rotation(payload.get("rotation"))
                        with selector._operation_lock:
                            selector._require_active()
                            preview = selector.session.preview(rotation)
                            with selector._state_lock:
                                selector._preview_bytes = preview
                                selector._latest_preview = rotation
                        self._send_json({"preview": f"/preview.png?v={secrets.token_hex(12)}"})
                        return
                    if path == "/api/complete":
                        selector._complete(payload.get("action"), payload.get("rotation"))
                        try:
                            self._send_json({"complete": True})
                        finally:
                            selector._completion.set()
                        return
                    self._send_json({"error": "Unknown selector action"}, HTTPStatus.NOT_FOUND)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except (OSError, RuntimeError) as exc:
                    logger.warning("Rink leveling request %s failed: %s", path, exc)
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                except Exception as exc:
                    logger.error("Unexpected rink leveling request failure: %s", exc)
                    traceback.print_exc()
                    self._send_json(
                        {"error": "Unexpected rink leveling failure"},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

            def _read_json(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("Invalid selector request length") from exc
                if not 0 < length <= 256 * 1024:
                    raise ValueError("Selector request is empty or too large")
                try:
                    value = json.loads(self.rfile.read(length))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Selector request is not valid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError("Selector request must be an object")
                return value

            def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
                self._send_bytes(
                    json.dumps(value, allow_nan=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status,
                )

            def _send_bytes(
                self,
                value: bytes,
                content_type: str,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(value)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self' blob:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "img-src 'self' blob:; frame-ancestors 'none'",
                )
                try:
                    self.end_headers()
                    self.wfile.write(value)
                except (BrokenPipeError, ConnectionResetError):
                    logger.info("Browser disconnected before %s completed", self.path)

            def log_message(self, format: str, *args: Any) -> None:
                logger.info(format, *args)

        return Handler

    def _build_page(self) -> str:
        state = {
            "gameId": self.game_id,
            "sizes": [list(size) for size in self.session.prepared.image_sizes],
            "rotation": list(self.initial_rotation),
            "draftKey": f"calibration-rink-leveling:{self.game_id}",
        }
        return PAGE.replace("__STATE__", json.dumps(state).replace("</", "<\\/"))


def select_calibration_leveling(
    *,
    aligned_project: str | Path,
    framed_project: str | Path,
    image_files: Sequence[str | Path],
    settings: StitchingSettings,
    game_id: str,
    run: CommandRunner,
    resolve_binary: Callable[[str], str],
    selector_factory: type[CalibrationLevelingSelector] = CalibrationLevelingSelector,
) -> CalibrationLevelingResult:
    """Open the selector and return its explicit Use, Skip or Cancel result."""
    session = CalibrationLevelingSession(
        aligned_project, framed_project, image_files, settings, run, resolve_binary
    )
    try:
        return selector_factory(session, game_id=game_id).run()
    finally:
        session.close()


__all__ = [
    "CalibrationLevelingCancelled",
    "CalibrationLevelingResult",
    "CalibrationLevelingSelector",
    "CalibrationLevelingSession",
    "select_calibration_leveling",
]
