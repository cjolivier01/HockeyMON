from __future__ import annotations

import argparse
import base64
import contextlib
import html
import io
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Integral
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from hmlib.config import (
    get_game_config,
    get_game_dir,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.utils.image import image_height, image_width, make_visible_image, resize_image

_DONE_BACKGROUND_PATH = (
    Path(__file__).resolve().parents[1] / "images" / "scoreboard_selector_thank_you.svg"
)

# Match HStream's display proxy limits. Full PNG source decoding still belongs
# to Pillow and remains subject to its decompression-bomb limits.
_MAXIMUM_PREVIEW_DIMENSION = 8192
_MAXIMUM_PREVIEW_PIXELS = 96 * 1024 * 1024 // 4
_MAXIMUM_SOURCE_DIMENSION = 65535
_MAXIMUM_SOURCE_PIXELS = 256 * 1024 * 1024


def _bounded_preview_size(
    source_size: Tuple[int, int], max_display_height: Optional[int] = None
) -> Tuple[int, int]:
    width, height = source_size
    if (
        width <= 0
        or height <= 0
        or max(width, height) > _MAXIMUM_SOURCE_DIMENSION
        or width * height > _MAXIMUM_SOURCE_PIXELS
    ):
        raise ValueError(f"Scoreboard source image exceeds supported dimensions: {width}x{height}")
    if max_display_height is not None and (
        isinstance(max_display_height, bool)
        or not isinstance(max_display_height, Integral)
        or max_display_height <= 0
    ):
        raise ValueError("max_display_height must be a positive integer")
    scale = min(
        1.0,
        _MAXIMUM_PREVIEW_DIMENSION / width,
        _MAXIMUM_PREVIEW_DIMENSION / height,
        math.sqrt(_MAXIMUM_PREVIEW_PIXELS / (width * height)),
        max_display_height / height if max_display_height is not None else 1.0,
    )
    return max(1, int(width * scale)), max(1, int(height * scale))


def _prepare_selector_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    max_display_height: Optional[int],
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Make a bounded display proxy and retain original selection coordinates."""
    if isinstance(image, torch.Tensor) and image.ndim == 4:
        image = image[0]
    source_size = (
        image.size
        if isinstance(image, Image.Image)
        else (int(image_width(image)), int(image_height(image)))
    )
    preview_size = _bounded_preview_size(source_size, max_display_height)
    if isinstance(image, torch.Tensor):
        # Reduce GPU tensors before copying a preview to the host.
        if preview_size != source_size:
            image = resize_image(image, new_width=preview_size[0], new_height=preview_size[1])
        image = Image.fromarray(make_visible_image(image, force_numpy=True))
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.size != preview_size:
        # Pillow may fully decode a PNG here; only the proxy is retained and
        # encoded for the browser, never a second full-size RGB panorama.
        image = image.resize(preview_size, resample=Image.Resampling.LANCZOS)
    return image.convert("RGB"), source_size


def _has_local_display() -> bool:
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _default_bind_host() -> str:
    return "0.0.0.0"


def _iter_detected_ipv4_addresses(raw_output: str) -> List[str]:
    addresses: set[str] = set()
    for match in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw_output):
        try:
            parsed = ipaddress.ip_address(match)
        except ValueError:
            continue
        if parsed.version == 4 and not parsed.is_loopback and str(parsed) != "0.0.0.0":
            addresses.add(str(parsed))
    return sorted(addresses)


def _iter_local_ipv4_addresses() -> List[str]:
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for result in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            address = result[4][0]
            if address and not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    for command in (["hostname", "-I"], ["ip", "-4", "addr", "show"], ["ifconfig"]):
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue
        if completed.stdout:
            addresses.update(_iter_detected_ipv4_addresses(completed.stdout))
    return sorted(addresses)


def _build_access_urls(bind_host: str, port: int) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()

    def _add(host: str) -> None:
        if not host:
            return
        url = f"http://{host}:{port}/"
        if url not in seen:
            seen.add(url)
            urls.append(url)

    if bind_host in ("0.0.0.0", "", "::"):
        _add(socket.gethostname())
        _add("localhost")
        _add("127.0.0.1")
        for address in _iter_local_ipv4_addresses():
            _add(address)
    else:
        _add(bind_host)
        if bind_host == "127.0.0.1":
            _add("localhost")

    return urls


def _build_browser_url(port: int) -> str:
    return f"http://localhost:{port}/"


def _parse_selector_cli_args(argv: Optional[List[str]] = None) -> Tuple[Any, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--selector-bind-host",
        dest="bind_host",
        default=None,
        help="Bind host for the scoreboard selector web server.",
    )
    parser.add_argument(
        "--selector-port",
        dest="port",
        type=int,
        default=0,
        help="Port for the scoreboard selector web server. Use 0 to pick a free port.",
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument(
        "--selector-open-browser",
        dest="open_browser",
        action="store_true",
        default=None,
        help="Force opening the scoreboard selector in a local browser.",
    )
    browser.add_argument(
        "--selector-no-browser",
        dest="open_browser",
        action="store_false",
        help="Do not open a browser automatically; only print selector URLs.",
    )
    return parser.parse_known_args(argv)


def get_max_screen_height() -> Optional[int]:
    return None


class ScoreboardSelector:
    NULL_POINTS = [(0, 0), (0, 0), (0, 0), (0, 0)]

    def __init__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        initial_points: Optional[List[Tuple[int, int]]] = None,
        max_display_height: Optional[int] = None,
        game_id: Optional[str] = None,
        bind_host: Optional[str] = None,
        port: int = 0,
        open_browser: Optional[bool] = None,
    ) -> None:
        self.image, (self._image_width, self._image_height) = _prepare_selector_image(
            image, max_display_height
        )

        self._game_id = game_id or "unknown-game"
        self._session_id = uuid.uuid4().hex[:12]
        self._bind_host = bind_host or _default_bind_host()
        self._port = int(port)
        self._open_browser = _has_local_display() if open_browser is None else open_browser
        self._completion_event = threading.Event()
        self._shutdown_timer: Optional[threading.Timer] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_lock = threading.Lock()
        self._completed = False

        image_buffer = io.BytesIO()
        self.image.save(image_buffer, format="PNG")
        self._image_bytes = image_buffer.getvalue()
        self._done_background_data_url = self._load_done_background_data_url()
        self._access_urls: List[str] = []
        self._browser_url: Optional[str] = None

        self.points: List[Tuple[int, int]] = []
        if initial_points == ScoreboardSelector.NULL_POINTS:
            initial_points = []
        if initial_points:
            if len(initial_points) != 4:
                print(
                    "Warning: initial scoreboard points were not exactly 4 points. Ignoring them."
                )
            else:
                self.points = self._coerce_points(initial_points, require_four=True)

    @property
    def primary_url(self) -> str:
        if not self._access_urls:
            raise RuntimeError("Scoreboard selector server has not been started.")
        return self._access_urls[0]

    @property
    def access_urls(self) -> List[str]:
        return list(self._access_urls)

    def _load_done_background_data_url(self) -> str:
        svg_bytes = _DONE_BACKGROUND_PATH.read_bytes()
        encoded = base64.b64encode(svg_bytes).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"

    def _coerce_points(
        self,
        points: List[Any],
        require_four: bool,
    ) -> List[Tuple[int, int]]:
        if require_four and len(points) != 4:
            raise ValueError("Please select exactly four scoreboard corners before saving.")

        coerced: List[Tuple[int, int]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Every scoreboard point must contain exactly two coordinates.")
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if not (0 <= x < self._image_width and 0 <= y < self._image_height):
                raise ValueError(
                    f"Point ({x}, {y}) is outside the image bounds "
                    f"{self._image_width}x{self._image_height}."
                )
            coerced.append((x, y))

        return coerced

    def order_points_clockwise(self, pts: Union[torch.Tensor, List[Tuple[int, int]]]):
        if isinstance(pts, torch.Tensor):
            pts_np = pts.to(torch.float32).cpu().numpy()
        else:
            pts_np = np.array(pts, dtype=np.float32)

        s = pts_np.sum(axis=1)
        diff = np.diff(pts_np, axis=1)

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts_np[np.argmin(s)]
        ordered[2] = pts_np[np.argmax(s)]
        ordered[1] = pts_np[np.argmin(diff)]
        ordered[3] = pts_np[np.argmax(diff)]

        return [(int(round(x)), int(round(y))) for x, y in ordered.tolist()]

    def process_ok(self, points: Optional[List[Any]] = None) -> None:
        selected_points = (
            self.points if points is None else self._coerce_points(points, require_four=True)
        )
        ordered_points = self.order_points_clockwise(selected_points)
        print("Selected points in clockwise order starting from the upper-left point:")
        for point in ordered_points:
            print(f"({point[0]}, {point[1]})")
        self.points = ordered_points
        self._completed = True
        self._completion_event.set()

    def process_none(self) -> None:
        self.points = ScoreboardSelector.NULL_POINTS.copy()
        print("Scoreboard selector marked this frame as having no scoreboard.")
        self._completed = True
        self._completion_event.set()

    def close(self) -> None:
        self._stop_server()

    def run(self) -> None:
        self._start_server()
        self._announce_urls()
        self._maybe_open_browser()
        try:
            self._completion_event.wait()
        finally:
            self._schedule_shutdown(delay_seconds=1.0)

    def _announce_urls(self) -> None:
        print("", flush=True)
        print("Scoreboard selector is ready.", flush=True)
        print("Open any of these links in a browser to select the scoreboard:", flush=True)
        for url in self._access_urls:
            print(f"  {url}", flush=True)
        print("", flush=True)

    def _maybe_open_browser(self) -> None:
        if not self._open_browser or self._browser_url is None:
            return

        def _open() -> None:
            try:
                webbrowser.open(self._browser_url, new=1, autoraise=True)
            except Exception as exc:
                print(f"Could not launch a browser automatically: {exc}")

        threading.Thread(target=_open, daemon=True).start()

    def _schedule_shutdown(self, delay_seconds: float) -> None:
        with self._server_lock:
            if self._shutdown_timer is not None:
                return
            timer = threading.Timer(delay_seconds, self._stop_server)
            timer.daemon = True
            timer.start()
            self._shutdown_timer = timer

    def _start_server(self) -> None:
        with self._server_lock:
            if self._server is not None:
                return

            server = ThreadingHTTPServer((self._bind_host, self._port), self._build_handler())
            server.daemon_threads = True
            server.allow_reuse_address = True
            self._server = server
            self._port = int(server.server_address[1])
            self._access_urls = _build_access_urls(self._bind_host, self._port)
            self._browser_url = _build_browser_url(self._port)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            self._server_thread = server_thread

    def _stop_server(self) -> None:
        with self._server_lock:
            server = self._server
            server_thread = self._server_thread
            self._server = None
            self._server_thread = None
            self._shutdown_timer = None

        if server is None:
            return

        try:
            server.shutdown()
        finally:
            server.server_close()
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=2.0)

    def _build_handler(self):
        selector = self

        class ScoreboardSelectorHandler(BaseHTTPRequestHandler):
            server_version = "HmScoreboardSelector/1.0"

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    if selector._completed:
                        self._send_html(selector._build_completion_page())
                    else:
                        self._send_html(selector._build_selection_page())
                    return
                if path == "/image":
                    self._send_bytes(selector._image_bytes, content_type="image/png")
                    return
                if path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self._send_json({"error": f"Unknown path: {path}"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path != "/api/complete":
                    self._send_json({"error": f"Unknown path: {path}"}, status=HTTPStatus.NOT_FOUND)
                    return

                try:
                    payload = self._read_json_body()
                    action = payload.get("action")
                    if action == "save":
                        selector.process_ok(points=payload.get("points", []))
                    elif action == "none":
                        selector.process_none()
                    else:
                        raise ValueError("Unsupported action. Use 'save' or 'none'.")
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                except Exception as exc:
                    traceback.print_exc()
                    self._send_json(
                        {"error": f"Unexpected selector failure: {exc}"},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return

                self._send_html(selector._build_completion_page())

            def _read_json_body(self) -> Any:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))

            def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
                self._send_bytes(
                    body.encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                    status=status,
                )

            def _send_json(self, payload: Any, status: HTTPStatus) -> None:
                self._send_bytes(
                    json.dumps(payload).encode("utf-8"),
                    content_type="application/json; charset=utf-8",
                    status=status,
                )

            def _send_bytes(
                self,
                payload: bytes,
                content_type: str,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return ScoreboardSelectorHandler

    def _build_selection_page(self) -> str:
        state = dict(
            gameId=self._game_id,
            imageWidth=self._image_width,
            imageHeight=self._image_height,
            imageUrl="/image",
            initialPoints=[[point[0], point[1]] for point in self.points],
            draftStorageKey=f"scoreboard-selector:{self._game_id}:{self._session_id}",
        )
        state_json = json.dumps(state)
        page = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scoreboard Selector</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07141d;
      --bg-alt: #0d2432;
      --panel: rgba(8, 28, 39, 0.88);
      --panel-border: rgba(186, 227, 255, 0.16);
      --text: #edf8ff;
      --muted: #9cc3d7;
      --accent: #5cd0ff;
      --accent-strong: #9fe7ff;
      --warn: #ffb95c;
      --danger: #ff7d88;
      --success: #79f0b0;
      --shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
      --radius: 22px;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(92, 208, 255, 0.12), transparent 34%),
        linear-gradient(180deg, #081521 0%, #0f2b3d 45%, #09131d 100%);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.02), transparent 48%),
        repeating-linear-gradient(
          120deg,
          rgba(255, 255, 255, 0.02) 0,
          rgba(255, 255, 255, 0.02) 2px,
          transparent 2px,
          transparent 26px
        );
      pointer-events: none;
    }

    .page {
      position: relative;
      min-height: 100vh;
      padding: 24px;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(320px, 420px);
      gap: 24px;
      min-height: calc(100vh - 48px);
    }

    .stage-card,
    .panel-card {
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }

    .stage-card {
      position: relative;
      padding: 18px;
      overflow: hidden;
    }

    .stage-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    .eyebrow {
      color: var(--accent-strong);
      font-size: 0.83rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }

    .stage-title {
      margin: 8px 0 0;
      font-size: clamp(1.5rem, 2.1vw, 2.2rem);
      line-height: 1.05;
    }

    .stage-subtitle {
      margin: 8px 0 0;
      max-width: 52rem;
      color: var(--muted);
      line-height: 1.45;
    }

    .status-pill {
      align-self: center;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(92, 208, 255, 0.13);
      border: 1px solid rgba(92, 208, 255, 0.28);
      font-weight: 700;
      white-space: nowrap;
    }

    .canvas-wrap {
      position: relative;
      min-height: 520px;
      height: calc(100vh - 210px);
      border-radius: 20px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.04), transparent 28%),
        linear-gradient(180deg, #0b1821 0%, #122736 100%);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    #selector-canvas {
      width: 100%;
      height: 100%;
      display: block;
      touch-action: none;
      cursor: crosshair;
    }

    .canvas-hint {
      position: absolute;
      left: 18px;
      bottom: 18px;
      padding: 12px 14px;
      max-width: min(34rem, calc(100% - 36px));
      border-radius: 16px;
      background: rgba(6, 21, 30, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--muted);
      line-height: 1.42;
      font-size: 0.93rem;
    }

    .panel-card {
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .panel-header h2 {
      margin: 0;
      font-size: 1.2rem;
    }

    .panel-header p {
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.45;
    }

    .status-box {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.09);
      background: rgba(255, 255, 255, 0.04);
    }

    .status-box[data-tone="error"] {
      border-color: rgba(255, 125, 136, 0.38);
      background: rgba(255, 125, 136, 0.08);
      color: #ffd7dc;
    }

    .status-box[data-tone="success"] {
      border-color: rgba(121, 240, 176, 0.36);
      background: rgba(121, 240, 176, 0.1);
      color: #dcffe9;
    }

    .status-box strong {
      display: block;
      margin-bottom: 4px;
      font-size: 0.94rem;
      letter-spacing: 0.01em;
    }

    .steps {
      display: grid;
      gap: 12px;
    }

    .step {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }

    .step-badge {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-weight: 800;
      color: #06141d;
      background: linear-gradient(180deg, #c5f4ff 0%, #66d7ff 100%);
    }

    .step p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }

    .button-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .primary-actions {
      display: grid;
      gap: 10px;
    }

    button {
      border: 0;
      border-radius: 16px;
      padding: 14px 16px;
      font: inherit;
      font-weight: 700;
      color: var(--text);
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }

    button:hover:not(:disabled) {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.07);
    }

    .btn-accent {
      color: #04131c;
      background: linear-gradient(180deg, #d8f7ff 0%, #63d7ff 100%);
      box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.3),
        0 14px 30px rgba(92, 208, 255, 0.28);
    }

    .btn-danger {
      background: linear-gradient(180deg, rgba(255, 160, 171, 0.27), rgba(255, 125, 136, 0.15));
      box-shadow: inset 0 0 0 1px rgba(255, 125, 136, 0.3);
    }

    .point-list {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .point-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.07);
    }

    .point-label {
      font-weight: 700;
    }

    .point-value {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .footnote {
      color: var(--muted);
      line-height: 1.45;
      font-size: 0.92rem;
    }

    @media (max-width: 1100px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .canvas-wrap {
        height: 62vh;
        min-height: 420px;
      }
    }

    @media (max-width: 720px) {
      .page {
        padding: 16px;
      }

      .stage-card,
      .panel-card {
        border-radius: 20px;
      }

      .canvas-wrap {
        height: 56vh;
        min-height: 340px;
      }

      .button-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      <section class="stage-card">
        <div class="stage-header">
          <div>
            <div class="eyebrow">Scoreboard Selection</div>
            <h1 class="stage-title">Pin the four scoreboard corners</h1>
            <p class="stage-subtitle">
              Click the scoreboard corners in any order. Drag a red point to fine-tune it,
              drag empty space to pan, and use the mouse wheel or the zoom buttons to dial in precision.
            </p>
          </div>
          <div class="status-pill" id="selection-count">0 / 4 points</div>
        </div>
        <div class="canvas-wrap">
          <canvas id="selector-canvas" aria-label="Scoreboard selector canvas"></canvas>
          <div class="canvas-hint" id="canvas-hint">
            Wheel = zoom. Drag empty ice = pan. Drag a red dot = adjust. Use "Focus Points" after placing
            points if you want to zoom straight into the scoreboard area.
          </div>
        </div>
      </section>

      <aside class="panel-card">
        <div class="panel-header">
          <h2>Game: __GAME_ID__</h2>
          <p>The save button unlocks automatically once all four corners are set.</p>
        </div>

        <div class="status-box" id="status-box" data-tone="info">
          <strong id="status-title">Ready when you are</strong>
          <div id="status-message">Load the image, zoom in, and place four points on the scoreboard corners.</div>
        </div>

        <div class="steps">
          <div class="step">
            <div class="step-badge">1</div>
            <p>Use the wheel or the zoom buttons to get the scoreboard large enough to place points accurately.</p>
          </div>
          <div class="step">
            <div class="step-badge">2</div>
            <p>Click each corner once. If you miss, drag the dot into place or use Undo/Clear.</p>
          </div>
          <div class="step">
            <div class="step-badge">3</div>
            <p>Press Save Selection. If there is no scoreboard in this frame, use No Scoreboard instead.</p>
          </div>
        </div>

        <div class="button-grid">
          <button class="btn-secondary" id="zoom-out-button" type="button">Zoom Out</button>
          <button class="btn-secondary" id="zoom-in-button" type="button">Zoom In</button>
          <button class="btn-secondary" id="fit-button" type="button">Fit Image</button>
          <button class="btn-secondary" id="actual-size-button" type="button">100% Zoom</button>
          <button class="btn-secondary" id="focus-button" type="button">Focus Points</button>
          <button class="btn-secondary" id="undo-button" type="button">Undo Last Point</button>
          <button class="btn-secondary" id="clear-button" type="button">Clear Points</button>
          <button class="btn-danger" id="none-button" type="button">No Scoreboard</button>
        </div>

        <div>
          <h2>Selected Points</h2>
          <ul class="point-list" id="point-list">
            <li class="point-item"><span class="point-label">Point 1</span><span class="point-value">Not set</span></li>
            <li class="point-item"><span class="point-label">Point 2</span><span class="point-value">Not set</span></li>
            <li class="point-item"><span class="point-label">Point 3</span><span class="point-value">Not set</span></li>
            <li class="point-item"><span class="point-label">Point 4</span><span class="point-value">Not set</span></li>
          </ul>
        </div>

        <div class="primary-actions">
          <button class="btn-accent" id="save-button" type="button" disabled>Save Selection</button>
        </div>

        <div class="footnote">
          The selector remembers your in-progress points in this browser tab while the page stays open.
          Closing the browser without saving leaves the scoreboard unchanged.
        </div>
      </aside>
    </div>
  </div>

  <script>
    const INITIAL_STATE = __STATE_JSON__;

    const canvas = document.getElementById("selector-canvas");
    const ctx = canvas.getContext("2d");
    const selectionCount = document.getElementById("selection-count");
    const statusBox = document.getElementById("status-box");
    const statusTitle = document.getElementById("status-title");
    const statusMessage = document.getElementById("status-message");
    const pointList = document.getElementById("point-list");
    const saveButton = document.getElementById("save-button");
    const noneButton = document.getElementById("none-button");
    const undoButton = document.getElementById("undo-button");
    const clearButton = document.getElementById("clear-button");
    const zoomInButton = document.getElementById("zoom-in-button");
    const zoomOutButton = document.getElementById("zoom-out-button");
    const fitButton = document.getElementById("fit-button");
    const actualSizeButton = document.getElementById("actual-size-button");
    const focusButton = document.getElementById("focus-button");

    const image = new Image();
    image.src = INITIAL_STATE.imageUrl;

    let points = loadStoredPoints();
    let imageReady = false;
    let saving = false;
    let viewScale = 1;
    let viewOffsetX = 0;
    let viewOffsetY = 0;
    let fitScale = 1;
    let activePointerId = null;
    let dragMode = null;
    let draggedPointIndex = null;
    let panStart = null;
    let hoverPoint = null;

    image.addEventListener("load", () => {
      imageReady = true;
      fitImageToView();
      if (points.length === 4) {
        focusPoints();
      }
      setStatus("Image ready", "Zoom in and click the four scoreboard corners.", "info");
      updateUI();
      render();
    });

    image.addEventListener("error", () => {
      setStatus("Image failed to load", "Refresh the page or rerun the selector.", "error");
    });

    window.addEventListener("resize", () => {
      if (!imageReady) {
        resizeCanvas();
        render();
        return;
      }
      const previousCenter = screenToImage(canvas.clientWidth / 2, canvas.clientHeight / 2);
      resizeCanvas();
      viewOffsetX = canvas.clientWidth / 2 - previousCenter.x * viewScale;
      viewOffsetY = canvas.clientHeight / 2 - previousCenter.y * viewScale;
      render();
    });

    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", resetPointerState);
    canvas.addEventListener("pointerleave", () => {
      hoverPoint = null;
      render();
    });

    zoomInButton.addEventListener("click", () => zoomAround(canvas.clientWidth / 2, canvas.clientHeight / 2, 1.25));
    zoomOutButton.addEventListener("click", () => zoomAround(canvas.clientWidth / 2, canvas.clientHeight / 2, 0.8));
    fitButton.addEventListener("click", () => {
      fitImageToView();
      render();
    });
    actualSizeButton.addEventListener("click", () => {
      setActualSize();
      render();
    });
    focusButton.addEventListener("click", () => {
      if (points.length > 0) {
        focusPoints();
        render();
      }
    });
    undoButton.addEventListener("click", () => {
      if (saving || points.length === 0) {
        return;
      }
      points.pop();
      saveDraft();
      updateUI();
      render();
      setStatus("Last point removed", "Click to place it again or drag another point into place.", "info");
    });
    clearButton.addEventListener("click", () => {
      if (saving || points.length === 0) {
        return;
      }
      points = [];
      saveDraft();
      updateUI();
      render();
      setStatus("Selection cleared", "All four points have been removed.", "info");
    });
    noneButton.addEventListener("click", async () => {
      if (saving) {
        return;
      }
      const confirmed = window.confirm("Save this frame as having no visible scoreboard?");
      if (!confirmed) {
        return;
      }
      await submitSelection("none");
    });
    saveButton.addEventListener("click", async () => {
      if (saving) {
        return;
      }
      if (points.length !== 4) {
        setStatus("Four corners required", "Place exactly four points before saving.", "error");
        return;
      }
      await submitSelection("save");
    });

    resizeCanvas();
    updateUI();
    render();

    function loadStoredPoints() {
      const fallback = normalizePointList(INITIAL_STATE.initialPoints);
      try {
        const raw = window.localStorage.getItem(INITIAL_STATE.draftStorageKey);
        if (!raw) {
          return fallback;
        }
        const parsed = JSON.parse(raw);
        const normalized = normalizePointList(parsed);
        return normalized.length > 0 ? normalized : fallback;
      } catch (error) {
        return fallback;
      }
    }

    function normalizePointList(input) {
      if (!Array.isArray(input)) {
        return [];
      }
      const normalized = [];
      for (const point of input) {
        if (!Array.isArray(point) || point.length !== 2) {
          continue;
        }
        const x = clamp(Math.round(Number(point[0])), 0, INITIAL_STATE.imageWidth - 1);
        const y = clamp(Math.round(Number(point[1])), 0, INITIAL_STATE.imageHeight - 1);
        if (Number.isFinite(x) && Number.isFinite(y)) {
          normalized.push({ x, y });
        }
      }
      return normalized.slice(0, 4);
    }

    function saveDraft() {
      try {
        const serialized = points.map((point) => [point.x, point.y]);
        window.localStorage.setItem(INITIAL_STATE.draftStorageKey, JSON.stringify(serialized));
      } catch (error) {
        console.warn("Could not store draft points.", error);
      }
    }

    function clearDraft() {
      try {
        window.localStorage.removeItem(INITIAL_STATE.draftStorageKey);
      } catch (error) {
        console.warn("Could not clear draft points.", error);
      }
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function fitImageToView() {
      resizeCanvas();
      const paddedWidth = Math.max(canvas.clientWidth - 36, 120);
      const paddedHeight = Math.max(canvas.clientHeight - 36, 120);
      fitScale = Math.min(
        paddedWidth / INITIAL_STATE.imageWidth,
        paddedHeight / INITIAL_STATE.imageHeight
      );
      if (!Number.isFinite(fitScale) || fitScale <= 0) {
        fitScale = 1;
      }
      viewScale = fitScale;
      viewOffsetX = (canvas.clientWidth - INITIAL_STATE.imageWidth * viewScale) / 2;
      viewOffsetY = (canvas.clientHeight - INITIAL_STATE.imageHeight * viewScale) / 2;
    }

    function setActualSize() {
      resizeCanvas();
      viewScale = 1;
      viewOffsetX = (canvas.clientWidth - INITIAL_STATE.imageWidth) / 2;
      viewOffsetY = (canvas.clientHeight - INITIAL_STATE.imageHeight) / 2;
    }

    function focusPoints() {
      resizeCanvas();
      if (points.length === 0) {
        fitImageToView();
        return;
      }
      const bounds = getPointBounds(points);
      const padding = 110;
      const width = Math.max(bounds.maxX - bounds.minX, 22);
      const height = Math.max(bounds.maxY - bounds.minY, 22);
      const scaleX = Math.max((canvas.clientWidth - padding) / width, fitScale);
      const scaleY = Math.max((canvas.clientHeight - padding) / height, fitScale);
      viewScale = clamp(Math.min(scaleX, scaleY), fitScale, 24);
      const centerX = (bounds.minX + bounds.maxX) / 2;
      const centerY = (bounds.minY + bounds.maxY) / 2;
      viewOffsetX = canvas.clientWidth / 2 - centerX * viewScale;
      viewOffsetY = canvas.clientHeight / 2 - centerY * viewScale;
    }

    function onWheel(event) {
      if (!imageReady) {
        return;
      }
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.14 : 0.88;
      const canvasPoint = getCanvasCoordinates(event);
      zoomAround(canvasPoint.x, canvasPoint.y, factor);
    }

    function zoomAround(screenX, screenY, factor) {
      const anchor = screenToImage(screenX, screenY);
      const nextScale = clamp(viewScale * factor, Math.min(fitScale * 0.5, 0.25), 30);
      viewScale = nextScale;
      viewOffsetX = screenX - anchor.x * viewScale;
      viewOffsetY = screenY - anchor.y * viewScale;
      render();
    }

    function onPointerDown(event) {
      if (!imageReady || saving) {
        return;
      }
      activePointerId = event.pointerId;
      canvas.setPointerCapture(event.pointerId);
      const canvasPoint = getCanvasCoordinates(event);
      const hitIndex = findPointNear(canvasPoint.x, canvasPoint.y);
      if (hitIndex >= 0) {
        dragMode = "point";
        draggedPointIndex = hitIndex;
      } else {
        dragMode = "pan";
        panStart = {
          x: canvasPoint.x,
          y: canvasPoint.y,
          offsetX: viewOffsetX,
          offsetY: viewOffsetY,
          moved: false,
        };
      }
    }

    function onPointerMove(event) {
      const canvasPoint = getCanvasCoordinates(event);
      if (imageReady) {
        hoverPoint = clampPoint(screenToImage(canvasPoint.x, canvasPoint.y));
      }

      if (activePointerId !== event.pointerId || saving) {
        render();
        return;
      }

      if (dragMode === "point" && draggedPointIndex !== null) {
        points[draggedPointIndex] = clampPoint(screenToImage(canvasPoint.x, canvasPoint.y));
        saveDraft();
        updateUI();
        render();
        return;
      }

      if (dragMode === "pan" && panStart) {
        const dx = canvasPoint.x - panStart.x;
        const dy = canvasPoint.y - panStart.y;
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
          panStart.moved = true;
        }
        viewOffsetX = panStart.offsetX + dx;
        viewOffsetY = panStart.offsetY + dy;
        render();
        return;
      }

      render();
    }

    function onPointerUp(event) {
      if (activePointerId !== event.pointerId) {
        return;
      }
      const canvasPoint = getCanvasCoordinates(event);
      if (dragMode === "pan" && panStart && !panStart.moved) {
        maybeAddPoint(screenToImage(canvasPoint.x, canvasPoint.y));
      }
      resetPointerState();
    }

    function resetPointerState() {
      activePointerId = null;
      dragMode = null;
      draggedPointIndex = null;
      panStart = null;
      render();
    }

    function maybeAddPoint(rawPoint) {
      if (points.length >= 4) {
        setStatus("Already have four points", "Drag a point to adjust it, or use Undo/Clear to make changes.", "info");
        return;
      }
      points.push(clampPoint(rawPoint));
      saveDraft();
      updateUI();
      render();
      if (points.length === 4) {
        setStatus("Ready to save", "All four corners are set. Save the selection when it looks right.", "success");
      } else {
        setStatus("Point added", `${4 - points.length} corner${points.length === 3 ? "" : "s"} left to place.`, "info");
      }
    }

    function render() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);

      ctx.fillStyle = "#0a1620";
      ctx.fillRect(0, 0, width, height);

      if (!imageReady) {
        drawCenteredText("Loading image...");
        return;
      }

      ctx.save();
      ctx.translate(viewOffsetX, viewOffsetY);
      ctx.scale(viewScale, viewScale);
      ctx.drawImage(image, 0, 0, INITIAL_STATE.imageWidth, INITIAL_STATE.imageHeight);
      ctx.restore();

      const polygonPoints = points.length === 4 ? orderPointsClockwise(points) : points;
      if (polygonPoints.length >= 2) {
        ctx.save();
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(145, 231, 255, 0.95)";
        ctx.beginPath();
        polygonPoints.forEach((point, index) => {
          const screen = imageToScreen(point);
          if (index === 0) {
            ctx.moveTo(screen.x, screen.y);
          } else {
            ctx.lineTo(screen.x, screen.y);
          }
        });
        if (polygonPoints.length === 4) {
          const first = imageToScreen(polygonPoints[0]);
          ctx.lineTo(first.x, first.y);
        }
        ctx.stroke();
        ctx.restore();
      }

      points.forEach((point, index) => {
        const screen = imageToScreen(point);
        ctx.beginPath();
        ctx.fillStyle = "#ff5b6f";
        ctx.arc(screen.x, screen.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 12px Avenir Next, Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(index + 1), screen.x, screen.y);
      });

      if (hoverPoint) {
        ctx.save();
        ctx.fillStyle = "rgba(5, 16, 24, 0.82)";
        ctx.strokeStyle = "rgba(159, 231, 255, 0.25)";
        ctx.lineWidth = 1;
        const label = `x=${hoverPoint.x}, y=${hoverPoint.y}`;
        ctx.font = "600 12px Avenir Next, Segoe UI, sans-serif";
        const boxWidth = ctx.measureText(label).width + 16;
        const boxX = Math.min(width - boxWidth - 12, 14);
        const boxY = 14;
        roundRect(ctx, boxX, boxY, boxWidth, 30, 12);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#dff7ff";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, boxX + 8, boxY + 15);
        ctx.restore();
      }
    }

    function drawCenteredText(message) {
      ctx.fillStyle = "#d3edf8";
      ctx.font = "600 18px Avenir Next, Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(message, canvas.clientWidth / 2, canvas.clientHeight / 2);
    }

    function roundRect(context, x, y, width, height, radius) {
      context.beginPath();
      context.moveTo(x + radius, y);
      context.arcTo(x + width, y, x + width, y + height, radius);
      context.arcTo(x + width, y + height, x, y + height, radius);
      context.arcTo(x, y + height, x, y, radius);
      context.arcTo(x, y, x + width, y, radius);
      context.closePath();
    }

    function updateUI() {
      selectionCount.textContent = `${points.length} / 4 points`;
      saveButton.disabled = saving || points.length !== 4;
      noneButton.disabled = saving;
      undoButton.disabled = saving || points.length === 0;
      clearButton.disabled = saving || points.length === 0;
      focusButton.disabled = points.length === 0;

      const items = pointList.querySelectorAll(".point-item");
      items.forEach((item, index) => {
        const value = item.querySelector(".point-value");
        const point = points[index];
        value.textContent = point ? `(${point.x}, ${point.y})` : "Not set";
      });
    }

    function setStatus(title, message, tone) {
      statusBox.dataset.tone = tone;
      statusTitle.textContent = title;
      statusMessage.textContent = message;
    }

    async function submitSelection(action) {
      saving = true;
      updateUI();
      setStatus(
        action === "save" ? "Saving selection" : "Saving no-scoreboard selection",
        "Please wait while the selector hands control back to hmtrack.",
        "info"
      );

      try {
        const response = await fetch("/api/complete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action,
            points: points.map((point) => [point.x, point.y]),
          }),
        });
        const body = await response.text();
        if (!response.ok) {
          let errorMessage = body;
          try {
            const parsed = JSON.parse(body);
            errorMessage = parsed.error || errorMessage;
          } catch (error) {
          }
          throw new Error(errorMessage);
        }
        clearDraft();
        document.open();
        document.write(body);
        document.close();
      } catch (error) {
        saving = false;
        updateUI();
        setStatus("Could not save", error.message || "Unknown selector failure.", "error");
      }
    }

    function orderPointsClockwise(pointList) {
      const pointsCopy = pointList.map((point) => ({ x: point.x, y: point.y }));
      if (pointsCopy.length !== 4) {
        return pointsCopy;
      }

      const sums = pointsCopy.map((point) => point.x + point.y);
      const diffs = pointsCopy.map((point) => point.y - point.x);

      const ordered = new Array(4);
      ordered[0] = pointsCopy[indexOfMin(sums)];
      ordered[2] = pointsCopy[indexOfMax(sums)];
      ordered[1] = pointsCopy[indexOfMin(diffs)];
      ordered[3] = pointsCopy[indexOfMax(diffs)];
      return ordered;
    }

    function getPointBounds(pointList) {
      return pointList.reduce(
        (bounds, point) => ({
          minX: Math.min(bounds.minX, point.x),
          minY: Math.min(bounds.minY, point.y),
          maxX: Math.max(bounds.maxX, point.x),
          maxY: Math.max(bounds.maxY, point.y),
        }),
        { minX: pointList[0].x, minY: pointList[0].y, maxX: pointList[0].x, maxY: pointList[0].y }
      );
    }

    function imageToScreen(point) {
      return {
        x: point.x * viewScale + viewOffsetX,
        y: point.y * viewScale + viewOffsetY,
      };
    }

    function screenToImage(screenX, screenY) {
      return {
        x: (screenX - viewOffsetX) / viewScale,
        y: (screenY - viewOffsetY) / viewScale,
      };
    }

    function clampPoint(point) {
      return {
        x: clamp(Math.round(point.x), 0, INITIAL_STATE.imageWidth - 1),
        y: clamp(Math.round(point.y), 0, INITIAL_STATE.imageHeight - 1),
      };
    }

    function getCanvasCoordinates(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    }

    function findPointNear(screenX, screenY) {
      for (let index = points.length - 1; index >= 0; index -= 1) {
        const screenPoint = imageToScreen(points[index]);
        const distance = Math.hypot(screenPoint.x - screenX, screenPoint.y - screenY);
        if (distance <= 14) {
          return index;
        }
      }
      return -1;
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function indexOfMin(values) {
      return values.reduce((best, value, index, array) => (value < array[best] ? index : best), 0);
    }

    function indexOfMax(values) {
      return values.reduce((best, value, index, array) => (value > array[best] ? index : best), 0);
    }
  </script>
</body>
</html>
"""
        return page.replace("__STATE_JSON__", state_json).replace(
            "__GAME_ID__", html.escape(self._game_id)
        )

    def _build_completion_page(self) -> str:
        page = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thank you for playing!</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      overflow: hidden;
      color: #f3fbff;
      background:
        linear-gradient(135deg, rgba(4, 16, 24, 0.68), rgba(9, 39, 56, 0.48)),
        url("__BACKGROUND_URL__") center / cover no-repeat;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(circle at top, rgba(255, 255, 255, 0.22), transparent 28%),
        linear-gradient(180deg, rgba(2, 8, 12, 0.12), rgba(2, 8, 12, 0.54));
      pointer-events: none;
    }

    .card {
      position: relative;
      z-index: 1;
      max-width: min(42rem, calc(100vw - 32px));
      padding: clamp(28px, 4vw, 42px);
      border-radius: 28px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: rgba(5, 16, 25, 0.56);
      box-shadow: 0 28px 64px rgba(0, 0, 0, 0.34);
      backdrop-filter: blur(12px);
      text-align: center;
    }

    .eyebrow {
      margin: 0 0 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #bcecff;
      font-size: 0.84rem;
      font-weight: 700;
    }

    h1 {
      margin: 0;
      font-size: clamp(2.2rem, 6vw, 4.8rem);
      line-height: 0.95;
      text-wrap: balance;
    }

    p {
      margin: 18px auto 0;
      max-width: 30rem;
      color: rgba(239, 250, 255, 0.9);
      font-size: 1.05rem;
      line-height: 1.55;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-top: 22px;
      padding: 12px 18px;
      border-radius: 999px;
      background: rgba(121, 240, 176, 0.16);
      border: 1px solid rgba(121, 240, 176, 0.32);
      color: #ddffeb;
      font-weight: 700;
    }
  </style>
</head>
<body>
  <main class="card">
    <p class="eyebrow">Scoreboard Saved</p>
    <h1>Thank you for playing!</h1>
    <p>
      Your selection is in. hmtrack can keep rolling now, and you can close this tab whenever you like.
    </p>
    <div class="tag">Selection complete</div>
  </main>
</body>
</html>
"""
        return page.replace("__BACKGROUND_URL__", self._done_background_data_url)


def parse_points(points_str_list: List[str]) -> Optional[List[Tuple[int, int]]]:
    if len(points_str_list) != 4:
        print("Error: Exactly four points must be provided.")
        return None
    points: List[Tuple[int, int]] = []
    for point_str in points_str_list:
        try:
            x_str, y_str = point_str.split(",")
            x = int(x_str)
            y = int(y_str)
            points.append((x, y))
        except ValueError:
            print(f"Error parsing point '{point_str}'. Points must be in the format x,y.")
            return None
    return points


def _untuple_points(points: List[Tuple[int, int]]) -> List[List[int]]:
    results: List[List[int]] = []
    for pt in points:
        assert len(pt) == 2
        results.append([int(pt[0]), int(pt[1])])
    return results


def configure_scoreboard(
    game_id: str,
    image: Optional[torch.Tensor] = None,
    force: bool = False,
    max_display_height: Optional[int] = None,
    bind_host: Optional[str] = None,
    port: int = 0,
    open_browser: Optional[bool] = None,
) -> List[List[int]]:
    assert game_id
    game_config = get_game_config(game_id=game_id)
    current_scoreboard = get_nested_value(game_config, "rink.scoreboard.perspective_polygon")
    if current_scoreboard and not force:
        return current_scoreboard

    if image is None:
        game_dir = get_game_dir(game_id=game_id)
        image_file = os.path.join(game_dir, "s.png")
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Could not find image file: {image_file}")
        image_context = Image.open(image_file)
    else:
        image_context = contextlib.nullcontext(image)
    with image_context as source:
        selector = ScoreboardSelector(
            image=source,
            initial_points=current_scoreboard,
            max_display_height=max_display_height,
            game_id=game_id,
            bind_host=bind_host,
            port=port,
            open_browser=open_browser,
        )
    selector.run()
    current_scoreboard = selector.points
    current_scoreboard = _untuple_points(current_scoreboard)
    set_nested_value(game_config, "rink.scoreboard.perspective_polygon", current_scoreboard)
    save_private_config(game_id=game_id, data=game_config, verbose=True)
    return current_scoreboard


def main() -> None:
    selector_args, remaining_args = _parse_selector_cli_args(sys.argv[1:])
    opts = hm_opts()
    args = opts.parse(remaining_args)

    configure_scoreboard(
        game_id=args.game_id,
        force=True,
        bind_host=selector_args.bind_host,
        port=selector_args.port,
        open_browser=selector_args.open_browser,
    )


if __name__ == "__main__":
    main()
