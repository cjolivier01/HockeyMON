"""Browser preview and live-stream helpers for headless environments."""

from __future__ import annotations

import contextlib
import html
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from hmlib.ui.display_env import env_flag, has_local_display_env, sanitize_display_env_for_cv2

sanitize_display_env_for_cv2()

import numpy as np
import torch

from hmlib.log import get_root_logger
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.utils.image import make_visible_image
from hmlib.utils.torch_backend import is_rocm_backend
from hmlib.video.ffmpeg_mux_cmd import (
    build_ffmpeg_live_bitstream_publish_cmd,
    build_ffmpeg_live_hls_bitstream_publish_cmd,
)

DEFAULT_BROWSER_PREVIEW_MAX_WIDTH = 1920
DEFAULT_BROWSER_PREVIEW_MAX_HEIGHT = 1080
HLS_JS_URL = "https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js"
HLS_JS_INTEGRITY = "sha384-9v3HcdYrO3D+OPDTjZ40RXocgE4GtXVCd3/mCS62JsM93JXgI1afJVuwjFvsu6ni"


def has_local_display() -> bool:
    """Return True when an interactive display is available."""
    if env_flag("HM_FORCE_HEADLESS_PREVIEW"):
        return False
    return has_local_display_env()


def _best_public_host() -> str | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
        finally:
            sock.close()
        if host and not host.startswith("127."):
            return host
    except OSError:
        return None
    return None


def _normalize_preview_frame(
    img: torch.Tensor | np.ndarray | StreamTensorBase,
    *,
    show_scaled: float | None = None,
) -> np.ndarray:
    visible = make_visible_image(
        unwrap_tensor(img),
        enable_resizing=show_scaled,
        force_numpy=True,
    )
    if visible.ndim == 4:
        if visible.shape[0] != 1:
            raise ValueError(
                f"Expected a single frame for preview publishing, got batch shape={visible.shape}"
            )
        visible = visible[0]
    if visible.dtype != np.uint8:
        visible = visible.astype(np.uint8)
    return np.ascontiguousarray(visible)


def _infer_frame_height_width(
    img: torch.Tensor | np.ndarray | StreamTensorBase,
) -> tuple[int, int] | None:
    frame = unwrap_tensor(img)
    shape = getattr(frame, "shape", None)
    if shape is None:
        return None
    dims = tuple(int(dim) for dim in shape)
    if len(dims) < 2:
        return None
    if len(dims) == 2:
        height, width = dims
        return (height, width) if height > 0 and width > 0 else None
    if dims[-1] in {1, 3, 4}:
        height, width = dims[-3], dims[-2]
    elif dims[-3] in {1, 3, 4}:
        height, width = dims[-2], dims[-1]
    else:
        return None
    return (height, width) if height > 0 and width > 0 else None


def _browser_preview_auto_scale(
    img: torch.Tensor | np.ndarray | StreamTensorBase,
    *,
    max_width: int = DEFAULT_BROWSER_PREVIEW_MAX_WIDTH,
    max_height: int = DEFAULT_BROWSER_PREVIEW_MAX_HEIGHT,
) -> float | None:
    shape = _infer_frame_height_width(img)
    if shape is None:
        return None
    height, width = shape
    if width <= max_width and height <= max_height:
        return None
    return min(float(max_width) / float(width), float(max_height) / float(height), 1.0)


def _ensure_even_video_frame(frame: torch.Tensor) -> torch.Tensor:
    """Crop preview frames to even dimensions for YUV420/NVENC compatibility."""
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame for video publishing, got shape={frame.shape}")
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height <= 0 or even_width <= 0:
        raise ValueError(f"Preview frame became too small after even-dimension crop: {frame.shape}")
    if even_height == height and even_width == width:
        return frame
    return frame[:even_height, :even_width, ...]


def _prof_ctx(profiler: Any, name: str):
    if profiler is not None and getattr(profiler, "enabled", False):
        return profiler.rf(name)
    return contextlib.nullcontext()


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, request_handler_class, preview: BrowserPreviewServer):
        super().__init__(server_address, request_handler_class)
        self.preview = preview


class _PreviewRequestHandler(BaseHTTPRequestHandler):
    server: _PreviewHttpServer

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        try:
            preview = self.server.preview
            preview._note_client_activity()
            path = urlsplit(self.path).path or "/"
            if path in ("", "/"):
                self._serve_index()
                return
            if self._serve_output_file(path.lstrip("/")):
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_index(self) -> None:
        title = html.escape(self.server.preview.label)
        manifest_path = "/" + self.server.preview.manifest_name
        body = (
            "<!doctype html><html><head>"
            f"<title>{title}</title>"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<style>"
            "body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
            "header{padding:12px 16px;font-size:14px;background:#1a1a1a;}"
            "main{display:flex;justify-content:center;align-items:center;min-height:calc(100vh - 48px);padding:16px;}"
            ".wrap{width:min(100%,1280px);}"
            "video{width:100%;max-height:calc(100vh - 120px);background:#000;}"
            "p{font-size:13px;line-height:1.5;}"
            "a{color:#9cd5ff;}"
            "</style>"
            "</head><body>"
            f'<header>{title} preview | <a href="{manifest_path}">HLS manifest</a></header>'
            '<main><div class="wrap">'
            '<video id="preview" controls autoplay muted playsinline></video>'
            '<p id="status">Waiting for preview data...</p>'
            "</div></main>"
            "<script>"
            f"const manifestUrl={manifest_path!r};"
            f"const hlsScriptUrl={HLS_JS_URL!r};"
            f"const hlsScriptIntegrity={HLS_JS_INTEGRITY!r};"
            "const video=document.getElementById('preview');"
            "const status=document.getElementById('status');"
            "let hls=null;"
            "function sleep(ms){return new Promise(resolve=>setTimeout(resolve,ms));}"
            "async function manifestReady(){try{"
            "const response=await fetch(manifestUrl,{cache:'no-store'});"
            "if(!response.ok){return false;}"
            "const text=await response.text();"
            "return text.includes('#EXTM3U');"
            "}catch(_){return false;}}"
            "async function waitForManifest(){status.textContent='Waiting for preview stream...';"
            "while(!(await manifestReady())){await sleep(1000);}}"
            "function attachNative(){video.src=manifestUrl;video.play().catch(()=>{});"
            "status.textContent='Loading live preview...';}"
            "function loadHlsScript(){if(window.Hls){return Promise.resolve();}"
            "return new Promise((resolve,reject)=>{"
            "const script=document.createElement('script');"
            "script.src=hlsScriptUrl;script.integrity=hlsScriptIntegrity;"
            "script.crossOrigin='anonymous';"
            "script.onload=resolve;script.onerror=reject;document.head.appendChild(script);"
            "});}"
            "async function attachHls(){try{await loadHlsScript();}catch(_){"
            "return false;}"
            "if(!(window.Hls&&window.Hls.isSupported())){"
            "return false;}"
            "hls=new window.Hls({liveDurationInfinity:true,backBufferLength:0});"
            "hls.loadSource(manifestUrl);hls.attachMedia(video);"
            "hls.on(window.Hls.Events.MANIFEST_PARSED,()=>{video.play().catch(()=>{});});"
            "hls.on(window.Hls.Events.ERROR,(_,data)=>{if(data&&data.fatal){"
            "status.textContent='Preview interrupted; reconnecting...';"
            "try{hls.destroy();}catch(_){};setTimeout(startPreview,1000);"
            "}});"
            "status.textContent='Loading live preview...';"
            "return true;"
            "}"
            "async function startPreview(){await waitForManifest();"
            "if(await attachHls()){return;}"
            "if(video.canPlayType('application/vnd.apple.mpegurl')){attachNative();}"
            "else{status.innerHTML='Browser HLS playback is unavailable. Open <a href=\"'+manifestUrl+'\">the HLS manifest</a> in an HLS-capable player.';}}"
            "startPreview();"
            "</script>"
            "</body></html>"
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_output_file(self, relative_name: str) -> bool:
        preview = self.server.preview
        output_path = preview.resolve_output_path(relative_name)
        if output_path is None:
            return False
        ready_path = preview.wait_for_output(relative_name, timeout=5.0)
        if ready_path is None or not ready_path.exists() or ready_path.stat().st_size <= 0:
            self.send_error(HTTPStatus.NOT_FOUND, "Preview stream is not ready")
            return True
        content_type = {
            ".m3u8": "application/vnd.apple.mpegurl",
            ".ts": "video/mp2t",
            ".m4s": "video/iso.segment",
            ".mp4": "video/mp4",
        }.get(ready_path.suffix.lower(), "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(ready_path.stat().st_size))
        self.end_headers()
        with open(ready_path, "rb") as stream:
            shutil.copyfileobj(stream, self.wfile)
        return True


class BrowserPreviewServer:
    """Tiny threaded HLS server for browser-based live previews."""

    def __init__(
        self,
        label: str,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        jpeg_quality: int = 80,
        logger: Any = None,
        profiler: Any = None,
    ) -> None:
        self.label = str(label or "Preview")
        self._host = str(host or "0.0.0.0")
        self._port = int(port or 0)
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._active_client_window_seconds = 10.0
        self._last_client_activity: float | None = None
        self._server: _PreviewHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._publisher: Any | None = None
        self._output_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
        self._output_dir: Path | None = None
        self._manifest_name = "stream.m3u8"
        self._segment_pattern = "preview-%06d.ts"
        self._announced = False
        self._auto_scale_announced = False

    @property
    def closed(self) -> bool:
        return self._server is None

    @property
    def port(self) -> int:
        return 0 if self._server is None else int(self._server.server_port)

    @property
    def urls(self) -> list[str]:
        return self._candidate_urls()

    @property
    def preferred_url(self) -> str | None:
        urls = self.urls
        for url in urls:
            host = urlsplit(url).hostname or ""
            if host not in {"127.0.0.1", "localhost"}:
                return url
        return urls[0] if urls else None

    @property
    def manifest_name(self) -> str:
        return self._manifest_name

    @property
    def active_client_count(self) -> int:
        return 1 if self.has_active_clients() else 0

    def has_active_clients(self) -> bool:
        with self._condition:
            if self._last_client_activity is None:
                return False
            return (
                time.monotonic() - float(self._last_client_activity)
            ) <= self._active_client_window_seconds

    def _client_connected(self) -> None:
        self._note_client_activity()

    def _client_disconnected(self) -> None:
        self._note_client_activity()

    def _note_client_activity(self) -> None:
        with self._condition:
            self._last_client_activity = time.monotonic()
            self._condition.notify_all()

    def _candidate_urls(self) -> list[str]:
        port = self.port
        if port <= 0:
            return []
        urls: list[str] = [f"http://127.0.0.1:{port}/"]
        if self._host not in {"0.0.0.0", "::", ""}:
            urls.append(f"http://{self._host}:{port}/")
        public_host = _best_public_host()
        if public_host:
            urls.append(f"http://{public_host}:{port}/")
        deduped: list[str] = []
        for url in urls:
            if url not in deduped:
                deduped.append(url)
        return deduped

    def start(self) -> None:
        if self._server is not None:
            return
        if self._output_dir_ctx is None:
            self._output_dir_ctx = tempfile.TemporaryDirectory(prefix="hm-headless-preview-")
            self._output_dir = Path(self._output_dir_ctx.name)
        self._server = _PreviewHttpServer((self._host, self._port), _PreviewRequestHandler, self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if not self._announced:
            urls = self._candidate_urls()
            if urls:
                self._logger.info(
                    "Headless preview for %s is available at: %s",
                    self.label,
                    ", ".join(urls),
                )
            self._announced = True

    def _select_publisher(self, img: torch.Tensor | np.ndarray | StreamTensorBase) -> Any:
        if self._output_dir is None:
            raise RuntimeError("Preview output directory is not initialized")
        frame = unwrap_tensor(img)
        if isinstance(frame, torch.Tensor) and frame.device.type == "cuda":
            try:
                if is_rocm_backend():
                    return _AmdHlsPreviewPublisher(
                        output_dir=self._output_dir,
                        manifest_name=self._manifest_name,
                        segment_pattern=self._segment_pattern,
                        label=self.label,
                        logger=self._logger,
                        profiler=self._profiler,
                    )
                return _NvencHlsPreviewPublisher(
                    output_dir=self._output_dir,
                    manifest_name=self._manifest_name,
                    segment_pattern=self._segment_pattern,
                    label=self.label,
                    logger=self._logger,
                    profiler=self._profiler,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self._logger.warning(
                    "Falling back to raw ffmpeg browser preview for %s because NVENC preview "
                    "initialization failed: %s",
                    self.label,
                    exc,
                )
        return _RawHlsPreviewPublisher(
            output_dir=self._output_dir,
            manifest_name=self._manifest_name,
            segment_pattern=self._segment_pattern,
            label=self.label,
            logger=self._logger,
            profiler=self._profiler,
        )

    def _effective_show_scale(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        show_scaled: float | None,
    ) -> float | None:
        if show_scaled is not None:
            return show_scaled
        scale = _browser_preview_auto_scale(img)
        if scale is None:
            return None
        if not self._auto_scale_announced:
            shape = _infer_frame_height_width(img)
            if shape is not None:
                height, width = shape
                self._logger.info(
                    "Downscaling headless browser preview for %s from %dx%d to fit "
                    "%dx%d (scale %.3f). Saved video output is unchanged; pass "
                    "--show-scaled to override the preview scale.",
                    self.label,
                    width,
                    height,
                    DEFAULT_BROWSER_PREVIEW_MAX_WIDTH,
                    DEFAULT_BROWSER_PREVIEW_MAX_HEIGHT,
                    scale,
                )
            self._auto_scale_announced = True
        return scale

    def publish(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
        require_clients: bool = False,
    ) -> bool:
        if self._server is None:
            self.start()
        if require_clients and not self.has_active_clients():
            return False
        if self._publisher is None:
            self._publisher = self._select_publisher(img)
        effective_show_scaled = self._effective_show_scale(img, show_scaled)
        with _prof_ctx(self._profiler, "headless_preview.publish"):
            self._publisher.write_frame(img, show_scaled=effective_show_scaled)
        return True

    def resolve_output_path(self, relative_name: str) -> Path | None:
        if self._output_dir is None:
            return None
        safe_name = Path(relative_name).name
        if not safe_name or safe_name != relative_name:
            return None
        if safe_name == self._manifest_name:
            return self._output_dir / safe_name
        if safe_name == "init.mp4":
            return self._output_dir / safe_name
        if safe_name.startswith("preview-") and Path(safe_name).suffix.lower() in {
            ".ts",
            ".m4s",
            ".mp4",
        }:
            return self._output_dir / safe_name
        return None

    def wait_for_output(self, relative_name: str, timeout: float = 5.0) -> Path | None:
        output_path = self.resolve_output_path(relative_name)
        if output_path is None:
            return None
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if output_path.exists():
                try:
                    if output_path.stat().st_size > 0:
                        return output_path
                except OSError:
                    pass
            if self.closed:
                break
            time.sleep(0.05)
        if output_path.exists():
            return output_path
        return None

    def close(self) -> None:
        publisher = self._publisher
        self._publisher = None
        if publisher is not None:
            publisher.close()
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
        with self._condition:
            self._condition.notify_all()
        output_dir_ctx = self._output_dir_ctx
        self._output_dir_ctx = None
        self._output_dir = None
        if output_dir_ctx is not None:
            output_dir_ctx.cleanup()


def mask_stream_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"rtmp", "rtmps"}:
        return url
    prefix, _, suffix = (parsed.path or "").rpartition("/")
    path = prefix + ("/***" if suffix else "/")
    host = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, host, path, "***" if parsed.query else "", ""))


def resolve_youtube_stream_url(base_url: str, stream_key: str | None) -> str:
    key = (stream_key or os.environ.get("HM_YOUTUBE_STREAM_KEY") or "").strip()
    if not key:
        key = (os.environ.get("YOUTUBE_STREAM_KEY") or "").strip()
    base = (base_url or "").strip()
    if not base:
        base = "rtmps://a.rtmps.youtube.com/live2"
    if key and not base.rstrip("/").endswith(key):
        return base.rstrip("/") + "/" + key
    return base


def validate_youtube_stream_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"rtmp", "rtmps"}:
        raise ValueError(f"YouTube live preview requires an rtmp:// or rtmps:// URL, got {url!r}")
    path = (parsed.path or "").strip("/")
    if not path or path.endswith(("live2", "live2/")):
        raise ValueError(
            "YouTube live preview requires a full publish URL with a stream key. "
            "Provide --youtube-stream-key or set HM_YOUTUBE_STREAM_KEY."
        )


class _HlsBitstreamMuxer:
    """Mux encoded H.264 packets into a rolling HLS playlist using ffmpeg copy mode."""

    def __init__(
        self,
        output_dir: Path,
        *,
        manifest_name: str,
        segment_pattern: str,
        fps: float,
        label: str,
        logger: Any = None,
        profiler: Any = None,
        hls_time: float = 1.0,
        hls_list_size: int = 6,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.manifest_name = str(manifest_name)
        self.segment_pattern = str(segment_pattern)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.label = str(label or "Preview")
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._hls_time = float(hls_time)
        self._hls_list_size = int(hls_list_size)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_name

    @property
    def segment_path_pattern(self) -> Path:
        return self.output_dir / self.segment_pattern

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _build_cmd(self) -> list[str]:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        return build_ffmpeg_live_hls_bitstream_publish_cmd(
            ffmpeg=ffmpeg,
            output_manifest_path=self.manifest_path,
            segment_path_pattern=self.segment_path_pattern,
            stream_format="h264",
            fps=self.fps,
            hls_time=self._hls_time,
            hls_list_size=self._hls_list_size,
        )

    def start(self) -> None:
        if self._process is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg is required for headless preview streaming")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with _prof_ctx(self._profiler, "headless_preview.start_muxer"):
            self._process = subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def write_packet(self, payload: bytes) -> None:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("HLS preview muxer is not open")
        try:
            with _prof_ctx(self._profiler, "headless_preview.packet_write"):
                process.stdin.write(payload)
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(f"HLS preview muxer failed for {self.label}\n{stderr_tail}") from exc

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        if process.returncode not in (0, None):
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"HLS preview muxer exited with code {process.returncode} for {self.label}\n"
                f"{stderr_tail}"
            )


class _NvencHlsPreviewPublisher:
    """Encode CUDA preview frames with NVENC and hand encoded packets to HLS muxing."""

    def __init__(
        self,
        output_dir: Path,
        *,
        manifest_name: str,
        segment_pattern: str,
        label: str,
        fps: float = 30.0,
        logger: Any = None,
        profiler: Any = None,
    ) -> None:
        self.label = str(label or "Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._encoder = None
        self._muxer = _HlsBitstreamMuxer(
            output_dir=output_dir,
            manifest_name=manifest_name,
            segment_pattern=segment_pattern,
            fps=self.fps,
            label=self.label,
            logger=self._logger,
            profiler=self._profiler,
        )

    def _ensure_open(self, frame: torch.Tensor) -> None:
        if self._encoder is not None:
            return
        from hmlib.video.py_nv_encoder import PyNvVideoEncoder

        if frame.device.type != "cuda":
            raise ValueError("NVENC browser preview requires CUDA frames")
        with _prof_ctx(self._profiler, "headless_preview.start_nvenc"):
            self._encoder = PyNvVideoEncoder(
                output_path=None,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=self.fps,
                codec="h264",
                preset="P2",
                device=frame.device,
                gpu_id=frame.device.index or 0,
                cuda_stream=torch.cuda.current_stream(frame.device).cuda_stream,
                bitstream_handler=self._muxer.write_packet,
                profiler=self._profiler,
            )
            self._encoder.open()

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        frame = unwrap_tensor(img)
        if not isinstance(frame, torch.Tensor) or frame.device.type != "cuda":
            raise ValueError("NVENC browser preview requires CUDA tensor frames")
        if frame.ndim != 3:
            raise ValueError(
                f"Expected single frame tensor with shape (H, W, C), got {frame.shape}"
            )
        with _prof_ctx(self._profiler, "headless_preview.prepare_cuda_frame"):
            frame = make_visible_image(frame, enable_resizing=show_scaled, force_numpy=False)
            if not isinstance(frame, torch.Tensor):
                raise TypeError("Expected make_visible_image to return a CUDA tensor for NVENC")
            frame = _ensure_even_video_frame(frame)
        self._ensure_open(frame)
        assert self._encoder is not None
        with _prof_ctx(self._profiler, "headless_preview.nvenc_encode"):
            self._encoder.write(frame)

    def close(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            encoder.close()
        self._muxer.close()


class _AmdHlsPreviewPublisher:
    """Encode ROCm preview frames and hand encoded bytes to HLS muxing."""

    def __init__(
        self,
        output_dir: Path,
        *,
        manifest_name: str,
        segment_pattern: str,
        label: str,
        fps: float = 30.0,
        logger: Any = None,
        profiler: Any = None,
    ) -> None:
        self.label = str(label or "Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._encoder = None
        self._muxer = _HlsBitstreamMuxer(
            output_dir=output_dir,
            manifest_name=manifest_name,
            segment_pattern=segment_pattern,
            fps=self.fps,
            label=self.label,
            logger=self._logger,
            profiler=self._profiler,
        )

    def _ensure_open(self, frame: torch.Tensor) -> None:
        if self._encoder is not None:
            return
        from hmlib.video.py_amd_codec import PyAmdVideoEncoder

        if frame.device.type != "cuda":
            raise ValueError("AMD browser preview requires ROCm/CUDA frames")
        with _prof_ctx(self._profiler, "headless_preview.start_amd"):
            self._encoder = PyAmdVideoEncoder(
                output_path=None,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=self.fps,
                codec="h264",
                device=frame.device,
                bitstream_handler=self._muxer.write_packet,
                profiler=self._profiler,
            )
            self._encoder.open()

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        frame = unwrap_tensor(img)
        if not isinstance(frame, torch.Tensor) or frame.device.type != "cuda":
            raise ValueError("AMD browser preview requires ROCm/CUDA tensor frames")
        if frame.ndim != 3:
            raise ValueError(
                f"Expected single frame tensor with shape (H, W, C), got {frame.shape}"
            )
        with _prof_ctx(self._profiler, "headless_preview.prepare_amd_frame"):
            frame = make_visible_image(frame, enable_resizing=show_scaled, force_numpy=False)
            if not isinstance(frame, torch.Tensor):
                raise TypeError(
                    "Expected make_visible_image to return a ROCm tensor for AMD encode"
                )
            frame = _ensure_even_video_frame(frame)
        self._ensure_open(frame)
        assert self._encoder is not None
        with _prof_ctx(self._profiler, "headless_preview.amd_encode"):
            self._encoder.write(frame)

    def close(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            encoder.close()
        self._muxer.close()


class _RawHlsPreviewPublisher:
    """Encode CPU preview frames into HLS through ffmpeg when NVENC is unavailable."""

    def __init__(
        self,
        output_dir: Path,
        *,
        manifest_name: str,
        segment_pattern: str,
        label: str,
        fps: float = 30.0,
        logger: Any = None,
        profiler: Any = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.manifest_name = str(manifest_name)
        self.segment_pattern = str(segment_pattern)
        self.label = str(label or "Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._width: int | None = None
        self._height: int | None = None

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_name

    @property
    def segment_path_pattern(self) -> Path:
        return self.output_dir / self.segment_pattern

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _build_cmd(self, width: int, height: int) -> list[str]:
        gop = max(1, round(self.fps * 0.5))
        return [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-progress",
            "pipe:2",
            "-nostats",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-f",
            "hls",
            "-hls_time",
            "1.0",
            "-hls_list_size",
            "6",
            "-hls_allow_cache",
            "0",
            "-hls_flags",
            "delete_segments+append_list+independent_segments+omit_endlist+temp_file",
            "-hls_segment_filename",
            str(self.segment_path_pattern),
            "-start_number",
            "0",
            str(self.manifest_path),
        ]

    def _ensure_open(self, width: int, height: int) -> None:
        if self._process is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg is required for headless preview streaming")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._width = int(width)
        self._height = int(height)
        with _prof_ctx(self._profiler, "headless_preview.start_raw_muxer"):
            self._process = subprocess.Popen(
                self._build_cmd(width=self._width, height=self._height),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        with _prof_ctx(self._profiler, "headless_preview.normalize"):
            frame = _normalize_preview_frame(img, show_scaled=show_scaled)
        self._ensure_open(width=frame.shape[1], height=frame.shape[0])
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Raw HLS preview publisher is not open")
        try:
            with _prof_ctx(self._profiler, "headless_preview.raw_write"):
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"Raw HLS preview publisher failed for {self.label}\n{stderr_tail}"
            ) from exc

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        if process.returncode not in (0, None):
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"Raw HLS preview publisher exited with code {process.returncode} for "
                f"{self.label}\n{stderr_tail}"
            )


def _live_transport(command: list[str], output_url: str, audio: dict) -> list[str]:
    """Use a real source audio stream when supplied and split RTP with tee."""
    audio_start = command.index("lavfi") - 1
    audio_end = audio_start + 4
    if audio.get("file"):
        inputs = []
        if audio.get("offset_seconds"):
            offset = float(audio["offset_seconds"])
            inputs += ["-ss", str(-offset)] if offset < 0 else ["-itsoffset", str(offset)]
        inputs += ["-i", str(audio["file"])]
        command[audio_start:audio_end] = inputs
        audio_end = audio_start + len(inputs)
        command[command.index("1:a:0")] = f"1:a:{int(audio.get('stream', 0))}"
    if urlsplit(output_url).scheme != "rtp":
        return command
    if audio.get("file") and audio.get("rtp_url"):
        command[-3:] = [
            "-f",
            "tee",
            (
                f"[select=v:f=rtp:payload_type=96]{output_url}|"
                f"[select=a:f=rtp:payload_type=97]{audio['rtp_url']}"
            ),
        ]
        return command
    # Video-only RTSP receives no synthetic silence; remove the second input.
    del command[audio_start:audio_end]
    audio_map = command.index("-map", command.index("-map") + 1)
    del command[audio_map : audio_map + 2]
    for option in ("-c:a", "-b:a", "-ar", "-ac"):
        index = command.index(option)
        del command[index : index + 2]
    command.remove("-shortest")
    command[-3:] = ["-an", "-f", "rtp", "-payload_type", "96", output_url]
    return command


class _RawVideoLivePublisher:
    """Pipe raw preview frames into ffmpeg for live RTMP(S) publishing."""

    def __init__(
        self,
        output_url: str,
        *,
        label: str,
        fps: float,
        logger: Any = None,
        profiler: Any = None,
        audio: dict | None = None,
    ) -> None:
        self.output_url = str(output_url)
        self.label = str(label or "Live Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._audio = audio or {}
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._width: int | None = None
        self._height: int | None = None

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _build_cmd(self, width: int, height: int) -> list[str]:
        gop = max(1, round(self.fps * 2.0))
        command = [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "flv",
            self.output_url,
        ]
        return _live_transport(command, self.output_url, self._audio)

    def _ensure_open(self, width: int, height: int) -> None:
        if self._process is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg is required for live preview streaming")
        self._width = int(width)
        self._height = int(height)
        cmd = self._build_cmd(width=self._width, height=self._height)
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()
        self._logger.info(
            "Publishing %s to %s",
            self.label,
            mask_stream_url(self.output_url),
        )

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        with _prof_ctx(self._profiler, "live_publish.normalize"):
            frame = _normalize_preview_frame(img, show_scaled=show_scaled)
        self._ensure_open(width=frame.shape[1], height=frame.shape[0])
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("ffmpeg live publisher is not open")
        try:
            with _prof_ctx(self._profiler, "live_publish.raw_write"):
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"ffmpeg live publisher failed for {mask_stream_url(self.output_url)}\n{stderr_tail}"
            ) from exc

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        if process.returncode not in (0, None):
            stderr_tail = "\n".join(self._stderr_tail)
            if any(
                marker in stderr_tail
                for marker in (
                    "Connection reset by peer",
                    "Broken pipe",
                    "Error writing trailer",
                )
            ):
                self._logger.warning(
                    "ffmpeg live publisher for %s ended after the remote peer closed the stream.\n%s",
                    mask_stream_url(self.output_url),
                    stderr_tail,
                )
                return
            raise RuntimeError(
                f"ffmpeg live publisher exited with code {process.returncode} for "
                f"{mask_stream_url(self.output_url)}\n{stderr_tail}"
            )


class _BitstreamLiveMuxer:
    """Mux encoded H.264 packets into RTMP(S) using ffmpeg copy mode."""

    def __init__(
        self,
        output_url: str,
        *,
        fps: float,
        label: str,
        logger: Any = None,
        profiler: Any = None,
        audio: dict | None = None,
    ) -> None:
        self.output_url = str(output_url)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.label = str(label or "Live Preview")
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._audio = audio or {}
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _build_cmd(self) -> list[str]:
        command = build_ffmpeg_live_bitstream_publish_cmd(
            ffmpeg=shutil.which("ffmpeg") or "ffmpeg",
            output_url=self.output_url,
            stream_format="h264",
            fps=self.fps,
        )
        if urlsplit(self.output_url).scheme == "rtp":
            command[1:1] = ["-probesize", "32", "-analyzeduration", "0"]
            # RTSP clients can attach after encoding starts. NVENC emits SPS/PPS
            # once; repeat the parsed headers at every IDR for late subscribers.
            command[command.index("-bsf:v") + 1] += ",dump_extra=freq=keyframe"
        return _live_transport(command, self.output_url, self._audio)

    def start(self) -> None:
        if self._process is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg is required for live preview streaming")
        with _prof_ctx(self._profiler, "live_publish.start_muxer"):
            self._process = subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()
        self._logger.info(
            "Publishing %s to %s",
            self.label,
            mask_stream_url(self.output_url),
        )

    def write_packet(self, payload: bytes) -> None:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("ffmpeg live muxer is not open")
        try:
            with _prof_ctx(self._profiler, "live_publish.packet_write"):
                process.stdin.write(payload)
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"ffmpeg live muxer failed for {mask_stream_url(self.output_url)}\n{stderr_tail}"
            ) from exc

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        if process.returncode not in (0, None):
            stderr_tail = "\n".join(self._stderr_tail)
            if any(
                marker in stderr_tail
                for marker in (
                    "Connection reset by peer",
                    "Broken pipe",
                    "Error writing trailer",
                )
            ):
                self._logger.warning(
                    "ffmpeg live muxer for %s ended after the remote peer closed the stream.\n%s",
                    mask_stream_url(self.output_url),
                    stderr_tail,
                )
                return
            raise RuntimeError(
                f"ffmpeg live muxer exited with code {process.returncode} for "
                f"{mask_stream_url(self.output_url)}\n{stderr_tail}"
            )


class _NvencLivePublisher:
    """Encode CUDA frames with NVENC and publish packets live through ffmpeg."""

    def __init__(
        self,
        output_url: str,
        *,
        label: str,
        fps: float,
        logger: Any = None,
        profiler: Any = None,
        audio: dict | None = None,
    ) -> None:
        self.output_url = str(output_url)
        self.label = str(label or "Live Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._audio = audio or {}
        self._encoder = None
        self._muxer = _BitstreamLiveMuxer(
            output_url=self.output_url,
            fps=self.fps,
            label=self.label,
            logger=self._logger,
            profiler=self._profiler,
            audio=self._audio,
        )

    def _ensure_open(self, frame: torch.Tensor) -> None:
        if self._encoder is not None:
            return
        from hmlib.video.py_nv_encoder import PyNvVideoEncoder

        if frame.device.type != "cuda":
            raise ValueError("NVENC live publisher requires CUDA frames")
        with _prof_ctx(self._profiler, "live_publish.start_nvenc"):
            self._encoder = PyNvVideoEncoder(
                output_path=None,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=self.fps,
                codec="h264",
                preset="P2",
                device=frame.device,
                gpu_id=frame.device.index or 0,
                cuda_stream=torch.cuda.current_stream(frame.device).cuda_stream,
                bitstream_handler=self._muxer.write_packet,
                profiler=self._profiler,
            )
            self._encoder.open()

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        frame = unwrap_tensor(img)
        if not isinstance(frame, torch.Tensor) or frame.device.type != "cuda":
            raise ValueError("NVENC live publisher requires CUDA tensor frames")
        if frame.ndim != 3:
            raise ValueError(
                f"Expected single frame tensor with shape (H, W, C), got {frame.shape}"
            )
        with _prof_ctx(self._profiler, "live_publish.prepare_cuda_frame"):
            frame = make_visible_image(frame, enable_resizing=show_scaled, force_numpy=False)
            if not isinstance(frame, torch.Tensor):
                raise TypeError("Expected make_visible_image to return a CUDA tensor for NVENC")
            frame = _ensure_even_video_frame(frame)
        self._ensure_open(frame)
        assert self._encoder is not None
        with _prof_ctx(self._profiler, "live_publish.nvenc_encode"):
            self._encoder.write(frame)

    def close(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            encoder.close()
        self._muxer.close()


class _AmdLivePublisher:
    """Encode ROCm frames with the AMD encoder and publish packets live through ffmpeg."""

    def __init__(
        self,
        output_url: str,
        *,
        label: str,
        fps: float,
        logger: Any = None,
        profiler: Any = None,
        audio: dict | None = None,
    ) -> None:
        self.output_url = str(output_url)
        self.label = str(label or "Live Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._audio = audio or {}
        self._encoder = None
        self._muxer = _BitstreamLiveMuxer(
            output_url=self.output_url,
            fps=self.fps,
            label=self.label,
            logger=self._logger,
            profiler=self._profiler,
            audio=self._audio,
        )

    def _ensure_open(self, frame: torch.Tensor) -> None:
        if self._encoder is not None:
            return
        from hmlib.video.py_amd_codec import PyAmdVideoEncoder

        if frame.device.type != "cuda":
            raise ValueError("AMD live publisher requires ROCm/CUDA frames")
        with _prof_ctx(self._profiler, "live_publish.start_amd"):
            self._encoder = PyAmdVideoEncoder(
                output_path=None,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                fps=self.fps,
                codec="h264",
                device=frame.device,
                bitstream_handler=self._muxer.write_packet,
                profiler=self._profiler,
            )
            self._encoder.open()

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        frame = unwrap_tensor(img)
        if not isinstance(frame, torch.Tensor) or frame.device.type != "cuda":
            raise ValueError("AMD live publisher requires ROCm/CUDA tensor frames")
        if frame.ndim != 3:
            raise ValueError(
                f"Expected single frame tensor with shape (H, W, C), got {frame.shape}"
            )
        with _prof_ctx(self._profiler, "live_publish.prepare_amd_frame"):
            frame = make_visible_image(frame, enable_resizing=show_scaled, force_numpy=False)
            if not isinstance(frame, torch.Tensor):
                raise TypeError(
                    "Expected make_visible_image to return a ROCm tensor for AMD encode"
                )
            frame = _ensure_even_video_frame(frame)
        self._ensure_open(frame)
        assert self._encoder is not None
        with _prof_ctx(self._profiler, "live_publish.amd_encode"):
            self._encoder.write(frame)

    def close(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            encoder.close()
        self._muxer.close()


class FFmpegLivePublisher:
    """Live RTMP(S) publisher.

    Uses a GPU encoder when CUDA/ROCm frames are available and falls back to
    the rawvideo->ffmpeg path for CPU frames or unsupported environments.
    """

    def __init__(
        self,
        output_url: str,
        *,
        label: str,
        fps: float,
        logger: Any = None,
        profiler: Any = None,
        audio: dict | None = None,
    ) -> None:
        self.output_url = str(output_url)
        self.label = str(label or "Live Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._profiler = profiler
        self._audio = audio or {}
        self._backend: Any | None = None

    def _select_backend(self, img: torch.Tensor | np.ndarray | StreamTensorBase) -> Any:
        frame = unwrap_tensor(img)
        if isinstance(frame, torch.Tensor) and frame.device.type == "cuda":
            try:
                if is_rocm_backend():
                    return _AmdLivePublisher(
                        output_url=self.output_url,
                        label=self.label,
                        fps=self.fps,
                        logger=self._logger,
                        profiler=self._profiler,
                        audio=self._audio,
                    )
                return _NvencLivePublisher(
                    output_url=self.output_url,
                    label=self.label,
                    fps=self.fps,
                    logger=self._logger,
                    profiler=self._profiler,
                    audio=self._audio,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self._logger.warning(
                    "Falling back to raw ffmpeg live publisher for %s because NVENC live "
                    "publisher initialization failed: %s",
                    mask_stream_url(self.output_url),
                    exc,
                )
        return _RawVideoLivePublisher(
            output_url=self.output_url,
            label=self.label,
            fps=self.fps,
            logger=self._logger,
            profiler=self._profiler,
            audio=self._audio,
        )

    def write_frame(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: float | None = None,
    ) -> None:
        if self._backend is None:
            self._backend = self._select_backend(img)
        self._backend.write_frame(img, show_scaled=show_scaled)

    def close(self) -> None:
        if self._backend is None:
            return
        self._backend.close()
        self._backend = None
