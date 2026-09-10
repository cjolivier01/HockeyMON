from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urlparse

import torch
from typeguard import typechecked

from hmlib.log import get_logger
from hmlib.utils.finalization import finalize_resources
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor
from hmlib.video.ffmpeg import BasicVideoInfo, preexec_fn

logger = get_logger(__name__)


def _ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def _require_ffmpeg_path() -> str:
    ffmpeg_path = _ffmpeg_path()
    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg is required for the AMD video backend, but it was not found on PATH."
        )
    return ffmpeg_path


def _is_stream_output(target: str) -> bool:
    parsed = urlparse(target)
    return bool(parsed.scheme and "://" in target)


def _read_ffmpeg_listing(*args: str) -> set[str]:
    ffmpeg_path = _ffmpeg_path()
    if ffmpeg_path is None:
        return set()
    try:
        raw = subprocess.check_output(
            [ffmpeg_path, "-hide_banner", *args],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return set()
    items: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].startswith("V"):
            items.add(tokens[1].strip())
        elif len(tokens) >= 1 and tokens[0].isalpha():
            items.add(tokens[0].strip())
    return items


_FFMPEG_ENCODERS: Optional[set[str]] = None
_FFMPEG_HWACCELS: Optional[set[str]] = None


def _get_ffmpeg_encoders() -> set[str]:
    global _FFMPEG_ENCODERS
    if _FFMPEG_ENCODERS is None:
        _FFMPEG_ENCODERS = _read_ffmpeg_listing("-encoders")
    return _FFMPEG_ENCODERS


def _get_ffmpeg_hwaccels() -> set[str]:
    global _FFMPEG_HWACCELS
    if _FFMPEG_HWACCELS is None:
        _FFMPEG_HWACCELS = _read_ffmpeg_listing("-hwaccels")
    return _FFMPEG_HWACCELS


def _best_vaapi_device() -> Optional[str]:
    env_device = (os.environ.get("HM_AMD_VAAPI_DEVICE") or "").strip()
    candidates = [env_device] if env_device else []
    candidates.extend(
        [
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            "/dev/dri/renderD130",
            "/dev/dri/renderD131",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _has_amf_runtime() -> bool:
    try:
        ctypes.CDLL("libamfrt64.so.1")
        return True
    except OSError:
        return False


class PyAmdVideoCodec:
    """Capability helpers for AMD-backed video encode/decode paths."""

    _CODEC_ALIASES = {
        "av1": "av1",
        "h264": "h264",
        "avc": "h264",
        "hevc": "hevc",
        "h265": "hevc",
    }

    @classmethod
    def normalize_codec(cls, codec: Optional[str]) -> str:
        key = (codec or "h264").strip().lower()
        for token, family in cls._CODEC_ALIASES.items():
            if token in key:
                return family
        raise ValueError(f"Unsupported AMD codec {codec!r}; expected h264, hevc/h265, or av1.")

    @classmethod
    def bitstream_format(cls, codec: Optional[str]) -> str:
        family = cls.normalize_codec(codec)
        if family == "av1":
            return "ivf"
        return family

    @classmethod
    def default_vaapi_device(cls) -> str:
        device = _best_vaapi_device()
        if device is None:
            raise RuntimeError(
                "VAAPI render device was not found. Set HM_AMD_VAAPI_DEVICE or ensure /dev/dri/renderD* exists."
            )
        return device

    @classmethod
    def is_encoder_available(cls, codec: Optional[str] = None, backend: str = "auto") -> bool:
        try:
            cls.resolve_encoder(codec or "h264", backend=backend)
            return True
        except RuntimeError:
            return False

    @classmethod
    def is_decoder_available(cls, backend: str = "auto") -> bool:
        chosen = cls.default_decoder_backend() if backend == "auto" else str(backend).lower()
        if chosen == "vaapi":
            return (
                _ffmpeg_path() is not None
                and "vaapi" in _get_ffmpeg_hwaccels()
                and _best_vaapi_device() is not None
            )
        if chosen == "software":
            return _ffmpeg_path() is not None
        raise RuntimeError(f"Unsupported AMD decoder backend {backend!r}.")

    @classmethod
    def default_encoder_backend(cls) -> str:
        if "vaapi" in _get_ffmpeg_hwaccels() and _best_vaapi_device() is not None:
            for name in ("hevc_vaapi", "h264_vaapi", "av1_vaapi"):
                if name in _get_ffmpeg_encoders():
                    return "vaapi"
        if _has_amf_runtime():
            for name in ("hevc_amf", "h264_amf", "av1_amf"):
                if name in _get_ffmpeg_encoders():
                    return "amf"
        raise RuntimeError(
            "No AMD video encoder backend is available. Checked VAAPI and AMF FFmpeg backends."
        )

    @classmethod
    def default_decoder_backend(cls) -> str:
        return "software"

    @classmethod
    def resolve_encoder(cls, codec: Optional[str], backend: str = "auto") -> str:
        family = cls.normalize_codec(codec)
        candidates = (
            [f"{family}_vaapi", f"{family}_amf"]
            if backend == "auto"
            else [f"{family}_{str(backend).lower()}"]
        )
        for name in candidates:
            if name not in _get_ffmpeg_encoders():
                continue
            if name.endswith("_vaapi") and _best_vaapi_device() is None:
                continue
            if name.endswith("_amf") and not _has_amf_runtime():
                continue
            return name
        raise RuntimeError(f"No FFmpeg AMD encoder is available for codec family {family!r}.")

    @classmethod
    def preferred_output_codec(cls) -> str:
        if cls.is_encoder_available("hevc", backend="vaapi"):
            return "hevc_vaapi"
        if cls.is_encoder_available("h264", backend="vaapi"):
            return "h264_vaapi"
        if cls.is_encoder_available("hevc", backend="amf"):
            return "hevc_amf"
        if cls.is_encoder_available("h264", backend="amf"):
            return "h264_amf"
        raise RuntimeError("No AMD output codec is available.")


class PyAmdVideoDecoder:
    """Sequential FFmpeg-backed AMD video decoder."""

    @typechecked
    def __init__(
        self,
        input_path: Union[str, Path],
        *,
        device: Optional[torch.device] = None,
        backend: str = "auto",
        batch_size: int = 1,
        output_format: str = "bgr24",
        vaapi_device: Optional[str] = None,
        loglevel: str = "warning",
    ) -> None:
        self.input_path = str(input_path)
        self.device = device if isinstance(device, torch.device) else torch.device(device or "cpu")
        self.backend = (
            PyAmdVideoCodec.default_decoder_backend()
            if backend == "auto"
            else str(backend).strip().lower()
        )
        self.batch_size = int(batch_size or 1)
        self.output_format = str(output_format or "bgr24")
        self.vaapi_device = vaapi_device or (
            PyAmdVideoCodec.default_vaapi_device() if self.backend == "vaapi" else None
        )
        self.loglevel = str(loglevel or "warning")
        self._ffmpeg_path = _require_ffmpeg_path()
        self.video_info = BasicVideoInfo(self.input_path)
        self.width = int(self.video_info.width)
        self.height = int(self.video_info.height)
        self.fps = float(self.video_info.fps)
        self.frame_count = int(self.video_info.frame_count)
        self.frames_delivered_count = 0
        self._frame_bytes = int(self.width * self.height * 3)
        self._start_time = 0.0
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail: deque[str] = deque(maxlen=40)
        if self.backend == "software":
            logger.warning(
                "PyAmdVideoDecoder is using FFmpeg software decode. "
                "Set backend='vaapi' explicitly once a stable AMD hardware decoder path is available."
            )
        self.open()

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _build_cmd(self) -> list[str]:
        cmd = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            self.loglevel,
            "-nostats",
            "-nostdin",
        ]
        if self.backend == "vaapi":
            cmd += [
                "-vaapi_device",
                str(self.vaapi_device),
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
            ]
        cmd += ["-i", self.input_path]
        if self._start_time > 0:
            # Output-side -ss keeps seeking frame-accurate for exact frame-number requests.
            cmd += ["-ss", f"{self._start_time:.6f}"]
        if self.backend == "vaapi":
            cmd += ["-vf", "hwdownload,format=bgr24"]
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.output_format,
            "pipe:1",
        ]
        return cmd

    def open(self) -> None:
        self.close()
        self._process = subprocess.Popen(
            self._build_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
            preexec_fn=preexec_fn,
        )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def get_index_from_time_in_seconds(self, timestamp: float) -> int:
        return max(0, int(round(float(timestamp) * self.fps)))

    def seek_to_index(self, frame_index: int) -> None:
        self._start_time = max(0.0, float(frame_index) / max(self.fps, 1e-6))
        self.open()

    def read_batch(
        self, batch_size: Optional[int] = None
    ) -> Optional[Union[torch.Tensor, StreamTensorBase]]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("AMD decoder is not open.")
        batch = int(batch_size or self.batch_size)
        raw = process.stdout.read(self._frame_bytes * batch)
        if not raw:
            return None
        frame_count = len(raw) // self._frame_bytes
        if frame_count <= 0:
            return None
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            frame_count, self.height, self.width, 3
        )
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        self.frames_delivered_count += int(frame_count)
        if self.device.type == "cuda":
            return wrap_tensor(tensor.to(self.device, non_blocking=True))
        return tensor

    def end(self) -> None:
        self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None


class PyAmdVideoEncoder:
    """FFmpeg-backed AMD video encoder for VAAPI/AMF."""

    @typechecked
    def __init__(
        self,
        output_path: Optional[Union[str, Path]],
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        device: Optional[torch.device] = None,
        backend: str = "auto",
        bitrate: Optional[int] = None,
        mux_audio_file: Optional[str] = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
        bitstream_handler: Optional[Callable[[bytes], None]] = None,
        profiler: Optional[Any] = None,
    ) -> None:
        self.output_path = str(output_path) if output_path is not None else None
        self._output_is_stream = bool(self.output_path and _is_stream_output(self.output_path))
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec_family = PyAmdVideoCodec.normalize_codec(codec)
        self.device = device if isinstance(device, torch.device) else torch.device(device or "cpu")
        requested_backend = str(backend or "auto").strip().lower()
        self.encoder_name = PyAmdVideoCodec.resolve_encoder(
            self.codec_family, backend=requested_backend
        )
        self.backend = self.encoder_name.rsplit("_", 1)[-1]
        self.stream_format = PyAmdVideoCodec.bitstream_format(self.codec_family)
        self.bitrate = int(bitrate) if bitrate is not None else None
        self._mux_audio_file = str(mux_audio_file) if mux_audio_file else None
        self._mux_audio_stream = int(mux_audio_stream or 0)
        self._mux_audio_offset_seconds = float(mux_audio_offset_seconds or 0.0)
        self._mux_audio_aac_bitrate = str(mux_audio_aac_bitrate or "192k")
        self._bitstream_handler = bitstream_handler
        self._profiler = profiler
        self.last_frame_id: Optional[int] = None
        self._opened = False
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._background_error: Optional[BaseException] = None
        self._vaapi_device = (
            PyAmdVideoCodec.default_vaapi_device() if self.backend == "vaapi" else None
        )
        self._ffmpeg_path = _require_ffmpeg_path()
        if self.width % 2 or self.height % 2:
            raise ValueError("Width and height must be even for AMD YUV420 encoders.")
        if self.output_path is None and self._bitstream_handler is None:
            raise ValueError("PyAmdVideoEncoder requires either output_path or bitstream_handler.")

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _stdout_worker(self, pipe) -> None:
        try:
            while True:
                payload = pipe.read(65536)
                if not payload:
                    break
                assert self._bitstream_handler is not None
                self._bitstream_handler(payload)
        except BaseException as exc:  # pragma: no cover - background thread
            self._background_error = exc
        finally:
            pipe.close()

    def _build_video_filters(self) -> Optional[str]:
        if self.backend == "vaapi":
            return "format=nv12,hwupload"
        return None

    def _build_cmd(self) -> list[str]:
        fps_frac = Fraction(float(self.fps)).limit_denominator(1001)
        fps_str = (
            f"{fps_frac.numerator}/{fps_frac.denominator}"
            if fps_frac.denominator != 1
            else str(fps_frac.numerator)
        )
        cmd = [
            self._ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-progress",
            "pipe:2",
            "-nostats",
            "-nostdin",
        ]
        if self.backend == "vaapi":
            cmd += ["-vaapi_device", str(self._vaapi_device)]
        cmd += [
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            fps_str,
            "-i",
            "-",
        ]
        if self.output_path is not None and self._mux_audio_file:
            if abs(self._mux_audio_offset_seconds) > 1e-9:
                cmd += ["-itsoffset", str(self._mux_audio_offset_seconds)]
            cmd += [
                "-i",
                self._mux_audio_file,
                "-map",
                "0:v:0",
                "-map",
                f"1:a:{self._mux_audio_stream}",
            ]
        else:
            cmd += ["-an"]
        filters = self._build_video_filters()
        if filters:
            cmd += ["-vf", filters]
        cmd += ["-c:v", self.encoder_name]
        if self.bitrate is not None:
            cmd += ["-b:v", str(self.bitrate)]
        if self.output_path is None:
            cmd += ["-f", self.stream_format, "pipe:1"]
        else:
            if self._mux_audio_file:
                cmd += [
                    "-c:a",
                    "aac",
                    "-b:a",
                    self._mux_audio_aac_bitrate,
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-shortest",
                ]
            output_suffix = (
                Path(self.output_path).suffix.lower() if not self._output_is_stream else ""
            )
            if output_suffix in {".mp4", ".m4v", ".mov"}:
                cmd += ["-movflags", "+faststart"]
                if self.codec_family == "hevc":
                    cmd += ["-tag:v", "hvc1"]
                elif self.codec_family == "av1":
                    cmd += ["-tag:v", "av01"]
            cmd.append(self.output_path)
        return cmd

    def _normalize_frames(self, frames: Union[torch.Tensor, StreamTensorBase]) -> torch.Tensor:
        if isinstance(frames, StreamTensorBase):
            frames = unwrap_tensor(frames)
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a torch.Tensor")
        frames = frames.detach()
        if self.device.type == "cuda":
            frames = frames.to(self.device, non_blocking=True)
        elif frames.is_cuda:
            frames = frames.cpu()
        if frames.ndim == 3:
            if frames.shape[0] == 3 and frames.shape[-1] != 3:
                frames = frames.permute(1, 2, 0)
            frames = frames.unsqueeze(0)
        elif frames.ndim == 4:
            if frames.shape[1] == 3 and frames.shape[-1] != 3:
                frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError("frames must have 3 or 4 dimensions")
        if frames.shape[-1] != 3:
            raise ValueError("Last dimension must be 3 (BGR channels)")
        if int(frames.shape[1]) != self.height or int(frames.shape[2]) != self.width:
            raise ValueError(
                f"Expected frames of shape (*, {self.height}, {self.width}, 3), got {tuple(frames.shape)}"
            )
        if frames.dtype.is_floating_point:
            if float(frames.max()) <= 1.0:
                frames = frames * 255.0
            frames = frames.clamp(0, 255).to(torch.uint8)
        elif frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)
        return frames.contiguous()

    def _validate_frame_ids(
        self, batch: torch.Tensor, frame_ids: Union[int, torch.Tensor, list[int], None]
    ) -> None:
        if frame_ids is None:
            return
        ids_list: list[int] = []
        if isinstance(frame_ids, StreamTensorBase):
            frame_ids = unwrap_tensor(frame_ids, get=True)
        if isinstance(frame_ids, torch.Tensor):
            ids_t = frame_ids.detach()
            if ids_t.is_cuda:
                ids_t = ids_t.cpu()
            ids_list = [int(x) for x in ids_t.reshape(-1).tolist()]
        elif isinstance(frame_ids, (list, tuple)):
            ids_list = [int(x) for x in frame_ids]
        else:
            ids_list = [int(frame_ids)]
        frame_count = int(batch.shape[0])
        if len(ids_list) != frame_count:
            raise ValueError(
                f"frame_ids length mismatch: expected {frame_count}, got {len(ids_list)}"
            )
        if self.last_frame_id is not None:
            expected = int(self.last_frame_id) + 1
            if ids_list[0] != expected:
                raise ValueError(f"Non-consecutive frame_id: expected {expected}, got {ids_list}")
        self.last_frame_id = int(ids_list[-1])

    def _check_background_error(self) -> None:
        if self._background_error is not None:
            exc = self._background_error
            self._background_error = None
            raise RuntimeError("AMD encoder background worker failed") from exc

    def open(self) -> None:
        if self._opened:
            return
        stdout = subprocess.PIPE if self._bitstream_handler is not None else subprocess.DEVNULL
        self._process = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=subprocess.PIPE,
            bufsize=0,
            preexec_fn=preexec_fn,
        )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()
        if self._bitstream_handler is not None:
            assert self._process.stdout is not None
            self._stdout_thread = threading.Thread(
                target=self._stdout_worker,
                args=(self._process.stdout,),
                daemon=True,
            )
            self._stdout_thread.start()
        self._opened = True

    def write(
        self,
        frames: Union[torch.Tensor, StreamTensorBase],
        frame_ids: Union[int, torch.Tensor, list[int], None] = None,
    ) -> None:
        if not self._opened or self._process is None or self._process.stdin is None:
            raise RuntimeError("Encoder is not open. Call open() before write().")
        batch = self._normalize_frames(frames)
        self._validate_frame_ids(batch, frame_ids)
        self._check_background_error()
        cpu_batch = batch.cpu() if batch.is_cuda else batch
        try:
            self._process.stdin.write(cpu_batch.numpy().tobytes())
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(f"AMD encoder failed while writing frames.\n{stderr_tail}") from exc
        self._check_background_error()

    def close(self) -> None:
        process = self._process
        self._process = None
        self._opened = False
        if process is None:
            return

        def wait_process():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError("AMD encoder timed out during finalization") from error

        def join_thread(attribute):
            thread = getattr(self, attribute)
            if thread is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError(f"AMD encoder worker {attribute} did not stop")
                setattr(self, attribute, None)

        def check_exit_status():
            if process.returncode != 0:
                stderr_tail = "\n".join(self._stderr_tail)
                raise RuntimeError(
                    f"AMD encoder exited with code {process.returncode}.\n{stderr_tail}"
                )

        actions = []
        if process.stdin is not None:
            actions.append(("AMD encoder input flush", process.stdin.close))
        actions.extend(
            [
                ("AMD encoder process", wait_process),
                ("AMD encoder bitstream worker", lambda: join_thread("_stdout_thread")),
                ("AMD encoder stderr worker", lambda: join_thread("_stderr_thread")),
                ("AMD encoder background output", self._check_background_error),
                ("AMD encoder exit status", check_exit_status),
            ]
        )
        finalize_resources(actions)


__all__ = ["PyAmdVideoCodec", "PyAmdVideoDecoder", "PyAmdVideoEncoder"]
