from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, Callable, List, Optional, Union

import torch
from typeguard import typechecked

from hmlib.log import get_logger
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.video.ffmpeg import build_ffmpeg_output_handler, iter_ffmpeg_output_lines
from hmlib.video.ffmpeg_mux_cmd import build_ffmpeg_raw_bitstream_mux_cmd
from hockeymon import bgr_to_i420_cuda

try:
    import PyNvVideoCodec as nvc  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    nvc = None  # type: ignore[assignment]

logger = get_logger(__name__)


class _DLPackFrame:
    """
    Lightweight wrapper exposing a torch CUDA tensor via the DLPack protocol.

    This exists to adapt to PyNvEncoder's expectation of calling
    __dlpack__(consumer_stream) with a positional argument, while newer
    PyTorch versions require keyword-only parameters. We forward the
    consumer stream so PyTorch can synchronize producer/consumer streams
    correctly before exporting the capsule.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def cuda(self) -> "_DLPackFrame":
        # PyNvEncoder's Encode() checks for a .cuda() method and calls it;
        # returning self keeps everything on the GPU while still allowing
        # the encoder to discover the DLPack interface on this wrapper.
        return self

    def __dlpack__(self, *args, **kwargs):
        # PyNvVideoCodec calls __dlpack__(consumer_stream) positionally, but
        # torch.Tensor.__dlpack__ requires keyword-only args. Forward the
        # consumer stream so PyTorch can perform the required synchronization
        # between the current stream (producer) and the consumer stream.
        if args and "stream" not in kwargs:
            kwargs["stream"] = args[0]
            args = args[1:]
        # Be tolerant of any extra positional args from other dlpack callers
        # (max_version, dl_device, copy), mapping them onto PyTorch keywords.
        if args and "max_version" not in kwargs:
            kwargs["max_version"] = args[0]
            args = args[1:]
        if args and "dl_device" not in kwargs:
            kwargs["dl_device"] = args[0]
            args = args[1:]
        if args and "copy" not in kwargs:
            kwargs["copy"] = args[0]
            args = args[1:]
        return self._tensor.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


class PyNvVideoEncoder:
    """
    High-level GPU-only video encoder backed by PyNvVideoCodec.

    - Accepts BGR torch.Tensors on CUDA (single frame or batch).
    - Uses jetson-utils (via hockeymon.bgr_to_i420_cuda) to convert
      BGR -> planar YUV420 (I420) entirely on the GPU.
    - Feeds YUV420 frames to PyNvVideoCodec's NVENC bindings using CUDA
      memory through the DLPack interface (no CPU copies of raw frames).
    - Encodes to an elementary H.26x/AV1 bitstream and, by default, writes
      it to a sidecar ``.h264`` / ``.h265`` / ``.ivf`` file before invoking
      ``ffmpeg`` once at ``close()`` time to mux into the requested
      container (MP4/MKV/etc., based on the output file extension).
    - Alternative backend using PyAV remains available via
      HM_VIDEO_ENCODER_BACKEND or the ``use_pyav`` argument.
    """

    @typechecked
    def __init__(
        self,
        output_path: Optional[Union[str, Path]],
        width: int,
        height: int,
        fps: float,
        codec: str = "h265",
        preset: str = "P1",
        device: Optional[torch.device] = None,
        gpu_id: Optional[int] = None,
        cuda_context: Optional[int] = None,
        cuda_stream: Optional[int] = None,
        bitrate: Optional[int] = None,
        mux_audio_file: Optional[str] = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
        use_pyav: Optional[bool] = None,
        profiler: Optional[Any] = None,
        ffmpeg_output_handler: Optional[Any] = None,
        bitstream_handler: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        if nvc is None:
            raise ImportError(
                "PyNvVideoEncoder requires the PyNvVideoCodec package "
                "(import PyNvVideoCodec failed)."
            )

        self.output_path = Path(output_path) if output_path is not None else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec.lower()
        self.preset = preset.upper()
        self.last_frame_id: Optional[int] = None

        if device is None:
            device = torch.device("cuda", 0)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("PyNvVideoEncoder requires a CUDA device.")
        self.device = device

        self.gpu_id = int(gpu_id if gpu_id is not None else (device.index or 0))
        self.cuda_context = cuda_context
        self.cuda_stream = cuda_stream
        self.bitrate = int(bitrate) if bitrate is not None else None
        self._mux_audio_file = str(mux_audio_file) if mux_audio_file else None
        self._mux_audio_stream = int(mux_audio_stream or 0)
        self._mux_audio_offset_seconds = float(mux_audio_offset_seconds or 0.0)
        self._mux_audio_aac_bitrate = str(mux_audio_aac_bitrate or "192k")
        # Optional HmProfiler instance injected by callers.
        self._profiler: Optional[Any] = profiler
        self._ffmpeg_output_handler = ffmpeg_output_handler
        self._bitstream_handler = bitstream_handler

        backend_env = os.environ.get("HM_VIDEO_ENCODER_BACKEND", "").lower()
        backend: str
        if self._bitstream_handler is not None:
            backend = "callback"
        elif use_pyav is not None:
            # Explicit caller override wins: True -> pyav, False -> ffmpeg pipe.
            backend = "pyav" if use_pyav else "raw"
        elif backend_env in {"pyav", "raw"}:
            # Environment variable controls backend when set to a known value.
            backend = backend_env
        elif backend_env:
            raise ValueError(
                f"Unsupported HM_VIDEO_ENCODER_BACKEND={backend_env!r}; supported backends are: raw, pyav."
            )
        else:
            # Default to writing a raw elementary bitstream sidecar and
            # invoking ffmpeg once at close() to mux into the container.
            backend = "raw"

        self._backend: str = backend

        if self.width % 2 or self.height % 2:
            raise ValueError("Width and height must be even for YUV420 encoding.")

        self._encoder: Optional[nvc.PyNvEncoder] = None  # type: ignore[assignment]
        self._av_container = None
        self._av_stream = None
        self._next_pts: int = 0
        self._opened = False
        self._frames_in_current_bitstream: int = 0
        self._frame_duration_units: Optional[int] = None
        self._bitstream_path: Optional[Path] = None
        self._bitstream_file: Optional[IO[bytes]] = None
        self._bgr_to_i420_cuda: Optional[torch.Tensor] = (
            self._profiler.rf("bgr_to_i420_cuda")
            if (self._profiler is not None and self._profiler.enabled)
            else contextlib.nullcontext()
        )
        self._bgr_contiguous_1: Optional[torch.Tensor] = (
            self._profiler.rf("bgr_contig_1")
            if (self._profiler is not None and self._profiler.enabled)
            else contextlib.nullcontext()
        )
        self._bgr_contiguous_2: Optional[torch.Tensor] = (
            self._profiler.rf("bgr_contig_2_dlpack")
            if (self._profiler is not None and self._profiler.enabled)
            else contextlib.nullcontext()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialize NVENC encoder and container muxer."""
        if self._opened:
            return

        self._encoder = self._create_encoder()
        if self._backend == "pyav":
            self._open_bitstream_file()
        elif self._backend == "raw":
            self._open_bitstream_file()
        elif self._backend == "callback":
            pass
        else:
            raise ValueError(f"Unsupported NVENC encoder backend: {self._backend}")
        self._opened = True

    def write(
        self, frames: torch.Tensor, frame_ids: Union[int, torch.Tensor, List[int], None] = None
    ) -> None:
        """
        Encode one or more BGR frames and append them to the output stream.

        Args:
            frames: torch.Tensor with shape:
                - [H, W, 3]
                - [N, H, W, 3]
                - [3, H, W]
                - [N, 3, H, W]
              Values are interpreted as 8-bit BGR (0–255) or floats in
              [0, 1] or [0, 255]. All processing stays on CUDA.
        """
        if not self._opened:
            raise RuntimeError("Encoder is not open. Call open() before write().")

        if self._encoder is None:
            raise RuntimeError("Encoder is not properly initialized.")

        batch = self._normalize_frames(frames)

        if frame_ids is not None:
            ids_list: List[int] = []

            if isinstance(frame_ids, StreamTensorBase):
                frame_ids = unwrap_tensor(frame_ids, get=True)

            if isinstance(frame_ids, torch.Tensor):
                ids_t = frame_ids.detach()
                if ids_t.is_cuda:
                    ids_t = ids_t.cpu()
                ids_list = [int(x) for x in ids_t.reshape(-1).tolist()]
            elif isinstance(frame_ids, (list, tuple)):
                for item in frame_ids:
                    if isinstance(item, torch.Tensor):
                        item = item.detach().reshape(-1).cpu().tolist()
                    if isinstance(item, (list, tuple)) and len(item) == 1:
                        item = item[0]
                    ids_list.append(int(item))
            else:
                ids_list = [int(frame_ids)]

            frame_count = int(batch.shape[0])
            if len(ids_list) != frame_count:
                raise ValueError(
                    f"frame_ids length mismatch: expected {frame_count}, got {len(ids_list)}"
                )

            if self.last_frame_id is not None:
                expected_frame_id = int(self.last_frame_id) + 1
                if ids_list[0] != expected_frame_id:
                    raise ValueError(
                        f"Non-consecutive frame_id: expected {expected_frame_id}, got {ids_list}"
                    )
            self.last_frame_id = int(ids_list[-1])

        prof = self._profiler
        batch_ctx = (
            prof.rf("video.nvenc.write_batch")  # type: ignore[union-attr]
            if getattr(prof, "enabled", False)
            else contextlib.nullcontext()
        )
        with batch_ctx:
            for frame in batch:
                frame_ctx = (
                    prof.rf("video.nvenc.encode_frame")  # type: ignore[union-attr]
                    if getattr(prof, "enabled", False)
                    else contextlib.nullcontext()
                )
                with frame_ctx:
                    yuv420 = self._bgr_to_yuv420(frame)
                    # yuv420 is a 2D CUDA tensor with shape [H*3/2, W], uint8.
                    bitstream = self._encoder.Encode(yuv420)  # type: ignore[union-attr]
                    self._frames_in_current_bitstream += 1
                    if bitstream:
                        if self._backend in {"pyav", "raw"}:
                            if self._bitstream_file is None:
                                raise RuntimeError(
                                    "Bitstream backend is selected but bitstream file is not open."
                                )
                            self._bitstream_file.write(bytearray(bitstream))
                        elif self._backend == "callback":
                            if self._bitstream_handler is None:
                                raise RuntimeError(
                                    "Callback backend is selected but bitstream handler is not set."
                                )
                            self._bitstream_handler(bytes(bitstream))
                        else:
                            raise RuntimeError(
                                f"Unsupported NVENC encoder backend: {self._backend}"
                            )

    def close(self) -> None:
        """Flush pending frames, finalize container, and release resources."""
        if not self._opened:
            return

        if self._encoder is not None:
            # Flush encoder
            bitstream = self._encoder.EndEncode()  # type: ignore[union-attr]
            if bitstream:
                if self._backend in {"pyav", "raw"}:
                    if self._bitstream_file is None:
                        raise RuntimeError(
                            "Bitstream backend is selected but bitstream file is not open."
                        )
                    self._bitstream_file.write(bytearray(bitstream))
                elif self._backend == "callback":
                    if self._bitstream_handler is None:
                        raise RuntimeError(
                            "Callback backend is selected but bitstream handler is not set."
                        )
                    self._bitstream_handler(bytes(bitstream))
                else:
                    raise RuntimeError(f"Unsupported NVENC encoder backend: {self._backend}")

        if self._backend in {"pyav", "raw"}:
            bitstream_path = self._bitstream_path
            bitstream_file = self._bitstream_file
            self._bitstream_file = None
            # Close the sidecar bitstream file before remuxing so ffmpeg/PyAV
            # sees a fully flushed file on disk.
            if bitstream_file is not None:
                try:
                    bitstream_file.flush()
                finally:
                    bitstream_file.close()
            if bitstream_path is not None:
                if self._backend == "pyav":
                    if self._mux_audio_file:
                        # PyAV remuxer currently does not support adding a second
                        # audio input; fall back to the ffmpeg muxer when audio is requested.
                        self._mux_bitstream_file_with_ffmpeg(bitstream_path)
                    else:
                        self._mux_bitstream_file_with_pyav(bitstream_path)
                elif self._backend == "raw":
                    self._mux_bitstream_file_with_ffmpeg(bitstream_path)
                else:
                    raise RuntimeError(f"Unsupported NVENC encoder backend: {self._backend}")
        elif self._backend == "callback":
            pass
        else:
            assert self._bitstream_file is None

        self._encoder = None
        self._av_container = None
        self._av_stream = None
        self._bitstream_path = None
        self._opened = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_encoder(self) -> nvc.PyNvEncoder:  # type: ignore[override]
        """
        Create a PyNvEncoder configured for YUV420 (I420) input and the requested codec.

        The encoder expects CUDA-accessible YUV420 surfaces; we provide them
        via torch tensors implementing __dlpack__/__cuda_array_interface__.
        """
        config: dict[str, str] = {
            "codec": self.codec,
            "preset": self.preset,
            "fps": str(self.fps),
            "gpu_id": str(self.gpu_id),
        }

        # For file-backed and callback-backed elementary bitstreams, request a
        # finite GOP length so NVENC inserts regular IDR keyframes. A 0.5-second
        # GOP is better for high-motion content like hockey and is required for
        # downstream live muxers such as HLS to cut clean segments.
        if self._backend in {"raw", "callback"} and self.fps > 0:
            gop = max(1, int(round(self.fps * 0.5)))
            config["gop"] = str(gop)

        if self.cuda_context is not None:
            config["cudacontext"] = int(self.cuda_context)
        if self.cuda_stream is not None:
            config["cudastream"] = int(self.cuda_stream)
        if self.bitrate is not None:
            config["bitrate"] = str(self.bitrate)

        # YUV420 (I420) is efficient for NVENC and maps to yuv420p in the container.
        # Set usecpuinputbuffer=False to keep frames on CUDA.
        return nvc.CreateEncoder(self.width, self.height, "YUV420", False, **config)

    def _bitstream_format(self) -> str:
        """
        Return the ffmpeg demuxer format corresponding to the configured codec.
        """
        if self.codec == "h264":
            return "h264"
        if self.codec in ("hevc", "h265"):
            return "hevc"
        if self.codec == "av1":
            return "av1"
        raise ValueError(f"Unsupported codec for muxing: {self.codec}")

    def _bitstream_extension(self) -> str:
        """
        Return the on-disk sidecar file extension for the elementary bitstream.

        H.264/H.265 use ``.h264`` / ``.h265``; AV1 uses ``.ivf``.
        """
        if self.codec == "h264":
            return ".h264"
        if self.codec in ("hevc", "h265"):
            return ".h265"
        if self.codec == "av1":
            return ".ivf"
        raise ValueError(f"Unsupported codec for bitstream extension: {self.codec}")

    def _container_format(self) -> Optional[str]:
        """Return an ffmpeg muxer name based on the output file extension."""
        if self.output_path is None:
            return None
        suffix = self.output_path.suffix.lower()
        if not suffix:
            return None
        mapping = {
            ".mp4": "mp4",
            ".m4v": "mp4",
            ".mov": "mov",
            ".mkv": "matroska",
            ".webm": "webm",
        }
        return mapping.get(suffix)

    def _open_bitstream_file(self) -> None:
        """
        Open the raw elementary bitstream sidecar file for writing.

        The filename is derived from the requested container path, e.g.:
          - ``output.mp4`` -> ``output.mp4.h265``
          - ``tracking_output.mkv`` -> ``tracking_output.mkv.h265``
        """
        ext = self._bitstream_extension()
        if self.output_path is None:
            raise RuntimeError("output_path is required for file-backed NVENC encoders")
        if self.output_path.suffix:
            raw_suffix = self.output_path.suffix + ext
            bitstream_path = self.output_path.with_suffix(raw_suffix)
        else:
            bitstream_path = self.output_path.with_suffix(ext)

        bitstream_path.parent.mkdir(parents=True, exist_ok=True)
        self._bitstream_path = bitstream_path
        self._bitstream_file = open(bitstream_path, "wb")
        if self._bitstream_file is None:
            raise RuntimeError(f"Failed to open bitstream file for writing: {bitstream_path}")

    def _mux_bitstream_file_with_ffmpeg(self, bitstream_path: Path) -> None:
        """
        Invoke ffmpeg once to mux the raw elementary bitstream into a container.

        The container type is inferred from ``self.output_path``.
        """
        from shutil import which

        ffmpeg = which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for container muxing but was not found in PATH.")

        mux_audio_file = self._mux_audio_file
        mux_audio_stream = int(self._mux_audio_stream or 0)
        mux_audio_offset_seconds = float(self._mux_audio_offset_seconds or 0.0)
        mux_audio_aac_bitrate = str(self._mux_audio_aac_bitrate or "192k")

        ffprobe = which("ffprobe") if mux_audio_file else None
        if mux_audio_file and ffprobe is None:
            raise RuntimeError("ffprobe is required for audio muxing but was not found in PATH.")

        stream_format = self._bitstream_format()
        expected_duration = None
        try:
            if self.fps > 0 and self._frames_in_current_bitstream > 0:
                expected_duration = float(self._frames_in_current_bitstream) / float(self.fps)
        except Exception:
            expected_duration = None
        muxer = self._container_format()

        def _probe_audio_codec(audio_file: str, stream_idx: int) -> Optional[str]:
            try:
                out = subprocess.check_output(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-select_streams",
                        f"a:{int(stream_idx)}",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "default=nk=1:nw=1",
                        audio_file,
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                codec = str(out).strip().lower()
                return codec or None
            except Exception:
                return None

        mux_audio_codec: Optional[str] = None
        if mux_audio_file:
            audio_path = Path(mux_audio_file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio mux source file not found: {audio_path}")
            if not audio_path.is_file():
                raise RuntimeError(f"Audio mux source is not a file: {audio_path}")
            mux_audio_codec = _probe_audio_codec(mux_audio_file, mux_audio_stream)
            if not mux_audio_codec:
                raise RuntimeError(
                    f"Could not detect audio stream a:{mux_audio_stream} in {mux_audio_file!r}"
                )

        def _build_cmd(with_audio: bool) -> List[str]:
            return build_ffmpeg_raw_bitstream_mux_cmd(
                ffmpeg=ffmpeg,
                bitstream_path=bitstream_path,
                output_path=self.output_path,
                stream_format=stream_format,
                fps=self.fps,
                muxer=muxer,
                audio_file=mux_audio_file if with_audio else None,
                audio_stream=mux_audio_stream,
                audio_offset_seconds=mux_audio_offset_seconds,
                audio_codec=mux_audio_codec if with_audio else None,
                aac_bitrate=mux_audio_aac_bitrate,
            )

        def _run(cmd: List[str], label: str) -> None:
            output_handler = build_ffmpeg_output_handler(
                self._ffmpeg_output_handler,
                total_seconds=expected_duration,
                label=label,
            )
            stdout_lines = deque(maxlen=200)
            stderr_lines = deque(maxlen=200)

            def _reader(stream, stream_name: str, sink: deque) -> None:
                try:
                    for line in iter_ffmpeg_output_lines(stream):
                        sink.append(line)
                        output_handler.handle_line(line, stream_name)
                except Exception:
                    pass
                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

            proc = None
            returncode: Optional[int] = None
            try:
                print("Muxing the raw bitstream with ffmpeg, this may take a moment...")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                threads = []
                if proc.stdout is not None:
                    t_out = threading.Thread(
                        target=_reader,
                        args=(proc.stdout, "stdout", stdout_lines),
                        daemon=True,
                    )
                    t_out.start()
                    threads.append(t_out)
                if proc.stderr is not None:
                    t_err = threading.Thread(
                        target=_reader,
                        args=(proc.stderr, "stderr", stderr_lines),
                        daemon=True,
                    )
                    t_err.start()
                    threads.append(t_err)
                returncode = proc.wait()
                for thread in threads:
                    thread.join()
                if returncode != 0:
                    stderr_output = "\n".join(stderr_lines)
                    stdout_output = "\n".join(stdout_lines)
                    raise RuntimeError(
                        "ffmpeg muxer failed while writing NVENC bitstream container. "
                        "Check codec/format settings, input bitstream, and disk space. "
                        f"(returncode={returncode}, stderr={stderr_output!r}, stdout={stdout_output!r})"
                    )
            finally:
                output_handler.close(returncode)
                if proc is not None and proc.poll() is None:
                    try:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass

        if mux_audio_file:
            try:
                _run(_build_cmd(with_audio=True), label="ffmpeg mux (video+audio)")
                return
            except Exception:
                if self.output_path.stem.endswith("-with-audio"):
                    logger.exception(
                        "ffmpeg mux with audio failed; refusing to fall back to video-only mux "
                        "because output_path implies audio. (audio_file=%s)",
                        mux_audio_file,
                    )
                    raise
                logger.exception(
                    "ffmpeg mux with audio failed; falling back to video-only mux. (audio_file=%s)",
                    mux_audio_file,
                )

        _run(_build_cmd(with_audio=False), label="ffmpeg mux (video-only)")

    def _mux_bitstream_file_with_pyav(self, bitstream_path: Path) -> None:
        """
        Remux the raw elementary bitstream into a container using PyAV.

        This avoids depending on an external ffmpeg binary while still using
        FFmpeg's demuxer/muxer logic via libavformat.
        """
        from fractions import Fraction

        import av

        stream_format = self._bitstream_format()
        input_options = {"framerate": str(self.fps)}

        try:
            input_container = av.open(
                str(bitstream_path),
                mode="r",
                format=stream_format,
                options=input_options,
            )
        except Exception as exc:
            raise RuntimeError(
                f"PyAV failed to open bitstream file for remuxing: {bitstream_path}"
            ) from exc

        try:
            if not input_container.streams.video:
                raise RuntimeError(f"No video stream found while remuxing {bitstream_path}")
            input_stream = input_container.streams.video[0]

            output_container = av.open(str(self.output_path), mode="w")
            try:
                fps = Fraction(int(round(self.fps * 1001)), 1001)
                codec_name = input_stream.codec_context.name
                output_stream = output_container.add_stream(codec_name, rate=fps)
                output_stream.width = self.width
                output_stream.height = self.height
                output_stream.pix_fmt = "yuv420p"
                output_stream.time_base = Fraction(1, 90000)
                if input_stream.codec_context.extradata:
                    output_stream.codec_context.extradata = input_stream.codec_context.extradata

                output_container.start_encoding()
                tb = output_stream.time_base
                ticks_per_frame = int(round((Fraction(1, 1) / fps) / tb))
                ticks_per_frame = max(ticks_per_frame, 1)
                next_pts = 0
                for packet in input_container.demux(input_stream):
                    if packet is None or packet.size == 0:
                        continue
                    packet.stream = output_stream
                    packet.time_base = tb
                    packet.pts = next_pts
                    packet.dts = next_pts
                    packet.duration = ticks_per_frame
                    next_pts += ticks_per_frame
                    output_container.mux(packet)
            finally:
                output_container.close()
        finally:
            input_container.close()

    def _open_pyav_container(self) -> None:
        import av

        self._av_container = av.open(str(self.output_path), mode="w")

        fps = Fraction(int(round(self.fps * 1001)), 1001)

        codec_name = "hevc" if self.codec in ("h265", "hevc") else self.codec
        self._av_stream = self._av_container.add_stream(codec_name, rate=fps)
        self._av_stream.width = self.width
        self._av_stream.height = self.height
        self._av_stream.pix_fmt = "yuv420p"

        # Give muxer a sensible hint (optional, but helps keep things stable).
        self._av_stream.time_base = Fraction(1, 90000)

        # IMPORTANT: finalize header so time_base is final
        self._av_container.start_encoding()

        tb = self._av_stream.time_base  # may have changed after start_encoding()
        self._ticks_per_frame = int(round((Fraction(1, 1) / fps) / tb))

        self._next_pts = 0  # in tb ticks
        self._frames_in_current_bitstream = 0

    def _mux_packet_pyav(self, bitstream: bytes) -> None:
        import av

        dur = int(self._frames_in_current_bitstream * self._ticks_per_frame)

        packet = av.packet.Packet(bitstream)
        packet.stream = self._av_stream
        packet.time_base = self._av_stream.time_base
        packet.pts = self._next_pts
        packet.dts = self._next_pts
        packet.duration = dur

        self._av_container.mux_one(packet)

        self._next_pts += dur

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to a CUDA tensor with shape [N, H, W, 3] and dtype uint8 (BGR).
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a torch.Tensor")

        # Move to the requested GPU and drop gradients; no CPU round-trip.
        frames = frames.to(device=self.device, non_blocking=True).detach()

        if frames.ndim == 3:
            # HWC or CHW
            if frames.shape[0] == 3 and frames.shape[-1] != 3:
                # CHW -> HWC
                frames = frames.permute(1, 2, 0)
            frames = frames.unsqueeze(0)
        elif frames.ndim == 4:
            # NHWC or NCHW
            if frames.shape[1] == 3 and frames.shape[-1] != 3:
                # NCHW -> NHWC
                frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError("frames must have 3 or 4 dimensions")

        if frames.shape[-1] != 3:
            raise ValueError("Last dimension must be 3 (BGR channels)")

        n, h, w, _c = frames.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"Expected frames of shape (*, {self.height}, {self.width}, 3), "
                f"got {tuple(frames.shape)}"
            )

        # Normalize dtype to uint8 in [0, 255]
        if frames.dtype.is_floating_point:
            max_val = frames.max()
            if float(max_val) <= 1.0:
                frames = frames * 255.0
            frames.clamp_(0, 255)
            frames = frames.to(torch.uint8)
        elif frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)

        return frames.contiguous()

    def _bgr_to_yuv420(self, frame: torch.Tensor) -> _DLPackFrame:
        """
        Convert a single BGR frame (H, W, 3) in uint8 (CUDA) to planar YUV420 (I420)
        layout suitable for PyNvEncoder when configured with format \"YUV420\".

        The resulting tensor has shape [H * 3 / 2, W] on CUDA.
        """
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Expected BGR frame with shape (H, W, 3)")

        h, w, _ = frame.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"Frame size mismatch: expected {self.width}x{self.height}, got {w}x{h}"
            )

        if frame.device != self.device:
            frame = frame.to(self.device, non_blocking=True)

        if frame.dtype != torch.uint8:
            frame.clamp_(0, 255)
            frame = frame.to(torch.uint8)

        with self._bgr_contiguous_1:
            frame = frame.contiguous()

        # Delegate BGR -> I420 conversion to jetson-utils via hockeymon binding.
        with self._bgr_to_i420_cuda:
            yuv420 = bgr_to_i420_cuda(frame)

        if yuv420.dim() != 2 or yuv420.size(0) != h * 3 // 2 or yuv420.size(1) != w:
            raise RuntimeError(
                "bgr_to_i420_cuda returned unexpected shape "
                f"{tuple(yuv420.shape)} for input {h}x{w}"
            )
        with self._bgr_contiguous_2:
            return _DLPackFrame(yuv420.contiguous())
