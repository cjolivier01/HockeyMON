"""Unified video reader/writer built on top of torchaudio/ffmpeg.

Supports streaming decode, NVENC-friendly sizing and optional tensor-aware
interfaces used across the stitching and tracking CLIs.

@see @ref hmlib.video.ffmpeg "ffmpeg" for low-level FFmpeg helpers.
"""

import contextlib
import os
import platform
import sys
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import torch
from typeguard import typechecked

from hmlib.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import show_image
from hmlib.utils.gpu import StreamTensorBase, wrap_tensor
from hmlib.utils.image import make_channels_first, make_channels_last, resize_image
from hmlib.video.ffmpeg import BasicVideoInfo, get_ffmpeg_decoder_process
from hmlib.video.py_amd_codec import PyAmdVideoCodec, PyAmdVideoDecoder, PyAmdVideoEncoder

JETSON_UTILS_PY_PATH = "/mnt/monster-data/colivier/src/jetson-utils/python/python"

#
# Live stream:
# https://www.digitalocean.com/community/tutorials/how-to-set-up-a-video-streaming-server-using-nginx-rtmp-on-ubuntu-22-04
#

_EXTENSION_MAPPING = {
    "matroska": "mkv",
    "mp4": "mp4",
}

_FOURCC_TO_CODEC = {
    "HEVC": "hevc_cuvid",
    "HVC1": "hevc_cuvid",
    "HEV1": "hevc_cuvid",
    "H264": "h264_cuvid",
    "MJPEG": "mjpeg_cuvid",
    "XVID": "mpeg4_cuvid",
    "MP4V": "mpeg4_cuvid",
    "FMP4": "mpeg4_cuvid",
}

MAX_VIDEO_WIDTH = 8000  # 8K is 7680 x 4320
MAX_NEVC_VIDEO_WIDTH: int = 8192

if TYPE_CHECKING:
    from torchaudio.io import StreamReader as StreamingMediaDecoder


def _load_torchaudio_io():
    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Torchaudio backend requires the torchaudio package, but it is not installed."
        ) from exc
    return torchaudio.io.StreamReader, torchaudio.io.StreamWriter, torchaudio.io.CodecConfig


def _normalize_decoder_name(decoder_name: str) -> str:
    if not isinstance(decoder_name, str):
        return decoder_name
    return decoder_name.strip().lower()


def _load_pynvcodec():
    try:
        import PyNvVideoCodec as nvc  # type: ignore[import-not-found]

        return nvc
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyNvVideoCodec backend requires the PyNvVideoCodec package. "
            f"Failed to import: {exc}"
        )


def _load_jetson_utils():
    """
    Best-effort import of jetson_utils Python bindings (used by the GStreamer backend).

    Falls back to the local checkout path when the package is not on sys.path.
    """
    bazel_ext_path = "/mnt/monster-data/colivier/src/jetson-utils/bazel-bin/python/bindings"
    if os.path.isdir(bazel_ext_path) and bazel_ext_path not in sys.path:
        sys.path.insert(0, bazel_ext_path)
    if os.path.isdir(JETSON_UTILS_PY_PATH) and JETSON_UTILS_PY_PATH not in sys.path:
        sys.path.insert(0, JETSON_UTILS_PY_PATH)
    try:
        import jetson_utils  # type: ignore

        return jetson_utils
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            "GStreamer backend requires jetson_utils (DeepStream). " f"Failed to import: {exc}"
        )


def _jetson_codec_from_fourcc(codec: str) -> str | None:
    codec = codec.lower()
    if "h265" in codec or "hevc" in codec or "hvc" in codec:
        return "h265"
    if "h264" in codec or "avc" in codec:
        return "h264"
    if "mjpeg" in codec or "jpeg" in codec:
        return "mjpeg"
    if "mpeg4" in codec or "mp4v" in codec:
        return "mpeg4"
    if "vp9" in codec:
        return "vp9"
    if "vp8" in codec:
        return "vp8"
    return None


def _max_video_width(codec: str) -> int:
    if "hevc" in codec:
        return MAX_NEVC_VIDEO_WIDTH
    return MAX_VIDEO_WIDTH


class VideoStreamWriterInterface:
    def close(self) -> None:
        raise NotImplementedError("close not implemented")

    def isOpened(self) -> bool:
        raise NotImplementedError("isOpened not implemented")


def video_size(
    width: int, height: int, codec: str | None = None, max_width: int | None = None
) -> tuple[int | Literal[True]] | tuple[int | Literal[False]]:
    assert codec or max_width
    if codec:
        assert not max_width
        max_width = _max_video_width(codec)
    h = height
    w = width
    if w > max_width:
        ar = w / h
        if ar > 2.1:
            # probably the whole panorama
            max_width = max_width + 500
        w = int(max_width)
        h = int(w / ar)
        return w, h, True
    return w, h, False


def clamp_max_video_dimensions(
    width: torch.Tensor,
    height: torch.Tensor,
    codec: str | None = None,
    max_width: None | int | torch.Tensor = None,
) -> tuple[int, int]:
    assert codec or max_width
    if codec:
        assert not max_width
        max_width = _max_video_width(codec)
    if width.ndim == 0:
        wh = torch.tensor([width, height])
    else:
        wh = torch.cat([width, height])
    wh_f = wh.to(torch.float)
    new_width = torch.ones_like(wh[0]) * max_width
    new_height = new_width.to(torch.float) / (wh_f[0] / wh_f[1])
    result_wh = torch.where(
        wh[0] <= new_width, wh, torch.tensor([new_width, new_height.to(new_width.dtype)])
    )
    return result_wh[0], result_wh[1]


def scale_down_for_live_video(tensor: torch.Tensor, max_width: int = MAX_VIDEO_WIDTH):
    assert tensor.ndim == 4 and (tensor.shape[-1] == 3 or tensor.shape[-1] == 4)
    h = tensor.shape[1]
    w = tensor.shape[2]
    w, h, resized = video_size(width=w, height=h, max_width=max_width)
    if resized:
        return resize_image(tensor, new_height=h, new_width=w)
    return tensor


def yuv_to_bgr_float(
    frames: torch.Tensor, dtype: torch.dtype = torch.float16, non_blocking: bool = True
):
    """
    Current HW decode returns only YUV
    """
    frames = make_channels_first(frames).contiguous()
    if not torch.is_floating_point(frames):
        frames = frames.to(dtype, non_blocking=non_blocking)

    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    # u = u / 255 - 0.5
    # v = v / 255 - 0.5

    u /= 255
    u -= 0.5

    v /= 255
    v -= 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([b, g, r], -1)
    rgb *= 255
    # rgb = (rgb * 255).clamp(0, 255)
    rgb.clamp_(0, 255)
    return rgb


def time_to_frame(time_str: str, fps: float):
    # Split the time duration string into components
    h = 0
    m = 0
    s = 0
    tokens = time_str.split(":")
    s = float(tokens[-1])
    if len(tokens) > 1:
        m = int(tokens[-2])
        if len(tokens) > 2:
            assert len(tokens) == 3
            h = int(tokens[0])
    # Extract seconds and milliseconds
    # Convert hours, minutes, seconds, and milliseconds to total seconds
    total_seconds = h * 3600 + m * 60 + s
    # Calculate the frame number
    frame_number = int(total_seconds * fps)
    return frame_number


class VideoStreamWriter(VideoStreamWriterInterface):
    @typechecked
    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        codec: str,
        format: str = "bgr24",
        batch_size: int = 3,
        cache_size: int = 4,
        bit_rate: int = int(55e6),
        device: torch.device = None,
        lossless: bool = int(os.environ.get("HM_LOSSLESS_OUT", "0")) != 0,
        container_type: str = "matroska",
        local_resize: bool = True,
        streaming_drop_frame_interval: int = 3,
        stream_fps: int = 15,
    ):
        assert False, "Stop using VideoStreamWriter; use PyNvVideoEncoderWriter instead"
        self._filename = filename
        self._container_type = container_type
        # Always keep a float internally even if callers pass a Fraction
        self._fps = float(fps)

        self._stream_fps = stream_fps
        self._stream_frame_indexes = {
            int(i)
            for i in np.linspace(0, np.round(self._fps) - 1, self._stream_fps, endpoint=False)
        }
        self._width = width
        self._height = height
        self._streaming_drop_frame_interval = streaming_drop_frame_interval
        self._streaming_drop_frame_interval_counter = 0
        self._codec = codec
        self._streaming = False
        self._local_resize = local_resize
        self._timer = Timer()
        if self._filename.startswith("rtmp://"):
            self._format = format
            self._codec = "hevc_nvenc"
            self._container_type = "flv"
            self._streaming = True
        elif self._filename.startswith("udp://"):
            self._format = "mpegts"
            self._container_type = "mpegts"
            self._streaming = True
        else:
            self._format = format

        if self._streaming and self._local_resize:
            self._width, self._height, _ = video_size(width=self._width, height=self._height)

        self._video_out = None
        self._video_f = None
        self._device = device
        self._lossless = lossless
        self._cache_size = cache_size
        assert batch_size >= 1
        self._batch_size = batch_size
        self._batch_count = 0
        self._batch_items = []
        self._in_flush = False
        self._bit_rate = bit_rate
        _, stream_writer_cls, codec_config_cls = _load_torchaudio_io()
        self._streaming_media_encoder_cls = stream_writer_cls
        self._codec_config = codec_config_cls(
            bit_rate=bit_rate,
        )
        self._frame_counter = 0

    def __enter__(self):
        if self._video_f is None:
            self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _make_proper_permute(self, image: torch.Tensor):
        if len(image.shape) == 3:
            if image.shape[-1] == 3 and self._device is not None:
                if isinstance(image, torch.Tensor):
                    image = image.permute(2, 0, 1)
                else:
                    image = image.transpose(2, 0, 1)
        else:
            if image.shape[-1] == 3 and self._device is not None:
                if not isinstance(image, np.ndarray):
                    image = image.permute(0, 3, 1, 2)
                else:
                    image = image.transpose(0, 3, 1, 2)
        assert image.shape[-2] == self._height
        assert image.shape[-1] == self._width
        return image

    def _add_stream(self):
        if self._lossless:
            rate_control = "constqp"
        else:
            rate_control = "cbr"
        rate_m = self._bit_rate // 1_000_000
        options = {
            "rc": rate_control,
            "minrate": f"{rate_m}M",
            "maxrate": f"{rate_m}M",
            "bufsize": f"{rate_m}M",
        }
        if self._lossless:
            options["qp"] = "0"

        if self._filename.startswith("rtmp://"):
            new_w, new_h, needs_resize = video_size(width=self._width, height=self._height)
            assert not self._local_resize or not needs_resize
            hw_accel = None
            if self._codec.endswith("_nvenc"):
                hw_accel = str(self._device)
            # Cut down the bit_rate
            self._codec_config.bit_rate //= 4
            self._video_out.add_video_stream(
                frame_rate=self._stream_fps,
                format="bgr24",
                encoder=self._codec,
                encoder_format="bgr0",
                encoder_width=self._width,
                encoder_height=self._height,
                height=new_h,
                width=new_w,
                codec_config=self._codec_config,
                hw_accel=hw_accel,
            )
        else:
            self._video_out.add_video_stream(
                frame_rate=self._fps,
                height=self._height,
                width=self._width,
                format=self._format,
                encoder=self._codec,
                encoder_format="bgr0",
                encoder_option=options,
                encoder_frame_rate=self._fps,
                codec_config=self._codec_config,
                hw_accel=str(self._device),
            )

    def close(self):
        if self._video_f is not None:
            self._finish()
            self._video_f.close()
            self._video_f = None
            print(f"VideoStreamWriter wrote {self._frame_counter} frames")

    def release(self):
        self.close()

    def _finish(self):
        self.flush(flush_video_file=True, flush_all=True)

    def flush(self, flush_video_file: bool = True, flush_all: bool = False):
        def _get_tensor(t):
            if isinstance(t, StreamTensorBase):
                # return t.get()
                return t.wait()
            return t

        if self._batch_items:
            if flush_all:
                batch_items = self._batch_items
                self._batch_items = []
                if not batch_items:
                    return
            elif len(self._batch_items) <= self._cache_size:
                # Don't have at least cache_size items yet
                return
            else:
                flush_item_count = len(self._batch_items) - self._cache_size
                assert flush_item_count
                batch_items = self._batch_items[0:flush_item_count]
                self._batch_items = self._batch_items[flush_item_count:]
            if not batch_items:
                return
            batch_items = [_get_tensor(img) for img in batch_items]
            if len(batch_items) == 1:
                image_batch = batch_items[0]
                if len(image_batch.shape) == 3:
                    image_batch = image_batch.unsqueeze(0)
            else:
                if len(batch_items[0].shape) == 3:
                    image_batch = torch.stack(batch_items)
                else:
                    image_batch = torch.cat(batch_items, dim=0)
            frame_count = len(image_batch)
            self._video_out.write_video_chunk(
                i=0,
                chunk=image_batch,
            )
            self._frame_counter += frame_count

        if flush_video_file and self._video_f is not None:
            self._video_f.flush()

    def isOpened(self) -> bool:
        return self._video_f is not None

    def open(self):
        assert self._video_f is None
        if not self._streaming:
            ext = _EXTENSION_MAPPING.get(self._container_type, self._container_type)
            if not self._filename.endswith("." + ext):
                self._filename += "." + ext
        self._video_out = self._streaming_media_encoder_cls(
            dst=self._filename, format=self._container_type
        )
        self._add_stream()
        self._video_f = self._video_out.open()

    def set(self, key: int, value: any):
        pass

    def get(self, key: int) -> any:
        return None

    def append(self, images: torch.Tensor):
        assert images.device == self._device
        if self._streaming:
            frame = self._streaming_drop_frame_interval_counter
            self._streaming_drop_frame_interval_counter += 1
            if (frame % int(self._fps)) not in self._stream_frame_indexes:
                assert images.shape[0] == 1
                return
            if self._local_resize:
                images = scale_down_for_live_video(images)
            assert images.device == self._device
            if not self._codec.endswith("_nvenc") and images.device.type != "cpu":
                images = images.cpu()
        self._batch_items.append(self._make_proper_permute(images))
        self.flush(flush_video_file=False)
        self._batch_count += 1
        if self._timer.start_time != 0:
            self._timer.toc()
        if self._batch_count % 100 == 0:
            logger.info(
                f"Writing output ({len(images) * 1.0 / max(1e-5, self._timer.average_time):.2f} fps)"
            )
            self._timer = Timer()
        self._timer.tic()

    def write(self, images: torch.Tensor):
        return self.append(images)


class GStreamerVideoReaderIterator:
    def __init__(
        self,
        video_source,
        device: torch.device,
        batch_size: int = 1,
        format: str = "rgb8",
        timeout_ms: int = 5000,
    ):
        self._video_source = video_source
        self._batch_size = batch_size
        self._format = format
        self._device = device
        self._timeout_ms = timeout_ms
        self.frames_delivered_count = 0

    def _capture_tensor(self) -> torch.Tensor | None:
        frame = self._video_source.Capture(format=self._format, timeout=self._timeout_ms)
        if frame is None:
            return None
        tensor = torch.as_tensor(frame, device=self._device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.shape[-1] == 4:
            tensor = tensor[..., :3]
        tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] == 3:
            tensor = tensor[[2, 1, 0], ...]
        tensor = tensor.contiguous()
        return tensor

    def __next__(self):
        frames = []
        for _ in range(self._batch_size):
            tensor = self._capture_tensor()
            if tensor is None:
                if not frames:
                    raise StopIteration()
                break
            frames.append(tensor)
            self.frames_delivered_count += 1
        if len(frames) == 1:
            return frames[0].unsqueeze(0)
        return torch.stack(frames, dim=0)

    def __iter__(self):
        return self


class PyNvVideoCodecIterator:
    def __init__(
        self,
        decoder,
        device: torch.device,
        cuda_stream: torch.cuda.Stream,
        batch_size: int = 1,
    ):
        self._decoder = decoder
        self._batch_size = batch_size
        self._device = device
        self._cuda_stream = cuda_stream
        self.frames_delivered_count: int = 0

    def __next__(self):
        with torch.cuda.stream(self._cuda_stream):
            frames = self._decoder.get_batch_frames(self._batch_size)
            if not frames:
                raise StopIteration()

            batch_tensors = []
            for frame in frames:
                planes = frame.cuda()
                if planes is None:
                    continue
                plane_tensors = [torch.as_tensor(p, device=self._device) for p in planes]
                if len(plane_tensors) >= 3:
                    # RGB->BGR
                    tmp = plane_tensors[0]
                    plane_tensors[0] = plane_tensors[2]
                    plane_tensors[2] = tmp
                tensor = torch.stack(plane_tensors, dim=0)
                batch_tensors.append(tensor)

            if not batch_tensors:
                raise StopIteration()

            self.frames_delivered_count += len(batch_tensors)
            return wrap_tensor(torch.stack(batch_tensors, dim=0))

    def __iter__(self):
        return self


class PyAmdVideoCodecIterator:
    def __init__(self, decoder: PyAmdVideoDecoder, batch_size: int = 1):
        self._decoder = decoder
        self._batch_size = batch_size
        self.frames_delivered_count: int = 0

    def __next__(self):
        batch = self._decoder.read_batch(self._batch_size)
        if batch is None:
            raise StopIteration()
        batch_ref = batch.ref() if isinstance(batch, StreamTensorBase) else batch
        self.frames_delivered_count += int(batch_ref.shape[0])
        return batch

    def __iter__(self):
        return self


class CVVideoCaptureIterator:
    def __init__(self, cap: cv2.VideoCapture, batch_size: int = 1):
        self._cap = cap
        self._batch_size = batch_size
        self.frames_delivered_count = 0

    def __next__(self):
        if self._batch_size == 1:
            res, frame = self._cap.read()
            if not res:
                raise StopIteration()
            self.frames_delivered_count += 1
            return np.expand_dims(frame.transpose(2, 0, 1), axis=0)
        else:
            frames = []
            for _ in range(self._batch_size):
                res, frame = self._cap.read()
                if not res:
                    raise StopIteration()
                frames.append(frame.transpose(2, 0, 1))
            self.frames_delivered_count += self._batch_size
            return np.stack(frames)

    def __del__(self):
        if hasattr(self, "frames_delivered_count") and self.frames_delivered_count:
            logger.info("CVVideoCaptureIterator delivered %d frames", self.frames_delivered_count)


class TAStreamReaderIterator:
    def __init__(self, sr: "StreamingMediaDecoder", batch_size: int = 1):
        self._iter = sr.stream()
        self._batch_size = batch_size
        self._device = None

    def __next__(self):
        with torch.no_grad():
            if self._device is None:
                torch.cuda.synchronize()
            else:
                torch.cuda.current_stream(self._device).synchronize()
            next_chunk = next(self._iter)
            if next_chunk is None:
                raise StopIteration()
            assert len(next_chunk) == 1
            frame = next_chunk[0]
            assert len(frame) == self._batch_size
            if self._device is None:
                self._device = frame.device
            current_stream = torch.cuda.current_stream(frame.device)

            current_stream.synchronize()

            frame = yuv_to_bgr_float(frame.clone())
            # current_stream.synchronize()
            # show_image(f"frame{str(id(self))}", frame, wait=False, enable_resizing=0.25)
            # show_image(f"frame{str(id(self))}", frame, wait=False)
        return frame


class FFMpegVideoReader:
    pass


class FFmpegVideoReaderIterator:
    def __init__(
        self,
        filename: str,
        gpu_index: int,
        time_s: float,
        format: str,
        vid_info: BasicVideoInfo,
        batch_size: int = 1,
    ):
        self._process = get_ffmpeg_decoder_process(
            input_video=filename, gpu_index=gpu_index, format=format, time_s=time_s
        )
        assert self._process is not None
        self._vid_info = vid_info
        self._channels = 3
        self._count = 0
        self._batch_size = batch_size

    def __del__(self):
        self._process.terminate()

    def __next__(self):
        if self._count:
            # Skip to the next frame in the buffer
            self._process.stdout.flush()
        raw_image = self._process.stdout.read(
            self._batch_size * self._vid_info.width * self._vid_info.height * self._channels
        )

        if not raw_image:
            self._process.terminate()
            raise StopIteration()

        frame = torch.frombuffer(buffer=raw_image, dtype=torch.uint8).reshape(
            (
                self._batch_size,
                self._vid_info.height,
                self._vid_info.width,
                self._channels,
            )
        )
        self._count += 1
        # Make channels-first
        return frame.permute(0, 3, 1, 2)


#
# VideoStreamReader
#
class VideoStreamReader:
    """Unified batched video reader supporting multiple backends."""

    def __init__(
        self,
        filename: str,
        type: str,
        codec: str | None = None,
        batch_size: int = 1,
        device: torch.device = None,
        pynvc_simple_decoder: bool | None = None,
    ):
        self._filename = filename
        self._type = _normalize_decoder_name(type)
        if os.environ.get("HM_SHARED_HIGH_BIT_DEPTH") == "1":
            self._type = "shared-high-bit"
        self._codec = codec
        self._fps = None
        self._width = None
        self._height = None
        self._batch_size = batch_size
        if device is None:
            device = torch.device("cpu")
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._video_in = None
        self._video_info = None
        self._iter = None
        self._ss = 0.0
        self._torchaudio_stream = False
        self._gstreamer_stream = False
        self._frames_delivered_count = 0
        self._cuda_stream: torch.cuda.Stream | None = None
        self._pynvc_simple_decoder = pynvc_simple_decoder
        self.open()

    @property
    def fps(self):
        return self._video_info.fps

    @property
    def width(self):
        return self._video_info.width

    @property
    def height(self):
        return self._video_info.height

    @property
    def bit_rate(self):
        return self._video_info.bit_rate

    @property
    def frame_count(self):
        return self._video_info.frame_count

    @property
    def codec(self):
        return self._video_info.codec

    def __len__(self):
        return self.frame_count

    def __iter__(self):
        if self._type == "shared-high-bit":
            from hmlib.video.high_bit_reader import HighBitVideoReader

            if self._iter is not None:
                self._iter.close()
            self._iter = HighBitVideoReader(
                self._filename,
                self.width,
                self.height,
                seconds=self._ss,
                batch_size=self._batch_size,
            )
            return self._iter
        elif self._type == "torchaudio":
            return TAStreamReaderIterator(self._video_in, batch_size=self._batch_size)
        elif self._type == "cv2":
            return CVVideoCaptureIterator(self._video_in, batch_size=self._batch_size)
        elif self._type == "gstreamer":
            return GStreamerVideoReaderIterator(
                self._video_in,
                batch_size=self._batch_size,
                device=self._device,
                timeout_ms=5000,
            )
        elif self._type == "pynvcodec":
            assert self._cuda_stream is not None
            return PyNvVideoCodecIterator(
                self._video_in,
                device=self._device,
                cuda_stream=self._cuda_stream,
                batch_size=self._batch_size,
            )
        elif self._type == "pyamdcodec":
            return PyAmdVideoCodecIterator(self._video_in, batch_size=self._batch_size)
        elif self._type == "ffmpeg":
            return FFmpegVideoReaderIterator(
                filename=self._filename,
                gpu_index=self._device.index if self._device is not None else 0,
                time_s=self._ss,
                format="bgr24",
                vid_info=self._video_info,
                batch_size=self._batch_size,
            )
        else:
            assert False

    def _add_stream(self):
        # Add the video stream
        hw_accel = None
        format = None
        decoder = None
        decoder_options: dict[str, str] = {}
        if self._device.type == "cuda":
            # hw_accel = "cuda"
            decoder = _FOURCC_TO_CODEC[self._video_info.codec.upper()]
            hw_accel = str(self._device)
        self._video_in.add_basic_video_stream(
            frames_per_chunk=self._batch_size,
            stream_index=0,
            decoder=decoder,
            decoder_option=decoder_options,
            format=format,
            hw_accel=hw_accel,
            buffer_chunk_size=-1,
        )

    def isOpened(self):
        return self._video_in is not None

    def seek(self, timestamp: float | None = None, frame_number: int | None = None):
        assert timestamp is None or frame_number is None
        if frame_number is not None:
            timestamp = float(frame_number) / self.fps
        else:
            frame_number = timestamp * self.fps
        if self._torchaudio_stream:
            self._video_in.seek(timestamp=timestamp, mode="precise")
        elif isinstance(self._video_in, cv2.VideoCapture):
            if isinstance(frame_number, Fraction):
                frame_number = frame_number.numerator / frame_number.denominator
            self._video_in.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )
        elif isinstance(self._video_in, FFMpegVideoReader):
            self._ss = timestamp
        elif self._gstreamer_stream:
            logger.warning("GStreamer backend does not currently support seeking; ignoring request")
        elif self._type == "pynvcodec":
            if timestamp is not None and frame_number is None:
                frame_number = int(self._video_in.get_index_from_time_in_seconds(timestamp))
            assert frame_number is not None
            nvc = _load_pynvcodec()
            if isinstance(self._video_in, nvc.SimpleDecoder):
                self._video_in.seek_to_index(
                    self._video_in.get_index_from_time_in_seconds(timestamp)
                )
            else:
                if frame_number:
                    new_args = self._video_in_args.copy()
                    new_args["start_frame"] = round(frame_number)
                    self._video_in.end()
                    torch.cuda.synchronize()
                    self._video_in = nvc.ThreadedDecoder(**new_args)
                    torch.cuda.synchronize()
        elif self._type == "pyamdcodec":
            if timestamp is not None and frame_number is None:
                frame_number = int(self._video_in.get_index_from_time_in_seconds(timestamp))
            assert frame_number is not None
            self._video_in.seek_to_index(int(frame_number))
        else:
            assert False

    def set(self, prop: int, value: any):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seek(frame_number=value)
        else:
            raise ValueError(f"Unsupported property: {prop}")

    def get(self, prop: int):
        assert False

    def open(self):
        assert self._video_in is None
        self._video_info = BasicVideoInfo(self._filename)
        if self._type == "shared-high-bit":
            self._video_in = FFMpegVideoReader()
            return
        if (self._video_info.width % 2 or self._video_info.height % 2) and self._type != "cv2":
            logger.warning(
                "Decoding exact odd-sized canvas %sx%s with CPU OpenCV; hardware YUV420 decoders cannot preserve this raster: %s",
                self._video_info.width,
                self._video_info.height,
                self._filename,
            )
            self._type = "cv2"
        if self._codec is None and self._type not in ("ffmpeg", "pynvcodec", "pyamdcodec"):
            self._codec = _FOURCC_TO_CODEC.get(self._video_info.codec.upper(), None)
            if self._codec is None and self._type != "cv2":
                print(
                    f"VideoStreamReader is changing decoder from {self._type} "
                    f"to cv2 due to video's codec type: {self._video_info.codec}"
                )
                self._type = "cv2"
        if self._type == "torchaudio":
            stream_reader_cls, _, _ = _load_torchaudio_io()
            self._video_in = stream_reader_cls(src=self._filename)
            self._torchaudio_stream = True
            self._add_stream()
        elif self._type == "cv2":
            self._video_in = cv2.VideoCapture(self._filename)
            if not self._video_in.isOpened():
                self._video_in.release()
                self._video_in = None
        elif self._type == "gstreamer":
            if self._device.type != "cuda":
                raise AssertionError(
                    "GStreamer backend requires a CUDA device (DeepStream decoder outputs to GPU)"
                )
            ju = _load_jetson_utils()
            codec_str = _jetson_codec_from_fourcc(self._video_info.codec) or "h264"
            arch = platform.machine().lower()
            codec_type = "v4l2" if arch == "aarch64" else "nvdec"
            if arch != "aarch64":
                os.environ.setdefault("JETSON_UTILS_NVDEC", "1")
            options = {
                "width": int(self._video_info.width),
                "height": int(self._video_info.height),
                "framerate": float(self._video_info.fps),
                "codec": codec_str,
                "codecType": codec_type,
                "zeroCopy": False,
            }
            self._video_in = ju.videoSource(self._filename, options=options)
            try:
                opts = self._video_in.GetOptions()
                if isinstance(opts, dict) and opts.get("codecType", "").lower() == "cpu":
                    logger.warning(
                        "GStreamer backend is using CPU decode; hardware decode may require DeepStream/V4L2 plugins."
                    )
            except AttributeError as error:
                logger.warning("GStreamer decoder cannot report its backend options: %s", error)
            self._gstreamer_stream = True
        elif self._type == "pynvcodec":
            if self._device.type != "cuda":
                raise AssertionError(
                    "PyNvVideoCodec backend requires a CUDA device (set decoder-device to cuda:N)."
                )
            nvc = _load_pynvcodec()
            gpu_id = self._device.index if self._device.index is not None else 0
            if self._cuda_stream is None:
                self._cuda_stream = torch.cuda.Stream(device=torch.device("cuda", index=gpu_id))
            if self._pynvc_simple_decoder:
                self._video_in_args: dict[str, Any] = {
                    "enc_file_path": self._filename,
                    "gpu_id": gpu_id,
                    "use_device_memory": True,
                    "max_width": int(self._video_info.width),
                    "max_height": int(self._video_info.height),
                    "need_scanned_stream_metadata": 0,
                    "output_color_type": nvc.OutputColorType.RGBP,
                    "cuda_stream": self._cuda_stream.cuda_stream,
                }
                self._video_in = nvc.SimpleDecoder(**self._video_in_args)
            else:
                self._video_in_args: dict[str, Any] = {
                    "enc_file_path": self._filename,
                    "buffer_size": self._batch_size * 2,
                    "gpu_id": gpu_id,
                    "use_device_memory": True,
                    "max_width": int(self._video_info.width),
                    "max_height": int(self._video_info.height),
                    "need_scanned_stream_metadata": 0,
                    "output_color_type": nvc.OutputColorType.RGBP,
                    "cuda_stream": self._cuda_stream.cuda_stream,
                }
                self._video_in = nvc.ThreadedDecoder(**self._video_in_args)
        elif self._type == "pyamdcodec":
            self._video_in = PyAmdVideoDecoder(
                self._filename,
                device=self._device,
                batch_size=self._batch_size,
            )
        elif self._type == "ffmpeg":
            self._video_in = FFMpegVideoReader()
        else:
            assert False

    def close(self):
        if self._video_in is not None:
            if self._type == "shared-high-bit" and self._iter is not None:
                self._iter.close()
            elif self._torchaudio_stream:
                self._video_in.remove_stream(0)
            elif isinstance(self._video_in, cv2.VideoCapture):
                self._video_in.release()
            elif self._gstreamer_stream and hasattr(self._video_in, "Close"):
                self._video_in.Close()
            elif self._type == "pynvcodec":
                if hasattr(self._video_in, "end"):
                    self._video_in.end()
            elif self._type == "pyamdcodec":
                self._video_in.end()
            elif isinstance(self._video_in, FFMpegVideoReader):
                pass
            else:
                assert False
            if self._video_in is not None and hasattr(self._video_in, "frames_delivered_count"):
                print(f"VideoStreamReader delivered {self._video_in.frames_delivered_count} frames")
            self._video_in = None
            self._iter = None
            self._gstreamer_stream = False

    def read(self):
        next_data = next(self._iter)
        if next_data is None:
            return False, None
        return True, next_data


class PyNvVideoEncoderWriter(VideoStreamWriterInterface):
    """
    VideoStreamWriterInterface adapter around PyNvVideoEncoder.

    This writer keeps all raw frames on CUDA, uses jetson-utils for BGR->YUV420
    conversion, encodes with NVENC via PyNvVideoCodec, and muxes the stream with
    ffmpeg into the requested container (MP4/MKV).
    """

    def __init__(
        self,
        filename: str,
        fps: Fraction,
        width: int,
        height: int,
        codec: str | None,
        device: torch.device,
        bit_rate: int = int(55e6),
        batch_size: int | None = 1,
        profiler: Any = None,
        mux_audio_file: str | None = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
        encoder_backend: str | None = None,
    ):
        if device is None or device.type != "cuda":
            raise AssertionError("PyNvVideoEncoderWriter requires a CUDA device.")
        from hmlib.video.py_nv_encoder import PyNvVideoEncoder

        self._filename = filename
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)
        self._bit_rate = int(bit_rate)
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._batch_size = batch_size or 1
        self._profiler = profiler

        codec_lower = (codec or "hevc_nvenc").lower()
        encoder_codec = "h264"
        if "hevc" in codec_lower or "h265" in codec_lower:
            encoder_codec = "hevc"
        elif "av1" in codec_lower:
            encoder_codec = "av1"

        use_pyav: bool | None
        backend = (encoder_backend or "").lower()
        if backend in {"", "auto"}:
            use_pyav = None
        elif backend == "pyav":
            use_pyav = True
        elif backend in {"raw", "ffmpeg"}:
            use_pyav = False
        else:
            raise ValueError(
                f"Unsupported NVENC encoder backend {encoder_backend!r}; "
                "supported backends are: auto, raw, ffmpeg, pyav."
            )

        self._encoder = PyNvVideoEncoder(
            output_path=self._filename,
            width=self._width,
            height=self._height,
            fps=self._fps,
            codec=encoder_codec,
            preset="P2",
            device=self._device,
            gpu_id=self._device.index or 0,
            cuda_stream=(
                torch.cuda.current_stream(self._device).cuda_stream
                if self._device is not None
                else None
            ),
            bitrate=self._bit_rate,
            mux_audio_file=mux_audio_file,
            mux_audio_stream=mux_audio_stream,
            mux_audio_offset_seconds=mux_audio_offset_seconds,
            mux_audio_aac_bitrate=mux_audio_aac_bitrate,
            use_pyav=use_pyav,
            profiler=self._profiler,
        )
        self._frame_counter = 0

    def isOpened(self) -> bool:
        return getattr(self._encoder, "_opened", False)

    @property
    def device(self) -> torch.device:
        return self._device

    def open(self):
        self._encoder.open()

    def append(self, images: torch.Tensor, **kwargs):
        prof = self._profiler
        ctx = (
            prof.rf("video.nvenc_writer.append")
            if getattr(prof, "enabled", False)
            else contextlib.nullcontext()
        )
        with ctx:
            assert images.device == self._device
            self._encoder.write(images, **kwargs)

            if images.ndim == 4:
                self._frame_counter += int(images.shape[0])
            else:
                self._frame_counter += 1

    def write(self, images: torch.Tensor, **kwargs):
        return self.append(images, **kwargs)

    def flush(self):
        # Flushing is handled in close() via encoder.EndEncode()/ffmpeg drain.
        pass

    def close(self):
        if getattr(self._encoder, "_opened", False):
            self._encoder.close()


class PyAmdVideoEncoderWriter(VideoStreamWriterInterface):
    """VideoStreamWriterInterface adapter around PyAmdVideoEncoder."""

    def __init__(
        self,
        filename: str,
        fps: Fraction,
        width: int,
        height: int,
        codec: str | None,
        device: torch.device,
        bit_rate: int = int(55e6),
        batch_size: int | None = 1,
        profiler: Any = None,
        mux_audio_file: str | None = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
    ):
        self._filename = filename
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)
        self._bit_rate = int(bit_rate)
        self._device = device if isinstance(device, torch.device) else torch.device(device or "cpu")
        self._batch_size = batch_size or 1
        self._profiler = profiler

        codec_lower = (codec or PyAmdVideoCodec.preferred_output_codec()).lower()
        encoder_codec = "h264"
        if "hevc" in codec_lower or "h265" in codec_lower:
            encoder_codec = "hevc"
        elif "av1" in codec_lower:
            encoder_codec = "av1"

        backend = (
            "vaapi"
            if codec_lower.endswith("_vaapi")
            else "amf" if codec_lower.endswith("_amf") else "auto"
        )
        self._encoder = PyAmdVideoEncoder(
            output_path=self._filename,
            width=self._width,
            height=self._height,
            fps=self._fps,
            codec=encoder_codec,
            device=self._device,
            backend=backend,
            bitrate=self._bit_rate,
            mux_audio_file=mux_audio_file,
            mux_audio_stream=mux_audio_stream,
            mux_audio_offset_seconds=mux_audio_offset_seconds,
            mux_audio_aac_bitrate=mux_audio_aac_bitrate,
            profiler=self._profiler,
        )
        self._frame_counter = 0

    def isOpened(self) -> bool:
        return getattr(self._encoder, "_opened", False)

    @property
    def device(self) -> torch.device:
        return self._device

    def open(self):
        self._encoder.open()

    def append(self, images: torch.Tensor | StreamTensorBase, **kwargs):
        assert not isinstance(images, torch.Tensor) or images.device == self._device
        self._encoder.write(images, **kwargs)
        img_ref = images.ref() if isinstance(images, StreamTensorBase) else images
        if img_ref.ndim == 4:
            self._frame_counter += int(img_ref.shape[0])
        else:
            self._frame_counter += 1

    def write(self, images: torch.Tensor | StreamTensorBase, **kwargs):
        return self.append(images, **kwargs)

    def flush(self):
        pass

    def close(self):
        if getattr(self._encoder, "_opened", False):
            self._encoder.close()


class VideoStreamWriterCV2(VideoStreamWriterInterface):
    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        device: torch.device,
        codec: str | None,
        bit_rate: int | None = None,
        batch_size: int | None = 1,
    ):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._bit_rate = bit_rate
        self._output_video = cv2.VideoWriter(
            filename=filename,
            fourcc=fourcc,
            fps=fps,
            frameSize=(
                int(width),
                int(height),
            ),
        )
        self._output_video.set(
            cv2.CAP_PROP_BITRATE,
            self.calculate_desired_bitrate(width=width, height=height),
        )
        assert self._output_video.isOpened()
        self._device = torch.device("cpu")

    def isOpened(self) -> bool:
        return self._output_video is not None and self._output_video.isOpened()

    def calculate_desired_bitrate(self, width: int, height: int):
        # 4K @ 55M
        desired_bit_rate = self._bit_rate
        if not desired_bit_rate:
            desired_bit_rate_per_pixel = 55e6 / (3840 * 2160)
            desired_bit_rate = int(desired_bit_rate_per_pixel * width * height)
            logger.info(
                f"Desired bit rate for output video ({int(width)} x {int(height)}): {desired_bit_rate//1000} kb/s"
            )
        return desired_bit_rate

    def open(self):
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    def flush(self):
        pass

    def close(self):
        self._output_video.release()
        self._output_video = None

    def write(self, img: torch.Tensor | np.ndarray, *, frame_ids=None):
        if isinstance(img, StreamTensorBase):
            img = img.get()
        if img.ndim == 4:
            assert img.shape[0] == 1  # batch size of one only
            img = img.squeeze(0)
        # OpenCV always wants channels last
        img = make_channels_last(img)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            img = np.ascontiguousarray(img)
        if isinstance(img, np.ndarray) and img.dtype != np.dtype("uint8"):
            img = img.astype("uint8")
        self._output_video.write(img)


@typechecked
def create_output_video_stream(
    filename: str,
    fps: float | Fraction,
    width: int,
    height: int,
    codec: str | None,
    device: torch.device,
    bit_rate: int = int(55e6),
    batch_size: int | None = 1,
    profiler: Any = None,
    mux_audio_file: str | None = None,
    mux_audio_stream: int = 0,
    mux_audio_offset_seconds: float = 0.0,
    mux_audio_aac_bitrate: str = "192k",
    encoder_backend: str | None = None,
) -> VideoStreamWriterInterface:
    # Normalize fps to a plain float so downstream writers and backends
    # never see Fraction instances.
    fps_val = float(fps)
    if width % 2 or height % 2:
        from hmlib.video.exact_raster_writer import ExactRasterWriter

        logger.warning(
            "Encoding exact odd-sized canvas %sx%s with software H.264 4:4:4; YUV420 would crop or pad tracking coordinates: %s",
            width,
            height,
            filename,
        )
        return ExactRasterWriter(
            filename=filename,
            fps=fps_val,
            width=int(width),
            height=int(height),
            bit_rate=int(bit_rate),
            audio=(
                {
                    "source": mux_audio_file,
                    "stream": mux_audio_stream,
                    "offset_seconds": mux_audio_offset_seconds,
                    "bitrate": mux_audio_aac_bitrate,
                }
                if mux_audio_file
                else None
            ),
        )
    codec_lower = (codec or "").lower()
    # use_pynvcodec_env = os.environ.get("HM_VIDEO_ENCODER", "").lower() == "pynvcodec"
    use_pynvcodec_env = True
    if codec_lower.endswith(("_vaapi", "_amf")):
        output_video = PyAmdVideoEncoderWriter(
            filename=filename,
            fps=fps,
            height=int(height),
            width=int(width),
            codec=codec,
            bit_rate=bit_rate,
            device=device,
            batch_size=batch_size,
            profiler=profiler,
            mux_audio_file=mux_audio_file,
            mux_audio_stream=mux_audio_stream,
            mux_audio_offset_seconds=mux_audio_offset_seconds,
            mux_audio_aac_bitrate=mux_audio_aac_bitrate,
        )
        output_video.open()
    elif ("_nvenc" in codec_lower) and use_pynvcodec_env:
        output_video = PyNvVideoEncoderWriter(
            filename=filename,
            fps=fps,
            height=int(height),
            width=int(width),
            codec=codec,
            bit_rate=bit_rate,
            device=device,
            batch_size=batch_size,
            profiler=profiler,
            mux_audio_file=mux_audio_file,
            mux_audio_stream=mux_audio_stream,
            mux_audio_offset_seconds=mux_audio_offset_seconds,
            mux_audio_aac_bitrate=mux_audio_aac_bitrate,
            encoder_backend=encoder_backend,
        )
        output_video.open()
    elif "_nvenc" in codec_lower or filename.startswith("rtmp://"):
        output_video = VideoStreamWriter(
            filename=filename,
            fps=fps_val,
            height=int(height),
            width=int(width),
            codec=codec if codec is not None else "hevc_nvenc",
            bit_rate=bit_rate,
            device=device,
            batch_size=batch_size,
        )
        output_video.open()
    else:
        output_video = VideoStreamWriterCV2(
            filename=filename,
            fps=fps_val,
            height=int(height),
            width=int(width),
            codec=codec,
            bit_rate=bit_rate,
            device=device,
            batch_size=batch_size,
        )
        output_video.open()
    return output_video


def extract_frame_image(
    source_video: str,
    frame_number: float,
    dest_image: str,
    stream_type: str = "cv2",
    # stream_type: str = "torchaudio",
) -> torch.Tensor | np.ndarray:
    print(f"Extracting frame {frame_number} from {source_video}...")
    reader = VideoStreamReader(
        filename=source_video,
        type=stream_type,
        device=torch.device("cuda", 0) if stream_type == "torchaudio" else None,
    )
    try:
        if frame_number:
            reader.seek(frame_number=frame_number)
        iterator = iter(reader)
        image = next(iterator)
    finally:
        reader.close()
    image = make_channels_last(image)
    if dest_image:
        img = image
        if isinstance(image, torch.Tensor):
            img = img.cpu().numpy()
        assert img.ndim == 4 and img.shape[0] == 1
        img = np.squeeze(img, axis=0)
        cv2.imwrite(dest_image, img)
    return image


if __name__ == "__main__":

    reader = VideoStreamReader(
        filename=f"{os.environ['HOME']}/Videos/ev-bs-short/GX010005.MP4",
        type="torchaudio",
        device=torch.device("cuda", 0),
    )
    iterator = iter(reader)
    image = next(iterator)
    show_image("image", image, wait=True)

    print("Done")
