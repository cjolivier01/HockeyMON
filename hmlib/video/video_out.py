"""High-level stitched video writer and visualization utilities.

This module coordinates GPU streams, color transforms, overlays and IO to
produce the final rendered videos used by many CLIs.

@see @ref hmlib.video.video_stream "video_stream" for the underlying encoder.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import math
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hmlib.log import logger
from hmlib.ui.shower import Shower
from hmlib.utils import MeanTracker
from hmlib.utils.cuda_graph import CudaGraphCallable
from hmlib.utils.gpu import get_gpu_capabilities, unwrap_tensor, wrap_tensor
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
    resize_image,
    to_uint8_image,
)
from hmlib.utils.path import add_suffix_to_filename
from hmlib.utils.progress_bar import ProgressBar
from hmlib.utils.torch_backend import is_rocm_backend
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH

from .py_amd_codec import PyAmdVideoCodec
from .video_stream import VideoStreamWriterInterface, create_output_video_stream

standard_8k_width: int = 7680
standard_8k_height: int = 4320


def get_and_pop(map: Dict[str, Any], key: str) -> Any:
    result = map.get(key, None)
    if result is not None:
        del map[key]
    return result


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        try:
            x = float(s)
        except ValueError:
            return False
        return math.isfinite(x)
    return False


def get_best_codec(
    gpu_number: int, width: int, height: int, allow_scaling: bool = False
) -> Tuple[str, bool]:
    if is_rocm_backend():
        try:
            return PyAmdVideoCodec.preferred_output_codec(), True
        except RuntimeError:
            return "XVID", False
    caps = get_gpu_capabilities()
    compute = float(caps[gpu_number]["compute_capability"])
    if compute >= 7 and (width <= MAX_NEVC_VIDEO_WIDTH or allow_scaling):
        return "hevc_nvenc", True
        # return "h264_nvenc", True
        # return "av1_nvenc", True
    elif compute >= 6 and width <= 4096:
        return "hevc_nvenc", True
        # return "h264_nvenc", True
        # return "av1_nvenc", True
    else:
        return "XVID", False
    # return "XVID", False


def is_nearly_8k(width, height, size_tolerance=0.10, aspect_ratio_tolerance=0.01):
    """
    Checks if a given width and height are within a configurable tolerance of 8K
    resolution and have a very similar aspect ratio.

    Args:
        width (int): The width dimension to check.
        height (int): The height dimension to check.
        size_tolerance (float): The maximum allowed percentage difference from 8K
                                dimensions (e.g., 0.10 for 10%).
        aspect_ratio_tolerance (float): The maximum allowed absolute difference
                                       in aspect ratio.

    Returns:
        tuple: A boolean indicating if the dimensions and aspect ratio match,
               and a string with details on the outcome.
    """
    # Define the reference 8K dimensions and aspect ratio
    ref_8k_width, ref_8k_height = standard_8k_width, standard_8k_height
    ref_aspect_ratio = ref_8k_width / ref_8k_height

    # Check if dimensions are within 10% of 8K
    width_ok = ref_8k_width * (1 - size_tolerance) <= width <= ref_8k_width * (1 + size_tolerance)
    height_ok = (
        ref_8k_height * (1 - size_tolerance) <= height <= ref_8k_height * (1 + size_tolerance)
    )

    # Check if the aspect ratio is very close
    try:
        current_aspect_ratio = width / height
        aspect_ratio_ok = math.isclose(
            current_aspect_ratio, ref_aspect_ratio, rel_tol=aspect_ratio_tolerance
        )
    except ZeroDivisionError:
        return False, "Height cannot be zero."

    # Return the combined result
    if width_ok and height_ok and aspect_ratio_ok:
        return True, "Dimensions are within 10% of 8K and have a very close aspect ratio."
    else:
        details = []
        if not width_ok:
            details.append(
                f"Width ({width}) is not within {size_tolerance*100}% of 8K width ({ref_8k_width})."
            )
        if not height_ok:
            details.append(
                f"Height ({height}) is not within {size_tolerance*100}% of 8K height ({ref_8k_height})."
            )
        if not aspect_ratio_ok:
            details.append(
                f"Aspect ratio ({current_aspect_ratio:.4f}) is not very close to 8K ({ref_aspect_ratio:.4f})."
            )

        return False, " and ".join(details)


_FP_TYPES: Set[torch.dtype] = {
    torch.float,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.half,
}


class VideoOutput(torch.nn.ModuleDict):
    """Synchronous video writer for final HockeyMON frames.

    This module owns the lifecycle of one or more output video streams and
    encapsulates tensor-to-video conversion and IO. It does **not** perform
    camera logic, cropping, or overlays; those are handled upstream
    (e.g., :class:`hmlib.camera.apply_camera_plugin.ApplyCameraPlugin`).

    Typical usage::

        video_out = VideoOutput(...)
        video_out = video_out.to(device)
        for results in dataset:
            # results must include:
            #   - "img": tensor[B, H, W, C] or StreamTensorBase
            #   - "frame_ids": tensor[B] (1-based ids)
            # optionally:
            #   - "end_zone_img": tensor[B, H, W, C]
            video_out(results)

    The return value is the updated ``results`` dict; most callers ignore it.
    """

    VIDEO_DEFAULT: str = "default"
    VIDEO_END_ZONES: str = "end_zones"

    def __init__(
        self,
        output_video_path: str,
        fps: float,
        fourcc: str = "auto",
        bit_rate: int = int(55e6),
        mux_audio_file: Optional[str] = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
        save_frame_dir: str | None = None,
        name: str = "",
        simple_save: bool = False,
        skip_final_save: bool = False,
        progress_bar: ProgressBar | None = None,
        cache_size: int = 2,
        clip_to_max_dimensions: bool = True,
        visualization_config: Dict[str, Any] | None = None,
        dtype: torch.dtype | None = None,
        device: Union[torch.device, str, None] = None,
        output_width: int | str | None = None,
        output_height: int | str | None = None,
        show_image: bool = False,
        show_scaled: Optional[float] = None,
        show_youtube: bool = False,
        youtube_stream_url: Optional[str] = None,
        youtube_stream_key: Optional[str] = None,
        headless_preview_host: str = "0.0.0.0",
        headless_preview_port: int = 0,
        always_stream: bool = False,
        profiler: Any = None,
        enable_end_zones: bool = False,
        encoder_backend: Optional[str] = None,
    ):
        """Construct a synchronous video writer.

        @param output_video_path: Destination filename for the main output video
                                  (e.g., ``tracking_output.mkv``). When
                                  ``skip_final_save`` is True, no file is written.
        @param fps: Frames per second to encode the output stream with.
        @param fourcc: Codec identifier. Use ``"auto"`` to pick a codec based
                       on GPU capabilities (e.g., ``"hevc_nvenc"`` or ``"XVID"``).
        @param bit_rate: Target bitrate in bits per second for the encoder.
        @param save_frame_dir: Optional directory for saving individual frames
                               as PNGs alongside the encoded video.
        @param name: Human-readable name used in logs (e.g., ``"TRACKING"``).
        @param simple_save: When True, disables some dynamic scaling logic and
                            assumes input frames are already at output size.
        @param skip_final_save: When True, the underlying writer is not asked to
                                finalize/flush the container at shutdown.
        @param image_channel_adjustment: Deprecated per-channel adjustment; kept
                                         for API compatibility but not used.
        @param print_interval: Interval (in frames) at which throughput logs can
                               be emitted (currently used only via callers).
        @param original_clip_box: Optional crop box applied earlier in the pipeline;
                                  used only for naming/logging, not for IO here.
        @param progress_bar: Optional :class:`ProgressBar` instance used by callers
                             to surface IO/throughput metrics.
        @param cache_size: Reserved for historical async batching; currently only
                           used to size the optional UI shower cache.
        @param clip_to_max_dimensions: When ``simple_save`` is True, scales video
                                      down to encoder-specific maximums (e.g., 8K).
        @param visualization_config: Reserved for future visualization options
                                     (not currently consumed).
        @param dtype: Preferred floating-point dtype for any internal tensors that
                      may be created (defaults to ``torch.get_default_dtype()``).
        @param device: Torch device or device string (e.g., ``"cuda:0"`` or
                       ``"cpu"``) on which encoding should run. When ``None``,
                       the device is inferred from input tensors at first call.
        @param output_width: Optional width (in pixels) to resize the final
                             rendered frames to before encoding. Aspect ratio
                             is preserved.
        @param output_height: Optional height (in pixels) to resize/letterbox
                              the final rendered frames to before encoding.
        @param show_image: When True, enables an interactive OpenCV window that
                           displays frames as they are written.
        @param show_scaled: Optional scale factor for the interactive viewer.
        @param show_youtube: When True, publish preview frames to a YouTube
                             RTMP(S) ingest URL instead of only local display.
        @param youtube_stream_url: Base YouTube ingest URL or a full RTMP(S)
                                   publish URL.
        @param youtube_stream_key: Stream key appended to
                                   ``youtube_stream_url`` when needed.
        @param headless_preview_host: Listen host for the browser-based
                                      fallback preview server.
        @param headless_preview_port: Listen port for the browser-based
                                      fallback preview server. Use ``0`` for an
                                      ephemeral free port.
        @param profiler: Optional profiler object exposing ``rf(label)`` for
                         scoped timings; see :mod:`hmlib.utils.profiler`.
        @param game_config: Optional game configuration dict; kept for parity with
                            older APIs but not consumed directly by this class.
        @param enable_end_zones: When True, a secondary ``VIDEO_END_ZONES`` stream
                                 will be opened and any ``"end_zone_img"`` tensors
                                 present in ``results`` will be written there.
        """
        super().__init__()
        self._allow_scaling = False
        self._clip_to_max_dimensions = clip_to_max_dimensions
        self._visualization_config = visualization_config
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        assert self._dtype in _FP_TYPES
        self._mux_audio_file = str(mux_audio_file) if mux_audio_file else None
        self._mux_audio_stream = int(mux_audio_stream or 0)
        self._mux_audio_offset_seconds = float(mux_audio_offset_seconds or 0.0)
        self._mux_audio_aac_bitrate = str(mux_audio_aac_bitrate or "192k")
        self._encoder_backend = (
            None if (not encoder_backend or encoder_backend == "auto") else encoder_backend
        )
        self._output_width = output_width
        self._output_height = output_height
        self._output_resize_wh: Optional[Tuple[int, int]] = None
        self._output_canvas_wh: Optional[Tuple[int, int]] = None

        if device is not None:
            self._device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self._device = None
        self._name = name
        self._simple_save = simple_save
        # Normalize to float even if upstream passes a Fraction
        self._fps = fps
        self._skip_final_save = skip_final_save
        self._progress_bar = progress_bar
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._output_videos: Dict[str, VideoStreamWriterInterface] = {}

        self._bit_rate = bit_rate
        self._enable_end_zones: bool = bool(enable_end_zones)
        self._last_frame_id: Optional[torch.Tensor] = None
        self._cuda_graph_enabled: bool = False
        self._img_prepare_cg: Optional[CudaGraphCallable] = None
        self._img_prepare_cg_signature: Optional[
            Tuple[torch.device, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]
        ] = None
        self._end_zone_prepare_cg: Optional[CudaGraphCallable] = None
        self._end_zone_prepare_cg_signature: Optional[
            Tuple[torch.device, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]
        ] = None

        self._fourcc = fourcc

        self._prof = profiler
        self._fctx = (
            self._prof.rf("video_out.forward")
            if getattr(self._prof, "enabled", False)
            else contextlib.nullcontext()
        )
        self._sctx = (
            self._prof.rf("video_out._save_frame")
            if getattr(self._prof, "enabled", False)
            else contextlib.nullcontext()
        )

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        self._show_image = bool(show_image)
        self._show_scaled = show_scaled
        self._show_youtube = bool(show_youtube)
        self._youtube_stream_url = youtube_stream_url
        self._youtube_stream_key = youtube_stream_key
        self._headless_preview_host = str(headless_preview_host or "0.0.0.0")
        self._headless_preview_port = int(headless_preview_port or 0)
        self._always_stream = bool(always_stream)
        self._shower = (
            Shower(
                label="Video Out",
                show_scaled=self._show_scaled,
                max_size=cache_size,
                profiler=self._prof,
                enable_local_display=self._show_image,
                show_youtube=self._show_youtube,
                youtube_stream_url=self._youtube_stream_url,
                youtube_stream_key=self._youtube_stream_key,
                headless_preview_host=self._headless_preview_host,
                headless_preview_port=self._headless_preview_port,
                always_stream=self._always_stream,
            )
            if (self._show_image or self._show_youtube)
            else None
        )
        self._progress_bar_stream_callback_installed = False
        self._attach_stream_status_progress_callback()

        self._mean_tracker: Optional[MeanTracker] = None

    def _attach_stream_status_progress_callback(self) -> None:
        if (
            self._progress_bar is None
            or self._shower is None
            or self._progress_bar_stream_callback_installed
        ):
            return

        def _table_callback(table_map: OrderedDict[Any, Any]) -> None:
            self._shower.update_progress_table(table_map)

        self._progress_bar.add_table_callback(_table_callback)
        self._progress_bar_stream_callback_installed = True

    def _parse_output_dim(self, value: Any, dim_label: str, natural: int) -> Optional[int]:
        if value is None:
            return None
        if not is_number(value):
            key = str(value).strip().lower()
            if key == "auto":
                return None
            if dim_label == "width":
                if key == "4k":
                    return 4096
                if key == "8k":
                    return 8192
            if dim_label == "height":
                if key == "4k":
                    return 2160
                if key == "8k":
                    return 4320
            raise ValueError(f'Invalid output {dim_label}: "{value}"')
        return int(value)

    def _coerce_even(self, value: int, label: str) -> int:
        if value % 2 != 0:
            adjusted = value + 1
            logger.warning(
                "output_%s=%d is not even; adjusting to %d for YUV420 encoders",
                label,
                value,
                adjusted,
            )
            return adjusted
        return value

    def _coerce_even_down(self, value: int, label: str) -> int:
        if value % 2 == 0:
            return value
        adjusted = value - 1 if value > 2 else value + 1
        logger.warning(
            "output_%s=%d is not even; adjusting to %d for YUV420 encoders",
            label,
            value,
            adjusted,
        )
        return adjusted

    @staticmethod
    def _is_auto_output_dim(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() == "auto"
        return False

    def _get_auto_resize_and_canvas(
        self, width: int, height: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        if self._skip_final_save or not self._clip_to_max_dimensions:
            return None, None
        if not self._is_auto_output_dim(self._output_width) or not self._is_auto_output_dim(
            self._output_height
        ):
            return None, None

        resize_w = int(width)
        resize_h = int(height)
        if resize_w > MAX_NEVC_VIDEO_WIDTH:
            scale = float(MAX_NEVC_VIDEO_WIDTH) / float(resize_w)
            resize_w = MAX_NEVC_VIDEO_WIDTH
            resize_h = int(float(resize_h) * scale)

        resize_w = self._coerce_even_down(resize_w, "width")
        resize_h = self._coerce_even_down(resize_h, "height")
        if resize_w == int(width) and resize_h == int(height):
            return None, None

        logger.info(
            "Auto-resizing output from %dx%d to %dx%d for encoder compatibility",
            int(width),
            int(height),
            resize_w,
            resize_h,
        )
        return (resize_w, resize_h), None

    def _get_output_resize_and_canvas(
        self, width: int, height: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Return (resize_wh, canvas_wh) for output scaling/letterboxing."""
        target_w = self._parse_output_dim(self._output_width, "width", width)
        target_h = self._parse_output_dim(self._output_height, "height", height)

        if target_w is None and target_h is None:
            return self._get_auto_resize_and_canvas(width, height)

        if target_w is None:
            if target_h is None or target_h <= 0:
                raise ValueError(f"output_height must be positive; got {target_h}")
            scale = float(target_h) / float(height)
            resize_w = int(round(float(width) * scale))
            resize_h = int(target_h)
            resize_w = self._coerce_even(resize_w, "width")
            resize_h = self._coerce_even(resize_h, "height")
            if resize_w == int(width) and resize_h == int(height):
                return None, None
            return (resize_w, resize_h), None

        if target_h is None:
            if target_w <= 0:
                raise ValueError(f"output_width must be positive; got {target_w}")
            target_w = self._coerce_even(int(target_w), "width")
            scale = float(target_w) / float(width)
            resize_h = int(round(float(height) * scale))
            resize_h = self._coerce_even(resize_h, "height")
            if target_w == int(width) and resize_h == int(height):
                return None, None
            return (target_w, resize_h), None

        target_w = self._coerce_even(int(target_w), "width")
        target_h = self._coerce_even(int(target_h), "height")
        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                f"output_width/output_height must be positive; got {target_w}x{target_h}"
            )
        scale = min(float(target_w) / float(width), float(target_h) / float(height))
        resize_w = int(round(float(width) * scale))
        resize_h = int(round(float(height) * scale))
        resize_w = self._coerce_even(max(1, resize_w), "width")
        resize_h = self._coerce_even(max(1, resize_h), "height")
        resize_w = min(resize_w, target_w)
        resize_h = min(resize_h, target_h)
        if resize_w == target_w and resize_h == target_h:
            if resize_w == int(width) and resize_h == int(height):
                return None, None
            return (resize_w, resize_h), None
        return (resize_w, resize_h), (target_w, target_h)

    @staticmethod
    def _letterbox_tensor(img: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
        """Pad a channels-last tensor to (target_w, target_h) with black bars."""
        w = image_width(img)
        h = image_height(img)
        pad_w = max(0, int(target_w) - int(w))
        pad_h = max(0, int(target_h) - int(h))
        if pad_w == 0 and pad_h == 0:
            return img
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        x = make_channels_first(img)
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], "constant", 0)
        return make_channels_last(x)

    def _reset_prepare_cuda_graphs(self) -> None:
        self._img_prepare_cg = None
        self._img_prepare_cg_signature = None
        self._end_zone_prepare_cg = None
        self._end_zone_prepare_cg_signature = None

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if not self._cuda_graph_enabled:
            self._reset_prepare_cuda_graphs()
        return True

    @staticmethod
    def _normalize_input_image(img: Any) -> Any:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(img, torch.Tensor) and img.ndim == 3:
            img = img.unsqueeze(0)
        return img

    def _prepare_output_layout(
        self, results: Dict[str, Any], online_im: Any
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        resize_wh = self._output_resize_wh
        canvas_wh = self._output_canvas_wh
        if self._output_width is None and self._output_height is None:
            return resize_wh, canvas_wh
        if resize_wh is None and canvas_wh is None:
            resize_wh, canvas_wh = self._get_output_resize_and_canvas(
                width=image_width(online_im),
                height=image_height(online_im),
            )
            self._output_resize_wh = resize_wh
            self._output_canvas_wh = canvas_wh
            self._reset_prepare_cuda_graphs()
        output_wh = canvas_wh or resize_wh
        if output_wh is not None:
            video_frame_cfg = results.get("video_frame_cfg")
            if isinstance(video_frame_cfg, dict):
                target_w, target_h = output_wh
                video_frame_cfg_local = dict(video_frame_cfg)
                video_frame_cfg_local["output_frame_width"] = target_w
                video_frame_cfg_local["output_frame_height"] = target_h
                video_frame_cfg_local["output_aspect_ratio"] = float(target_w) / float(target_h)
                results["video_frame_cfg"] = video_frame_cfg_local
        return resize_wh, canvas_wh

    def _apply_output_transforms(
        self,
        img: Any,
        resize_wh: Optional[Tuple[int, int]],
        canvas_wh: Optional[Tuple[int, int]],
    ) -> Any:
        out = img
        if resize_wh is not None:
            target_w, target_h = resize_wh
            if image_width(out) != target_w or image_height(out) != target_h:
                out_t = unwrap_tensor(out)
                out_t = resize_image(
                    img=out_t,
                    new_width=target_w,
                    new_height=target_h,
                )
                out = to_uint8_image(out_t).contiguous()
        if canvas_wh is not None:
            target_w, target_h = canvas_wh
            out = self._letterbox_tensor(unwrap_tensor(out), target_w, target_h)
            out = to_uint8_image(out).contiguous()
        return out

    def _maybe_move_to_output_device(self, img: Any) -> Any:
        out = img
        if not self._skip_final_save and self._device is not None:
            if str(out.device) != str(self._device):
                out = unwrap_tensor(out).to(self._device)
            if out.device.type != "cpu" and self._device.type == "cpu":
                out = out.to("cpu", non_blocking=True)
                out = wrap_tensor(out)
            else:
                assert self._device is None or out.device == self._device
        return out

    def _graph_prepare_tensor(
        self,
        img: Any,
        *,
        resize_wh: Optional[Tuple[int, int]],
        canvas_wh: Optional[Tuple[int, int]],
        slot: str,
    ) -> Any:
        if resize_wh is None and canvas_wh is None:
            return img
        if not self._cuda_graph_enabled:
            return self._apply_output_transforms(img, resize_wh, canvas_wh)
        if self._device is None or self._device.type != "cuda":
            return self._apply_output_transforms(img, resize_wh, canvas_wh)
        img_t = unwrap_tensor(img)
        if not isinstance(img_t, torch.Tensor) or img_t.device.type != "cuda":
            return self._apply_output_transforms(img, resize_wh, canvas_wh)
        signature = (img_t.device, resize_wh, canvas_wh)
        if slot == self.VIDEO_DEFAULT:
            cg = self._img_prepare_cg
            cg_sig = self._img_prepare_cg_signature
        else:
            cg = self._end_zone_prepare_cg
            cg_sig = self._end_zone_prepare_cg_signature
        if cg is None or cg_sig != signature:
            cg = CudaGraphCallable(
                lambda t: self._apply_output_transforms(t, resize_wh, canvas_wh),
                (img_t,),
                warmup=0,
                name=f"video_out_{slot}_prep",
            )
            if slot == self.VIDEO_DEFAULT:
                self._img_prepare_cg = cg
                self._img_prepare_cg_signature = signature
            else:
                self._end_zone_prepare_cg = cg
                self._end_zone_prepare_cg_signature = signature
        return cg(img_t)

    def _prepare_single_image(
        self,
        img: Any,
        *,
        resize_wh: Optional[Tuple[int, int]],
        canvas_wh: Optional[Tuple[int, int]],
        slot: str,
    ) -> Any:
        out = self._normalize_input_image(img)
        out = self._maybe_move_to_output_device(out)
        return self._graph_prepare_tensor(out, resize_wh=resize_wh, canvas_wh=canvas_wh, slot=slot)

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        """Resolve device and codec configuration before the first write.

        This method is intentionally idempotent and cheap to call. It:
          - Infers the writer device from the registered output-size buffers
            when no explicit device was provided.
          - Resolves ``"auto"`` codecs to concrete GPU/CPU codecs using
            :func:`get_best_codec`.
        """
        if self._device is None:
            img = context.get("img")
            if img is None:
                img = context.get("end_zone_img")
            img = self._normalize_input_image(img)
            img_t = unwrap_tensor(img) if img is not None else None
            if isinstance(img_t, torch.Tensor):
                self._device = img_t.device
            else:
                self._device = torch.device("cpu")
        if self._fourcc == "auto":
            video_frame_cfg = context["video_frame_cfg"]
            output_frame_width = int(video_frame_cfg["output_frame_width"])
            output_frame_height = int(video_frame_cfg["output_frame_height"])
            if self._device.type == "cuda":
                device_index = self._device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                self._fourcc, is_gpu = get_best_codec(
                    device_index,
                    width=output_frame_width,
                    height=output_frame_height,
                    allow_scaling=self._allow_scaling,
                )
                if not is_gpu:
                    logger.info(f"Can't use GPU for output video {self._output_video_path}")
                    self._device = torch.device("cpu")
                    self._reset_prepare_cuda_graphs()
            else:
                self._fourcc = "XVID"
            logger.info(
                f"Output video {self._name} {output_frame_width}x"
                f"{output_frame_height} will use codec: {self._fourcc}"
            )

    def set_progress_bar(self, progress_bar: ProgressBar):
        """Attach a progress bar instance used for external UI updates."""
        self._progress_bar = progress_bar
        self._attach_stream_status_progress_callback()

    def start(self):
        """Legacy no-op; synchronous VideoOutput does not require start()."""
        return None

    def stop(self):
        """Close any interactive UI resources (e.g., OpenCV shower)."""
        for stream in self._output_videos.values():
            stream.close()
        self._output_videos.clear()
        if self._shower is not None:
            self._shower.close()
            self._shower = None

    def create_output_videos(self, context: Dict[str, Any]) -> None:
        """Create underlying VideoStreamWriter instances if not already open."""
        if self._output_video_path and not self._skip_final_save:
            video_frame_cfg = context["video_frame_cfg"]
            output_frame_width = int(video_frame_cfg["output_frame_width"])
            output_frame_height = int(video_frame_cfg["output_frame_height"])
            if self.VIDEO_DEFAULT not in self._output_videos:
                self._output_videos[self.VIDEO_DEFAULT] = create_output_video_stream(
                    filename=self._output_video_path,
                    fps=self._fps,
                    height=output_frame_height,
                    width=output_frame_width,
                    codec=self._fourcc,
                    bit_rate=self._bit_rate,
                    device=self._device,
                    batch_size=1,
                    profiler=self._prof,
                    mux_audio_file=self._mux_audio_file,
                    mux_audio_stream=self._mux_audio_stream,
                    mux_audio_offset_seconds=self._mux_audio_offset_seconds,
                    mux_audio_aac_bitrate=self._mux_audio_aac_bitrate,
                    encoder_backend=self._encoder_backend,
                )
                assert self._output_videos[self.VIDEO_DEFAULT].isOpened()

            if self._enable_end_zones and self.VIDEO_END_ZONES not in self._output_videos:
                self._output_videos[self.VIDEO_END_ZONES] = create_output_video_stream(
                    filename=str(add_suffix_to_filename(self._output_video_path, "-end-zones")),
                    fps=self._fps,
                    height=output_frame_height,
                    width=output_frame_width,
                    codec=self._fourcc,
                    bit_rate=self._bit_rate,
                    device=self._device,
                    batch_size=1,
                    profiler=self._prof,
                    encoder_backend=self._encoder_backend,
                )
                assert self._output_videos[self.VIDEO_END_ZONES].isOpened()

    def prepare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(results)
        online_im = prepared["img"]
        online_im = self._normalize_input_image(online_im)

        resize_wh, canvas_wh = self._prepare_output_layout(prepared, online_im)

        self._ensure_initialized(prepared)

        prepared["img"] = self._prepare_single_image(
            online_im,
            resize_wh=resize_wh,
            canvas_wh=canvas_wh,
            slot=self.VIDEO_DEFAULT,
        )

        ez_img = prepared.get("end_zone_img")
        if ez_img is not None:
            prepared["end_zone_img"] = self._prepare_single_image(
                ez_img,
                resize_wh=resize_wh,
                canvas_wh=canvas_wh,
                slot=self.VIDEO_END_ZONES,
            )
        return prepared

    def _validate_frame_ids(self, results: Dict[str, Any]) -> None:
        frame_ids = results["frame_ids"]
        if self._last_frame_id is not None and torch.any(frame_ids <= self._last_frame_id):
            raise ValueError(
                "VideoOutput received non-monotonic frame_ids. "
                f"last_frame_id={self._last_frame_id}, "
                f"current_frame_ids={frame_ids}"
            )
        self._last_frame_id = frame_ids.max()

    def write_prepared_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized(results)
        self._validate_frame_ids(results)
        if not self._output_videos:
            self.create_output_videos(results)
        with self._sctx:
            out = self._save_frame(results)
        assert "img" not in out
        return out

    def _save_frame(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Write a batch of frames to the configured video streams.

        Expected keys in ``context`` on entry:
          - ``"img"``: tensor[B, H, W, C] or StreamTensorBase
          - ``"frame_ids"``: tensor[B] (used for logging and PNG naming)
          - ``"end_zone_img"``: optional tensor[B, H, W, C] for far-end output

        The ``"img"`` key is consumed and reattached in-place.
        """

        # Consume the image tensor
        online_im = context.pop("img")

        image_w = image_width(online_im)
        image_h = image_height(online_im)
        assert online_im.ndim == 4  # Should have a batch dimension

        assert online_im.dtype == torch.uint8
        assert online_im.is_contiguous

        video_frame_cfg = context["video_frame_cfg"]
        output_frame_width = int(video_frame_cfg["output_frame_width"])
        output_frame_height = int(video_frame_cfg["output_frame_height"])

        # Output (and maybe show) the final image
        assert int(output_frame_width) == image_w
        assert int(output_frame_height) == image_h

        online_im = unwrap_tensor(online_im, verbose=True)

        if self._mean_tracker is not None:
            self._mean_tracker(online_im)

        if self._show_image and self._shower is not None:
            for show_img in online_im:
                self._shower.show(show_img, clone=True)

        # torch.cuda.synchronize()
        # torch.cuda.current_stream(online_im.device).synchronize()

        if not self._skip_final_save:
            if self.VIDEO_DEFAULT in self._output_videos:
                self._output_videos[self.VIDEO_DEFAULT].write(
                    online_im, frame_ids=context.get("frame_ids")
                )

            if self.VIDEO_END_ZONES in self._output_videos:
                ez_img = context.get("end_zone_img")
                if ez_img is None:
                    ez_img = online_im
                self._output_videos[self.VIDEO_END_ZONES].write(
                    ez_img, frame_ids=context.get("frame_ids")
                )
        else:
            # Sync the stream if skipping final save
            # online_im = wrap_tensor(online_im, verbose=True).get()
            pass

        # Save frames as individual frames
        if self._save_frame_dir:
            # frame_id should start with 1
            assert context["frame_ids"][0] != 0
            cv2.imwrite(
                os.path.join(
                    self._save_frame_dir,
                    "frame_{:06d}.png".format(int(context["frame_ids"][0]) - 1),
                ),
                online_im,
            )
        return context

    def forward(self, results: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        """Normalize input images and synchronously write them to disk.

        This is the primary public API. It:
          1. Lazily initializes device/codec/streams.
          2. Opens output video writers on first use.
          3. Writes the frames via :meth:`_save_frame`.

        @param results: Dict containing at least ``"img"`` and ``"frame_ids"``.
        @return: The updated ``results`` dict (for chaining if desired).
        """
        with self._fctx:
            prepared = self.prepare_results(results)
            return self.write_prepared_results(prepared)
