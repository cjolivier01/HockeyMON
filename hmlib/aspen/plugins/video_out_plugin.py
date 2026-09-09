from __future__ import annotations

import os
from typing import Any, Dict, Optional

from hmlib.builder import HM
from hmlib.config import get_nested_value, normalize_runtime_config
from hmlib.utils.path import add_prefix_to_filename
from hmlib.video.video_out import VideoOutput

from .base import Plugin


@HM.register_module()
class VideoOutPlugin(Plugin):
    """Sink plugin that writes prepared frames, with legacy prep fallback."""

    disable_in_cuda_graph_pipeline = True

    def __init__(
        self,
        enabled: bool = True,
        output_video_path: Optional[str] = None,
        cache_size: int = 2,
        preview_via_sink: bool = True,
        video_out_device: Optional[str] = None,
        skip_final_save: bool = False,
        save_frame_dir: Optional[str] = None,
        encoder_backend: Optional[str] = None,
        mux_audio_file: Optional[str] = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
        output_width: Optional[str | int] = None,
        output_height: Optional[str | int] = None,
        show_image: Optional[bool] = None,
        show_scaled: Optional[float] = None,
        show_youtube: Optional[bool] = None,
        youtube_stream_url: Optional[str] = None,
        youtube_stream_key: Optional[str] = None,
        headless_preview_host: Optional[str] = None,
        headless_preview_port: Optional[int] = None,
        always_stream: Optional[bool] = None,
        bit_rate: Optional[int] = None,
    ) -> None:
        """Construct the Aspen video sink plugin.

        This plugin owns the stateful pieces of video output: writer creation,
        frame-id validation, optional muxing, optional UI display, and final
        frame emission. When the prep plugin is present upstream, this plugin
        should mostly just consume already-prepared tensors.

        @param enabled: Whether this plugin should execute.
        @param output_video_path: Explicit output path or basename override.
        @param cache_size: Small writer/UI cache size carried through to
                           :class:`VideoOutput`.
        @param preview_via_sink: When False, keep this plugin write-only and
                                 leave preview/streaming to a separate sink.
        @param video_out_device: Optional preferred writer device.
        @param skip_final_save: When True, suppress final encoded output writes.
        @param save_frame_dir: Optional PNG frame dump directory.
        @param encoder_backend: Optional encoder backend override.
        @param mux_audio_file: Optional audio file to mux into the output.
        @param mux_audio_stream: Audio stream index to mux from the source.
        @param mux_audio_offset_seconds: Audio offset applied during muxing.
        @param mux_audio_aac_bitrate: AAC bitrate when re-encoding audio.
        @param output_width: Optional final output width for legacy sink-only use.
        @param output_height: Optional final output height for legacy sink-only use.
        @param show_image: Whether to show frames live during writing.
        @param show_scaled: Optional scale factor for the live preview window.
        @param show_youtube: Whether to publish preview frames to a YouTube
                             RTMP(S) ingest URL.
        @param youtube_stream_url: Base YouTube ingest URL or a full RTMP(S)
                                   publish URL.
        @param youtube_stream_key: Stream key appended to the YouTube ingest URL.
        @param headless_preview_host: Listen host for browser-based fallback
                                      preview when no display is available.
        @param headless_preview_port: Listen port for browser-based fallback
                                      preview when no display is available.
        @param bit_rate: Optional target encoded video bitrate.
        """
        super().__init__(enabled=enabled)
        self._vo: Optional[VideoOutput] = None
        self._out_path = output_video_path
        self._cache = int(cache_size)
        self._preview_via_sink = bool(preview_via_sink)
        self._vo_dev = video_out_device
        self._skip_final_save = bool(skip_final_save)
        self._save_dir = save_frame_dir
        self._encoder_backend = encoder_backend
        self._mux_audio_file = mux_audio_file
        self._mux_audio_stream = int(mux_audio_stream or 0)
        self._mux_audio_offset_seconds = float(mux_audio_offset_seconds or 0.0)
        self._mux_audio_aac_bitrate = str(mux_audio_aac_bitrate or "192k")
        self._output_width = output_width
        self._output_height = output_height
        self._show_image = bool(show_image) if show_image is not None else None
        self._show_scaled = show_scaled
        self._show_youtube = bool(show_youtube) if show_youtube is not None else None
        self._youtube_stream_url = youtube_stream_url
        self._youtube_stream_key = youtube_stream_key
        self._headless_preview_host = headless_preview_host
        self._headless_preview_port = headless_preview_port
        self._always_stream = bool(always_stream) if always_stream is not None else None
        self._bit_rate = bit_rate

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if self._vo is not None and hasattr(self._vo, "set_cuda_graph_enabled"):
            self._vo.set_cuda_graph_enabled(enabled)
        return True

    def _resolve_output_path(self, cfg: Dict[str, Any], context: Dict[str, Any]) -> str:
        out_path = self._out_path
        if not out_path or "/" not in out_path:
            # Prefer the resolved game config path, but still allow a bare
            # filename override to land under the active work directory.
            candidate = get_nested_value(cfg, "video_out.output_video_path", default_value=None)
            if not candidate:
                candidate = get_nested_value(
                    cfg, "aspen.video_out.output_video_path", default_value=None
                )
            if not candidate:
                work_dir = context.get("work_dir") or os.path.join(os.getcwd(), "output_workdirs")
                candidate = os.path.join(str(work_dir), out_path or "tracking_output.mkv")
            out_path = candidate
        shared = context.get("shared", {})
        label = None
        if isinstance(shared, dict):
            label = shared.get("output_label") or shared.get("label")
        if label:
            try:
                out_path = str(add_prefix_to_filename(out_path, str(label)))
            except Exception:
                pass
        return str(out_path)

    def _resolve_fps(self, context: Dict[str, Any]) -> float:
        shared = context.get("shared", {})
        fps = None
        try:
            fps_val = context.get("fps")
            if fps_val is None and isinstance(shared, dict):
                fps_val = shared.get("fps")
            fps = float(fps_val) if fps_val is not None else None
        except Exception:
            fps = None
        return 30.0 if fps is None else float(fps)

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._vo is not None:
            return
        shared = context.get("shared", {})
        img = context.get("img")
        if img is None:
            img = context.get("original_images")
        assert img is not None, "VideoOutPlugin requires 'img' in context"

        device = shared.get("device") if isinstance(shared, dict) else None
        cfg = shared.get("game_config") if isinstance(shared, dict) else {}
        cfg = cfg or {}
        normalize_runtime_config(cfg)
        out_path = self._resolve_output_path(cfg, context)
        self._out_path = out_path
        # If the sink is used without the prep plugin, it still needs enough
        # layout information to prepare frames correctly on its own.
        vo_dev = self._vo_dev if self._vo_dev is not None else device
        fps = self._resolve_fps(context)

        bit_rate = self._bit_rate
        progress_bar = shared.get("progress_bar") if isinstance(shared, dict) else None
        if bit_rate is None:
            bit_rate = get_nested_value(cfg, "video_out.bit_rate", default_value=None)
        if bit_rate is None:
            bit_rate = get_nested_value(cfg, "aspen.video_out.bit_rate", default_value=None)
        if bit_rate is None:
            bit_rate = int(55e6)

        mux_audio_file = self._mux_audio_file
        mux_audio_stream = self._mux_audio_stream
        mux_audio_offset_seconds = self._mux_audio_offset_seconds
        mux_audio_aac_bitrate = self._mux_audio_aac_bitrate
        if isinstance(shared, dict):
            if not mux_audio_file:
                mux_audio_file = shared.get("mux_audio_file")
            if shared.get("mux_audio_stream") is not None:
                mux_audio_stream = int(shared.get("mux_audio_stream") or 0)
            if shared.get("mux_audio_offset_seconds") is not None:
                mux_audio_offset_seconds = float(shared.get("mux_audio_offset_seconds") or 0.0)
            if shared.get("mux_audio_aac_bitrate") is not None:
                mux_audio_aac_bitrate = str(shared.get("mux_audio_aac_bitrate") or "192k")

        show_image = False
        show_scaled = None
        show_youtube = False
        youtube_stream_url = None
        youtube_stream_key = None
        headless_preview_host = "0.0.0.0"
        headless_preview_port = 0
        always_stream = False
        if self._preview_via_sink:
            show_image = bool(
                self._show_image
                if self._show_image is not None
                else get_nested_value(
                    cfg,
                    "video_out.show_image",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.show_image", default_value=False
                    ),
                )
            )
            show_scaled = (
                self._show_scaled
                if self._show_scaled is not None
                else get_nested_value(
                    cfg,
                    "video_out.show_scaled",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.show_scaled", default_value=None
                    ),
                )
            )
            show_youtube = bool(
                self._show_youtube
                if self._show_youtube is not None
                else get_nested_value(
                    cfg,
                    "video_out.show_youtube",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.show_youtube", default_value=False
                    ),
                )
            )
            youtube_stream_url = (
                self._youtube_stream_url
                if self._youtube_stream_url is not None
                else get_nested_value(
                    cfg,
                    "video_out.youtube_stream_url",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.youtube_stream_url", default_value=None
                    ),
                )
            )
            youtube_stream_key = (
                self._youtube_stream_key
                if self._youtube_stream_key is not None
                else get_nested_value(
                    cfg,
                    "video_out.youtube_stream_key",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.youtube_stream_key", default_value=None
                    ),
                )
            )
            headless_preview_host = (
                self._headless_preview_host
                if self._headless_preview_host is not None
                else get_nested_value(
                    cfg,
                    "video_out.headless_preview_host",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.headless_preview_host", default_value="0.0.0.0"
                    ),
                )
            )
            headless_preview_port = (
                self._headless_preview_port
                if self._headless_preview_port is not None
                else get_nested_value(
                    cfg,
                    "video_out.headless_preview_port",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.headless_preview_port", default_value=0
                    ),
                )
            )
            always_stream = bool(
                self._always_stream
                if self._always_stream is not None
                else get_nested_value(
                    cfg,
                    "video_out.always_stream",
                    default_value=get_nested_value(
                        cfg, "aspen.video_out.always_stream", default_value=False
                    ),
                )
            )

        self._vo = VideoOutput(
            output_video_path=out_path,
            fps=fps,
            bit_rate=bit_rate,
            mux_audio_file=mux_audio_file,
            mux_audio_stream=mux_audio_stream,
            mux_audio_offset_seconds=mux_audio_offset_seconds,
            mux_audio_aac_bitrate=mux_audio_aac_bitrate,
            save_frame_dir=self._save_dir,
            name="TRACKING",
            skip_final_save=self._skip_final_save,
            progress_bar=progress_bar,
            cache_size=self._cache,
            device=vo_dev,
            output_width=self._output_width,
            output_height=self._output_height,
            show_image=show_image,
            show_scaled=show_scaled,
            show_youtube=show_youtube,
            youtube_stream_url=youtube_stream_url,
            youtube_stream_key=youtube_stream_key,
            headless_preview_host=headless_preview_host,
            headless_preview_port=headless_preview_port,
            always_stream=always_stream,
            profiler=shared.get("profiler", None) if isinstance(shared, dict) else None,
            enable_end_zones=False,
            encoder_backend=self._encoder_backend,
        )
        if hasattr(self._vo, "set_cuda_graph_enabled"):
            self._vo.set_cuda_graph_enabled(self._cuda_graph_enabled)

    def finalize(self) -> None:
        if self._vo is not None:
            with self.profile_scope("video_out.finalize"):
                self._vo.stop()

    def input_keys(self):
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "original_images",
                "current_box",
                "frame_ids",
                "player_bottom_points",
                "player_ids",
                "rink_profile",
                "shared",
                "fps",
                "game_id",
                "work_dir",
                "video_frame_cfg",
                "end_zone_img",
                "video_out_prepared",
            }
        return self._input_keys

    def is_output(self) -> bool:
        return self.enabled

    def output_keys(self):
        return set()

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        with self.profile_scope("video_out.ensure_initialized"):
            self._ensure_initialized(context)
        assert self._vo is not None
        prepared = dict(context)
        if not prepared.get("video_out_prepared", False):
            # Preserve backward compatibility for configs that instantiate only
            # the sink plugin: it can still perform prep internally.
            with self.profile_scope("video_out.prepare_fallback"):
                prepared = self._vo.prepare_results(prepared)
        with self.profile_scope("video_out.write"):
            self._vo.write_prepared_results(prepared)
        return {}


from .video_out_prep_plugin import VideoOutPrepPlugin

__all__ = ["VideoOutPlugin", "VideoOutPrepPlugin"]
