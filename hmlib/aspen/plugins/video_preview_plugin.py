from __future__ import annotations

import os
from typing import Any

from hmlib.builder import HM
from hmlib.config import get_nested_value, normalize_runtime_config
from hmlib.log import logger
from hmlib.ui.output_routes import OutputRoutes
from hmlib.ui.shower import Shower
from hmlib.utils.gpu import unwrap_tensor

from .base import Plugin


@HM.register_module()
class VideoPreviewPlugin(Plugin):
    """Async, lossy preview sink for local display, headless preview, and RTMP."""

    disable_in_cuda_graph_pipeline = True

    def __init__(
        self,
        enabled: bool = True,
        cache_size: int = 1,
        preview_fps: float | None = None,
        show_image: bool | None = None,
        show_scaled: float | None = None,
        show_youtube: bool | None = None,
        youtube_stream_url: str | None = None,
        youtube_stream_key: str | None = None,
        headless_preview_host: str | None = None,
        headless_preview_port: int | None = None,
        always_stream: bool | None = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self._cache_size = max(1, int(cache_size or 1))
        self._preview_fps = float(preview_fps) if preview_fps not in (None, 0) else None
        self._show_image = bool(show_image) if show_image is not None else None
        self._show_scaled = show_scaled
        self._show_youtube = bool(show_youtube) if show_youtube is not None else None
        self._youtube_stream_url = youtube_stream_url
        self._youtube_stream_key = youtube_stream_key
        self._headless_preview_host = headless_preview_host
        self._headless_preview_port = headless_preview_port
        self._always_stream = bool(always_stream) if always_stream is not None else None
        self._shower: Shower | None = None
        self._routes: OutputRoutes | None = None
        self._local_display_suppressed_for_hm_ui = False
        self._progress_bar_callback_installed = False

    def _resolve_preview_enabled(self, cfg: dict[str, Any]) -> tuple[bool, bool]:
        show_image = bool(
            self._show_image
            if self._show_image is not None
            else get_nested_value(
                cfg,
                "video_out.show_image",
                default_value=get_nested_value(cfg, "aspen.video_out.show_image", False),
            )
        )
        show_youtube = bool(
            self._show_youtube
            if self._show_youtube is not None
            else get_nested_value(
                cfg,
                "video_out.show_youtube",
                default_value=get_nested_value(cfg, "aspen.video_out.show_youtube", False),
            )
        )
        return show_image, show_youtube

    def _ensure_initialized(self, context: dict[str, Any]) -> None:
        shared = context.get("shared", {})
        shared_cfg = shared if isinstance(shared, dict) else {}
        cfg = shared_cfg.get("game_config") if shared_cfg else {}
        cfg = cfg or {}
        normalize_runtime_config(cfg)
        if self._routes is None:
            from hmlib.ui.output_cadence import recording_fps

            route_config = dict(cfg.get("video_out", {}))
            route_config["output_fps"] = recording_fps(
                cfg, float(context.get("fps") or shared_cfg.get("fps") or 30)
            )
            for key, value in (
                ("show_youtube", self._show_youtube),
                ("youtube_stream_url", self._youtube_stream_url),
                ("youtube_stream_key", self._youtube_stream_key),
            ):
                if value is not None:
                    route_config[key] = value
            self._routes = OutputRoutes(
                route_config,
                fps=float(context.get("fps") or shared_cfg.get("fps") or 30),
                profiler=self._profiler,
                audio={
                    "file": shared_cfg.get("mux_audio_file"),
                    "stream": shared_cfg.get("mux_audio_stream") or 0,
                    "offset_seconds": shared_cfg.get("mux_audio_offset_seconds") or 0,
                },
                session_dir=os.environ.get("HM_UI_SESSION_DIR"),
            )
        hm_ui_preview_active = bool(shared_cfg.get("hm_ui_preview_active"))
        hm_ui_owns_local_preview = hm_ui_preview_active or bool(shared_cfg.get("camera_ui"))

        if self._shower is not None:
            if hm_ui_owns_local_preview and not self._local_display_suppressed_for_hm_ui:
                self._shower.close()
                self._shower = None
                self._progress_bar_callback_installed = False
            elif self._local_display_suppressed_for_hm_ui and not hm_ui_owns_local_preview:
                self._shower.close()
                self._shower = None
                self._local_display_suppressed_for_hm_ui = False
                self._progress_bar_callback_installed = False
            else:
                return

        show_image, _ = self._resolve_preview_enabled(cfg)
        # Network routes consume every output frame before the lossy display queue.
        show_youtube = False
        if not show_image and not show_youtube:
            return
        effective_show_image = show_image and not hm_ui_owns_local_preview
        self._local_display_suppressed_for_hm_ui = show_image and hm_ui_owns_local_preview
        if not effective_show_image and not show_youtube:
            return

        fps_val = context.get("fps")
        if fps_val is None:
            fps_val = shared_cfg.get("fps")
        fps = self._preview_fps
        if fps is None:
            cfg_preview_fps = get_nested_value(
                cfg,
                "video_out.preview_fps",
                default_value=get_nested_value(cfg, "aspen.video_out.preview_fps", None),
            )
            if cfg_preview_fps is not None:
                fps = float(cfg_preview_fps)
        if fps is None and fps_val is not None:
            fps = float(fps_val)
        if fps is not None and fps <= 0:
            raise ValueError("Preview FPS must be positive")

        label = "Preview"
        if shared_cfg.get("game_id"):
            label = f"{shared_cfg['game_id']} preview"

        headless_preview_host = (
            self._headless_preview_host
            if self._headless_preview_host is not None
            else get_nested_value(
                cfg,
                "video_out.headless_preview_host",
                default_value=get_nested_value(
                    cfg, "aspen.video_out.headless_preview_host", "0.0.0.0"
                ),
            )
        )
        headless_preview_port = (
            self._headless_preview_port
            if self._headless_preview_port is not None
            else get_nested_value(
                cfg,
                "video_out.headless_preview_port",
                default_value=get_nested_value(cfg, "aspen.video_out.headless_preview_port", 0),
            )
        )

        self._shower = Shower(
            label=label,
            show_scaled=(
                self._show_scaled
                if self._show_scaled is not None
                else get_nested_value(
                    cfg,
                    "video_out.show_scaled",
                    default_value=get_nested_value(cfg, "aspen.video_out.show_scaled", None),
                )
            ),
            max_size=self._cache_size,
            fps=fps,
            profiler=self._profiler,
            skip_frame_when_full=True,
            drop_oldest_when_full=True,
            enable_local_display=effective_show_image,
            show_youtube=show_youtube,
            youtube_stream_url=(
                self._youtube_stream_url
                if self._youtube_stream_url is not None
                else get_nested_value(
                    cfg,
                    "video_out.youtube_stream_url",
                    default_value=get_nested_value(cfg, "aspen.video_out.youtube_stream_url", None),
                )
            ),
            youtube_stream_key=(
                self._youtube_stream_key
                if self._youtube_stream_key is not None
                else get_nested_value(
                    cfg,
                    "video_out.youtube_stream_key",
                    default_value=get_nested_value(cfg, "aspen.video_out.youtube_stream_key", None),
                )
            ),
            headless_preview_host=str(headless_preview_host or "0.0.0.0"),
            headless_preview_port=int(headless_preview_port or 0),
            always_stream=bool(
                self._always_stream
                if self._always_stream is not None
                else get_nested_value(
                    cfg,
                    "video_out.always_stream",
                    default_value=get_nested_value(cfg, "aspen.video_out.always_stream", False),
                )
            ),
        )

        progress_bar = shared_cfg.get("progress_bar")
        if progress_bar is not None and not self._progress_bar_callback_installed:
            progress_bar.add_table_callback(self._shower.update_progress_table)
            self._progress_bar_callback_installed = True

    def input_keys(self):
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "shared",
                "fps",
                "original_images",
                "rink_map_overlay",
                "shared_preview_metadata",
            }
        return self._input_keys

    def is_output(self) -> bool:
        return self.enabled

    def output_keys(self):
        return set()

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        with self.profile_scope("video_preview.forward"):
            with self.profile_scope("video_preview.ensure_initialized"):
                self._ensure_initialized(context)
            img = context.get("img")
            if img is None:
                return {}
            if self._routes is not None:
                self._routes.write(context)
            preview_img = img
            overlay = context.get("rink_map_overlay")
            if overlay:
                from hmlib.camera.rink_framing import draw_rink_map

                preview_img = draw_rink_map(unwrap_tensor(img).clone(), overlay)
            shared = context.get("shared", {})
            hm_ui_process = shared.get("hm_ui_process") if isinstance(shared, dict) else None
            publish_preview = getattr(hm_ui_process, "publish_preview", None)
            if callable(publish_preview):
                try:
                    publish_preview(
                        preview_img, name="Final", metadata=context.get("shared_preview_metadata")
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as ex:
                    logger.warning("Failed to publish final hm-ui preview frame: %s", ex)
            if self._shower is None:
                return {}
            with self.profile_scope("video_preview.enqueue"):
                if getattr(preview_img, "ndim", None) == 4:
                    for frame in preview_img:
                        self._shower.show(frame, clone=True)
                else:
                    self._shower.show(preview_img, clone=True)
        return {}

    def finalize(self) -> None:
        try:
            if self._routes is not None:
                self._routes.close()
        finally:
            self._routes = None
            self._close_preview()

    def _close_preview(self) -> None:
        if self._shower is not None:
            with self.profile_scope("video_preview.finalize"):
                self._shower.close()
            self._shower = None


__all__ = ["VideoPreviewPlugin"]
