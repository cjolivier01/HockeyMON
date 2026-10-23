from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from hmlib.builder import HM
from hmlib.config import get_nested_value, normalize_runtime_config
from hmlib.utils.gpu import unwrap_tensor
from hmlib.utils.image import make_channels_last
from hmlib.utils.path import add_prefix_to_filename
from hmlib.video.bitrate import select_source_bitrate_density
from hmlib.video.video_out import VideoOutput

from .base import Plugin


@HM.register_module()
class VideoOutPlugin(Plugin):
    """Sink plugin that writes prepared frames, with legacy prep fallback."""

    disable_in_cuda_graph_pipeline = True

    def __init__(
        self,
        enabled: bool = True,
        output_video_path: str | None = None,
        cache_size: int = 2,
        preview_via_sink: bool = True,
        video_out_device: str | None = None,
        skip_final_save: bool = False,
        save_frame_dir: str | None = None,
        encoder_backend: str | None = None,
        mux_audio_file: str | None = None,
        mux_audio_stream: int = 0,
        mux_audio_offset_seconds: float = 0.0,
        mux_audio_aac_bitrate: str = "192k",
        output_width: str | int | None = None,
        output_height: str | int | None = None,
        show_image: bool | None = None,
        show_scaled: float | None = None,
        show_youtube: bool | None = None,
        youtube_stream_url: str | None = None,
        youtube_stream_key: str | None = None,
        headless_preview_host: str | None = None,
        headless_preview_port: int | None = None,
        always_stream: bool | None = None,
        bit_rate: int | None = None,
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
        self._vo: VideoOutput | None = None
        self._archives: dict[str, VideoOutput] = {}
        self._archive_paths: list[Path] = []
        self._archive_dir: Path | None = None
        self._archive_audio: dict | None = None
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
        self._frames_written = 0
        self._primary_active = False
        self._primary_path: Path | None = None
        self._main_options: dict = {}
        self._archive_options: dict = {}
        self._segment_counts: dict[str, int] = {}
        self._segments: list[dict] = []
        self._path_audio: dict[Path, dict] = {}
        self._recording_capabilities: list[str] = []
        self._segments_manifest: Path | None = None

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if self._vo is not None and hasattr(self._vo, "set_cuda_graph_enabled"):
            self._vo.set_cuda_graph_enabled(enabled)
        return True

    def _resolve_output_path(self, cfg: dict[str, Any], context: dict[str, Any]) -> str:
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
            out_path = str(add_prefix_to_filename(out_path, str(label)))
        return str(out_path)

    def _resolve_fps(self, context: dict[str, Any]) -> float:
        shared = context.get("shared", {})
        fps_val = context.get("fps")
        if fps_val is None and isinstance(shared, dict):
            fps_val = shared.get("fps")
        fps = 30.0 if fps_val is None else float(fps_val)
        if fps <= 0:
            raise ValueError("Recording FPS must be positive")
        return fps

    def _ensure_initialized(self, context: dict[str, Any]) -> None:
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
        source_bitrate_density = None
        if bit_rate is None and isinstance(shared, dict):
            source_bitrate_density = select_source_bitrate_density(
                shared.get("source_video_paths", ())
            )

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

        if mux_audio_file:
            self._archive_audio = {
                "source": mux_audio_file,
                "stream": mux_audio_stream,
                "offset_seconds": mux_audio_offset_seconds,
                "bitrate": mux_audio_aac_bitrate,
            }
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

        self._primary_active = bool(
            cfg.get("video_out", {}).get("archive_program", not self._skip_final_save)
        )
        self._main_options = {
            "output_video_path": out_path,
            "fps": fps,
            "bit_rate": bit_rate,
            "source_bitrate_density": source_bitrate_density,
            "mux_audio_file": mux_audio_file,
            "mux_audio_stream": mux_audio_stream,
            "mux_audio_offset_seconds": mux_audio_offset_seconds,
            "mux_audio_aac_bitrate": mux_audio_aac_bitrate,
            "save_frame_dir": self._save_dir,
            "name": "TRACKING",
            "skip_final_save": not self._primary_active,
            "progress_bar": progress_bar,
            "cache_size": self._cache,
            "device": vo_dev,
            "output_width": self._output_width,
            "output_height": self._output_height,
            "show_image": show_image,
            "show_scaled": show_scaled,
            "show_youtube": show_youtube,
            "youtube_stream_url": youtube_stream_url,
            "youtube_stream_key": youtube_stream_key,
            "headless_preview_host": headless_preview_host,
            "headless_preview_port": headless_preview_port,
            "always_stream": always_stream,
            "profiler": shared.get("profiler", None) if isinstance(shared, dict) else None,
            "enable_end_zones": False,
            "encoder_backend": self._encoder_backend,
        }
        self._vo = VideoOutput(**self._main_options)
        if hasattr(self._vo, "set_cuda_graph_enabled"):
            self._vo.set_cuda_graph_enabled(self._cuda_graph_enabled)
        self._archive_options = {
            "fps": fps,
            "cache_size": self._cache,
            "device": vo_dev,
            "skip_final_save": False,
            "encoder_backend": self._encoder_backend,
            "enable_end_zones": False,
            "profiler": shared.get("profiler"),
        }
        if self._primary_active:
            self._primary_path = Path(out_path)
            self._segment_counts["archive_program"] = 1
            self._register_segment("archive_program", self._primary_path)

    def _register_segment(self, role: str, path: Path) -> None:
        self._segments.append(
            {
                "role": role,
                "path": str(path),
                "start_frame": self._frames_written,
                "state": "recording",
            }
        )
        if self._archive_audio:
            self._path_audio[path] = {
                **self._archive_audio,
                "offset_seconds": self._archive_audio["offset_seconds"]
                - self._frames_written / self._archive_options["fps"],
            }

    def _segment_path(self, role: str, basename: str) -> Path:
        if self._archive_dir is None:
            Path(self._out_path).parent.mkdir(parents=True, exist_ok=True)
            self._archive_dir = Path(
                tempfile.mkdtemp(prefix=".archives-", dir=Path(self._out_path).parent)
            )
        number = self._segment_counts.get(role, 0) + 1
        self._segment_counts[role] = number
        name = Path(basename)
        if number > 1:
            name = name.with_stem(f"{name.stem}-part{number}")
        path = self._archive_dir / name
        self._archive_paths.append(path)
        return path

    def _finish_segment(self, path: Path, writer: VideoOutput) -> None:
        writer.stop()
        if path.is_file() and path in self._path_audio:
            from hmlib.ui.output_audio import ensure_archive_audio

            ensure_archive_audio(path, **self._path_audio[path])
        for segment in self._segments:
            if segment["path"] == str(path):
                segment.update(state="finalized", end_frame=self._frames_written)

    def _publish_recordings(self, *, state: str = "running") -> None:
        session = os.environ.get("HM_UI_SESSION_DIR")
        if session:
            from hmlib.ui.workspace.files import write_json

            write_json(
                Path(session) / "recording-state.json",
                {
                    "state": state,
                    "capabilities": self._recording_capabilities,
                    "active": {
                        "archive_program": self._primary_active,
                        "archive_stitched": "original_images" in self._archives,
                        "archive_program_4k": "program_4k_img" in self._archives,
                    },
                    "segments": self._segments,
                },
            )

    def _sync_recordings(self, context: dict) -> None:
        shared = context.get("shared", {})
        config = shared.get("game_config", {}).get("video_out", {})
        self._recording_capabilities = ["archive_program"]
        if context.get("original_images") is not None:
            self._recording_capabilities.append("archive_stitched")
        if shared.get("program_4k_available") or context.get("program_4k_img") is not None:
            self._recording_capabilities.append("archive_program_4k")
        shared["recording_capabilities"] = self._recording_capabilities
        primary = bool(config.get("archive_program", not self._skip_final_save))
        changed = False
        if primary != self._primary_active:
            if self._primary_active:
                self._finish_segment(self._primary_path, self._vo)
                self._vo = VideoOutput(
                    **{**self._main_options, "skip_final_save": True, "mux_audio_file": None}
                )
                self._vo.set_cuda_graph_enabled(False)
            else:
                path = self._segment_path("archive_program", Path(self._out_path).name)
                # Segments trim audio after encoding so all encoder backends
                # use the same current playback position and zero-based PTS.
                self._vo = VideoOutput(
                    **{
                        **self._main_options,
                        "output_video_path": str(path),
                        "skip_final_save": False,
                        "mux_audio_file": None,
                    }
                )
                self._vo.set_cuda_graph_enabled(False)
                self._primary_path = path
                self._register_segment("archive_program", path)
            self._primary_active = primary
            changed = True
        for flag, source, basename in (
            ("archive_stitched", "original_images", "stitched_output.mkv"),
            ("archive_program_4k", "program_4k_img", "program_4k_output.mkv"),
        ):
            wanted = bool(config.get(flag, False))
            if wanted and flag not in self._recording_capabilities:
                raise ValueError(f"{flag} is unavailable in this pipeline")
            writer = self._archives.get(source)
            if writer is not None and not wanted:
                path = next(
                    Path(item["path"])
                    for item in reversed(self._segments)
                    if item["role"] == flag and item["state"] == "recording"
                )
                self._finish_segment(path, writer)
                del self._archives[source]
                changed = True
            elif wanted and writer is None and context.get(source) is not None:
                if Path(self._out_path).stem == Path(basename).stem:
                    basename = f"{Path(basename).stem}-full{Path(basename).suffix}"
                path = self._segment_path(flag, basename)
                writer = VideoOutput(
                    output_video_path=str(path), name=flag.upper(), **self._archive_options
                )
                writer.set_cuda_graph_enabled(False)
                self._archives[source] = writer
                self._register_segment(flag, path)
                changed = True
        if changed or not self._frames_written:
            self._publish_recordings()

    def output_paths(self) -> tuple[Path, ...]:
        return tuple(Path(item["path"]) for item in self._segments if Path(item["path"]).is_file())

    def supplementary_paths(self) -> tuple[Path, ...]:
        paths = [path for path in self._archive_paths if path.is_file()]
        if self._segments_manifest is not None:
            paths.append(self._segments_manifest)
        return tuple(paths)

    def finalize(self) -> None:
        from hmlib.utils.finalization import finalize_resources

        actions = []
        for source, writer in self._archives.items():
            role = "archive_stitched" if source == "original_images" else "archive_program_4k"
            path = next(
                Path(item["path"])
                for item in reversed(self._segments)
                if item["role"] == role and item["state"] == "recording"
            )
            actions.append((role, lambda p=path, w=writer: self._finish_segment(p, w)))
        if self._vo is not None and self._primary_active:
            actions.append(
                ("program archive", lambda: self._finish_segment(self._primary_path, self._vo))
            )
        elif self._vo is not None:
            actions.append(("program preview", self._vo.stop))
        with self.profile_scope("video_out.finalize"):
            finalize_resources(actions)
        self._archives.clear()
        self._primary_active = False
        if any(
            item["start_frame"] or item.get("end_frame") != self._frames_written
            for item in self._segments
        ):
            from hmlib.ui.workspace.files import write_json

            if self._archive_dir is None:
                self._archive_dir = Path(
                    tempfile.mkdtemp(prefix=".archives-", dir=Path(self._out_path).parent)
                )
            self._segments_manifest = self._archive_dir / "recording_segments.json"
            fps = self._archive_options["fps"]
            write_json(
                self._segments_manifest,
                {
                    "schema": 1,
                    "fps": fps,
                    "segments": [
                        {
                            **{key: value for key, value in item.items() if key != "path"},
                            "filename": Path(item["path"]).name,
                            "timeline_offset_seconds": item["start_frame"] / fps,
                            "duration_seconds": (item["end_frame"] - item["start_frame"]) / fps,
                        }
                        for item in self._segments
                        if Path(item["path"]).is_file()
                    ],
                },
            )
        self._publish_recordings(state="finalized")

    def input_keys(self):
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "original_images",
                "program_4k_img",
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

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        with self.profile_scope("video_out.ensure_initialized"):
            self._ensure_initialized(context)
            self._sync_recordings(context)
        assert self._vo is not None
        prepared = dict(context)
        if not prepared.get("video_out_prepared", False):
            # Preserve backward compatibility for configs that instantiate only
            # the sink plugin: it can still perform prep internally.
            with self.profile_scope("video_out.prepare_fallback"):
                prepared = self._vo.prepare_results(prepared)
        with self.profile_scope("video_out.write"):
            self._vo.write_prepared_results(prepared)
        for source, writer in self._archives.items():
            frames = make_channels_last(unwrap_tensor(context[source]))
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            frames = frames.contiguous()
            height, width = frames.shape[-3:-1]
            archive = dict(context)
            archive.update(
                img=frames,
                video_out_prepared=False,
                video_frame_cfg={
                    **context["video_frame_cfg"],
                    "output_frame_width": width,
                    "output_frame_height": height,
                    "output_aspect_ratio": width / height,
                },
            )
            archive.pop("end_zone_img", None)
            writer.write_prepared_results(writer.prepare_results(archive))
        frames = unwrap_tensor(context["img"])
        self._frames_written += int(frames.shape[0]) if frames.ndim == 4 else 1
        return {}


from .video_out_prep_plugin import VideoOutPrepPlugin

__all__ = ["VideoOutPlugin", "VideoOutPrepPlugin"]
