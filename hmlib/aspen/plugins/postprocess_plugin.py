import argparse
import os
from typing import Any, Dict, List, Optional

import torch

from hmlib.bbox.box_functions import center, clamp_box, height, make_box_at_center, width
from hmlib.camera.camera import HockeyMON
from hmlib.config import get_nested_value
from hmlib.utils.image import image_height, image_width
from hmlib.utils.path import add_prefix_to_filename
from hmlib.video.video_stream import MAX_VIDEO_WIDTH

from .base import Plugin


class CamPostProcessPlugin(Plugin):
    """
    Computes the camera arena/play box and initializes HockeyMON geometry.

    Expects in context:
      - shared.initial_args: original CLI args dict (required)
      - shared.game_config: full game config dict
      - shared.work_dir: output/results directory
      - shared.original_clip_box: optional clip box
      - data_samples: TrackDataSample (or list)
      - fps: float (optional)
      - rink_profile: rink profile with combined_bbox (for play box seeding)
      - original_images: original input frames tensor

    Produces in context:
      - arena: TLBR tensor describing the play/arena region.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

        # Configuration populated either from Aspen context or from legacy
        # CamTrackPostProcessor-style arguments via _configure().
        self._opt: Optional[argparse.Namespace] = None
        self._args: Optional[argparse.Namespace] = None
        self._camera_name: str = ""
        self._data_type: str = "mot"
        self._postprocess: bool = True
        self._video_out_pipeline: Dict[str, Any] = {}
        self._original_clip_box: Any = None
        self._fps: float = 0.0
        self._save_dir: str = "."
        self._output_video_path: Optional[str] = None
        self._save_frame_dir: Optional[str] = None
        self._device: Optional[torch.device] = None
        self._video_out_device: Optional[torch.device] = None
        self._no_cuda_streams: bool = False
        self._counter: int = 0

        # Core post-processing state (initialized on first frame)
        self._hockey_mon: Optional[HockeyMON] = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._arena_box: Optional[torch.Tensor] = None
        self.final_frame_width: Optional[int] = None
        self.final_frame_height: Optional[int] = None

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Legacy CamTrackPostProcessor configuration helpers
    # ------------------------------------------------------------------
    def _configure(
        self,
        opt: argparse.Namespace,
        args: argparse.Namespace,
        device: torch.device,
        fps: float,
        save_dir: str,
        output_video_path: Optional[str],
        camera_name: str,
        original_clip_box,
        video_out_pipeline: Dict[str, Any],
        save_frame_dir: Optional[str] = None,
        data_type: str = "mot",
        postprocess: bool = True,
        video_out_device: Optional[torch.device] = None,
        no_cuda_streams: bool = False,
    ) -> None:
        """Populate configuration previously owned by CamTrackPostProcessor."""
        self._opt = opt
        self._args = args
        self._camera_name = camera_name
        self._data_type = data_type
        self._postprocess = postprocess
        self._video_out_pipeline = video_out_pipeline
        self._original_clip_box = original_clip_box
        self._fps = fps
        self._save_dir = save_dir
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._device = device
        self._video_out_device = video_out_device or device
        self._no_cuda_streams = no_cuda_streams
        self._counter = 0

        # Reset per-run state
        self._hockey_mon = None
        self._arena_box = None
        self.final_frame_width = None
        self.final_frame_height = None
        self._initialized = True

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        """Lazily configure the plugin from the Aspen shared context."""
        if self._initialized:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        game_cfg = shared.get("game_config") or {}
        init_args = shared.get("initial_args") or {}
        work_dir = shared.get("work_dir") or "."
        original_clip_box = shared.get("original_clip_box")
        device = shared.get("camera_device") or shared.get("device") or torch.device("cuda")
        video_out_device = shared.get("encoder_device") or device

        try:
            args_ns = argparse.Namespace(**init_args)
        except Exception:
            args_ns = argparse.Namespace()
        setattr(args_ns, "game_config", game_cfg)

        if not hasattr(args_ns, "cam_ignore_largest"):
            cam_ignore = get_nested_value(
                game_cfg, "rink.tracking.cam_ignore_largest", default_value=False
            )
            setattr(args_ns, "cam_ignore_largest", bool(cam_ignore))
        if not hasattr(args_ns, "crop_output_image"):
            no_crop = bool(getattr(args_ns, "no_crop", False))
            setattr(args_ns, "crop_output_image", not no_crop)
        if not hasattr(args_ns, "crop_play_box"):
            setattr(args_ns, "crop_play_box", bool(getattr(args_ns, "crop_play_box", False)))
        if not hasattr(args_ns, "plot_individual_player_tracking"):
            pit = bool(getattr(args_ns, "plot_tracking", False))
            setattr(args_ns, "plot_individual_player_tracking", pit)
        if not hasattr(args_ns, "plot_boundaries"):
            setattr(
                args_ns,
                "plot_boundaries",
                bool(getattr(args_ns, "plot_individual_player_tracking", False)),
            )
        if not hasattr(args_ns, "plot_cluster_tracking"):
            setattr(args_ns, "plot_cluster_tracking", False)
        if not hasattr(args_ns, "plot_speed"):
            setattr(args_ns, "plot_speed", False)

        fps = None
        try:
            fps_val = context.get("fps")
            if fps_val is None and isinstance(shared, dict):
                fps_val = shared.get("fps")
            fps = float(fps_val) if fps_val is not None else None
        except Exception:
            fps = None
        if fps is None:
            fps = 30.0

        camera_name = None
        try:
            camera_name = get_nested_value(game_cfg, "camera.name", None)
        except Exception:
            camera_name = None

        output_video_path = getattr(args_ns, "output_video_path", None) or shared.get(
            "output_video_path"
        )
        if not output_video_path:
            output_video_path = os.path.join(work_dir, "tracking_output.mkv")
        output_label = (
            shared.get("output_label") or shared.get("label") or getattr(args_ns, "label", None)
        )
        if output_label:
            try:
                output_video_path = str(
                    add_prefix_to_filename(output_video_path, str(output_label))
                )
            except Exception:
                pass

        video_out_pipeline = getattr(args_ns, "video_out_pipeline", None) or shared.get(
            "video_out_pipeline"
        )
        if video_out_pipeline is None:
            video_out_pipeline = {}

        save_frame_dir = getattr(args_ns, "save_frame_dir", None)
        no_cuda_streams = bool(getattr(args_ns, "no_cuda_streams", False))

        self._configure(
            opt=args_ns,
            args=args_ns,
            device=device,
            fps=fps,
            save_dir=work_dir,
            output_video_path=output_video_path,
            save_frame_dir=save_frame_dir,
            original_clip_box=original_clip_box,
            video_out_pipeline=video_out_pipeline,
            data_type="mot",
            postprocess=True,
            video_out_device=video_out_device,
            no_cuda_streams=no_cuda_streams,
            camera_name=camera_name or "",
        )

    # ------------------------------------------------------------------
    # Core CamTrackPostProcessor behavior
    # ------------------------------------------------------------------
    @property
    def data_type(self) -> str:
        return self._data_type

    def filter_outputs(self, outputs: torch.Tensor, output_results):
        return outputs, output_results

    def _maybe_init(
        self,
        frame_id: int,
        img_width: int,
        img_height: int,
        arena: List[int],
        device: torch.device,
    ) -> None:
        if not self.is_initialized():
            self.on_first_image(
                frame_id=frame_id,
                img_width=img_width,
                img_height=img_height,
                device=device,
                arena=arena,
            )

    def is_initialized(self) -> bool:
        return self._hockey_mon is not None

    @staticmethod
    def calculate_play_box(
        results: Dict[str, Any], context: Dict[str, Any], scale: float = 1.3
    ) -> List[int]:
        play_box = torch.tensor(context["rink_profile"]["combined_bbox"], dtype=torch.int64)
        ww, hh = width(play_box), height(play_box)
        cc = center(play_box)
        play_box = make_box_at_center(cc, ww * scale, hh * scale)
        # Prefer original_images from the results dict when present; otherwise
        # fall back to a top-level context entry. Avoid boolean checks on
        # tensors so we don't hit \"ambiguous\" errors.
        images = results.get("original_images", None)
        if images is None:
            images = context.get("original_images")
        return clamp_box(
            play_box,
            [
                0,
                0,
                image_width(images),
                image_height(images),
            ],
        )

    def process_tracking(
        self,
        results: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._counter += 1
        if not self._postprocess:
            return results
        if self._hockey_mon is None:
            data_samples = results.get("data_samples")
            if data_samples is None or not hasattr(data_samples, "video_data_samples"):
                return results
            video_data_sample = data_samples.video_data_samples[0]
            metainfo = video_data_sample.metainfo
            original_shape = metainfo["ori_shape"]

            arena = self.calculate_play_box(results, context)

            if not isinstance(original_shape, torch.Size):
                original_shape = torch.Size(list(original_shape))
            frame_id = getattr(video_data_sample, "frame_id", None)
            if frame_id is None:
                frame_id = metainfo.get("frame_id", 0)
            assert self._device is not None
            self._maybe_init(
                frame_id=int(frame_id),
                img_height=int(original_shape[0]),
                img_width=int(original_shape[1]),
                arena=arena,
                device=self._device,
            )
        # Expose arena/play box for downstream plugins (e.g., PlayTrackerPlugin, VideoOutPlugin)
        results["arena"] = self.get_arena_box()
        return results

    def on_first_image(
        self,
        frame_id: int,
        img_width: int,
        img_height: int,
        arena: List[int],
        device: torch.device,
    ) -> None:
        assert self._hockey_mon is None

        # Initialize HockeyMON video geometry and cache arena box
        self._hockey_mon = HockeyMON(
            image_width=img_width,
            image_height=img_height,
            fps=self._fps,
            device=device,
            camera_name=self._camera_name,
        )
        self._arena_box = (
            self._hockey_mon.video.bounding_box()
            if not getattr(self._args, "crop_play_box", False)
            else torch.as_tensor(arena, dtype=torch.float, device=device)
        )
        self.secondary_init()

    def secondary_init(self) -> None:
        assert self._hockey_mon is not None
        play_box: torch.Tensor = (
            self._arena_box
            if self._arena_box is not None
            else self._hockey_mon.video.bounding_box()
        )
        play_width, play_height = width(play_box), height(play_box)

        if getattr(self._args, "crop_play_box", False):
            if getattr(self._args, "crop_output_image", True):
                self.final_frame_height = play_height
                self.final_frame_width = play_height * self._final_aspect_ratio
                if self.final_frame_width > MAX_VIDEO_WIDTH:
                    self.final_frame_width = MAX_VIDEO_WIDTH
                    self.final_frame_height = self.final_frame_width / self._final_aspect_ratio

            else:
                self.final_frame_height = play_height
                self.final_frame_width = play_width
        else:
            if getattr(self._args, "crop_output_image", True):
                self.final_frame_height = self._hockey_mon.video.height
                self.final_frame_width = self._hockey_mon.video.height * self._final_aspect_ratio
                if self.final_frame_width > MAX_VIDEO_WIDTH:
                    self.final_frame_width = MAX_VIDEO_WIDTH
                    self.final_frame_height = self.final_frame_width / self._final_aspect_ratio

            else:
                self.final_frame_height = self._hockey_mon.video.height
                self.final_frame_width = self._hockey_mon.video.width

        self.final_frame_width = int(self.final_frame_width + 0.5)  # type: ignore[arg-type]
        self.final_frame_height = int(self.final_frame_height + 0.5)  # type: ignore[arg-type]

    @property
    def output_video_path(self) -> Optional[str]:
        return self._output_video_path

    def start(self) -> None:
        # CamPostProcessPlugin runs synchronously; kept for API compatibility.
        return None

    def stop(self) -> None:
        # Kept for API compatibility; nothing to stop in synchronous mode.
        return None

    def get_arena_box(self) -> torch.Tensor:
        assert self._hockey_mon is not None
        if self._arena_box is not None:
            return self._arena_box
        return self._hockey_mon.video.bounding_box()

    # ------------------------------------------------------------------
    # Aspen Plugin API
    # ------------------------------------------------------------------
    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        self._ensure_initialized(context)
        results: Dict[str, Any] = {
            "data_samples": context.get("data_samples"),
            "original_images": context.get("original_images"),
            "fps": context.get("fps"),
        }
        results = self.process_tracking(results=results, context=context)
        arena = results.get("arena") if isinstance(results, dict) else None
        if arena is None:
            return {}
        return {
            "arena": arena,
            "final_frame_size": (self.final_frame_width, self.final_frame_height),
        }

    def input_keys(self):
        return {"data_samples", "fps", "rink_profile", "shared", "original_images"}

    def output_keys(self):
        return {"arena", "final_frame_size"}
