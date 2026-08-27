from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch

from hmlib.bbox.box_functions import center, height, make_box_at_center, width
from hmlib.builder import HM
from hmlib.camera.camera import HockeyMON
from hmlib.camera.camera_controller_modes import normalize_play_tracker_camera_controller
from hmlib.camera.play_tracker import PlayTracker as _PlayTracker
from hmlib.config import get_nested_value
from hmlib.utils.image import image_height, image_width

from .base import Plugin


@HM.register_module()
class PlayTrackerPlugin(Plugin):
    """
    Plugin wrapping the PlayTracker to compute per-frame camera boxes and
    attach visualization images.

    Expects context keys (minimal_context ready):
      - data_samples: TrackDataSample (or list)
      - original_images: stitched panorama frames
      - shared.game_config: full game config dict
      - shared.original_clip_box: optional clip box
      - shared.device: torch.device
      - shared.game_config: full game config dict
      - arena: optional TLBR arena/play region tensor (from CamTrackPostProcessor)

    Produces:
      - img: image tensor batch after overlays (StreamCheckpoint)
      - current_box: TLBR camera boxes per frame (tensor[B,4])
      - frame_ids: tensor[B]
      - player_bottom_points, player_ids, rink_profile (when available)
      - play_box: current arena/play region tensor (TLBR)
    """

    def __init__(
        self,
        enabled: bool = True,
        cam_ignore_largest: Optional[bool] = None,
        no_wide_start: bool = False,
        track_ids: Optional[str] = None,
        debug_play_tracker: bool = False,
        plot_moving_boxes: bool = False,
        plot_individual_player_tracking: bool = False,
        plot_tracking_circles: bool = True,
        plot_boundaries: bool = False,
        plot_all_detections: Optional[float] = None,
        plot_trajectories: bool = False,
        plot_speed: bool = False,
        plot_jersey_numbers: bool = False,
        plot_actions: bool = False,
        camera_controller: Optional[str] = None,
        camera_model: Optional[str] = None,
        camera_window: Optional[int] = None,
        force_stitching: bool = False,
        cluster_centroids_path: Optional[str] = None,
        save_cluster_centroids: bool = False,
    ) -> None:
        super().__init__(enabled=enabled)
        self._hockey_mon: Optional[HockeyMON] = None
        self._play_tracker: Optional[_PlayTracker] = None
        self._device: Optional[torch.device] = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._clip_box: Optional[torch.Tensor] = None
        self._cam_ignore_largest = cam_ignore_largest
        self._no_wide_start = bool(no_wide_start)
        self._track_ids = track_ids
        self._debug_play_tracker = bool(debug_play_tracker)
        self._plot_moving_boxes = bool(plot_moving_boxes)
        self._plot_individual_player_tracking = bool(plot_individual_player_tracking)
        self._plot_tracking_circles = bool(plot_tracking_circles)
        self._plot_boundaries = bool(plot_boundaries)
        self._plot_all_detections = plot_all_detections
        self._plot_trajectories = bool(plot_trajectories)
        self._plot_speed = bool(plot_speed)
        self._plot_jersey_numbers = bool(plot_jersey_numbers)
        self._plot_actions = bool(plot_actions)
        self._camera_controller = camera_controller
        self._camera_model = camera_model
        self._camera_window = camera_window
        self._force_stitching = bool(force_stitching)
        self._cluster_centroids_path = cluster_centroids_path
        self._save_cluster_centroids = bool(save_cluster_centroids)

    def _update_hm_ui_preview_status(self, context: Dict[str, Any]) -> None:
        shared = context.get("shared") if isinstance(context, dict) else None
        if not isinstance(shared, dict) or self._play_tracker is None:
            return
        status_fn = getattr(self._play_tracker, "hm_ui_preview_active", None)
        shared["hm_ui_preview_active"] = bool(status_fn()) if callable(status_fn) else False
        process_fn = getattr(self._play_tracker, "hm_ui_process", None)
        shared["hm_ui_process"] = process_fn() if callable(process_fn) else None

    # region helpers
    @staticmethod
    def _calc_seed_play_box(results: Dict[str, Any]) -> torch.Tensor:
        # Use rink_profile combined bbox if present; scale a bit and clamp
        track_container = results["data_samples"]
        vds = track_container.video_data_samples[0]
        prof = results.get("rink_profile")
        if prof is None:
            # Backwards-compat: older pipelines stored rink_profile on metainfo.
            prof = getattr(vds, "metainfo", {}).get("rink_profile", None)
        image = results["original_images"]
        if prof and "combined_bbox" in prof:
            play_box = torch.tensor(prof["combined_bbox"], dtype=torch.int64, device=image.device)
            ww, hh = width(play_box), height(play_box)
            cc = center(play_box)
            play_box = make_box_at_center(cc, ww * 1.3, hh * 1.3)
        else:
            # fallback to whole frame
            play_box = torch.tensor(
                [
                    0,
                    0,
                    image_width(image),
                    image_height(image),
                ],
                dtype=torch.int64,
                device=image.device,
            )
        # Clamp to image bounds
        zero = torch.zeros((), dtype=torch.int64, device=image.device)
        play_box[0] = torch.max(zero, play_box[0].long())
        play_box[1] = torch.max(zero, play_box[1].long())
        play_box[2] = torch.min(
            torch.tensor(image_width(image), dtype=torch.int64, device=image.device),
            play_box[2].long(),
        )
        play_box[3] = torch.min(
            torch.tensor(image_height(image), dtype=torch.int64, device=image.device),
            play_box[3].long(),
        )
        return play_box

    def _ensure_initialized(self, context: Dict[str, Any], results: Dict[str, Any]) -> None:
        if self._play_tracker is not None:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        game_cfg = shared.get("game_config") or {}
        self._clip_box = shared.get("original_clip_box")
        # Prefer provided device; fallback to image's device
        dev = context.get("device")
        if dev is None and isinstance(results.get("original_images"), torch.Tensor):
            dev = results["original_images"].device
        self._device = (
            dev if isinstance(dev, torch.device) else torch.device(str(dev) if dev else "cuda")
        )
        # Determine video geometry and fps
        ori = results["original_images"]
        H = int(ori.shape[-2])
        W = int(ori.shape[-1])
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

        cam_name = None
        try:
            cam_name = game_cfg.get("camera", {}).get("name")
        except Exception:
            pass

        self._hockey_mon = HockeyMON(
            image_width=W,
            image_height=H,
            fps=fps,
            device=self._device,
            camera_name=cam_name,
        )
        # Prefer arena/play box provided by upstream CamTrackPostProcessor when available.
        arena = context.get("arena")
        if isinstance(arena, torch.Tensor):
            seed_box = arena.to(device=self._device, dtype=torch.int64)
        elif arena is not None:
            seed_box = torch.as_tensor(arena, dtype=torch.int64, device=self._device)
        else:
            seed_box = self._calc_seed_play_box(results)
        # Camera + plotting configuration
        cam_ignore = (
            self._cam_ignore_largest
            if self._cam_ignore_largest is not None
            else bool(get_nested_value(game_cfg, "rink.tracking.cam_ignore_largest", False))
        )
        controller = self._camera_controller or get_nested_value(
            game_cfg, "rink.camera.controller", "rule"
        )
        cam_model = self._camera_model or get_nested_value(
            game_cfg, "rink.camera.camera_model", None
        )
        controller = normalize_play_tracker_camera_controller(controller, cam_model)
        cam_window = (
            self._camera_window
            if self._camera_window is not None
            else int(get_nested_value(game_cfg, "rink.camera.camera_window", 8))
        )

        # Optional track id whitelist parsed here so tests/YAML can pass strings
        track_ids: Optional[Set[int]] = None
        if self._track_ids:
            if isinstance(self._track_ids, str):
                track_ids = {int(i) for i in self._track_ids.split(",") if i}
            else:
                try:
                    track_ids = {int(i) for i in self._track_ids}  # type: ignore[arg-type]
                except Exception:
                    track_ids = None

        self._play_tracker = _PlayTracker(
            hockey_mon=self._hockey_mon,
            play_box=seed_box,
            device=self._device,
            original_clip_box=self._clip_box,
            progress_bar=None,
            game_config=game_cfg,
            game_id=shared.get("game_id"),
            cam_ignore_largest=cam_ignore,
            no_wide_start=self._no_wide_start,
            track_ids=track_ids,
            debug_play_tracker=self._debug_play_tracker,
            plot_moving_boxes=self._plot_moving_boxes,
            plot_individual_player_tracking=self._plot_individual_player_tracking,
            plot_tracking_circles=self._plot_tracking_circles,
            plot_boundaries=self._plot_boundaries,
            plot_all_detections=self._plot_all_detections,
            plot_trajectories=self._plot_trajectories,
            plot_speed=self._plot_speed,
            plot_jersey_numbers=self._plot_jersey_numbers,
            plot_actions=self._plot_actions,
            camera_ui=int(context.get("shared", {}).get("camera_ui", 0)),
            camera_controller=controller,
            camera_model=cam_model,
            camera_window=int(cam_window),
            force_stitching=self._force_stitching,
            stitch_rotation_controller=shared.get("stitch_rotation_controller"),
            cluster_centroids=shared.get("cluster_centroids"),
            cluster_centroids_path=self._cluster_centroids_path
            or shared.get("cluster_centroids_path"),
            save_cluster_centroids=self._save_cluster_centroids
            or bool(shared.get("save_cluster_centroids", False)),
        )
        self._play_tracker.eval()
        self._update_hm_ui_preview_status(context)

    def finalize(self) -> None:
        if self._play_tracker is None:
            return
        closer = getattr(self._play_tracker, "close_ui", None)
        if callable(closer):
            closer()

    # endregion

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        track_data_sample = track_samples[0] if isinstance(track_samples, list) else track_samples

        original_images = context.get("original_images")
        if original_images is None:
            return {}
        # Build results dictionary expected by PlayTracker
        results: Dict[str, Any] = {
            "data_samples": track_data_sample,
            "original_images": original_images,
        }
        # Optional rink_profile for play-box seeding and boundary overlays
        rp = context.get("rink_profile")
        if rp is not None:
            results["rink_profile"] = rp
        # Optional per-frame annotations propagated if present
        for k in ("jersey_results", "action_results"):
            v = context.get(k)
            if v is not None:
                results[k] = v
        # Optional external camera controller output (e.g., GPT/transformer trunk).
        cam_boxes = context.get("camera_boxes")
        if cam_boxes is not None:
            results["camera_boxes"] = cam_boxes
        cam_fast_boxes = context.get("camera_fast_boxes")
        if cam_fast_boxes is not None:
            results["camera_fast_boxes"] = cam_fast_boxes

        self._ensure_initialized(context, results)
        assert self._play_tracker is not None
        with torch.no_grad():
            out = self._play_tracker.forward(results=results)
        self._update_hm_ui_preview_status(context)
        # Surface current play_box for downstream video out sizing
        try:
            out["play_box"] = self._play_tracker.play_box
        except Exception:
            pass
        results = {k: v for k, v in out.items() if k in self.output_keys()}
        return results

    def input_keys(self) -> set[str]:
        return {
            "data_samples",
            "original_images",
            "jersey_results",
            "action_results",
            "camera_boxes",
            "camera_fast_boxes",
            "fps",
            "device",
            "shared",
            "arena",
            "tracker_stream_token",
        }

    def output_keys(self) -> set[str]:
        if not hasattr(self, "_output_keys"):
            self._output_keys: set[str] = {
                "img",
                "current_box",
                "current_fast_box_list",
                "frame_ids",
                "player_bottom_points",
                "player_ids",
                "play_box",
                "rink_profile",
            }
        return self._output_keys
