from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from mmcv.transforms import Compose

from hmlib.aspen.plugins.base import Plugin
from hmlib.camera.end_zones import EndZones, load_lines_from_config
from hmlib.config import get_nested_value
from hmlib.log import logger
from hmlib.tracking_utils.boundaries import adjust_point_for_clip_box
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor
from hmlib.utils.image import image_height, image_width, make_channels_last
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH


def _parse_requested_output_dim(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "auto", "same"):
            return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class ApplyCameraPlugin(Plugin):
    """
    Aspen trunk that applies camera cropping/rotation and video-out transforms.

    This trunk mirrors the high-level camera pipeline previously implemented
    inside :class:`hmlib.video.video_out.VideoOutput.forward`, but runs it as
    an explicit Aspen stage before video encoding.

    Expects in context:
      - img: batched image tensor (from PlayTrackerPlugin)
      - current_box: TLBR camera boxes per frame
      - frame_ids: tensor[B]
      - play_box: TLBR arena/play box (for sizing when crop_play_box)
      - current_fast_box_list: optional fast camera boxes (for EndZones)
      - dataset_results: optional far-end frames for EndZones
      - shared.game_config: full game config dict
      - shared.original_clip_box: optional clip box
      - shared.game_config: full game config dict

    Params:
      - video_out_pipeline: dict/list pipeline; applied via mmcv.Compose
      - crop_play_box: bool; crop output to play-box region when available
      - crop_output_image: bool; enforce 16:9 crop/output sizing
      - end_zones: bool; enable end-zone camera overlays
    """

    def __init__(
        self,
        enabled: bool = True,
        video_out_pipeline: dict[str, Any] | None = None,
        crop_play_box: bool = False,
        crop_output_image: bool = True,
        end_zones: bool = False,
    ) -> None:
        super().__init__(enabled=enabled)
        self._pipeline_cfg = video_out_pipeline or {}
        self._pipeline: Compose | None = None
        self._color_adjust_tf = None
        self._perspective_rotation_tf = None
        self._video_frame_cfg: dict[str, Any] | None = None
        self._end_zones: EndZones | None = None
        self._game_config: dict[str, Any] | None = None
        self._iter_num: int = 0
        self._initialized: bool = False
        self._crop_play_box = bool(crop_play_box)
        self._crop_output_image = bool(crop_output_image)
        self._use_end_zones = bool(end_zones)
        self._archive_camera = None

    def _ensure_initialized(self, context: dict[str, Any]) -> None:
        if self._initialized:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        self._game_config = shared.get("game_config") or {}

        img = context.get("img")
        if img is None:
            img = context.get("original_images")
        if img is None:
            raise AssertionError("ApplyCameraPlugin requires 'img' or 'original_images'")

        # Decide final output size (mirrors VideoOutPlugin logic)
        H = int(image_height(img))
        W = int(image_width(img))
        ar = 16.0 / 9.0
        play_box = context.get("play_box")

        if self._crop_play_box:
            if self._crop_output_image:
                final_h = H if play_box is None else int(play_box[3] - play_box[1])
                final_w = int(final_h * ar)
                if final_w > MAX_NEVC_VIDEO_WIDTH:
                    final_w = MAX_NEVC_VIDEO_WIDTH
                    final_h = int(final_w / ar)
            else:
                final_h = H if play_box is None else int(play_box[3] - play_box[1])
                final_w = W if play_box is None else int(play_box[2] - play_box[0])
        else:
            if self._crop_output_image:
                final_h = H
                final_w = int(final_h * ar)
                if final_w > MAX_NEVC_VIDEO_WIDTH:
                    final_w = MAX_NEVC_VIDEO_WIDTH
                    final_h = int(final_w / ar)
            else:
                final_h = H
                final_w = W

        final_w = int(final_w)
        final_h = int(final_h)

        requested_w = _parse_requested_output_dim(
            get_nested_value(self._game_config, "video_out.output_width", None)
        )
        requested_h = _parse_requested_output_dim(
            get_nested_value(self._game_config, "video_out.output_height", None)
        )
        if requested_w is not None or requested_h is not None:
            scale_w = (
                float(requested_w) / float(final_w)
                if requested_w is not None and final_w > requested_w
                else 1.0
            )
            scale_h = (
                float(requested_h) / float(final_h)
                if requested_h is not None and final_h > requested_h
                else 1.0
            )
            scale = min(scale_w, scale_h, 1.0)
            if scale < 1.0:
                final_w = max(2, int(final_w * scale))
                final_h = max(2, int(final_h * scale))

        # Width and height should be even numbers
        if final_w % 2 != 0:
            # Round down to avoid creating small upsample/distortion steps
            final_w = final_w - 1 if final_w > 1 else 2
        if final_h % 2 != 0:
            final_h = final_h - 1 if final_h > 1 else 2

        self._video_frame_cfg = {
            "output_frame_width": final_w,
            "output_frame_height": final_h,
            "output_aspect_ratio": float(final_w) / float(final_h),
            "no_crop": not self._crop_output_image,
        }

        if self._pipeline_cfg:
            pipeline = Compose(self._pipeline_cfg)
            for transform in pipeline:
                if transform.__class__.__name__ == "HmImageColorAdjust":
                    transform.channel_order = "bgr"
                    transform.config_ref = self._game_config
                    if not transform._config_paths:
                        transform._config_paths = transform._normalize_paths(
                            [("rink", "camera", "color"), ("rink", "camera")]
                        )
            for mod in pipeline:
                if isinstance(mod, torch.nn.Module):
                    name = str(mod)
                    self.add_module(name, mod)
            self._pipeline = pipeline

        enable_end_zones = self._use_end_zones
        if enable_end_zones and self._game_config:
            lines = load_lines_from_config(self._game_config)
            if lines:
                original_clip_box = shared.get("original_clip_box")
                if original_clip_box is not None:
                    orig_lines = lines
                    lines = {}
                    for key, line in orig_lines.items():
                        line[0] = adjust_point_for_clip_box(line[0], original_clip_box)
                        line[1] = adjust_point_for_clip_box(line[1], original_clip_box)
                        lines[key] = line
                self._end_zones = EndZones(
                    lines=lines,
                    output_width=final_w,
                    output_height=final_h,
                )

        # Ensure any registered pipeline modules/buffers match the image device
        try:
            self.to(img.device)
        except (RuntimeError, TypeError) as error:
            raise RuntimeError(f"Could not move the camera transforms to {img.device}") from error

        self._initialized = True

    def __call__(self, *args, **kwds):
        self._iter_num += 1
        # do_trace = self._iter_num == 4
        # if do_trace:
        #     pass
        # from cuda_stacktrace import CudaStackTracer

        # with CudaStackTracer(functions="cudaStreamSynchronize", enabled=do_trace):
        with contextlib.nullcontext():
            results = super().__call__(*args, **kwds)
        # if do_trace:
        #     pass
        return results

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {"img": context.get("img", None)}
        self._ensure_initialized(context)
        assert self._video_frame_cfg is not None

        img = context.get("img")
        if img is None:
            img = context.get("original_images")
        if img is None:
            return {}

        img = unwrap_tensor(img)

        # frame_ids = context.get("frame_ids")
        current_boxes = context.get("current_box")
        if isinstance(current_boxes, StreamTensorBase):
            current_boxes = unwrap_tensor(current_boxes)

        results: dict[str, Any] = {}
        pano_size_wh = [image_width(img), image_height(img)]
        results["pano_size_wh"] = pano_size_wh
        pipeline_outputs: dict[str, Any] = {}

        if current_boxes is None:
            # Fallback: cover the entire frame
            if img.ndim == 4:
                batch_size = img.size(0)
            else:
                batch_size = 1
            whole_box = torch.zeros((4,), dtype=torch.float32, device=img.device)
            whole_box[2] += image_width(img)
            whole_box[3] += image_height(img)
            current_boxes = whole_box.repeat(batch_size, 1)
        else:
            current_boxes = current_boxes.clone().to(img.device, non_blocking=True)

        # Resolve streamed / numpy images to concrete tensors
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = make_channels_last(img)

        program_4k = None
        shared = context.get("shared", {})
        shared["program_4k_available"] = True
        live_config = shared.get("game_config", self._game_config)
        if get_nested_value(live_config, "video_out.archive_program_4k", False):
            if self._archive_camera is None:
                self._archive_camera = ApplyCameraPlugin(
                    video_out_pipeline=self._pipeline_cfg,
                    crop_play_box=self._crop_play_box,
                    crop_output_image=self._crop_output_image,
                    end_zones=self._use_end_zones,
                )
            # Reuse the full source-resolution image and identical transforms.
            # Only the recording dimensions differ; runtime grading remains shared.
            archive_config = {
                **self._game_config,
                "video_out": {
                    **self._game_config.get("video_out", {}),
                    "archive_program_4k": False,
                    "output_width": 3840,
                    "output_height": 2160,
                },
            }
            archive_context = {
                **context,
                "img": img.clone(),
                "shared": {**context.get("shared", {}), "game_config": archive_config},
            }
            self._archive_camera._ensure_initialized(archive_context)
            self._archive_camera._video_frame_cfg.update(
                output_frame_width=3840,
                output_frame_height=2160,
                output_aspect_ratio=16 / 9,
            )
            program_4k = self._archive_camera.forward(archive_context)["img"]

        # Optional end-zone camera handling
        if self._end_zones is not None:
            ez_data: dict[str, Any] = {
                "img": img,
                "pano_size_wh": pano_size_wh,
            }
            dataset_results = context.get("dataset_results")
            if dataset_results is not None:
                ez_data["dataset_results"] = dataset_results
            fast_boxes = context.get("current_fast_box_list")
            if isinstance(fast_boxes, StreamTensorBase):
                fast_boxes = unwrap_tensor(fast_boxes)
            if fast_boxes is not None:
                ez_data["current_fast_box_list"] = fast_boxes
            ez_data = self._end_zones(ez_data)
            img = ez_data.get("img", img)
            if "end_zone_img" in ez_data:
                results["end_zone_img"] = ez_data["end_zone_img"]

        # Video-out pipeline (perspective, cropping, overlays, etc.)
        video_frame_cfg = self._video_frame_cfg
        if self._pipeline is not None:
            try:
                if self._color_adjust_tf is None:
                    for tf in getattr(self._pipeline, "transforms", []):
                        if tf.__class__.__name__ == "HmImageColorAdjust":
                            self._color_adjust_tf = tf
                            break
                if self._color_adjust_tf is not None and self._game_config is not None:
                    cam = self._game_config.get("rink", {}).get("camera", {})
                    if isinstance(cam, dict):
                        color = cam.get("color", {}) or {}
                        wb = color.get("white_balance", cam.get("white_balance"))
                        wbk = color.get(
                            "white_balance_temp",
                            cam.get("white_balance_k", cam.get("white_balance_temp")),
                        )
                        bright = color.get("brightness", cam.get("color_brightness"))
                        exposure_ev = color.get("exposure_ev", cam.get("exposure_ev"))
                        contr = color.get("contrast", cam.get("color_contrast"))
                        gamma = color.get("gamma", cam.get("color_gamma"))
                        if wbk is not None and wb is None:
                            try:
                                self._color_adjust_tf.white_balance = (
                                    self._color_adjust_tf._gains_from_kelvin(wbk)
                                )
                            except (TypeError, ValueError) as ex:
                                logger.warning("Invalid runtime white_balance_temp %r: %s", wbk, ex)
                        elif wb is not None:
                            try:
                                if isinstance(wb, (list, tuple)) and len(wb) == 3:
                                    self._color_adjust_tf.white_balance = [float(x) for x in wb]
                            except (TypeError, ValueError) as ex:
                                logger.warning("Invalid runtime white_balance %r: %s", wb, ex)
                        if bright is not None:
                            try:
                                self._color_adjust_tf.brightness = float(bright)
                            except (TypeError, ValueError) as ex:
                                logger.warning("Invalid runtime brightness %r: %s", bright, ex)
                        if exposure_ev is not None:
                            try:
                                self._color_adjust_tf.exposure_ev = float(exposure_ev)
                            except (TypeError, ValueError) as ex:
                                logger.warning(
                                    "Invalid runtime exposure_ev %r: %s", exposure_ev, ex
                                )
                        if contr is not None:
                            try:
                                self._color_adjust_tf.contrast = float(contr)
                            except (TypeError, ValueError) as ex:
                                logger.warning("Invalid runtime contrast %r: %s", contr, ex)
                        if gamma is not None:
                            try:
                                self._color_adjust_tf.gamma = float(gamma)
                            except (TypeError, ValueError) as ex:
                                logger.warning("Invalid runtime gamma %r: %s", gamma, ex)
            except (AttributeError, KeyError, TypeError, ValueError) as ex:
                logger.warning("Failed to refresh runtime camera color controls: %s", ex)

            if self._perspective_rotation_tf is None:
                for transform in getattr(self._pipeline, "transforms", []):
                    if transform.__class__.__name__ == "HmPerspectiveRotation":
                        self._perspective_rotation_tf = transform
                        # Geometry changes take effect when calibration/tracking
                        # restarts. Freeze suppression with the loaded panorama,
                        # while retaining live adjustment of saved edge angles.
                        from hmlib.stitching.settings import read_stitching_settings

                        settings = read_stitching_settings(self._game_config)
                        transform.set_camera_space_leveling(
                            settings.mapping_backend == "nona"
                            and any(settings.framing.rotation_degrees[1:])
                        )
                        break
            if self._perspective_rotation_tf is not None and self._game_config is not None:
                angle = get_nested_value(
                    self._game_config,
                    "rink.camera.fixed_edge_rotation_angle",
                    None,
                )
                if angle is not None:
                    try:
                        self._perspective_rotation_tf.set_fixed_edge_rotation_angle(angle)
                    except (TypeError, ValueError) as ex:
                        logger.warning(
                            "Invalid runtime rink.camera.fixed_edge_rotation_angle %r: %s",
                            angle,
                            ex,
                        )

            pipeline_inputs: dict[str, Any] = {
                "img": img,
                "game_config": self._game_config,
                "camera_box": current_boxes,
                "video_frame_cfg": self._video_frame_cfg,
            }
            # Preserve optional extras for overlays
            for key in ("player_bottom_points", "player_ids", "rink_profile", "game_id"):
                if key in context:
                    pipeline_inputs[key] = context[key]
            pipeline_inputs["pano_size_wh"] = pano_size_wh
            if context.get("shared_preview_metadata") is not None:
                from hmlib.ui.preview_overlays import copy_metadata

                pipeline_inputs["shared_preview_metadata"] = copy_metadata(
                    context["shared_preview_metadata"]
                )

            pipeline_outputs = self._pipeline(pipeline_inputs)
            img = pipeline_outputs.pop("img")
            current_boxes = pipeline_outputs.pop("camera_box", current_boxes)
            video_frame_cfg = pipeline_outputs.pop("video_frame_cfg", video_frame_cfg)
            results.update(pipeline_outputs)

        if isinstance(video_frame_cfg, dict):
            video_frame_cfg = dict(video_frame_cfg)
            actual_w = int(image_width(img))
            actual_h = int(image_height(img))
            video_frame_cfg["output_frame_width"] = actual_w
            video_frame_cfg["output_frame_height"] = actual_h
            video_frame_cfg["output_aspect_ratio"] = float(actual_w) / float(actual_h)

        results = {
            "img": wrap_tensor(img),
            "video_frame_cfg": video_frame_cfg,
        }
        #  results["current_box"] = wrap_tensor(current_boxes)
        # if frame_ids is not None:
        #     results["frame_ids"] = frame_ids
        if "end_zone_img" in pipeline_outputs:
            results["end_zone_img"] = pipeline_outputs["end_zone_img"]
        if context.get("rink_map_overlay") is not None:
            results["rink_map_overlay"] = context["rink_map_overlay"]
        if pipeline_outputs.get("shared_preview_metadata") is not None:
            results["shared_preview_metadata"] = pipeline_outputs["shared_preview_metadata"]
        if program_4k is not None:
            results["program_4k_img"] = program_4k
        return results

    def input_keys(self) -> set[str]:
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "original_images",
                "current_box",
                "frame_ids",
                "current_fast_box_list",
                "play_box",
                "shared",
                "dataset_results",
                "rink_profile",
                "rink_map_overlay",
                "shared_preview_metadata",
                "game_id",
            }
        return self._input_keys

    def output_keys(self) -> set[str]:
        return {
            "img",
            "rink_map_overlay",
            "shared_preview_metadata",
            "program_4k_img",
            # "current_box",
            # "frame_ids",
            # "pano_size_wh",
            "video_frame_cfg",
            "end_zone_img",
        }
