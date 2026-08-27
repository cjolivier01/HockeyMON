"""Aspen plugin that stitches multi-camera inputs into a panorama frame."""

from __future__ import annotations

import contextlib
import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import Compose

from hmlib.config import get_game_config, get_nested_value
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.log import logger
from hmlib.stitching.blender2 import create_stitcher
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor
from hmlib.utils.hockeymon_compat import (
    CudaStitchPanoF32,
    CudaStitchPanoNF32,
    CudaStitchPanoNU8,
    CudaStitchPanoU8,
)
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
    make_visible_image,
    resize_image,
)

from .base import Plugin


def _to_tensor(tensor: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> torch.Tensor:
    tensor = unwrap_tensor(tensor)
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    raise TypeError(f"Unexpected tensor type: {type(tensor)}")


def _parse_dtype(dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in ("float", "float32", "fp32"):
            return torch.float32
        if key in ("float16", "fp16", "half"):
            return torch.float16
        if key in ("uint8", "u8"):
            return torch.uint8
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


class StitchingPlugin(Plugin):
    """Aspen plugin that stitches multi-camera frames into a panorama.

    Expects in context:
      - stitch_inputs: list/tuple of >=2 dicts (views) from MultiDataLoaderWrapper, or a dict
        with left/right (and optionally a list/tuple under `views`)
      - stitch_data_pipeline: optional mmcv.Compose pipeline to build detection inputs

    Produces in context:
      - inputs: model input tensor produced by stitch_data_pipeline (StreamTensorBase/torch.Tensor)
      - data_samples: TrackDataSample (or list) produced by stitch_data_pipeline
      - original_images: stitched frame tensor (channels-first)
      - img: stitched frame (channels-last) for video output plugins
    """

    def __init__(
        self,
        enabled: bool = True,
        pto_project_file: Optional[str] = None,
        dir_name: Optional[str] = None,
        blend_mode: str = "laplacian",
        python_blender: bool = False,
        use_cuda_pano_n: bool = False,
        minimize_blend: bool = True,
        dtype: Optional[Union[str, torch.dtype]] = None,
        max_blend_levels: Optional[int] = None,
        no_cuda_streams: bool = False,
        post_stitch_rotate_degrees: Optional[float] = None,
        left_color_pipeline: Optional[List[Dict[str, Any]]] = None,
        right_color_pipeline: Optional[List[Dict[str, Any]]] = None,
        capture_rgb_stats: bool = False,
        max_output_width: Optional[int] = None,
        cache_rotation_grid: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._pto_project_file = pto_project_file
        self._dir_name = Path(dir_name) if dir_name else None
        if self._pto_project_file and not self._dir_name:
            try:
                self._dir_name = Path(self._pto_project_file).parent
            except Exception:
                self._dir_name = None
        self._blend_mode = str(blend_mode)
        self._python_blender = bool(python_blender)
        self._use_cuda_pano_n = bool(use_cuda_pano_n)
        self._minimize_blend = bool(minimize_blend)
        self._dtype = _parse_dtype(dtype)
        self._max_blend_levels = max_blend_levels
        self._no_cuda_streams = bool(no_cuda_streams)
        self._post_stitch_rotate_degrees = post_stitch_rotate_degrees
        self._left_color_pipeline_cfg = left_color_pipeline
        self._right_color_pipeline_cfg = right_color_pipeline
        self._capture_rgb_stats = bool(capture_rgb_stats)
        self._max_output_width = int(max_output_width) if max_output_width else None
        self._cache_rotation_grid = bool(cache_rotation_grid)

        self._left_color_pipeline: Optional[Compose] = None
        self._right_color_pipeline: Optional[Compose] = None
        self._config_ref: Optional[Dict[str, Any]] = None
        self._initialized: bool = False
        self._stitcher = None
        self._rotate_cache: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = {}
        self._rotate_grid_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}
        self._width_t: Optional[torch.Tensor] = None
        self._height_t: Optional[torch.Tensor] = None
        self._iter_num: int = 0

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._initialized:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        cfg = shared.get("game_config")
        if cfg is None:
            game_id = shared.get("game_id")
            if game_id:
                try:
                    cfg = get_game_config(game_id=game_id)
                except Exception:
                    cfg = None
        if isinstance(cfg, dict):
            self._config_ref = cfg
        self._build_color_pipelines()
        self._initialized = True

    def _build_color_pipelines(self) -> None:
        self._left_color_pipeline = None
        self._right_color_pipeline = None

        def _make_pipeline(spec: Optional[List[Dict[str, Any]]]) -> Optional[Compose]:
            if not spec:
                return None
            try:
                pipeline = Compose(copy.deepcopy(spec))
                if self._config_ref is not None:
                    for tf in getattr(pipeline, "transforms", []):
                        if tf.__class__.__name__ == "HmImageColorAdjust":
                            setattr(tf, "config_ref", self._config_ref)
                return pipeline
            except Exception:
                logger.debug("StitchingPlugin failed to build color pipeline", exc_info=True)
                return None

        self._left_color_pipeline = _make_pipeline(self._left_color_pipeline_cfg)
        self._right_color_pipeline = _make_pipeline(self._right_color_pipeline_cfg)

    def _resolve_dir_name(self, context: Dict[str, Any]) -> Path:
        if self._dir_name is not None:
            return self._dir_name
        if self._pto_project_file:
            try:
                self._dir_name = Path(self._pto_project_file).parent
                return self._dir_name
            except Exception:
                pass
        shared = context.get("shared", {})
        game_dir = shared.get("game_dir")
        if game_dir:
            self._dir_name = Path(game_dir)
            return self._dir_name
        raise RuntimeError("StitchingPlugin needs dir_name or pto_project_file")

    def _ensure_rgba(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = make_channels_first(tensor)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        b, c, h, w = tensor.shape
        if c == 4:
            return tensor[0] if squeezed else tensor
        if c == 3:
            alpha = torch.empty((b, 1, h, w), dtype=tensor.dtype, device=tensor.device)
            alpha.fill_(255)
            out = torch.cat([tensor, alpha], dim=1)
            return out[0] if squeezed else out
        raise ValueError(f"Expected 3 or 4 channels, got {c}")

    def _ensure_rgba_channels_last(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = unwrap_tensor(tensor)
        if tensor.ndim == 3:
            squeezed = True
            if tensor.shape[-1] in (3, 4):
                tensor = tensor.unsqueeze(0)
                channels_last = True
            else:
                tensor = tensor.unsqueeze(0)
                channels_last = False
        elif tensor.ndim == 4:
            squeezed = False
            if tensor.shape[-1] in (3, 4):
                channels_last = True
            else:
                channels_last = False
        else:
            raise ValueError(f"Expected tensor with 3 or 4 dims, got {tensor.shape}")

        if channels_last:
            b, h, w, c = tensor.shape
            if c == 4:
                return tensor[0] if squeezed else tensor
            if c != 3:
                raise ValueError(f"Expected 3 or 4 channels, got {c}")
            out = torch.empty((b, h, w, 4), dtype=tensor.dtype, device=tensor.device)
            out[..., :3] = tensor
            out[..., 3].fill_(255)
            return out[0] if squeezed else out

        if tensor.shape[1] not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {tensor.shape}")
        b, c, h, w = tensor.shape
        if c == 4:
            out = tensor.permute(0, 2, 3, 1)
            return out[0] if squeezed else out
        rgb = tensor.permute(0, 2, 3, 1)
        out = torch.empty((b, h, w, 4), dtype=tensor.dtype, device=tensor.device)
        out[..., :3] = rgb
        out[..., 3].fill_(255)
        return out[0] if squeezed else out

    def _create_stitcher(
        self,
        context: Dict[str, Any],
        imgs: List[torch.Tensor],
        device: torch.device,
    ) -> None:
        if self._stitcher is not None:
            return
        if device.type == "cpu":
            raise RuntimeError("StitchingPlugin requires a CUDA device")
        if len(imgs) < 2:
            raise RuntimeError("StitchingPlugin needs at least 2 input views")
        dir_name = self._resolve_dir_name(context)
        if self._blend_mode == "laplacian":
            levels_arg = (
                int(self._max_blend_levels)
                if self._max_blend_levels is not None and self._max_blend_levels > 0
                else 11
            )
        else:
            levels_arg = 0

        batch_size = int(imgs[0].shape[0])
        for idx, img in enumerate(imgs):
            if img.device != device:
                raise RuntimeError("All stitch inputs must be on the same device")
            if img.ndim != imgs[0].ndim:
                raise RuntimeError("All stitch inputs must have the same tensor rank")
            if img.ndim != 4:
                raise RuntimeError(
                    f"Expected stitch input tensors with 4 dimensions (B, C/H, H/W, W/C); got {img.shape}"
                )
            if int(img.shape[0]) != batch_size:
                raise RuntimeError(
                    "All stitch inputs must have the same batch size; "
                    f"got input[0].shape[0]={batch_size} and input[{idx}].shape[0]={int(img.shape[0])}"
                )

        input_image_sizes_wh = [(image_width(img), image_height(img)) for img in imgs]
        left_size_wh = input_image_sizes_wh[0]
        right_size_wh = input_image_sizes_wh[1]
        dtype = self._dtype if self._python_blender else torch.uint8
        self._stitcher = create_stitcher(
            dir_name=str(dir_name),
            batch_size=batch_size,
            left_image_size_wh=left_size_wh,
            right_image_size_wh=right_size_wh,
            input_image_sizes_wh=input_image_sizes_wh,
            device=device,
            dtype=dtype,
            python_blender=self._python_blender,
            use_cuda_pano=not self._python_blender,
            use_cuda_pano_n=self._use_cuda_pano_n,
            minimize_blend=self._minimize_blend,
            max_output_width=self._max_output_width,
            blend_mode=self._blend_mode,
            levels=levels_arg,
        )

    def _resolve_rotation_degrees(self, context: Dict[str, Any]) -> Optional[float]:
        shared = context.get("shared", {})
        cfg = shared.get("game_config") or {}
        try:
            val = get_nested_value(cfg, "stitching.post_stitch_rotate_degrees", None)
            if val is not None:
                return float(val)
        except Exception:
            pass
        return self._post_stitch_rotate_degrees

    def _prepare_frame_for_video(
        self, image: torch.Tensor, image_roi: Optional[List[int]] = None
    ) -> torch.Tensor:
        if not image_roi:
            if image.shape[-1] == 4:
                image = make_channels_last(image)[:, :, :, :3]
        else:
            image_roi = _fix_clip_box(image_roi, [image_height(image), image_width(image)])
            if image.ndim == 4:
                image = make_channels_last(image)[
                    :, image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
            else:
                image = make_channels_last(image)[
                    image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
        return image

    def _rotate_tensor_keep_size(
        self,
        tensor: torch.Tensor,
        degrees: float,
    ) -> torch.Tensor:
        if tensor is None or degrees is None or abs(degrees) < 1e-6:
            return tensor
        assert tensor.ndim == 4
        was_channels_last = tensor.shape[-1] in (1, 3, 4)
        orig_dtype = tensor.dtype
        device = tensor.device
        # Keep rotation math in float32 even on low-memory paths so Aspen and
        # legacy stitch paths do not diverge from half-precision resampling.
        work_dtype = torch.float32

        x = tensor.permute(0, 3, 1, 2) if was_channels_last else tensor
        b, c, h, w = x.shape
        x_work = x.to(dtype=work_dtype, non_blocking=True)

        grid_cache_key = (int(b), int(c), int(h), int(w), float(degrees), device, work_dtype)
        grid = self._rotate_grid_cache.get(grid_cache_key) if self._cache_rotation_grid else None
        cache_key = (int(h), int(w), device, work_dtype)
        cache = self._rotate_cache.get(cache_key)
        if cache is None:
            center = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], device=device, dtype=work_dtype)
            s = torch.tensor(
                [
                    [(w - 1) / 2.0, 0.0, (w - 1) / 2.0],
                    [0.0, (h - 1) / 2.0, (h - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=work_dtype,
            )
            s_inv = torch.tensor(
                [
                    [2.0 / (w - 1), 0.0, -1.0],
                    [0.0, 2.0 / (h - 1), -1.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=work_dtype,
            )
            s_001 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=work_dtype)
            cache = {"center": center, "s": s, "s_inv": s_inv, "s_001": s_001}
            self._rotate_cache[cache_key] = cache

        if grid is None:
            angle = torch.zeros((), device=device, dtype=work_dtype) + (-degrees * math.pi / 180.0)
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            cx, cy = cache["center"][0], cache["center"][1]
            tx = (1.0 - cos_a) * cx - sin_a * cy
            ty = sin_a * cx + (1.0 - cos_a) * cy
            m_inv = torch.stack(
                [
                    torch.stack([cos_a, sin_a, tx]),
                    torch.stack([-sin_a, cos_a, ty]),
                    cache["s_001"],
                ]
            )
            a = cache["s_inv"] @ m_inv @ cache["s"]
            theta = a[:2, :].unsqueeze(0).repeat(b, 1, 1)
            grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
            if self._cache_rotation_grid:
                # The full panorama grid is large; keep only the active shape/angle.
                self._rotate_grid_cache.clear()
                self._rotate_grid_cache[grid_cache_key] = grid
        y = F.grid_sample(x_work, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        if orig_dtype == torch.uint8:
            y = y.clamp_(min=0.0, max=255.0).to(dtype=torch.uint8)
        else:
            y = y.to(dtype=orig_dtype)
        if was_channels_last:
            y = y.permute(0, 2, 3, 1)
        return y

    def _maybe_resize_output(self, img: torch.Tensor) -> torch.Tensor:
        max_w = self._max_output_width
        if not max_w or max_w <= 0:
            return img
        width = int(image_width(img))
        height = int(image_height(img))
        if width <= max_w:
            return img
        scale = float(max_w) / float(width)
        new_w = max_w
        new_h = int(height * scale)
        if new_w % 2 != 0:
            new_w -= 1
        if new_h % 2 != 0:
            new_h -= 1
        return resize_image(img, new_width=new_w, new_height=new_h)

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

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        stitch_inputs = context.get("stitch_inputs")
        if stitch_inputs is None:
            raise RuntimeError("StitchingPlugin requires 'stitch_inputs' in context")
        image_data_list: List[Dict[str, Any]]
        if isinstance(stitch_inputs, dict):
            views = stitch_inputs.get("views")
            if isinstance(views, (list, tuple)):
                image_data_list = list(views)
            else:
                left = stitch_inputs.get("left")
                right = stitch_inputs.get("right")
                if left is None or right is None:
                    raise RuntimeError(
                        "StitchingPlugin needs at least 2 input views (left/right or a list under 'views')"
                    )
                image_data_list = [left, right]
        elif isinstance(stitch_inputs, (list, tuple)):
            image_data_list = list(stitch_inputs)
        else:
            raise TypeError(f"Unexpected stitch_inputs type: {type(stitch_inputs)}")

        if len(image_data_list) < 2:
            raise RuntimeError("StitchingPlugin needs at least 2 input views")
        for idx, image_data in enumerate(image_data_list):
            if not isinstance(image_data, dict):
                raise TypeError(
                    f"StitchingPlugin expected stitch_inputs[{idx}] to be a dict, got {type(image_data)}"
                )
            if "img" not in image_data:
                raise RuntimeError(f"StitchingPlugin stitch_inputs[{idx}] missing 'img' key")

        self._ensure_initialized(context)

        imgs = [_to_tensor(image_data["img"]) for image_data in image_data_list]
        ids = _first_not_none(
            image_data_list[0].get("frame_ids"),
            image_data_list[0].get("ids"),
            image_data_list[0].get("img_id"),
        )
        if isinstance(ids, (list, tuple)):
            ids = torch.tensor(ids, dtype=torch.int64)
        if isinstance(ids, torch.Tensor) and ids.is_cuda:
            ids = ids.detach().cpu()
        if ids is None:
            raise RuntimeError("StitchingPlugin could not resolve frame ids")

        pre_stats_list: Optional[List[Optional[Dict[str, Tuple[float, float, float]]]]] = None
        if self._capture_rgb_stats:
            pre_stats_list = []
            for img in imgs:
                try:
                    pre_stats_list.append(MOTLoadVideoWithOrig.compute_rgb_stats(img))
                except Exception:
                    pre_stats_list.append(None)

        if self._left_color_pipeline is not None:
            try:
                data_0 = {"img": imgs[0]}
                data_0 = self._left_color_pipeline(data_0)
                if "img" in data_0:
                    imgs[0] = data_0["img"]
            except Exception:
                logger.debug("Left stitch color pipeline failed", exc_info=True)
        if self._right_color_pipeline is not None:
            try:
                data_1 = {"img": imgs[1]}
                data_1 = self._right_color_pipeline(data_1)
                if "img" in data_1:
                    imgs[1] = data_1["img"]
            except Exception:
                logger.debug("Right stitch color pipeline failed", exc_info=True)

        device = imgs[0].device
        self._create_stitcher(context=context, imgs=imgs, device=device)
        if isinstance(self._stitcher, (CudaStitchPanoF32, CudaStitchPanoU8)):
            if len(imgs) != 2:
                raise RuntimeError(
                    f"Expected 2 stitch inputs for {type(self._stitcher).__name__}; got {len(imgs)}"
                )
            imgs_0 = self._ensure_rgba_channels_last(imgs[0])
            imgs_1 = self._ensure_rgba_channels_last(imgs[1])
            blended = torch.empty(
                [
                    imgs_0.shape[0],
                    self._stitcher.canvas_height(),
                    self._stitcher.canvas_width(),
                    imgs_0.shape[-1],
                ],
                dtype=imgs_0.dtype,
                device=imgs_0.device,
            )
            stream_handle = torch.cuda.current_stream(imgs_0.device).cuda_stream
            self._stitcher.process(
                imgs_0.contiguous(),
                imgs_1.contiguous(),
                blended,
                stream_handle,
            )
        elif isinstance(self._stitcher, (CudaStitchPanoNF32, CudaStitchPanoNU8)):
            imgs_rgba = [self._ensure_rgba_channels_last(img) for img in imgs]
            blended = torch.empty(
                [
                    imgs_rgba[0].shape[0],
                    self._stitcher.canvas_height(),
                    self._stitcher.canvas_width(),
                    imgs_rgba[0].shape[-1],
                ],
                dtype=imgs_rgba[0].dtype,
                device=imgs_rgba[0].device,
            )
            stream_handle = torch.cuda.current_stream(imgs_rgba[0].device).cuda_stream
            self._stitcher.process([img.contiguous() for img in imgs_rgba], blended, stream_handle)
        else:
            blended = self._stitcher.forward(inputs=imgs)

        blended = self._prepare_frame_for_video(blended, image_roi=None)

        rotate_degrees = self._resolve_rotation_degrees(context)
        if rotate_degrees is not None and abs(rotate_degrees) > 1e-6:
            blended = self._rotate_tensor_keep_size(blended, rotate_degrees)

        rgb_stats: Optional[Dict[str, Any]] = None
        if self._capture_rgb_stats:
            try:
                post_stats = MOTLoadVideoWithOrig.compute_rgb_stats(blended)
            except Exception:
                post_stats = None
            rgb_stats = {"inputs": pre_stats_list, "stitched": post_stats}
            if pre_stats_list is not None and len(pre_stats_list) >= 1:
                rgb_stats["left"] = pre_stats_list[0]
            if pre_stats_list is not None and len(pre_stats_list) >= 2:
                rgb_stats["right"] = pre_stats_list[1]

        stitched_for_output = self._maybe_resize_output(blended)

        if self._width_t is None or self._height_t is None:
            self._width_t = torch.tensor([image_width(blended)], dtype=torch.int64)
            self._height_t = torch.tensor([image_height(blended)], dtype=torch.int64)

        # Prepare a lightweight dict for the MMEngine pipeline. We keep this local
        # and surface the pipeline outputs as top-level Aspen context keys rather
        # than nesting under a shared `data` object.
        data_item: Dict[str, Any] = {}
        fps_value = context.get("stitch_fps") or context.get("fps")
        hm_real_time_fps = None
        fps_scalar = None
        if fps_value:
            try:
                fps_scalar = float(fps_value)
                hm_real_time_fps = [fps_scalar for _ in range(len(ids))]
                data_item["hm_real_time_fps"] = hm_real_time_fps
                data_item["fps"] = fps_scalar
            except Exception:
                fps_scalar = None
                hm_real_time_fps = None

        data_item.update(
            dict(
                img=blended,
                img_info=dict(
                    frame_id=int(ids[0].item()) if isinstance(ids, torch.Tensor) else int(ids[0])
                ),
                img_prefix=None,
                img_id=ids,
            )
        )
        if rgb_stats is not None:
            data_item.setdefault("debug_rgb_stats", {})["stitch"] = rgb_stats

        pipeline = context.get("stitch_data_pipeline")
        inputs = None
        data_samples = None
        if pipeline is not None:
            data_item = data_item.copy()
            pipeline_input = data_item.copy()
            pipeline_input.pop("dataset_results", None)
            pipeline_result = pipeline(pipeline_input)
            if "img" in pipeline_result:
                raise RuntimeError("StitchingPlugin pipeline returned unexpected 'img' key")
            inputs = pipeline_result.pop("inputs")
            data_samples = pipeline_result.get("data_samples")

        original_images = make_channels_first(blended)
        original_images = wrap_tensor(original_images)
        stitched_debug = data_item.get("debug_rgb_stats")
        if not isinstance(stitched_debug, dict):
            stitched_debug = {}
        debug_rgb_stats = context.get("debug_rgb_stats")
        if isinstance(debug_rgb_stats, dict):
            # Merge any upstream stats into a copy to avoid mutating shared state.
            merged_stats = dict(debug_rgb_stats)
            merged_stats.update(stitched_debug)
            stitched_debug = merged_stats

        if self._iter_num == 1 and self._dir_name is not None:
            frame_path = os.path.join(self._dir_name, "s.png")
            stitched_frame = unwrap_tensor(stitched_for_output)
            print(
                f"Stitched frame resolution: {image_width(stitched_frame)} x {image_height(stitched_frame)}"
            )
            print(f"Saving first stitched frame to {frame_path}")
            cv2.imwrite(frame_path, make_visible_image(stitched_frame[0], force_numpy=True))

        out: Dict[str, Any] = {
            "original_images": original_images,
            "ids": ids,
            "frame_ids": ids,
            "debug_rgb_stats": stitched_debug,
            "img": wrap_tensor(stitched_for_output),
        }
        # Only surface model inputs when a stitch_data_pipeline is configured.
        if inputs is not None:
            out["inputs"] = wrap_tensor(tensor=inputs)
        if data_samples is not None:
            out["data_samples"] = data_samples
        if hm_real_time_fps is not None:
            out["hm_real_time_fps"] = hm_real_time_fps
        if fps_scalar is not None:
            out["fps"] = fps_scalar

        return out

    def input_keys(self) -> set:
        return {"stitch_inputs", "stitch_data_pipeline", "stitch_fps", "fps", "debug_rgb_stats"}

    def output_keys(self) -> set:
        return {
            "inputs",
            "data_samples",
            "original_images",
            "debug_rgb_stats",
            "hm_real_time_fps",
            "fps",
            "ids",
            "frame_ids",
            "img",
        }

    def get_post_stitch_rotate_degrees(self) -> Optional[float]:
        return self._post_stitch_rotate_degrees

    def set_post_stitch_rotate_degrees(self, degrees: Optional[float]) -> None:
        self._post_stitch_rotate_degrees = degrees


def _is_none(val: Any) -> bool:
    if isinstance(val, str) and val == "None":
        return True
    return val is None


def _fix_clip_box(clip_box: Any, hw: List[int]) -> Any:
    if isinstance(clip_box, list):
        if _is_none(clip_box[0]):
            clip_box[0] = 0
        if _is_none(clip_box[1]):
            clip_box[1] = 0
        if _is_none(clip_box[2]):
            clip_box[2] = hw[1]
        if _is_none(clip_box[3]):
            clip_box[3] = hw[0]
        clip_box = np.array(clip_box, dtype=np.int64)
    return clip_box
