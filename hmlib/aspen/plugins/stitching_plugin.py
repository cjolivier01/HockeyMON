"""Aspen plugin that stitches multi-camera inputs into a panorama frame."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import Compose

from hmlib.config import get_game_config, get_nested_value
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.log import logger
from hmlib.stitching.blend import FEATHERED_BLEND_MODES, resolve_blend_settings
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


def _to_tensor(tensor: torch.Tensor | StreamTensorBase | np.ndarray) -> torch.Tensor:
    tensor = unwrap_tensor(tensor)
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    raise TypeError(f"Unexpected tensor type: {type(tensor)}")


def _parse_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
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
        pto_project_file: str | None = None,
        dir_name: str | None = None,
        blend_mode: str = "laplacian",
        blend_feather_fraction: float | None = None,
        python_blender: bool = False,
        use_cuda_pano_n: bool = False,
        minimize_blend: bool = True,
        dtype: str | torch.dtype | None = None,
        max_blend_levels: int | None = None,
        no_cuda_streams: bool = False,
        post_stitch_rotate_degrees: float | None = None,
        left_color_pipeline: list[dict[str, Any]] | None = None,
        right_color_pipeline: list[dict[str, Any]] | None = None,
        capture_rgb_stats: bool = False,
        max_output_width: int | None = None,
        cache_rotation_grid: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._pto_project_file = pto_project_file
        self._dir_name = Path(dir_name) if dir_name else None
        if self._pto_project_file and not self._dir_name:
            self._dir_name = Path(self._pto_project_file).parent
        # Not str(): an explicit YAML null resolves to None, which means "inherit
        # the default" the same way it does in HStream. str() would make it the
        # literal "None" and turn an inherit into a rejected mode.
        #
        # Resolved here rather than at the first frame: a disabled trunk becomes a
        # no-op stub without reaching this constructor, so a bad blend key fails
        # at graph construction instead of after video readers and models are up.
        self._blend = resolve_blend_settings(
            blend_mode=blend_mode,
            blend_feather_fraction=blend_feather_fraction,
            max_blend_levels=max_blend_levels,
        )
        self._python_blender = bool(python_blender)
        self._use_cuda_pano_n = bool(use_cuda_pano_n)
        self._minimize_blend = bool(minimize_blend)
        self._dtype = _parse_dtype(dtype)
        self._no_cuda_streams = bool(no_cuda_streams)
        self._post_stitch_rotate_degrees = post_stitch_rotate_degrees
        self._left_color_pipeline_cfg = left_color_pipeline
        self._right_color_pipeline_cfg = right_color_pipeline
        self._capture_rgb_stats = bool(capture_rgb_stats)
        self._max_output_width = int(max_output_width) if max_output_width else None
        self._cache_rotation_grid = bool(cache_rotation_grid)

        self._left_color_pipeline: Compose | None = None
        self._right_color_pipeline: Compose | None = None
        self._config_ref: dict[str, Any] | None = None
        self._initialized: bool = False
        self._stitcher = None
        self._mask_generation_signature = None
        self._geometry_stitcher = None
        self._geometry_session = None
        self._geometry_source_identity = None
        self._rotate_cache: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        self._rotate_grid_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._width_t: torch.Tensor | None = None
        self._height_t: torch.Tensor | None = None
        self._iter_num: int = 0

    def _ensure_initialized(self, context: dict[str, Any]) -> None:
        if self._initialized:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        cfg = shared.get("game_config")
        if cfg is None:
            game_id = shared.get("game_id")
            if game_id:
                cfg = get_game_config(game_id=game_id)
        if isinstance(cfg, dict):
            self._config_ref = cfg
        self._build_color_pipelines()
        self._initialized = True

    def _build_color_pipelines(self) -> None:
        self._left_color_pipeline = None
        self._right_color_pipeline = None

        def _make_pipeline(spec: list[dict[str, Any]] | None) -> Compose | None:
            if not spec:
                return None
            pipeline = Compose(copy.deepcopy(spec))
            for tf in pipeline:
                if tf.__class__.__name__ == "HmImageColorAdjust":
                    tf.channel_order = "bgr"
                    if self._config_ref is not None:
                        tf.config_ref = self._config_ref
            return pipeline

        self._left_color_pipeline = _make_pipeline(self._left_color_pipeline_cfg)
        self._right_color_pipeline = _make_pipeline(self._right_color_pipeline_cfg)

    def _resolve_dir_name(self, context: dict[str, Any]) -> Path:
        from hmlib.stitching.native_calibration import shared_native_calibration

        native = shared_native_calibration()
        if native is not None:
            self._dir_name = native.directory
            return self._dir_name
        if self._dir_name is not None:
            return self._dir_name
        if self._pto_project_file:
            self._dir_name = Path(self._pto_project_file).parent
            return self._dir_name
        shared = context.get("shared", {})
        game_dir = shared.get("game_dir")
        if game_dir:
            self._dir_name = Path(game_dir)
            return self._dir_name
        raise RuntimeError("StitchingPlugin needs dir_name or pto_project_file")

    def _geometry_source(self, context: dict[str, Any]) -> dict[str, str]:
        """Return a stable identity for the calibration behind the stitcher.

        Native HStream configs carry a revision generated from the actual
        stitch artifacts.  For older configs, the PTO project is the stable
        source from which those artifacts are generated, so hash its contents.
        If neither is available, deliberately fall back to a process-local
        token: a mask without a durable geometry identity must never be reused
        across runs merely because its dimensions happen to match.
        """
        if self._geometry_source_identity is not None:
            return self._geometry_source_identity

        from hmlib.stitching.native_calibration import shared_native_calibration

        native = shared_native_calibration()
        if native is not None:
            self._geometry_source_identity = {
                "kind": "native-hugin-generation",
                "value": native.generation,
            }
            return self._geometry_source_identity

        cfg = self._config_ref
        if cfg is None:
            shared = context.get("shared", {})
            candidate = shared.get("game_config") if isinstance(shared, dict) else None
            cfg = candidate if isinstance(candidate, dict) else None

        native_revision = None
        if isinstance(cfg, dict):
            for path in (
                "stitching.calibration_frame_selection.context.rink_mask_revision",
                "stitching.rink_mask_revision",
                "rink.rink_mask_revision",
            ):
                native_revision = get_nested_value(cfg, path, None)
                if native_revision is not None:
                    break

        if native_revision is not None:
            self._geometry_source_identity = {
                "kind": "native-rink-mask-revision",
                "value": str(native_revision),
            }
            return self._geometry_source_identity

        pto_project_file = self._pto_project_file
        if pto_project_file is None and isinstance(cfg, dict):
            pto_project_file = get_nested_value(cfg, "stitching.pto_project_file", None)
        if pto_project_file and "://" not in str(pto_project_file):
            pto_path = Path(str(pto_project_file)).expanduser()
            if pto_path.is_file():
                digest = hashlib.sha256()
                with pto_path.open("rb") as pto_file:
                    for chunk in iter(lambda: pto_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                self._geometry_source_identity = {
                    "kind": "pto-sha256",
                    "value": digest.hexdigest(),
                }
                return self._geometry_source_identity

        if self._geometry_session is None:
            self._geometry_session = uuid.uuid4().hex
        self._geometry_source_identity = {
            "kind": "process-local",
            "value": self._geometry_session,
        }
        return self._geometry_source_identity

    def _make_geometry_revision(
        self,
        context: dict[str, Any],
        imgs: list[torch.Tensor],
        blended: torch.Tensor,
        applied_rotation: float,
    ) -> str:
        shared = context.get("shared", {})
        game_id = context.get("game_id")
        if game_id is None and isinstance(shared, dict):
            game_id = shared.get("game_id")
        blend = self._blend
        payload = {
            "schema": "hm-stitch-geometry-v2",
            "game_id": str(game_id) if game_id is not None else None,
            "source": self._geometry_source(context),
            "input_shapes": [list(img.shape[1:]) for img in imgs],
            "stitched_shape": list(blended.shape[1:]),
            "post_stitch_rotate_degrees": float(applied_rotation),
            # The normalized mode, not the raw spelling: alias spellings render
            # identically, so they must not key different cached rink masks, and
            # this cache deletes entries filed under other revisions.
            "blend_mode": blend.mode,
            "max_output_width": self._max_output_width,
        }
        if blend.mode in FEATHERED_BLEND_MODES:
            # Only a feathered mode moves pixels with the width, and adding the key
            # only for those keeps every already-cached Laplacian/hard-seam mask
            # valid. Keyed off the set so a future feathered mode is covered
            # rather than silently reusing another mode's mask.
            payload["blend_feather_fraction"] = blend.feather_fraction
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

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
        context: dict[str, Any],
        imgs: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        if self._stitcher is not None:
            return
        if device.type == "cpu":
            raise RuntimeError("StitchingPlugin requires a CUDA device")
        if len(imgs) < 2:
            raise RuntimeError("StitchingPlugin needs at least 2 input views")
        dir_name = self._resolve_dir_name(context)
        blend = self._blend

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
            blend_mode=blend.mode,
            feather_fraction=blend.feather_fraction,
            levels=blend.levels,
        )
        self._mask_generation_signature = self._mask_source_signature(dir_name)

    @staticmethod
    def _mask_source_signature(directory):
        paths = sorted(
            path
            for path in Path(directory).iterdir()
            if path.name in (".stitching_artifacts.json", "autooptimiser_out.pto", "seam_file.png")
            or path.name.startswith("mapping_")
        )
        return tuple(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
            for path in paths
            if path.is_file()
        )

    def _render_images(self, imgs):
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

        return blended

    def _make_mask_renderer(self, context, imgs, rotation):
        device = imgs[0].device
        sizes = [(image_width(image), image_height(image)) for image in imgs]
        dtypes = [image.dtype for image in imgs]
        generation = self._mask_generation_signature

        def render(frames):
            directory = str(self._resolve_dir_name(context))
            if generation is not None and self._mask_source_signature(directory) != generation:
                raise ValueError(
                    "Calibration changed during playback; restart before sampling a rink mask"
                )
            if len(frames) != len(sizes):
                raise ValueError("Mask renderer needs the active source camera count")
            # Own maps, scratch space and transforms: the pipeline may already
            # be rendering a later playback frame on another worker.
            renderer = StitchingPlugin(
                dir_name=directory,
                blend_mode=self._blend.mode,
                blend_feather_fraction=self._blend.feather_fraction,
                max_blend_levels=self._blend.levels,
                python_blender=self._python_blender,
                use_cuda_pano_n=self._use_cuda_pano_n,
                minimize_blend=self._minimize_blend,
                dtype=self._dtype,
                max_output_width=self._max_output_width,
                left_color_pipeline=self._left_color_pipeline_cfg,
                right_color_pipeline=self._right_color_pipeline_cfg,
            )
            renderer._ensure_initialized(context)
            prepared = []
            for index, frame in enumerate(frames):
                if (image_width(frame), image_height(frame)) != sizes[index]:
                    raise ValueError("Mask source dimensions differ from the active stitch input")
                prepared.append(frame.unsqueeze(0).to(device=device, dtype=dtypes[index]))
            for index, pipeline in enumerate(
                (renderer._left_color_pipeline, renderer._right_color_pipeline)
            ):
                if pipeline is not None:
                    prepared[index] = pipeline({"img": prepared[index]})["img"]
            renderer._create_stitcher(context, prepared, device)
            if generation is not None and self._mask_source_signature(directory) != generation:
                raise ValueError("Calibration changed while constructing the mask renderer")
            with torch.inference_mode():
                blended = renderer._prepare_frame_for_video(renderer._render_images(prepared))
                if rotation is not None and abs(rotation) > 1e-6:
                    blended = renderer._rotate_tensor_keep_size(blended, rotation)
                result = make_channels_first(blended)[0].detach().cpu()
            return result

        return render

    def _resolve_rotation_degrees(self, context: dict[str, Any]) -> float | None:
        shared = context.get("shared", {})
        cfg = shared.get("game_config") or {}
        val = get_nested_value(cfg, "stitching.post_stitch_rotate_degrees", None)
        if val is not None:
            return float(val)
        return self._post_stitch_rotate_degrees

    def _prepare_frame_for_video(
        self, image: torch.Tensor, image_roi: list[int] | None = None
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

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        from hmlib.stitching.native_calibration import shared_native_calibration

        native_calibration = shared_native_calibration()
        # The runtime replaces this mapping atomically after validating an epoch.
        # Capture it before rendering, so queued frames retain their own geometry.
        native_epoch = (context.get("shared", {}).get("game_config") or {}).get("hm_shared_runtime")

        stitch_inputs = context.get("stitch_inputs")
        if stitch_inputs is None:
            raise RuntimeError("StitchingPlugin requires 'stitch_inputs' in context")
        image_data_list: list[dict[str, Any]]
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

        pre_stats_list: list[dict[str, tuple[float, float, float]] | None] | None = None
        if self._capture_rgb_stats:
            pre_stats_list = []
            for img in imgs:
                try:
                    pre_stats_list.append(MOTLoadVideoWithOrig.compute_rgb_stats(img))
                except (RuntimeError, ValueError, TypeError) as error:
                    logger.warning("Source color statistics unavailable: %s", error)
                    pre_stats_list.append(None)

        # Source tabs show decoded pixels before downstream color/stitch work.
        # The bridge queues only the selected channel in shared desktop runs.
        ui = context.get("shared", {}).get("hm_ui_process")
        if ui is not None and not ui.closed:
            from hmlib.ui.preview_overlays import source_metadata

            for index, source in enumerate(imgs[:2], start=1):
                ui.publish_preview(
                    wrap_tensor(source),
                    name=f"Camera {index}",
                    snapshot=True,
                    metadata=source_metadata(
                        image_data_list[index - 1], role="left" if index == 1 else "right"
                    ),
                )

        if self._left_color_pipeline is not None:
            try:
                data_0 = {"img": imgs[0]}
                data_0 = self._left_color_pipeline(data_0)
                if "img" in data_0:
                    imgs[0] = data_0["img"]
            except (RuntimeError, ValueError, TypeError, KeyError):
                logger.warning(
                    "Left stitch color pipeline failed; using unadjusted input", exc_info=True
                )
        if self._right_color_pipeline is not None:
            try:
                data_1 = {"img": imgs[1]}
                data_1 = self._right_color_pipeline(data_1)
                if "img" in data_1:
                    imgs[1] = data_1["img"]
            except (RuntimeError, ValueError, TypeError, KeyError):
                logger.warning(
                    "Right stitch color pipeline failed; using unadjusted input", exc_info=True
                )

        device = imgs[0].device
        self._create_stitcher(context=context, imgs=imgs, device=device)
        blended = self._render_images(imgs)

        blended = self._prepare_frame_for_video(blended, image_roi=None)

        if native_calibration is not None:
            rotate_degrees = (
                float(native_epoch["post_stitch_rotate_degrees"])
                if native_epoch is not None
                else native_calibration.post_stitch_rotate_degrees
            )
        else:
            rotate_degrees = self._resolve_rotation_degrees(context)
        applied_rotation = 0.0
        if rotate_degrees is not None and abs(rotate_degrees) > 1e-6:
            blended = self._rotate_tensor_keep_size(blended, rotate_degrees)
            applied_rotation = float(rotate_degrees)

        rgb_stats: dict[str, Any] | None = None
        if self._capture_rgb_stats:
            try:
                post_stats = MOTLoadVideoWithOrig.compute_rgb_stats(blended)
            except (RuntimeError, ValueError, TypeError) as error:
                logger.warning("Stitched color statistics unavailable: %s", error)
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
        data_item: dict[str, Any] = {}
        fps_value = context.get("stitch_fps") or context.get("fps")
        hm_real_time_fps = None
        fps_scalar = None
        if fps_value:
            fps_scalar = float(fps_value)
            hm_real_time_fps = [fps_scalar for _ in range(len(ids))]
            data_item["hm_real_time_fps"] = hm_real_time_fps
            data_item["fps"] = fps_scalar

        data_item.update(
            {
                "img": blended,
                "img_info": {
                    "frame_id": int(ids[0].item()) if isinstance(ids, torch.Tensor) else int(ids[0])
                },
                "img_prefix": None,
                "img_id": ids,
            }
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

        # Rotation and canvas/input shape belong to the identity of the frame
        # actually rendered, not a later UI request.  The source identity is
        # stable when the calibration is durable, allowing rink masks to be
        # reused safely across processes.
        if self._geometry_stitcher is not self._stitcher:
            # Stitchers own immutable calibration maps, so a replacement must not
            # inherit the previous identity. Drop the memoized source: a durable
            # source recomputes to the same value, and a process-local one has to
            # become a new token rather than vouch for a different calibration.
            self._geometry_stitcher = self._stitcher
            self._geometry_source_identity = None
            self._geometry_session = None
        geometry_revision = self._make_geometry_revision(
            context=context,
            imgs=imgs,
            blended=blended,
            applied_rotation=applied_rotation,
        )
        out: dict[str, Any] = {
            "original_images": original_images,
            "rink_mask_renderer": self._make_mask_renderer(context, imgs, applied_rotation),
            "camera_input_geometry": {
                "post_stitch_rotate_degrees": applied_rotation,
                "stitched_geometry_revision": geometry_revision,
            },
            "ids": ids,
            "frame_ids": ids,
            "debug_rgb_stats": stitched_debug,
            "img": wrap_tensor(stitched_for_output),
        }
        if native_calibration is not None:
            out["camera_input_geometry"].update(
                {
                    "native_output_generation": (
                        native_epoch["output_generation"]
                        if native_epoch is not None
                        else native_calibration.output_generation
                    ),
                    "native_output_authorization_id": (
                        native_epoch["authorization_id"] if native_epoch is not None else ""
                    ),
                }
            )
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
            "rink_mask_renderer",
            "camera_input_geometry",
            "debug_rgb_stats",
            "hm_real_time_fps",
            "fps",
            "ids",
            "frame_ids",
            "img",
        }

    def get_post_stitch_rotate_degrees(self) -> float | None:
        return self._post_stitch_rotate_degrees

    def set_post_stitch_rotate_degrees(self, degrees: float | None) -> None:
        self._post_stitch_rotate_degrees = degrees


def _is_none(val: Any) -> bool:
    if isinstance(val, str) and val == "None":
        return True
    return val is None


def _fix_clip_box(clip_box: Any, hw: list[int]) -> Any:
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
