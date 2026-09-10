"""Aspen trunk that computes per-frame camera boxes (pan/zoom)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from mmengine.structures import InstanceData

from hmlib.bbox.box_functions import center, clamp_box, make_box_at_center
from hmlib.builder import HM
from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, unpack_gpt_checkpoint
from hmlib.camera.camera_transformer import (
    CameraNorm,
    CameraPanZoomTransformer,
    build_frame_base_features_torch,
    build_frame_features_torch,
    build_player_box_features_torch,
    unpack_checkpoint,
)
from hmlib.camera.clusters import ClusterMan
from hmlib.camera.rink_context import StaticRinkEncoderCache
from hmlib.log import logger
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor

from .base import Plugin


@HM.register_module()
class CameraControllerPlugin(Plugin):
    """
    Camera controller trunk that computes per-frame camera boxes (pan/zoom).

    Modes:
      - controller="rule": cluster-based heuristic similar to PlayTracker.
      - controller="transformer": use trained transformer checkpoint.
      - controller="gpt" / "drivegpt": use trained causal transformer checkpoint.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - frame_id: int for first frame in batch

    Produces in context:
      - camera_boxes: List[torch.Tensor] (TLBR per frame)
      - Side-effects: updates each img_data_sample.pred_cam_box with TLBR tensor
    """

    def __init__(
        self,
        enabled: bool = True,
        controller: str = "rule",
        model_path: Optional[str] = None,
        window: int = 8,
        aspect_ratio: float = 16.0 / 9.0,
    ) -> None:
        super().__init__(enabled=enabled)
        self._requested_controller = str(controller)
        self._controller = "gpt" if self._requested_controller == "drivegpt" else str(controller)
        self._model: Optional[CameraPanZoomTransformer] = None
        self._gpt_model: Optional[CameraPanZoomGPT] = None
        self._gpt_cfg: Optional[CameraGPTConfig] = None
        self._norm: Optional[CameraNorm] = None
        self._window = int(window)
        self._feat_buf: deque = deque(maxlen=self._window)
        self._prev_center: Optional[torch.Tensor] = None
        self._prev_h: Optional[torch.Tensor] = None
        self._prev_y: Optional[torch.Tensor] = None
        self._cluster_man: Optional[ClusterMan] = None
        self._ar = float(aspect_ratio)
        self._feat_device: Optional[torch.device] = None
        self._rink_cache = StaticRinkEncoderCache()

        controller = self._controller

        if controller == "transformer":
            if model_path:
                try:
                    ckpt = torch.load(model_path, map_location="cpu")
                    sd, norm, w = unpack_checkpoint(ckpt)
                    self._model = CameraPanZoomTransformer(d_in=11)
                    self._model.load_state_dict(sd)
                    self._model.eval()
                    self._norm = norm
                    self._window = int(w)
                    self._feat_buf = deque(maxlen=self._window)
                except Exception as ex:
                    raise RuntimeError(
                        f"Failed to load transformer camera controller checkpoint {model_path!r}"
                    ) from ex
            else:
                # No checkpoint -> do not override PlayTracker.
                self._controller = "rule"
        elif controller == "gpt":
            if model_path:
                try:
                    ckpt = torch.load(model_path, map_location="cpu")
                    sd, norm, w, cfg = unpack_gpt_checkpoint(ckpt)
                    if (
                        self._requested_controller == "drivegpt"
                        and str(getattr(cfg, "model_kind", "gpt")) != "drivegpt"
                    ):
                        raise ValueError(
                            "controller='drivegpt' requires a checkpoint with "
                            "model_kind='drivegpt'"
                        )
                    self._gpt_cfg = cfg
                    self._gpt_model = CameraPanZoomGPT(cfg)
                    self._gpt_model.load_state_dict(sd)
                    self._gpt_model.eval()
                    self._norm = norm
                    self._window = int(w)
                    self._feat_buf = deque(maxlen=self._window)
                except Exception as ex:
                    raise RuntimeError(
                        f"Failed to load {controller} camera controller checkpoint "
                        f"{model_path!r}: {ex}"
                    ) from ex
            else:
                if self._requested_controller == "drivegpt":
                    raise ValueError("controller='drivegpt' requires model_path")
                self._controller = "rule"

    def _ensure_cluster_man(self, sizes: List[int] = [3, 2]):
        if self._cluster_man is None:
            self._cluster_man = ClusterMan(sizes=sizes, device="cpu")

    @staticmethod
    def _default_prev_y(d_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if int(d_out) == 3:
            return torch.tensor([0.5, 0.5, 1.0], device=device, dtype=dtype)
        if int(d_out) == 4:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype)
        if int(d_out) == 8:
            return torch.tensor(
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype
            )
        return torch.zeros((int(d_out),), device=device, dtype=dtype)

    def _output_scales(self, frame_w: int, frame_h: int) -> tuple[float, float]:
        if self._norm is None:
            return max(1.0, float(frame_w)), max(1.0, float(frame_h))
        return (
            max(1.0, float(self._norm.scale_x)),
            max(1.0, float(self._norm.scale_y)),
        )

    @staticmethod
    def _fit_box_inside_bounds(box: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        box = box.to(dtype=torch.float32, device=bounds.device)
        bounds = bounds.to(dtype=torch.float32, device=box.device)
        w = torch.clamp(box[2] - box[0], min=1.0)
        h = torch.clamp(box[3] - box[1], min=1.0)
        bw = torch.clamp(bounds[2] - bounds[0], min=1.0)
        bh = torch.clamp(bounds[3] - bounds[1], min=1.0)
        scale = torch.minimum(box.new_tensor(1.0), torch.minimum(bw / w, bh / h))
        w2 = w * scale
        h2 = h * scale
        half_w = w2 * 0.5
        half_h = h2 * 0.5
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        cx = torch.clamp(cx, min=bounds[0] + half_w, max=bounds[2] - half_w)
        cy = torch.clamp(cy, min=bounds[1] + half_h, max=bounds[3] - half_h)
        return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h])

    @staticmethod
    def _play_bounds(
        context: Dict[str, Any],
        frame_index: int,
        frame_w: int,
        frame_h: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        frame_bounds = torch.tensor([0, 0, frame_w, frame_h], device=device, dtype=dtype)
        arena = context.get("arena")
        if arena is None:
            return frame_bounds
        arena_t = arena if isinstance(arena, torch.Tensor) else torch.as_tensor(arena)
        if arena_t.ndim == 0:
            raise RuntimeError("camera_controller arena must be a TLBR box, got scalar")
        if arena_t.ndim > 1:
            if arena_t.shape[0] == 1:
                arena_t = arena_t[0]
            elif frame_index < int(arena_t.shape[0]):
                arena_t = arena_t[frame_index]
            else:
                raise RuntimeError(
                    "camera_controller arena batch is shorter than the tracking batch"
                )
        arena_t = arena_t.reshape(-1)
        if int(arena_t.numel()) < 4:
            raise RuntimeError(
                f"camera_controller arena must contain at least 4 values, got {arena_t.numel()}"
            )
        return clamp_box(arena_t[:4].to(device=device, dtype=dtype), frame_bounds)

    def _box_to_prev_y(
        self,
        box: torch.Tensor,
        fast_box: Optional[torch.Tensor],
        d_out: int,
        frame_w: int,
        frame_h: int,
    ) -> torch.Tensor:
        x_scale, y_scale = self._output_scales(frame_w, frame_h)
        slow_tlwh = torch.stack(
            [
                torch.clamp(box[0] / x_scale, 0.0, 1.0),
                torch.clamp(box[1] / y_scale, 0.0, 1.0),
                torch.clamp((box[2] - box[0]) / x_scale, 0.0, 1.0),
                torch.clamp((box[3] - box[1]) / y_scale, 0.0, 1.0),
            ],
            dim=0,
        ).to(dtype=torch.float32)
        if int(d_out) == 3:
            return torch.stack(
                [
                    slow_tlwh[0] + slow_tlwh[2] * 0.5,
                    slow_tlwh[1] + slow_tlwh[3] * 0.5,
                    slow_tlwh[3],
                ],
                dim=0,
            )
        if int(d_out) == 4:
            return slow_tlwh
        if int(d_out) == 8:
            fast = fast_box if fast_box is not None else box
            fast_tlwh = torch.stack(
                [
                    torch.clamp(fast[0] / x_scale, 0.0, 1.0),
                    torch.clamp(fast[1] / y_scale, 0.0, 1.0),
                    torch.clamp((fast[2] - fast[0]) / x_scale, 0.0, 1.0),
                    torch.clamp((fast[3] - fast[1]) / y_scale, 0.0, 1.0),
                ],
                dim=0,
            ).to(dtype=torch.float32)
            return torch.cat([slow_tlwh, fast_tlwh], dim=0)
        return self._default_prev_y(int(d_out), device=box.device, dtype=torch.float32)

    @staticmethod
    def _uses_prev_y_feature(cfg: Optional[CameraGPTConfig]) -> bool:
        if cfg is None:
            return False
        return str(getattr(cfg, "feature_mode", "legacy_prev_slow")) in {
            "base_prev_y",
            "players_prev_y",
        }

    @staticmethod
    def _resolve_device(context: Dict[str, Any], fallback: Optional[torch.device] = None):
        # Prefer flattened Aspen context keys.
        for key in ("inputs", "img", "original_images"):
            t = context.get(key)
            if isinstance(t, torch.Tensor):
                return t.device
            dev = getattr(t, "device", None)
            if isinstance(dev, torch.device):
                return dev
        shared = context.get("shared", {}) or {}
        device = shared.get("camera_device") or shared.get("device")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception:
                pass
        if fallback is not None:
            return fallback
        if torch.cuda.is_available():
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                pass
        return torch.device("cpu")

    def _pose_features(
        self, pose_results: Optional[List[Any]], frame_index: int, device: torch.device
    ) -> torch.Tensor:
        """Extract a fixed-length (8) pose feature vector for the current frame."""
        feat = torch.zeros((8,), device=device, dtype=torch.float32)
        if self._norm is None:
            return feat
        try:
            if not isinstance(pose_results, list) or frame_index >= len(pose_results):
                return feat
            pr = pose_results[frame_index]
            preds = pr.get("predictions") if isinstance(pr, dict) else None
            ds0 = preds[0] if isinstance(preds, list) and preds else None
            inst0 = getattr(ds0, "pred_instances", ds0)
            if isinstance(inst0, dict):
                bxs = inst0.get("bboxes")
                kps = inst0.get("keypoint_scores")
                bbox_scores = inst0.get("bbox_scores")
                scores = inst0.get("scores")
            else:
                bxs = getattr(inst0, "bboxes", None)
                kps = getattr(inst0, "keypoint_scores", None)
                bbox_scores = getattr(inst0, "bbox_scores", None)
                scores = getattr(inst0, "scores", None)
        except Exception:
            return feat

        if bxs is not None:
            try:
                if torch.is_tensor(bxs):
                    bxs_t = bxs.to(device=device, dtype=torch.float32)
                else:
                    bxs_t = torch.as_tensor(bxs, device=device, dtype=torch.float32)
                bxs_t = bxs_t.reshape(-1, 4)
                if bxs_t.numel() > 0:
                    cxn = (bxs_t[:, 0] + bxs_t[:, 2]) * 0.5 / max(1e-6, float(self._norm.scale_x))
                    cyn = (bxs_t[:, 1] + bxs_t[:, 3]) * 0.5 / max(1e-6, float(self._norm.scale_y))
                    hn = (bxs_t[:, 3] - bxs_t[:, 1]) / max(1e-6, float(self._norm.scale_y))
                    feat[0] = min(float(bxs_t.shape[0]) / max(1, int(self._norm.max_players)), 1.0)
                    feat[1] = torch.clamp(torch.mean(cxn), 0.0, 1.0)
                    feat[2] = torch.clamp(torch.mean(cyn), 0.0, 1.0)
                    feat[3] = torch.clamp(torch.std(cxn, unbiased=False), 0.0, 1.0)
                    feat[4] = torch.clamp(torch.std(cyn, unbiased=False), 0.0, 1.0)
                    feat[5] = torch.clamp(torch.mean(hn), 0.0, 1.0)
            except Exception:
                pass

        score_val = None
        for vv in (kps, bbox_scores, scores):
            if vv is None:
                continue
            try:
                if torch.is_tensor(vv):
                    vv_t = vv.to(device=device, dtype=torch.float32)
                else:
                    vv_t = torch.as_tensor(vv, device=device, dtype=torch.float32)
                if vv_t is not None and vv_t.numel() > 0:
                    score_val = torch.mean(vv_t)
                    break
            except Exception:
                continue
        if score_val is not None:
            feat[6] = torch.clamp(score_val, 0.0, 1.0)

        if kps is not None:
            try:
                if torch.is_tensor(kps):
                    kk = kps.to(device=device, dtype=torch.float32)
                else:
                    kk = torch.as_tensor(kps, device=device, dtype=torch.float32)
                if kk is not None and kk.numel() > 0:
                    feat[7] = torch.mean((kk > 0.5).to(dtype=feat.dtype))
            except Exception:
                pass

        return feat

    def _static_rink_embedding(self, context, frame_w, frame_h, device):
        if self._gpt_model is None or self._gpt_cfg is None or self._norm is None:
            raise RuntimeError("Static rink context requires a loaded camera model")
        profile = context.get("rink_profile")
        if (
            not isinstance(profile, dict)
            or profile.get("coordinate_space") != "original_stitched_pixels"
        ):
            raise ValueError(
                "Static rink input requires a profile in original stitched tracking pixels; "
                "set ice_config.params.require_geometry_provenance=true for this checkpoint"
            )
        mask = unwrap_tensor(profile.get("combined_mask"))
        if mask is None:
            raise ValueError("Static rink input requires the complete rink mask")
        if profile.get("frame_size") != [frame_w, frame_h]:
            raise ValueError("Rink profile geometry does not match the tracked frame")
        if next(self._gpt_model.parameters()).device != device:
            self._gpt_model.to(device)
        embedding, changed = self._rink_cache.get(
            self._gpt_model,
            mask,
            self._norm,
            (frame_w, frame_h),
            game_id=context.get("game_id") or context.get("shared", {}).get("game_id"),
            revision=profile.get("geometry_revision"),
        )
        if changed:
            self._feat_buf.clear()
            self._prev_center = self._prev_h = self._prev_y = None
        return embedding

    def _rink_features(self, context: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Fixed-length rink features (7,) derived from rink_profile or rink_mask_0.png."""
        feat = torch.zeros((7,), device=device, dtype=torch.float32)
        if self._norm is None:
            return feat
        sx = max(1e-6, float(self._norm.scale_x))
        sy = max(1e-6, float(self._norm.scale_y))

        rp = context.get("rink_profile")
        if isinstance(rp, dict):
            try:
                bbox = rp.get("combined_bbox")
                centroid = rp.get("centroid")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
                    feat[0] = torch.clamp(
                        torch.tensor(x1 / sx, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[1] = torch.clamp(
                        torch.tensor(y1 / sy, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[2] = torch.clamp(
                        torch.tensor(x2 / sx, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[3] = torch.clamp(
                        torch.tensor(y2 / sy, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                if centroid is not None and len(centroid) == 2:
                    cx, cy = float(centroid[0]), float(centroid[1])
                    feat[4] = torch.clamp(
                        torch.tensor(cx / sx, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[5] = torch.clamp(
                        torch.tensor(cy / sy, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                # area fraction if mask present
                mask = rp.get("combined_mask")
                if mask is not None:
                    try:
                        if torch.is_tensor(mask):
                            m = mask.to(device=device)
                        else:
                            m = torch.as_tensor(mask, device=device)
                        feat[6] = torch.clamp(torch.mean((m > 0).to(dtype=feat.dtype)), 0.0, 1.0)
                    except Exception:
                        pass
                return feat
            except Exception:
                pass

        # Fallback: load rink_mask_0.png from game_dir if present.
        try:
            import cv2

            shared = context.get("shared", {}) or {}
            game_dir = shared.get("game_dir")
            if not game_dir:
                return feat
            p = Path(str(game_dir)) / "rink_mask_0.png"
            if not p.is_file():
                return feat
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                return feat
            mask_t = torch.as_tensor(mask, device=device)
            ys, xs = torch.nonzero(mask_t > 0, as_tuple=True)
            if xs.numel() == 0 or ys.numel() == 0:
                return feat
            x1 = xs.min().to(dtype=feat.dtype)
            y1 = ys.min().to(dtype=feat.dtype)
            x2 = xs.max().to(dtype=feat.dtype)
            y2 = ys.max().to(dtype=feat.dtype)
            cx = xs.to(dtype=feat.dtype).mean()
            cy = ys.to(dtype=feat.dtype).mean()
            area = xs.numel() / float(mask.shape[0] * mask.shape[1])
            feat[0] = torch.clamp(x1 / sx, 0.0, 1.0)
            feat[1] = torch.clamp(y1 / sy, 0.0, 1.0)
            feat[2] = torch.clamp(x2 / sx, 0.0, 1.0)
            feat[3] = torch.clamp(y2 / sy, 0.0, 1.0)
            feat[4] = torch.clamp(cx / sx, 0.0, 1.0)
            feat[5] = torch.clamp(cy / sy, 0.0, 1.0)
            feat[6] = torch.clamp(torch.tensor(area, device=device, dtype=feat.dtype), 0.0, 1.0)
        except Exception:
            pass
        return feat

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        # In rule mode, defer entirely to PlayTracker's native camera controller.
        # This avoids accidentally overriding the default camera behavior on Python-only runs.
        if self._controller == "rule":
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)
        pose_results = context.get("pose_results")
        if (
            self._controller == "gpt"
            and self._gpt_cfg is not None
            and bool(getattr(self._gpt_cfg, "include_pose", False))
            and (not isinstance(pose_results, list) or len(pose_results) < video_len)
        ):
            raise RuntimeError(
                "Camera GPT checkpoint expects pose features, but pose_results are missing or "
                "shorter than the tracking batch. Use a no-pose checkpoint or run camera_controller "
                "after the pose trunk."
            )

        cam_boxes: List[torch.Tensor] = []
        cam_fast_boxes: List[torch.Tensor] = []
        self._ensure_cluster_man()

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            inst: InstanceData = getattr(img_data_sample, "pred_track_instances", None)
            raw_boxes = (
                unwrap_tensor(inst.bboxes) if inst is not None and "bboxes" in inst else None
            )
            device = (
                raw_boxes.device if torch.is_tensor(raw_boxes) else self._resolve_device(context)
            )
            if self._feat_device is None:
                self._feat_device = device
            elif self._feat_device != device:
                self._feat_device = device
                self._feat_buf.clear()
                self._prev_center = None
                self._prev_h = None
                self._prev_y = None

            ori_shape = img_data_sample.metainfo.get("ori_shape")
            H = int(ori_shape[0]) if isinstance(ori_shape, (list, tuple, torch.Size)) else int(1080)
            W = int(ori_shape[1]) if isinstance(ori_shape, (list, tuple, torch.Size)) else int(1920)
            rink_embedding = None
            if (
                self._gpt_cfg is not None
                and self._gpt_cfg.include_rink
                and self._gpt_cfg.rink_input == "grid"
            ):
                if not isinstance(ori_shape, (list, tuple, torch.Size)) or len(ori_shape) < 2:
                    raise ValueError("Static rink input requires original frame dimensions")
                rink_embedding = self._static_rink_embedding(context, W, H, device)
            play_bounds = self._play_bounds(context, frame_index, W, H, device, dtype=torch.float32)
            if inst is None or not hasattr(inst, "bboxes"):
                # Default to centered wide shot
                h_px = H * 0.8
                w_px = h_px * self._ar
                cx, cy = W / 2.0, H / 2.0
                box = torch.tensor(
                    [cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2],
                    dtype=torch.float32,
                    device=device,
                )
                box = self._fit_box_inside_bounds(box, play_bounds)
                cam_boxes.append(box)
                setattr(img_data_sample, "pred_cam_box", box)
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and int(self._gpt_cfg.d_out) == 8
                ):
                    cam_fast_boxes.append(box)
                    setattr(img_data_sample, "pred_cam_fast_box", box)
                try:
                    w_denom, h_denom = self._output_scales(W, H)
                    self._prev_center = torch.stack(
                        [
                            (box[0] + box[2]) / (2.0 * w_denom),
                            (box[1] + box[3]) / (2.0 * h_denom),
                        ],
                        dim=0,
                    )
                    self._prev_h = (box[3] - box[1]) / h_denom
                except Exception:
                    pass
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and self._uses_prev_y_feature(self._gpt_cfg)
                ):
                    # Seed prev_y from the default wide shot.
                    try:
                        w_denom, h_denom = self._output_scales(W, H)
                        slow_tlwh = torch.stack(
                            [
                                torch.clamp(box[0] / w_denom, 0.0, 1.0),
                                torch.clamp(box[1] / h_denom, 0.0, 1.0),
                                torch.clamp((box[2] - box[0]) / w_denom, 0.0, 1.0),
                                torch.clamp((box[3] - box[1]) / h_denom, 0.0, 1.0),
                            ],
                            dim=0,
                        ).to(dtype=torch.float32)
                        d_out = int(self._gpt_cfg.d_out)
                        if d_out == 3:
                            self._prev_y = torch.stack(
                                [
                                    slow_tlwh[0] + slow_tlwh[2] * 0.5,
                                    slow_tlwh[1] + slow_tlwh[3] * 0.5,
                                    slow_tlwh[3],
                                ],
                                dim=0,
                            )
                        elif d_out == 4:
                            self._prev_y = slow_tlwh
                        elif d_out == 8:
                            self._prev_y = torch.cat([slow_tlwh, slow_tlwh], dim=0)
                        else:
                            self._prev_y = self._default_prev_y(
                                d_out, device=device, dtype=torch.float32
                            )
                    except Exception:
                        self._prev_y = self._default_prev_y(
                            int(self._gpt_cfg.d_out), device=device, dtype=torch.float32
                        )
                continue

            det_tlbr = unwrap_tensor(inst.bboxes)
            inst.bboxes = wrap_tensor(det_tlbr)

            if not isinstance(det_tlbr, torch.Tensor):
                det_tlbr = torch.as_tensor(det_tlbr, device=device, dtype=torch.float32)
            device = det_tlbr.device
            play_bounds = play_bounds.to(device=device, dtype=torch.float32)
            if self._feat_device is None:
                self._feat_device = device
            elif self._feat_device != device:
                self._feat_device = device
                self._feat_buf.clear()
                self._prev_center = None
                self._prev_h = None
                self._prev_y = None
            track_mask = get_track_mask(inst)
            if isinstance(track_mask, torch.Tensor):
                det_tlbr = det_tlbr[track_mask]
            if det_tlbr.numel() == 0:
                h_px = H * 0.8
                w_px = h_px * self._ar
                cx, cy = W / 2.0, H / 2.0
                box = torch.tensor(
                    [cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2],
                    dtype=torch.float32,
                    device=device,
                )
                box = self._fit_box_inside_bounds(box, play_bounds)
                cam_boxes.append(box)
                setattr(img_data_sample, "pred_cam_box", box)
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and int(self._gpt_cfg.d_out) == 8
                ):
                    cam_fast_boxes.append(box)
                    setattr(img_data_sample, "pred_cam_fast_box", box)
                try:
                    w_denom, h_denom = self._output_scales(W, H)
                    self._prev_center = torch.stack(
                        [
                            (box[0] + box[2]) / (2.0 * w_denom),
                            (box[1] + box[3]) / (2.0 * h_denom),
                        ],
                        dim=0,
                    )
                    self._prev_h = (box[3] - box[1]) / h_denom
                except Exception:
                    pass
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and self._uses_prev_y_feature(self._gpt_cfg)
                ):
                    try:
                        w_denom, h_denom = self._output_scales(W, H)
                        slow_tlwh = torch.stack(
                            [
                                torch.clamp(box[0] / w_denom, 0.0, 1.0),
                                torch.clamp(box[1] / h_denom, 0.0, 1.0),
                                torch.clamp((box[2] - box[0]) / w_denom, 0.0, 1.0),
                                torch.clamp((box[3] - box[1]) / h_denom, 0.0, 1.0),
                            ],
                            dim=0,
                        ).to(dtype=torch.float32)
                        d_out = int(self._gpt_cfg.d_out)
                        if d_out == 3:
                            self._prev_y = torch.stack(
                                [
                                    slow_tlwh[0] + slow_tlwh[2] * 0.5,
                                    slow_tlwh[1] + slow_tlwh[3] * 0.5,
                                    slow_tlwh[3],
                                ],
                                dim=0,
                            )
                        elif d_out == 4:
                            self._prev_y = slow_tlwh
                        elif d_out == 8:
                            self._prev_y = torch.cat([slow_tlwh, slow_tlwh], dim=0)
                        else:
                            self._prev_y = self._default_prev_y(
                                d_out, device=device, dtype=torch.float32
                            )
                    except Exception:
                        self._prev_y = self._default_prev_y(
                            int(self._gpt_cfg.d_out), device=device, dtype=torch.float32
                        )
                continue

            # Convert to TLWH for features
            tlwh = det_tlbr.clone()
            tlwh[:, 2] = tlwh[:, 2] - tlwh[:, 0]
            tlwh[:, 3] = tlwh[:, 3] - tlwh[:, 1]

            box_out: Optional[torch.Tensor] = None
            box_fast_out: Optional[torch.Tensor] = None
            if (
                self._controller == "transformer"
                and self._model is not None
                and self._norm is not None
            ):
                model_device = next(self._model.parameters()).device
                if model_device != device:
                    self._model.to(device)
                feat = build_frame_features_torch(
                    tlwh=tlwh,
                    norm=self._norm,
                    prev_cam_center=self._prev_center,
                    prev_cam_h=self._prev_h,
                ).to(device=device, dtype=torch.float32)
                self._feat_buf.append(feat.unsqueeze(0))
                if len(self._feat_buf) >= self._window:
                    x = torch.cat(list(self._feat_buf), dim=0).unsqueeze(0)
                    with torch.no_grad():
                        pred = self._model(x).squeeze(0)
                    cx, cy, hr = pred[0], pred[1], pred[2]
                    self._prev_center = torch.stack([cx, cy], dim=0)
                    self._prev_h = hr
                    w_denom, h_denom = self._output_scales(W, H)
                    h_px = torch.clamp(hr * h_denom, min=1.0)
                    w_px = h_px * self._ar
                    cx_px = cx * w_denom
                    cy_px = cy * h_denom
                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.stack([left, top, right, bottom], dim=0).to(
                        dtype=det_tlbr.dtype, device=device
                    )
                    box_out = clamp_box(
                        box_out,
                        torch.tensor([0, 0, W, H], dtype=box_out.dtype, device=device),
                    )
            elif (
                self._controller == "gpt"
                and self._gpt_model is not None
                and self._norm is not None
                and self._gpt_cfg is not None
            ):
                pose_feat = (
                    self._pose_features(pose_results, frame_index, device)
                    if bool(getattr(self._gpt_cfg, "include_pose", False))
                    else None
                )
                rink_feat = (
                    self._rink_features(context, device)
                    if self._gpt_cfg.include_rink and self._gpt_cfg.rink_input == "stats"
                    else None
                )

                feature_mode = str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow"))
                if feature_mode in {"base_prev_y", "players_prev_y"}:
                    base_feat = build_frame_base_features_torch(tlwh=tlwh, norm=self._norm)
                    if feature_mode == "players_prev_y":
                        base_feat = torch.cat(
                            [
                                base_feat,
                                build_player_box_features_torch(
                                    tlwh=tlwh,
                                    norm=self._norm,
                                    max_players=int(self._norm.max_players),
                                ),
                            ],
                            dim=0,
                        )
                    if pose_feat is not None:
                        base_feat = torch.cat([base_feat, pose_feat], dim=0)
                    if rink_feat is not None:
                        base_feat = torch.cat([base_feat, rink_feat], dim=0)
                    if self._prev_y is None or int(self._prev_y.shape[0]) != int(
                        self._gpt_cfg.d_out
                    ):
                        self._prev_y = self._default_prev_y(
                            int(self._gpt_cfg.d_out), device=device, dtype=base_feat.dtype
                        )
                    feat = torch.cat([base_feat, self._prev_y.to(device=device)], dim=0)
                else:
                    feat = build_frame_features_torch(
                        tlwh=tlwh,
                        norm=self._norm,
                        prev_cam_center=self._prev_center,
                        prev_cam_h=self._prev_h,
                    )
                    if pose_feat is not None:
                        feat = torch.cat([feat, pose_feat], dim=0)
                    if rink_feat is not None:
                        feat = torch.cat([feat, rink_feat], dim=0)

                expected_dim = int(self._gpt_cfg.d_in)
                actual_dim = int(feat.shape[0])
                if actual_dim != expected_dim:
                    msg = (
                        "Camera GPT feature schema mismatch: "
                        f"feature_mode={feature_mode!r} include_pose="
                        f"{bool(getattr(self._gpt_cfg, 'include_pose', False))} include_rink="
                        f"{bool(getattr(self._gpt_cfg, 'include_rink', False))} "
                        f"expected d_in={expected_dim}, got {actual_dim}"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                gpt_device = next(self._gpt_model.parameters()).device
                if gpt_device != device:
                    self._gpt_model.to(device)
                self._feat_buf.append(feat.unsqueeze(0))
                # GPT supports variable context length; emit a prediction as soon as we have any history.
                x = torch.cat(list(self._feat_buf), dim=0).unsqueeze(0)
                with torch.no_grad():
                    pred_seq = self._gpt_model(x, rink_embedding=rink_embedding).squeeze(0)
                pred_last = pred_seq[-1]

                if int(self._gpt_cfg.d_out) == 3:
                    cx, cy, hr = pred_last[0], pred_last[1], pred_last[2]
                    w_denom, h_denom = self._output_scales(W, H)
                    h_px = torch.clamp(hr * h_denom, min=1.0)
                    w_px = h_px * self._ar
                    cx_px = cx * w_denom
                    cy_px = cy * h_denom
                    # Keep box fully inside the frame without shrinking (avoid clamp distortion).
                    w_over = w_px > float(W)
                    w_px = torch.where(w_over, det_tlbr.new_tensor(float(W)), w_px)
                    h_px = torch.where(w_over, w_px / self._ar, h_px)
                    h_over = h_px > float(H)
                    h_px = torch.where(h_over, det_tlbr.new_tensor(float(H)), h_px)
                    w_px = torch.where(h_over, h_px * self._ar, w_px)
                    cx_px = torch.clamp(cx_px, w_px * 0.5, float(W) - w_px * 0.5)
                    cy_px = torch.clamp(cy_px, h_px * 0.5, float(H) - h_px * 0.5)
                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.stack([left, top, right, bottom], dim=0).to(
                        dtype=det_tlbr.dtype, device=device
                    )
                    box_out = clamp_box(
                        box_out,
                        torch.tensor([0, 0, W, H], dtype=box_out.dtype, device=device),
                    )
                    try:
                        w_denom, h_denom = self._output_scales(W, H)
                        self._prev_center = torch.stack(
                            [
                                (box_out[0] + box_out[2]) / (2.0 * w_denom),
                                (box_out[1] + box_out[3]) / (2.0 * h_denom),
                            ],
                            dim=0,
                        )
                        self._prev_h = (box_out[3] - box_out[1]) / h_denom
                    except Exception:
                        pass
                elif int(self._gpt_cfg.d_out) in (4, 8):
                    slow_tlwh = pred_last[:4]
                    w_denom, h_denom = self._output_scales(W, H)
                    cx_px = (slow_tlwh[0] + slow_tlwh[2] * 0.5) * w_denom
                    cy_px = (slow_tlwh[1] + slow_tlwh[3] * 0.5) * h_denom
                    h0 = torch.clamp(slow_tlwh[3] * h_denom, min=1.0)

                    # Enforce output aspect ratio (avoid mixed up/downscale in crop pipeline).
                    h_px = h0
                    w_px = h_px * self._ar
                    # Clamp to frame bounds while preserving aspect ratio (shrink only).
                    w_over = w_px > float(W)
                    w_px = torch.where(w_over, det_tlbr.new_tensor(float(W)), w_px)
                    h_px = torch.where(w_over, w_px / self._ar, h_px)
                    h_over = h_px > float(H)
                    h_px = torch.where(h_over, det_tlbr.new_tensor(float(H)), h_px)
                    w_px = torch.where(h_over, h_px * self._ar, w_px)

                    # Clamp center so the box fits without shrinking.
                    cx_px = torch.clamp(cx_px, w_px * 0.5, float(W) - w_px * 0.5)
                    cy_px = torch.clamp(cy_px, h_px * 0.5, float(H) - h_px * 0.5)

                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.stack([left, top, right, bottom], dim=0).to(
                        dtype=det_tlbr.dtype, device=device
                    )
                    box_out = clamp_box(
                        box_out,
                        torch.tensor([0, 0, W, H], dtype=box_out.dtype, device=device),
                    )
                    try:
                        w_denom, h_denom = self._output_scales(W, H)
                        cxn = (box_out[0] + box_out[2]) / (2.0 * w_denom)
                        cyn = (box_out[1] + box_out[3]) / (2.0 * h_denom)
                        hrn = (box_out[3] - box_out[1]) / h_denom
                        self._prev_center = torch.stack([cxn, cyn], dim=0)
                        self._prev_h = hrn
                    except Exception:
                        pass

                    if int(self._gpt_cfg.d_out) == 8:
                        fast_tlwh = pred_last[4:8]
                        xf = fast_tlwh[0] * w_denom
                        yf = fast_tlwh[1] * h_denom
                        wf = torch.clamp(fast_tlwh[2] * w_denom, min=1.0)
                        hf = torch.clamp(fast_tlwh[3] * h_denom, min=1.0)
                        box_fast_out = torch.stack([xf, yf, xf + wf, yf + hf], dim=0).to(
                            dtype=det_tlbr.dtype, device=device
                        )
                        box_fast_out = clamp_box(
                            box_fast_out,
                            torch.tensor([0, 0, W, H], dtype=box_fast_out.dtype, device=device),
                        )

            if box_out is None:
                # Rule-based fallback: union of detections -> fixed height with aspect
                left = (
                    torch.min(det_tlbr[:, 0])
                    if len(det_tlbr)
                    else torch.tensor(0.0).to(device=det_tlbr.device, non_blocking=True)
                )
                right = (
                    torch.max(det_tlbr[:, 2])
                    if len(det_tlbr)
                    else det_tlbr.new_tensor(float(W)).to(device=det_tlbr.device, non_blocking=True)
                )
                top = (
                    torch.min(det_tlbr[:, 1])
                    if len(det_tlbr)
                    else torch.tensor(0.0).to(device=det_tlbr.device, non_blocking=True)
                )
                bottom = (
                    torch.max(det_tlbr[:, 3])
                    if len(det_tlbr)
                    else det_tlbr.new_tensor(float(H)).to(device=det_tlbr.device, non_blocking=True)
                )
                uni = torch.stack([left, top, right, bottom])
                c = center(uni)
                h_px = torch.clamp((bottom - top) * 1.4, min=H * 0.35, max=H * 0.95)
                w_px = h_px * self._ar
                box_out = make_box_at_center(c, w=w_px, h=h_px)
                if not hasattr(self, "_wh_box"):
                    self._wh_box = torch.tensor([0, 0, W, H], dtype=box_out.dtype).to(
                        device=box_out.device, non_blocking=True
                    )
                box_out = clamp_box(box_out, self._wh_box)

            box_out = self._fit_box_inside_bounds(box_out, play_bounds)
            if box_fast_out is not None:
                box_fast_out = self._fit_box_inside_bounds(box_fast_out, play_bounds)

            # For base_prev_y checkpoints, feed back the *actual* box used after clamping/aspect enforcement.
            if (
                self._controller == "gpt"
                and self._gpt_cfg is not None
                and self._uses_prev_y_feature(self._gpt_cfg)
            ):
                try:
                    self._prev_y = self._box_to_prev_y(
                        box_out,
                        box_fast_out,
                        int(self._gpt_cfg.d_out),
                        W,
                        H,
                    )
                except Exception:
                    self._prev_y = self._default_prev_y(
                        int(self._gpt_cfg.d_out), device=device, dtype=torch.float32
                    )

            cam_boxes.append(box_out)
            if box_fast_out is not None:
                cam_fast_boxes.append(box_fast_out)
            setattr(img_data_sample, "pred_cam_box", wrap_tensor(box_out))
            if box_fast_out is not None:
                setattr(img_data_sample, "pred_cam_fast_box", wrap_tensor(box_fast_out))

        out: Dict[str, Any] = {"camera_boxes": wrap_tensor(torch.stack(cam_boxes, dim=0))}
        if cam_fast_boxes and len(cam_fast_boxes) == len(cam_boxes):
            out["camera_fast_boxes"] = wrap_tensor(torch.stack(cam_fast_boxes, dim=0))
        return out

    def input_keys(self):
        return {
            "data_samples",
            "pose_results",
            "inputs",
            "img",
            "original_images",
            "shared",
            "rink_profile",
            "arena",
        }

    def output_keys(self):
        return {"camera_boxes", "camera_fast_boxes"}
