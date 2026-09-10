"""GPT-style causal transformer for camera pan/zoom prediction.

This extends the existing camera transformer approach to a decoder-only,
autoregressive-friendly model that emits a camera state per frame.

The model consumes per-frame feature vectors (tracking/pose aggregates plus
previous camera state) and predicts normalized camera center/height:
  - cx, cy in [0, 1]
  - h_ratio in [0, 1] (relative to the global frame height)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from hmlib.camera.camera_transformer import CameraNorm, PositionalEncoding

OPENDRIVE_UNIAD_MODEL_ID = "OpenDriveLab/UniAD2.0_R101_nuScenes"
OPENDRIVE_UNIAD_PLANNING_FILE = "ckpts/uniad_base_e2e.pth"


@dataclass(frozen=True)
class CameraGPTConfig:
    d_in: int = 11
    d_out: int = 3
    model_kind: str = "gpt"
    feature_mode: str = "legacy_prev_slow"
    include_pose: bool = False
    include_rink: bool = False
    rink_input: str = "stats"
    rink_grid_height: int = 32
    rink_grid_width: int = 64
    rink_encoder_version: int = 1
    source_model_id: str = ""
    source_checkpoint: str = ""
    source_init: str = ""
    residual_prev_y: bool = False
    residual_scale: float = 0.1
    # Defaults chosen to be ~10M params with typical camgpt training settings.
    d_model: int = 448
    nhead: int = 8
    nlayers: int = 4
    dim_feedforward: int = 1792
    dropout: float = 0.1


class CameraPanZoomGPT(nn.Module):
    """Causal transformer that predicts a camera state per timestep."""

    def __init__(self, cfg: CameraGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(int(cfg.d_in), int(cfg.d_model))
        self.pe = PositionalEncoding(int(cfg.d_model))
        if cfg.rink_input not in {"stats", "grid"}:
            raise ValueError(f"Unknown rink input: {cfg.rink_input}")
        self.rink_encoder = None
        if cfg.include_rink and cfg.rink_input == "grid":
            if cfg.rink_encoder_version != 1:
                raise ValueError(f"Unsupported rink encoder version: {cfg.rink_encoder_version}")
            if min(cfg.rink_grid_height, cfg.rink_grid_width) < 2:
                raise ValueError("Rink grid dimensions must be >=2")
            # Flattening retains absolute placement in the tracking coordinate plane.
            self.rink_encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(cfg.rink_grid_height * cfg.rink_grid_width, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.Tanh(),
            )
        self.encoder: Optional[nn.TransformerEncoder] = None
        self.decoder: Optional[nn.TransformerDecoder] = None
        if str(getattr(cfg, "model_kind", "gpt")) == "drivegpt":
            dec_layer = nn.TransformerDecoderLayer(
                d_model=int(cfg.d_model),
                nhead=int(cfg.nhead),
                dim_feedforward=int(cfg.dim_feedforward),
                batch_first=True,
                dropout=float(cfg.dropout),
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(cfg.nlayers))
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=int(cfg.d_model),
                nhead=int(cfg.nhead),
                dim_feedforward=int(cfg.dim_feedforward),
                batch_first=True,
                dropout=float(cfg.dropout),
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.nlayers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(cfg.d_model)),
            nn.Linear(int(cfg.d_model), int(cfg.d_model)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cfg.d_model), int(cfg.d_out)),
            nn.Sigmoid(),
        )

    def encode_rink(self, rink: torch.Tensor) -> torch.Tensor:
        if self.rink_encoder is None:
            raise ValueError("This checkpoint does not use static rink grids")
        expected = (1, self.cfg.rink_grid_height, self.cfg.rink_grid_width)
        if rink.ndim != 4 or tuple(rink.shape[1:]) != expected:
            raise ValueError(f"Expected rink [B,{expected}], got {tuple(rink.shape)}")
        return self.rink_encoder(rink)

    def forward(
        self, x: torch.Tensor, *, rink_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: [B, T, D]
        if x.ndim != 3:
            raise ValueError(f"Expected x to be [B,T,D], got shape {tuple(x.shape)}")
        t = int(x.shape[1])
        h = self.input(x)
        if self.rink_encoder is not None:
            if rink_embedding is None or rink_embedding.shape != (x.shape[0], self.cfg.d_model):
                raise ValueError("Static rink checkpoint requires one rink embedding per sequence")
            h = h + rink_embedding.unsqueeze(1)
        elif rink_embedding is not None:
            raise ValueError("Unexpected rink embedding for a checkpoint without static rink input")
        h = self.pe(h)
        # Causal (GPT-style) mask: prevent attending to future timesteps.
        mask = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)
        if self.decoder is not None:
            h = self.decoder(h, h, tgt_mask=mask, memory_mask=mask)
        elif self.encoder is not None:
            h = self.encoder(h, mask=mask)
        else:
            raise RuntimeError("CameraPanZoomGPT has neither encoder nor decoder initialized")
        out = self.head(h)
        if bool(getattr(self.cfg, "residual_prev_y", False)):
            d_out = int(self.cfg.d_out)
            if int(x.shape[-1]) < d_out:
                raise ValueError(
                    "residual_prev_y requires previous target values at the end of the input"
                )
            prev_y = x[..., -d_out:].to(device=out.device, dtype=out.dtype)
            delta = (out - 0.5) * (2.0 * float(getattr(self.cfg, "residual_scale", 0.1)))
            out = torch.clamp(prev_y + delta, 0.0, 1.0)
        return out


def pack_gpt_checkpoint(
    model: nn.Module, norm: CameraNorm, window: int, cfg: CameraGPTConfig
) -> Dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "norm": {
            "scale_x": float(norm.scale_x),
            "scale_y": float(norm.scale_y),
            "max_players": int(norm.max_players),
        },
        "window": int(window),
        "model": {
            "d_in": int(cfg.d_in),
            "d_out": int(cfg.d_out),
            "model_kind": str(getattr(cfg, "model_kind", "gpt")),
            "feature_mode": str(cfg.feature_mode),
            "include_pose": bool(cfg.include_pose),
            "include_rink": bool(getattr(cfg, "include_rink", False)),
            "rink_input": cfg.rink_input,
            "rink_encoder_version": cfg.rink_encoder_version,
            "rink_grid_height": cfg.rink_grid_height,
            "rink_grid_width": cfg.rink_grid_width,
            "source_model_id": str(getattr(cfg, "source_model_id", "")),
            "source_checkpoint": str(getattr(cfg, "source_checkpoint", "")),
            "source_init": str(getattr(cfg, "source_init", "")),
            "residual_prev_y": bool(getattr(cfg, "residual_prev_y", False)),
            "residual_scale": float(getattr(cfg, "residual_scale", 0.1)),
            "d_model": int(cfg.d_model),
            "nhead": int(cfg.nhead),
            "nlayers": int(cfg.nlayers),
            "dim_feedforward": int(cfg.dim_feedforward),
            "dropout": float(cfg.dropout),
        },
    }


def unpack_gpt_checkpoint(
    ckpt: Dict[str, Any],
) -> Tuple[Dict[str, Any], CameraNorm, int, CameraGPTConfig]:
    sd = ckpt["state_dict"]
    n = ckpt["norm"]
    window = int(ckpt.get("window", 16))
    model_cfg = ckpt.get("model") or {}
    include_pose = model_cfg.get("include_pose", ckpt.get("include_pose", False))
    include_rink = model_cfg.get("include_rink", ckpt.get("include_rink", False))
    cfg = CameraGPTConfig(
        d_in=int(model_cfg.get("d_in", 11)),
        d_out=int(model_cfg.get("d_out", 3)),
        model_kind=str(model_cfg.get("model_kind", "gpt")),
        feature_mode=str(model_cfg.get("feature_mode", "legacy_prev_slow")),
        include_pose=bool(include_pose),
        include_rink=bool(include_rink),
        rink_input=str(model_cfg.get("rink_input", "stats")),
        rink_encoder_version=int(model_cfg.get("rink_encoder_version", 1)),
        rink_grid_height=int(model_cfg.get("rink_grid_height", 32)),
        rink_grid_width=int(model_cfg.get("rink_grid_width", 64)),
        source_model_id=str(model_cfg.get("source_model_id", "")),
        source_checkpoint=str(model_cfg.get("source_checkpoint", "")),
        source_init=str(model_cfg.get("source_init", "")),
        residual_prev_y=bool(model_cfg.get("residual_prev_y", False)),
        residual_scale=float(model_cfg.get("residual_scale", 0.1)),
        d_model=int(model_cfg.get("d_model", 448)),
        nhead=int(model_cfg.get("nhead", 8)),
        nlayers=int(model_cfg.get("nlayers", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 1792)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    norm = CameraNorm(
        scale_x=float(n["scale_x"]),
        scale_y=float(n["scale_y"]),
        max_players=int(n["max_players"]),
    )
    return sd, norm, window, cfg


def resolve_opendrive_checkpoint(
    *,
    model_id: str = OPENDRIVE_UNIAD_MODEL_ID,
    filename: str = OPENDRIVE_UNIAD_PLANNING_FILE,
    checkpoint_path: Optional[str] = None,
) -> str:
    """Resolve an OpenDriveLab checkpoint path, downloading it from Hugging Face if needed."""
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"OpenDriveLab checkpoint does not exist: {path}")
        return str(path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as ex:
        raise RuntimeError(
            "huggingface_hub is required to download OpenDriveLab weights. "
            "Install it or pass --drivegpt-source-checkpoint."
        ) from ex

    return str(hf_hub_download(repo_id=str(model_id), filename=str(filename)))


def init_from_opendrive_uniad_planning(
    model: CameraPanZoomGPT, checkpoint_path: str
) -> Dict[str, Any]:
    """Initialize matching decoder blocks from UniAD's planning head.

    UniAD's public OpenDriveLab checkpoint contains a planning ``TransformerDecoder``
    head with self-attention, cross-attention, FFN, and three layer norms. DriveGPT uses
    the same decoder layer layout with causal masks over both target and memory streams;
    non-matching tensors are reported and skipped.
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"OpenDriveLab checkpoint does not exist: {ckpt_path}")
    if str(getattr(model.cfg, "model_kind", "gpt")) != "drivegpt":
        raise ValueError(
            "OpenDriveLab UniAD planning initialization requires model_kind='drivegpt'"
        )
    if (
        int(model.cfg.d_model) != 256
        or int(model.cfg.nhead) != 8
        or int(model.cfg.dim_feedforward) != 512
    ):
        raise ValueError(
            "OpenDriveLab UniAD planning initialization requires "
            "d_model=256, nhead=8, and dim_feedforward=512."
        )
    if int(model.cfg.nlayers) != 3:
        raise ValueError(
            "OpenDriveLab UniAD planning initialization requires nlayers=3 to match "
            "the public UniAD planning decoder."
        )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    source_sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(source_sd, dict):
        raise ValueError(f"Unsupported OpenDriveLab checkpoint format: {type(source_sd).__name__}")

    target_sd = model.state_dict()
    copied = 0
    skipped = 0
    initialized_layers: set[int] = set()
    mapping = {
        "self_attn.in_proj_weight": "self_attn.in_proj_weight",
        "self_attn.in_proj_bias": "self_attn.in_proj_bias",
        "self_attn.out_proj.weight": "self_attn.out_proj.weight",
        "self_attn.out_proj.bias": "self_attn.out_proj.bias",
        "multihead_attn.in_proj_weight": "multihead_attn.in_proj_weight",
        "multihead_attn.in_proj_bias": "multihead_attn.in_proj_bias",
        "multihead_attn.out_proj.weight": "multihead_attn.out_proj.weight",
        "multihead_attn.out_proj.bias": "multihead_attn.out_proj.bias",
        "linear1.weight": "linear1.weight",
        "linear1.bias": "linear1.bias",
        "linear2.weight": "linear2.weight",
        "linear2.bias": "linear2.bias",
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "norm3.weight": "norm3.weight",
        "norm3.bias": "norm3.bias",
    }

    for layer_idx in range(int(model.cfg.nlayers)):
        source_prefix = f"planning_head.attn_module.layers.{layer_idx}."
        target_prefix = f"decoder.layers.{layer_idx}."
        layer_copied = 0
        for source_suffix, target_suffix in mapping.items():
            source_key = source_prefix + source_suffix
            target_key = target_prefix + target_suffix
            source_tensor = source_sd.get(source_key)
            target_tensor = target_sd.get(target_key)
            if source_tensor is None or target_tensor is None:
                skipped += 1
                continue
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                skipped += 1
                continue
            target_sd[target_key] = source_tensor.to(dtype=target_tensor.dtype)
            copied += 1
            layer_copied += 1
        if layer_copied:
            initialized_layers.add(layer_idx)

    expected_copied = len(mapping) * int(model.cfg.nlayers)
    if copied != expected_copied:
        raise ValueError(
            f"Incomplete UniAD planning initialization: copied {copied}/{expected_copied} "
            "compatible decoder tensors. Use d_model=256, nhead=8, dim_feedforward=512, "
            "and nlayers matching the OpenDriveLab planning checkpoint."
        )

    model.load_state_dict(target_sd, strict=True)
    return {
        "source": "opendrive-uniad-planning",
        "checkpoint_path": str(ckpt_path),
        "copied_tensors": int(copied),
        "skipped_tensors": int(skipped),
        "initialized_layers": int(len(initialized_layers)),
    }
