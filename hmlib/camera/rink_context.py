"""Compact, generation-bound rink context in the tracking coordinate plane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from hmlib.camera.camera_gpt import CameraPanZoomGPT

from hmlib.camera.camera_transformer import CameraNorm

RINK_CONTEXT_SCHEMA = "hockey-drivegpt-rink-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rink_context_path(tracking_csv: str) -> Path:
    tracking = Path(tracking_csv)
    suffix = tracking.stem.removeprefix("tracking")
    return tracking.with_name(f"rink_context{suffix}.json")


def read_rink_context(tracking_csv: str, *, verify: bool = True) -> dict[str, Any]:
    """Require an explicit binding to this export, never guess a mask's scale."""
    path = rink_context_path(tracking_csv)
    context = json.loads(path.read_text())
    required = {"schema", "coordinate_space", "frame_size", "tracking", "masks", "evidence"}
    if set(context) != required or context["schema"] != RINK_CONTEXT_SCHEMA:
        raise ValueError(f"Invalid rink context schema: {path}")
    if context["coordinate_space"] != "original_stitched_pixels":
        raise ValueError(f"Rink context must use original stitched tracking pixels: {path}")
    size = np.asarray(context["frame_size"], dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError(f"Invalid rink frame_size [width, height]: {path}")
    if not context["evidence"] or not isinstance(context["evidence"], str):
        raise ValueError(f"Rink context needs coordinate-provenance evidence: {path}")
    binding = context["tracking"]
    if set(binding) != {"file", "sha256"} or binding["file"] != Path(tracking_csv).name:
        raise ValueError(f"Rink context is bound to a different tracking export: {path}")
    if verify and file_sha256(Path(tracking_csv)) != binding["sha256"]:
        raise ValueError(f"Rink context tracking checksum mismatch: {path}")
    if not isinstance(context["masks"], list) or not context["masks"]:
        raise ValueError(f"Rink context must include at least one mask: {path}")
    for mask in context["masks"]:
        if set(mask) != {"file", "sha256", "mask_to_tracking"}:
            raise ValueError(f"Invalid rink mask binding: {path}")
        if Path(mask["file"]).name != mask["file"]:
            raise ValueError(f"Rink masks must use game-local filenames: {path}")
        source = (path.parent / mask["file"]).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError(f"Rink mask must be inside its game directory: {source}")
        affine = np.asarray(mask["mask_to_tracking"], dtype=np.float64)
        if (
            affine.shape != (2, 3)
            or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:, :2])) < 1e-12
        ):
            raise ValueError(f"Invalid mask-to-tracking affine: {path}")
        if verify and file_sha256(source) != mask["sha256"]:
            raise ValueError(f"Rink mask checksum mismatch: {source}")
    return context


def _sample_mask(
    mask: np.ndarray,
    norm: CameraNorm,
    height: int,
    width: int,
    *,
    mask_to_tracking: np.ndarray | None = None,
) -> np.ndarray:
    """Sample occupancy in the SAME normalized x/y plane as player boxes.

    Four-by-four subcell sampling preserves fractional boundary occupancy without
    allocating a full normalized canvas. Input pixels use edge coordinates;
    out-of-image samples are background. Returns unpooled binary subcell samples.
    """
    if height < 2 or width < 2 or norm.scale_x <= 0 or norm.scale_y <= 0:
        raise ValueError("Rink grid dimensions and normalization scales must be positive")
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.any(mask):
        raise ValueError("Rink mask must be a nonempty 2D foreground mask")
    affine = np.eye(3, dtype=np.float64)
    if mask_to_tracking is not None:
        affine[:2] = mask_to_tracking
    inverse = np.linalg.inv(affine)
    samples = 4
    x = (np.arange(width * samples) + 0.5) * norm.scale_x / (width * samples)
    y = (np.arange(height * samples) + 0.5) * norm.scale_y / (height * samples)
    xx, yy = np.meshgrid(x, y)
    map_x = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2] - 0.5).astype(np.float32)
    map_y = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2] - 0.5).astype(np.float32)
    sampled = cv2.remap(
        cv2.compare(mask.astype(np.uint8, copy=False), 0, cv2.CMP_GT),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled


def _pool_samples(sampled: np.ndarray, height: int, width: int) -> np.ndarray:
    return (sampled.reshape(height, 4, width, 4).mean(axis=(1, 3), dtype=np.float32) / 255.0)[None]


def mask_to_grid(
    mask: np.ndarray,
    norm: CameraNorm,
    height: int,
    width: int,
    *,
    mask_to_tracking: np.ndarray | None = None,
    frame_size: tuple[float, float] | None = None,
) -> np.ndarray:
    sampled = _sample_mask(mask, norm, height, width, mask_to_tracking=mask_to_tracking)
    if frame_size is not None:
        _clip_samples(sampled, norm, height, width, frame_size)
    return _pool_samples(sampled, height, width)


def _clip_samples(sampled, norm, height, width, frame_size):
    # Shared by offline and live rasterization, including subcell frame edges.
    frame_w, frame_h = frame_size
    x = (np.arange(width * 4) + 0.5) * norm.scale_x / (width * 4)
    y = (np.arange(height * 4) + 0.5) * norm.scale_y / (height * 4)
    sampled[:, x >= frame_w] = 0
    sampled[y >= frame_h, :] = 0


def load_rink_grid(tracking_csv: str, norm: CameraNorm, height: int, width: int) -> np.ndarray:
    context = read_rink_context(tracking_csv)
    frame_w, frame_h = context["frame_size"]
    if frame_w > norm.scale_x + 1e-3 or frame_h > norm.scale_y + 1e-3:
        raise ValueError(f"Rink tracking canvas exceeds model normalization: {tracking_csv}")
    grids = []
    for binding in context["masks"]:
        path = Path(tracking_csv).parent / binding["file"]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot decode rink mask: {path}")
        grids.append(
            _sample_mask(
                mask, norm, height, width, mask_to_tracking=np.asarray(binding["mask_to_tracking"])
            )
        )
    combined = np.maximum.reduce(grids)
    # An affine may map a source mask past the visible tracking canvas.
    _clip_samples(combined, norm, height, width, (frame_w, frame_h))
    result = _pool_samples(combined, height, width)
    if not np.any(result):
        raise ValueError(f"Rink has no occupancy in the tracking coordinate plane: {tracking_csv}")
    return result


class StaticRinkEncoderCache:
    """Inference-only cache; raw profile masks must be immutable per revision."""

    def __init__(self) -> None:
        self._key = None
        self._mask = None
        self._embedding = None

    def get(
        self,
        model: CameraPanZoomGPT,
        mask: np.ndarray | torch.Tensor,
        norm: CameraNorm,
        frame_size: tuple[int, int],
        *,
        game_id: str | None,
        revision: str | None,
    ) -> tuple[torch.Tensor, bool]:

        if model.training:
            raise ValueError("Learned rink embeddings may only be cached in evaluation mode")
        w, h = frame_size
        if tuple(mask.shape) != (h, w):
            raise ValueError("Live rink mask and original tracking-frame dimensions differ")
        if w > norm.scale_x + 1e-3 or h > norm.scale_y + 1e-3:
            raise ValueError("Live rink canvas exceeds checkpoint normalization")
        if revision is None:
            raise ValueError("Static rink profiles require an explicit immutable geometry revision")
        parameter = next(model.parameters())
        key = (
            id(model),
            tuple(p._version for p in model.parameters()),
            id(mask),
            mask._version if torch.is_tensor(mask) and not torch.is_inference(mask) else None,
            revision,
            game_id,
            frame_size,
            norm.scale_x,
            norm.scale_y,
            model.cfg.rink_grid_height,
            model.cfg.rink_grid_width,
            model.cfg.rink_encoder_version,
            parameter.device,
            parameter.dtype,
        )
        changed = key != self._key
        if changed:
            raw = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
            grid = mask_to_grid(
                raw,
                norm,
                model.cfg.rink_grid_height,
                model.cfg.rink_grid_width,
                frame_size=frame_size,
            )
            if not np.any(grid):
                raise ValueError("Live rink has no occupancy in the checkpoint coordinate plane")
            value = (
                torch.from_numpy(grid)
                .unsqueeze(0)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            with torch.no_grad():
                self._embedding = model.encode_rink(value)
            self._key, self._mask = key, mask
        assert self._embedding is not None
        return self._embedding, changed
