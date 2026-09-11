import argparse
import json
import math
import os
import random
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from hmlib.camera.camera_gpt import (
    OPENDRIVE_UNIAD_MODEL_ID,
    OPENDRIVE_UNIAD_PLANNING_FILE,
    CameraGPTConfig,
    CameraPanZoomGPT,
    init_from_opendrive_uniad_planning,
    pack_gpt_checkpoint,
    resolve_opendrive_checkpoint,
)
from hmlib.camera.camera_gpt_dataset import (
    CameraPanZoomGPTIterableDataset,
    GameCsvPaths,
    resolve_csv_paths,
    scan_game_max_xy,
    validate_csv_paths,
)
from hmlib.camera.camera_training_config import catalog_split, expand_training_config
from hmlib.camera.camera_database import (
    discover_database_games,
    geometry as database_geometry,
    load_database_rink,
    scan_database_max_xy,
    split_database_games,
)
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.camera.rink_context import (
    file_sha256,
    load_rink_grid,
    read_rink_context,
    rink_context_path,
)
from hmlib.log import logger


def _tlwh_to_tlbr(tlwh: torch.Tensor) -> torch.Tensor:
    x1 = tlwh[..., 0]
    y1 = tlwh[..., 1]
    w = torch.clamp(tlwh[..., 2], min=1e-6)
    h = torch.clamp(tlwh[..., 3], min=1e-6)
    x2 = x1 + w
    y2 = y1 + h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _mean_iou_tlwh(a_tlwh: torch.Tensor, b_tlwh: torch.Tensor) -> torch.Tensor:
    a = _tlwh_to_tlbr(a_tlwh)
    b = _tlwh_to_tlbr(b_tlwh)
    ix1 = torch.maximum(a[..., 0], b[..., 0])
    iy1 = torch.maximum(a[..., 1], b[..., 1])
    ix2 = torch.minimum(a[..., 2], b[..., 2])
    iy2 = torch.minimum(a[..., 3], b[..., 3])
    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih
    a_area = torch.clamp(a[..., 2] - a[..., 0], min=0.0) * torch.clamp(
        a[..., 3] - a[..., 1], min=0.0
    )
    b_area = torch.clamp(b[..., 2] - b[..., 0], min=0.0) * torch.clamp(
        b[..., 3] - b[..., 1], min=0.0
    )
    union = a_area + b_area - inter
    iou = inter / torch.clamp(union, min=1e-6)
    return iou.mean()


def _runtime_aspect_slow_tlwh(tlwh: torch.Tensor, aspect_norm: float) -> torch.Tensor:
    """Mirror Aspen's slow-box postprocess: keep center/height, derive width from aspect."""
    x = tlwh[..., 0]
    y = tlwh[..., 1]
    src_w = torch.clamp(tlwh[..., 2], min=1e-6)
    src_h = torch.clamp(tlwh[..., 3], min=1e-6)
    aspect = max(1e-6, float(aspect_norm))
    max_h = min(1.0, 1.0 / aspect)
    h = torch.clamp(src_h, min=1e-6, max=max_h)
    out_w = torch.clamp(h * aspect, min=1e-6, max=1.0)
    cx = torch.clamp(x + src_w * 0.5, out_w * 0.5, 1.0 - out_w * 0.5)
    cy = torch.clamp(y + src_h * 0.5, h * 0.5, 1.0 - h * 0.5)
    left = cx - out_w * 0.5
    top = cy - h * 0.5
    return torch.stack([left, top, out_w, h], dim=-1)


def _center_h_to_runtime_tlwh(center_h: torch.Tensor, aspect_norm: float) -> torch.Tensor:
    aspect = max(1e-6, float(aspect_norm))
    max_h = min(1.0, 1.0 / aspect)
    h = torch.clamp(center_h[..., 2], min=1e-6, max=max_h)
    w = torch.clamp(h * aspect, min=1e-6, max=1.0)
    cx = torch.clamp(center_h[..., 0], min=w * 0.5, max=1.0 - w * 0.5)
    cy = torch.clamp(center_h[..., 1], min=h * 0.5, max=1.0 - h * 0.5)
    left = cx - w * 0.5
    top = cy - h * 0.5
    return torch.stack([left, top, w, h], dim=-1)


def _clamp_unit_tlwh(tlwh: torch.Tensor) -> torch.Tensor:
    x1 = torch.clamp(tlwh[..., 0], min=0.0, max=1.0)
    y1 = torch.clamp(tlwh[..., 1], min=0.0, max=1.0)
    x2 = torch.clamp(tlwh[..., 0] + tlwh[..., 2], min=0.0, max=1.0)
    y2 = torch.clamp(tlwh[..., 1] + tlwh[..., 3], min=0.0, max=1.0)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def _runtime_feedback_target(
    pred: torch.Tensor, runtime_slow_aspect_norm: Optional[float]
) -> torch.Tensor:
    if runtime_slow_aspect_norm is None:
        return pred
    d_out = int(pred.shape[-1])
    if d_out == 3:
        tlwh = _center_h_to_runtime_tlwh(pred, runtime_slow_aspect_norm)
        return torch.stack(
            [
                tlwh[..., 0] + tlwh[..., 2] * 0.5,
                tlwh[..., 1] + tlwh[..., 3] * 0.5,
                tlwh[..., 3],
            ],
            dim=-1,
        )
    if d_out == 4:
        return _runtime_aspect_slow_tlwh(pred, runtime_slow_aspect_norm)
    if d_out == 8:
        slow = _runtime_aspect_slow_tlwh(pred[..., :4], runtime_slow_aspect_norm)
        fast = _clamp_unit_tlwh(pred[..., 4:8])
        return torch.cat([slow, fast], dim=-1)
    return pred


def _legacy_prev_slow_feedback_target(
    pred: torch.Tensor, runtime_slow_aspect_norm: Optional[float]
) -> torch.Tensor:
    """Return normalized [cx, cy, h] feedback for legacy_prev_slow inputs."""
    d_out = int(pred.shape[-1])
    if d_out == 3:
        return _runtime_feedback_target(pred, runtime_slow_aspect_norm)
    if d_out == 4:
        slow = pred
    elif d_out == 8:
        slow = pred[..., :4]
    else:
        raise ValueError(f"legacy_prev_slow free-run feedback does not support d_out={d_out}")

    slow = (
        _runtime_aspect_slow_tlwh(slow, runtime_slow_aspect_norm)
        if runtime_slow_aspect_norm is not None
        else _clamp_unit_tlwh(slow)
    )
    cx = torch.clamp(slow[..., 0] + slow[..., 2] * 0.5, 0.0, 1.0)
    cy = torch.clamp(slow[..., 1] + slow[..., 3] * 0.5, 0.0, 1.0)
    h = torch.clamp(slow[..., 3], 0.0, 1.0)
    return torch.stack([cx, cy, h], dim=-1)


def _diff1(x: torch.Tensor) -> torch.Tensor:
    # [B,T,D] -> [B,T-1,D]
    return x[:, 1:, :] - x[:, :-1, :]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    # [B,T,D] -> [B,T-2,D]
    return x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]


def _compute_losses(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    w_l1: float,
    w_iou: float,
    w_vel: float,
    w_acc: float,
    fast_mult: float,
    runtime_slow_aspect_norm: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    d_out = int(y.shape[-1])

    if d_out == 8:
        pred_slow = pred[..., :4]
        y_slow = y[..., :4]
        pred_fast = pred[..., 4:8]
        y_fast = y[..., 4:8]
        pred_slow_loss = (
            _runtime_aspect_slow_tlwh(pred_slow, runtime_slow_aspect_norm)
            if runtime_slow_aspect_norm is not None
            else pred_slow
        )
        pred_fast_loss = _clamp_unit_tlwh(pred_fast)
        y_fast_loss = _clamp_unit_tlwh(y_fast)

        l1_slow = nn.functional.l1_loss(pred_slow_loss, y_slow)
        l1_fast = nn.functional.l1_loss(pred_fast_loss, y_fast_loss)
        l1 = l1_slow + float(fast_mult) * l1_fast
        metrics["l1_slow"] = float(l1_slow.detach().item())
        metrics["l1_fast"] = float(l1_fast.detach().item())

        iou_slow = _mean_iou_tlwh(pred_slow_loss, y_slow)
        iou_fast = _mean_iou_tlwh(pred_fast_loss, y_fast_loss)
        iou_loss = (1.0 - iou_slow) + float(fast_mult) * (1.0 - iou_fast)
        metrics["iou_slow"] = float(iou_slow.detach().item())
        metrics["iou_fast"] = float(iou_fast.detach().item())

        vel_slow = (
            nn.functional.l1_loss(_diff1(pred_slow_loss), _diff1(y_slow))
            if pred.size(1) > 1
            else pred.new_zeros(())
        )
        vel_fast = (
            nn.functional.l1_loss(_diff1(pred_fast_loss), _diff1(y_fast_loss))
            if pred.size(1) > 1
            else pred.new_zeros(())
        )
        vel = vel_slow + float(fast_mult) * vel_fast
        metrics["vel_slow"] = float(vel_slow.detach().item()) if pred.size(1) > 1 else 0.0
        metrics["vel_fast"] = float(vel_fast.detach().item()) if pred.size(1) > 1 else 0.0

        acc_slow = (
            nn.functional.l1_loss(_diff2(pred_slow_loss), _diff2(y_slow))
            if pred.size(1) > 2
            else pred.new_zeros(())
        )
        acc_fast = (
            nn.functional.l1_loss(_diff2(pred_fast_loss), _diff2(y_fast_loss))
            if pred.size(1) > 2
            else pred.new_zeros(())
        )
        acc = acc_slow + float(fast_mult) * acc_fast
        metrics["acc_slow"] = float(acc_slow.detach().item()) if pred.size(1) > 2 else 0.0
        metrics["acc_fast"] = float(acc_fast.detach().item()) if pred.size(1) > 2 else 0.0

    else:
        pred_loss = pred
        y_loss = y
        if runtime_slow_aspect_norm is not None:
            if d_out == 3:
                pred_loss = _runtime_feedback_target(pred, runtime_slow_aspect_norm)
                y_loss = _runtime_feedback_target(y, runtime_slow_aspect_norm)
            elif d_out == 4:
                pred_loss = _runtime_aspect_slow_tlwh(pred, runtime_slow_aspect_norm)
        l1 = nn.functional.l1_loss(pred_loss, y_loss)
        metrics["l1"] = float(l1.detach().item())
        iou_loss = pred.new_zeros(())
        if d_out == 4:
            iou = _mean_iou_tlwh(pred_loss, y)
            iou_loss = 1.0 - iou
            metrics["iou"] = float(iou.detach().item())
        elif d_out == 3 and runtime_slow_aspect_norm is not None:
            pred_tlwh = _center_h_to_runtime_tlwh(pred_loss, runtime_slow_aspect_norm)
            y_tlwh = _center_h_to_runtime_tlwh(y_loss, runtime_slow_aspect_norm)
            iou = _mean_iou_tlwh(pred_tlwh, y_tlwh)
            iou_loss = 1.0 - iou
            metrics["iou"] = float(iou.detach().item())
        vel = (
            nn.functional.l1_loss(_diff1(pred_loss), _diff1(y_loss))
            if pred.size(1) > 1
            else pred.new_zeros(())
        )
        acc = (
            nn.functional.l1_loss(_diff2(pred_loss), _diff2(y_loss))
            if pred.size(1) > 2
            else pred.new_zeros(())
        )
        metrics["vel"] = float(vel.detach().item()) if pred.size(1) > 1 else 0.0
        metrics["acc"] = float(acc.detach().item()) if pred.size(1) > 2 else 0.0

    loss = float(w_l1) * l1 + float(w_iou) * iou_loss + float(w_vel) * vel + float(w_acc) * acc
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


def _split_and_strip(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _load_game_ids(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    for item in args.game_id or []:
        out.extend(_split_and_strip(item))
    out.extend(_split_and_strip(args.game_ids))
    if args.game_ids_file:
        p = Path(args.game_ids_file)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    # de-dupe, keep order
    seen = set()
    dedup = []
    for gid in out:
        if gid in seen:
            continue
        seen.add(gid)
        dedup.append(gid)
    return dedup


def _arg_in_argv(flag: str, argv: list[str]) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _load_game_dirs_from_list(list_path: str) -> List[Path]:
    p = Path(list_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"file list does not exist: {p}")
    root = p.parent
    out: List[Path] = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        d = Path(line).expanduser()
        if not d.is_absolute():
            d = root / d
        out.append(d)
    # de-dupe, keep order
    seen: set[str] = set()
    dedup: List[Path] = []
    for d in out:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(d)
    return dedup


def _scan_games_max_xy(game_csvs: List[GameCsvPaths]) -> Tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for p in game_csvs:
        mx, my = (
            scan_database_max_xy(p)
            if p.database_path
            else scan_game_max_xy(p.tracking_csv, p.camera_csv, p.camera_fast_csv)
        )
        max_x = max(max_x, float(mx))
        max_y = max(max_y, float(my))
    return max_x, max_y


def _validate_validation_scale(val_games: List[GameCsvPaths], norm: CameraNorm) -> None:
    """Reject validation splits that would be clipped by train-only normalization."""
    if not val_games:
        return
    val_max_x, val_max_y = _scan_games_max_xy(val_games)
    tol_x = max(1.0, float(norm.scale_x)) * 1e-6
    tol_y = max(1.0, float(norm.scale_y)) * 1e-6
    over_x = float(val_max_x) > float(norm.scale_x) + tol_x
    over_y = float(val_max_y) > float(norm.scale_y) + tol_y
    if over_x or over_y:
        raise SystemExit(
            "Validation extents exceed train-only normalization scale: "
            f"val_max=({val_max_x:.3f}, {val_max_y:.3f}) "
            f"train_scale=({float(norm.scale_x):.3f}, {float(norm.scale_y):.3f}). "
            "This would clip validation labels/features and invalidate IoU/target-iou. "
            "Use a split whose training games cover the validation resolution/extents."
        )


def _parse_checkpoint_naming(out: str) -> Tuple[Path, str, str]:
    """Return (best_path, prefix, ext)."""
    best_path = Path(out)
    ext = "".join(best_path.suffixes) or ".pt"
    name = best_path.name
    stem = name[: -len(ext)] if name.endswith(ext) else best_path.stem
    prefix = stem
    if prefix.endswith("_best"):
        prefix = prefix[: -len("_best")]
    if not prefix:
        prefix = "camera_gpt"
    if not stem.endswith("_best"):
        best_path = best_path.with_name(f"{prefix}_best{ext}")
    return best_path, prefix, ext


def _checkpoint_path(best_path: Path, prefix: str, ext: str, step_num: int) -> Path:
    return best_path.with_name(f"{prefix}_{int(step_num)}{ext}")


def _list_numbered_checkpoints(best_path: Path, prefix: str, ext: str) -> List[tuple[int, Path]]:
    out: List[tuple[int, Path]] = []
    pat = f"{prefix}_"
    for p in best_path.parent.glob(f"{prefix}_*{ext}"):
        if p.name == best_path.name:
            continue
        name = p.name
        if not name.startswith(pat) or not name.endswith(ext):
            continue
        mid = name[len(pat) : -len(ext)]
        if not mid.isdigit():
            continue
        out.append((int(mid), p))
    out.sort(key=lambda t: t[0])
    return out


def _checkpoint_compatible(ckpt: dict, cfg: CameraGPTConfig, norm: CameraNorm) -> bool:
    try:
        m = ckpt.get("model") or {}
        n = ckpt.get("norm") or {}
        return (
            int(m.get("d_in")) == int(cfg.d_in)
            and int(m.get("d_out")) == int(cfg.d_out)
            and str(m.get("model_kind", "gpt")) == str(cfg.model_kind)
            and str(m.get("feature_mode")) == str(cfg.feature_mode)
            and bool(m.get("include_pose")) == bool(cfg.include_pose)
            and bool(m.get("include_rink", False)) == bool(cfg.include_rink)
            and (
                not cfg.include_rink
                or (
                    m.get("rink_input", "stats") == cfg.rink_input
                    and m.get("rink_grid_height", 32) == cfg.rink_grid_height
                    and m.get("rink_grid_width", 64) == cfg.rink_grid_width
                    and m.get("rink_encoder_version", 1) == cfg.rink_encoder_version
                )
            )
            and bool(m.get("residual_prev_y", False)) == bool(cfg.residual_prev_y)
            and float(m.get("residual_scale", 0.1)) == float(cfg.residual_scale)
            and int(m.get("d_model")) == int(cfg.d_model)
            and int(m.get("nhead")) == int(cfg.nhead)
            and int(m.get("nlayers")) == int(cfg.nlayers)
            and int(m.get("dim_feedforward", CameraGPTConfig().dim_feedforward))
            == int(cfg.dim_feedforward)
            and float(m.get("dropout", CameraGPTConfig().dropout)) == float(cfg.dropout)
            and math.isclose(
                float(n.get("scale_x", -1.0)),
                float(norm.scale_x),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(n.get("scale_y", -1.0)),
                float(norm.scale_y),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and int(n.get("max_players", -1)) == int(norm.max_players)
        )
    except Exception:
        return False


def _maybe_resume(
    *,
    best_path: Path,
    prefix: str,
    ext: str,
    resume_mode: str,
    device: torch.device,
    model: nn.Module,
    opt: optim.Optimizer,
    cfg: CameraGPTConfig,
    norm: CameraNorm,
    training_state: Optional[dict] = None,
) -> int:
    if resume_mode == "none":
        return 1

    numbered = _list_numbered_checkpoints(best_path, prefix, ext)
    candidates = [p for _, p in numbered]
    if best_path.is_file():
        candidates.append(best_path)
    checkpoints = [
        (p, torch.load(str(p), map_location="cpu", weights_only=False)) for p in candidates
    ]
    latest, ckpt = (
        max(checkpoints, key=lambda pair: int(pair[1].get("step", 0)))
        if checkpoints
        else (None, None)
    )

    if latest is None:
        if resume_mode == "force":
            raise SystemExit(f"--resume specified but no checkpoint found for prefix={prefix}")
        return 1

    assert ckpt is not None
    if not _checkpoint_compatible(ckpt, cfg, norm):
        msg = f"Checkpoint incompatible with current model cfg: {latest}"
        if resume_mode == "force":
            raise SystemExit(msg)
        logger.warning("%s (skipping resume)", msg)
        return 1

    if training_state is not None:
        saved_state = ckpt.get("training_state", {})
        if saved_state.get("data_identity") != training_state.get("data_identity"):
            raise ValueError(
                f"Checkpoint dataset/split differs from this run: {latest}; use --no-resume with a new output"
            )
        if saved_state.get("evaluation") != training_state.get("evaluation"):
            raise ValueError(f"Checkpoint evaluation settings differ from this run: {latest}")
        training_state.update(saved_state)

    model.load_state_dict(ckpt["state_dict"])
    opt_sd = ckpt.get("optimizer")
    if opt_sd:
        opt.load_state_dict(opt_sd)
    model.to(device)
    step0 = int(ckpt.get("step", 0) or 0)
    logger.info("Resumed from %s (step=%d)", latest, step0)
    return max(1, step0 + 1)


def _save_training_checkpoint(
    *,
    path: Path,
    step_num: int,
    model: nn.Module,
    opt: optim.Optimizer,
    train_ds: CameraPanZoomGPTIterableDataset,
    context_window: int,
    cfg: CameraGPTConfig,
    game_csvs: List[GameCsvPaths],
    target_mode: str,
    include_pose: bool,
    source_init_report: Optional[dict] = None,
    training_state: Optional[dict] = None,
) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    ckpt = pack_gpt_checkpoint(model, norm=train_ds.norm, window=context_window, cfg=cfg)
    ckpt["step"] = int(step_num)
    ckpt["games"] = [p.game_id for p in game_csvs]
    ckpt["target_mode"] = str(target_mode)
    ckpt["include_pose"] = bool(include_pose)
    if source_init_report is not None:
        ckpt["source_init_report"] = dict(source_init_report)
    ckpt["optimizer"] = opt.state_dict()
    ckpt["training_state"] = dict(training_state or {})
    os.makedirs(str(path.parent), exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(ckpt, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _predict_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    free_run: bool,
    runtime_slow_aspect_norm: Optional[float],
    context_window: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = (
        {"rink_embedding": model.encode_rink(batch["rink"].to(device))} if "rink" in batch else {}
    )

    def predict(features):
        return model(features, **kwargs)

    y = batch["y"].to(device)
    if "x" in batch:
        x_full = batch["x"].to(device)
        if not free_run:
            return predict(x_full), y
        if int(x_full.shape[-1]) < 11:
            raise ValueError(
                "legacy_prev_slow free-run expects previous [cx,cy,h] at feature indices 8:11"
            )
        prev = x_full[:, 0, 8:11]
        preds = []
        prefix = []
        for t in range(int(x_full.shape[1])):
            x_t = x_full[:, t, :].clone()
            x_t[:, 8:11] = prev
            prefix.append(x_t)
            x = torch.stack(prefix, dim=1)
            pred_t = predict(x[:, -context_window:] if context_window else x)[:, -1, :]
            preds.append(pred_t)
            prev = _legacy_prev_slow_feedback_target(pred_t, runtime_slow_aspect_norm).detach()
        return torch.stack(preds, dim=1), y

    base = batch["base"].to(device)
    prev0 = batch["prev0"].to(device)
    if not free_run:
        prev_y = torch.cat([prev0.unsqueeze(1), y[:, :-1, :]], dim=1)
        x = torch.cat([base, prev_y], dim=-1)
        return predict(x), y

    prev = prev0
    preds = []
    prefix = []
    for t in range(int(base.shape[1])):
        x_t = torch.cat([base[:, t, :], prev], dim=-1)
        prefix.append(x_t)
        x = torch.stack(prefix, dim=1)
        pred_t = predict(x[:, -context_window:] if context_window else x)[:, -1, :]
        preds.append(pred_t)
        prev = _runtime_feedback_target(pred_t, runtime_slow_aspect_norm).detach()
    return torch.stack(preds, dim=1), y


@torch.no_grad()
def _eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    *,
    w_l1: float,
    w_iou: float,
    w_vel: float,
    w_acc: float,
    fast_mult: float,
    free_run: bool,
    runtime_slow_aspect_norm: Optional[float],
    context_window: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    metric_sums: dict[str, float] = {}
    it = iter(loader)
    for _ in range(int(steps)):
        batch = next(it)
        pred, y = _predict_batch(
            model,
            batch,
            device,
            free_run=free_run,
            runtime_slow_aspect_norm=runtime_slow_aspect_norm,
            context_window=context_window,
        )
        loss, metrics = _compute_losses(
            pred,
            y,
            w_l1=w_l1,
            w_iou=w_iou,
            w_vel=w_vel,
            w_acc=w_acc,
            fast_mult=fast_mult,
            runtime_slow_aspect_norm=runtime_slow_aspect_norm,
        )
        batch_size = int(y.size(0))
        loss_sum += float(loss.item()) * batch_size
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size
        total += batch_size
    if dist.is_initialized():
        keys = sorted(metric_sums)
        sums = torch.tensor(
            [total, loss_sum] + [metric_sums[k] for k in keys], dtype=torch.float64, device=device
        )
        dist.all_reduce(sums)
        total, loss_sum = float(sums[0]), float(sums[1])
        metric_sums = {key: float(sums[i + 2]) for i, key in enumerate(keys)}
    out = {key: value / max(1, total) for key, value in metric_sums.items()}
    out["loss"] = loss_sum / max(1, total)
    return out


def _target_met(metrics: dict, target_iou: float) -> bool:
    if target_iou <= 0:
        return False
    if "iou_slow" in metrics and "iou_fast" in metrics:
        return min(metrics["iou_slow"], metrics["iou_fast"]) >= target_iou
    return metrics.get("iou", -1.0) >= target_iou


class TrainingRollout(nn.Module):
    """Train a bounded-context rollout inside a single DDP forward call."""

    def __init__(self, model: nn.Module, aspect: Optional[float], context_window: int) -> None:
        super().__init__()
        if context_window < 2:
            raise ValueError("Training context window must be >=2")
        self.model = model
        self.aspect = aspect
        self.context_window = context_window

    def _teacher_forced(
        self, features: torch.Tensor, kwargs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if features.shape[1] <= self.context_window:
            return self.model(features, **kwargs)
        return torch.stack(
            [
                self.model(features[:, max(0, t + 1 - self.context_window) : t + 1], **kwargs)[
                    :, -1
                ]
                for t in range(features.shape[1])
            ],
            dim=1,
        )

    def forward(self, batch: dict[str, torch.Tensor], probability: float) -> torch.Tensor:
        # The trainable encoder stays inside this single DDP forward. Its graph
        # is reused across prefixes, then recomputed after the next optimizer step.
        kwargs = (
            {"rink_embedding": self.model.encode_rink(batch["rink"])} if "rink" in batch else {}
        )
        if "x" in batch:
            return self._teacher_forced(batch["x"], kwargs)
        base, prev, y = batch["base"], batch["prev0"], batch["y"]
        if probability <= 0:
            previous = torch.cat([prev.unsqueeze(1), y[:, :-1]], dim=1)
            return self._teacher_forced(torch.cat([base, previous], dim=-1), kwargs)
        prefix, predictions = [], []
        for t in range(y.shape[1]):
            prefix.append(torch.cat([base[:, t], prev], dim=-1))
            prefix = prefix[-self.context_window :]
            prediction = self.model(torch.stack(prefix, dim=1), **kwargs)[:, -1]
            predictions.append(prediction)
            feedback = _runtime_feedback_target(prediction, self.aspect).detach()
            use_prediction = torch.rand((len(y), 1), device=y.device) < probability
            prev = torch.where(use_prediction, feedback, y[:, t])
        return torch.stack(predictions, dim=1)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser("Train GPT camera model from saved tracking/camera CSVs")
    ap.add_argument(
        "--config", type=str, help="Training YAML; explicit CLI arguments override YAML"
    )
    ap.add_argument(
        "--database",
        action="append",
        default=[],
        help="Telemetry database, directory, or glob (repeatable; merged and single-run files may be mixed)",
    )
    ap.add_argument("--dataset-config", type=str, help="Dataset catalog/selection YAML")
    ap.add_argument("--dataset-root", type=str, help="Override the dataset YAML root on this host")
    ap.add_argument(
        "--sample-stride", type=int, default=1, help="Stride between eligible window starts"
    )
    ap.add_argument("--run-sampling", choices=["uniform", "windows"], default="windows")
    ap.add_argument(
        "--val-seq-len",
        type=int,
        default=None,
        help="Validation rollout length; defaults to --seq-len",
    )
    ap.add_argument(
        "--rollout-len",
        type=int,
        default=None,
        help="Training rollout length (>=seq-len); defaults to --seq-len runtime context",
    )
    ap.add_argument("--rink-input", choices=["stats", "grid"], default="stats")
    ap.add_argument("--rink-grid-height", type=int, default=32)
    ap.add_argument("--rink-grid-width", type=int, default=64)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--ddp-backend", choices=["nccl", "gloo"], default=None)
    ap.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Append training/validation JSONL on rank zero",
    )
    ap.add_argument("--game-id", action="append", default=[], help="Game id (repeatable)")
    ap.add_argument("--game-ids", type=str, default=None, help="Comma-separated game ids")
    ap.add_argument("--game-ids-file", type=str, default=None, help="Text file with game ids")
    ap.add_argument(
        "--file-list",
        type=str,
        default=None,
        help=(
            "Path to a .lst text file containing game directories. Each line should be either an "
            "absolute path or a path relative to the .lst file's directory. Each game directory "
            "must contain tracking.csv and camera.csv (pose.csv optional). When set, this overrides "
            "--game-id/--game-ids/--game-ids-file."
        ),
    )
    ap.add_argument(
        "--videos-root",
        type=str,
        default=str(Path(os.environ.get("HOME", "")) / "Videos"),
        help="Root directory containing <game-id>/ directories (default: $HOME/Videos)",
    )
    ap.add_argument("--seq-len", type=int, default=32, help="Sequence length in frames")
    ap.add_argument(
        "--sample-seconds",
        type=float,
        default=3.0,
        help="Window length to sample (seconds). Used when --file-list is set and --seq-len isn't provided.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used to convert --sample-seconds into frames (used only for --file-list mode).",
    )
    ap.add_argument(
        "--target-mode",
        type=str,
        default="slow_fast_tlwh",
        choices=["slow_center_h", "slow_tlwh", "slow_fast_tlwh"],
        help="Training target format (slow camera only or slow+fast boxes).",
    )
    ap.add_argument(
        "--feature-mode",
        type=str,
        default="base_prev_y",
        choices=["legacy_prev_slow", "base_prev_y", "players_prev_y"],
        help=(
            "Input feature schema. 'base_prev_y' feeds previous camera target/prediction as input; "
            "'players_prev_y' also includes sorted padded player boxes."
        ),
    )
    ap.add_argument(
        "--pose",
        "--include-pose",
        dest="include_pose",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable pose.csv feature aggregation. Defaults off for DriveGPT to keep default "
            "checkpoints runnable in the non-pose tracking graph; use --pose for pose-enriched runs."
        ),
    )
    ap.add_argument(
        "--include-rink",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Include rink features derived from rink_mask_0.png / rink_profile (requires retraining). "
            "Use --no-include-rink to disable."
        ),
    )
    ap.add_argument(
        "--scheduled-sampling",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable scheduled sampling (only with previous-target feature modes). "
            "Defaults on for DriveGPT previous-target modes."
        ),
    )
    ap.add_argument(
        "--ss-prob-start", type=float, default=0.0, help="Scheduled-sampling prob at step 0."
    )
    ap.add_argument(
        "--ss-prob-end", type=float, default=0.3, help="Scheduled-sampling prob after warmup."
    )
    ap.add_argument(
        "--ss-warmup-steps",
        type=int,
        default=1000,
        help="Linearly ramp scheduled-sampling prob over this many steps (0=immediate).",
    )
    ap.add_argument("--steps", type=int, default=2000, help="Training optimizer steps")
    ap.add_argument(
        "--max-iters",
        dest="max_iters",
        type=int,
        default=None,
        help="Alias for --steps (max training iterations).",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Optional training budget in frames (overrides --steps when >0)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Per-rank validation batch; defaults to --batch-size",
    )
    ap.add_argument(
        "--data-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 runs data pipeline in the main process).",
    )
    ap.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory (useful for GPU training).",
    )
    ap.add_argument(
        "--preload-csv",
        type=str,
        default="none",
        choices=["none", "shard", "all"],
        help=(
            "Preload CSVs into each dataset worker process. "
            "'shard' loads only the worker's slice of games; 'all' loads all games in every worker."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--model-kind",
        type=str,
        default="gpt",
        choices=["gpt", "drivegpt"],
        help=(
            "Model family to train. 'drivegpt' initializes compatible causal blocks from "
            "OpenDriveLab UniAD planning weights before fine-tuning on camera trajectories."
        ),
    )
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=8)
    ap.add_argument(
        "--dim-feedforward",
        type=int,
        default=None,
        help=(
            "Transformer feed-forward dimension. Defaults to the legacy CameraGPT setting, "
            "or 512 for --model-kind=drivegpt to match OpenDriveLab UniAD planning weights."
        ),
    )
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--drivegpt-init",
        type=str,
        default="auto",
        choices=["auto", "require", "none"],
        help=(
            "OpenDriveLab initialization policy for --model-kind=drivegpt. 'auto' warns and "
            "continues if unavailable; 'require' fails; 'none' trains from scratch."
        ),
    )
    ap.add_argument(
        "--drivegpt-source-model",
        type=str,
        default=OPENDRIVE_UNIAD_MODEL_ID,
        help="Hugging Face model id for OpenDriveLab source weights.",
    )
    ap.add_argument(
        "--drivegpt-source-file",
        type=str,
        default=OPENDRIVE_UNIAD_PLANNING_FILE,
        help="Checkpoint filename inside --drivegpt-source-model.",
    )
    ap.add_argument(
        "--drivegpt-source-checkpoint",
        type=str,
        default=None,
        help="Local OpenDriveLab checkpoint path; skips Hugging Face download when set.",
    )
    ap.add_argument(
        "--residual-prev-y",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Predict bounded deltas from the previous camera target instead of absolute boxes. "
            "Defaults on for --model-kind=drivegpt previous-target feature modes."
        ),
    )
    ap.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="Maximum normalized residual delta per output dimension when --residual-prev-y is enabled.",
    )
    ap.add_argument("--loss-l1-weight", type=float, default=1.0, help="Weight for L1 loss.")
    ap.add_argument(
        "--loss-iou-weight",
        type=float,
        default=0.5,
        help="Weight for (1-IoU) box loss (TLWH modes only).",
    )
    ap.add_argument(
        "--loss-vel-weight",
        type=float,
        default=0.2,
        help="Weight for velocity matching loss (L1 on first differences).",
    )
    ap.add_argument(
        "--loss-acc-weight",
        type=float,
        default=0.1,
        help="Weight for acceleration matching loss (L1 on second differences).",
    )
    ap.add_argument(
        "--fast-loss-mult",
        type=float,
        default=2.0,
        help="Multiplier for fast-box losses when training slow_fast_tlwh.",
    )
    ap.add_argument("--max-players", type=int, default=22)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--val-steps", type=int, default=50)
    ap.add_argument(
        "--target-iou",
        type=float,
        default=0.0,
        help=(
            "Early-stop target validation IoU. For slow_fast_tlwh, both slow and fast "
            "mean IoU must meet this threshold. 0 disables early stopping."
        ),
    )
    ap.add_argument(
        "--eval-free-run",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Run validation autoregressively by feeding model predictions back as previous camera "
            "targets. Slower, but matches runtime behavior for previous-target feature modes."
        ),
    )
    ap.add_argument(
        "--runtime-slow-iou",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Compute slow-box IoU/feedback after applying Aspen runtime aspect-ratio "
            "postprocessing. Defaults on for DriveGPT camera targets."
        ),
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help=(
            "Save a training checkpoint every N steps (0=disabled). "
            "Checkpoints are saved alongside --out as <prefix>_<step>.pt (best is <prefix>_best.pt)."
        ),
    )
    ap.add_argument(
        "--max-checkpoints",
        type=int,
        default=10,
        help="Keep only the last N numbered checkpoints (0=keep all). Best checkpoint is kept.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if found (or error if none found).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume even if checkpoints are present.",
    )
    ap.add_argument(
        "--camera-csv-name",
        type=str,
        default=None,
        help="Optional fixed camera CSV filename inside each <videos-root>/<game-id>/ (e.g. camera_annotated.csv).",
    )
    ap.add_argument(
        "--camera-fast-csv-name",
        type=str,
        default=None,
        help="Optional fixed fast camera CSV filename inside each <videos-root>/<game-id>/ (e.g. camera_fast_annotated.csv).",
    )
    ap.add_argument(
        "--pose-csv-name",
        type=str,
        default=None,
        help="Optional fixed pose CSV filename inside each <videos-root>/<game-id>/ (e.g. pose.csv).",
    )
    ap.add_argument("--out", type=str, default="camera_gpt_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-cached-games", type=int, default=8)
    ap.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Optional cap on number of games loaded from --file-list or ids (0=no cap).",
    )
    argv = expand_training_config(ap, list(sys.argv[1:] if argv is None else argv))
    args = ap.parse_args(argv)
    if min(args.rink_grid_height, args.rink_grid_width) < 2:
        ap.error("Rink grid dimensions must be >=2")
    # Resolve seconds-based legacy sampling before deriving any horizon defaults.
    if args.file_list and not _arg_in_argv("--seq-len", argv):
        args.seq_len = int(round(max(0.1, float(args.sample_seconds)) * max(1.0, float(args.fps))))
    if args.rollout_len is None:
        args.rollout_len = args.seq_len
    if args.rollout_len < args.seq_len:
        ap.error("rollout-len must be >=seq-len")
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.val_batch_size < 1:
        ap.error("val-batch-size must be positive")
    if args.seq_len < 2 or args.batch_size < 1 or args.sample_stride < 1 or args.cpu_threads < 1:
        ap.error("seq-len must be >=2; batch-size, sample-stride, cpu-threads must be positive")
    if args.val_seq_len is None:
        args.val_seq_len = args.seq_len
    if args.val_seq_len < 2:
        ap.error("val-seq-len must be >=2")
    if not all(0 <= p <= 1 for p in (args.ss_prob_start, args.ss_prob_end, args.target_iou)):
        ap.error("Scheduled sampling probabilities and target-iou must be in [0, 1]")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        if world_size > 1 and device.index not in (None, local_rank):
            ap.error("DDP device must match LOCAL_RANK")
        device = torch.device("cuda", local_rank if device.index is None else device.index)
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    if world_size > 1:
        dist.init_process_group(
            args.ddp_backend or ("nccl" if device.type == "cuda" else "gloo"),
            timeout=timedelta(minutes=30),
        )
    if rank != 0:
        logger.setLevel("WARNING")

    drivegpt_prev_mode = str(args.model_kind) == "drivegpt" and str(args.feature_mode) in {
        "base_prev_y",
        "players_prev_y",
    }
    if args.include_pose is None:
        args.include_pose = str(args.model_kind) != "drivegpt"
    if args.scheduled_sampling is None:
        args.scheduled_sampling = drivegpt_prev_mode

    if str(args.model_kind) == "drivegpt":
        if not _arg_in_argv("--d-model", argv):
            args.d_model = 256
        if not _arg_in_argv("--nhead", argv):
            args.nhead = 8
        if not _arg_in_argv("--nlayers", argv):
            args.nlayers = 3
        if args.dim_feedforward is None:
            args.dim_feedforward = 512
        if not _arg_in_argv("--out", argv):
            args.out = "drivegpt_best.pt"
    elif args.dim_feedforward is None:
        args.dim_feedforward = int(CameraGPTConfig().dim_feedforward)
    if args.residual_prev_y is None:
        args.residual_prev_y = drivegpt_prev_mode
    elif bool(args.residual_prev_y) and str(args.feature_mode) not in {
        "base_prev_y",
        "players_prev_y",
    }:
        raise SystemExit(
            "--residual-prev-y requires --feature-mode=base_prev_y or "
            "--feature-mode=players_prev_y"
        )
    if args.eval_free_run is None:
        args.eval_free_run = drivegpt_prev_mode
    if args.runtime_slow_iou is None:
        args.runtime_slow_iou = str(args.model_kind) == "drivegpt" and str(args.target_mode) in {
            "slow_center_h",
            "slow_tlwh",
            "slow_fast_tlwh",
        }
    if (
        float(args.target_iou) > 0.0
        and str(args.target_mode) == "slow_center_h"
        and not bool(args.runtime_slow_iou)
    ):
        raise SystemExit(
            "--target-iou with --target-mode=slow_center_h requires --runtime-slow-iou"
        )

    # --max-iters is an alias for --steps. If both are provided explicitly, require they match.
    if args.max_iters is not None:
        steps_in_argv = _arg_in_argv("--steps", argv)
        max_iters_in_argv = _arg_in_argv("--max-iters", argv)
        if steps_in_argv and max_iters_in_argv and int(args.steps) != int(args.max_iters):
            raise SystemExit("--steps and --max-iters both provided but differ; please use one.")
        if (not steps_in_argv) and max_iters_in_argv:
            args.steps = int(args.max_iters)

    resume_mode = "auto"
    if bool(args.no_resume):
        resume_mode = "none"
    elif bool(args.resume):
        resume_mode = "force"

    game_csvs: List[GameCsvPaths] = []
    data_identity = None

    if args.dataset_config:
        if args.database or args.file_list or args.game_id or args.game_ids or args.game_ids_file:
            ap.error("--dataset-config cannot be combined with legacy game/file-list selection")
        catalog_error = None
        try:
            train_games, val_games, data_identity = catalog_split(
                args.dataset_config,
                args.dataset_root,
                min_train_frames=args.rollout_len,
                min_val_frames=args.val_seq_len,
                require_rink_grid=args.include_rink and args.rink_input == "grid",
            )
        except Exception as error:
            catalog_error = f"rank {rank}: {type(error).__name__}: {error}"
        errors = [catalog_error]
        if dist.is_initialized():
            errors = [None] * world_size
            dist.all_gather_object(errors, catalog_error)
        if any(errors):
            raise RuntimeError(
                "Dataset validation failed: " + "; ".join(error for error in errors if error)
            )
        game_csvs = train_games + val_games
        if args.include_pose:
            ap.error("The official catalog omits pose data; use --no-pose")
        if args.max_games:
            ap.error("Use dataset YAML include/exclude selectors instead of --max-games")
    elif args.database:
        if args.file_list or args.game_id or args.game_ids or args.game_ids_file:
            ap.error("--database cannot be combined with game/file-list selection")
        if args.include_pose:
            ap.error("Database recordings do not yet contain pose features; use --no-pose")
        game_csvs, data_identity = discover_database_games(args.database)
        train_games, val_games = split_database_games(
            game_csvs, float(args.val_split), int(args.seed)
        )
        data_identity["train_games"] = [game.game_id for game in train_games]
        data_identity["validation_games"] = [game.game_id for game in val_games]
    elif args.file_list:
        game_dirs = _load_game_dirs_from_list(args.file_list)
        if int(args.max_games) > 0:
            game_dirs = game_dirs[: int(args.max_games)]
        if not game_dirs:
            raise SystemExit(f"No game directories found in {args.file_list}")

        for game_dir in game_dirs:
            if not game_dir.is_dir():
                logger.warning("Skipping missing game dir: %s", game_dir)
                continue
            databases = sorted(game_dir.glob("hstream_telemetry*.db"))
            if databases and not (
                args.camera_csv_name or args.camera_fast_csv_name or args.pose_csv_name
            ):
                selected, _ = discover_database_games(databases)
                game_csvs.extend(selected)
                continue
            paths = resolve_csv_paths(
                game_id=game_dir.name,
                game_dir=str(game_dir),
                camera_csv_name=args.camera_csv_name,
                camera_fast_csv_name=args.camera_fast_csv_name,
                pose_csv_name=args.pose_csv_name,
            )
            if paths is None:
                logger.warning("Skipping %s (missing tracking.csv/camera.csv)", game_dir)
                continue
            if args.target_mode == "slow_fast_tlwh" and not paths.camera_fast_csv:
                logger.warning(
                    "Skipping %s (missing camera_fast.csv for --target-mode=slow_fast_tlwh)",
                    game_dir,
                )
                continue
            try:
                validate_csv_paths(paths)
            except Exception as ex:
                logger.warning("Skipping %s (bad CSV paths): %s", game_dir, ex)
                continue
            game_csvs.append(paths)
    else:
        game_ids = _load_game_ids(args)
        if not game_ids:
            raise SystemExit("No game ids provided (use --game-id/--game-ids/--game-ids-file)")

        videos_root = Path(args.videos_root)
        if not videos_root.is_dir():
            raise SystemExit(f"videos_root does not exist: {videos_root}")

        if int(args.max_games) > 0:
            game_ids = game_ids[: int(args.max_games)]

        for gid in game_ids:
            gdir = videos_root / gid
            if not gdir.is_dir():
                logger.warning("Skipping %s (missing dir %s)", gid, gdir)
                continue
            databases = sorted(gdir.glob("hstream_telemetry*.db"))
            if databases and not (
                args.camera_csv_name or args.camera_fast_csv_name or args.pose_csv_name
            ):
                selected, _ = discover_database_games(databases)
                game_csvs.extend(selected)
                continue
            paths = resolve_csv_paths(
                game_id=gid,
                game_dir=str(gdir),
                camera_csv_name=args.camera_csv_name,
                camera_fast_csv_name=args.camera_fast_csv_name,
                pose_csv_name=args.pose_csv_name,
            )
            if paths is None:
                logger.warning("Skipping %s (missing tracking.csv/camera.csv)", gid)
                continue
            if args.target_mode == "slow_fast_tlwh" and not paths.camera_fast_csv:
                logger.warning(
                    "Skipping %s (missing camera_fast.csv for --target-mode=slow_fast_tlwh)", gid
                )
                continue
            try:
                validate_csv_paths(paths)
            except Exception as ex:
                logger.warning("Skipping %s (bad CSV paths): %s", gid, ex)
                continue
            game_csvs.append(paths)

    if not game_csvs:
        raise SystemExit(
            "No usable games found (need tracking.csv + camera.csv [+ camera_fast.csv])."
        )

    if data_identity is None and any(game.database_path for game in game_csvs):
        if not all(game.database_path for game in game_csvs):
            ap.error(
                "Mixed legacy CSV and database game selection is ambiguous; select databases explicitly"
            )
        if args.include_pose:
            ap.error("Database recordings do not yet contain pose features; use --no-pose")
        game_csvs, data_identity = discover_database_games(
            sorted({game.database_path for game in game_csvs})
        )
        train_games, val_games = split_database_games(
            game_csvs, float(args.val_split), int(args.seed)
        )
        data_identity["train_games"] = [game.game_id for game in train_games]
        data_identity["validation_games"] = [game.game_id for game in val_games]
    if data_identity is None:
        rng = random.Random(int(args.seed))
        rng.shuffle(game_csvs)
        n_val = int(round(len(game_csvs) * float(args.val_split)))
        val_games = game_csvs[:n_val] if n_val > 0 else []
        train_games = game_csvs[n_val:] if n_val > 0 else game_csvs
        if not train_games:
            train_games = game_csvs
            val_games = []
        data_identity = {
            "train_games": [p.game_id for p in train_games],
            "validation_games": [p.game_id for p in val_games],
        }
    geometry_error = None
    try:
        # Use a train-only normalization scale for train/val consistency without validation leakage.
        max_x, max_y = _scan_games_max_xy(train_games)
        if args.include_rink and args.rink_input == "grid":
            # Frame geometry, rather than observed player extrema, defines the world.
            contexts = {
                game.game_id: (
                    {"frame_size": scan_database_max_xy(game)}
                    if game.database_path
                    else read_rink_context(game.tracking_csv)
                )
                for game in game_csvs
            }
            max_x = max(max_x, *(contexts[game.game_id]["frame_size"][0] for game in train_games))
            max_y = max(max_y, *(contexts[game.game_id]["frame_size"][1] for game in train_games))
            for game in val_games:
                w, h = contexts[game.game_id]["frame_size"]
                if w > max_x or h > max_y:
                    raise ValueError(
                        f"Validation rink exceeds training coordinate extent: {game.game_id}"
                    )
            data_identity["rink_contexts"] = {
                game.game_id: (
                    database_geometry(game)["mask_sha256"]
                    if game.database_path
                    else file_sha256(rink_context_path(game.tracking_csv))
                )
                for game in game_csvs
            }
        norm = CameraNorm(scale_x=max_x, scale_y=max_y, max_players=int(args.max_players))
        _validate_validation_scale(val_games, norm)

        rink_grids = (
            {
                game.game_id: (
                    load_database_rink(game, norm, args.rink_grid_height, args.rink_grid_width)
                    if game.database_path
                    else load_rink_grid(
                        game.tracking_csv, norm, args.rink_grid_height, args.rink_grid_width
                    )
                )
                for game in game_csvs
            }
            if args.include_rink and args.rink_input == "grid"
            else {}
        )
    except (Exception, SystemExit) as error:
        geometry_error = f"rank {rank}: {type(error).__name__}: {error}"
    errors = [geometry_error]
    if dist.is_initialized():
        errors = [None] * world_size
        dist.all_gather_object(errors, geometry_error)
    if any(errors):
        raise RuntimeError(
            "Dataset geometry preflight failed: " + "; ".join(error for error in errors if error)
        )

    train_ds = CameraPanZoomGPTIterableDataset(
        games=train_games,
        norm=norm,
        seq_len=int(args.rollout_len),
        target_mode=str(args.target_mode),
        feature_mode=str(args.feature_mode),
        include_pose=bool(args.include_pose),
        include_rink=bool(args.include_rink),
        rink_input=args.rink_input,
        rink_grid_height=args.rink_grid_height,
        rink_grid_width=args.rink_grid_width,
        rink_grids=rink_grids,
        max_players_for_norm=int(args.max_players),
        seed=int(args.seed),
        max_cached_games=int(args.max_cached_games),
        preload_csv=str(args.preload_csv),
        shard_games_by_worker=(int(args.data_workers) > 1),
        rank=rank,
        world_size=world_size,
        sample_stride=args.sample_stride,
        run_sampling=args.run_sampling,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.data_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=int(args.data_workers) > 0,
        multiprocessing_context="spawn" if args.data_workers > 0 else None,
    )

    val_loader = None
    if val_games and int(args.val_steps) > 0:
        val_ds = CameraPanZoomGPTIterableDataset(
            games=val_games,
            norm=norm,
            seq_len=int(args.val_seq_len),
            target_mode=str(args.target_mode),
            feature_mode=str(args.feature_mode),
            include_pose=bool(args.include_pose),
            include_rink=bool(args.include_rink),
            rink_input=args.rink_input,
            rink_grid_height=args.rink_grid_height,
            rink_grid_width=args.rink_grid_width,
            rink_grids=rink_grids,
            max_players_for_norm=int(args.max_players),
            seed=int(args.seed) + 999,
            max_cached_games=max(1, int(args.max_cached_games // 2)),
            preload_csv=str(args.preload_csv),
            shard_games_by_worker=(int(args.data_workers) > 1),
            rank=rank,
            world_size=world_size,
            sample_stride=args.sample_stride,
            run_sampling=args.run_sampling,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.val_batch_size),
            num_workers=int(args.data_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=int(args.data_workers) > 0,
            multiprocessing_context="spawn" if args.data_workers > 0 else None,
        )
    if float(args.target_iou) > 0.0 and (
        val_loader is None or int(args.eval_every) <= 0 or int(args.val_steps) <= 0
    ):
        raise SystemExit(
            "--target-iou requires validation to be enabled: use --val-split > 0, "
            "--val-steps > 0, and --eval-every > 0."
        )

    if args.dataset_config and args.target_iou > 0 and not args.eval_free_run:
        ap.error("Official dataset target-iou requires autoregressive --eval-free-run")
    cfg = CameraGPTConfig(
        d_in=int(train_ds.feature_dim),
        d_out=int(train_ds.target_dim),
        model_kind=str(args.model_kind),
        feature_mode=str(args.feature_mode),
        include_pose=bool(args.include_pose),
        include_rink=bool(args.include_rink),
        rink_input=args.rink_input,
        rink_grid_height=args.rink_grid_height,
        rink_grid_width=args.rink_grid_width,
        source_model_id=(
            str(args.drivegpt_source_model) if str(args.model_kind) == "drivegpt" else ""
        ),
        source_checkpoint="",
        source_init="",
        residual_prev_y=bool(args.residual_prev_y),
        residual_scale=float(args.residual_scale),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        nlayers=int(args.nlayers),
        dim_feedforward=int(args.dim_feedforward),
        dropout=float(args.dropout),
    )

    source_init_report: Optional[dict] = None
    if str(args.model_kind) == "drivegpt" and str(args.drivegpt_init) != "none":
        try:
            source_checkpoint = resolve_opendrive_checkpoint(
                model_id=str(args.drivegpt_source_model),
                filename=str(args.drivegpt_source_file),
                checkpoint_path=args.drivegpt_source_checkpoint,
            )
            cfg = replace(
                cfg,
                source_model_id=str(args.drivegpt_source_model),
                source_checkpoint=str(source_checkpoint),
                source_init="opendrive-uniad-planning",
            )
        except Exception as ex:
            if str(args.drivegpt_init) == "require":
                raise
            logger.warning("OpenDriveLab DriveGPT initialization unavailable: %s", ex)

    model = CameraPanZoomGPT(cfg)
    if str(getattr(cfg, "source_init", "")) == "opendrive-uniad-planning":
        try:
            source_init_report = init_from_opendrive_uniad_planning(
                model, str(cfg.source_checkpoint)
            )
            logger.info(
                "Initialized DriveGPT from %s: copied=%d layers=%d",
                cfg.source_checkpoint,
                int(source_init_report["copied_tensors"]),
                int(source_init_report["initialized_layers"]),
            )
        except Exception as ex:
            if str(args.drivegpt_init) == "require":
                raise
            logger.warning("OpenDriveLab DriveGPT initialization failed: %s", ex)
            cfg = replace(cfg, source_init="")
            model.cfg = cfg

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %.2fM", float(n_params) / 1e6)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))
    runtime_slow_aspect_norm = (
        (16.0 / 9.0) * float(norm.scale_y) / max(1e-6, float(norm.scale_x))
        if bool(args.runtime_slow_iou)
        else None
    )

    steps = int(args.steps)
    if int(args.frames) > 0:
        steps = int(
            math.ceil(float(args.frames) / float(args.batch_size * args.rollout_len * world_size))
        )
    logger.info(
        "Training %s: games=%d train=%d val=%d context=%d rollout=%d steps=%d bs=%d val_bs=%d device=%s",
        str(args.model_kind),
        len(game_csvs),
        len(train_games),
        len(val_games),
        int(args.seq_len),
        int(args.rollout_len),
        steps,
        int(args.batch_size),
        int(args.val_batch_size),
        device,
    )

    training_state = {
        "data_identity": data_identity,
        "evaluation": {
            "seq_len": args.val_seq_len,
            "context_window": args.seq_len,
            "free_run": args.eval_free_run,
            "runtime_slow_iou": args.runtime_slow_iou,
            "val_steps_per_rank": args.val_steps,
            "batch_size_per_rank": args.val_batch_size,
            "world_size": world_size,
            "seed": args.seed,
            "sample_stride": args.sample_stride,
            "run_sampling": args.run_sampling,
            "data_workers": args.data_workers,
            "loss_weights": [
                args.loss_l1_weight,
                args.loss_iou_weight,
                args.loss_vel_weight,
                args.loss_acc_weight,
                args.fast_loss_mult,
            ],
        },
        "best_val": float("inf"),
        "target_met": False,
        "target_iou": args.target_iou,
        "resolved_args": vars(args),
    }
    best_path, prefix, ext = _parse_checkpoint_naming(args.out)
    start_step = _maybe_resume(
        best_path=best_path,
        prefix=prefix,
        ext=ext,
        resume_mode=resume_mode,
        device=device,
        model=model,
        opt=opt,
        cfg=cfg,
        norm=norm,
        training_state=training_state,
    )
    # Resume preserves validation history, while a caller can raise the target.
    training_state["target_iou"] = args.target_iou
    training_state["target_met"] = training_state.get(
        "validation_step"
    ) == start_step - 1 and _target_met(training_state.get("validation", {}), args.target_iou)
    training_state["resolved_args"] = vars(args)
    for parameter_group in opt.param_groups:
        parameter_group["lr"] = args.lr
    rollout = TrainingRollout(model, runtime_slow_aspect_norm, args.seq_len)
    if world_size > 1:
        rollout = DistributedDataParallel(
            rollout,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    metrics_path = (
        Path(args.metrics_file) if args.metrics_file else best_path.with_suffix(".metrics.jsonl")
    )

    def record(kind: str, step_num: int, values: dict) -> None:
        if rank == 0:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a") as stream:
                stream.write(
                    json.dumps({"kind": kind, "step": step_num, **values}, allow_nan=False) + "\n"
                )

    def save(path: Path, step_num: int) -> None:
        _save_training_checkpoint(
            path=path,
            step_num=step_num,
            model=model,
            opt=opt,
            train_ds=train_ds,
            context_window=args.seq_len,
            cfg=cfg,
            game_csvs=game_csvs,
            target_mode=args.target_mode,
            include_pose=args.include_pose,
            source_init_report=source_init_report,
            training_state=training_state,
        )

    record(
        "run",
        start_step - 1,
        {
            "data_identity": data_identity,
            "evaluation": training_state["evaluation"],
            "args": vars(args),
            "step_budget": steps,
            "torch": torch.__version__,
        },
    )
    if training_state["target_met"]:
        logger.info("Resumed checkpoint already meets target validation IoU %.4f", args.target_iou)
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    # Vary the stochastic stream after resume instead of replaying the first windows.
    train_ds._seed += max(0, start_step - 1) * 1_000_033
    it = iter(train_loader)
    last_step = start_step - 1
    for step in range(start_step, steps + 1):
        last_step = step
        rollout.train()
        batch = {
            key: value.to(device, non_blocking=args.pin_memory) for key, value in next(it).items()
        }
        fraction = (
            min(step / max(1, args.ss_warmup_steps), 1.0) if args.ss_warmup_steps > 0 else 1.0
        )
        probability = (
            args.ss_prob_start + (args.ss_prob_end - args.ss_prob_start) * fraction
            if args.scheduled_sampling
            else 0.0
        )
        pred = rollout(batch, probability)
        loss, metrics = _compute_losses(
            pred,
            batch["y"],
            w_l1=args.loss_l1_weight,
            w_iou=args.loss_iou_weight,
            w_vel=args.loss_vel_weight,
            w_acc=args.loss_acc_weight,
            fast_mult=args.fast_loss_mult,
            runtime_slow_aspect_norm=runtime_slow_aspect_norm,
        )
        finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0:
            if dist.is_initialized():
                keys = sorted(metrics)
                values = torch.tensor([metrics[k] for k in keys], device=device)
                dist.all_reduce(values)
                metrics = {k: float(values[i]) / world_size for i, k in enumerate(keys)}
            logger.info("step %d/%d: train %s", step, steps, metrics)
            record("train", step, metrics)
        do_eval = (
            val_loader is not None
            and args.eval_every > 0
            and (step % args.eval_every == 0 or step == steps)
        )
        if do_eval:
            val_metrics = _eval_metrics(
                model,
                val_loader,
                device,
                steps=args.val_steps,
                w_l1=args.loss_l1_weight,
                w_iou=args.loss_iou_weight,
                w_vel=args.loss_vel_weight,
                w_acc=args.loss_acc_weight,
                fast_mult=args.fast_loss_mult,
                free_run=args.eval_free_run,
                runtime_slow_aspect_norm=runtime_slow_aspect_norm,
                context_window=args.seq_len,
            )
            if not all(math.isfinite(v) for v in val_metrics.values()):
                raise RuntimeError(f"Nonfinite validation metrics: {val_metrics}")
            training_state["validation"] = val_metrics
            training_state["validation_step"] = step
            training_state["target_met"] = _target_met(val_metrics, args.target_iou)
            logger.info("step %d/%d: validation %s", step, steps, val_metrics)
            record("validation", step, {**val_metrics, "target_met": training_state["target_met"]})
            improved = val_metrics["loss"] < training_state["best_val"]
            if improved:
                training_state["best_val"] = val_metrics["loss"]
            if improved or training_state["target_met"]:
                save(best_path, step)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save(_checkpoint_path(best_path, prefix, ext, step), step)
            if rank == 0 and args.max_checkpoints > 0:
                numbered = _list_numbered_checkpoints(best_path, prefix, ext)
                for _, old_path in numbered[: -args.max_checkpoints]:
                    old_path.unlink()
        if training_state["target_met"]:
            logger.info("Reached target validation IoU %.4f at step %d", args.target_iou, step)
            break
    if last_step >= start_step:
        save(_checkpoint_path(best_path, prefix, ext, last_step), last_step)
        if val_loader is None:
            save(best_path, last_step)
    record(
        "finished",
        last_step,
        {
            "target_met": training_state["target_met"],
            "validation": training_state.get("validation"),
            "target_iou": args.target_iou,
        },
    )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
