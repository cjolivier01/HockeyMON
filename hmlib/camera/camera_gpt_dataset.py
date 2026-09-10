"""Datasets for training the GPT-style camera controller from saved CSVs."""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from hmlib.camera.camera_policy import read_camera_policy_boundaries
from hmlib.camera.camera_transformer import (
    CameraNorm,
    build_frame_base_features,
    build_frame_features,
    build_player_box_features,
)


@dataclass(frozen=True)
class GameCsvPaths:
    game_id: str
    tracking_csv: str
    camera_csv: str
    camera_fast_csv: Optional[str] = None
    pose_csv: Optional[str] = None


def _read_tracking_dataframe(tracking_csv: str) -> pd.DataFrame:
    tracks = pd.read_csv(tracking_csv, header=None)
    # Support legacy tracking CSVs with a PoseIndex column (11/14) and current ones (10/13).
    base_cols = [
        "Frame",
        "ID",
        "BBox_X",
        "BBox_Y",
        "BBox_W",
        "BBox_H",
        "Scores",
        "Labels",
        "Visibility",
        "JerseyInfo",
    ]
    extra_cols = ["ActionLabel", "ActionScore", "ActionIndex"]
    legacy_cols = base_cols + ["PoseIndex"]
    if tracks.shape[1] >= len(legacy_cols) + len(extra_cols):
        tracks = tracks.iloc[:, : len(legacy_cols) + len(extra_cols)]
        tracks.columns = legacy_cols + extra_cols
        tracks = tracks.drop(columns=["PoseIndex"])
    elif tracks.shape[1] == len(legacy_cols):
        tracks.columns = legacy_cols
        tracks = tracks.drop(columns=["PoseIndex"])
    elif tracks.shape[1] >= len(base_cols) + len(extra_cols):
        tracks = tracks.iloc[:, : len(base_cols) + len(extra_cols)]
        tracks.columns = base_cols + extra_cols
    elif tracks.shape[1] == len(base_cols):
        tracks.columns = base_cols
    else:
        need = len(base_cols) - tracks.shape[1]
        for _ in range(need):
            tracks[tracks.shape[1]] = None
        tracks.columns = base_cols
    return tracks


def _read_camera_dataframe(camera_csv: str) -> pd.DataFrame:
    cams = pd.read_csv(camera_csv, header=None)
    cams.columns = ["Frame", "BBox_X", "BBox_Y", "BBox_W", "BBox_H"]
    return cams


def _read_pose_dataframe(pose_csv: str) -> pd.DataFrame:
    poses = pd.read_csv(pose_csv, header=None)
    poses.columns = ["Frame", "PoseJSON"]
    return poses


def scan_game_max_xy(
    tracking_csv: str, camera_csv: str, camera_fast_csv: Optional[str] = None
) -> Tuple[float, float]:
    """Return (max_x, max_y) across tracking + camera CSVs for a game."""
    max_x = float("nan")
    max_y = float("nan")
    try:
        tracks = _read_tracking_dataframe(tracking_csv)
        x2 = pd.to_numeric(tracks["BBox_X"], errors="coerce") + pd.to_numeric(
            tracks["BBox_W"], errors="coerce"
        )
        y2 = pd.to_numeric(tracks["BBox_Y"], errors="coerce") + pd.to_numeric(
            tracks["BBox_H"], errors="coerce"
        )
        max_x = float(np.nanmax(x2.to_numpy(dtype=np.float64)))
        max_y = float(np.nanmax(y2.to_numpy(dtype=np.float64)))
    except Exception:
        pass
    try:
        cams = _read_camera_dataframe(camera_csv)
        x2 = pd.to_numeric(cams["BBox_X"], errors="coerce") + pd.to_numeric(
            cams["BBox_W"], errors="coerce"
        )
        y2 = pd.to_numeric(cams["BBox_Y"], errors="coerce") + pd.to_numeric(
            cams["BBox_H"], errors="coerce"
        )
        max_x = float(np.nanmax([max_x, float(np.nanmax(x2.to_numpy(dtype=np.float64)))]))
        max_y = float(np.nanmax([max_y, float(np.nanmax(y2.to_numpy(dtype=np.float64)))]))
    except Exception:
        pass
    if camera_fast_csv:
        try:
            cams = _read_camera_dataframe(camera_fast_csv)
            x2 = pd.to_numeric(cams["BBox_X"], errors="coerce") + pd.to_numeric(
                cams["BBox_W"], errors="coerce"
            )
            y2 = pd.to_numeric(cams["BBox_Y"], errors="coerce") + pd.to_numeric(
                cams["BBox_H"], errors="coerce"
            )
            max_x = float(np.nanmax([max_x, float(np.nanmax(x2.to_numpy(dtype=np.float64)))]))
            max_y = float(np.nanmax([max_y, float(np.nanmax(y2.to_numpy(dtype=np.float64)))]))
        except Exception:
            pass
    if not np.isfinite(max_x) or max_x <= 0:
        max_x = 1920.0
    if not np.isfinite(max_y) or max_y <= 0:
        max_y = 1080.0
    return max_x, max_y


def _contiguous_frame_runs(
    frames: list[int], policy_boundaries: Optional[set[int]] = None
) -> list[list[int]]:
    """Split sorted frames at numeric gaps or effective camera-policy changes."""
    runs: list[list[int]] = []
    for frame in frames:
        if not runs or frame != runs[-1][-1] + 1 or frame in (policy_boundaries or ()):
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


@dataclass
class _LoadedGame:
    game_id: str
    frames: list[int]
    frame_runs: list[list[int]]
    tracks_by_frame: Dict[int, np.ndarray]
    cam_slow_tlwh_by_frame: Dict[int, np.ndarray]  # normalized TLWH
    cam_fast_tlwh_by_frame: Dict[int, np.ndarray]  # normalized TLWH
    pose_feat_by_frame: Dict[int, np.ndarray]
    rink_feat: np.ndarray  # fixed per-game features


def _pose_features_from_json(pose_json: str, norm: CameraNorm) -> np.ndarray:
    """Return fixed-length pose features for a frame (zeros on missing/parse errors)."""
    feat = np.zeros((8,), dtype=np.float32)
    if not pose_json:
        return feat
    try:
        obj = json.loads(pose_json)
    except Exception:
        return feat
    preds = obj.get("predictions") if isinstance(obj, dict) else None
    if not isinstance(preds, list) or not preds:
        return feat
    ds0 = preds[0] if isinstance(preds[0], dict) else None
    if ds0 is None:
        return feat

    bboxes = ds0.get("bboxes")
    bboxes_arr = None
    try:
        bboxes_arr = (
            np.asarray(bboxes, dtype=np.float32).reshape(-1, 4) if bboxes is not None else None
        )
    except Exception:
        bboxes_arr = None
    if bboxes_arr is not None and bboxes_arr.size:
        x1 = bboxes_arr[:, 0]
        y1 = bboxes_arr[:, 1]
        x2 = bboxes_arr[:, 2]
        y2 = bboxes_arr[:, 3]
        cx = (x1 + x2) * 0.5 / max(1e-6, float(norm.scale_x))
        cy = (y1 + y2) * 0.5 / max(1e-6, float(norm.scale_y))
        hh = (y2 - y1) / max(1e-6, float(norm.scale_y))
        feat[0] = float(min(len(bboxes_arr) / max(1, int(norm.max_players)), 1.0))
        feat[1] = float(np.clip(np.mean(cx), 0.0, 1.0))
        feat[2] = float(np.clip(np.mean(cy), 0.0, 1.0))
        feat[3] = float(np.clip(np.std(cx), 0.0, 1.0))
        feat[4] = float(np.clip(np.std(cy), 0.0, 1.0))
        feat[5] = float(np.clip(np.mean(hh), 0.0, 1.0))

    # Scores: prefer keypoint_scores; fall back to bbox_scores or scores.
    score = None
    for key in ("keypoint_scores", "bbox_scores", "scores"):
        val = ds0.get(key)
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=np.float32)
            if arr.size:
                score = float(np.mean(arr))
                break
        except Exception:
            continue
    if score is not None:
        feat[6] = float(np.clip(score, 0.0, 1.0))

    # Fraction of keypoints above 0.5 if available.
    kps = ds0.get("keypoint_scores")
    if kps is not None:
        try:
            arr = np.asarray(kps, dtype=np.float32)
            if arr.size:
                feat[7] = float(np.mean(arr > 0.5))
        except Exception:
            pass
    return feat


def _rink_features_from_mask_png(game_dir: str, norm: CameraNorm) -> np.ndarray:
    """Return fixed-length rink features derived from rink_mask_0.png (zeros if missing)."""
    feat = np.zeros((7,), dtype=np.float32)
    try:
        import cv2

        p = Path(game_dir) / "rink_mask_0.png"
        if not p.is_file():
            return feat
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            return feat
        # Treat non-zero pixels as rink.
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return feat
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max())
        y2 = float(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        area = float(xs.size) / float(mask.shape[0] * mask.shape[1])
        sx = max(1e-6, float(norm.scale_x))
        sy = max(1e-6, float(norm.scale_y))
        feat[0] = float(np.clip(x1 / sx, 0.0, 1.0))
        feat[1] = float(np.clip(y1 / sy, 0.0, 1.0))
        feat[2] = float(np.clip(x2 / sx, 0.0, 1.0))
        feat[3] = float(np.clip(y2 / sy, 0.0, 1.0))
        feat[4] = float(np.clip(cx / sx, 0.0, 1.0))
        feat[5] = float(np.clip(cy / sy, 0.0, 1.0))
        feat[6] = float(np.clip(area, 0.0, 1.0))
        return feat
    except Exception:
        return feat


def _load_game(
    paths: GameCsvPaths, norm: CameraNorm, target_mode: str, include_pose: bool, include_rink: bool
) -> _LoadedGame:
    tracks = _read_tracking_dataframe(paths.tracking_csv)
    cams = _read_camera_dataframe(paths.camera_csv)
    cams_fast = _read_camera_dataframe(paths.camera_fast_csv) if paths.camera_fast_csv else None

    # Frames present in both tracking and (slow) camera.
    frames_set = set(tracks["Frame"].unique()).intersection(set(cams["Frame"].unique()))
    if target_mode == "slow_fast_tlwh":
        if cams_fast is None:
            frames_set = set()
        else:
            frames_set = frames_set.intersection(set(cams_fast["Frame"].unique()))
    frames = sorted(frames_set)
    frames_int = [int(f) for f in frames]
    policy_boundaries = read_camera_policy_boundaries(paths.camera_csv, cams["Frame"])
    if target_mode == "slow_fast_tlwh" and cams_fast is not None:
        policy_boundaries.update(
            read_camera_policy_boundaries(paths.camera_fast_csv, cams_fast["Frame"])
        )
    frame_runs = _contiguous_frame_runs(frames_int, policy_boundaries)

    tracks_by_frame: Dict[int, np.ndarray] = {}
    if not tracks.empty:
        for frame_id, group in tracks.groupby("Frame"):
            fid = int(frame_id)
            arr = group[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy(dtype=np.float32)
            tracks_by_frame[fid] = arr

    cam_slow_tlwh_by_frame: Dict[int, np.ndarray] = {}
    if not cams.empty:
        for row in cams.itertuples(index=False):
            fid = int(getattr(row, "Frame"))
            x = float(getattr(row, "BBox_X"))
            y = float(getattr(row, "BBox_Y"))
            w = float(getattr(row, "BBox_W"))
            h = float(getattr(row, "BBox_H"))
            x1 = float(np.clip(x / max(1e-6, float(norm.scale_x)), 0.0, 1.0))
            y1 = float(np.clip(y / max(1e-6, float(norm.scale_y)), 0.0, 1.0))
            w1 = float(np.clip(w / max(1e-6, float(norm.scale_x)), 0.0, 1.0))
            h1 = float(np.clip(h / max(1e-6, float(norm.scale_y)), 0.0, 1.0))
            cam_slow_tlwh_by_frame[fid] = np.asarray([x1, y1, w1, h1], dtype=np.float32)

    cam_fast_tlwh_by_frame: Dict[int, np.ndarray] = {}
    if cams_fast is not None and not cams_fast.empty:
        for row in cams_fast.itertuples(index=False):
            fid = int(getattr(row, "Frame"))
            x = float(getattr(row, "BBox_X"))
            y = float(getattr(row, "BBox_Y"))
            w = float(getattr(row, "BBox_W"))
            h = float(getattr(row, "BBox_H"))
            x1 = float(np.clip(x / max(1e-6, float(norm.scale_x)), 0.0, 1.0))
            y1 = float(np.clip(y / max(1e-6, float(norm.scale_y)), 0.0, 1.0))
            w1 = float(np.clip(w / max(1e-6, float(norm.scale_x)), 0.0, 1.0))
            h1 = float(np.clip(h / max(1e-6, float(norm.scale_y)), 0.0, 1.0))
            cam_fast_tlwh_by_frame[fid] = np.asarray([x1, y1, w1, h1], dtype=np.float32)

    pose_feat_by_frame: Dict[int, np.ndarray] = {}
    if include_pose and paths.pose_csv:
        try:
            poses = _read_pose_dataframe(paths.pose_csv)
            for row in poses.itertuples(index=False):
                fid = int(getattr(row, "Frame"))
                pose_json = getattr(row, "PoseJSON")
                pose_feat_by_frame[fid] = _pose_features_from_json(str(pose_json), norm=norm)
        except Exception as ex:
            raise RuntimeError(f"Failed to parse pose CSV {paths.pose_csv!r}") from ex

    rink_feat = (
        _rink_features_from_mask_png(str(Path(paths.tracking_csv).parent), norm=norm)
        if include_rink
        else np.zeros((7,), dtype=np.float32)
    )

    return _LoadedGame(
        game_id=paths.game_id,
        frames=frames_int,
        frame_runs=frame_runs,
        tracks_by_frame=tracks_by_frame,
        cam_slow_tlwh_by_frame=cam_slow_tlwh_by_frame,
        cam_fast_tlwh_by_frame=cam_fast_tlwh_by_frame,
        pose_feat_by_frame=pose_feat_by_frame,
        rink_feat=rink_feat,
    )


class CameraPanZoomGPTIterableDataset(IterableDataset):
    """Randomly samples fixed-length sequences from multiple games."""

    def __init__(
        self,
        games: list[GameCsvPaths],
        norm: CameraNorm,
        seq_len: int = 32,
        target_mode: str = "slow_center_h",
        feature_mode: str = "base_prev_y",
        include_pose: bool = True,
        include_rink: bool = False,
        max_players_for_norm: int = 22,
        seed: int = 0,
        max_cached_games: int = 8,
        *,
        preload_csv: str = "none",
        shard_games_by_worker: bool = False,
    ) -> None:
        super().__init__()
        self._games = games
        self._norm = CameraNorm(
            scale_x=float(norm.scale_x),
            scale_y=float(norm.scale_y),
            max_players=int(max_players_for_norm),
        )
        self._seq_len = int(seq_len)
        self._target_mode = str(target_mode)
        self._feature_mode = str(feature_mode)
        self._include_pose = bool(include_pose)
        self._include_rink = bool(include_rink)
        self._pose_feat_dim = 8 if self._include_pose else 0
        self._rink_feat_dim = 7 if self._include_rink else 0
        self._seed = int(seed)
        self._max_cached = int(max_cached_games)
        self._preload_csv = str(preload_csv)
        self._shard_games_by_worker = bool(shard_games_by_worker)
        self._cache: Dict[str, _LoadedGame] = {}
        self._cache_order: deque[str] = deque()
        self._unusable_reasons: Dict[str, str] = {}
        if self._feature_mode not in {"legacy_prev_slow", "base_prev_y", "players_prev_y"}:
            raise ValueError(f"Unknown feature_mode: {self._feature_mode}")
        if self._preload_csv not in {"none", "shard", "all"}:
            raise ValueError(f"Unknown preload_csv: {self._preload_csv}")

    @property
    def norm(self) -> CameraNorm:
        return self._norm

    @property
    def base_dim(self) -> int:
        """Dimension of per-frame inputs excluding previous camera state."""
        player_dim = (
            4 * int(self._norm.max_players) if self._feature_mode == "players_prev_y" else 0
        )
        return 8 + player_dim + int(self._pose_feat_dim) + int(self._rink_feat_dim)

    @property
    def feature_dim(self) -> int:
        if self._feature_mode == "legacy_prev_slow":
            return 11 + int(self._pose_feat_dim) + int(self._rink_feat_dim)
        if self._feature_mode in {"base_prev_y", "players_prev_y"}:
            return int(self.base_dim) + int(self.target_dim)
        raise ValueError(f"Unknown feature_mode: {self._feature_mode}")

    @property
    def target_dim(self) -> int:
        if self._target_mode == "slow_center_h":
            return 3
        if self._target_mode == "slow_tlwh":
            return 4
        if self._target_mode == "slow_fast_tlwh":
            return 8
        raise ValueError(f"Unknown target_mode: {self._target_mode}")

    def _get_game(self, paths: GameCsvPaths) -> Optional[_LoadedGame]:
        game_id = paths.game_id
        cached = self._cache.get(game_id)
        if cached is not None:
            return cached
        if self._target_mode == "slow_fast_tlwh" and not paths.camera_fast_csv:
            self._unusable_reasons[game_id] = "missing camera_fast.csv"
            return None
        try:
            loaded = _load_game(
                paths,
                norm=self._norm,
                target_mode=self._target_mode,
                include_pose=self._include_pose,
                include_rink=self._include_rink,
            )
        except Exception as ex:
            raise RuntimeError(f"Failed to load camera GPT CSVs for game {game_id!r}") from ex
        longest_run = max((len(run) for run in loaded.frame_runs), default=0)
        if longest_run < self._seq_len:
            reason = f"longest contiguous run has {longest_run} frames for seq_len={self._seq_len}"
            self._unusable_reasons[game_id] = reason
            return None
        self._unusable_reasons.pop(game_id, None)
        self._cache[game_id] = loaded
        self._cache_order.append(game_id)
        while len(self._cache_order) > self._max_cached:
            old = self._cache_order.popleft()
            self._cache.pop(old, None)
        return loaded

    def _usable_game_paths(self, games: List[GameCsvPaths]) -> List[GameCsvPaths]:
        usable = [
            p for p in games if p.game_id in self._cache or p.game_id not in self._unusable_reasons
        ]
        if usable:
            return usable
        reasons = ", ".join(
            f"{gid}: {reason}" for gid, reason in sorted(self._unusable_reasons.items())
        )
        raise RuntimeError(f"No usable camera GPT games remain. {reasons}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = int(worker.num_workers) if worker is not None else 1
        rng = random.Random(self._seed + worker_id)

        games = self._games
        if self._shard_games_by_worker and worker is not None and num_workers > 1:
            shard = list(games[worker_id::num_workers])
            if shard:
                games = shard

        if self._preload_csv != "none":
            to_load = games if self._preload_csv == "shard" else self._games
            # Ensure the cache can hold all preloaded games without eviction.
            self._max_cached = max(self._max_cached, len(to_load))
            for paths in to_load:
                self._get_game(paths)
            games = self._usable_game_paths(games)

        while True:
            paths = rng.choice(games)
            game = self._get_game(paths)
            if game is None:
                games = self._usable_game_paths(games)
                continue

            eligible_runs = [run for run in game.frame_runs if len(run) >= self._seq_len]
            frames = rng.choice(eligible_runs)
            start = rng.randint(0, len(frames) - self._seq_len)
            seq_frames = frames[start : start + self._seq_len]

            feats = []
            targets = []
            prev0: Optional[np.ndarray] = None
            if self._feature_mode in {"base_prev_y", "players_prev_y"}:
                prev_frame = frames[start - 1] if start > 0 else None
                prev_slow = (
                    game.cam_slow_tlwh_by_frame.get(prev_frame) if prev_frame is not None else None
                )
                if prev_slow is None:
                    prev_slow = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
                if self._target_mode == "slow_center_h":
                    prev0 = np.asarray(
                        [
                            float(prev_slow[0] + prev_slow[2] * 0.5),
                            float(prev_slow[1] + prev_slow[3] * 0.5),
                            float(prev_slow[3]),
                        ],
                        dtype=np.float32,
                    )
                elif self._target_mode == "slow_tlwh":
                    prev0 = prev_slow.astype(np.float32, copy=False)
                elif self._target_mode == "slow_fast_tlwh":
                    prev_fast = (
                        game.cam_fast_tlwh_by_frame.get(prev_frame)
                        if prev_frame is not None
                        else None
                    )
                    if prev_fast is None:
                        prev_fast = prev_slow
                    prev0 = np.concatenate(
                        [
                            prev_slow.astype(np.float32, copy=False),
                            prev_fast.astype(np.float32, copy=False),
                        ],
                        axis=0,
                    ).astype(np.float32, copy=False)
                else:
                    raise ValueError(f"Unknown target_mode: {self._target_mode}")

            for f in seq_frames:
                tlwh = game.tracks_by_frame.get(f)
                if tlwh is None:
                    tlwh = np.zeros((0, 4), dtype=np.float32)
                if self._feature_mode == "legacy_prev_slow":
                    # Teacher forcing on the slow camera state, for legacy checkpoints.
                    prev_cx, prev_cy, prev_h = (0.0, 0.0, 0.0)
                    if targets:
                        y_prev = targets[-1]
                        if self._target_mode == "slow_center_h":
                            prev_cx, prev_cy, prev_h = (
                                float(y_prev[0]),
                                float(y_prev[1]),
                                float(y_prev[2]),
                            )
                        else:
                            prev_slow = y_prev[:4]
                            prev_cx = float(prev_slow[0] + prev_slow[2] * 0.5)
                            prev_cy = float(prev_slow[1] + prev_slow[3] * 0.5)
                            prev_h = float(prev_slow[3])
                    elif start > 0:
                        prev_slow = game.cam_slow_tlwh_by_frame.get(frames[start - 1])
                        if prev_slow is not None and len(prev_slow) == 4:
                            prev_cx = float(prev_slow[0] + prev_slow[2] * 0.5)
                            prev_cy = float(prev_slow[1] + prev_slow[3] * 0.5)
                            prev_h = float(prev_slow[3])
                    feat = build_frame_features(
                        tlwh=tlwh,
                        norm=self._norm,
                        prev_cam_center=(prev_cx, prev_cy),
                        prev_cam_h=prev_h,
                    )
                else:
                    feat = build_frame_base_features(tlwh=tlwh, norm=self._norm)
                    if self._feature_mode == "players_prev_y":
                        feat = np.concatenate(
                            [
                                feat,
                                build_player_box_features(
                                    tlwh=tlwh,
                                    norm=self._norm,
                                    max_players=int(self._norm.max_players),
                                ),
                            ],
                            axis=0,
                        )

                if self._include_pose:
                    pose_feat = game.pose_feat_by_frame.get(f)
                    if pose_feat is None:
                        pose_feat = np.zeros((self._pose_feat_dim,), dtype=np.float32)
                    feat = np.concatenate([feat, pose_feat.astype(np.float32, copy=False)], axis=0)
                if self._include_rink:
                    rf = game.rink_feat
                    if rf is None or rf.shape[0] != int(self._rink_feat_dim):
                        rf = np.zeros((self._rink_feat_dim,), dtype=np.float32)
                    feat = np.concatenate([feat, rf.astype(np.float32, copy=False)], axis=0)
                feats.append(feat.astype(np.float32, copy=False))

                slow_tlwh = game.cam_slow_tlwh_by_frame.get(f)
                if slow_tlwh is None:
                    slow_tlwh = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

                if self._target_mode == "slow_center_h":
                    cx = float(slow_tlwh[0] + slow_tlwh[2] * 0.5)
                    cy = float(slow_tlwh[1] + slow_tlwh[3] * 0.5)
                    hr = float(slow_tlwh[3])
                    y = np.asarray([cx, cy, hr], dtype=np.float32)
                elif self._target_mode == "slow_tlwh":
                    y = slow_tlwh.astype(np.float32, copy=False)
                elif self._target_mode == "slow_fast_tlwh":
                    fast_tlwh = game.cam_fast_tlwh_by_frame.get(f)
                    if fast_tlwh is None:
                        fast_tlwh = slow_tlwh
                    y = np.concatenate(
                        [
                            slow_tlwh.astype(np.float32, copy=False),
                            fast_tlwh.astype(np.float32, copy=False),
                        ],
                        axis=0,
                    )
                else:
                    raise ValueError(f"Unknown target_mode: {self._target_mode}")

                targets.append(y.astype(np.float32, copy=False))

            x = torch.from_numpy(np.stack(feats, axis=0))  # [T, D] (legacy) or base features
            y = torch.from_numpy(np.stack(targets, axis=0))  # [T, target_dim]
            out: Dict[str, torch.Tensor] = {"y": y}
            if self._feature_mode == "legacy_prev_slow":
                out["x"] = x
            else:
                out["base"] = x
                assert prev0 is not None
                out["prev0"] = torch.from_numpy(prev0.astype(np.float32, copy=False))
            yield out


def _hstream_manifest_allows_generation(
    base_path: Path,
    suffix_part: str,
    tracking_path: Path,
    camera_path: Path,
    camera_fast_path: Path,
) -> bool:
    """Reject incomplete hstream publications while preserving legacy CSV discovery."""
    manifest_path = base_path / f"hstream_telemetry{suffix_part}.json"
    if not manifest_path.exists():
        return True
    if not manifest_path.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    if not isinstance(manifest, dict):
        return False

    compatibility = manifest.get("hm_compatibility")
    if not isinstance(compatibility, dict):
        return False

    def compatibility_file(key: str) -> Optional[str]:
        artifact = compatibility.get(key)
        if not isinstance(artifact, dict):
            return None
        filename = artifact.get("file")
        return filename if isinstance(filename, str) else None

    return (
        manifest.get("schema") == "hstream-playtracker-telemetry-v1"
        and manifest.get("publication_state") == "committed"
        and manifest.get("completed") is True
        and manifest.get("eligible_for_training") is True
        and compatibility_file("tracking_csv") == tracking_path.name
        and compatibility_file("camera_csv") == camera_path.name
        and compatibility_file("camera_fast_csv") == camera_fast_path.name
        and camera_fast_path.is_file()
    )


def _latest_complete_camera_generation(
    game_dir: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return matching tracking/camera paths from the newest complete generation."""
    base_path = Path(game_dir)
    if not base_path.is_dir():
        return None, None, None

    tracking_candidates: list[tuple[int, str, Path]] = []
    bare_tracking = base_path / "tracking.csv"
    if bare_tracking.is_file():
        tracking_candidates.append((0, "", bare_tracking))
    for tracking_path in base_path.glob("tracking-*.csv"):
        suffix = tracking_path.stem[len("tracking-") :]
        if suffix.isdigit() and tracking_path.is_file():
            tracking_candidates.append((int(suffix), suffix, tracking_path))

    for _, suffix, tracking_path in sorted(
        tracking_candidates, key=lambda candidate: candidate[0], reverse=True
    ):
        suffix_part = f"-{suffix}" if suffix else ""
        camera_path = base_path / f"camera{suffix_part}.csv"
        if not camera_path.is_file():
            continue
        camera_fast_path = base_path / f"camera_fast{suffix_part}.csv"
        if not _hstream_manifest_allows_generation(
            base_path, suffix_part, tracking_path, camera_path, camera_fast_path
        ):
            continue
        return (
            str(tracking_path),
            str(camera_path),
            str(camera_fast_path) if camera_fast_path.is_file() else None,
        )

    return None, None, None


def resolve_csv_paths(
    game_id: str,
    game_dir: str,
    *,
    tracking_csv_name: Optional[str] = None,
    camera_csv_name: Optional[str] = None,
    camera_fast_csv_name: Optional[str] = None,
    pose_csv_name: Optional[str] = None,
) -> Optional[GameCsvPaths]:
    """Resolve required CSVs inside a game directory.

    The optional ``*_csv_name`` arguments allow selecting fixed filenames such as
    ``camera_annotated.csv`` produced by manual editing tools. When both required
    names are automatic, tracking and camera are selected from the newest complete
    suffix generation so an interrupted publisher cannot mix two generations.
    """
    from hmlib.datasets.dataframe import find_latest_dataframe_file

    automatic_generation = not tracking_csv_name and not camera_csv_name
    if automatic_generation:
        tracking, camera, generation_camera_fast = _latest_complete_camera_generation(game_dir)
    elif tracking_csv_name:
        tracking_path = str(Path(game_dir) / tracking_csv_name)
        tracking = tracking_path if Path(tracking_path).is_file() else None
    else:
        tracking = find_latest_dataframe_file(game_dir, "tracking")

    if automatic_generation:
        pass
    elif camera_csv_name:
        camera_path = str(Path(game_dir) / camera_csv_name)
        camera = camera_path if Path(camera_path).is_file() else None
    else:
        camera = find_latest_dataframe_file(game_dir, "camera")

    if not tracking or not camera:
        return None

    if camera_fast_csv_name:
        fast_path = str(Path(game_dir) / camera_fast_csv_name)
        camera_fast = fast_path if Path(fast_path).is_file() else None
    elif automatic_generation:
        camera_fast = generation_camera_fast
    else:
        camera_fast = find_latest_dataframe_file(game_dir, "camera_fast")

    if pose_csv_name:
        pose_path = str(Path(game_dir) / pose_csv_name)
        pose = pose_path if Path(pose_path).is_file() else None
    else:
        pose = find_latest_dataframe_file(game_dir, "pose")

    return GameCsvPaths(
        game_id=game_id,
        tracking_csv=tracking,
        camera_csv=camera,
        camera_fast_csv=camera_fast,
        pose_csv=pose,
    )


def validate_csv_paths(paths: GameCsvPaths) -> None:
    for p in (paths.tracking_csv, paths.camera_csv):
        if not Path(p).is_file():
            raise FileNotFoundError(p)
