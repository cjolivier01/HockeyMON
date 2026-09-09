"""Dataset for training transformer-based camera pan/zoom models.

Wraps tracking and camera CSVs into sliding windows of frame-level features
and target camera boxes.

@see @ref hmlib.camera.camera_transformer "camera_transformer" for model details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hmlib.camera.camera_transformer import CameraNorm, build_frame_features


@dataclass
class Sample:
    x: torch.Tensor  # [T, D]
    y: torch.Tensor  # [3]  (cx, cy, h)


class CameraPanZoomDataset(Dataset):
    def __init__(
        self,
        tracking_csv: str,
        camera_csv: str,
        window: int = 8,
        max_players_for_norm: int = 22,
    ) -> None:
        super().__init__()
        self.window = int(window)
        self.tracks = pd.read_csv(tracking_csv, header=None)
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
        if self.tracks.shape[1] >= len(legacy_cols) + len(extra_cols):
            self.tracks = self.tracks.iloc[:, : len(legacy_cols) + len(extra_cols)]
            self.tracks.columns = legacy_cols + extra_cols
            self.tracks = self.tracks.drop(columns=["PoseIndex"])
        elif self.tracks.shape[1] == len(legacy_cols):
            self.tracks.columns = legacy_cols
            self.tracks = self.tracks.drop(columns=["PoseIndex"])
        elif self.tracks.shape[1] >= len(base_cols) + len(extra_cols):
            self.tracks = self.tracks.iloc[:, : len(base_cols) + len(extra_cols)]
            self.tracks.columns = base_cols + extra_cols
        elif self.tracks.shape[1] == len(base_cols):
            self.tracks.columns = base_cols
        else:
            # Fallback: pad with missing columns as needed
            need = len(base_cols) - self.tracks.shape[1]
            for _ in range(need):
                self.tracks[self.tracks.shape[1]] = None
            self.tracks.columns = base_cols
        self.cams = pd.read_csv(camera_csv, header=None)
        self.cams.columns = ["Frame", "BBox_X", "BBox_Y", "BBox_W", "BBox_H"]
        # compute normalization scales from observed maxima
        max_x = float((self.tracks["BBox_X"] + self.tracks["BBox_W"]).max())
        max_y = float((self.tracks["BBox_Y"] + self.tracks["BBox_H"]).max())
        if not np.isfinite(max_x) or max_x <= 0:
            max_x = float((self.cams["BBox_X"] + self.cams["BBox_W"]).max())
        if not np.isfinite(max_y) or max_y <= 0:
            max_y = float((self.cams["BBox_Y"] + self.cams["BBox_H"]).max())
        if not np.isfinite(max_x) or max_x <= 0:
            max_x = 1920.0
        if not np.isfinite(max_y) or max_y <= 0:
            max_y = 1080.0
        self.norm = CameraNorm(scale_x=max_x, scale_y=max_y, max_players=max_players_for_norm)
        # list of frames present in both
        frames = sorted(
            set(self.tracks["Frame"].unique()).intersection(set(self.cams["Frame"].unique()))
        )
        self.frames: List[int] = [int(f) for f in frames]
        self.frame_set = set(self.frames)
        self.valid_indices: List[int] = [
            i
            for i in range(self.window, len(self.frames))
            if all(self.frames[j + 1] == self.frames[j] + 1 for j in range(i - self.window, i))
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_tlwh(self, frame_id: int) -> np.ndarray:
        df = self.tracks[self.tracks["Frame"] == frame_id]
        if df.empty:
            return np.zeros((0, 4), dtype=np.float32)
        tlwh = df[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy(dtype=np.float32)
        return tlwh

    def _get_cam(self, frame_id: int) -> Tuple[float, float, float]:
        df = self.cams[self.cams["Frame"] == frame_id]
        if df.empty:
            return 0.5, 0.5, 1.0
        x, y, w, h = df.iloc[0][["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy(dtype=np.float32)
        cx = (x + w / 2.0) / self.norm.scale_x
        cy = (y + h / 2.0) / self.norm.scale_y
        hr = h / self.norm.scale_y
        return float(cx), float(cy), float(hr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.valid_indices[idx]
        ts = self.frames[t - self.window : t]
        feats: List[np.ndarray] = []
        prev_cx, prev_cy, prev_h = None, None, None
        for f in ts:
            if prev_cx is None:
                # Never bridge a missing numeric frame with stale camera state.
                previous_frame = f - 1
                if previous_frame in self.frame_set:
                    pcx, pcy, ph = self._get_cam(previous_frame)
                    prev_cx, prev_cy, prev_h = pcx, pcy, ph
            tlwh = self._get_tlwh(f)
            feat = build_frame_features(
                tlwh=tlwh,
                norm=self.norm,
                prev_cam_center=None if prev_cx is None else (prev_cx, prev_cy),
                prev_cam_h=prev_h,
            )
            feats.append(feat)
            # update prev from ground-truth at f
            prev_cx, prev_cy, prev_h = self._get_cam(f)
        x = torch.from_numpy(np.stack(feats, axis=0))  # [T, D]
        y = torch.tensor(self._get_cam(self.frames[t]), dtype=torch.float32)  # [3]
        return {"x": x, "y": y}
