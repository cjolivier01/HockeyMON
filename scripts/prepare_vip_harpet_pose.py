#!/usr/bin/env python3
"""
Prepare VIP-HARPET skeleton annotations for MMACTION2 PoseDataset.

Reads HDF5 annotations and image folders from --data-root (defaulting to
/mnt/data/datasets/VIP-HARPET),
groups frames into short sequences per action and sequence id, and writes
per-split pickle files at openmm/mmaction2/data/skeleton/:
  - vipharpet_train.pkl
  - vipharpet_val.pkl
  - vipharpet_test.pkl

Each annotation item contains:
  - frame_dir: unique sequence id (e.g., Backward_3)
  - label: 0..3 mapped from {Backward, Forward, Passing, Shooting}
  - keypoint: (M=1, T, V=18, C=2) float32 xy pixels
  - keypoint_score: (1, T, 18) float32 ones
  - total_frames: T
  - img_shape: (H, W)

Requires: h5py, pillow, mmengine.
"""

from __future__ import annotations

import argparse
import os.path as osp
import re
from typing import Dict, List

import h5py
import numpy as np
from PIL import Image

import mmengine

ROOT = "/mnt/data/datasets/VIP-HARPET"
SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}

# Map action strings to integer labels
CLASS_MAP = {"Backward": 0, "Forward": 1, "Passing": 2, "Shooting": 3}

# Regex for filenames like: ImageSequencesBackward_3_323.jpg
FILENAME_RE = re.compile(
    r"^ImageSequences(?P<action>Backward|Forward|Passing|Shooting)_(?P<seq>\d+)_(?P<frame>\d+)\.jpg$"
)


def _decode_imgname(arr: np.ndarray) -> str:
    # convert ASCII float64 array to string
    return "".join(chr(int(x)) for x in arr if 0 < x < 128).strip()


def load_split(split: str, data_root: str = ROOT) -> Dict[str, Dict]:
    """Load a split and group by action+sequence.

    Returns a dict keyed by sequence id (e.g., Backward_3) with fields:
      - label
      - frames: list of (frame_idx, filename, (18,2) keypoints)
    """
    suffix = SPLIT_NAMES[split]
    h5_path = osp.join(data_root, f"annot_{suffix}.h5")
    img_dir = osp.join(data_root, f"images_{suffix}")
    assert osp.isfile(h5_path), f"Missing {h5_path}"
    assert osp.isdir(img_dir), f"Missing {img_dir}"

    with h5py.File(h5_path, "r") as f:
        names = f["imgname"][:]
        parts = f["part"][:]

    groups: Dict[str, Dict] = {}
    for i in range(len(names)):
        name = _decode_imgname(names[i])
        m = FILENAME_RE.match(name)
        if not m:
            continue
        action = m.group("action")
        seq = m.group("seq")
        frame = int(m.group("frame"))
        key = f"{action}_{seq}"
        if key not in groups:
            groups[key] = {"label": CLASS_MAP[action], "frames": []}
        kp = parts[i].astype(np.float32)  # (18,2)
        groups[key]["frames"].append((frame, name, kp))

    # sort frames within each group
    for key, val in groups.items():
        val["frames"].sort(key=lambda x: x[0])
    return groups


def build_annotations(groups: Dict[str, Dict], img_dir: str) -> List[Dict]:
    """Build list of mmaction pose annotations for a split."""
    annos: List[Dict] = []
    for key, val in groups.items():
        frames = val["frames"]
        label = val["label"]
        T = len(frames)
        if T == 0:
            continue

        # Load image size from first frame
        first_path = osp.join(img_dir, frames[0][1])
        with Image.open(first_path) as im:
            W, H = im.size

        V = 18
        M = 1
        kp = np.zeros((M, T, V, 2), dtype=np.float32)
        kpscore = np.ones((M, T, V), dtype=np.float32)

        for t, (_, fname, pose) in enumerate(frames):
            # Ensure coordinates are within image bounds (best-effort clamp)
            pose[:, 0] = np.clip(pose[:, 0], 0, W - 1)
            pose[:, 1] = np.clip(pose[:, 1], 0, H - 1)
            kp[0, t, :, :] = pose

        ann = {
            "frame_dir": key,
            "label": int(label),
            "keypoint": kp,
            "keypoint_score": kpscore,
            "total_frames": int(T),
            "img_shape": (int(H), int(W)),
        }
        annos.append(ann)
    return annos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=ROOT,
        help="VIP-HARPET directory containing annot_*.h5 and images_* directories",
    )
    parser.add_argument(
        "--out-dir",
        default="openmm/mmaction2/data/skeleton",
        help="Output directory for annotation PKLs",
    )
    args = parser.parse_args()

    mmengine.mkdir_or_exist(args.out_dir)

    for split in ["train", "val", "test"]:
        print(f"Preparing split: {split}")
        groups = load_split(split, args.data_root)
        img_dir = osp.join(args.data_root, f"images_{SPLIT_NAMES[split]}")
        annos = build_annotations(groups, img_dir)
        out_path = osp.join(args.out_dir, f"vipharpet_{split}.pkl")
        mmengine.dump(annos, out_path)
        print(f"  wrote {len(annos)} sequences to {out_path}")


if __name__ == "__main__":
    main()
