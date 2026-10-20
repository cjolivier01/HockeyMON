"""VIP-HARPET's MPII16 + stick-top + stick-tip contract and checked conversion."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

JOINTS = (
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "thorax",
    "neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "stick_top",
    "stick_tip",
)
FLIP = (5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10, 16, 17)
EDGES = (
    (0, 1),
    (1, 2),
    (2, 6),
    (3, 6),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (8, 9),
    (7, 12),
    (7, 13),
    (12, 11),
    (11, 10),
    (13, 14),
    (14, 15),
    (16, 17),
)
LABELS = ("backward skating", "forward skating", "passing", "shooting")
ACTION_IDS = {name: i for i, name in enumerate(("Backward", "Forward", "Passing", "Shooting"))}
NAME = re.compile(r"ImageSequences(Backward|Forward|Passing|Shooting)_(\d+)_(\d+)\.jpg")
METAINFO = {
    "dataset_name": "harpet18",
    "keypoint_info": {
        i: {
            "name": name,
            "id": i,
            "color": [255, 128, 0],
            "type": "lower" if i < 7 else "upper",
            "swap": JOINTS[FLIP[i]] if FLIP[i] != i else "",
        }
        for i, name in enumerate(JOINTS)
    },
    "skeleton_info": {
        i: {"link": (JOINTS[a], JOINTS[b]), "id": i, "color": [255, 128, 0]}
        for i, (a, b) in enumerate(EDGES)
    },
    "joint_weights": [1.0] * 18,
    "sigmas": [0.025] * 18,
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def software_provenance() -> dict:
    """Record actual imported packages and source files, including editable installs."""
    packages = {}
    for name in ("torch", "mmengine", "mmpose", "mmaction", "mmdet", "mmcv"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Required training package is unavailable: {name}")
        path = Path(spec.origin).resolve()
        revision = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        packages[name] = {
            "file": str(path),
            "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        }
    scripts = Path(__file__).parent
    return {
        "python": sys.version,
        "packages": packages,
        "distributions": sorted(
            f"{package.metadata['Name']}=={package.version}"
            for package in importlib.metadata.distributions()
        ),
        "source_sha256": {path.name: digest(path) for path in sorted(scripts.glob("*hockey*.py"))},
    }


def configure_native_crops() -> None:
    """Use the OpenCV interpolation table implemented by HStream's CUDA sampler."""
    import cv2

    if cv2.__version__ != "4.12.0":
        raise RuntimeError(
            "Native crop parity requires opencv-python-headless==4.12.0.88; "
            f"found OpenCV {cv2.__version__}. OpenCV 5's IPP HAL changes border interpolation."
        )
    cv2.ipp.setUseIPP(False)


def load_dataset(root: Path) -> tuple[dict, dict]:
    """Validate triplets and remove lower-priority sequences sharing holdout pixels."""
    result, seen_sequences = {}, set()
    irregular = {split: [] for split in ("train", "valid", "test")}
    for split in ("train", "valid", "test"):
        with h5py.File(root / f"annot_{split}.h5") as data:
            names, points = data["imgname"][:], data["part"][:].astype(np.float32)
        if points.shape != (len(names), 18, 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid {split} joint array: {points.shape}")
        groups = {}
        for row, joints in zip(names, points, strict=True):
            name = "".join(chr(int(x)) for x in row if x)
            match = NAME.fullmatch(name)
            if match is None:
                raise ValueError(f"Unrecognized HARPET image: {name}")
            action, sequence, frame = match.groups()
            key = f"{action}_{sequence}"
            path = root / f"images_{split}" / name
            checksum = digest(path)
            with Image.open(path) as image:
                width, height = image.size
            groups.setdefault(key, []).append(
                {
                    "name": name,
                    "path": str(path),
                    "points": joints,
                    "frame": int(frame),
                    "label": ACTION_IDS[action],
                    "width": width,
                    "height": height,
                    "sha256": checksum,
                }
            )
        if seen_sequences.intersection(groups):
            raise ValueError(f"Sequence leakage in {split}")
        seen_sequences.update(groups)
        for key, frames in groups.items():
            frames.sort(key=lambda f: f["frame"])
            indices = [f["frame"] for f in frames]
            if len(frames) != 3 or len(set(indices)) != 3:
                raise ValueError(f"Expected one consecutive triplet for {key}: {indices}")
            if indices != list(range(indices[0], indices[0] + 3)):
                irregular[split].append(key)
            if len({(f["width"], f["height"]) for f in frames}) != 1:
                raise ValueError(f"Image dimensions change within {key}")
        result[split] = groups
    # Official splits contain identical images in different triplets. Retain the
    # test split, then validation; discard whole connected training/validation
    # triplets, including chains linked through another duplicate frame.
    owner, parents = {}, {}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for split, groups in result.items():
        for key, frames in groups.items():
            identity = (split, key)
            parents[identity] = identity
            for frame in frames:
                previous = owner.setdefault(frame["sha256"], identity)
                parents[find(identity)] = find(previous)
    components = {}
    for key in parents:
        components.setdefault(find(key), []).append(key)
    rank = {"train": 0, "valid": 1, "test": 2}
    excluded = {split: [] for split in result}
    for members in components.values():
        priority = max(rank[split] for split, key in members)
        for split, key in members:
            if rank[split] < priority:
                del result[split][key]
                excluded[split].append(key)
    for split, keys in irregular.items():
        for key in keys:
            result[split].pop(key, None)
    return result, {"overlap": excluded, "irregular_frames": irregular}


def prepare(root: Path, output: Path) -> dict:
    """COCO-shaped annotations with body-only boxes; stick labels cannot set crops."""
    dataset, excluded = load_dataset(root)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": "https://vip.uwaterloo.ca/vip-harpet-dataset/",
        "topology": "harpet18",
        "joints": JOINTS,
        "labels": LABELS,
        "splits": {},
        "excluded_sequences": excluded,
    }
    for split, groups in dataset.items():
        images, annotations = [], []
        for frames in groups.values():
            for frame in frames:
                i = len(images)
                p = frame["points"]
                low, high = p[:16].min(axis=0), p[:16].max(axis=0)
                width, height = high - low
                if min(width, height) <= 0:
                    raise ValueError(f"Degenerate body box: {frame['path']}")
                images.append(
                    {
                        "id": i,
                        "file_name": frame["path"],
                        "width": frame["width"],
                        "height": frame["height"],
                    }
                )
                annotations.append(
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": 1,
                        "iscrowd": 0,
                        "bbox": [float(x) for x in (*low, width, height)],
                        "area": float(width * height),
                        "num_keypoints": 18,
                        "keypoints": np.c_[p, np.full(18, 2)].ravel().tolist(),
                    }
                )
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "player",
                    "keypoints": JOINTS,
                    "skeleton": [[a + 1, b + 1] for a, b in EDGES],
                }
            ],
        }
        (output / f"{split}.json").write_text(json.dumps(payload) + "\n")
        report["splits"][split] = {
            "sequences": len(groups),
            "images": len(images),
            "annotation_sha256": digest(root / f"annot_{split}.h5"),
            "image_sha256": {f["name"]: f["sha256"] for frames in groups.values() for f in frames},
            "classes": {
                label: sum(frames[0]["label"] == i for frames in groups.values())
                for i, label in enumerate(LABELS)
            },
        }
    (output / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
