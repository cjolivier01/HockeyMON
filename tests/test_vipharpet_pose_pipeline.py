import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

import mmengine

REPO_ROOT = Path(__file__).resolve().parents[1]
MMACTION_ROOT = REPO_ROOT / "openmm" / "mmaction2"
CONFIG = MMACTION_ROOT / "configs" / "skeleton" / "stgcn" / "vipharpet_stgcn_openpose_3f.py"
ACTIONS = ("Backward", "Forward", "Passing", "Shooting")
FRAME_INDICES = (20, 3, 11)
IMAGE_SHAPE = (64, 96)


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(MMACTION_ROOT), env.get("PYTHONPATH", "")))
    # The smoke test covers the CPU training path and needs no accelerator.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


@pytest.fixture(scope="module")
def pose_annotations(tmp_path_factory):
    root = tmp_path_factory.mktemp("vipharpet")
    data_root = root / "input"
    data_root.mkdir()
    out_dir = root / "annotations"

    for split in ("train", "valid", "test"):
        image_dir = data_root / f"images_{split}"
        image_dir.mkdir()
        names = []
        poses = []
        for label, action in enumerate(ACTIONS):
            for frame_idx in FRAME_INDICES:
                name = f"ImageSequences{action}_{label + 1}_{frame_idx}.jpg"
                names.append(name)
                pose = np.full((18, 2), frame_idx, dtype=np.float64)
                pose[0] = (-10, 100)
                poses.append(pose)
                Image.new("RGB", IMAGE_SHAPE[::-1]).save(image_dir / name)

        # VIP-HARPET stores zero-padded ASCII image names as float arrays.
        encoded_names = np.zeros((len(names), max(map(len, names)) + 1), dtype=np.float64)
        for row, name in zip(encoded_names, names):
            row[: len(name)] = list(name.encode("ascii"))
        with h5py.File(data_root / f"annot_{split}.h5", "w") as annotations:
            annotations["imgname"] = encoded_names
            annotations["part"] = np.stack(poses)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_vip_harpet_pose.py"),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
        ],
        env=_subprocess_env(),
        check=True,
    )
    return out_dir


def should_prepare_vipharpet_pose_generates_pkls(pose_annotations):
    for split in ("train", "val", "test"):
        annotations = mmengine.load(pose_annotations / f"vipharpet_{split}.pkl")
        assert len(annotations) == len(ACTIONS)
        for label, sample in enumerate(annotations):
            assert sample["frame_dir"] == f"{ACTIONS[label]}_{label + 1}"
            assert sample["label"] == label
            assert sample["total_frames"] == len(FRAME_INDICES)
            assert sample["img_shape"] == IMAGE_SHAPE
            assert sample["keypoint"].shape == (1, 3, 18, 2)
            assert sample["keypoint"].dtype == np.float32
            assert sample["keypoint_score"].shape == (1, 3, 18)
            assert sample["keypoint_score"].dtype == np.float32
            np.testing.assert_array_equal(sample["keypoint_score"], 1)
            # Frames must be numerically sorted and coordinates clamped.
            np.testing.assert_array_equal(sample["keypoint"][0, :, 1, 0], (3, 11, 20))
            np.testing.assert_array_equal(sample["keypoint"][0, :, 0, 0], 0)
            np.testing.assert_array_equal(sample["keypoint"][0, :, 0, 1], IMAGE_SHAPE[0] - 1)


def should_load_training_config():
    from mmengine.config import Config

    cfg = Config.fromfile(CONFIG)
    assert cfg.model.type == "RecognizerGCN"
    assert cfg.train_dataloader["dataset"]["type"] == "PoseDataset"
    assert "UniformSampleFrames" in [t["type"] for t in cfg.train_pipeline]


def should_build_and_forward_minimal_model(pose_annotations, monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(MMACTION_ROOT))
    from mmaction.models import STGCN

    sample = mmengine.load(pose_annotations / "vipharpet_train.pkl")[0]
    # STGCN consumes (N, M, T, V, C), including confidence as channel 3.
    x = torch.from_numpy(sample["keypoint"]).unsqueeze(0)
    score = torch.from_numpy(sample["keypoint_score"]).unsqueeze(0).unsqueeze(-1)
    x = torch.cat([x, score], dim=-1)
    model = STGCN(
        graph_cfg=dict(layout="openpose", mode="stgcn_spatial"), in_channels=3, num_person=1
    ).eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 1, 18)
    assert torch.isfinite(y).all()


def should_train_and_test_entrypoints_smoke(pose_annotations, tmp_path):
    env = _subprocess_env()
    work = tmp_path / "train"
    loader_options = []
    for split in ("train", "val", "test"):
        loader_options.extend(
            [
                f"{split}_dataloader.dataset.ann_file={pose_annotations / f'vipharpet_{split}.pkl'}",
                f"{split}_dataloader.batch_size=4",
                f"{split}_dataloader.num_workers=0",
                f"{split}_dataloader.persistent_workers=False",
            ]
        )

    subprocess.run(
        [
            sys.executable,
            str(MMACTION_ROOT / "tools" / "train.py"),
            str(CONFIG),
            "--work-dir",
            str(work),
            "--seed",
            "0",
            "--cfg-options",
            "train_cfg.max_epochs=1",
            *loader_options,
        ],
        env=env,
        check=True,
    )

    ckpt = work / "epoch_1.pth"
    assert ckpt.is_file()
    subprocess.run(
        [
            sys.executable,
            str(MMACTION_ROOT / "tools" / "test.py"),
            str(CONFIG),
            str(ckpt),
            "--work-dir",
            str(tmp_path / "test"),
            "--cfg-options",
            *loader_options,
        ],
        env=env,
        check=True,
    )
