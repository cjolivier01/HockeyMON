#!/usr/bin/env python3
"""Fine-tune RTMPose on checked HARPET splits with deployment-matched crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hockey_data import METAINFO, configure_native_crops, digest, prepare, software_provenance
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.runner import Runner
from mmpose.registry import METRICS, MODELS
from mmpose.utils import register_all_modules


@METRICS.register_module()
class HockeyPoseMetric(BaseMetric):
    """Head-normalized accuracy, including separately reported stick endpoints."""

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            truth = sample["gt_instances"]["keypoints"]
            predicted = sample["pred_instances"]["keypoints"]
            if isinstance(truth, torch.Tensor):
                truth = truth.detach().cpu().numpy()
            if isinstance(predicted, torch.Tensor):
                predicted = predicted.detach().cpu().numpy()
            head = np.linalg.norm(truth[:, 9] - truth[:, 8], axis=-1)
            if np.any(head <= 0):
                raise ValueError("Zero-length head segment in evaluation")
            error = np.linalg.norm(predicted - truth, axis=-1)
            self.results.append((error / head[:, None], error))

    def compute_metrics(self, results):
        normalized = np.concatenate([r[0] for r in results])
        pixels = np.concatenate([r[1] for r in results])
        correct = normalized < 0.5
        return {
            "PCKh": float(correct.mean()),
            "body_PCKh": float(correct[:, :16].mean()),
            "stick_top_PCKh": float(correct[:, 16].mean()),
            "stick_tip_PCKh": float(correct[:, 17].mean()),
            "EPE": float(pixels.mean()),
        }


def make_config(args):
    repo = Path(__file__).resolve().parents[1]
    name = "rtmpose-m_8xb64-crowdpose_harpe-256x192.py"
    cfg = Config.fromfile(repo / "openmm/mmpose/configs/body_2d_keypoint/rtmpose/harpe" / name)
    cfg.model.backbone.norm_cfg = {"type": "BN"}
    cfg.model.test_cfg.flip_test = False
    cfg.randomness = {"seed": args.seed}
    cfg.work_dir = str(args.work_dir)
    cfg.load_from = str(args.work_dir / "initial.pth")
    cfg.train_cfg = {"by_epoch": True, "max_epochs": args.epochs, "val_interval": 1}
    cfg.optim_wrapper = {
        "type": "AmpOptimWrapper",
        "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 0.01},
        "clip_grad": {"max_norm": 1.0, "norm_type": 2},
        "paramwise_cfg": {"norm_decay_mult": 0, "bias_decay_mult": 0, "bypass_duplicate": True},
    }
    cfg.param_scheduler = [
        {"type": "LinearLR", "start_factor": 0.1, "by_epoch": True, "begin": 0, "end": 5},
        {
            "type": "CosineAnnealingLR",
            "eta_min": args.lr * 0.05,
            "by_epoch": True,
            "begin": 5,
            "end": args.epochs,
            "T_max": args.epochs - 5,
        },
    ]
    codec = cfg.codec
    prefix = [{"type": "LoadImage"}, {"type": "GetBBoxCenterScale", "padding": args.padding}]
    end = [{"type": "TopdownAffine", "input_size": codec.input_size}]
    train_pipeline = (
        prefix
        + [
            {"type": "RandomFlip", "direction": "horizontal"},
            {
                "type": "RandomBBoxTransform",
                "shift_factor": 0.03 if args.refine else 0.08,
                "shift_prob": 0.5,
                "scale_factor": [0.95, 1.05] if args.refine else [0.85, 1.15],
                "rotate_factor": 10 if args.refine else 25,
            },
        ]
        + end
        + [
            {"type": "mmdet.YOLOXHSVRandomAug"},
            {"type": "GenerateTarget", "encoder": codec},
            {"type": "PackPoseInputs"},
        ]
    )
    val_pipeline = prefix + end + [{"type": "PackPoseInputs"}]
    for split, key in (
        ("train", "train_dataloader"),
        ("valid", "val_dataloader"),
        ("test", "test_dataloader"),
    ):
        cfg[key] = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "persistent_workers": args.workers > 0,
            "sampler": {"type": "DefaultSampler", "shuffle": split == "train"},
            "dataset": {
                "type": "CocoDataset",
                "data_mode": "topdown",
                "metainfo": METAINFO,
                "ann_file": str(args.work_dir / "data" / f"{split}.json"),
                "data_prefix": {"img": ""},
                "test_mode": split != "train",
                "pipeline": train_pipeline if split == "train" else val_pipeline,
            },
        }
    cfg.val_evaluator = {"type": "HockeyPoseMetric"}
    cfg.test_evaluator = {"type": "HockeyPoseMetric"}
    cfg.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "interval": 10,
        "save_best": "PCKh",
        "rule": "greater",
        "max_keep_ckpts": 2,
    }
    cfg.default_hooks.logger.interval = 15
    cfg.custom_hooks = [
        {
            "type": "EarlyStoppingHook",
            "monitor": "PCKh",
            "rule": "greater",
            "patience": args.patience,
            "min_delta": 0.0005,
        }
    ]
    cfg.env_cfg = {
        "cudnn_benchmark": True,
        "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
        "dist_cfg": {"backend": "nccl"},
    }
    return cfg


def initial_checkpoint(cfg, path, destination):
    model = MODELS.build(cfg.model)
    model.init_weights()
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    target = model.state_dict()
    # Copy actual shared anatomy from COCO to MPII order; new stick filters remain random.
    mapping = {0: 16, 1: 14, 2: 12, 3: 11, 4: 13, 5: 15, 10: 10, 11: 8, 12: 6, 13: 5, 14: 7, 15: 9}
    copied, remapped, fresh = [], [], []
    for key, value in target.items():
        if key not in state:
            fresh.append(key)
        elif state[key].shape == value.shape:
            target[key] = state[key]
            copied.append(key)
        elif key.startswith("head.final_layer.") and state[key].shape[0] == 17:
            for dest, src in mapping.items():
                value[dest] = state[key][src]
            for dest, sources in {6: (11, 12), 7: (5, 6), 8: (5, 6), 9: (0,)}.items():
                value[dest] = state[key][list(sources)].mean(0)
            remapped.append(key)
        else:
            raise ValueError(
                f"Unaccounted pretrained shape: {key}: {state[key].shape} -> {value.shape}"
            )
    torch.save({"state_dict": target, "meta": {"dataset_meta": METAINFO}}, destination)
    return {"copied": copied, "remapped": remapped, "new": fresh, "source_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--padding", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--refine", action="store_true", help="Reduce geometry augmentation for final refinement"
    )
    args = parser.parse_args()
    configure_native_crops()
    if args.work_dir.exists():
        raise ValueError("Choose a new work directory; existing runs are immutable")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    register_all_modules()
    report = prepare(args.data_root.resolve(), args.work_dir / "data")
    if any(not split["images"] for split in report["splits"].values()):
        raise ValueError("Training requires nonempty train, validation and test splits")
    cfg = make_config(args)
    provenance = initial_checkpoint(cfg, args.checkpoint, args.work_dir / "initial.pth")
    provenance.update(
        padding=args.padding,
        seed=args.seed,
        source=str(args.checkpoint),
        torch=torch.__version__,
        selection="validation PCKh@0.5; no test evaluation",
        software=software_provenance(),
    )
    (args.work_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    cfg.dump(str(args.work_dir / "config.py"))
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
