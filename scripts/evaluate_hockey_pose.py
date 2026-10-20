#!/usr/bin/env python3
"""Evaluate one frozen pose checkpoint and retain temporal inputs for action tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hockey_data import configure_native_crops, digest, load_dataset
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.utils import register_all_modules
from train_hockey_pose import HockeyPoseMetric


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_native_crops()
    torch.set_num_threads(4)
    register_all_modules()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(args.output)
    cfg.load_from = str(args.checkpoint)
    cfg.resume = False
    cfg.custom_hooks = []
    loader = cfg.val_dataloader
    loader.dataset.ann_file = str(args.config.parent / "data" / f"{args.split}.json")
    loader.dataset.pipeline = cfg.val_dataloader.dataset.pipeline
    cfg.val_dataloader = loader
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.eval()
    metric = HockeyPoseMetric()
    frames = {}
    with torch.inference_mode():
        for batch in runner.val_loop.dataloader:
            samples = runner.model.val_step(batch)
            dictionaries = [sample.to_dict() for sample in samples]
            metric.process(batch, dictionaries)
            for sample in samples:
                points = array(sample.pred_instances.keypoints)[0]
                confidence = np.clip(array(sample.pred_instances.keypoint_scores)[0], 0, 1)
                frames[Path(sample.img_path).name] = np.c_[points, confidence].astype(np.float32)
    metrics = metric.compute_metrics(metric.results)
    dataset, _ = load_dataset(args.data_root)
    sequences = {
        key: np.stack([frames[f["name"]] for f in triplet])
        for key, triplet in dataset[args.split].items()
    }
    np.savez_compressed(args.output / "predictions.npz", **sequences)
    metrics.update(
        checkpoint_sha256=digest(args.checkpoint),
        split=args.split,
        images=len(frames),
        sequences=len(sequences),
        config_sha256=digest(args.config),
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
