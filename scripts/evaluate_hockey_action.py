#!/usr/bin/env python3
"""Evaluate a frozen action checkpoint, including the native confidence gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hockey_action import HockeyAction
from hockey_data import LABELS, digest, load_dataset
from train_hockey_action import evaluate, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Evaluation output already exists: {args.output}")
    torch.set_num_threads(4)
    dataset, _ = load_dataset(args.data_root)
    truth, labels = tensors(dataset[args.split])
    with np.load(args.predictions) as predictions:
        estimated = torch.tensor(
            np.stack([predictions[key] for key in dataset[args.split]])[:, None], device="cuda"
        )
    model = HockeyAction().cuda().eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    )
    metrics = {
        "ground_truth_pose": evaluate(model, truth, labels),
        "estimated_pose": evaluate(model, estimated, labels),
        "split": args.split,
        "labels": LABELS,
        "checkpoint_sha256": digest(args.checkpoint),
        "predictions_sha256": digest(args.predictions),
    }
    with torch.no_grad():
        probability = model(estimated).softmax(-1)
        confidence, prediction = probability.max(-1)
        body_count = (estimated[:, 0, :, :16, 2] >= 0.3).sum(-1)
        eligible = ((body_count >= 8) & (estimated[:, 0, :, 6, 2] >= 0.3)).all(-1)
        accepted = eligible & (confidence >= 0.5)
        metrics["native_gates"] = {
            "total": len(labels),
            "eligible": int(eligible.sum()),
            "accepted": int(accepted.sum()),
            "correct_accepted": int(((prediction == labels) & accepted).sum()),
            "coverage": float(accepted.float().mean()),
        }
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
