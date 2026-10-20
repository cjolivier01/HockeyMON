#!/usr/bin/env python3
"""Train hockey STGCN++ without changing or tuning against the held-out test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hockey_action import HockeyAction
from hockey_data import FLIP, LABELS, digest, load_dataset, software_provenance
from torch.nn import functional as F


def tensors(groups):
    if not groups:
        raise ValueError("Action training/evaluation requires a nonempty split")
    points = np.stack([np.stack([f["points"] for f in frames]) for frames in groups.values()])
    scores = np.ones((*points.shape[:-1], 1), dtype=np.float32)
    return (
        torch.from_numpy(np.concatenate((points, scores), axis=-1)[:, None]).cuda(),
        torch.tensor([frames[0]["label"] for frames in groups.values()], device="cuda"),
    )


def augment(data, noise):
    data = data.clone()
    xy = data[..., :2]
    flip = torch.rand(len(data), device=data.device) < 0.5
    data[flip] = data[flip][..., list(FLIP), :]
    data[flip, ..., 0] *= -1
    height = (xy[..., :16, 1].amax(-1) - xy[..., :16, 1].amin(-1))[..., None, None]
    xy += torch.randn_like(xy) * height * noise
    missing = torch.rand_like(data[..., 2]) < 0.04
    missing[..., 6] = False
    data[..., 2] = torch.where(missing, 0, torch.empty_like(data[..., 2]).uniform_(0.5, 1))
    return data


def evaluate(model, data, labels):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = F.cross_entropy(logits, labels).item()
        pred = logits.argmax(-1)
        matrix = torch.bincount(labels * 4 + pred, minlength=16).reshape(4, 4)
        recall = matrix.diag() / matrix.sum(1).clamp(min=1)
    return {
        "accuracy": float((pred == labels).float().mean()),
        "balanced_accuracy": float(recall.mean()),
        "loss": loss,
        "confusion_matrix": matrix.cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--noise", type=float, default=0.025)
    parser.add_argument("--pose-predictions", type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset, excluded = load_dataset(args.data_root)
    train, train_y = tensors(dataset["train"])
    valid, valid_y = tensors(dataset["valid"])
    if args.pose_predictions:
        with np.load(args.pose_predictions) as predictions:
            valid = torch.tensor(
                np.stack([predictions[key] for key in dataset["valid"]])[:, None], device="cuda"
            )
    model = HockeyAction()
    pretrained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    target = model.state_dict()
    copied, fresh = [], []
    for key in target:
        if key in pretrained and target[key].shape == pretrained[key].shape and ".A" not in key:
            target[key] = pretrained[key]
            copied.append(key)
        else:
            fresh.append(key)
    model.load_state_dict(target)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.05
    )
    weights = len(train_y) / (4 * torch.bincount(train_y, minlength=4).float())
    provenance = {
        "source_sha256": digest(args.checkpoint),
        "copied": copied,
        "fresh": fresh,
        "excluded_sequences": excluded,
        "seed": args.seed,
        "lr": args.lr,
        "noise": args.noise,
        "labels": LABELS,
        "selection": "validation balanced accuracy, then cross entropy",
        "pose_predictions": str(args.pose_predictions) if args.pose_predictions else None,
        "pose_predictions_sha256": digest(args.pose_predictions) if args.pose_predictions else None,
        "epochs": args.epochs,
        "patience": args.patience,
        "software": software_provenance(),
    }
    (args.work_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best, best_epoch = (-1, -float("inf")), 0
    with (args.work_dir / "metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            model.train()
            permutation = torch.randperm(len(train), device="cuda")
            losses = []
            for indices in permutation.split(32):
                optimizer.zero_grad(set_to_none=True)
                logits = model(augment(train[indices], args.noise))
                loss = F.cross_entropy(
                    logits, train_y[indices], weight=weights, label_smoothing=0.05
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            metrics = evaluate(model, valid, valid_y)
            metrics.update(epoch=epoch + 1, train_loss=float(np.mean(losses)))
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            score = (metrics["balanced_accuracy"], -metrics["loss"])
            if score > best:
                best, best_epoch = score, epoch
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "metrics": metrics,
                        "provenance": provenance,
                    },
                    args.work_dir / "best.pth",
                )
                print(json.dumps(metrics), flush=True)
            if epoch - best_epoch >= args.patience:
                print(f"Validation plateau: stopping at epoch {epoch + 1}", flush=True)
                break


if __name__ == "__main__":
    main()
