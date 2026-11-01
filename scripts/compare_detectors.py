#!/usr/bin/env python3
"""Compare a hockey-fine-tuned detector against the CrowdHuman model it came from.

`coco/bbox_mAP` from the training run cannot answer this. The new head has
four classes and the old one has a single ``person``, so their mAPs are not
comparable numbers -- and the question that matters is not whether the new
model names roles well, it is whether it still *finds people* as reliably as
the model currently in production. Both are scored class-agnostically on
known-box recall: how many human-annotated on-ice boxes they recover at a
fixed score threshold and IoU. This measures coverage, not precision.

The dataset does not label spectators exhaustively. AP would count correct
spectator detections as false positives, while scoring against spectator
pseudo-labels would favor the baseline that generated them. Neither gives a
fair comparison. Unmatched detections are therefore not scored, and there
is no additional COCO top-100 truncation. False-positive rates and spectator
coverage need a separate, exhaustively human-labelled evaluation set.

Both models are served at the graph's 1632x608 because that is what
`tracking.yaml` runs and the deployed number is the one worth knowing. YOLOv8
is fully convolutional: the baseline was trained at 1408x544 and the new mixed
model at 1984x736, while both are compared at the same deployed resolution.

The new model is evaluated from its EMA weights, which is what mmengine's
`EMAHook` swaps in for validation.

Usage::

    python scripts/compare_detectors.py --new openmm/work_dirs/.../epoch_10.pth
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hockey_tracking_to_coco import build_transforms, detect, iou_matrix

#: Classes that are a person standing on the ice. `spectator` is excluded --
#: it is a pseudo-label, not an annotation.
ON_ICE = {"player", "goalie", "official"}


def sample_images(ann_path: Path, count: int, seed: int):
    """`count` val images spread evenly over the sequences, with their GT."""
    data = json.loads(ann_path.read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}

    by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("pseudo_label"):
            continue
        if names[annotation["category_id"]] in ON_ICE:
            by_image[annotation["image_id"]].append(annotation)

    per_sequence: dict[str, list[dict]] = collections.defaultdict(list)
    for image in data["images"]:
        per_sequence[image.get("sequence", "all")].append(image)

    rng = random.Random(seed)
    if not per_sequence:
        raise SystemExit(f"{ann_path} has no images")
    share = max(1, count // len(per_sequence))
    chosen: list[dict] = []
    for sequence in sorted(per_sequence):
        frames = per_sequence[sequence]
        chosen.extend(rng.sample(frames, min(share, len(frames))))
    chosen = chosen[:count]
    return chosen, by_image, names


def build_baseline(graph: str, device: str):
    """The deployed CrowdHuman detector, through the pipeline's own factory."""
    import argparse as _argparse

    from scripts.hockey_tracking_to_coco import build_detector

    return build_detector(
        _argparse.Namespace(
            no_detector=False,
            graph=graph,
            detector_config=None,
            detector_model="trained",
            detector=None,
            device=device,
        )
    )


def load_model_config(config_path: Path):
    """Parse relative training data paths without leaving the caller's directory."""
    from mmengine.config import Config

    # Parsed from the openmm root. The config resolves `data/crowdhuman/`
    # relative to the cwd, as every config in that tree does, and it reads
    # the file at parse time to size the mix -- so parsing it from anywhere
    # else raises even though only `cfg.model` is wanted here.
    config_path = config_path.resolve()
    previous = Path.cwd()
    try:
        os.chdir(config_path.parents[2])
        return Config.fromfile(str(config_path), import_custom_modules=False)
    finally:
        os.chdir(previous)


def build_finetuned(config_path: Path, checkpoint: Path, device: str, use_ema: bool):
    import mmdet.models  # noqa: F401
    import mmyolo.models  # noqa: F401
    import torch
    from mmengine.registry import MODELS, init_default_scope

    init_default_scope("mmyolo")
    cfg = load_model_config(config_path)
    # Roles are mutually exclusive. Suppress across roles before collapsing
    # to person, just as the deployed multi-class detector does.
    cfg.model.test_cfg.multi_label = False
    cfg.model.test_cfg.nms.class_agnostic = True
    model = MODELS.build(cfg.model)

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state, source = checkpoint_weights(blob, use_ema)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{checkpoint} {source}: missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    print(f"fine-tuned: {checkpoint.name} [{source}], {cfg.num_classes} classes")
    return model.to(device).eval(), cfg.metainfo["classes"]


def checkpoint_weights(blob, use_ema: bool):
    """Select weights from an MMEngine EMAHook checkpoint.

    before_save_checkpoint swaps EMA into state_dict for deployment. The
    module-prefixed ema_state_dict holds the raw weights for resuming.
    """
    if use_ema:
        return blob["state_dict"], "state_dict (deployment/EMA when saved by EMAHook)"
    if "ema_state_dict" not in blob:
        raise SystemExit("--raw-weights requires an EMAHook checkpoint with ema_state_dict")
    state = {
        k[len("module.") :]: v for k, v in blob["ema_state_dict"].items() if k.startswith("module.")
    }
    return state, "ema_state_dict (raw training weights)"


def known_box_recall(gt_images, gt_by_image, predictions, score_thr):
    """Mean recall over IoU .50:.05:.95, plus R50/R75, with one-to-one matches.

    Unmatched predictions cannot be classified as false positives with this
    partially labelled dataset. Keep all predictions above the score floor;
    additional crowd boxes must not displace annotated skaters via maxDets.
    """
    by_image = collections.defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= score_thr:
            by_image[prediction["image_id"]].append(prediction)

    def xyxy(box):
        x, y, w, h = box["bbox"]
        return x, y, x + w, y + h

    thresholds = [0.5 + 0.05 * i for i in range(10)]
    hits = [0] * len(thresholds)
    total = 0
    for image in gt_images:
        annotations = gt_by_image.get(image["id"], [])
        total += len(annotations)
        found = sorted(by_image[image["id"]], key=lambda p: p["score"], reverse=True)
        overlaps = iou_matrix([xyxy(p) for p in found], [xyxy(a) for a in annotations])
        for index, threshold in enumerate(thresholds):
            available = set(range(len(annotations)))
            for row in overlaps:
                if not available:
                    break
                best = max(available, key=lambda j: row[j])
                if row[best] >= threshold:
                    available.remove(best)
                    hits[index] += 1
    if not total:
        return None
    recalls = [hit / total for hit in hits]
    return sum(recalls) / len(recalls), recalls[0], recalls[5]


def evaluate_model(model, classes, images, gt_by_image, transforms, args):
    predictions = []
    per_class: collections.Counter = collections.Counter()
    # Match score/NMS limits for both models. No evaluator top-100 limit is
    # applied; the detector's normal 30k pre-NMS candidate cap still applies.
    model.bbox_head.test_cfg.update(
        score_thr=args.score_thr,
        multi_label=False,
        nms_pre=30000,
        max_per_img=30000,
        nms={"type": "nms", "iou_threshold": 0.7, "class_agnostic": True},
    )
    for start in range(0, len(images), args.batch_size):
        chunk = images[start : start + args.batch_size]
        found = detect(
            model,
            transforms,
            [args.images / i["file_name"] for i in chunk],
            args.score_thr,
            with_labels=True,
        )
        for image, detections in zip(chunk, found):
            for d in detections:
                x1, y1, x2, y2 = d["xyxy"]
                predictions.append(
                    {
                        "image_id": image["id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": d["score"],
                    }
                )
                if classes and d["score"] >= args.report_thr:
                    per_class[classes[d["label"]]] += 1
    stats = known_box_recall(images, gt_by_image, predictions, args.report_thr)
    kept = sum(1 for p in predictions if p["score"] >= args.report_thr)
    return stats, kept, per_class


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ann",
        type=Path,
        default=Path("/data/datasets/hockeyTrackingDataset-coco/annotations/val.json"),
    )
    parser.add_argument("--images", type=Path, default=Path("/data/datasets/hockeyTrackingDataset"))
    parser.add_argument("--new", required=True, type=Path, help="fine-tuned checkpoint")
    parser.add_argument(
        "--new-config",
        type=Path,
        default=Path("openmm/configs/hm/hm_hockey_tracking_yolov8_m_1984_736.py"),
    )
    parser.add_argument("--graph", default="hmlib/config/aspen/tracking.yaml")
    parser.add_argument("--count", type=int, default=600, help="val images to score")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=0.001, help="inference confidence floor")
    parser.add_argument(
        "--report-thr", type=float, default=0.3, help="confidence threshold for recall and counts"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="use raw training weights saved in ema_state_dict",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.08,
        help="hard cap on this process's share of the GPU, so a training run "
        "sharing the card cannot be starved. 0 disables the cap.",
    )
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        parser.error("--count and --batch-size must be positive")
    if not 0 <= args.score_thr <= args.report_thr <= 1:
        parser.error("require 0 <= --score-thr <= --report-thr <= 1")
    if not 0 <= args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in [0,1]")

    if args.device.startswith("cuda") and args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device))
        total = torch.cuda.get_device_properties(args.device).total_memory / 2**20
        print(
            f"capped at {args.memory_fraction:.0%} of the GPU ({total * args.memory_fraction:.0f} MiB)"
        )

    images, gt_by_image, _ = sample_images(args.ann, args.count, args.seed)
    boxes = sum(len(gt_by_image.get(i["id"], [])) for i in images)
    sequences = len({i.get("sequence") for i in images})
    print(f"{len(images)} val images from {sequences} sequences, {boxes} annotated on-ice boxes\n")

    transforms, scale = build_transforms(args.graph)
    print(f"both models served at {scale[0]}x{scale[1]}\n")

    results = {}
    # Factories, not models: do not retain both sets of GPU weights in the
    # loop's iterator. Release the baseline before loading the fine-tune.
    for label, factory in (
        ("baseline (CrowdHuman, 1-class)", lambda: (build_baseline(args.graph, args.device), None)),
        (
            "fine-tuned (hockey)",
            lambda: build_finetuned(args.new_config, args.new, args.device, not args.raw_weights),
        ),
    ):
        model, classes = factory()
        try:
            results[label] = evaluate_model(model, classes, images, gt_by_image, transforms, args)
        finally:
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\nKnown-box recall at score >= {args.report_thr}; precision is not measured.")
    print(f"{'':34s} {'R50:95':>8s} {'R50':>7s} {'R75':>7s}")
    for label, (stats, _, _) in results.items():
        if stats is None:
            print(f"{label:34s}  no human-annotated boxes")
            continue
        print(f"{label:34s} {stats[0]:8.3f} {stats[1]:7.3f} {stats[2]:7.3f}")
    base, new = (results[k][0] for k in results)
    if base is not None and new is not None:
        print(
            f"\n{'delta (fine-tuned - baseline)':34s} {new[0] - base[0]:+8.3f} "
            f"{new[1] - base[1]:+7.3f} {new[2] - base[2]:+7.3f}"
        )

    print(f"\nboxes kept at score >= {args.report_thr}:")
    for label, (_, kept, per_class) in results.items():
        detail = "  ".join(f"{k}={v}" for k, v in per_class.most_common())
        print(f"  {label:34s} {kept:6d}   {detail}")
    print(f"  {'annotated on-ice ground truth':34s} {boxes:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
