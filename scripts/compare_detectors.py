#!/usr/bin/env python3
"""Compare a hockey-fine-tuned detector against the CrowdHuman model it came from.

`coco/bbox_mAP` from the training run cannot answer this. The new head has
four classes and the old one has a single ``person``, so their mAPs are not
comparable numbers -- and the question that matters is not whether the new
model names roles well, it is whether it still *finds people* as reliably as
the model currently in production. So both are scored class-agnostically:
every prediction and every ground-truth box collapses to one ``person``
class, and the two models are measured on identical images with identical
geometry.

Ground truth is the human-annotated boxes only. The ``spectator`` boxes in
the converted dataset are pseudo-labels produced *by* the CrowdHuman model,
so scoring against them would mark the old model correct by construction and
the comparison would be meaningless.

Both models are served at the graph's 1632x608 rather than the 1408x544 they
were trained at, because that is what `tracking.yaml` runs and the deployed
number is the one worth knowing. YOLOv8 is fully convolutional, and the
baseline was trained at 1408x544 and deployed at 1632x608 too, so neither
model is favoured.

The new model is evaluated from its EMA weights, which is what mmengine's
`EMAHook` swaps in for validation.

Usage::

    python scripts/compare_detectors.py --new openmm/work_dirs/.../epoch_10.pth
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hockey_tracking_to_coco import build_transforms, detect

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


def build_finetuned(config_path: Path, checkpoint: Path, device: str, use_ema: bool):
    import mmdet.models  # noqa: F401
    import mmyolo.models  # noqa: F401
    import torch
    from mmengine.config import Config
    from mmengine.registry import MODELS, init_default_scope

    init_default_scope("mmyolo")
    # Parsed from the openmm root. The config resolves `data/crowdhuman/`
    # relative to the cwd, as every config in that tree does, and it reads
    # the file at parse time to size the mix -- so parsing it from anywhere
    # else raises even though only `cfg.model` is wanted here.
    config_path = config_path.resolve()
    with contextlib.chdir(config_path.parents[2]):
        cfg = Config.fromfile(str(config_path))
    model = MODELS.build(cfg.model)

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if use_ema and "ema_state_dict" in blob:
        # EMAHook stores the shadow under `module.`, and carries its own
        # bookkeeping buffers that the bare model has no slot for.
        state = {
            k[len("module.") :]: v
            for k, v in blob["ema_state_dict"].items()
            if k.startswith("module.")
        }
        source = "ema_state_dict"
    else:
        state, source = blob["state_dict"], "state_dict"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{checkpoint} {source}: missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    print(f"fine-tuned: {checkpoint.name} [{source}], {cfg.num_classes} classes")
    return model.to(device).eval(), cfg.metainfo["classes"]


def coco_eval(gt_images, gt_by_image, predictions, label):
    """Class-agnostic COCO bbox eval. Returns the 12-value stats array."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = {
        "images": [{"id": i["id"], "width": i["width"], "height": i["height"]} for i in gt_images],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }
    next_id = 1
    for image in gt_images:
        for annotation in gt_by_image.get(image["id"], []):
            x, y, w, h = annotation["bbox"]
            gt["annotations"].append(
                {
                    "id": next_id,
                    "image_id": image["id"],
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            next_id += 1

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions) if predictions else None
        if coco_dt is None:
            return None
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.stats


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
        default=Path("openmm/configs/hm/hm_hockey_tracking_yolov8_m_1408_544.py"),
    )
    parser.add_argument("--graph", default="hmlib/config/aspen/tracking.yaml")
    parser.add_argument("--count", type=int, default=600, help="val images to score")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=0.001, help="floor for AP curves")
    parser.add_argument(
        "--report-thr", type=float, default=0.3, help="threshold for the raw counts"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-weights", action="store_true", help="use state_dict, not the EMA")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.08,
        help="hard cap on this process's share of the GPU, so a training run "
        "sharing the card cannot be starved. 0 disables the cap.",
    )
    args = parser.parse_args()

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
    for label, model, classes in (
        ("baseline (CrowdHuman, 1-class)", build_baseline(args.graph, args.device), None),
        (
            "fine-tuned (hockey, 4-class)",
            *build_finetuned(args.new_config, args.new, args.device, not args.raw_weights),
        ),
    ):
        predictions = []
        per_class: collections.Counter = collections.Counter()
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
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": d["score"],
                        }
                    )
                    if classes and d["score"] >= args.report_thr:
                        per_class[classes[d["label"]]] += 1
        stats = coco_eval(images, gt_by_image, predictions, label)
        kept = sum(1 for p in predictions if p["score"] >= args.report_thr)
        results[label] = (stats, kept, per_class)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'':34s} {'AP':>7s} {'AP50':>7s} {'AP75':>7s} {'AR@100':>7s} {'APsmall':>8s}")
    for label, (stats, kept, _) in results.items():
        if stats is None:
            print(f"{label:34s}  no detections")
            continue
        print(
            f"{label:34s} {stats[0]:7.3f} {stats[1]:7.3f} {stats[2]:7.3f} {stats[8]:7.3f} {stats[3]:8.3f}"
        )
    base, new = (results[k][0] for k in results)
    if base is not None and new is not None:
        print(
            f"\n{'delta (fine-tuned - baseline)':34s} {new[0] - base[0]:+7.3f} "
            f"{new[1] - base[1]:+7.3f} {new[2] - base[2]:+7.3f} {new[8] - base[8]:+7.3f} "
            f"{new[3] - base[3]:+8.3f}"
        )

    print(f"\nboxes kept at score >= {args.report_thr}:")
    for label, (_, kept, per_class) in results.items():
        detail = "  ".join(f"{k}={v}" for k, v in per_class.most_common())
        print(f"  {label:34s} {kept:6d}   {detail}")
    print(f"  {'annotated on-ice ground truth':34s} {boxes:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
