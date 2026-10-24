#!/usr/bin/env python3
"""Export COCO frames with only ice and annotated on-ice boxes preserved.

Writes a separate dataset; never modifies the source. Spectator annotations
are removed. A missing rink prediction fails the export rather than silently
creating an almost entirely gray training image. Output must be a new directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paint_background(image, rink_mask, annotations, gray=128):
    """Union the binary ice mask with boxes, without allocating an alpha image."""
    keep = np.array(rink_mask, dtype=bool, copy=True)
    height, width = image.shape[:2]
    if keep.shape != (height, width):
        raise ValueError(f"Mask shape {keep.shape} differs from image {(height, width)}")
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid box: {annotation['bbox']}")
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
        if right > left and bottom > top:
            keep[top:bottom, left:right] = True
    output = image.copy()
    output[~keep] = gray
    return output


def select_annotations(data, limit=0):
    """Keep original IDs and attributes, dropping the spectator category/boxes."""
    categories = data["categories"]
    allowed = {"player", "goalie", "official", "keeper", "ref"}
    unknown = {c["name"] for c in categories} - allowed - {"spectator"}
    if unknown:
        raise ValueError(f"Unexpected categories: {sorted(unknown)}")
    result = copy.deepcopy(data)
    result["categories"] = [c for c in categories if c["name"] in allowed]
    kept = {c["id"] for c in result["categories"]}
    result["images"] = data["images"][:limit] if limit else data["images"]
    image_ids = {im["id"] for im in result["images"]}
    result["annotations"] = [
        a for a in data["annotations"] if a["category_id"] in kept and a["image_id"] in image_ids
    ]
    return result


def read_frame(item, image_root):
    path = Path(item["file_name"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected relative image path: {path}")
    image = cv2.imread(str(image_root / path))
    if image is None:
        raise ValueError(f"Cannot read {image_root / path}")
    if image.shape[:2] != (item["height"], item["width"]):
        raise ValueError(f"Image dimensions disagree with COCO: {path}")
    return image


def write_frame(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Frames per split; 0 means all")
    parser.add_argument("--gray", type=int, default=128)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "hmlib/config/models/ice_rink/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path.home() / ".cache/torch/hub/checkpoints/ice_rink_iter_60000.pth"),
    )
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.cpu_threads) < 1 or args.limit < 0:
        parser.error("Batch size, workers and threads must be positive; limit nonnegative")
    if not 0 <= args.gray <= 255 or not 0 <= args.score_threshold <= 1:
        parser.error("Gray must be in [0,255]; score threshold in [0,1]")
    if len({p.name for p in args.annotations}) != len(args.annotations):
        parser.error("Annotation files must have distinct basenames")
    if args.output.exists():
        parser.error("Output directory already exists; use a new directory")

    import torch
    from mmcv.transforms import Compose
    from mmdet.apis import init_detector
    from mmdet.utils import get_test_pipeline_cfg

    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    model = init_detector(str(args.config), args.checkpoint, device=args.device)
    classes = model.dataset_meta["classes"]
    rink_labels = [i for i, name in enumerate(classes) if "rink" in name.lower()]
    if len(rink_labels) != 1:
        raise ValueError(f"Expected one rink class, got {classes}")
    pipeline_cfg = copy.deepcopy(get_test_pipeline_cfg(model.cfg))
    pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
    pipeline_cfg = [p for p in pipeline_cfg if p["type"] != "LoadAnnotations"]
    pipeline = Compose(pipeline_cfg)
    args.output.mkdir(parents=True)
    # A failed export never has a completion manifest; annotations publish per split.
    metadata = {k: str(v) for k, v in vars(args).items()}
    metadata["annotations"] = [str(p.resolve()) for p in args.annotations]
    metadata["rule"] = "preserve ice OR on-ice annotation box; remove spectator annotations"
    (args.output / "export-settings.json").write_text(json.dumps(metadata, indent=2))
    start = time.perf_counter()
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for annotation_path in args.annotations:
            data = select_annotations(json.loads(annotation_path.read_text()), args.limit)
            by_image = defaultdict(list)
            for annotation in data["annotations"]:
                by_image[annotation["image_id"]].append(annotation)
            records = data["images"]
            for offset in range(0, len(records), args.batch_size):
                batch = records[offset : offset + args.batch_size]
                images = list(pool.map(lambda item: read_frame(item, args.image_root), batch))
                packed = [
                    pipeline({"img": img, "img_id": item["id"]}) for item, img in zip(batch, images)
                ]
                # inference_detector(list) loops over single images; test_step is a real batch.
                with torch.inference_mode():
                    predictions = model.test_step(
                        {
                            "inputs": [p["inputs"] for p in packed],
                            "data_samples": [p["data_samples"] for p in packed],
                        }
                    )
                writes = []
                for item, image, prediction in zip(batch, images, predictions, strict=True):
                    instances = prediction.pred_instances
                    selected = (instances.labels == rink_labels[0]) & (
                        instances.scores > args.score_threshold
                    )
                    mask = instances.masks[selected].any(dim=0).cpu().numpy()
                    if not mask.any():
                        raise RuntimeError(f"No ice mask: {item['file_name']}; export incomplete")
                    painted = paint_background(image, mask, by_image[item["id"]], args.gray)
                    # PNG preserves protected pixels and the exact gray value losslessly.
                    relative = Path("images") / Path(item["file_name"]).with_suffix(".png")
                    writes.append(pool.submit(write_frame, args.output / relative, painted))
                    item["file_name"] = str(relative)
                for future in writes:
                    future.result()
                total += len(batch)
                print(
                    f"{annotation_path.name}: {offset + len(batch)}/{len(records)}; "
                    f"{total / (time.perf_counter() - start):.2f} frames/s",
                    flush=True,
                )
            destination = args.output / "annotations" / annotation_path.name
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(json.dumps(data))
    (args.output / "COMPLETE.json").write_text(json.dumps({"images": total}))


if __name__ == "__main__":
    main()
