#!/usr/bin/env python3
"""Load a converted COCO dataset into FiftyOne and open the app on it.

Samples are built explicitly rather than through
``fo.types.COCODetectionDataset``. That importer resolves images by matching
the annotation ``file_name`` against a directory listing, and every sequence
in this dataset numbers its frames from ``000001.jpg`` -- 25 sequences, 25
files with the same basename, and no way to tell which one a label belongs
to. Building samples from the full relative path cannot get that wrong.

Doing it by hand also keeps the converter's extra fields, which is most of
why the app is worth opening: ``pseudo_label`` separates detector guesses
from what the dataset asserts, and ``team``, ``jersey``, ``track_id``,
``occluded`` and ``interpolated`` are all filterable in the sidebar. 88% of
this dataset's boxes are interpolated, so being able to isolate the 12% that
a human actually drew is the difference between spot-checking the conversion
and spot-checking the dataset.

Usage::

    python scripts/fiftyone_review.py --ann DST/annotations/val.json
    python scripts/fiftyone_review.py --ann DST/annotations/train.json --stride 10
"""

from __future__ import annotations

import argparse
import collections
import json
import socket
import sys
from pathlib import Path


def build(
    ann_path: Path,
    images_root: Path,
    stride: int,
    sequences: list[str] | None,
    name: str,
    *,
    replace: bool = False,
):
    import fiftyone as fo

    if stride < 1:
        raise SystemExit(f"--stride must be at least 1, got {stride}")
    if fo.dataset_exists(name) and not replace:
        raise SystemExit(
            f"FiftyOne dataset {name!r} already exists; choose a different --name or use "
            "--replace to delete its saved labels, tags, and views and rebuild it"
        )

    data = json.loads(ann_path.read_text())
    category = {c["id"]: c["name"] for c in data["categories"]}
    available = {image.get("sequence", ann_path.stem) for image in data["images"]}
    unknown = sorted(set(sequences or ()) - available)
    if unknown:
        raise SystemExit(f"{ann_path} has no sequence {', '.join(unknown)}")

    by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    samples = []
    counts: collections.Counter = collections.Counter()
    for index, image in enumerate(data["images"]):
        sequence = image.get("sequence", ann_path.stem)
        if sequences and sequence not in sequences:
            continue
        if index % stride:
            continue

        width, height = image["width"], image["height"]
        if width <= 0 or height <= 0:
            raise SystemExit(f"image {image['id']} has invalid dimensions {width}x{height}")
        path = images_root / image["file_name"]
        if not path.is_file():
            raise SystemExit(f"cannot read {path} (is --images the right root?)")
        detections = []
        for annotation in by_image.get(image["id"], []):
            x, y, w, h = annotation["bbox"]
            label = category.get(annotation["category_id"], str(annotation["category_id"]))
            pseudo = bool(annotation.get("pseudo_label", False))
            detections.append(
                fo.Detection(
                    label=label,
                    # FiftyOne wants [x, y, w, h] as fractions of the frame.
                    bounding_box=[x / width, y / height, w / width, h / height],
                    pseudo_label=pseudo,
                    # `confidence` rather than a plain field so the app's
                    # score slider filters the detector guesses directly.
                    confidence=annotation.get("detector_score") if pseudo else None,
                    iscrowd=bool(annotation.get("iscrowd", 0)),
                    track_id=annotation.get("track_id"),
                    team=annotation.get("team"),
                    jersey=annotation.get("jersey"),
                    occluded=annotation.get("occluded"),
                    interpolated=annotation.get("interpolated"),
                )
            )
            counts[("detector " if pseudo else "dataset ") + label] += 1

        sample = fo.Sample(filepath=str(path))
        sample["sequence"] = sequence
        sample["frame_index"] = image.get("frame_index", index)
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    if not samples:
        raise SystemExit(f"nothing to load from {ann_path} with the given --sequence/--stride")

    # Validate and construct the selection before replacing any saved review.
    if fo.dataset_exists(name):
        if not replace:
            raise SystemExit(f"FiftyOne dataset {name!r} already exists; choose a different --name")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples(samples)
    # Sorting in the app is by field, so the frame order has to be a field.
    dataset.create_index("sequence")
    dataset.create_index("frame_index")
    return dataset, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("/data/datasets/hockeyTrackingDataset"),
        help="root that image file_name entries are relative to",
    )
    parser.add_argument("--name", default=None, help="FiftyOne dataset name")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete an existing dataset's saved labels, tags, and views before rebuilding",
    )
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth image")
    parser.add_argument("--sequence", action="append", default=None, help="repeatable")
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument(
        "--address",
        default="0.0.0.0",
        help="bind address; the default is reachable from another machine",
    )
    parser.add_argument(
        "--no-serve", action="store_true", help="build the dataset and exit without the app"
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit(f"--stride must be at least 1, got {args.stride}")
    if not args.ann.is_file():
        raise SystemExit(f"{args.ann} not found")
    if not args.images.is_dir():
        raise SystemExit(f"--images {args.images} is not a directory")

    import fiftyone as fo

    name = args.name or f"{args.ann.parent.parent.name}-{args.ann.stem}"
    dataset, counts = build(
        args.ann, args.images, args.stride, args.sequence, name, replace=args.replace
    )

    print(f"\ndataset {name!r}: {len(dataset)} samples")
    for key, value in sorted(counts.items()):
        print(f"  {key:22s} {value:8d}")
    print(f"  {'total':22s} {sum(counts.values()):8d}")

    if args.no_serve:
        return 0

    session = fo.launch_app(dataset, address=args.address, port=args.port, remote=True)
    host = socket.gethostname()
    print(f"\n  http://{host}:{args.port}/")
    print(f"  http://localhost:{args.port}/   (if you are on this machine)")
    print("\nCtrl-C to stop.")
    session.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
