#!/usr/bin/env python3
"""Rewrite CrowdHuman's COCO annotations into the hockey category space.

Mixing CrowdHuman into a hockey run needs both annotation files to agree on
the category block, and not merely on the class *names*. mmdet builds its
label mapping as::

    cat_ids  = coco.get_cat_ids(cat_names=metainfo['classes'])
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

-- it enumerates the categories it *finds*, so a file declaring only
``person`` gets label 0 whatever ``classes`` says, and label 0 is ``player``.
Every CrowdHuman pedestrian would arrive labelled as a hockey player, silently
and with no shape error to catch it. So the whole category block is copied
from the converted hockey annotations and every box is remapped onto one of
those classes.

``spectator`` is the right target. In the hierarchy the converter writes, it
means "a person with no on-ice role", which is exactly what a CrowdHuman
pedestrian is -- and sourcing that class from 439k street-scene boxes rather
than from pseudo-labelled arena crowd is what stops it meaning "person in NHL
stands", which is the failure mode that would cost on-ice recall at a youth
rink.

``iscrowd`` is preserved: CrowdHuman's own group boxes stay ignore regions.

Usage::

    python scripts/crowdhuman_to_hockey_classes.py \\
        --src openmm/data/crowdhuman/annotations/train.json \\
        --categories-from /data/datasets/hockeyTrackingDataset-coco/annotations/train.json \\
        --dst openmm/data/crowdhuman/annotations/train_hockey_classes.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, type=Path, help="CrowdHuman COCO annotations")
    parser.add_argument(
        "--categories-from",
        required=True,
        type=Path,
        help="converted hockey annotations whose category block is authoritative",
    )
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument(
        "--as",
        dest="target",
        default="spectator",
        help="hockey class every CrowdHuman box becomes",
    )
    args = parser.parse_args()

    categories = json.loads(args.categories_from.read_text())["categories"]
    by_name = {c["name"]: c["id"] for c in categories}
    if args.target not in by_name:
        raise SystemExit(
            f"{args.categories_from} declares {sorted(by_name)}, which does not include "
            f"{args.target!r}. Convert with --crowd-mode spectator, or pick a class that "
            "exists -- mapping onto a category the hockey file never declares would put "
            "every CrowdHuman box on a label the head cannot reach."
        )
    target_id = by_name[args.target]

    data = json.loads(args.src.read_text())
    source_names = {c["id"]: c["name"] for c in data["categories"]}
    was = collections.Counter(source_names[a["category_id"]] for a in data["annotations"])

    data["categories"] = categories
    for annotation in data["annotations"]:
        annotation["category_id"] = target_id

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(data))

    ignored = sum(1 for a in data["annotations"] if a["iscrowd"])
    print(f"{len(data['images'])} images, {len(data['annotations'])} boxes")
    print(f"  {dict(was)} -> {args.target} (id {target_id})")
    print(f"  {ignored} kept as ignore regions (iscrowd=1)")
    print(f"  categories now {[c['name'] for c in categories]}")
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
