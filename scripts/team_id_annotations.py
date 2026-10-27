"""Read the labeled, spatially complete subset of Koshkina's York team-ID data."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "ref"}]
ROLE_IDS = {"1": 1, "2": 2, "3": 1}


def load_team_id(root: Path, val_games: list[str]):
    """Join gt.txt crop names to xyxy boxes; never guess missing roles/geometry.

    Completeness here means all *supplied* boxes are labeled, not a guarantee
    that the source detector found every visible person.
    """
    root = root.resolve()
    sequences = sorted(p for p in root.iterdir() if p.is_dir() and (p / "gt.txt").is_file())
    known = {p.name for p in sequences}
    if not sequences or set(val_games) - known:
        raise ValueError(f"Invalid team-ID root or validation games: {root}, {val_games}")
    splits = {
        split: {"images": [], "annotations": [], "categories": CATEGORIES}
        for split in ("train", "val")
    }
    report = {"class_mapping": ROLE_IDS, "sequences": {}}
    for directory in sequences:
        info = {"included_frames": 0, "included_boxes": 0, "excluded_frames": {}}
        report["sequences"][directory.name] = info
        box_path = directory / "bbox_data.txt"
        if not box_path.is_file():
            info["excluded_sequence_reason"] = (
                "No full bbox_data.txt; players_bbox_data.txt omits referees"
            )
            continue
        pattern = re.compile(rf"{re.escape(directory.name)}_(\d+)_\d+\.png")

        def frame_id(name, pattern=pattern, directory=directory):
            match = pattern.fullmatch(name)
            if match is None:
                raise ValueError(f"Unexpected crop name in {directory}: {name}")
            return int(match[1])

        labels = {}
        labeled_frames = defaultdict(set)
        for line in (directory / "gt.txt").read_text().splitlines():
            name, label = line.strip().rsplit(",", 1)
            if label not in ROLE_IDS or name in labels:
                raise ValueError(f"Invalid or duplicate role label: {directory}/{line}")
            labels[name] = label
            labeled_frames[frame_id(name)].add(name)
        boxes = defaultdict(dict)
        for line in box_path.read_text().splitlines():
            for record in filter(None, line.strip().split(";")):
                x1, y1, x2, y2, name = record.split(",")
                coords = tuple(map(float, (x1, y1, x2, y2)))
                if not all(math.isfinite(v) for v in coords):
                    raise ValueError(f"Nonfinite box: {record}")
                frame = frame_id(name)
                if name in boxes[frame]:
                    raise ValueError(f"Duplicate box: {name}")
                boxes[frame][name] = coords
        data = splits["val" if directory.name in val_games else "train"]
        for frame in sorted(boxes.keys() | labeled_frames.keys()):
            names = set(boxes[frame])
            if not names or names != labeled_frames[frame]:
                reason = (
                    "unlabeled_boxes_or_missing_geometry"
                    if labeled_frames[frame]
                    else "no_role_labels"
                )
                info["excluded_frames"][str(frame)] = reason
                continue
            relative = Path(directory.name) / "all_frames" / f"{directory.name}_{frame}.png"
            with Image.open(root / relative) as image:
                width, height = image.size
            clipped = {}
            for name, (x1, y1, x2, y2) in boxes[frame].items():
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(width), x2), min(float(height), y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Empty/outside box: {directory.name}/{name}")
                clipped[name] = [x1, y1, x2 - x1, y2 - y1]
            image_id = len(data["images"]) + 1
            data["images"].append(
                {
                    "id": image_id,
                    "file_name": str(relative),
                    "width": width,
                    "height": height,
                    "sequence": directory.name,
                    "frame_index": frame - 1,
                }
            )
            for name, box in clipped.items():
                data["annotations"].append(
                    {
                        "id": len(data["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": ROLE_IDS[labels[name]],
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "source_crop": name,
                    }
                )
            info["included_frames"] += 1
            info["included_boxes"] += len(clipped)
    for split, data in splits.items():
        if not data["images"]:
            raise ValueError(f"No fully labeled frames for {split}; inspect source labels/split")
    return splits, report
