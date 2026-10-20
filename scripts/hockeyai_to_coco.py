"""Convert the HockeyAI dataset to two-class COCO for referee-aware detection.

HockeyAI (SimulaMet-HOST, MIT) is 2,101 Swedish Hockey League broadcast frames
in YOLO format with seven classes. Only the person-like ones matter here:

    3 goalie    1,281 instances
    4 player   13,660 instances
    6 referee   1,948 instances

Those collapse to two output classes:

    0 player   (goalie merged in)
    1 referee

Goalies merge into ``player`` rather than being dropped. Dropping them would
leave a visible person unlabelled in every frame, which teaches the detector to
*suppress* goalies -- they would vanish from tracking. Merging keeps today's
behaviour, where the single-class detector finds everyone, and adds only the
referee distinction. It also means no new class has to be plumbed downstream.

The non-person classes (centriod, faceoff, goal, puck) are discarded: they are
rink furniture, and this model's job is people.

Usage::

    python scripts/hockeyai_to_coco.py --src ~/datasets/hockey_ref/raw/SHL \\
        --dst ~/datasets/hockey_ref --val-fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

#: HockeyAI YOLO class id -> output class id. Anything absent is discarded.
CLASS_MAP: dict[int, int] = {
    3: 0,  # goalie -> player
    4: 0,  # player -> player
    6: 1,  # referee -> referee
}

#: Output categories, COCO ids are 1-based.
CATEGORIES = [
    {"id": 1, "name": "player", "supercategory": "person"},
    {"id": 2, "name": "referee", "supercategory": "person"},
]


def _read_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in CLASS_MAP:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        rows.append((CLASS_MAP[cls], cx, cy, w, h))
    return rows


def _build_split(
    stems: list[str],
    frames: Path,
    labels: Path,
    image_dir_name: str,
) -> tuple[dict, dict[str, int]]:
    images, annotations = [], []
    counts = {"player": 0, "referee": 0}
    ann_id = 1
    for image_id, stem in enumerate(sorted(stems), start=1):
        image_path = next(frames.glob(f"{stem}.*"))
        with Image.open(image_path) as handle:
            width, height = handle.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for cls, cx, cy, w, h in _read_yolo(labels / f"{stem}.txt"):
            bw, bh = w * width, h * height
            x, y = (cx * width) - bw / 2.0, (cy * height) - bh / 2.0
            # Clamp: a few HockeyAI boxes run a pixel or two past the edge.
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, width - x), min(bh, height - y)
            if bw <= 1 or bh <= 1:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            counts["player" if cls == 0 else "referee"] += 1
            ann_id += 1

    document = {
        "info": {"description": f"HockeyAI two-class ({image_dir_name})"},
        "licenses": [{"id": 1, "name": "MIT"}],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    return document, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, type=Path, help="HockeyAI SHL directory")
    parser.add_argument("--dst", required=True, type=Path, help="output dataset root")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--group-by",
        choices=("clip", "none"),
        default="clip",
        help=(
            "Hold out whole source clips rather than individual frames. Frames extracted "
            "from broadcast video are near-duplicates of their neighbours, so a frame-level "
            "split inflates validation scores. `none` opts out explicitly."
        ),
    )
    args = parser.parse_args()

    frames = args.src / "frames"
    labels = args.src / "annotations"
    if not frames.is_dir() or not labels.is_dir():
        raise SystemExit(f"expected {frames} and {labels} to exist")

    stems = sorted(p.stem for p in labels.glob("*.txt"))
    if not stems:
        raise SystemExit(f"no YOLO .txt annotations under {labels}")

    # Split by source clip, not by frame. HockeyAI's 2101 frames are extracted
    # from broadcast video, so consecutive frames are near-duplicates: a
    # frame-level shuffle puts the same play on both sides of the boundary and
    # inflates validation scores by an unknown amount. Stems look like
    # `<clip>_<frame>`, so the clip is everything before the last underscore.
    if args.group_by == "none":
        groups = {stem: [stem] for stem in stems}
    else:
        groups = {}
        for stem in stems:
            clip = stem.rsplit("_", 1)[0] if "_" in stem else stem
            groups.setdefault(clip, []).append(stem)
    clips = sorted(groups)

    # Refuse to pretend. If the filename convention does not actually group --
    # nearly as many "clips" as frames -- the split is frame-level whatever
    # this says, and the operator should know rather than read a val score
    # inflated by near-duplicates on both sides.
    if args.group_by == "clip" and len(clips) > 0.5 * len(stems):
        raise SystemExit(
            f"{len(clips)} clip groups for {len(stems)} frames: the `<clip>_<frame>` convention "
            f"does not hold for {labels}. Pass --group-by none to accept a frame-level split, "
            "understanding that consecutive frames are near-duplicates and will leak across it."
        )
    if len(clips) < 2:
        raise SystemExit(f"only {len(clips)} group(s) in {labels}; nothing to hold out")

    # Deterministic so a re-run does not silently reshuffle and move clips
    # across the boundary between experiments.
    random.Random(args.seed).shuffle(clips)
    wanted_val = len(stems) * args.val_fraction
    val_clips: set[str] = set()
    held = 0
    for clip in clips:
        if held >= wanted_val or len(val_clips) == len(clips) - 1:
            break
        val_clips.add(clip)
        held += len(groups[clip])
    splits = {
        "train": [s for c in clips if c not in val_clips for s in groups[c]],
        "val": [s for c in clips if c in val_clips for s in groups[c]],
    }
    print(
        f"split {len(clips)} clips: {len(clips) - len(val_clips)} train "
        f"({len(splits['train'])} frames), {len(val_clips)} val ({len(splits['val'])} frames)"
    )

    ann_dir = args.dst / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for name, split_stems in splits.items():
        document, counts = _build_split(split_stems, frames, labels, name)
        out = ann_dir / f"{name}.json"
        out.write_text(json.dumps(document))
        print(
            f"{name:5s}: {len(document['images']):5d} images  "
            f"{counts['player']:6d} player  {counts['referee']:5d} referee  -> {out}"
        )
    print(f"\nimages stay at {frames} (point data_prefix there)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
