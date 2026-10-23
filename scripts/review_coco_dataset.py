#!/usr/bin/env python3
"""Render a converted COCO dataset back onto its frames, to look at.

Built for `hockey_tracking_to_coco.py` output, but it will read any COCO
detection file whose images carry the ``sequence`` and ``frame_index`` keys
that converter writes; without them every image is treated as one sequence in
file order.

Video by default rather than stills, because the things worth catching here
are temporal. A box that is 10 px off looks fine in isolation and obviously
wrong when it lags the skater across a frame; an interpolated CVAT track that
has drifted away from its player only shows up while the player moves. 88% of
this dataset's boxes are interpolated, so that matters.

Annotations and pseudo-labels are drawn differently on purpose -- solid and
labelled for what the dataset asserts, thin and scored for what our detector
guessed -- because the question being asked of this output is usually "is the
crowd filled in sensibly", and that is only answerable if you can see which
boxes came from where.

Examples::

    # one mp4 per sequence
    python scripts/review_coco_dataset.py --ann DST/annotations/val.json --out /tmp/review

    # one sequence, every frame, faster to eyeball
    python scripts/review_coco_dataset.py --ann DST/annotations/train.json \\
        --sequence CAR_VS_BOS_2019_002 --out /tmp/review

    # stills instead, for flicking through over ssh
    python scripts/review_coco_dataset.py --ann DST/annotations/val.json \\
        --stills 40 --out /tmp/review
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: BGR per class. Deliberately far apart in hue: the point is to tell three
#: on-ice roles apart at 1280x720 while the video is moving.
COLORS = {
    "player": (220, 60, 60),
    "goalie": (0, 160, 240),
    "official": (240, 240, 0),
    "spectator": (140, 140, 140),
}
FALLBACK_COLOR = (200, 200, 200)


def load(ann_path: Path) -> tuple[dict[int, str], dict[str, list[dict[str, Any]]]]:
    """Category names by id, and image records grouped into ordered sequences."""
    data = json.loads(ann_path.read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}

    by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    sequences: dict[str, list[dict]] = collections.defaultdict(list)
    for index, image in enumerate(data["images"]):
        record = dict(image)
        record["annotations"] = by_image.get(image["id"], [])
        record.setdefault("sequence", ann_path.stem)
        record.setdefault("frame_index", index)
        sequences[record["sequence"]].append(record)
    for frames in sequences.values():
        frames.sort(key=lambda r: r["frame_index"])
    return names, dict(sequences)


def draw(frame, record: dict, names: dict[int, str], sequence: str):
    """Boxes, labels and a per-frame tally, drawn in place."""
    import cv2

    counts: collections.Counter = collections.Counter()
    for annotation in record["annotations"]:
        name = names.get(annotation["category_id"], str(annotation["category_id"]))
        color = COLORS.get(name, FALLBACK_COLOR)
        x, y, w, h = annotation["bbox"]
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))

        pseudo = annotation.get("pseudo_label", False)
        ignored = annotation.get("iscrowd", 0) == 1
        cv2.rectangle(frame, p1, p2, color, 1 if pseudo else 2)
        if ignored:
            # An ignore region is not a claim about what is there, so it gets
            # a cross rather than a label -- it should not read as a box the
            # model is being asked to predict.
            cv2.line(frame, p1, p2, color, 1)
            cv2.line(frame, (p1[0], p2[1]), (p2[0], p1[1]), color, 1)

        if pseudo:
            caption = f"{annotation.get('detector_score', 0):.2f}"
            counts["~" + name] += 1
        else:
            caption = name
            jersey = annotation.get("jersey")
            team = annotation.get("team")
            extra = "/".join(str(v) for v in (team, jersey) if v)
            if extra:
                caption += f" {extra}"
            if annotation.get("interpolated"):
                caption += " i"
            counts[name] += 1
        scale = 0.4 if pseudo else 0.45
        cv2.putText(
            frame, caption, (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1
        )

    tally = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    header = f"{sequence}  frame {record['frame_index']}   {tally}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(frame, header, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # "~name" is a detector guess, "i" an interpolated CVAT box, a cross an
    # ignore region. Spelled out once per frame so a clip is self-describing.
    cv2.putText(
        frame,
        "solid=annotation  thin+score=detector  x=ignored  i=interpolated",
        (6, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )
    return frame


def render(record: dict, images_root: Path, names: dict[int, str], sequence: str, scale: float):
    """Draw in the annotations' original coordinates, then scale the whole preview."""
    import cv2

    path = images_root / record["file_name"]
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read {path} (is --images the right root?)")
    frame = draw(frame, record, names, sequence)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame


def encode(frames, out_path: Path, size: tuple[int, int], fps: int, crf: int):
    """Pipe BGR frames to ffmpeg as H.264.

    Through ffmpeg rather than `cv2.VideoWriter`: the OpenCV build here has no
    H.264 encoder, and its `mp4v` output does not play in a browser or in
    QuickTime, which is where these get looked at.
    """
    width, height = size
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(out_path),
    ]  # fmt: skip
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    written = 0
    for frame in frames:
        process.stdin.write(frame.tobytes())
        written += 1
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {out_path}")
    return written


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ann", required=True, type=Path, help="COCO annotations to review")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("/data/datasets/hockeyTrackingDataset"),
        help="root that image file_name entries are relative to",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/coco-review"))
    parser.add_argument("--sequence", action="append", default=None, help="repeatable")
    parser.add_argument("--list", action="store_true", help="print sequences and exit")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=23, help="ffmpeg quality, lower is bigger")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="per sequence, 0 means all")
    parser.add_argument(
        "--stills",
        type=int,
        default=0,
        help="write this many evenly spaced annotated jpegs per sequence instead of a video",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="output size multiplier")
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise SystemExit(f"--frame-stride must be at least 1, got {args.frame_stride}")
    if args.scale <= 0:
        raise SystemExit(f"--scale must be positive, got {args.scale}")

    names, sequences = load(args.ann)

    if args.list:
        for name in sorted(sequences):
            total = sum(len(f["annotations"]) for f in sequences[name])
            print(f"{name:28s} {len(sequences[name]):6d} frames  {total:7d} boxes")
        return 0

    wanted = sorted(sequences) if args.sequence is None else args.sequence
    unknown = [s for s in wanted if s not in sequences]
    if unknown:
        raise SystemExit(
            f"{args.ann} has no sequence {', '.join(unknown)}; --list shows what it does have"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        records = sequences[name][:: args.frame_stride]
        if args.max_frames:
            records = records[: args.max_frames]
        if not records:
            raise SystemExit(f"{name} has no frames left after --frame-stride/--max-frames")

        def rendered(records=records, sequence=name):
            for record in records:
                yield render(record, args.images, names, sequence, args.scale)

        if args.stills:
            step = max(1, len(records) // args.stills)
            chosen = records[::step][: args.stills]
            directory = args.out / name
            directory.mkdir(parents=True, exist_ok=True)
            for record in chosen:
                cv2.imwrite(
                    str(directory / f"{record['frame_index']:06d}.jpg"),
                    render(record, args.images, names, name, args.scale),
                )
            print(f"{name}: {len(chosen)} stills -> {directory}")
            continue

        frames = rendered()
        first = next(frames)
        height, width = first.shape[:2]
        out_path = args.out / f"{name}.mp4"
        written = encode(
            itertools.chain((first,), frames), out_path, (width, height), args.fps, args.crf
        )
        print(f"{name}: {written} frames -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
