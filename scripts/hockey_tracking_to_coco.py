#!/usr/bin/env python3
"""Convert the Montreal hockey tracking dataset to COCO, with the crowd filled in.

The dataset annotates only on-ice people. Everyone in the stands is left
unlabelled, and an unlabelled person is not neutral to a detector -- it is a
negative, so every frame teaches the model that humans in the bowl are
background. hm's pipeline wants the opposite: it detects people everywhere and
lets the rink mask drop the crowd downstream, so training this way works
against the architecture.

So this runs our CrowdHuman-trained detector over every frame and merges what
it finds with the dataset's own boxes. The dataset always wins: a detection
that overlaps an annotated box is discarded and the annotation keeps its class
and geometry. Only detections that match nothing survive, and what happens to
them is the one real choice here -- see ``--crowd-mode``.

The class labels come from ``CVAT_Style_labels``, not from the COCO or MOT
files shipped with the dataset: those collapse everything to a single
``pedestrian`` class and throw the distinction away. The CVAT tracks carry
``player`` / ``ref`` / ``keeper``, plus a ``team`` colour and a jersey number
that are preserved here as annotation attributes.

The hierarchy written out is::

    person                  <- not a class; the detector's whole job
    |
    +-- on-ice
    |   +-- player          <- class 1
    |   +-- goalie          <- class 2   (CVAT "keeper")
    |   +-- official        <- class 3   (CVAT "ref")
    |
    +-- off-ice
        +-- spectator       <- class 4, or an ignore region (--crowd-mode)

``team`` is deliberately not in that tree. It is an annotation attribute,
because "home" has no transferable visual definition -- it is a per-game
assignment of one of two colours, so a detector head cannot learn it.

IGNORE REGIONS
--------------
``--crowd-mode ignore`` is the better design and does nothing today. Under the
stock mmyolo YOLOv8 stack an ``iscrowd=1`` box is dropped twice over:

* ``mmyolo``'s ``LoadAnnotations._load_bboxes`` appends an instance only when
  ``ignore_flag == 0``, so the box never leaves the dataset;
* ``yolov5_collate`` carries no ignore flags, and ``YOLOv8Head`` names
  ``batch_gt_instances_ignore`` in its signature and docstring and never reads
  it. (``YOLOv5Head`` does honour it, via ``ignore_iof_thr`` against the
  objectness branch -- which YOLOv8, being anchor-free, does not have.)

So ``ignore`` produces a dataset that trains exactly as if the crowd had never
been labelled, which is the failure this script exists to remove. Making it
real needs a ``LoadAnnotations`` that keeps ignore boxes and a ``YOLOv8Head``
that masks the classification loss at anchor points covered by one -- a
contained change, but a change to the training stack, so it is not assumed
here. Until then ``spectator`` is the mode that does something.

Images are referenced in place rather than copied; the tree is 14 GB.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

#: CVAT label -> output category name. ``keeper`` is renamed because every
#: other part of this codebase says goalie, and ``ref`` because the on-ice
#: crew includes linesmen, who wear the same stripes and are labelled ``ref``
#: here -- the class is the role, not the title.
LABELS = {"player": "player", "keeper": "goalie", "ref": "official"}

#: Category ids, 1-based as COCO expects.
CATEGORY_IDS = {"player": 1, "goalie": 2, "official": 3, "spectator": 4}

#: Order the categories are declared in, which fixes the contiguous 0-based
#: label ids mmdet derives from them.
CATEGORY_ORDER = ["player", "goalie", "official", "spectator"]

#: Default detector confidence per crowd mode. Measured on a broadcast frame
#: with ~300 visible spectators: 0.5 finds 11 people, 0.05 finds 122 and the
#: empty ice stays clean. The low floor is only safe because an ignored box
#: costs nothing to be wrong about.
SCORE_THR = {"ignore": 0.05, "spectator": 0.5}


def _sequence_key(name: str) -> str:
    """Normalise a sequence name for matching across the two label trees.

    The CVAT filenames and the MOT directory names disagree: ``CAR_BOS_2019_001``
    against ``CAR_VS_BOS_2019_001``, ``all_star_2019_001`` against
    ``allstar_2019_001``. Dropping the separators and the ``vs`` token makes
    all but one pair line up, and the caller checks that the mapping is
    one-to-one rather than trusting it.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower()).replace("vs", "")


def map_sequences(src: Path) -> tuple[dict[str, tuple[Path, Path]], list[str], list[Path]]:
    """Pair each MOT sequence directory with its CVAT label file.

    Returns ``(mapping, unlabelled, orphans)``. A sequence with no CVAT file
    cannot be used -- its officials would silently become background, which is
    the failure this script exists to remove -- so the caller excludes it
    rather than converting it as players.
    """
    sequences: dict[str, Path] = {}
    for split in ("train", "test"):
        split_dir = src / "MOT_Challenge_Sytle_Label" / split
        if not split_dir.is_dir():
            raise SystemExit(f"expected {split_dir} to exist")
        for entry in sorted(split_dir.iterdir()):
            if (entry / "img1").is_dir():
                sequences[entry.name] = entry

    by_key: dict[str, list[Path]] = collections.defaultdict(list)
    for path in sorted((src / "CVAT_Style_labels").rglob("*.xml")):
        by_key[_sequence_key(path.stem)].append(path)

    mapping: dict[str, tuple[Path, Path]] = {}
    unlabelled: list[str] = []
    for name, directory in sequences.items():
        candidates = by_key.get(_sequence_key(name), [])
        if len(candidates) > 1:
            raise SystemExit(
                f"{name} matches {len(candidates)} CVAT files: "
                + ", ".join(str(c) for c in candidates)
            )
        if not candidates:
            unlabelled.append(name)
            continue
        mapping[name] = (directory, candidates[0])

    used = {_sequence_key(name) for name in mapping}
    orphans = [p for key, paths in by_key.items() if key not in used for p in paths]
    return mapping, unlabelled, orphans


def read_cvat(path: Path) -> tuple[dict[int, list[dict[str, Any]]], tuple[int, int], int]:
    """Boxes per 0-based frame index, plus ``(width, height)`` and frame count.

    ``outside="1"`` marks a frame where the tracked person has left the shot,
    so those boxes are skipped. Interpolated boxes (``keyframe="0"``) are kept:
    they are most of the dataset, and dropping them would leave the majority of
    frames with annotated people missing, which is the same poison as the
    unlabelled crowd.
    """
    root = ET.parse(path).getroot()
    meta = root.find("meta")
    if meta is None:
        raise SystemExit(f"{path} has no <meta>")
    size_node = meta.find(".//original_size")
    frames_node = meta.find(".//size")
    if size_node is None or frames_node is None:
        raise SystemExit(f"{path} is missing original_size or size")
    width = int(size_node.findtext("width", "0"))
    height = int(size_node.findtext("height", "0"))

    per_frame: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for track in root.iter("track"):
        label = track.get("label", "")
        if label not in LABELS:
            raise SystemExit(f"{path} has unexpected CVAT label {label!r}")
        for box in track.iter("box"):
            if box.get("outside") == "1":
                continue
            attributes = {a.get("name"): (a.text or "") for a in box.iter("attribute")}
            per_frame[int(box.get("frame", "0"))].append(
                {
                    "category": LABELS[label],
                    "xyxy": (
                        float(box.get("xtl", "0")),
                        float(box.get("ytl", "0")),
                        float(box.get("xbr", "0")),
                        float(box.get("ybr", "0")),
                    ),
                    "track_id": int(track.get("id", "-1")),
                    "occluded": box.get("occluded") == "1",
                    "interpolated": box.get("keyframe") == "0",
                    "team": attributes.get("team"),
                    "jersey": attributes.get("id"),
                }
            )
    return per_frame, (width, height), int(frames_node.text or "0")


def iou_matrix(a: list[tuple[float, float, float, float]], b: list[tuple[float, ...]]):
    """Pairwise IoU between two xyxy lists, as a plain nested list."""
    out = []
    for ax1, ay1, ax2, ay2 in a:
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        row = []
        for bx1, by1, bx2, by2 in b:
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            row.append(inter / union if union > 0 else 0.0)
        out.append(row)
    return out


def unmatched_detections(detections, annotated, threshold: float) -> list[int]:
    """Indices of detections that do not overlap any authoritative annotation.

    Suppress all duplicates on a labelled person. An annotation can claim
    multiple detections; a one-to-one match would relabel the losers as crowd.
    """
    if not detections or not annotated:
        return list(range(len(detections)))
    ious = iou_matrix([d["xyxy"] for d in detections], [a["xyxy"] for a in annotated])
    return [i for i, row in enumerate(ious) if max(row) < threshold]


def build_detector(args):
    """Load the CrowdHuman detector, or return None in --no-detector mode.

    Built through `DetectorFactoryPlugin` rather than `mmdet.apis.init_detector`
    because the shipped `.detector.yaml` files are model fragments, not whole
    mmdet configs, and the factory already handles the mmyolo scope, the
    checkpoint prefixes and the distilled-student weight layout. Running a
    different construction path here would mean the labels came from a model
    that is not quite the one the pipeline runs.

    TensorRT and CUDA graphs are off: this is a batch job over 86k stills, so
    engine build time dominates and the static-detection padding would only
    have to be stripped again.
    """
    if args.no_detector:
        return None

    from hmlib.aspen.plugins.detector_factory_plugin import DetectorFactoryPlugin
    from hmlib.models import detector_selection

    graph = yaml.safe_load(Path(args.graph).read_text())
    try:
        params = dict(graph["aspen"]["plugins"]["detector_factory"]["params"])
    except (KeyError, TypeError) as error:
        raise SystemExit(f"{args.graph} has no aspen.plugins.detector_factory.params") from error

    if args.detector_config is not None:
        params["detector_yaml"] = args.detector_config
    # Through `resolve_detector`, not the raw graph params: the shipped
    # `.detector.yaml` declares the 80-class COCO head, and the profile is
    # what narrows it -- one class for the CrowdHuman person detectors, two
    # for `hockeyref`. Building from the YAML alone loads nothing.
    spec = detector_selection.resolve_detector(args.detector_model, params, args.detector)
    params.pop("detector_yaml", None)
    params["detector"] = spec["detector"]
    params["checkpoint"] = spec["checkpoint"]
    params["trt"] = {"enable": False}
    params["onnx"] = {"enable": False}
    params["cuda_graph"] = False
    params["nms_backend"] = "head"
    params["static_detections"] = {"enable": False}

    classes = params["detector"]["bbox_head"]["head_module"]["num_classes"]
    print(
        f"detector: profile={args.detector_model} classes={classes} "
        f"@ {Path(params['checkpoint']).name}"
    )
    import torch

    factory = DetectorFactoryPlugin(**params)
    # A `torch.device`, not the string: `DetectorFactoryPlugin` moves the model
    # only when `isinstance(context["device"], torch.device)`, so a string
    # leaves it on the CPU with no error and no log line -- about 2.5 frames/s
    # against 82k frames.
    context: dict[str, Any] = {"device": torch.device(args.device)}
    context.update(factory.forward(context) or {})
    model = context.get("detector_model")
    if model is None:
        raise SystemExit("DetectorFactoryPlugin did not publish a detector_model")

    placed = next(model.parameters()).device
    if placed != torch.device(args.device):
        raise SystemExit(f"asked for {args.device}, detector landed on {placed}")
    print(f"detector on {placed}")
    return model


def build_transforms(graph_path: str):
    """The graph's resize and pad, instantiated, plus the serving size.

    Replayed rather than reimplemented. The detection head consumes
    `scale_factor` straight out of the metainfo (`yolov5_head.py:432`), and
    `HmResize` publishes it as a device tensor derived from the *achieved*
    integer size rather than from the requested ratio; `HmPad` then fills a
    fixed 1632x608 canvas with 114, not a round-up to the backbone divisor.
    Hand geometry that is merely close moves every box, so the two steps that
    decide where a box lands come from the graph itself.

    The graph's other steps are deliberately skipped: `LoadImageFromFile` and
    `HmImageToTensor` are done here, `HmCrop` needs per-game crop parameters a
    still does not carry, `HmImageColorAdjust` is identity at the shipped
    defaults, and `PackTrackInputs` packs a video clip rather than a frame.
    """
    from hmlib.hm_transforms import HmPad, HmResize

    wanted = {"HmResize": HmResize, "HmPad": HmPad}
    graph = yaml.safe_load(Path(graph_path).read_text())
    steps = graph.get("aspen", {}).get("inference_pipeline", [])

    built = []
    seen = set()
    for step in steps:
        cls = wanted.get(step.get("type"))
        if cls is None:
            continue
        built.append(cls(**{k: v for k, v in step.items() if k != "type"}))
        seen.add(step["type"])
    missing = sorted(set(wanted) - seen)
    if missing:
        raise SystemExit(
            f"{graph_path} inference_pipeline has no {', '.join(missing)}; "
            "the detector geometry is taken from the graph, not assumed"
        )

    resize = built[0]
    if not isinstance(resize, HmResize) or not resize.img_scale:
        raise SystemExit(f"{graph_path} must apply HmResize with an img_scale before HmPad")
    width, height = resize.img_scale[0]
    return built, (int(width), int(height))


def detect(model, transforms, image_paths: list[Path], score_thr: float, with_labels: bool = False):
    """People the detector finds, per frame, in original-frame coordinates.

    Colour order is BGR, matching the pipeline rather than the config: the
    graph's `LoadImageFromFile` decodes BGR and nothing swaps the channels
    before `DetectorInferencePlugin` calls `detector.predict` directly, which
    bypasses the data preprocessor that would otherwise honour `bgr_to_rgb`.
    Feeding RGB here would label the frames with a model that behaves
    differently from the one the pipeline runs.

    Detection classes are ignored unless ``with_labels`` asks for them. Every
    box is treated as "a person", which is all the merge needs -- a detection
    that lands on an annotated person is dropped whatever the detector called
    it, and one that lands on nobody is crowd. That also keeps a two-class
    profile like `hockeyref` usable here without its referee opinion silently
    overriding the dataset's. ``with_labels`` adds the predicted class index
    for callers that are comparing heads rather than merging boxes.
    """
    import cv2
    import numpy as np
    import torch
    from mmdet.structures import DetDataSample

    from hmlib.utils.image import make_channels_first

    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot decode {path}")
        images.append(image)

    # Frames of different sizes cannot share a padded batch, and this dataset
    # is uniformly 1280x720, so the grouping is a formality -- but a silent
    # reshape would be a wrong box, not a slow one.
    groups: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for position, image in enumerate(images):
        groups[image.shape[:2]].append(position)

    device = next(model.parameters()).device
    found: list[list[dict[str, Any]]] = [[] for _ in images]

    for (height, width), positions in groups.items():
        # Channels-last through the transforms and channels-first only at the
        # end, which is the order `DetectorInferencePlugin` uses: `HmResize`
        # asserts NHWC, and `make_channels_first` runs after the pipeline.
        stacked = torch.from_numpy(np.stack([images[p] for p in positions]))
        results = {"img": stacked.to(device).float()}
        for transform in transforms:
            results = transform(results)
        batch = make_channels_first(results["img"]).contiguous() / 255.0

        # `HmResize` publishes `img_shape` and `scale_factor` per image, but
        # `HmPad` overwrites `pad_shape` with a single flat [h, w] -- and the
        # padded canvas is the batch, so take it from there and avoid relying
        # on which of the two wrote the key last.
        padded = tuple(batch.shape[-2:])
        samples = []
        for index in range(len(positions)):
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "ori_shape": (height, width),
                    "img_shape": tuple(results["img_shape"][index]),
                    "pad_shape": padded,
                    "scale_factor": results["scale_factor"][index],
                    "batch_input_shape": padded,
                }
            )
            samples.append(sample)

        with torch.no_grad():
            # rescale=True by default, so the head undoes `scale_factor` and
            # clamps to `ori_shape`: boxes come back in frame coordinates.
            predictions = model.predict(batch, samples)

        for position, prediction in zip(positions, predictions):
            instances = prediction.pred_instances
            keep = instances.scores >= score_thr
            boxes = instances.bboxes[keep].detach().cpu().tolist()
            scores = instances.scores[keep].detach().cpu().tolist()
            labels = instances.labels[keep].detach().cpu().tolist() if with_labels else None
            found[position] = [
                {
                    "xyxy": tuple(float(v) for v in box),
                    "score": float(score),
                    **({"label": int(labels[i])} if with_labels else {}),
                }
                for i, (box, score) in enumerate(zip(boxes, scores))
            ]

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("/data/datasets/hockeyTrackingDataset"))
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument(
        "--crowd-mode",
        choices=("spectator", "ignore"),
        default="spectator",
        help=(
            "What to do with people the detector found and the dataset never annotated. "
            "`spectator` writes them as a fourth trainable class, which makes today's "
            "false positives permanent ground truth but is the only mode the stock mmyolo "
            "YOLOv8 stack acts on. `ignore` writes them with iscrowd=1, which is the "
            "better idea and currently a no-op -- see IGNORE REGIONS in the module "
            "docstring before choosing it."
        ),
    )
    parser.add_argument("--detector-model", default="trained", help="detector profile to run")
    parser.add_argument("--detector", default=None, help="checkpoint override")
    parser.add_argument("--detector-config", default=None, help="mmdet config override")
    parser.add_argument(
        "--graph",
        default="hmlib/config/aspen/tracking.yaml",
        help="aspen graph to take the detector architecture from",
    )
    parser.add_argument("--device", default="cuda:0")
    # 16 measured 93 frames/s against 71 at 8 and 92 at 32 on a 4090.
    parser.add_argument("--batch-size", type=int, default=16, help="frames per detector call")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help=(
            "Detector confidence floor for crowd boxes. Defaults per --crowd-mode, because "
            f"the two modes want opposite things: {SCORE_THR['ignore']} for `ignore`, where a "
            "duplicate or a false positive only widens an ignore region but a missed spectator "
            "is exactly the 'humans are background' signal being removed, and "
            f"{SCORE_THR['spectator']} for `spectator`, where every box becomes a hard positive "
            "and precision is what matters."
        ),
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--limit-sequences", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="convert the CVAT labels only, skipping inference. Useful to check the "
        "mapping and the split before committing to a full pass over 82k frames.",
    )
    args = parser.parse_args()

    if not 0.0 < args.match_iou <= 1.0:
        raise SystemExit(f"--match-iou must be in (0,1], got {args.match_iou}")
    if args.score_thr is None:
        args.score_thr = SCORE_THR[args.crowd_mode]
        print(f"--score-thr {args.score_thr} (default for --crowd-mode {args.crowd_mode})")
    if args.crowd_mode == "ignore" and not args.no_detector:
        print(
            "WARNING: --crowd-mode ignore writes iscrowd=1 boxes, which the stock mmyolo "
            "YOLOv8 stack discards at load time and never reads in the head. Unless you "
            "are training with ignore-aware code, this dump trains identically to one "
            "with no crowd boxes at all. See IGNORE REGIONS in the module docstring."
        )
    if not 0.0 <= args.score_thr <= 1.0:
        raise SystemExit(f"--score-thr must be in [0,1], got {args.score_thr}")
    if args.batch_size < 1:
        raise SystemExit(f"--batch-size must be at least 1, got {args.batch_size}")
    if args.frame_stride < 1:
        raise SystemExit(f"--frame-stride must be at least 1, got {args.frame_stride}")

    mapping, unlabelled, orphans = map_sequences(args.src)
    print(f"{len(mapping)} sequences with CVAT labels")
    if unlabelled:
        # Converting these as player-only would put officials into the
        # background class, which is the failure this script removes.
        print(f"  excluded, no CVAT labels: {', '.join(sorted(unlabelled))}")
    if orphans:
        print(f"  CVAT files matching no sequence: {', '.join(str(o) for o in orphans)}")

    names = sorted(mapping)
    if args.limit_sequences:
        names = names[: args.limit_sequences]

    # Hold out whole sequences. A frame-level split is meaningless here: 86k
    # frames come from 26 clips, so neighbouring frames are near-identical and
    # would land on both sides of the boundary.
    #
    # Chosen by frame count, not sequence count. The clips run from 226 to
    # 6027 frames, so holding out a fixed number of them lands anywhere
    # between 3% and 30% of the data; a sequence is only taken if it brings
    # the held-out total closer to the target than stopping would.
    lengths = {name: len(list((mapping[name][0] / "img1").glob("*.jpg"))) for name in names}
    total_frames = sum(lengths.values())
    target = total_frames * args.val_fraction
    shuffled = list(names)
    random.Random(args.seed).shuffle(shuffled)
    val: set[str] = set()
    held = 0
    for name in shuffled:
        if len(val) == len(shuffled) - 1:
            break
        if abs(held + lengths[name] - target) >= abs(held - target) and val:
            continue
        val.add(name)
        held += lengths[name]
    if not val:
        raise SystemExit(
            f"cannot hold out a validation set from {len(shuffled)} sequence(s) at "
            f"--val-fraction {args.val_fraction}: whole sequences are the unit, because "
            "neighbouring frames are near-identical and a frame-level split would leak"
        )
    print(
        f"split: {len(shuffled) - len(val)} train sequences / {total_frames - held} frames, "
        f"{len(val)} val / {held} frames ({held / total_frames:.1%}, target {args.val_fraction:.0%})"
    )
    print(f"  val sequences: {', '.join(sorted(val))}")

    model = build_detector(args)
    transforms = None
    if model is None:
        print("running without a detector; crowd boxes will not be added")
    else:
        transforms, scale = build_transforms(args.graph)
        print(f"serving at {scale[0]}x{scale[1]} (from {args.graph})")

    # All four are declared whatever the crowd mode, and `spectator` is last.
    # mmdet maps categories by enumerating the ones it finds, so a training
    # config wanting only the on-ice roles declares the first three in its
    # `metainfo` and gets labels 0,1,2 unchanged -- while a config that mixes
    # CrowdHuman in as `spectator` finds the class already here. Making the
    # block depend on the mode would mean a checkpoint trained on one dump
    # could not be evaluated against the other.
    categories = [{"id": CATEGORY_IDS[n], "name": n} for n in CATEGORY_ORDER]
    out: dict[str, dict[str, list]] = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "val")
    }
    counts: dict[str, collections.Counter] = {
        "train": collections.Counter(),
        "val": collections.Counter(),
    }
    image_id = annotation_id = 0

    for name in names:
        directory, xml_path = mapping[name]
        per_frame, (width, height), frame_count = read_cvat(xml_path)
        split = "val" if name in val else "train"
        images = sorted((directory / "img1").glob("*.jpg"))
        if len(images) != frame_count:
            print(f"  {name}: CVAT declares {frame_count} frames, {len(images)} jpegs on disk")

        frames: list[tuple[int, int, Path]] = []
        for index, image_path in enumerate(images):
            if index % args.frame_stride:
                continue
            image_id += 1
            frames.append((image_id, index, image_path))
            out[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": str(image_path.relative_to(args.src)),
                    "width": width,
                    "height": height,
                    "sequence": name,
                    "frame_index": index,
                }
            )
            for box in per_frame.get(index, []):
                x1, y1, x2, y2 = box["xyxy"]
                annotation_id += 1
                out[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_IDS[box["category"]],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "track_id": box["track_id"],
                        "occluded": box["occluded"],
                        "interpolated": box["interpolated"],
                        "team": box["team"],
                        "jersey": box["jersey"],
                    }
                )
                counts[split][box["category"]] += 1

        crowd = args.crowd_mode == "spectator"
        added = 0
        for start in range(0, len(frames) if model is not None else 0, args.batch_size):
            chunk = frames[start : start + args.batch_size]
            batch = detect(model, transforms, [path for _, _, path in chunk], args.score_thr)
            for (frame_image_id, index, _), found in zip(chunk, batch):
                annotated = per_frame.get(index, [])
                for i in unmatched_detections(found, annotated, args.match_iou):
                    x1, y1, x2, y2 = found[i]["xyxy"]
                    annotation_id += 1
                    out[split]["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": frame_image_id,
                            # Ignored boxes still carry a real class, rather
                            # than `spectator`, so that a training config
                            # declaring only the three on-ice classes keeps
                            # them: mmdet drops annotations whose category is
                            # not in `metainfo`, which would silently delete
                            # the ignore regions and restore the very signal
                            # this mode exists to remove.
                            "category_id": CATEGORY_IDS["spectator" if crowd else "player"],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0 if crowd else 1,
                            "detector_score": found[i]["score"],
                            "pseudo_label": True,
                        }
                    )
                    counts[split]["crowd"] += 1
                    added += 1
        print(f"  {name} -> {split}: {len(frames)} frames, {added} crowd", flush=True)

    annotations_dir = args.dst / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with open(annotations_dir / f"{split}.json", "w", encoding="utf-8") as stream:
            json.dump(out[split], stream)

    provenance = {
        "source": str(args.src),
        "crowd_mode": args.crowd_mode,
        "detector_model": None if model is None else args.detector_model,
        "detector_checkpoint": None if model is None else (args.detector or "profile default"),
        "score_thr": args.score_thr,
        "match_iou": args.match_iou,
        "frame_stride": args.frame_stride,
        "seed": args.seed,
        "val_sequences": sorted(val),
        "excluded_sequences": sorted(unlabelled),
        "counts": {split: dict(counts[split]) for split in counts},
        "images": {split: len(out[split]["images"]) for split in out},
    }
    with open(args.dst / "provenance.json", "w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)

    print("\ncounts:")
    for split in ("train", "val"):
        total = sum(counts[split].values())
        detail = "  ".join(f"{k}={v}" for k, v in sorted(counts[split].items()))
        print(f"  {split}: {len(out[split]['images'])} images, {total} boxes   {detail}")
    print(f"\nwrote {annotations_dir}/ and {args.dst / 'provenance.json'}")


if __name__ == "__main__":
    sys.exit(main())
