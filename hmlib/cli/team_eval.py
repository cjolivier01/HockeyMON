"""Score team affiliation against SoccerNet GSR labels.

This exists so "no training was necessary" is a measurement rather than an
assumption. It runs the same backend and clusterer the trunk uses over
SoccerNet Game State Reconstruction clips and reports **team purity**: how
often the two-way split agrees with the annotated ``team`` attribute, scored
under the better of the two cluster-to-team assignments, because cluster
indices are arbitrary.

GSR is soccer, so treat the number as evidence that the appearance split
transfers, not as a hockey result. Officials are excluded from scoring rather
than detected -- referee handling is not implemented.

Download the data first::

    python -c "from SoccerNet.Downloader import SoccerNetDownloader as D; \\
        D(LocalDirectory='data/SoccerNetGS').downloadDataTask( \\
            task='gamestate-2024', split=['valid'])"

Then::

    python hmlib/cli/team_eval.py --gsr-root data/SoccerNetGS/valid --backend lab
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from hmlib.log import logger
from hmlib.team.backends import BACKEND_NAMES, build_backend, stack_crops
from hmlib.team.clusterer import OnlineTeamClusterer

#: Roles that belong to one of the two sides. Everything else is skipped.
_TEAM_ROLES = ("player", "goalkeeper")


@dataclass
class Tally:
    """Running counts for one clip."""

    cluster_vs_team: dict[tuple[int, str], int] = field(default_factory=dict)
    assigned: int = 0
    unassigned: int = 0

    def team_purity(self) -> float | None:
        """Best-permutation agreement between clusters and annotated teams."""
        total = sum(self.cluster_vs_team.values())
        if total == 0:
            return None
        straight = sum(
            n
            for (cluster, team), n in self.cluster_vs_team.items()
            if (cluster == 0) == (team == "left")
        )
        return max(straight, total - straight) / total


def _load_labels(sequence: Path) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Return ``(image_id -> file_name, image_id -> annotations)``."""
    labels = json.loads((sequence / "Labels-GameState.json").read_text())
    images = {str(entry["image_id"]): entry["file_name"] for entry in labels["images"]}
    by_image: dict[str, list[dict]] = {}
    for annotation in labels["annotations"]:
        if "bbox_image" not in annotation:
            # Pitch-only entries (the ball has no image box in some clips).
            continue
        by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
    return images, by_image


def _crop(frame: np.ndarray, box: dict) -> torch.Tensor | None:
    """Upper-torso crop from a GSR ``bbox_image`` dict, as RGB HWC uint8.

    Mirrors :func:`hmlib.bbox.torso.bbox_upper_torso_fallback` so the eval
    measures the same pixels the trunk would see.
    """
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    x1 = max(0, x + int(0.25 * w))
    x2 = min(frame.shape[1], x + w - int(0.25 * w))
    y1 = max(0, y + int(0.15 * h))
    y2 = min(frame.shape[0], y + int(0.50 * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = frame[y1:y2, x1:x2, ::-1].copy()  # BGR -> RGB
    return torch.from_numpy(patch)


def evaluate_sequence(
    sequence: Path,
    backend_name: str,
    max_frames: int | None,
    warmup_samples: int,
    min_margin: float,
) -> Tally:
    """Run the backend and clusterer over one clip and tally agreement."""
    images, by_image = _load_labels(sequence)
    backend = build_backend(backend_name)
    clusterer = OnlineTeamClusterer(
        warmup_samples=warmup_samples, min_margin=min_margin, refit_interval=10**9
    )
    tally = Tally()

    ordered = sorted(images)
    if max_frames is not None:
        ordered = ordered[:max_frames]

    # Two passes: the first fills the clusterer, the second scores against
    # settled centroids. Scoring during warmup would mark every early crop
    # wrong for a reason that has nothing to do with the backend.
    for scoring in (False, True):
        for image_id in ordered:
            annotations = by_image.get(image_id)
            if not annotations:
                continue
            path = sequence / "img1" / Path(images[image_id]).name
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"could not read frame {path}")

            crops, metas = [], []
            for annotation in annotations:
                attributes = annotation.get("attributes") or {}
                if str(attributes.get("role", "")) not in _TEAM_ROLES:
                    continue
                crop = _crop(frame, annotation["bbox_image"])
                if crop is None:
                    continue
                crops.append(crop)
                metas.append(attributes)
            if not crops:
                continue

            assignment = clusterer.observe(*_signals(backend, crops))
            if not scoring:
                continue

            for index, attributes in enumerate(metas):
                team = str(attributes.get("team", ""))
                if team not in ("left", "right"):
                    continue
                cluster = int(assignment.clusters[index])
                if cluster < 0:
                    tally.unassigned += 1
                    continue
                tally.assigned += 1
                key = (cluster, team)
                tally.cluster_vs_team[key] = tally.cluster_vs_team.get(key, 0) + 1

    return tally


def _signals(backend, crops):
    """Batch the per-annotation crops the way the trunk batches its ROIs."""
    out = backend.extract(stack_crops(crops))
    return out.features, out.lightness


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--gsr-root",
        required=True,
        type=Path,
        help="SoccerNet GSR split directory containing sequence folders",
    )
    parser.add_argument("--backend", default="lab", choices=list(BACKEND_NAMES))
    parser.add_argument("--warmup-samples", type=int, default=200)
    parser.add_argument("--min-margin", type=float, default=0.12)
    parser.add_argument("--max-sequences", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=60, help="per sequence; 0 for all")
    return parser


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    sequences = sorted(p for p in args.gsr_root.iterdir() if (p / "Labels-GameState.json").exists())
    if not sequences:
        raise SystemExit(
            f"no GSR sequences under {args.gsr_root} "
            "(expected subdirectories each containing Labels-GameState.json)"
        )
    sequences = sequences[: args.max_sequences]
    max_frames = args.max_frames or None

    totals = Tally()
    for sequence in sequences:
        tally = evaluate_sequence(
            sequence=sequence,
            backend_name=args.backend,
            max_frames=max_frames,
            warmup_samples=args.warmup_samples,
            min_margin=args.min_margin,
        )
        logger.info(
            "%-28s team purity %s  assigned %d  unassigned %d",
            sequence.name,
            _format(tally.team_purity()),
            tally.assigned,
            tally.unassigned,
        )
        for key, count in tally.cluster_vs_team.items():
            totals.cluster_vs_team[key] = totals.cluster_vs_team.get(key, 0) + count
        totals.assigned += tally.assigned
        totals.unassigned += tally.unassigned

    print(f"\nbackend        : {args.backend}")
    print(f"sequences      : {len(sequences)}")
    print(f"team purity    : {_format(totals.team_purity())}")
    print(f"assigned crops : {totals.assigned}")
    print(f"left UNKNOWN   : {totals.unassigned} (below the margin threshold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
