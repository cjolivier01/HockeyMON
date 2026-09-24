#!/usr/bin/env python3
"""Compare HM detectors using a game's normal decoding/stitching pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--trained-checkpoint")
    parser.add_argument("--distilled-checkpoint")
    parser.add_argument("--include-distilled", action="store_true")
    parser.add_argument(
        "--tracking-model", choices=("deployed", "trained", "distilled"), default="deployed"
    )
    parser.add_argument(
        "--annotations", help="COCO labels with explicit hmtrack frame_id per image."
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--sample-every", type=int, default=30)
    parser.add_argument("--preview-frames", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "hmtrack_args",
        nargs=argparse.REMAINDER,
        help="Additional hmtrack flags after -- (e.g. --input-video clip.mkv).",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.max_frames < 1 or args.sample_every < 1:
        raise ValueError("--max-frames and --sample-every must be positive")
    from hmlib.config import get_game_dir

    game_dir = get_game_dir(args.game_id, assert_exists=False)
    annotations = args.annotations
    if annotations is None and game_dir:
        candidate = Path(game_dir) / "detector_ground_truth.json"
        if candidate.is_file():
            annotations = str(candidate)
    output = Path(
        args.output_dir
        or (
            Path("output_workdirs")
            / args.game_id
            / "detector_comparisons"
            / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        )
    )
    cmd = [
        sys.executable,
        "-m",
        "hmlib.cli.hmtrack",
        "--game-id",
        args.game_id,
        "--detector-model",
        args.tracking_model,
        "--compare-detectors",
        str(output),
        "--detector-sample-every",
        str(args.sample_every),
        "--detector-preview-frames",
        str(args.preview_frames),
        "--detector-review-threshold",
        str(args.score_threshold),
        "--max-frames",
        str(args.max_frames),
        "--no-audio",
        "--skip-final-video-save",
        "--camera-ui=0",
    ]
    for flag, value in (
        ("--trained-checkpoint", args.trained_checkpoint),
        ("--distilled-checkpoint", args.distilled_checkpoint),
        ("--detector-annotations", annotations),
        ("-s", args.start_time),
    ):
        if value:
            cmd.extend([flag, value])
    if args.include_distilled or args.distilled_checkpoint or args.tracking_model == "distilled":
        cmd.append("--compare-distilled")
    extra = args.hmtrack_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    cmd.extend(extra)
    print(shlex.join(cmd), flush=True)
    if not annotations:
        print(
            "No ground truth: comparison will report detections/agreement and previews; "
            "mAP requires --annotations.",
            flush=True,
        )
    if args.dry_run:
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "openmm/mmdetection"),
            str(ROOT / "openmm/mmyolo"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.setdefault("HM_NMS_BACKEND", "mmcv")
    subprocess.run(cmd, check=True, env=env)
    summary = output / "summary.txt"
    if not summary.is_file():
        raise RuntimeError(f"hmtrack completed without a comparison report: {summary}")
    print(summary.read_text())
    print(f"Report and previews: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
