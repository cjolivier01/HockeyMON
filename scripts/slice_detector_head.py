"""Reshape a CrowdHuman YOLOv8 checkpoint's 80-class head down to two classes.

The shipped detector (``hm_crowdhuman_e85_yolov8_m_1984_736.pth``) keeps COCO's
80-wide classification head and trains people into COCO's ``person`` slot,
index 0. Fine-tuning for a referee class needs a 2-wide head, and letting
mmengine discard the mismatched layers would throw away the one thing worth
keeping -- a person classifier trained on ~340k CrowdHuman instances -- and
relearn it from 14k hockey boxes.

So the head is sliced rather than dropped:

    out class 0 (player)  <- COCO row 0 (person), verbatim
    out class 1 (referee) <- COCO row 0 (person), copied

Copying the person row into the referee slot is a deliberate init: a referee
*is* a person, so the new logit starts already firing on people and fine-tuning
only has to learn the stripes-versus-sweater distinction. Starting it from an
unrelated COCO row, or from noise, would make it relearn "is a person" from the
1,654 referee boxes that exist.

Only the final 1x1 conv of each ``cls_preds`` branch is class-shaped; the
regression branch and everything upstream are class-agnostic and pass through
untouched.

Usage::

    python scripts/slice_detector_head.py --src in.pth --dst out.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

#: COCO index the CrowdHuman run trains people into.
PERSON_ROW = 0

#: Output head width: player, referee.
NUM_CLASSES = 2


def _is_class_layer(key: str) -> bool:
    """Final 1x1 conv of a cls_preds branch -- the only class-shaped tensors."""
    return "cls_preds" in key and key.endswith((".2.weight", ".2.bias"))


def slice_state_dict(state_dict: dict, source_rows: int) -> tuple[dict, int]:
    out, touched = {}, 0
    for key, value in state_dict.items():
        if _is_class_layer(key) and value.shape[0] == source_rows:
            rows = [PERSON_ROW] * NUM_CLASSES
            out[key] = value[rows].clone()
            touched += 1
        else:
            out[key] = value
    return out, touched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--source-classes", type=int, default=80)
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    sliced, touched = slice_state_dict(state_dict, args.source_classes)
    if touched == 0:
        raise SystemExit(
            f"no {args.source_classes}-row cls_preds layers found in {args.src}; "
            "the checkpoint is not the 80-class CrowdHuman head this expects"
        )

    # Drop optimizer/EMA state: it is shaped for the old head and for a finished
    # schedule, and carrying it into a fresh fine-tune would fight the new LR.
    out = {"state_dict": sliced, "meta": checkpoint.get("meta", {})}
    torch.save(out, args.dst)
    print(f"rewrote {touched} class layers {args.source_classes} -> {NUM_CLASSES}")
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
