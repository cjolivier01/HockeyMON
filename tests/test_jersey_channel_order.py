"""The jersey OCR trunk must hand its models RGB.

`original_images` is BGR -- `hmlib/video/video_stream.py` decodes `bgr24` and
the stitcher tags its output `channel_order = "bgr"`, which is why every
detector and pose config carries `bgr_to_rgb=True`. Both consumers in the
Koshkina trunk want RGB: PARSeq through `Image.fromarray(..., mode="RGB")`
and the legibility model through ImageNet RGB statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from mmengine.structures import InstanceData

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aspen_plugin_harness import make_track_data_sample

from hmlib.aspen.plugins.jersey_koshkina_plugin import (
    KoshkinaJerseyNumberPlugin,
    _bgr_to_rgb,
)


def should_hand_the_recogniser_rgb_pixels():
    """Drive the real trunk and inspect the crop its model would have seen.

    Asserting on the source text, or reimplementing the flip in the test, both
    pass while the production path is broken. This captures the crop at the
    point PARSeq receives it.
    """
    seen: list[torch.Tensor] = []

    plugin = KoshkinaJerseyNumberPlugin(enabled=True, roi_mode="bbox")
    plugin._run_parseq_on_crop = lambda crop: seen.append(crop.clone()) or None

    # A frame whose subject is red. In BGR that is channel 2.
    frame = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    frame[:, 2] = 255

    tracks = InstanceData()
    tracks.bboxes = torch.tensor([[8.0, 8.0, 56.0, 56.0]])
    tracks.instances_id = torch.tensor([1])
    tracks.labels = torch.tensor([0])
    tracks.scores = torch.tensor([0.9])
    samples = make_track_data_sample(
        num_frames=1, ori_shape=(64, 64), pred_track_instances=[tracks]
    )
    plugin.forward(
        {"data_samples": samples, "original_images": frame, "frame_id": 0, "pose_results": None}
    )

    assert seen, "the trunk never reached the recogniser; the test is not exercising it"
    crop = seen[0].to(dtype=torch.float32)
    red, green, blue = (float(crop[c].mean()) for c in range(3))
    assert red > 200 and blue < 55, (
        f"PARSeq would read this crop as R={red:.0f} G={green:.0f} B={blue:.0f}; "
        "a red sweater must not arrive as blue"
    )


def should_leave_the_mmocr_trunk_on_bgr():
    """The sibling pose trunk feeds MMOCR, which expects BGR; it must not flip.

    Applying this fix there would introduce the bug it removes here. Nothing
    in either file said so, so a later reader could reasonably get it wrong.
    """
    source = Path("hmlib/aspen/plugins/jersey_pose_plugin.py").read_text()
    assert "_bgr_to_rgb" not in source and ".flip(0)" not in source, (
        "jersey_pose_plugin feeds MMOCRInferencer, whose configs set "
        "bgr_to_rgb=True and therefore expect BGR ndarray input"
    )


def should_reverse_the_channel_axis():
    """The helper itself, on the (C, H, W) layout it is given."""
    frame = torch.zeros(3, 4, 4, dtype=torch.uint8)
    frame[0] = 10
    frame[1] = 20
    frame[2] = 30
    out = _bgr_to_rgb(frame)
    assert [int(out[c, 0, 0]) for c in range(3)] == [30, 20, 10]
