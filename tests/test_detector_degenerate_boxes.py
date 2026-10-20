"""Degenerate-detection compaction in the detector trunk.

A model can emit a box with non-positive width or height. Every consumer
treats that as meaningless, but the telemetry writer rejects the entire batch
over one of them (`hmlib/telemetry/capture.py::_boxes`), which takes the run
down rather than losing a box.

Filtering is not as simple as masking rows out. With static detection outputs
the tensors are fixed-size and a scalar count says how many *leading* rows are
real; dropping rows outright leaves that count larger than the array and trips
a second telemetry assertion ("observation count exceeds allocation"). These
tests pin the prefix invariant that makes the fix correct.
"""

from __future__ import annotations

import torch
from mmengine.structures import InstanceData

from hmlib.aspen.plugins.detector_plugin import _drop_degenerate_boxes


def _instances(boxes, valid=None) -> InstanceData:
    metainfo = {"num_detections": torch.tensor([valid])} if valid is not None else None
    inst = InstanceData(metainfo=metainfo)
    inst.bboxes = torch.tensor(boxes, dtype=torch.float32)
    count = len(boxes)
    # Distinct scores so the test can tell whether sibling fields moved with
    # their rows rather than being left behind.
    inst.scores = torch.arange(count, dtype=torch.float32)
    inst.labels = torch.zeros((count,), dtype=torch.int64)
    return inst


def should_leave_clean_detections_untouched():
    """The common path must not allocate or copy."""
    inst = _instances([[0, 0, 2, 2], [0, 0, 3, 3]], valid=2)
    assert _drop_degenerate_boxes(inst) is inst


def should_preserve_the_static_allocation_size():
    """Row count is fixed by the static output shape; only the count shrinks."""
    inst = _instances(
        [[0, 0, 10, 10], [5, 5, 5, 9], [1, 1, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0]], valid=3
    )
    out = _drop_degenerate_boxes(inst)
    assert len(out) == 5
    assert int(out.num_detections[0]) == 2


def should_compact_survivors_to_the_front():
    """The count is a prefix, so the kept rows have to lead."""
    inst = _instances([[0, 0, 10, 10], [5, 5, 5, 9], [1, 1, 4, 4]], valid=3)
    out = _drop_degenerate_boxes(inst)
    assert out.bboxes[:2].tolist() == [[0, 0, 10, 10], [1, 1, 4, 4]]


def should_move_sibling_fields_with_their_rows():
    """Scores and labels must stay aligned to the boxes they describe."""
    inst = _instances([[0, 0, 10, 10], [5, 5, 5, 9], [1, 1, 4, 4]], valid=3)
    out = _drop_degenerate_boxes(inst)
    # Rows 0 and 2 survived, so their original scores lead.
    assert out.scores[:2].tolist() == [0.0, 2.0]


def should_ignore_rows_beyond_the_valid_prefix():
    """Padding rows are already garbage; they must not shrink the count."""
    inst = _instances([[0, 0, 10, 10], [1, 1, 4, 4], [0, 0, 0, 0]], valid=2)
    assert _drop_degenerate_boxes(inst) is inst


def should_handle_instances_with_no_count_metadata():
    """Without a count every row is real, so plain filtering is correct."""
    inst = _instances([[0, 0, 2, 2], [1, 1, 1, 5]])
    out = _drop_degenerate_boxes(inst)
    assert out.bboxes[0].tolist() == [0, 0, 2, 2]


def should_drop_non_finite_boxes():
    inst = _instances([[0, 0, 2, 2], [0, 0, float("inf"), 3]], valid=2)
    out = _drop_degenerate_boxes(inst)
    assert int(out.num_detections[0]) == 1


def should_survive_an_empty_detection_set():
    inst = InstanceData()
    inst.bboxes = torch.zeros((0, 4), dtype=torch.float32)
    inst.scores = torch.zeros((0,), dtype=torch.float32)
    inst.labels = torch.zeros((0,), dtype=torch.int64)
    assert _drop_degenerate_boxes(inst) is inst
