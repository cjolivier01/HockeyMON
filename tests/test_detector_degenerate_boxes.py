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

import pytest
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
    """Clean input comes back with its rows and count exactly as they were.

    Not identity: deciding there is nothing to do would mean reading the count
    on the host, which blocks on the detector's kernels every frame. The
    compaction runs unconditionally on device instead, so what matters is that
    it is a no-op in content and in order.
    """
    inst = _instances([[0, 0, 2, 2], [0, 0, 3, 3]], valid=2)
    out = _drop_degenerate_boxes(inst)
    assert out.bboxes.tolist() == [[0, 0, 2, 2], [0, 0, 3, 3]]
    assert int(out.num_detections[0]) == 2


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
    """Padding is garbage and must not shrink the count; a bad row inside must.

    The degenerate row sits *inside* the prefix and a valid row *outside* it.
    Without the prefix mask both versions agree on a count of 2 and nothing
    moves, so an input with the padding last cannot tell them apart -- the
    collapsed box would stay inside the prefix and reach the telemetry writer.
    """
    inst = _instances([[0, 0, 10, 10], [0, 0, 0, 0], [1, 1, 4, 4]], valid=2)
    out = _drop_degenerate_boxes(inst)
    assert int(out.num_detections[0]) == 1
    assert out.bboxes[0].tolist() == [0, 0, 10, 10]
    # The row outside the prefix stays outside it rather than being promoted.
    assert out.bboxes[: int(out.num_detections[0])].tolist() == [[0, 0, 10, 10]]


def should_clamp_a_count_larger_than_the_array():
    """A count past the end must not reach telemetry's allocation check.

    Deciding whether to reject it needs the value on the host, so it is
    clamped -- but the clamped figure is what gets written back, which is the
    part that matters: `hmlib/telemetry/capture.py` raises on a count larger
    than the rows it was given.
    """
    inst = _instances([[0, 0, 10, 10], [1, 1, 4, 4]], valid=7)
    out = _drop_degenerate_boxes(inst)
    assert int(out.num_detections[0]) == 2


def should_reject_an_empty_count_tensor():
    """Three other call sites raise on this; disagreeing silently is worse."""
    inst = _instances([[0, 0, 2, 2]], valid=1)
    inst.set_metainfo({"num_detections": torch.zeros((0,), dtype=torch.int32)})
    with pytest.raises(ValueError, match="at least one value"):
        _drop_degenerate_boxes(inst)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to observe a host sync")
def should_not_synchronize_the_host_on_the_detector_hot_path():
    """Both helpers run on every frame, so neither may read a count back.

    A host read blocks until the detector's kernels retire. `hmlib/utils/nms.py`
    keeps `num_valid` on device for this reason and `_strip_static_padding`
    carries a GPU-mask branch for it; these two sat in the same path and
    undid it. Covered for a count on either side, because the static TRT path
    leaves it on device while other callers build it on the host.
    """
    from hmlib.aspen.plugins.detector_plugin import _cap_referees

    for count_device in ("cpu", "cuda"):
        boxes = torch.rand(256, 4, device="cuda")
        boxes[:, 2:] += boxes[:, :2] + 0.1
        inst = InstanceData(metainfo={"num_detections": torch.tensor([200], device=count_device)})
        inst.bboxes = boxes
        inst.labels = (torch.rand(256, device="cuda") > 0.9).long()
        inst.scores = torch.rand(256, device="cuda")

        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            _drop_degenerate_boxes(inst)
            _cap_referees(inst, referee_label=1, max_referees=3)
        finally:
            torch.cuda.set_sync_debug_mode("default")
