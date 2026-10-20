"""Capping referees to what can physically be on the ice.

The detector scores every box independently and cannot know how many officials
a game has. Measured on a vegas-gknights-1 warmup frame it reported five
referees where there were none, all distant gold-sweater skaters scoring
0.48-0.77, while real officials in live frames scored 0.79-0.90. The confidence
gap between the worst true positive and the best false positive was 0.014,
which is far too thin to threshold on -- so the count prior does the work
instead.
"""

from __future__ import annotations

import torch
from mmengine.structures import InstanceData

from hmlib.aspen.plugins.detector_plugin import _cap_referees

REFEREE = 1
PLAYER = 0


def _instances(labels, scores, valid=None) -> InstanceData:
    metainfo = {"num_detections": torch.tensor([valid])} if valid is not None else None
    inst = InstanceData(metainfo=metainfo)
    count = len(labels)
    inst.bboxes = torch.tensor(
        [[10.0 * i, 0.0, 10.0 * i + 8.0, 20.0] for i in range(count)], dtype=torch.float32
    )
    inst.labels = torch.tensor(labels, dtype=torch.int64)
    inst.scores = torch.tensor(scores, dtype=torch.float32)
    return inst


def should_leave_an_allowed_count_untouched():
    inst = _instances([1, 0, 1, 0], [0.9, 0.8, 0.85, 0.7])
    out = _cap_referees(inst, referee_label=REFEREE, max_referees=3)
    assert out.labels.tolist() == [1, 0, 1, 0]


def should_demote_the_lowest_scoring_surplus():
    """Five referees, cap of three: the two weakest become players."""
    inst = _instances(
        [1, 1, 1, 1, 1],
        [0.48, 0.90, 0.65, 0.87, 0.58],
    )
    out = _cap_referees(inst, referee_label=REFEREE, max_referees=3)
    # 0.90, 0.87 and 0.65 survive; 0.58 and 0.48 are demoted.
    assert out.labels.tolist() == [0, 1, 1, 1, 0]


def should_keep_the_boxes_rather_than_dropping_them():
    """A demoted detection is still a person -- losing it would lose a skater."""
    inst = _instances([1, 1, 1, 1], [0.9, 0.8, 0.7, 0.6])
    out = _cap_referees(inst, referee_label=REFEREE, max_referees=2)
    assert len(out) == 4
    assert out.labels.tolist().count(PLAYER) == 2


def should_not_count_rows_past_the_valid_prefix():
    """Static padding rows are not real detections and must not use up the cap."""
    inst = _instances([1, 1, 1, 1], [0.9, 0.8, 0.1, 0.1], valid=2)
    out = _cap_referees(inst, referee_label=REFEREE, max_referees=2)
    # Only the first two are real, so nothing is demoted.
    assert out.labels[:2].tolist() == [1, 1]


def should_handle_a_cap_of_zero():
    inst = _instances([1, 1], [0.9, 0.8])
    out = _cap_referees(inst, referee_label=REFEREE, max_referees=0)
    assert out.labels.tolist() == [0, 0]


def should_ignore_frames_with_no_referees():
    inst = _instances([0, 0, 0], [0.9, 0.8, 0.7])
    out = _cap_referees(inst, referee_label=REFEREE, max_referees=3)
    assert out.labels.tolist() == [0, 0, 0]


def should_survive_empty_detections():
    inst = InstanceData()
    inst.bboxes = torch.zeros((0, 4), dtype=torch.float32)
    inst.labels = torch.zeros((0,), dtype=torch.int64)
    inst.scores = torch.zeros((0,), dtype=torch.float32)
    assert _cap_referees(inst, referee_label=REFEREE, max_referees=3) is inst


def should_reproduce_the_measured_warmup_frame():
    """The five false positives actually observed, against two real officials.

    Scores are the ones the model produced on vegas-gknights-1 GX010080.MP4
    at 8s (no officials on the ice) and GX030080.MP4 at 40s (two officials).
    A cap of three removes two of the five warmup errors; it is a bound, not
    a classifier, which is why in-domain training data is still the fix.
    """
    warmup = _instances([1] * 5, [0.772, 0.656, 0.650, 0.584, 0.480])
    capped = _cap_referees(warmup, referee_label=REFEREE, max_referees=3)
    assert capped.labels.tolist().count(REFEREE) == 3

    live = _instances([1, 1], [0.806, 0.786])
    untouched = _cap_referees(live, referee_label=REFEREE, max_referees=3)
    assert untouched.labels.tolist() == [1, 1]
