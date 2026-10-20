"""Track-level aggregation and home/away naming."""

from __future__ import annotations

import pytest

from hmlib.team.team_tracker import TeamLabelRenderer, TeamTracker
from hmlib.team.types import TeamLabel, TrackTeamInfo, name_clusters


def _observe(tracker: TeamTracker, tid: int, cluster: int, frames):
    for frame_id in frames:
        tracker.observe(
            frame_id=frame_id,
            info=TrackTeamInfo(tracking_id=tid, cluster=cluster, margin=1.0, score=1.0),
        )


def should_name_clusters_only_when_the_game_says_which_side_is_dark():
    assert name_clusters(dark_cluster=0, home_is_dark=None) == (TeamLabel.DARK, TeamLabel.LIGHT)
    assert name_clusters(dark_cluster=0, home_is_dark=True) == (TeamLabel.HOME, TeamLabel.AWAY)
    assert name_clusters(dark_cluster=0, home_is_dark=False) == (TeamLabel.AWAY, TeamLabel.HOME)
    # The mapping follows the dark cluster, not the index.
    assert name_clusters(dark_cluster=1, home_is_dark=True) == (TeamLabel.AWAY, TeamLabel.HOME)


def should_reject_a_cluster_index_that_is_not_zero_or_one():
    with pytest.raises(ValueError, match="dark_cluster"):
        name_clusters(dark_cluster=2, home_is_dark=True)


def should_report_unknown_before_the_dark_side_is_known():
    tracker = TeamTracker()
    _observe(tracker, tid=7, cluster=0, frames=range(10))
    # Centroids exist upstream but set_dark_cluster has not been called.
    assert tracker.label_for(7) == TeamLabel.UNKNOWN


def should_label_a_settled_track():
    tracker = TeamTracker(home_is_dark=True)
    tracker.set_dark_cluster(0)
    _observe(tracker, tid=7, cluster=0, frames=range(10))
    assert tracker.label_for(7) == TeamLabel.HOME

    _observe(tracker, tid=8, cluster=1, frames=range(10))
    assert tracker.label_for(8) == TeamLabel.AWAY


def should_read_dark_and_light_when_the_game_does_not_say():
    tracker = TeamTracker(home_is_dark=None)
    tracker.set_dark_cluster(1)
    _observe(tracker, tid=3, cluster=1, frames=range(10))
    assert tracker.label_for(3) == TeamLabel.DARK


def should_hold_its_label_against_a_few_stray_frames():
    tracker = TeamTracker(home_is_dark=True)
    tracker.set_dark_cluster(0)
    _observe(tracker, tid=1, cluster=0, frames=range(40))
    assert tracker.label_for(1) == TeamLabel.HOME

    # Two frames of the other team is a bad crop, not a line change.
    _observe(tracker, tid=1, cluster=1, frames=range(40, 42))
    assert tracker.label_for(1) == TeamLabel.HOME


def should_switch_when_the_contender_wins_decisively():
    tracker = TeamTracker(home_is_dark=True)
    tracker.set_dark_cluster(0)
    _observe(tracker, tid=1, cluster=0, frames=range(5))
    _observe(tracker, tid=1, cluster=1, frames=range(5, 60))
    assert tracker.label_for(1) == TeamLabel.AWAY


def should_weight_low_margin_observations_less():
    confident = TeamTracker(home_is_dark=True)
    confident.set_dark_cluster(0)
    uncertain = TeamTracker(home_is_dark=True)
    uncertain.set_dark_cluster(0)

    for frame_id in range(4):
        confident.observe(
            frame_id=frame_id,
            info=TrackTeamInfo(tracking_id=1, cluster=0, margin=1.0, score=1.0),
        )
        uncertain.observe(
            frame_id=frame_id,
            info=TrackTeamInfo(tracking_id=1, cluster=0, margin=0.05, score=1.0),
        )
    assert confident.label_for(1) == TeamLabel.HOME
    assert uncertain.label_for(1) == TeamLabel.UNKNOWN


def should_ignore_negative_tracking_ids():
    tracker = TeamTracker()
    tracker.set_dark_cluster(0)
    tracker.observe(frame_id=0, info=TrackTeamInfo(tracking_id=-1, cluster=0, margin=1.0))
    assert tracker.label_for(-1) == TeamLabel.UNKNOWN


def should_render_only_labels_it_was_given():
    renderer = TeamLabelRenderer()
    renderer.update([TrackTeamInfo(tracking_id=4, label=TeamLabel.HOME)])
    assert renderer._labels == {4: TeamLabel.HOME}


def should_reject_mismatched_draw_inputs():
    import torch

    renderer = TeamLabelRenderer()
    with pytest.raises(ValueError, match="same length"):
        renderer.draw(
            image=torch.zeros((3, 16, 16), dtype=torch.uint8),
            tracking_ids=torch.tensor([1, 2]),
            bboxes=torch.zeros((1, 4)),
        )


def should_decay_evidence_with_frame_distance():
    """Decay is what lets a track recover from an early run of bad crops.

    It does not blank a settled label -- affiliation does not lapse -- so the
    observable effect is that a stale incumbent is easier to overtake. The
    same contrary reads that flip a decayed track leave an undecayed one
    alone, which is what pins the constant: with `evidence_decay` at 1.0 both
    halves below behave identically.
    """

    def run(decay: float) -> TeamLabel:
        tracker = TeamTracker(evidence_decay=decay, min_display_evidence=1.0, switch_ratio=1.35)
        tracker.set_dark_cluster(0)
        for _ in range(4):
            tracker.observe(
                frame_id=0, info=TrackTeamInfo(tracking_id=1, cluster=0, margin=1.0, score=1.0)
            )
        # A long gap, then a short run of contrary reads.
        for frame in range(60, 64):
            tracker.observe(
                frame_id=frame,
                info=TrackTeamInfo(tracking_id=1, cluster=1, margin=1.0, score=1.0),
            )
        return tracker.label_for(1)

    assert run(0.92) is TeamLabel.LIGHT, "a stale incumbent was not overtaken"
    assert run(1.0) is TeamLabel.DARK, "evidence aged without any decay configured"


def should_hold_the_label_until_the_switch_ratio_is_exceeded():
    """A challenger below the ratio must not take the label, one above must.

    The existing hysteresis test only ever feeds the incumbent's own cluster,
    so `_select_cluster` returns before reaching the switch branch and the
    ratio is never bounded from below.
    """
    tracker = TeamTracker(evidence_decay=1.0, min_display_evidence=1.0, switch_ratio=1.35)
    tracker.set_dark_cluster(0)
    for _ in range(4):
        tracker.observe(
            frame_id=0, info=TrackTeamInfo(tracking_id=1, cluster=0, margin=1.0, score=1.0)
        )
    assert tracker.label_for(1) is TeamLabel.DARK

    # Four contrary reads: level with the incumbent, short of 1.35x.
    for frame in range(1, 5):
        tracker.observe(
            frame_id=frame, info=TrackTeamInfo(tracking_id=1, cluster=1, margin=1.0, score=1.0)
        )
    assert tracker.label_for(1) is TeamLabel.DARK, "switched before clearing the ratio"

    # Keep going until the challenger is decisively ahead.
    for frame in range(5, 20):
        tracker.observe(
            frame_id=frame, info=TrackTeamInfo(tracking_id=1, cluster=1, margin=1.0, score=1.0)
        )
    assert tracker.label_for(1) is TeamLabel.LIGHT, "never switched on decisive evidence"


def should_keep_its_aggregation_constants():
    """The three constants that decide when a badge appears and when it moves.

    They are not graph parameters, so `_BASELINE_BACKED` does not reach them
    and nothing else would notice them drifting. The behavioural tests above
    pass explicit values -- deliberately, so they test behaviour rather than
    defaults -- which leaves the defaults themselves unpinned without this.
    """
    import inspect

    defaults = {
        name: param.default
        for name, param in inspect.signature(TeamTracker.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    # Roughly two confident reads before a badge appears, and a challenger
    # must beat the incumbent by a third to take it.
    assert defaults["min_display_evidence"] == 1.0
    assert defaults["switch_ratio"] == 1.35
    # Below 1.0 or evidence never ages; at 1.0 a stale read keeps full weight.
    assert 0.0 < defaults["evidence_decay"] < 1.0
    assert defaults["evidence_decay"] == 0.92
