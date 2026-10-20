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
