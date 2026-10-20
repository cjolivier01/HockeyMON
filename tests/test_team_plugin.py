"""The team affiliation trunk and its overlay rendering.

The trunk is read-only with respect to tracking: it consumes
``pred_track_instances`` and publishes ``team_results``, mutating nothing. These
tests pin that contract -- what it publishes, which tracks it skips, and that it
honours the tracker's own mask rather than working around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from mmengine.structures import InstanceData

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aspen_plugin_harness import make_track_data_sample

from hmlib.aspen.plugins.overlay_plugin import OverlayPlugin
from hmlib.aspen.plugins.team_plugin import TeamAffiliationPlugin
from hmlib.team.types import TeamLabel, TrackTeamInfo


def _tracks(count: int = 4, num_tracks: int | None = None) -> InstanceData:
    # `num_tracks` is metainfo, not a data field: InstanceData requires every
    # data field to match the instance count, and this one is a scalar.
    metainfo = (
        {"num_tracks": torch.tensor([num_tracks], dtype=torch.int64)}
        if num_tracks is not None
        else None
    )
    inst = InstanceData(metainfo=metainfo)
    inst.bboxes = torch.tensor(
        [[10.0 * i, 10.0, 10.0 * i + 40.0, 110.0] for i in range(count)],
        dtype=torch.float32,
    )
    inst.instances_id = torch.arange(count, dtype=torch.int64)
    inst.scores = torch.ones((count,), dtype=torch.float32)
    inst.labels = torch.zeros((count,), dtype=torch.int64)
    return inst


def _solid_frame(height: int = 160, width: int = 200, value: int = 90) -> torch.Tensor:
    return torch.full((1, 3, height, width), value, dtype=torch.uint8)


def should_declare_the_key_it_publishes():
    """Keys missing from output_keys() are dropped by AspenNet, silently."""
    plugin = TeamAffiliationPlugin(backend="lab")
    assert plugin.output_keys() == {"team_results"}
    # Under minimal_context these are the only keys the trunk will be handed.
    assert {"data_samples", "original_images", "pose_results"} <= plugin.input_keys()


def should_return_nothing_without_tracks():
    plugin = TeamAffiliationPlugin(backend="lab")
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1),
            "original_images": torch.zeros((1, 3, 32, 32), dtype=torch.uint8),
            "pose_results": [],
            "frame_id": 0,
        }
    )
    assert out["team_results"] == [[]]


def should_reject_an_unknown_backend():
    """A typo must not silently run a different model than was configured."""
    plugin = TeamAffiliationPlugin(backend="siglip")
    with pytest.raises(ValueError, match="unknown team.backend"):
        plugin(
            {
                "data_samples": make_track_data_sample(
                    num_frames=1, pred_track_instances=[_tracks()]
                ),
                "original_images": _solid_frame(),
                "pose_results": [],
                "frame_id": 0,
            }
        )


def should_publish_one_result_entry_per_visible_track():
    plugin = TeamAffiliationPlugin(backend="lab", warmup_samples=2, min_track_height=10)
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1, pred_track_instances=[_tracks()]),
            "original_images": _solid_frame(),
            "pose_results": [],
            "frame_id": 0,
        }
    )
    assert len(out["team_results"]) == 1
    assert [info.tracking_id for info in out["team_results"][0]] == [0, 1, 2, 3]


def should_honour_the_trackers_own_mask():
    """Slots past `num_tracks` are dead; the trunk must not label them."""
    inst = _tracks(count=4, num_tracks=2)
    plugin = TeamAffiliationPlugin(backend="lab", warmup_samples=2, min_track_height=10)
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1, pred_track_instances=[inst]),
            "original_images": _solid_frame(),
            "pose_results": [],
            "frame_id": 0,
        }
    )
    assert [info.tracking_id for info in out["team_results"][0]] == [0, 1]


def should_leave_the_track_instances_untouched():
    """The trunk publishes labels and mutates nothing upstream."""
    inst = _tracks(count=3)
    sample = make_track_data_sample(num_frames=1, pred_track_instances=[inst])
    before = set(inst.all_keys())

    plugin = TeamAffiliationPlugin(backend="lab", warmup_samples=2, min_track_height=10)
    for frame_id in range(6):
        plugin(
            {
                "data_samples": sample,
                "original_images": _solid_frame(),
                "pose_results": [],
                "frame_id": frame_id,
            }
        )
    live = sample.video_data_samples[0].pred_track_instances
    assert set(live.all_keys()) == before
    assert "track_mask" not in live.all_keys()


def should_skip_tracks_too_short_to_crop():
    """A distant player's torso is mostly ice and would poison the clusterer."""
    inst = _tracks(count=1)
    inst.bboxes = torch.tensor([[10.0, 10.0, 30.0, 25.0]], dtype=torch.float32)
    plugin = TeamAffiliationPlugin(backend="lab", warmup_samples=2, min_track_height=60)
    plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1, pred_track_instances=[inst]),
            "original_images": _solid_frame(),
            "pose_results": [],
            "frame_id": 0,
        }
    )
    assert plugin._clusterer.buffered == 0


def should_draw_team_labels_in_the_overlay():
    """End-to-end through the shared label renderer."""
    sample = make_track_data_sample(num_frames=1, pred_track_instances=[_tracks(count=2)])
    background = 200
    img = torch.full((1, 3, 240, 320), background, dtype=torch.uint8)

    plugin = OverlayPlugin(plot_teams=True, print_available=False)
    out = plugin(
        {
            "img": img,
            "data_samples": sample,
            "frame_id": 0,
            "team_results": [
                [
                    TrackTeamInfo(tracking_id=0, label=TeamLabel.HOME),
                    TrackTeamInfo(tracking_id=1, label=TeamLabel.AWAY),
                ]
            ],
        }
    )
    rendered = out["img"]
    rendered = rendered.wait() if hasattr(rendered, "wait") else rendered
    assert int((rendered != background).sum()) > 0, "plot_teams drew nothing"


def should_draw_nothing_for_unknown_labels():
    """An undecided track gets no label rather than a '?' badge."""
    sample = make_track_data_sample(num_frames=1, pred_track_instances=[_tracks(count=2)])
    background = 200
    img = torch.full((1, 3, 240, 320), background, dtype=torch.uint8)

    plugin = OverlayPlugin(plot_teams=True, print_available=False)
    out = plugin(
        {
            "img": img,
            "data_samples": sample,
            "frame_id": 0,
            "team_results": [[TrackTeamInfo(tracking_id=0, label=TeamLabel.UNKNOWN)]],
        }
    )
    rendered = out["img"]
    rendered = rendered.wait() if hasattr(rendered, "wait") else rendered
    assert int((rendered != background).sum()) == 0
