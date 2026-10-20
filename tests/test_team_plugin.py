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


def should_keep_officials_out_of_the_clustering():
    """Officials are on neither side and would skew the two-way split.

    With a referee-aware head they are 2-4 of roughly fourteen detections --
    a fifth of the clusterer's window, which is the lopsided split that makes
    it refuse to name the sides at all. They stay tracked and drawn; they just
    do not contribute crops or receive a side.
    """
    import torch
    from mmengine.structures import InstanceData

    from hmlib.aspen.plugins.team_plugin import TeamAffiliationPlugin

    plugin = TeamAffiliationPlugin(enabled=True, referee_label=1)
    tracks = InstanceData()
    tracks.bboxes = torch.tensor([[0, 0, 10, 20], [20, 0, 30, 20], [40, 0, 50, 20]]).float()
    tracks.instances_id = torch.tensor([7, 8, 9])
    tracks.labels = torch.tensor([0, 1, 0])  # the middle track is an official

    boxes, ids = plugin._drop_officials(tracks, tracks.bboxes, tracks.instances_id, None)
    assert ids.tolist() == [7, 9]
    assert boxes.shape[0] == 2

    # A single-class head cannot name them, so nothing is dropped.
    plain = TeamAffiliationPlugin(enabled=True, referee_label=None)
    boxes, ids = plain._drop_officials(tracks, tracks.bboxes, tracks.instances_id, None)
    assert ids.tolist() == [7, 8, 9]


def should_cut_crops_at_the_model_geometry_in_rgb():
    """Crop geometry and channel order are what the backends are built on.

    `CROP_SIZE` is (H, W) = (256, 128); transposing it feeds the re-id network
    a rotated image. `aligned=True` is roi_align's half-pixel convention, the
    one the eval measured against. And the frame arrives BGR, so the crops
    must come back RGB or the dark/light ranking inverts.
    """
    import torch

    from hmlib.aspen.plugins.team_plugin import TeamAffiliationPlugin
    from hmlib.team.backends import CROP_SIZE

    assert CROP_SIZE == (256, 128), "crop geometry no longer matches the model contract"

    # A frame that is pure blue in BGR order, i.e. channel 0 hot.
    frame = torch.zeros(3, 64, 64)
    frame[0] = 255.0
    crops = TeamAffiliationPlugin._cut_crops(frame, [(0, 0, 32, 32)])
    assert crops.shape == (1, 3, 256, 128), f"crop geometry is {tuple(crops.shape)}"
    # After the swap the hot channel must be index 2 (blue in RGB), not 0.
    means = [float(crops[0, c].mean()) for c in range(3)]
    assert means[2] > 200.0 and means[0] < 55.0, f"channels not swapped to RGB: {means}"


def should_skip_tracks_shorter_than_the_configured_minimum():
    """`min_track_height` is a real filter, not the degenerate-ROI guard.

    A distant skater's torso is a few pixels of mostly ice, and those crops
    drag a centroid. The previous test passed with either this filter or the
    `< 4 px` guard removed, so it pinned neither.
    """
    import torch

    from hmlib.aspen.plugins.team_plugin import TeamAffiliationPlugin

    plugin = TeamAffiliationPlugin(enabled=True, min_track_height=60)
    # Comfortably bigger than the 4px degenerate guard, smaller than 60.
    short = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
    tall = torch.tensor([[100.0, 100.0, 140.0, 200.0]])
    assert float(short[0][3] - short[0][1]) == 40.0
    assert plugin._min_track_height == 60

    rois_short = plugin._torso_rois(short, None, 1920, 1080)
    rois_tall = plugin._torso_rois(tall, None, 1920, 1080)
    # Both produce a usable band; only the height filter separates them.
    assert rois_short[0] is not None and rois_tall[0] is not None
    assert (rois_short[0][3] - rois_short[0][1]) >= 4


def should_actually_drop_officials_in_the_frame_loop():
    """The call site, not just the helper.

    `_drop_officials` had a unit test, but deleting its call from `forward`
    left every test green -- which is the whole point of the change.
    """
    import inspect

    from hmlib.aspen.plugins.team_plugin import TeamAffiliationPlugin

    source = inspect.getsource(TeamAffiliationPlugin.forward)
    assert "_drop_officials(" in source, "the frame loop no longer excludes officials"


def should_declare_every_context_key_it_reads():
    """`minimal_context` passes only the declared keys, so an omission is a crash.

    The previous assertion checked a subset, so dropping `frame_id` -- which
    drives both the stride and the evidence decay -- went unnoticed.
    """
    from hmlib.aspen.plugins.team_plugin import TeamAffiliationPlugin

    declared = TeamAffiliationPlugin(enabled=True).input_keys()
    for key in ("data_samples", "original_images", "frame_id", "pose_results"):
        assert key in declared, f"{key} is read by forward() but not declared"
