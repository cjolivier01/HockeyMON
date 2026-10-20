"""Referee badging in the tracking overlay, and the profile that enables it.

The referee class index is a property of the trained detector head, so it has
to travel with the detector selection rather than being configured separately.
These tests pin that coupling and the overlay behaviour on both sides of it:
badges when a two-class head is active, nothing at all otherwise.
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
from hmlib.models import detector_selection

BACKGROUND = 200


def _tracks(labels: list[int]) -> InstanceData:
    count = len(labels)
    inst = InstanceData()
    inst.bboxes = torch.tensor(
        [[40.0 + 90.0 * i, 60.0, 110.0 + 90.0 * i, 200.0] for i in range(count)],
        dtype=torch.float32,
    )
    inst.instances_id = torch.arange(count, dtype=torch.int64)
    inst.scores = torch.ones((count,), dtype=torch.float32)
    inst.labels = torch.tensor(labels, dtype=torch.int64)
    return inst


def _render(labels: list[int], **kwargs) -> torch.Tensor:
    sample = make_track_data_sample(num_frames=1, pred_track_instances=[_tracks(labels)])
    img = torch.full((1, 3, 320, 420), BACKGROUND, dtype=torch.uint8)
    plugin = OverlayPlugin(plot_tracking=True, print_available=False, **kwargs)
    out = plugin({"img": img, "data_samples": sample, "frame_id": 0})
    rendered = out["img"]
    return rendered.wait() if hasattr(rendered, "wait") else rendered


def _changed(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).sum())


def should_map_only_the_hockeyref_profile_to_a_referee_class():
    assert detector_selection.PROFILE_REFEREE_LABEL["hockeyref"] == 1
    for profile in ("deployed", "trained", "distilled"):
        assert detector_selection.PROFILE_REFEREE_LABEL[profile] is None
    # Every profile needs an entry or the overlay silently gets no label.
    assert set(detector_selection.PROFILE_REFEREE_LABEL) == set(detector_selection.PROFILES)
    assert set(detector_selection.PROFILE_CLASSES) == set(detector_selection.PROFILES)


def should_size_the_head_from_the_profile_class_list():
    """`hockeyref` is the only profile whose head is wider than one class."""
    assert len(detector_selection.PROFILE_CLASSES["hockeyref"]) == 2
    assert detector_selection.PROFILE_CLASSES["hockeyref"][1] == "referee"
    for profile in ("deployed", "trained", "distilled"):
        assert len(detector_selection.PROFILE_CLASSES[profile]) == 1


def should_badge_a_referee_track():
    plain = _render([0, 0], referee_label=1)
    with_ref = _render([0, 1], referee_label=1)
    assert _changed(with_ref, plain) > 0, "referee track drew no badge"


def should_draw_no_badge_without_a_two_class_detector():
    """On a single-class head `referee_label` is None and this is inert."""
    baseline = _render([0, 0], referee_label=None)
    same_labels = _render([0, 1], referee_label=None)
    assert _changed(baseline, same_labels) == 0


def should_respect_the_plot_referees_switch():
    enabled = _render([0, 1], referee_label=1, plot_referees=True)
    disabled = _render([0, 1], referee_label=1, plot_referees=False)
    assert _changed(enabled, disabled) > 0


def should_ignore_labels_that_do_not_match_the_track_count():
    """A malformed label tensor must not mis-badge; it should badge nothing."""
    inst = _tracks([0, 1])
    inst.set_metainfo({"stale": 1})
    plugin = OverlayPlugin(plot_tracking=True, print_available=False, referee_label=1)
    labels = plugin._track_labels(InstanceData(), torch.arange(2))
    assert labels is None


def should_find_the_best_checkpoint_under_any_metric_name(monkeypatch, tmp_path):
    """`save_best="auto"` names the file after whichever metric came first.

    Under `classwise=True` the evaluator reports per-class precision ahead of
    overall mAP, so the file is `best_coco_player_precision_epoch_N.pth` and a
    glob pinned to `bbox_mAP` silently finds nothing.
    """
    directory = tmp_path / "openmm/work_dirs/hm_hockeyref_yolov8_m_1984_736"
    directory.mkdir(parents=True)
    (directory / "best_coco_player_precision_epoch_10.pth").write_bytes(b"x")
    monkeypatch.setattr(detector_selection, "ROOT", tmp_path)

    found = detector_selection.default_checkpoint("hockeyref")
    assert found.endswith("best_coco_player_precision_epoch_10.pth")

    # When a pinned mAP checkpoint is also present it wins, so a run that set
    # save_best explicitly stays deterministic.
    (directory / "best_coco_bbox_mAP_epoch_20.pth").write_bytes(b"x")
    assert detector_selection.default_checkpoint("hockeyref").endswith(
        "best_coco_bbox_mAP_epoch_20.pth"
    )


def should_report_a_missing_checkpoint_clearly():
    """A fresh checkout has not trained this profile; say so, do not guess."""
    directory = detector_selection.ROOT / "openmm/work_dirs/hm_hockeyref_yolov8_m_1984_736"
    if list(directory.glob("best_*.pth")):
        pytest.skip("hockeyref checkpoint is present in this tree")
    with pytest.raises(ValueError, match="hm_hockeyref_yolov8_m_1984_736"):
        detector_selection.default_checkpoint("hockeyref")
