"""Referee badging in the tracking overlay, and the profile that enables it.

The referee class index is a property of the trained detector head, so it has
to travel with the detector selection rather than being configured separately.
These tests pin that coupling and the overlay behaviour on both sides of it:
badges when a two-class head is active, nothing at all otherwise.
"""

from __future__ import annotations

import hashlib
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


def should_verify_the_pinned_checkpoint_digest(monkeypatch, tmp_path):
    """The weights are fetched by digest, not found on the training machine.

    `openmm/work_dirs/` only exists where the fine-tune ran, so a profile
    resolved from it is unusable anywhere else. The digest is what makes the
    download trustworthy: an interrupted transfer leaves a plausible file
    behind, and half a checkpoint loads as a model that detects nothing rather
    than raising.
    """
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    payload = b"weights"
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def fake_download(url, target, progress=True):
        calls.append(url)
        Path(target).write_bytes(payload)

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)

    first = detector_selection.download_pinned_checkpoint("http://x/w.pth", digest, "w.pth")
    assert Path(first).read_bytes() == payload
    assert len(calls) == 1

    # A good cache is reused rather than fetched again.
    assert detector_selection.download_pinned_checkpoint("http://x/w.pth", digest, "w.pth") == first
    assert len(calls) == 1

    # A corrupted cache is replaced, not trusted.
    Path(first).write_bytes(b"truncated")
    assert detector_selection.download_pinned_checkpoint("http://x/w.pth", digest, "w.pth") == first
    assert Path(first).read_bytes() == payload
    assert len(calls) == 2

    # A download that does not match the pin raises, and leaves nothing behind
    # for the next run to mistake for a good copy.
    with pytest.raises(ValueError, match="expected"):
        detector_selection.download_pinned_checkpoint("http://x/w.pth", "0" * 64, "w.pth")
    assert not Path(first).exists()


def should_resolve_hockeyref_from_the_published_asset(monkeypatch):
    """The profile uses the pin, not whatever happens to be on this machine."""
    seen = {}

    def fake_download(url, sha256, filename):
        seen.update(url=url, sha256=sha256, filename=filename)
        return "/cache/" + filename

    monkeypatch.setattr(detector_selection, "download_pinned_checkpoint", fake_download)
    assert detector_selection.default_checkpoint("hockeyref") == (
        "/cache/" + Path(detector_selection.HOCKEYREF_WEIGHTS_URL).name
    )
    assert seen["url"] == detector_selection.HOCKEYREF_WEIGHTS_URL
    assert seen["sha256"] == detector_selection.HOCKEYREF_WEIGHTS_SHA256


def should_not_reach_the_network_from_a_unit_test(monkeypatch):
    """Resolving the profile must not fetch 104 MB from a test.

    This replaces a test that guarded the deleted work-dirs glob. Its skip
    fired only when `openmm/work_dirs/.../best_*.pth` existed -- on the one
    machine that ran the fine-tune, the exact condition the release asset was
    introduced to stop depending on. Anywhere else it did not skip, and the
    profile now downloads instead of raising, so the test both failed and
    pulled the asset into a target with no network tag.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("default_checkpoint must not download during tests")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(detector_selection, "_file_sha256", lambda path: "deadbeef")

    # A cache hit resolves without touching the network at all.
    monkeypatch.setattr(
        detector_selection, "download_pinned_checkpoint", lambda url, sha256, filename: "/cached"
    )
    assert detector_selection.default_checkpoint("hockeyref") == "/cached"


def should_report_a_profile_without_an_automatic_checkpoint():
    """`deployed` has no work dir to search; say so rather than guess."""
    with pytest.raises(ValueError, match="No automatic checkpoint"):
        detector_selection.default_checkpoint("deployed")
