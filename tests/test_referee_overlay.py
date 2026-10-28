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


def should_resolve_a_cached_checkpoint_without_the_network(monkeypatch, tmp_path):
    """A warm cache must not reach the network.

    Stubbing `download_pinned_checkpoint` -- the only thing
    `default_checkpoint("hockeyref")` calls -- would make this tautological,
    so the real resolver runs against a pre-seeded cache with
    `torch.hub.download_url_to_file` replaced by a tripwire. That is what
    keeps 104 MB out of a Bazel target with no network tag.
    """
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    def tripwire(*args, **kwargs):
        raise AssertionError("default_checkpoint reached the network on a warm cache")

    monkeypatch.setattr(torch.hub, "download_url_to_file", tripwire)

    cached = tmp_path / "checkpoints" / Path(detector_selection.HOCKEYREF_WEIGHTS_URL).name
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"weights")
    monkeypatch.setattr(
        detector_selection, "HOCKEYREF_WEIGHTS_SHA256", hashlib.sha256(b"weights").hexdigest()
    )

    assert detector_selection.default_checkpoint("hockeyref") == str(cached)


def should_report_a_profile_without_an_automatic_checkpoint():
    """`deployed` has no work dir to search; say so rather than guess."""
    with pytest.raises(ValueError, match="No automatic checkpoint"):
        detector_selection.default_checkpoint("deployed")


def should_badge_both_hockey_roles_without_labelling_people(monkeypatch):
    from hmlib.aspen.plugins import overlay_plugin

    texts = []

    def capture(**kwargs):
        texts.append(kwargs["text"])
        return kwargs["image"]

    monkeypatch.setattr(overlay_plugin, "draw_label_box", capture)
    _render([0, 1, 2], referee_label=2, goalie_label=1)
    assert texts == ["GOALIE", "REF"]
    assert detector_selection.PROFILE_CLASSES["hockey-roles"] == ("person", "goalie", "official")
    assert detector_selection.PROFILE_INPUT_SCALE["hockey-roles"] == (1408, 544)


@pytest.mark.parametrize(
    "ignore_ref,ignore_goalie,expected",
    [
        (False, False, [True, True, True]),
        (True, False, [True, True, False]),
        (False, True, [True, False, True]),
        (True, True, [True, False, False]),
    ],
)
def should_filter_only_camera_subjects(ignore_ref, ignore_goalie, expected):
    from types import SimpleNamespace

    from hmlib.camera.play_tracker import PlayTracker

    tracks = _tracks([0, 1, 2])
    tracker = SimpleNamespace(
        _game_config={
            "detector": {"role_labels": {"referee": 2, "goalie": 1}},
            "rink": {
                "tracking": {"cam_ignore_referees": ignore_ref, "cam_ignore_goalies": ignore_goalie}
            },
        }
    )
    mask = PlayTracker._play_role_mask(tracker, tracks.instances_id, tracks)
    assert mask.tolist() == expected
    assert tracks.labels.tolist() == [0, 1, 2]
    assert tracks.instances_id.tolist() == [0, 1, 2]


def should_preserve_roles_when_pose_replaces_detection_boxes():
    from hmlib.aspen.plugins.pose_to_det_plugin import PoseToDetPlugin

    original = _tracks([0, 1, 2])
    sample = make_track_data_sample(num_frames=1)
    sample[0].pred_instances = original
    boxes = original.bboxes[[2, 0, 1]].tolist()
    pose = [{"predictions": [{"bboxes": boxes, "bbox_scores": [0.9, 0.9, 0.9]}]}]
    PoseToDetPlugin()({"data_samples": sample, "pose_results": pose})
    assert sample[0].pred_instances.labels.tolist() == [2, 0, 1]


def should_keep_goalie_labels_when_referee_labels_are_hidden(monkeypatch):
    from hmlib.aspen.plugins import overlay_plugin

    texts = []

    def capture(**kwargs):
        texts.append(kwargs["text"])
        return kwargs["image"]

    monkeypatch.setattr(overlay_plugin, "draw_label_box", capture)
    _render([0, 1, 2], referee_label=2, goalie_label=1, plot_referees=False)
    assert texts == ["GOALIE"]


def should_filter_before_learned_camera_features(monkeypatch):
    from hmlib.aspen.plugins import camera_controller_plugin as module
    from hmlib.camera.camera_transformer import CameraNorm

    plugin = module.CameraControllerPlugin(controller="rule", window=2)
    plugin._controller = "transformer"
    plugin._model = torch.nn.Linear(11, 3)
    plugin._norm = CameraNorm(420, 320, 16)
    captured = []
    original = module.build_frame_features_torch

    def capture(**kwargs):
        captured.append(kwargs["tlwh"].clone())
        return original(**kwargs)

    monkeypatch.setattr(module, "build_frame_features_torch", capture)
    tracks = _tracks([0, 1, 2])
    sample = make_track_data_sample(
        num_frames=1, ori_shape=(320, 420), pred_track_instances=[tracks]
    )
    config = {
        "detector": {"role_labels": {"referee": 2, "goalie": 1}},
        "rink": {"tracking": {"cam_ignore_referees": True, "cam_ignore_goalies": True}},
    }
    plugin({"data_samples": sample, "shared": {"game_config": config}})
    assert len(captured) == 1 and captured[0].shape[0] == 1
    assert captured[0][0, 0] == tracks.bboxes[0, 0]
    assert len(tracks) == 3


@pytest.mark.parametrize("pose_format", ["dict", "sample"])
@pytest.mark.parametrize("score_field", ["keypoint_scores", "bbox_scores", "scores"])
@pytest.mark.parametrize("excluded", [(), (1,), (2,), (1, 2)])
def should_filter_gpt_pose_features_without_mutating_published_poses(
    pose_format, score_field, excluded
):
    from types import SimpleNamespace

    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin
    from hmlib.camera.camera_gpt import CameraGPTConfig
    from hmlib.camera.camera_transformer import CameraNorm

    class CaptureCamera(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.features = None

        def forward(self, features, rink_embedding=None):
            self.features = features.clone()
            return features.new_tensor([0.5, 0.5, 0.5]).expand(*features.shape[:2], 3)

    plugin = CameraControllerPlugin(controller="rule")
    plugin._controller = "gpt"
    plugin._gpt_cfg = CameraGPTConfig(d_in=19, include_pose=True)
    plugin._norm = CameraNorm(420, 320, 16)
    plugin._gpt_model = CaptureCamera()
    tracks = _tracks([0, 1, 2, 0])
    tracks.set_metainfo({"track_mask": torch.tensor([True, True, True, False])})
    # Pose cadence/reordering need not match the tracks. Include a padded track
    # and an unmatched pose to ensure neither leaks into filtered camera input.
    boxes = torch.cat([tracks.bboxes[[2, 0, 1, 3]], torch.tensor([[800, 0, 820, 40]])])
    confidence = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.4])
    if score_field == "keypoint_scores":
        confidence = confidence[:, None].expand(-1, 2).clone()
    fields = {"bboxes": boxes, score_field: confidence, "keypoints": torch.ones(5, 2, 2)}
    snapshot = {key: value.clone() for key, value in fields.items()}
    prediction = (
        fields if pose_format == "dict" else SimpleNamespace(pred_instances=InstanceData(**fields))
    )
    poses = [{"predictions": [prediction]}]
    sample = make_track_data_sample(
        num_frames=1, ori_shape=(320, 420), pred_track_instances=[tracks]
    )
    context = {
        "data_samples": sample,
        "pose_results": poses,
        "shared": {
            "game_config": {
                "detector": {"role_labels": {"referee": 2, "goalie": 1}},
                "rink": {
                    "tracking": {
                        "cam_ignore_referees": 2 in excluded,
                        "cam_ignore_goalies": 1 in excluded,
                    }
                },
            }
        },
    }
    plugin(context)
    keep = [i for i, role in enumerate([2, 0, 1]) if role not in excluded]
    if not excluded:
        keep = list(range(5))
    expected = plugin._pose_features(
        [{"predictions": [{key: value[keep] for key, value in fields.items()}]}],
        0,
        torch.device("cpu"),
    )
    torch.testing.assert_close(plugin._gpt_model.features[0, 0, 11:], expected)
    assert context["pose_results"] is poses
    assert poses[0]["predictions"][0] is prediction
    published = fields if pose_format == "dict" else prediction.pred_instances
    for key, value in snapshot.items():
        torch.testing.assert_close(published[key], value)
    assert tracks.labels.tolist() == [0, 1, 2, 0]
    torch.testing.assert_close(tracks.bboxes, _tracks([0, 1, 2, 0]).bboxes)


@pytest.mark.parametrize("pose_indices", [[], [1, 2]])
def should_zero_camera_pose_features_when_no_pose_matches_a_kept_role(pose_indices):
    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin
    from hmlib.camera.camera_transformer import CameraNorm

    plugin = CameraControllerPlugin(controller="rule")
    plugin._norm = CameraNorm(420, 320, 16)
    tracks = _tracks([0, 1, 2])
    poses = [
        {
            "predictions": [
                {
                    "bboxes": tracks.bboxes[pose_indices],
                    "keypoint_scores": torch.ones(len(pose_indices), 2),
                }
            ]
        }
    ]
    features = plugin._pose_features(
        poses,
        0,
        torch.device("cpu"),
        role_tracks=(tracks.bboxes, torch.tensor([True, False, False])),
    )
    torch.testing.assert_close(features, torch.zeros(8))
    assert len(poses[0]["predictions"][0]["bboxes"]) == len(pose_indices)
