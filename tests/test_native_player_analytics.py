"""Native causal history, PARSeq preprocessing and Aspen integration contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hmlib.aspen.player_semantics import (
    ActionHistory,
    JerseyConsensus,
    decode_parseq,
    jersey_roi,
    parseq_input,
)
from hmlib.aspen.plugins.native_analytics_plugin import (
    ParseqJerseyPlugin,
    TemporalActionPlugin,
    fresh_poses,
)
from hmlib.models.player_analytics import PlayerModel, model_contract, prepare_player_model


def _pose(x=250, score=0.9):
    return np.tile([x, 500, score], (17, 1)).astype(np.float32)


def should_require_100_causal_samples_and_normalize_against_full_canvas():
    history = ActionHistory()
    for index in range(100):
        history.observe(index * 100_000_000, _pose(), 1000, 1000)
        if index < 99:
            assert history.tensor(index * 100_000_000) is None
    tensor = history.tensor(9_900_000_000)
    assert tensor.shape == (2, 100, 17, 3)
    np.testing.assert_allclose(tensor[0, 0, 0], [-0.5, 0, 0.9])
    assert not tensor[1].any()
    assert history.samples[0][0] == 0 and history.samples[-1][0] == 9_900_000_000
    assert not history.observe(9_900_000_000, _pose(), 1000, 1000)
    assert len(history.samples) == 100
    assert history.tensor(10_050_000_001) is None
    history.observe(10_100_000_000, _pose(), 1000, 1000)
    assert len(history.samples) == 1


def should_interpolate_jitter_only_between_fresh_endpoints_and_reset_invalid_evidence():
    history = ActionHistory()
    history.observe(0, _pose(0), 1000, 1000)
    history.observe(120_000_000, _pose(120, 0.6), 1000, 1000)
    assert history.samples[-1][0] == 100_000_000
    np.testing.assert_allclose(history.samples[-1][1][0], [-0.8, 0, 0.6], atol=1e-7)
    assert len(history.samples) == 2
    invalid = _pose()
    invalid[0, 0] = np.nan
    assert not history.observe(220_000_000, invalid, 1000, 1000)
    assert len(history.samples) == 0
    history.observe(300_000_000, _pose(), 1000, 1000)
    history.observe(400_000_000, _pose(), 2000, 1000)
    assert len(history.samples) == 1
    history.observe(200_000_000, _pose(), 2000, 1000)
    assert len(history.samples) == 1
    invalid = _pose()
    invalid[:10, 2] = 0.2
    assert not history.observe(300_000_000, invalid, 2000, 1000)
    assert len(history.samples) == 0


def should_match_native_roi_geometry_and_rgb_uint8_bicubic_preprocessing():
    assert jersey_roi([10, 20, 110, 220], None, "bbox", 320, 960) == (30, 70, 90, 210)
    pose = _pose()
    pose[[5, 6, 11, 12], :2] = [[10, 20], [110, 20], [10, 220], [110, 220]]
    assert jersey_roi([0, 0, 200, 300], pose, "pose", 320, 960) == (5, 10, 115, 230)
    pose[5, 2] = 0.3
    assert jersey_roi([0, 0, 200, 300], pose, "pose", 320, 960) is None
    assert jersey_roi([0, 0, 200, 300], None, "pose", 320, 960) is None
    image = np.full((32, 128, 3), [0, 127, 255], dtype=np.uint8)
    inputs = parseq_input(image, (0, 0, 128, 32))
    np.testing.assert_allclose(inputs[:, 0, 0], [1, 127 / 127.5 - 1, -1], atol=1e-7)
    assert inputs.shape == (3, 32, 128)
    # Outside pixels retain black padding; clamping would change the model crop.
    padded = parseq_input(image, (-128, 0, 128, 32))
    assert np.all(padded[:, :, :50] == -1)


def _jersey_logits(tokens):
    logits = np.full((3, 95), -20, dtype=np.float32)
    for index, token in enumerate(tokens):
        logits[index, token] = 20
    return logits


def should_decode_all_95_tokens_and_require_digits_then_eos():
    assert decode_parseq(_jersey_logits([4, 8, 0])) == ("37", 1.0)
    assert decode_parseq(_jersey_logits([1, 0, 0])) == ("0", 1.0)
    assert decode_parseq(_jersey_logits([1, 1, 0])) == ("00", 1.0)
    for tokens in ([0, 0, 0], [11, 0, 0], [4, 8, 1]):
        assert decode_parseq(_jersey_logits(tokens)) is None
    scores = _jersey_logits([4, 8, 0])
    scores[1] = 0
    assert decode_parseq(scores)[1] == pytest.approx(1 / 95)


def should_accumulate_jersey_evidence_with_expiry_and_hysteresis():
    consensus = JerseyConsensus()
    consensus.observe(0, "0", 0.9)
    assert consensus.result(0) is None
    consensus.observe(500_000_000, "0", 0.9)
    assert consensus.result(500_000_000) == ("0", 1.0)
    consensus.observe(500_000_000, "37", 1)
    assert "37" not in consensus.evidence  # one attempt per source timestamp
    consensus.observe(1_000_000_000, "37", 0.9)
    assert consensus.result(1_000_000_000)[0] == "0"
    consensus.observe(1_500_000_000, "37", 0.9)
    assert consensus.result(1_500_000_000)[0] == "37"
    assert consensus.result(4_500_000_001) is None
    consensus.observe(0, "0", 0.9)
    assert consensus.result(0) is None


def _context(start, count=1, *, fresh=True, track_id=2**42):
    images, frames, poses = [], [], []
    for index in range(start, start + count):
        box = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        instances = SimpleNamespace(instances_id=torch.tensor([track_id]), bboxes=box)
        frames.append(
            SimpleNamespace(
                metainfo={"pts_ns": index * 100_000_000}, pred_track_instances=instances
            )
        )
        image = torch.zeros(3, 100, 100, dtype=torch.uint8)
        images.append(image)
        pose = SimpleNamespace(
            pred_instances=SimpleNamespace(
                keypoints=torch.full((1, 17, 2), 50.0),
                keypoint_scores=torch.full((1, 17), 0.9),
                bboxes=box,
            )
        )
        poses.append({"predictions": [pose] if fresh else []})
    return {
        "original_images": torch.stack(images),
        "data_samples": SimpleNamespace(video_data_samples=frames),
        "pose_results": poses,
        "shared": {"game_id": "test", "source_video_paths": ["source.mp4"]},
    }


class FakeAction:
    def __init__(self):
        self.inputs = []

    def __call__(self, inputs):
        self.inputs.append(inputs.copy())
        logits = np.full((len(inputs), 60), -20, dtype=np.float32)
        logits[:, 42] = 20
        return logits


def should_accumulate_across_batches_without_leaking_future_actions_into_earlier_frames():
    plugin = TemporalActionPlugin()
    model = plugin.model = FakeAction()
    for index in range(0, 96, 4):
        assert plugin.forward(_context(index, 4))["action_results"] == [[], [], [], []]
    output = plugin.forward(_context(96, 4))["action_results"]
    assert output[:3] == [[], [], []]
    assert output[3][0]["tracking_id"] == 2**42
    assert output[3][0]["label_index"] == 42
    assert output[3][0]["window_start_ns"] == 0
    assert len(model.inputs) == 1
    np.testing.assert_allclose(model.inputs[0][0, 0, :, :, :2], 0)
    assert plugin.forward(_context(103))["action_results"] == [[]]  # 400ms gap resets history
    assert len(plugin.states[2**42]["history"].samples) == 1
    assert len(model.inputs) == 1
    changed = _context(104)
    changed["shared"]["source_video_paths"] = ["different.mp4"]
    plugin.forward(changed)
    assert len(plugin.states[2**42]["history"].samples) == 1


def should_never_use_held_poses_as_action_evidence_or_pose_guided_jersey_crops():
    action = TemporalActionPlugin()
    action.model = FakeAction()
    jersey = ParseqJerseyPlugin(roi_mode="pose")
    jersey.model = lambda inputs: pytest.fail("A held pose must not trigger jersey inference")
    for index in range(101):
        context = _context(index, fresh=False)
        context["pose_display_results"] = _context(index)["pose_results"]
        assert action.forward(context)["action_results"] == [[]]
        assert jersey.forward(context)["jersey_results"] == [[]]
    assert action.model.inputs == []


def should_support_bbox_jerseys_without_pose_and_return_native_consensus():
    plugin = ParseqJerseyPlugin()
    calls = []

    def infer(inputs):
        calls.append(inputs.shape)
        logits = _jersey_logits([4, 8, 0])
        logits[2, 1] = 20 - np.log(9)  # 0.9 sequence confidence needs repeat evidence.
        return np.stack([logits for _ in inputs])

    plugin.model = infer
    assert plugin.forward(_context(0, fresh=False))["jersey_results"] == [[]]
    assert plugin.forward(_context(1, fresh=False))["jersey_results"] == [[]]
    result = plugin.forward(_context(5, fresh=False))["jersey_results"][0][0]
    assert result.number == 37 and result.tracking_id == 2**42
    assert calls == [(1, 3, 32, 128), (1, 3, 32, 128)]


def should_reject_wrong_pose_topology_and_preserve_one_to_one_association():
    context = _context(0)
    pose = context["pose_results"][0]
    tracks = {10: np.array([0, 0, 100, 100]), 20: np.array([0, 0, 100, 100])}
    assert list(fresh_poses(pose, tracks)) == [10]
    pose["predictions"][0].pred_instances.keypoints = torch.zeros(1, 14, 2)
    with pytest.raises(ValueError, match="COCO17"):
        fresh_poses(pose, tracks)


def should_validate_pinned_cached_bytes_before_model_loading(tmp_path):
    path = tmp_path / "parseq-hockey-cvprw2024-df48ad7463aa89a5.onnx"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_player_model("jersey", tmp_path)
    assert "CC-BY-NC" in model_contract("jersey")["source"]["license"]
    assert len(model_contract("action")["labels"]) == 60


def should_reject_incompatible_custom_models_and_explicit_cuda_failures(tmp_path, monkeypatch):
    import onnxruntime as ort

    path = tmp_path / "custom.onnx"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDA provider unavailable"):
        PlayerModel("action", str(path), "cuda").ensure_session()
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(type="tensor(float)", shape=["N", 2, 100, 14, 3])],
        get_outputs=lambda: [SimpleNamespace(type="tensor(float)", shape=["N", 60])],
    )
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    with pytest.raises(ValueError, match="native|float32"):
        PlayerModel("action", str(path), "cpu").ensure_session()
