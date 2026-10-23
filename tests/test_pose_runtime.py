"""Exercise source-time scheduling and the plugin's fresh/display split."""

from types import SimpleNamespace

import pytest
import torch

from hmlib.aspen.plugins.pose_plugin import PosePlugin
from hmlib.aspen.pose_runtime import PoseRuntime, pose_settings


def should_schedule_per_track_cadence_and_admit_new_tracks_immediately():
    runtime = PoseRuntime()
    assert runtime.due(0, [2**42], source="game", rate_hz=10) == [2**42]
    assert runtime.due(1 / 30, [2**42], source="game", rate_hz=10) == []
    assert runtime.due(2 / 30, [2**42, 7], source="game", rate_hz=10) == [7]
    assert runtime.due(3 / 30, [2**42, 7], source="game", rate_hz=10) == [2**42]
    assert runtime.due(4 / 30, [2**42, 7], source="game", rate_hz=10) == []


def should_bound_inference_and_retire_disappeared_tracks():
    runtime = PoseRuntime()
    first = runtime.due(0, list(range(400)), source="game", rate_hz=None)
    assert len(first) == 32
    second = runtime.due(1 / 29.97, list(range(400)), source="game", rate_hz=None)
    assert second == list(range(32, 64))
    assert len(runtime.next_due) <= 256
    assert runtime.due(2 / 29.97, [3], source="game", rate_hz=None) == [3]
    assert set(runtime.next_due) == {3}


def should_reproject_only_for_drawing_preserve_confidence_and_expire_at_250_ms():
    runtime = PoseRuntime()
    runtime.due(0, [1], source="game", rate_hz=1)
    original_points = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    scores = torch.tensor([0.8, 0.9])
    runtime.observe(1, 0, torch.tensor([0.0, 0.0, 10.0, 10.0]), original_points, scores)
    new_box = {1: torch.tensor([10.0, 20.0, 30.0, 40.0])}
    held = runtime.drawing(0.2, new_box, continuous=True)[0][1]
    torch.testing.assert_close(held.keypoints, torch.tensor([[14.0, 26.0], [18.0, 30.0]]))
    torch.testing.assert_close(held.scores, scores)
    torch.testing.assert_close(original_points, torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    assert held.timestamp == 0
    assert runtime.drawing(0.2, new_box, continuous=False) == []
    assert runtime.drawing(0.251, new_box, continuous=True) == []
    runtime.due(0.22, [], source="game", rate_hz=1)
    assert runtime.observations == {}


@pytest.mark.parametrize("new_source,new_time", [("other", 0.05), ("game", 0.0)])
def should_reset_display_and_cadence_on_source_changes_or_backwards_time(new_source, new_time):
    runtime = PoseRuntime()
    runtime.due(0.04, [1], source="game", rate_hz=1)
    runtime.observe(1, 0.04, torch.tensor([0.0, 0.0, 10.0, 10.0]), torch.ones(2, 2), torch.ones(2))
    assert runtime.due(new_time, [1], source=new_source, rate_hz=1) == [1]
    assert runtime.observations == {}


def should_reject_invalid_policy_and_timestamp_instead_of_silently_falling_back():
    for value in [0, -1, True, float("inf"), "not-a-rate"]:
        with pytest.raises((ValueError, TypeError)):
            pose_settings({"pipeline": {"player-analytics": {"pose": {"rate-hz": value}}}})
    assert (
        pose_settings({"pipeline": {"player-analytics": {"pose": {"rate-hz": "fps"}}}})[0] is None
    )
    with pytest.raises(ValueError, match="source timestamps"):
        PoseRuntime().due(float("nan"), [1], source="game", rate_hz=10)


class Instances(SimpleNamespace):
    def __getitem__(self, indices):
        return Instances(**{key: value[indices] for key, value in vars(self).items()})


class Visualizer:
    def __init__(self):
        self.drawn = []

    def add_datasample(self, **kwargs):
        self.drawn.append(kwargs["data_sample"].pred_instances.keypoints.clone())
        return kwargs["image"]


def pose_context(count, *, start=0, rate=10, continuous=True):
    frames = []
    for index in range(count):
        frame_id = start + index
        instances = Instances(
            bboxes=torch.tensor([[frame_id + 1.0, 2.0, frame_id + 11.0, 12.0]]),
            instances_id=torch.tensor([2**42]),
            labels=torch.tensor([0]),
            scores=torch.tensor([0.9]),
        )
        frames.append(
            SimpleNamespace(metainfo={"frame_id": frame_id}, pred_track_instances=instances)
        )
    visualizer = Visualizer()
    context = {
        "original_images": torch.zeros(count, 16, 32, 3),
        "data_samples": SimpleNamespace(video_data_samples=frames),
        "fps": 30.0,
        "frame_id": start,
        "pose_inferencer": SimpleNamespace(
            inferencer=SimpleNamespace(visualizer=visualizer), filter_args={}
        ),
        "shared": {
            "game_id": "fixture",
            "plot_pose": True,
            "game_config": {
                "pipeline": {
                    "player-analytics": {
                        "pose": {"rate-hz": rate, "continuous-drawing": continuous}
                    }
                }
            },
        },
    }
    return context, visualizer


def fake_inference(context):
    frame = context["data_samples"].video_data_samples[0]
    boxes = frame.pred_track_instances.bboxes
    points = boxes[:, None, :2] + torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    sample = SimpleNamespace(
        pred_instances=Instances(
            bboxes=boxes.clone(), keypoints=points, keypoint_scores=torch.ones(len(boxes), 2)
        )
    )
    return {"pose_results": [{"predictions": [sample]}]}


def should_keep_held_skeletons_out_of_jersey_and_action_evidence(monkeypatch):
    plugin = PosePlugin()
    calls = []

    def infer(context):
        calls.append(context["data_samples"].video_data_samples[0].metainfo["frame_id"])
        return fake_inference(context)

    monkeypatch.setattr(plugin, "_forward_inference", infer)
    context, visualizer = pose_context(9)
    original_tracks = list(context["data_samples"].video_data_samples)
    result = plugin.forward(context)
    assert calls == [0, 3, 6]
    assert [bool(frame["predictions"]) for frame in result["pose_results"]] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert len(visualizer.drawn) == 9
    assert result["pose_display_metadata"][1]["poses"][0] == {
        "track_id": 2**42,
        "observed_time": 0.0,
        "held": True,
    }
    assert context["data_samples"].video_data_samples == original_tracks
    torch.testing.assert_close(
        visualizer.drawn[1] - visualizer.drawn[0], torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    )


def should_disable_hold_without_disabling_fresh_inference(monkeypatch):
    plugin = PosePlugin()
    monkeypatch.setattr(plugin, "_forward_inference", fake_inference)
    context, visualizer = pose_context(4, continuous=False)
    result = plugin.forward(context)
    assert len(visualizer.drawn) == 2
    assert result["pose_display_metadata"][1]["poses"] == []
    context, visualizer = pose_context(4, start=4, rate="fps")
    result = plugin.forward(context)
    assert all(frame["predictions"] for frame in result["pose_results"])
    assert len(visualizer.drawn) == 4


def should_schedule_source_pts_without_frame_rate(monkeypatch):
    plugin = PosePlugin()
    monkeypatch.setattr(plugin, "_forward_inference", fake_inference)
    context, _ = pose_context(4)
    del context["fps"]
    del context["frame_id"]
    for index, frame in enumerate(context["data_samples"].video_data_samples):
        frame.metainfo = {"pts_ns": index * 40_000_000}
    result = plugin.forward(context)
    assert [bool(frame["predictions"]) for frame in result["pose_results"]] == [
        True,
        False,
        False,
        True,
    ]


def should_report_missing_timing_and_use_fresh_inference(monkeypatch, caplog):
    plugin = PosePlugin()
    marker = {"pose_results": []}
    monkeypatch.setattr(plugin, "_forward_inference", lambda context: marker)
    context, _ = pose_context(1)
    del context["frame_id"]
    del context["data_samples"]
    assert plugin.forward(context) is marker
    assert "source timing unavailable" in caplog.text


def should_compact_stream_wrapped_tracks_without_reusing_padded_validity_metadata(monkeypatch):
    from mmengine.structures import InstanceData

    from hmlib.utils.gpu import StreamTensorX

    plugin = PosePlugin()
    captured = []

    def infer(context):
        instances = context["data_samples"].video_data_samples[0].pred_track_instances
        captured.append(instances)
        return fake_inference(context)

    monkeypatch.setattr(plugin, "_forward_inference", infer)
    context, _ = pose_context(1)
    source = InstanceData(
        metainfo={"track_mask": torch.tensor([True, False]), "num_tracks": torch.tensor([1])},
        instances_id=StreamTensorX(torch.tensor([2**42, -1])),
        bboxes=StreamTensorX(torch.tensor([[1.0, 2.0, 11.0, 12.0], [0.0, 0.0, 0.0, 0.0]])),
        labels=StreamTensorX(torch.tensor([0, 0])),
        scores=StreamTensorX(torch.tensor([0.9, 0.0])),
    )
    context["data_samples"].video_data_samples[0].pred_track_instances = source
    plugin.forward(context)
    assert captured[0].track_mask.tolist() == [True]
    assert len(captured[0].bboxes) == 1
    assert len(source.track_mask) == 2
    assert isinstance(source.instances_id, StreamTensorX)
