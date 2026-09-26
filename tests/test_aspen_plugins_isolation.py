from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

if torch is not None:
    from hmlib.aspen.plugins.base import DeleteKey, Plugin
    from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
    from hmlib.jersey.number_classifier import TrackJerseyInfo
    from hmlib.tracking_utils.action_dataframe import ActionDataFrame
    from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
    from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
    from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
    from hmlib.utils.gpu import wrap_tensor
else:
    DeleteKey = object  # type: ignore[assignment]
    Plugin = object  # type: ignore[assignment]
    CameraTrackingDataFrame = None  # type: ignore[assignment]
    TrackJerseyInfo = None  # type: ignore[assignment]
    ActionDataFrame = None  # type: ignore[assignment]
    DetectionDataFrame = None  # type: ignore[assignment]
    PoseDataFrame = None  # type: ignore[assignment]
    TrackingDataFrame = None  # type: ignore[assignment]
    wrap_tensor = None  # type: ignore[assignment]

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

if torch is not None:
    from aspen_plugin_harness import (
        FakeCompose,
        FakeDetectorModel,
        FakeEndZones,
        FakePlayTrackerRuntime,
        FakeProgressBar,
        FakePoseInferencer,
        FakeShower,
        FakeVideoOutput,
        FakeVideoOutputPreparer,
        discover_production_plugin_classes,
        install_fake_module,
        make_det_sample,
        make_instance_data,
        make_pose_results,
        make_track_data_sample,
    )
else:
    FakeCompose = None
    FakeDetectorModel = None
    FakeEndZones = None
    FakePlayTrackerRuntime = None
    FakeProgressBar = None
    FakePoseInferencer = None
    FakeShower = None
    FakeVideoOutput = None
    FakeVideoOutputPreparer = None
    discover_production_plugin_classes = None
    install_fake_module = None
    make_det_sample = None
    make_instance_data = None
    make_pose_results = None
    make_track_data_sample = None

REPO_ROOT = Path(__file__).resolve().parents[1]
_TORCH_MODULE_BASE = torch.nn.Module if torch is not None else object
requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


class HarnessModel(_TORCH_MODULE_BASE):
    def __init__(self, **kwargs: Any):
        if torch is None:  # pragma: no cover - guarded by requires_torch
            raise RuntimeError("HarnessModel requires torch")
        super().__init__()
        self.kwargs = kwargs
        self.eval_called = False
        self.to_device = None

    def eval(self):  # type: ignore[override]
        self.eval_called = True
        return super().eval()

    def to(self, device: Any):  # type: ignore[override]
        self.to_device = device
        return self


class HarnessTrackingDelegate(Plugin):
    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        return {
            "data_samples": context["data_samples"],
            "nr_tracks": 2,
            "max_tracking_id": 9,
        }


class HarnessTrackerRuntime:
    def __init__(self, result: dict[str, torch.Tensor]):
        self.result = result
        self.payloads: list[dict[str, torch.Tensor]] = []

    def track(self, payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.payloads.append(payload)
        return dict(self.result)


def _assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert isinstance(actual, torch.Tensor)
    assert torch.equal(actual.detach().cpu(), expected.detach().cpu())


def _maybe_enable_cuda_graph(plugin: Any, enabled: bool) -> None:
    if not enabled:
        return
    accepted = plugin.set_cuda_graph_enabled(True)
    if accepted:
        assert getattr(plugin, "_cuda_graph_enabled", False) is True
    else:
        assert getattr(plugin, "_cuda_graph_enabled", False) is False


def _write_detection_csv(path: Path, frame_id: int = 1) -> None:
    df = DetectionDataFrame(output_file=str(path), write_interval=1)
    inst = make_instance_data(
        bboxes=torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32),
        scores=torch.tensor([0.95], dtype=torch.float32),
        labels=torch.tensor([3], dtype=torch.long),
    )
    sample = make_det_sample(frame_id=frame_id, pred_instances=inst)
    df.add_frame_sample(frame_id, sample)
    df.close()


def _write_tracking_csv(path: Path, frame_id: int = 1) -> None:
    df = TrackingDataFrame(output_file=str(path), input_batch_size=1, write_interval=1)
    inst = make_instance_data(
        bboxes=torch.tensor([[5.0, 6.0, 25.0, 26.0]], dtype=torch.float32),
        scores=torch.tensor([0.9], dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        instances_id=torch.tensor([7], dtype=torch.long),
    )
    sample = make_det_sample(frame_id=frame_id, pred_track_instances=inst)
    df.add_frame_sample(frame_id, sample)
    df.close()


def _write_pose_csv(path: Path, frame_id: int = 1) -> None:
    df = PoseDataFrame(output_file=str(path), write_interval=1)
    df.add_frame_sample(frame_id=frame_id, pose_item=make_pose_results(num_frames=1)[0])
    df.close()


def _case_action_factory(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.action_factory_plugin import ActionRecognizerFactoryPlugin

    label_map_path = tmp_path / "labels.txt"
    label_map_path.write_text("idle\nskate\n", encoding="utf-8")

    install_fake_module(monkeypatch, "mmaction")
    install_fake_module(
        monkeypatch,
        "mmaction.apis",
        init_recognizer=lambda cfg, ckpt, dev: {
            "cfg": cfg,
            "ckpt": ckpt,
            "dev": dev,
        },
    )

    plugin = ActionRecognizerFactoryPlugin(
        action_config="action.py",
        action_checkpoint="weights.pth",
        device="cpu",
        label_map_path=str(label_map_path),
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({})
    assert out["action_recognizer"]["cfg"] == "action.py"
    assert out["action_recognizer"]["dev"] == "cpu"
    assert out["action_label_map"] == ["idle", "skate"]


def _case_action_pose(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.action_pose_plugin import ActionFromPosePlugin

    def _inference_skeleton(_recognizer, _poses, _shape):
        return SimpleNamespace(pred_score=torch.tensor([0.1, 0.9], dtype=torch.float32))

    install_fake_module(monkeypatch, "mmaction")
    install_fake_module(monkeypatch, "mmaction.apis", inference_skeleton=_inference_skeleton)

    track_inst = make_instance_data(
        bboxes=torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32),
        scores=torch.tensor([0.95], dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        instances_id=torch.tensor([5], dtype=torch.long),
    )
    track_data_sample = make_track_data_sample(
        num_frames=2,
        pred_track_instances=[track_inst, track_inst],
    )
    pose_kpts = torch.tensor(
        [
            [
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
            ]
        ],
        dtype=torch.float32,
    )
    pose_scores = torch.ones((1, 4), dtype=torch.float32)
    pose_sample = SimpleNamespace(
        pred_instances=make_instance_data(
            bboxes=torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32),
            keypoints=pose_kpts,
            keypoint_scores=pose_scores,
        )
    )
    pose_results = [{"predictions": [pose_sample]}, {"predictions": [pose_sample]}]
    plugin = ActionFromPosePlugin(top_k=2, score_threshold=0.2)
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    monkeypatch.setattr(
        plugin, "_map_tracks_to_pose_indices", lambda inst, pose_kpts: torch.tensor([0])
    )
    out = plugin(
        {
            "data_samples": track_data_sample,
            "original_images": torch.zeros((2, 3, 16, 16), dtype=torch.float32),
            "pose_results": pose_results,
            "action_recognizer": object(),
            "action_label_map": ["idle", "skate"],
        }
    )
    assert len(out["action_results"]) == 2
    assert out["action_results"][0][0]["tracking_id"] == 5
    assert out["action_results"][0][0]["label"] == "skate"


def _case_boundaries(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.boundaries_plugin import BoundariesPlugin

    updates: list[tuple[str, dict[str, Any]]] = []

    def _update_pipeline_item(pipeline, kind, params):
        updates.append((kind, dict(params)))
        if pipeline:
            pipeline[0].update(params)
        return True

    monkeypatch.setattr(
        "hmlib.aspen.plugins.boundaries_plugin.update_pipeline_item",
        _update_pipeline_item,
    )
    model = SimpleNamespace(post_detection_pipeline=[{"type": "IceRinkSegmBoundaries"}])
    plugin = BoundariesPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.forward({"model": model, "game_id": "game-1", "plot_ice_mask": True})
    assert plugin.factory_complete is True
    assert updates[0][0] == "IceRinkSegmBoundaries"
    assert updates[0][1]["game_id"] == "game-1"


def _case_camera_controller(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin

    plugin = CameraControllerPlugin(controller="rule")
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"data_samples": make_track_data_sample(num_frames=1)})
    assert out == {}


def _case_camera_train(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.camera_train_plugin import CameraTrainPlugin

    class _Dataset:
        def __init__(self, **kwargs: Any):
            self.norm = None

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return SimpleNamespace(
                x=torch.full((4,), float(index + 1), dtype=torch.float32),
                y=torch.zeros((3,), dtype=torch.float32),
            )

    class _Model(torch.nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.linear = torch.nn.Linear(d_in, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    def _data_loader(dataset, **kwargs):
        return [dataset[0], dataset[1]]

    monkeypatch.setattr("hmlib.aspen.plugins.camera_train_plugin.CameraPanZoomDataset", _Dataset)
    monkeypatch.setattr("hmlib.aspen.plugins.camera_train_plugin.CameraPanZoomTransformer", _Model)
    monkeypatch.setattr("hmlib.aspen.plugins.camera_train_plugin.DataLoader", _data_loader)
    monkeypatch.setattr(
        "hmlib.aspen.plugins.camera_train_plugin.random_split",
        lambda dataset, splits: (dataset, dataset),
    )
    monkeypatch.setattr(
        "hmlib.aspen.plugins.camera_train_plugin.pack_checkpoint",
        lambda model, norm, window: {"window": window, "state_dict": model.state_dict()},
    )
    monkeypatch.setattr(
        "hmlib.aspen.plugins.camera_train_plugin.torch.save",
        lambda obj, path: Path(path).write_text(json.dumps({"window": obj["window"]})),
    )

    output_path = tmp_path / "camera.pt"
    plugin = CameraTrainPlugin(
        tracking_csv="tracking.csv",
        camera_csv="camera.csv",
        out=str(output_path),
        epochs=1,
        batch_size=1,
        lr=1e-3,
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({})
    assert out["camera_model_path"] == str(output_path)
    assert output_path.exists()


def _case_dataloader(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.dataloader_plugin import DataLoaderPlugin

    plugin = DataLoaderPlugin(dataloader=[{"x": torch.tensor(3)}], batch_index_key="batch_index")
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({})
    assert int(out["x"].item()) == 3
    assert out["batch_index"] == 0


def _case_debug_rgb_stats(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.debug_rgb_stats_plugin import RgbStatsCheckPlugin

    monkeypatch.setattr(
        "hmlib.aspen.plugins.debug_rgb_stats_plugin.MOTLoadVideoWithOrig.check_rgb_stats",
        lambda **kwargs: (True, False),
    )
    plugin = RgbStatsCheckPlugin(log_on_change=False)
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "debug_rgb_stats": {"stitch": {"stitched": {"mean": [1, 2, 3]}}},
            "original_images": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        }
    )
    assert out["debug_rgb_stats_checks"]["stitch.stitched"]["changed"] is True


def _case_detector_factory(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.detector_factory_plugin import DetectorFactoryPlugin

    fake_model = FakeDetectorModel()
    install_fake_module(monkeypatch, "mmyolo")
    install_fake_module(
        monkeypatch,
        "mmyolo.registry",
        MODELS=SimpleNamespace(build=lambda cfg: fake_model),
    )
    plugin = DetectorFactoryPlugin(detector={"type": "FakeDetector"})
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"device": torch.device("cpu")})
    assert out["detector_model"] is fake_model
    assert fake_model.init_weights_called is True
    assert fake_model.eval_called is True
    assert fake_model.to_device == torch.device("cpu")


def _case_detector_inference(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.detector_plugin import DetectorInferencePlugin

    class _Result:
        def __init__(self, pred_instances):
            self.pred_instances = pred_instances

    class _Detector:
        def predict(self, inputs, data_samples):
            assert inputs.shape[0] == len(data_samples)
            return [
                _Result(
                    make_instance_data(
                        bboxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32),
                        scores=torch.tensor([0.8], dtype=torch.float32),
                        labels=torch.tensor([1], dtype=torch.long),
                    )
                )
                for _ in data_samples
            ]

    track_data_sample = make_track_data_sample(num_frames=2)
    plugin = DetectorInferencePlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "inputs": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
            "data_samples": track_data_sample,
            "detector_model": _Detector(),
            "fp16": False,
        }
    )
    assert out == {}
    assert hasattr(track_data_sample[0], "pred_instances")
    assert int(track_data_sample[1].pred_instances.labels[0].item()) == 1
    assert int(track_data_sample[0].metainfo["frame_id"]) == 0


def _case_detector_compare(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    import json

    from hmlib.aspen.plugins.detector_compare_plugin import DetectorComparePlugin

    class _Result:
        def __init__(self, pred_instances):
            self.pred_instances = pred_instances

    class _Detector:
        def predict(self, inputs, data_samples):
            return [
                _Result(
                    make_instance_data(
                        bboxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]], dtype=torch.float32),
                        scores=torch.tensor([0.8], dtype=torch.float32),
                        labels=torch.tensor([0], dtype=torch.long),
                    )
                )
                for _ in data_samples
            ]

    output_dir = tmp_path / "detector_comparison"
    # Only the tracking model is compared, so no extra detector is built here;
    # preview rendering is exercised by the detector comparison tests instead.
    plugin = DetectorComparePlugin(
        models={"deployed": {"detector": {"type": "FakeDetector"}}},
        selected="deployed",
        output_dir=str(output_dir),
        sample_every=1,
        preview_frames=0,
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    track_data_sample = make_track_data_sample(num_frames=1)
    out = plugin(
        {
            "inputs": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "data_samples": track_data_sample,
            "detector_model": _Detector(),
            "fp16": False,
            "device": torch.device("cpu"),
            "game_id": "game-1",
            "work_dir": str(tmp_path),
        }
    )
    assert out == {}
    plugin.finalize()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["frames"] == 1
    assert summary["frame_ids"] == [0]
    assert summary["metadata"]["game_id"] == "game-1"
    assert summary["models"]["deployed"]["mean_person_detections"] == 1.0


def _case_ice_boundaries(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.ice_rink_boundaries_plugins import IceRinkSegmBoundariesPlugin

    class _Segm:
        def __call__(self, payload):
            return {
                "det_bboxes": payload["det_bboxes"][:1],
                "labels": payload["labels"][:1],
                "scores": payload["scores"][:1],
                "num_detections": torch.tensor([1], dtype=torch.int32),
            }

    det_inst = make_instance_data(
        bboxes=torch.tensor([[0.0, 0.0, 4.0, 4.0], [5.0, 5.0, 8.0, 8.0]], dtype=torch.float32),
        scores=torch.tensor([0.4, 0.2], dtype=torch.float32),
        labels=torch.tensor([1, 2], dtype=torch.long),
    )
    track_data_sample = make_track_data_sample(num_frames=1, pred_instances=[det_inst])
    plugin = IceRinkSegmBoundariesPlugin(
        raise_bbox_center_by_height_ratio=0.0,
        lower_bbox_bottom_by_height_ratio=0.0,
        max_detections_in_mask=1,
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin._segm = _Segm()
    out = plugin({"data_samples": track_data_sample, "rink_profile": {"mask": True}})
    assert out == {}
    assert track_data_sample[0].pred_instances.bboxes.shape[0] == 1
    assert int(track_data_sample[0].pred_instances.metainfo["num_detections"].item()) == 1


def _case_ice_config(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.ice_rink_boundaries_plugins import IceRinkSegmConfigPlugin

    monkeypatch.setattr(
        "hmlib.segm.ice_rink.configure_ice_rink_mask",
        lambda **kwargs: {"game_id": kwargs["game_id"], "shape": tuple(kwargs["expected_shape"])},
    )
    plugin = IceRinkSegmConfigPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1, ori_shape=(200, 300)),
            "original_images": wrap_tensor(torch.zeros((1, 20, 30, 3), dtype=torch.float32)),
            "shared": {"game_id": "game-1"},
        }
    )
    assert out["rink_profile"]["game_id"] == "game-1"
    assert out["rink_profile"]["shape"] == (20, 30)


def _case_image_prep(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.image_prep_plugin import ImagePrepPlugin

    plugin = ImagePrepPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "inputs": torch.arange(48, dtype=torch.float32).reshape(1, 4, 4, 3),
            "device": torch.device("cpu"),
        }
    )
    assert out["inputs"].shape == (1, 3, 4, 4)
    _assert_tensor_close(out["inputs"], out["detection_image"])


def _case_jersey_koshkina(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.jersey_koshkina_plugin import KoshkinaJerseyNumberPlugin

    monkeypatch.setattr(KoshkinaJerseyNumberPlugin, "_ensure_legibility", lambda self, device: None)
    monkeypatch.setattr(KoshkinaJerseyNumberPlugin, "_ensure_parseq", lambda self, device_str: None)
    plugin = KoshkinaJerseyNumberPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1),
            "original_images": torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            "pose_results": [],
        }
    )
    assert out["jersey_results"] == [[]]


def _case_jersey_pose(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.jersey_pose_plugin import JerseyNumberFromPosePlugin

    monkeypatch.setattr(JerseyNumberFromPosePlugin, "_ensure_inferencer", lambda self: None)
    plugin = JerseyNumberFromPosePlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin._inferencer = lambda img, progress_bar=False: []  # type: ignore[assignment]
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1),
            "original_images": torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            "pose_results": [],
        }
    )
    assert out["jersey_results"] == [[]]


def _case_join(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.join_plugin import JoinPlugin

    plugin = JoinPlugin(required_plugins=["a", "b"], output_key="joined")
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"_aspen_seq": 1, "shared": {}, "plugins": {"a": {"x": 1}, "b": {"y": 2}}})
    assert out == {"joined": {"x": 1, "y": 2}}


def _case_load_detections(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.load_plugins import LoadDetectionsPlugin

    csv_path = tmp_path / "detections.csv"
    _write_detection_csv(csv_path)
    track_data_sample = make_track_data_sample(num_frames=1)
    plugin = LoadDetectionsPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "detection_data_path": str(csv_path),
            "frame_id": 1,
            "data_samples": track_data_sample,
        }
    )
    assert out["detection_dataframe"] is not None
    assert track_data_sample[0].pred_instances.bboxes.shape == (1, 4)


def _case_load_tracking(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.load_plugins import LoadTrackingPlugin

    csv_path = tmp_path / "tracking.csv"
    _write_tracking_csv(csv_path)
    track_data_sample = make_track_data_sample(num_frames=1)
    plugin = LoadTrackingPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "tracking_data_path": str(csv_path),
            "frame_id": 1,
            "data_samples": track_data_sample,
        }
    )
    assert out["tracking_dataframe"] is not None
    assert int(out["nr_tracks"]) == 1
    assert int(out["max_tracking_id"]) == 7
    assert int(track_data_sample[0].pred_track_instances.instances_id[0].item()) == 7


def _case_load_pose(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.load_plugins import LoadPosePlugin

    csv_path = tmp_path / "pose.csv"
    _write_pose_csv(csv_path)
    plugin = LoadPosePlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "pose_data_path": str(csv_path),
            "frame_id": 1,
            "data_samples": make_track_data_sample(num_frames=1),
        }
    )
    assert out["pose_dataframe"] is not None
    assert len(out["pose_results"]) == 1
    assert "predictions" in out["pose_results"][0]


def _case_load_camera(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.load_plugins import LoadCameraPlugin

    csv_path = tmp_path / "camera.csv"
    df = CameraTrackingDataFrame(output_file=str(csv_path), input_batch_size=1)
    df.add_frame_records(
        frame_id=1,
        tlbr=torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32).numpy(),
    )
    df.close()

    plugin = LoadCameraPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"camera_data_path": str(csv_path), "frame_id": 1})
    assert out["camera_dataframe"] is not None
    assert out["camera_frame_id"] == 1
    assert out["camera_bboxes"].shape == (1, 4)


def _case_mmtracking(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.mmtracking_plugin import MMTrackingPlugin

    track_data_sample = make_track_data_sample(num_frames=1)
    plugin = MMTrackingPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"model": HarnessTrackingDelegate(), "data_samples": track_data_sample})
    assert out["data_samples"] is track_data_sample
    assert out["nr_tracks"] == 2
    assert out["max_tracking_id"] == 9


def _case_model_config(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.model_config_plugin import ModelConfigPlugin

    model = SimpleNamespace(_enabled=True, post_tracking_pipeline=None, some_flag=False)
    plugin = ModelConfigPlugin(
        post_tracking_pipeline=[{"type": "X"}], enabled=False, some_flag=True
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"model": model})
    assert out == {}
    assert model.post_tracking_pipeline == [{"type": "X"}]
    assert model._enabled is False
    assert model.some_flag is True


def _case_model_factory(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.model_factory_plugin import ModelFactoryPlugin

    plugin = ModelFactoryPlugin(
        model_class=f"{__name__}.HarnessModel",
        detector={"type": "D", "score": 1},
        tracker={"type": "T"},
        post_tracking_pipeline=[{"type": "Pipe"}],
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"device": torch.device("cpu")})
    model = out["model"]
    assert isinstance(model, HarnessModel)
    assert model.kwargs["detector"]["type"] == "D"
    assert model.eval_called is True
    assert model.to_device == torch.device("cpu")


def _case_overlay(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.overlay_plugin import OverlayPlugin

    plugin = OverlayPlugin(print_available=False)
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"original_images": torch.zeros((1, 4, 4, 3), dtype=torch.float32)})
    assert out == {}


def _case_play_tracker(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.play_tracker_plugin import PlayTrackerPlugin

    expected_box = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    plugin = PlayTrackerPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin._play_tracker = FakePlayTrackerRuntime(
        outputs={
            "img": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "current_box": expected_box,
            "frame_ids": torch.tensor([1], dtype=torch.long),
        }
    )
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1),
            "original_images": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "shared": {"game_config": {}, "game_id": "game-1"},
        }
    )
    _assert_tensor_close(out["current_box"], expected_box)
    assert "play_box" in out


def _case_pose_factory(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.pose_factory_plugin import PoseInferencerFactoryPlugin

    class _Inferencer:
        def __init__(self, pose2d, pose2d_weights, device, det_model, show_progress):
            self.pose2d = pose2d
            self.pose2d_weights = pose2d_weights
            self.device = device
            self.det_model = det_model
            self.show_progress = show_progress
            self.filter_args = {}
            self.inferencer = SimpleNamespace(
                model=torch.nn.Linear(1, 1), cfg=SimpleNamespace(data_mode="bottomup")
            )

    install_fake_module(monkeypatch, "mmpose")
    install_fake_module(monkeypatch, "mmpose.apis")
    install_fake_module(monkeypatch, "mmpose.apis.inferencers", MMPoseInferencer=_Inferencer)

    plugin = PoseInferencerFactoryPlugin(
        pose_config="pose.py",
        pose_checkpoint="weights.pth",
        device="cpu",
    )
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"device": torch.device("cpu")})
    assert out["pose_inferencer"].pose2d.endswith("pose.py")
    assert out["pose_inferencer"].device == "cpu"


def _case_pose_plugin(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.pose_plugin import PosePlugin

    results = make_pose_results(num_frames=2)
    plugin = PosePlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "pose_inferencer": FakePoseInferencer(results),
            "original_images": torch.zeros((2, 3, 16, 16), dtype=torch.float32),
            "data_samples": make_track_data_sample(num_frames=2),
        }
    )
    assert len(out["pose_results"]) == 2
    assert "predictions" in out["pose_results"][0]


def _case_pose_to_det(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.pose_to_det_plugin import PoseToDetPlugin

    track_data_sample = make_track_data_sample(num_frames=1)
    pose_results = [
        {
            "predictions": [
                {
                    "bboxes": [[1.0, 2.0, 10.0, 12.0]],
                    "bbox_scores": [0.5],
                    "keypoints": [[[1.0, 2.0], [3.0, 4.0]]],
                    "keypoint_scores": [[0.8, 0.9]],
                }
            ]
        }
    ]
    plugin = PoseToDetPlugin(score_adder=0.25)
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"data_samples": track_data_sample, "pose_results": pose_results})
    assert out == {}
    assert torch.allclose(
        track_data_sample[0].pred_instances.scores,
        torch.tensor([0.75], dtype=torch.float32),
    )


def _case_postprocess(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.postprocess_plugin import CamPostProcessPlugin

    plugin = CamPostProcessPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.final_frame_width = 1280
    plugin.final_frame_height = 720
    monkeypatch.setattr(plugin, "_ensure_initialized", lambda context: None)
    monkeypatch.setattr(
        plugin,
        "process_tracking",
        lambda results, context: {"arena": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )
    out = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1),
            "original_images": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "fps": 30.0,
        }
    )
    assert out["final_frame_size"] == (1280, 720)
    _assert_tensor_close(out["arena"], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def _case_prune(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.prune_plugin import PruneKeysPlugin

    plugin = PruneKeysPlugin(remove_keys=["a", "b"])
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin({"a": 1, "b": 2, "c": 3})
    assert isinstance(out["a"], DeleteKey)
    assert isinstance(out["b"], DeleteKey)


def _case_save_detections(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.save_plugins import SaveDetectionsPlugin

    work_dir = tmp_path / "detections"
    inst = make_instance_data(
        bboxes=torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32),
        scores=torch.tensor([0.95], dtype=torch.float32),
        labels=torch.tensor([1], dtype=torch.long),
    )
    plugin = SaveDetectionsPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.forward(
        {
            "work_dir": str(work_dir),
            "frame_id": 1,
            "data_samples": make_track_data_sample(num_frames=1, pred_instances=[inst]),
        }
    )
    plugin.finalize()
    loaded = DetectionDataFrame(input_file=str(work_dir / "detections.csv"), input_batch_size=1)
    sample = loaded.get_sample_by_frame(0)
    assert sample.pred_instances.bboxes.shape == (1, 4)


def _case_save_tracking(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.save_plugins import SaveTrackingPlugin

    work_dir = tmp_path / "tracking"
    inst = make_instance_data(
        bboxes=torch.tensor([[5.0, 6.0, 25.0, 26.0]], dtype=torch.float32),
        scores=torch.tensor([0.9], dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        instances_id=torch.tensor([7], dtype=torch.long),
    )
    plugin = SaveTrackingPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.forward(
        {
            "work_dir": str(work_dir),
            "frame_id": 1,
            "data_samples": make_track_data_sample(num_frames=1, pred_track_instances=[inst]),
            "jersey_results": [[TrackJerseyInfo(tracking_id=7, number=12, score=0.9)]],
        }
    )
    plugin.finalize()
    loaded = TrackingDataFrame(input_file=str(work_dir / "tracking.csv"), input_batch_size=1)
    sample = loaded.get_sample_by_frame(1)
    assert int(sample[0].pred_track_instances.instances_id[0].item()) == 7


def _case_save_pose(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.save_plugins import SavePosePlugin

    work_dir = tmp_path / "pose"
    plugin = SavePosePlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.forward(
        {
            "work_dir": str(work_dir),
            "frame_id": 1,
            "data_samples": make_track_data_sample(num_frames=1),
            "pose_results": make_pose_results(num_frames=1),
        }
    )
    plugin.finalize()
    loaded = PoseDataFrame(input_file=str(work_dir / "pose.csv"), write_interval=100)
    assert loaded.get_sample_by_frame(1) is not None


def _case_save_actions(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.save_plugins import SaveActionsPlugin

    work_dir = tmp_path / "actions"
    inst = make_instance_data(
        bboxes=torch.tensor([[5.0, 6.0, 25.0, 26.0]], dtype=torch.float32),
        scores=torch.tensor([0.9], dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        instances_id=torch.tensor([7], dtype=torch.long),
    )
    plugin = SaveActionsPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.forward(
        {
            "work_dir": str(work_dir),
            "frame_id": 1,
            "data_samples": make_track_data_sample(num_frames=1, pred_track_instances=[inst]),
            "action_results": [
                [{"tracking_id": 7, "label": "skate", "label_index": 1, "score": 0.8}]
            ],
        }
    )
    plugin.finalize()
    loaded = ActionDataFrame(input_file=str(work_dir / "actions.csv"), input_batch_size=1)
    actions = loaded.get_data_dict_by_frame(1)["actions"]
    assert actions[0]["label"] == "skate"


def _case_save_camera(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.save_plugins import SaveCameraPlugin

    work_dir = tmp_path / "camera"
    plugin = SaveCameraPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin.forward(
        {
            "work_dir": str(work_dir),
            "frame_id": 1,
            "current_box": torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32),
            "current_fast_box_list": torch.tensor([[2.0, 3.0, 9.0, 11.0]], dtype=torch.float32),
        }
    )
    plugin.finalize()
    loaded = CameraTrackingDataFrame(input_file=str(work_dir / "camera.csv"), input_batch_size=1)
    assert loaded.get_data_dict_by_frame(1)["bboxes"].shape == (1, 4)


def _case_stitching(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    plugin = StitchingPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    with pytest.raises(RuntimeError, match="at least 2 input views"):
        plugin({"stitch_inputs": [{"img": torch.zeros((1, 4, 4, 3), dtype=torch.float32)}]})


@requires_torch
def should_cache_stitch_rotation_grid_without_changing_pixels() -> None:
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    plugin = StitchingPlugin()
    image = torch.arange(1 * 8 * 10 * 3, dtype=torch.uint8).reshape(1, 8, 10, 3)
    out_first = plugin._rotate_tensor_keep_size(image, 9.0)

    assert len(plugin._rotate_grid_cache) == 1
    grid = next(iter(plugin._rotate_grid_cache.values()))
    grid_ptr = grid.data_ptr()

    out_cached = plugin._rotate_tensor_keep_size(image.clone(), 9.0)

    assert torch.equal(out_first, out_cached)
    assert len(plugin._rotate_grid_cache) == 1
    assert next(iter(plugin._rotate_grid_cache.values())).data_ptr() == grid_ptr


@requires_torch
def should_allow_disabling_stitch_rotation_grid_cache() -> None:
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    plugin = StitchingPlugin(cache_rotation_grid=False)
    image = torch.arange(1 * 8 * 10 * 3, dtype=torch.uint8).reshape(1, 8, 10, 3)

    out_first = plugin._rotate_tensor_keep_size(image, 9.0)
    out_second = plugin._rotate_tensor_keep_size(image.clone(), 9.0)

    assert torch.equal(out_first, out_second)
    assert plugin._rotate_grid_cache == {}


def _case_tracker(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.tracker_plugin import TrackerPlugin

    det_inst = make_instance_data(
        bboxes=torch.tensor([[1.0, 2.0, 10.0, 12.0]], dtype=torch.float32),
        scores=torch.tensor([0.9], dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
    )
    track_data_sample = make_track_data_sample(num_frames=1, pred_instances=[det_inst])
    tracker_result = {
        "ids": torch.tensor([7], dtype=torch.long),
        "bboxes": torch.tensor([[2.0, 3.0, 11.0, 13.0]], dtype=torch.float32),
        "scores": torch.tensor([0.85], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.long),
        "num_tracks": torch.tensor([1], dtype=torch.long),
        "max_id": torch.tensor([7], dtype=torch.long),
    }
    plugin = TrackerPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    plugin._hm_tracker = HarnessTrackerRuntime(tracker_result)
    plugin._static_tracker_max_detections = None
    out = plugin(
        {
            "frame_id": 1,
            "original_images": torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            "data_samples": track_data_sample,
        }
    )
    assert int(out["nr_tracks"].item()) == 1
    assert int(out["max_tracking_id"].item()) == 7
    assert int(track_data_sample[0].pred_track_instances.instances_id[0].item()) == 7


def _case_video_out_prep(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.video_out_prep_plugin import VideoOutPrepPlugin

    FakeVideoOutputPreparer.instances.clear()
    monkeypatch.setattr(
        "hmlib.aspen.plugins.video_out_prep_plugin.VideoOutputPreparer",
        FakeVideoOutputPreparer,
    )
    plugin = VideoOutPrepPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "img": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "video_frame_cfg": {
                "output_frame_width": 8,
                "output_frame_height": 8,
                "output_aspect_ratio": 1.0,
            },
            "work_dir": str(tmp_path),
            "shared": {"device": torch.device("cpu"), "game_config": {}},
        }
    )
    assert out["video_out_prepared"] is True
    assert len(FakeVideoOutputPreparer.instances) == 1
    assert len(FakeVideoOutputPreparer.instances[0].prepare_calls) == 1
    assert FakeVideoOutputPreparer.instances[0].cuda_graph_enabled is cuda_graph_enabled


def _case_video_out(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.video_out_plugin import VideoOutPlugin

    FakeVideoOutput.instances.clear()
    monkeypatch.setattr("hmlib.aspen.plugins.video_out_plugin.VideoOutput", FakeVideoOutput)
    plugin = VideoOutPlugin(output_video_path="tracking.mkv")
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "img": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "work_dir": str(tmp_path),
            "shared": {"device": torch.device("cpu"), "game_config": {}},
        }
    )
    assert out == {}
    assert len(FakeVideoOutput.instances) == 1
    assert len(FakeVideoOutput.instances[0].prepare_calls) == 1
    assert len(FakeVideoOutput.instances[0].calls) == 1
    assert FakeVideoOutput.instances[0].cuda_graph_enabled is cuda_graph_enabled


def _case_video_preview(monkeypatch, _tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.video_preview_plugin import VideoPreviewPlugin

    FakeShower.instances.clear()
    monkeypatch.setattr("hmlib.aspen.plugins.video_preview_plugin.Shower", FakeShower)
    progress_bar = FakeProgressBar()
    plugin = VideoPreviewPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "img": torch.zeros((1, 8, 8, 3), dtype=torch.uint8),
            "fps": 30.0,
            "shared": {
                "game_id": "game-1",
                "game_config": {"video_out": {"show_image": True}},
                "progress_bar": progress_bar,
            },
        }
    )
    assert out == {}
    assert len(FakeShower.instances) == 1
    assert FakeShower.instances[0].kwargs["skip_frame_when_full"] is True
    assert FakeShower.instances[0].kwargs["drop_oldest_when_full"] is True
    assert len(FakeShower.instances[0].calls) == 1
    assert FakeShower.instances[0].calls[0][1] is True
    assert len(progress_bar.callbacks) == 1
    plugin.finalize()
    assert FakeShower.instances[0].closed is True

    FakeShower.instances.clear()
    plugin = VideoPreviewPlugin()
    published = []
    publisher = SimpleNamespace(publish_preview=lambda img, name: published.append((img, name)))
    out = plugin(
        {
            "img": torch.zeros((1, 8, 8, 3), dtype=torch.uint8),
            "fps": 30.0,
            "shared": {
                "game_config": {"video_out": {"show_image": True}},
                "hm_ui_preview_active": True,
                "hm_ui_process": publisher,
            },
        }
    )
    assert out == {}
    assert FakeShower.instances == []
    assert len(published) == 1
    assert published[0][1] == "Final"


def _case_stitch_ui(monkeypatch, _tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.aspen.plugins.stitch_ui_plugin import StitchUiPlugin

    plugin = StitchUiPlugin()
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    img = torch.zeros((1, 8, 8, 3), dtype=torch.uint8)
    out = plugin(
        {
            "img": img,
            "shared": {
                "camera_ui": 0,
                "game_config": {},
            },
        }
    )
    _assert_tensor_close(out["img"], img)
    plugin.finalize()


def _case_apply_camera(monkeypatch, tmp_path: Path, cuda_graph_enabled: bool = False) -> None:
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin

    monkeypatch.setattr("hmlib.camera.apply_camera_plugin.Compose", FakeCompose)
    monkeypatch.setattr("hmlib.camera.apply_camera_plugin.EndZones", FakeEndZones)
    plugin = ApplyCameraPlugin(video_out_pipeline=None, crop_output_image=False)
    _maybe_enable_cuda_graph(plugin, cuda_graph_enabled)
    out = plugin(
        {
            "img": torch.zeros((1, 8, 8, 3), dtype=torch.float32),
            "current_box": torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float32),
            "shared": {"game_config": {}},
        }
    )
    assert out["img"].shape == (1, 8, 8, 3)
    assert out["video_frame_cfg"]["output_frame_width"] == 8
    assert out["video_frame_cfg"]["output_frame_height"] == 8


PLUGIN_CASES: dict[str, Callable[[Any, Path, bool], None]] = {
    "hmlib.aspen.plugins.action_factory_plugin.ActionRecognizerFactoryPlugin": _case_action_factory,
    "hmlib.aspen.plugins.action_pose_plugin.ActionFromPosePlugin": _case_action_pose,
    "hmlib.aspen.plugins.boundaries_plugin.BoundariesPlugin": _case_boundaries,
    "hmlib.aspen.plugins.camera_controller_plugin.CameraControllerPlugin": _case_camera_controller,
    "hmlib.aspen.plugins.camera_train_plugin.CameraTrainPlugin": _case_camera_train,
    "hmlib.aspen.plugins.dataloader_plugin.DataLoaderPlugin": _case_dataloader,
    "hmlib.aspen.plugins.debug_rgb_stats_plugin.RgbStatsCheckPlugin": _case_debug_rgb_stats,
    "hmlib.aspen.plugins.detector_compare_plugin.DetectorComparePlugin": _case_detector_compare,
    "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin": _case_detector_factory,
    "hmlib.aspen.plugins.detector_plugin.DetectorInferencePlugin": _case_detector_inference,
    "hmlib.aspen.plugins.ice_rink_boundaries_plugins.IceRinkSegmBoundariesPlugin": _case_ice_boundaries,
    "hmlib.aspen.plugins.ice_rink_boundaries_plugins.IceRinkSegmConfigPlugin": _case_ice_config,
    "hmlib.aspen.plugins.image_prep_plugin.ImagePrepPlugin": _case_image_prep,
    "hmlib.aspen.plugins.jersey_koshkina_plugin.KoshkinaJerseyNumberPlugin": _case_jersey_koshkina,
    "hmlib.aspen.plugins.jersey_pose_plugin.JerseyNumberFromPosePlugin": _case_jersey_pose,
    "hmlib.aspen.plugins.join_plugin.JoinPlugin": _case_join,
    "hmlib.aspen.plugins.load_plugins.LoadDetectionsPlugin": _case_load_detections,
    "hmlib.aspen.plugins.load_plugins.LoadTrackingPlugin": _case_load_tracking,
    "hmlib.aspen.plugins.load_plugins.LoadPosePlugin": _case_load_pose,
    "hmlib.aspen.plugins.load_plugins.LoadCameraPlugin": _case_load_camera,
    "hmlib.aspen.plugins.mmtracking_plugin.MMTrackingPlugin": _case_mmtracking,
    "hmlib.aspen.plugins.model_config_plugin.ModelConfigPlugin": _case_model_config,
    "hmlib.aspen.plugins.model_factory_plugin.ModelFactoryPlugin": _case_model_factory,
    "hmlib.aspen.plugins.overlay_plugin.OverlayPlugin": _case_overlay,
    "hmlib.aspen.plugins.play_tracker_plugin.PlayTrackerPlugin": _case_play_tracker,
    "hmlib.aspen.plugins.pose_factory_plugin.PoseInferencerFactoryPlugin": _case_pose_factory,
    "hmlib.aspen.plugins.pose_plugin.PosePlugin": _case_pose_plugin,
    "hmlib.aspen.plugins.pose_to_det_plugin.PoseToDetPlugin": _case_pose_to_det,
    "hmlib.aspen.plugins.postprocess_plugin.CamPostProcessPlugin": _case_postprocess,
    "hmlib.aspen.plugins.prune_plugin.PruneKeysPlugin": _case_prune,
    "hmlib.aspen.plugins.save_plugins.SaveDetectionsPlugin": _case_save_detections,
    "hmlib.aspen.plugins.save_plugins.SaveTrackingPlugin": _case_save_tracking,
    "hmlib.aspen.plugins.save_plugins.SavePosePlugin": _case_save_pose,
    "hmlib.aspen.plugins.save_plugins.SaveActionsPlugin": _case_save_actions,
    "hmlib.aspen.plugins.save_plugins.SaveCameraPlugin": _case_save_camera,
    "hmlib.aspen.plugins.stitching_plugin.StitchingPlugin": _case_stitching,
    "hmlib.aspen.plugins.stitch_ui_plugin.StitchUiPlugin": _case_stitch_ui,
    "hmlib.aspen.plugins.tracker_plugin.TrackerPlugin": _case_tracker,
    "hmlib.aspen.plugins.video_out_prep_plugin.VideoOutPrepPlugin": _case_video_out_prep,
    "hmlib.aspen.plugins.video_preview_plugin.VideoPreviewPlugin": _case_video_preview,
    "hmlib.aspen.plugins.video_out_plugin.VideoOutPlugin": _case_video_out,
    "hmlib.camera.apply_camera_plugin.ApplyCameraPlugin": _case_apply_camera,
}


def should_report_torch_runtime_requirement_for_plugin_isolation() -> None:
    if torch is None:
        pytest.skip("torch is unavailable in this Bazel test runtime")


@requires_torch
@pytest.mark.parametrize("cuda_graph_enabled", [False, True], ids=["normal", "cuda_graph"])
@pytest.mark.parametrize("plugin_class", sorted(PLUGIN_CASES))
def should_validate_each_aspen_plugin_in_isolation(
    plugin_class: str, cuda_graph_enabled: bool, monkeypatch, tmp_path: Path
) -> None:
    PLUGIN_CASES[plugin_class](monkeypatch, tmp_path, cuda_graph_enabled)


@requires_torch
def should_cover_every_production_aspen_plugin_class() -> None:
    assert set(PLUGIN_CASES) == discover_production_plugin_classes(REPO_ROOT)
