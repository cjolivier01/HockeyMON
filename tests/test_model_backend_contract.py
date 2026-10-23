"""An explicit inference backend must stay selected or report a failure."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from hmlib.aspen.plugins import detector_factory_plugin as detector
from hmlib.aspen.plugins import pose_factory_plugin as pose


@pytest.mark.parametrize("backend", ["ONNX", "TensorRT"])
def should_surface_detector_initialization_failure(monkeypatch, backend):
    def fail(**_kwargs):
        raise ValueError("unsupported checkpoint")

    monkeypatch.setattr(detector, "_runtime_supports_tensorrt", lambda: True)
    monkeypatch.setattr(
        detector, "_OnnxDetectorWrapper" if backend == "ONNX" else "_TrtDetectorWrapper", fail
    )
    plugin = detector.DetectorFactoryPlugin(
        onnx={"enable": backend == "ONNX"}, trt={"enable": backend == "TensorRT"}
    )
    plugin._model = torch.nn.Identity()
    with pytest.raises(RuntimeError, match=f"requested {backend} detector backend") as raised:
        plugin.forward({})
    assert str(raised.value.__cause__) == "unsupported checkpoint"
    assert plugin._runtime_detector is None


@pytest.mark.parametrize("kind", ["detector", "pose"])
def should_reuse_supplied_int8_model_without_export_or_quantization(tmp_path, monkeypatch, kind):
    path = tmp_path / "prepared.int8.onnx"
    path.write_bytes(b"session fixture")
    opened = []
    if kind == "detector":
        runner_type = detector._OnnxDetectorWrapper
        monkeypatch.setattr(
            runner_type, "_create_session", lambda self, value: opened.append(value)
        )
    else:
        runner_type = pose._OnnxPoseRunner
        monkeypatch.setattr(
            runner_type,
            "_export_and_create_session",
            lambda self: opened.append(self.onnx_path),
        )

    def unexpected_export(*_args, **_kwargs):
        raise AssertionError("prepared INT8 model must not be exported")

    monkeypatch.setattr(torch.onnx, "export", unexpected_export)
    runner = runner_type(torch.nn.Identity(), str(path), quantize_int8=True)
    runner._maybe_quantize_after_calib()
    assert opened == [str(path)]
    assert runner._quantized and runner._quantized_path == str(path)
    with pytest.raises(ValueError, match="existing INT8"):
        runner_type(torch.nn.Identity(), str(path), force_export=True)


@pytest.mark.parametrize("runner_type", [detector._OnnxDetectorWrapper, pose._OnnxPoseRunner])
def should_surface_int8_quantization_failure(monkeypatch, runner_type):
    def fail(**_kwargs):
        raise ValueError("unsupported quantization node")

    quantization = ModuleType("onnxruntime.quantization")
    quantization.CalibrationDataReader = object
    quantization.QuantFormat = SimpleNamespace(QDQ="QDQ")
    quantization.QuantType = SimpleNamespace(QUInt8="uint8", QInt8="int8")
    quantization.quantize_static = fail
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization)
    runner = object.__new__(runner_type)
    runner.quantize_int8 = True
    runner._quantized = False
    runner.calib_target = 1
    runner._calib_inputs = [object()]
    runner.onnx_path = "prepared.onnx"
    runner._ort_sess = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="images")])
    with pytest.raises(RuntimeError, match="INT8 quantization failed") as raised:
        runner._maybe_quantize_after_calib()
    assert str(raised.value.__cause__) == "unsupported quantization node"
    assert not runner._quantized


def should_not_run_pytorch_when_wrapped_pose_execution_fails():
    class Model:
        def test_step(self, _data):
            raise AssertionError("PyTorch must not run")

    class Runner:
        def forward(self, _data):
            raise RuntimeError("requested backend failed")

    model = Model()
    pose.PoseInferencerFactoryPlugin._wrap_model_test_step(model, Runner())
    with pytest.raises(RuntimeError, match="requested backend failed"):
        model.test_step({})


@pytest.mark.parametrize("backend", ["onnx", "trt"])
def should_preserve_mmpose_preprocessing_flip_and_image_coordinates(monkeypatch, tmp_path, backend):
    from mmengine.structures import InstanceData
    from mmpose.models.data_preprocessors import PoseDataPreprocessor
    from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
    from mmpose.structures import PoseDataSample

    observed = []

    class Head(torch.nn.Module):
        def predict(self, features, samples, test_cfg):
            assert test_cfg["flip_test"] is True
            assert len(features) == 2
            assert torch.equal(features[1], features[0].flip(-1))
            observed.append(features[0])
            return [
                InstanceData(
                    keypoints=torch.tensor([[[2.0, 2.0]]]),
                    keypoint_scores=torch.tensor([[0.9]]),
                )
                for _ in samples
            ]

    class Model(torch.nn.Module):
        add_pred_to_datasample = TopdownPoseEstimator.add_pred_to_datasample

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Identity()
            self.head = Head()
            self.data_preprocessor = PoseDataPreprocessor(mean=[1, 2, 3], std=[2, 2, 2])
            self.test_cfg = {"flip_test": True}

    model = Model()
    if backend == "onnx":
        monkeypatch.setattr(pose._OnnxPoseRunner, "_export_and_create_session", lambda self: None)
        runner = pose._OnnxPoseRunner(model, str(tmp_path / "model.onnx"))
        monkeypatch.setattr(
            runner,
            "_run_feats",
            lambda inputs, *, preprocessed, collect_calibration=True: inputs,
        )
    else:

        class ReusedOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output = None

            def forward(self, inputs):
                if self.output is None:
                    self.output = torch.empty_like(inputs)
                self.output.copy_(inputs)
                return self.output

        monkeypatch.setattr(
            pose, "compile_torch_tensorrt", lambda *_args, **_kwargs: ReusedOutput()
        )
        runner = pose._TrtPoseRunner(model, str(tmp_path / "model.engine"), batch_size=2)

    sample = PoseDataSample(
        metainfo={
            "input_center": [100.0, 200.0],
            "input_scale": [40.0, 80.0],
            "input_size": [4, 4],
        }
    )
    sample.gt_instances = InstanceData(
        bboxes=torch.tensor([[80.0, 160.0, 120.0, 240.0]]), bbox_scores=torch.tensor([1.0])
    )
    inputs = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
    predictions = runner.forward({"inputs": [inputs], "data_samples": [sample]})
    normalized = (inputs - torch.tensor([1, 2, 3]).reshape(3, 1, 1)) / 2
    assert torch.equal(observed[0].cpu(), normalized.unsqueeze(0))
    assert len(predictions) == 1 and isinstance(predictions[0], PoseDataSample)
    assert predictions[0].pred_instances.keypoints.tolist() == [[[100.0, 200.0]]]
    assert torch.equal(predictions[0].pred_instances.bboxes, sample.gt_instances.bboxes)
