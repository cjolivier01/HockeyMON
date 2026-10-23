"""Preparation must publish real, validated artifacts and reject stale selections."""

from types import SimpleNamespace

import numpy as np
import pytest

from hmlib.models.detector_selection import configure_detector_selection, detector_identity


def should_invalidate_prepared_pose_when_custom_architecture_changes(tmp_path):
    from hmlib.ui.model_preparation import cache_directory

    model_config = tmp_path / "pose.py"
    model_config.write_text("model = {'width': 192}\n")
    config = {"hm_ui": {"run": {"pose_model": "custom", "pose_config": str(model_config)}}}
    original = cache_directory(tmp_path, "pose", config)
    model_config.write_text("model = {'width': 288}\n")
    assert cache_directory(tmp_path, "pose", config) != original


from hmlib.ui import model_preparation as preparation
from hmlib.ui.workspace.files import write_json


def quantized_model(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["images", "scale", "zero"], ["quantized"]),
            helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["features"]),
        ],
        "calibration-test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 2, 2])],
        initializer=[
            numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.save(model, str(path))


def should_validate_real_quantized_onnx_and_reuse_the_exact_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_effective", lambda game: {})
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--config-override" not in command
        assert command.count("--config") == 2
        path = command[command.index("--detector-onnx") + 1]
        quantized_model(preparation.Path(path).with_name("detector-modelidentity.int8.onnx"))

    monkeypatch.setattr(preparation.subprocess, "run", run)
    status = preparation.prepare(tmp_path, "detector", 3)
    assert status["state"] == "ready"
    assert status["frames"] == 3
    args = preparation.prepared_onnx_arguments(tmp_path, "detector")
    assert args == ["--detector-onnx-enable", "--detector-onnx", status["path"]]
    assert "--detector-onnx-quantize-int8" not in args
    assert len(calls) == 1
    preparation.Path(status["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="preparation is required"):
        preparation.prepared_onnx_arguments(tmp_path, "detector")


def should_fail_preparation_when_legacy_runner_did_not_generate_int8(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_effective", lambda game: {})
    monkeypatch.setattr(preparation.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="exactly one INT8"):
        preparation.prepare(tmp_path, "detector", 3)
    assert preparation.model_status(tmp_path, "detector")["state"] == "missing"


def should_reject_unquantized_artifacts(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "float.onnx"
    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])], "float", [value], [output]
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9), path
    )
    with pytest.raises(RuntimeError, match="no INT8"):
        preparation._validate(path, "detector")


def should_preserve_model_identity_and_reject_conflicting_detector_selection(tmp_path):
    model = {"type": "FixtureDetector"}
    identity = detector_identity({"detector": model})
    artifact = tmp_path / f"detector-{identity}.int8.onnx"
    write_json(tmp_path / "ready.json", {"file": artifact.name})
    args = SimpleNamespace(
        detector_model="deployed",
        detector=None,
        compare_detectors=None,
        input_detection_data=None,
        input_tracking_data=None,
        trained_checkpoint=None,
        distilled_checkpoint=None,
        game_id="fixture",
    )

    def config():
        return {
            "aspen": {
                "plugins": {
                    "detector_factory": {
                        "params": {"detector": model, "onnx": {"path": str(artifact)}}
                    },
                    "detector": {"params": {}},
                }
            }
        }

    current = config()
    configure_detector_selection(args, current)
    assert current["aspen"]["plugins"]["detector_factory"]["params"]["onnx"]["path"] == str(
        artifact
    )
    changed = config()
    changed["aspen"]["plugins"]["detector_factory"]["params"]["detector"] = {
        "type": "OtherDetector"
    }
    with pytest.raises(ValueError, match="does not match"):
        configure_detector_selection(args, changed)


@pytest.mark.parametrize("frames", [0, -1, 10001, True, 1.5])
def should_reject_invalid_calibration_budgets(tmp_path, frames):
    with pytest.raises(ValueError, match="frame count"):
        preparation.preparation_command(tmp_path, "detector", frames)
