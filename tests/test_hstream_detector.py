"""Exercise shared preparation publication with real images, video and ONNX."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hmlib.ui import hstream_detector as detector


def should_build_detector_only_preparation_without_breaking_structural_joins(tmp_path, monkeypatch):
    import torch

    from hmlib.aspen.net import AspenNet

    game = tmp_path / "game"
    game.mkdir()
    (game / "config.yaml").write_text("{}\n")
    config, _, _ = detector._configuration(game, "hockeyref")
    instantiate = AspenNet._instantiate

    def without_models(class_path, params):
        if class_path == "hmlib.aspen.plugins.join_plugin.JoinPlugin":
            return instantiate(class_path, params)
        return torch.nn.Identity()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(AspenNet, "_instantiate", staticmethod(without_models))
    graph = AspenNet("detector-preparation", config["aspen"])
    order = [node.name for node in graph.exec_order]
    assert order.index("detector_factory") < order.index("detector")
    assert graph.node_map["apply_camera_join"].module.allow_multi_path_inputs


def quantized_model(path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["images", "scale", "zero"], ["quantized"]),
            helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["features"]),
        ],
        "shared-calibration-test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 2, 2])],
        initializer=[
            numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), "zero"),
        ],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9), path
    )


def samples(directory, count=16, shape=(32, 16)):
    directory.mkdir()
    records = []
    for index in range(count):
        name = f"frame-{index:04d}.png"
        Image.new("RGB", shape, (index * 8, 40, 80)).save(directory / name)
        records.append({"image": name, "timeline_ns": str(index * 1_000_000_000)})
    (directory / "samples.json").write_text(
        json.dumps(
            {"schema": "1", "width": str(shape[0]), "height": str(shape[1]), "samples": records}
        )
    )


def should_prepare_from_native_recording_samples_and_publish_validated_python_artifact(
    tmp_path, monkeypatch, capsys
):
    game = tmp_path / "game"
    game.mkdir()
    (game / "config.yaml").write_text("{}\n")
    graph = tmp_path / "tracking.yaml"
    graph.write_text("aspen: {}\n")
    config = {"hm_ui": {"run": {"detector": "hockeyref", "precision": "onnx"}}}
    monkeypatch.setenv("HM_GAME_DIR", str(tmp_path))
    monkeypatch.setenv("HM_UI_SESSION_DIR", "/must/not/inherit")
    monkeypatch.setenv("HSTREAM_NATIVE_RUNNER", "/bin/true")
    monkeypatch.setattr(detector, "_native_config", lambda runner: graph)
    monkeypatch.setattr(detector, "_configuration", lambda game, profile: (config, graph, (32, 16)))
    run = detector._run
    calls = []

    def preparation_step(command, environment, *, cwd=None):
        calls.append(command)
        assert "HM_UI_SESSION_DIR" not in environment
        assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(
            Path(detector.__file__).resolve().parents[2]
        )
        if any(part.startswith("--int8-sample-output=") for part in command):
            directory = Path(
                next(
                    part.partition("=")[2]
                    for part in command
                    if part.startswith("--int8-sample-output=")
                )
            )
            samples(directory)
            print("HSTREAM_INT8_SAMPLE completed=16 total=16 timeline_ns=15000000000")
        elif "hmlib.cli.hmtrack" in command:
            assert command[command.index("--detector-model") + 1] == "hockeyref"
            assert "--detector-onnx-calib-frames=16" in command
            assert "--camera-ui=0" in command and "--no-save-video" in command
            reel = Path(command[command.index("--input-video") + 1])
            probe = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "csv=p=0",
                    str(reel),
                ],
                text=True,
            )
            assert int(probe.strip()) == 16
            quantized_model(
                Path(command[command.index("--detector-onnx") + 1]).with_name(
                    "detector-fixture.int8.onnx"
                )
            )
            print("\x1b[?25h", end="")
        else:
            run(command, environment, cwd=cwd)

    monkeypatch.setattr(detector, "_run", preparation_step)
    result = detector.prepare_detector("game", 16, "default")
    manifest = json.loads(Path(result["manifest"]).read_text())
    model = Path(result["engine"])
    assert model.is_file() and model.parent == Path(result["manifest"]).parent
    assert manifest["model"] == "default" and manifest["profile"] == "hockeyref"
    assert manifest["sha256"] == detector.model_preparation._digest(model)
    assert manifest["frames"] == 16 and len(manifest["image_sha256"]) == 16
    selection = {
        "hstream_ui": {
            "hockeymon": {"detector": {"model": "default", "precision": "int8", **result}}
        }
    }
    assert detector.validated_detector_arguments(game, selection) == [
        "--detector-onnx-enable",
        "--detector-onnx",
        str(model),
    ]
    assert any("--int8-sample-verify" in command for command in calls)
    assert "HSTREAM_INT8_READY " + json.dumps(result) in capsys.readouterr().out.splitlines()
    assert (game / "config.yaml").read_text() == "{}\n"
    before = model.read_bytes(), Path(result["manifest"]).read_bytes()

    def failed_step(command, environment, *, cwd=None):
        if "hmlib.cli.hmtrack" in command:
            return  # Exit zero without an INT8 artifact is still a failure.
        preparation_step(command, environment, cwd=cwd)

    monkeypatch.setattr(detector, "_run", failed_step)
    with pytest.raises(RuntimeError, match="exactly one INT8"):
        detector.prepare_detector("game", 16, "default")
    assert before == (model.read_bytes(), Path(result["manifest"]).read_bytes())
    assert "HSTREAM_INT8_READY" not in capsys.readouterr().out
    model.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="changed on disk"):
        detector.validated_detector_arguments(game, selection)


def should_ignore_native_and_non_int8_models_in_shared_validation(tmp_path):
    assert (
        detector.validated_detector_arguments(
            tmp_path, {"pipeline": {"primary-gie": {"model-engine-file": "native.engine"}}}
        )
        == []
    )
    assert (
        detector.validated_detector_arguments(
            tmp_path,
            {"hstream_ui": {"hockeymon": {"detector": {"model": "default", "precision": "fp32"}}}},
        )
        == []
    )


@pytest.mark.parametrize("change", ["timestamp", "path", "symlink", "shape", "float", "boolean"])
def should_reject_invalid_native_sample_capture(tmp_path, change):
    directory = tmp_path / "samples"
    samples(directory)
    manifest = json.loads((directory / "samples.json").read_text())
    if change == "timestamp":
        manifest["samples"][1]["timeline_ns"] = 0
    elif change == "float":
        manifest["width"] = 32.5
    elif change == "boolean":
        manifest["schema"] = True
    elif change == "path":
        manifest["samples"][0]["image"] = "../frame-0000.png"
    elif change == "symlink":
        (directory / "frame-0000.png").unlink()
        (directory / "frame-0000.png").symlink_to(directory / "frame-0001.png")
    else:
        Image.new("RGB", (64, 16)).save(directory / "frame-0000.png")
    (directory / "samples.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        detector._samples(directory, 16, (32, 16))


@pytest.mark.parametrize("count", [0, 15, 257, True, 16.5])
def should_validate_budget_before_starting_preparation(count):
    with pytest.raises(ValueError, match="samples"):
        detector.prepare_detector("game", count, "default")


def should_cancel_the_preparation_process_group(tmp_path):
    ready = tmp_path / "ready"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib,time,os\npathlib.Path("
        + repr(str(ready))
        + ").write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys\nfrom hmlib.ui.hstream_detector import _cancellation,_run\nwith _cancellation():\n _run([sys.executable, sys.argv[1]], dict(os.environ))\n",
            str(child),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        child_pid = int(ready.read_text())
        parent.send_signal(signal.SIGTERM)
        _, error = parent.communicate(timeout=10)
        assert parent.returncode != 0 and "Detector preparation cancelled" in error
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate()
