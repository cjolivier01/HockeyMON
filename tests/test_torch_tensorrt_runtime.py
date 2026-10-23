import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from hmlib.utils import torch_tensorrt_runtime


class _PoseModelFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.test_cfg = {}

    @staticmethod
    def data_preprocessor(data, training=False):
        return data

    @staticmethod
    def add_pred_to_datasample(instances, fields, samples):
        return instances


class _PreallocationController:
    def __init__(self) -> None:
        self.enabled = False

    def set_pre_allocated_output(self, enabled: bool) -> None:
        self.enabled = enabled


class _FakeTorchTensorRT:
    def __init__(self) -> None:
        self.input_specs = []
        self.compile_calls = []
        self.preallocation = _PreallocationController()
        self.runtime = SimpleNamespace(
            enable_pre_allocated_outputs=lambda _module: self.preallocation
        )

    def Input(self, **kwargs):
        self.input_specs.append(kwargs)
        return kwargs

    def compile(self, module, **kwargs):
        self.compile_calls.append((module, kwargs))
        return torch.nn.Identity()


@pytest.fixture
def fake_cuda_compilation(monkeypatch):
    device = torch.device("cuda:0")
    monkeypatch.setattr(torch_tensorrt_runtime.sys, "version_info", (3, 13))
    monkeypatch.setattr(
        torch_tensorrt_runtime,
        "_validate_cuda_device",
        lambda _module, _inputs: device,
    )
    monkeypatch.setattr(
        torch_tensorrt_runtime,
        "_cache_fingerprint",
        lambda _module, _torch_tensorrt, _device: "test-runtime",
    )
    monkeypatch.setattr(torch_tensorrt_runtime.torch.cuda, "device", lambda _device: nullcontext())


def should_compile_static_fp16_with_performance_settings(
    monkeypatch, tmp_path, fake_cuda_compilation
) -> None:
    fake = _FakeTorchTensorRT()
    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        lambda name: fake,
    )
    model = torch.nn.Conv2d(3, 8, kernel_size=3).eval()
    sample = torch.randn(1, 3, 32, 48)

    compiled = torch_tensorrt_runtime.compile_torch_tensorrt(
        model,
        [sample],
        engine_path=tmp_path / "detector.engine",
    )

    assert isinstance(compiled, torch.nn.Identity)
    assert fake.input_specs == [
        {
            "shape": (1, 3, 32, 48),
            "dtype": torch.float32,
            "format": torch.contiguous_format,
        }
    ]
    _, options = fake.compile_calls[0]
    assert options["ir"] == "dynamo"
    assert options["device"] == torch.device("cuda:0")
    assert options["enabled_precisions"] == {torch.float16}
    assert options["use_explicit_typing"] is False
    assert options["require_full_compilation"] is True
    assert options["pass_through_build_failures"] is True
    assert options["min_block_size"] == 1
    assert options["use_fast_partitioner"] is False
    assert options["optimization_level"] == 5
    assert options["num_avg_timing_iters"] == 8
    assert options["tiling_optimization_level"] == "full"
    assert options["workspace_size"] == 4 << 30
    assert options["immutable_weights"] is False
    assert options["cache_built_engines"] is True
    assert options["reuse_cached_engines"] is True
    assert options["engine_cache_dir"] == str(
        tmp_path / "detector.engine.torch_tensorrt_cache" / "test-runtime"
    )
    assert fake.preallocation.enabled is True


def should_force_fp32_rebuild_without_reusing_cached_engine(
    monkeypatch, tmp_path, fake_cuda_compilation
) -> None:
    fake = _FakeTorchTensorRT()
    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        lambda name: fake,
    )

    torch_tensorrt_runtime.compile_torch_tensorrt(
        torch.nn.Identity(),
        [torch.randn(2, 4)],
        engine_path=tmp_path / "pose.engine",
        force_build=True,
        fp16=False,
    )

    _, options = fake.compile_calls[0]
    assert options["enabled_precisions"] == {torch.float32}
    assert options["use_explicit_typing"] is True
    assert options["reuse_cached_engines"] is False


def should_retry_once_without_a_stale_cached_engine(
    monkeypatch, tmp_path, fake_cuda_compilation
) -> None:
    fake = _FakeTorchTensorRT()
    original_compile = fake.compile

    def _flaky_compile(module, **kwargs):
        if not fake.compile_calls:
            fake.compile_calls.append((module, kwargs))
            raise RuntimeError("incompatible serialized engine")
        return original_compile(module, **kwargs)

    fake.compile = _flaky_compile
    monkeypatch.setattr(torch_tensorrt_runtime.importlib, "import_module", lambda _name: fake)
    cache_dir = tmp_path / "detector.engine.torch_tensorrt_cache" / "test-runtime"
    cache_dir.mkdir(parents=True)
    (cache_dir / "existing-entry").mkdir()

    compiled = torch_tensorrt_runtime.compile_torch_tensorrt(
        torch.nn.Identity(),
        [torch.randn(1, 4)],
        engine_path=tmp_path / "detector.engine",
    )

    assert isinstance(compiled, torch.nn.Identity)
    assert len(fake.compile_calls) == 2
    assert fake.compile_calls[0][1]["reuse_cached_engines"] is True
    assert fake.compile_calls[1][1]["reuse_cached_engines"] is False


def should_reject_legacy_int8_calibration_before_import(monkeypatch, tmp_path) -> None:
    imported = False

    def _unexpected_import(_name):
        nonlocal imported
        imported = True

    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        _unexpected_import,
    )

    with pytest.raises(
        torch_tensorrt_runtime.TorchTensorRTConfigurationError,
        match="Q/DQ-quantized model",
    ):
        torch_tensorrt_runtime.compile_torch_tensorrt(
            torch.nn.Identity(),
            [torch.randn(1, 4)],
            engine_path=tmp_path / "detector.engine",
            int8=True,
        )

    assert imported is False


def should_reject_cpu_compilation_before_import(monkeypatch, tmp_path) -> None:
    imported = False

    def _unexpected_import(_name):
        nonlocal imported
        imported = True

    monkeypatch.setattr(torch_tensorrt_runtime.importlib, "import_module", _unexpected_import)

    with pytest.raises(
        torch_tensorrt_runtime.TorchTensorRTConfigurationError,
        match="must be CUDA tensors",
    ):
        torch_tensorrt_runtime.compile_torch_tensorrt(
            torch.nn.Identity(),
            [torch.randn(1, 4)],
            engine_path=tmp_path / "detector.engine",
        )

    assert imported is False


def should_surface_actionable_optional_dependency_error(monkeypatch) -> None:
    def _missing(_name):
        raise ModuleNotFoundError("No module named 'torch_tensorrt'")

    monkeypatch.setattr(
        torch_tensorrt_runtime.importlib,
        "import_module",
        _missing,
    )
    monkeypatch.setattr(torch_tensorrt_runtime.sys, "version_info", (3, 13))

    with pytest.raises(
        torch_tensorrt_runtime.TorchTensorRTUnavailableError,
        match="release matching the installed PyTorch and CUDA versions",
    ):
        torch_tensorrt_runtime.load_torch_tensorrt()


def should_surface_python_314_incompatibility_before_import(monkeypatch) -> None:
    imported = False

    def _unexpected_import(_name):
        nonlocal imported
        imported = True

    monkeypatch.setattr(torch_tensorrt_runtime.sys, "version_info", (3, 14))
    monkeypatch.setattr(torch_tensorrt_runtime.importlib, "import_module", _unexpected_import)

    with pytest.raises(
        torch_tensorrt_runtime.TorchTensorRTUnavailableError,
        match="use a Python 3.13 environment",
    ):
        torch_tensorrt_runtime.load_torch_tensorrt()

    assert imported is False


def should_lookup_nms_plugins_with_tensor_rt_11_registry_api() -> None:
    from hmlib.utils.nms import _get_plugin_creator

    expected = object()

    class _Registry:
        def get_creator(self, name, version, namespace):
            assert (name, version, namespace) == ("EfficientNMS_TRT", "1", "")
            return expected

    assert _get_plugin_creator(_Registry(), "EfficientNMS_TRT") is expected


def should_enumerate_tensor_rt_11_plugins_without_probing_missing_creators() -> None:
    from hmlib.utils.nms import _get_plugin_creator

    creator = SimpleNamespace(
        name="EfficientNMS_TRT",
        plugin_version="1",
        plugin_namespace="",
    )

    class _Registry:
        all_creators: ClassVar = [creator]

        def get_creator(self, _name, _version, _namespace):
            raise AssertionError(
                "creator lookup should not be called when enumeration is available"
            )

    registry = _Registry()
    assert _get_plugin_creator(registry, "EfficientNMS_TRT") is creator
    assert _get_plugin_creator(registry, "BatchedNMSDynamic_TRT") is None


def should_lookup_nms_plugins_with_legacy_registry_api() -> None:
    from hmlib.utils.nms import _get_plugin_creator

    expected = object()

    class _Registry:
        def get_plugin_creator(self, name, version, namespace):
            assert (name, version, namespace) == ("BatchedNMSDynamic_TRT", "1", "")
            return expected

    assert _get_plugin_creator(_Registry(), "BatchedNMSDynamic_TRT") is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def should_preserve_nonzero_labels_during_tensor_rt_nms_fallback() -> None:
    pytest.importorskip("tensorrt")
    from hmlib.utils.nms import TrtBatchedNMS, TrtNmsConfig

    cfg = TrtNmsConfig(
        num_classes=3,
        max_num_boxes=32,
        top_k=32,
        keep_top_k=8,
        score_threshold=0.1,
        iou_threshold=0.5,
        max_per_img=8,
        plugin="batched",
    )
    nms = TrtBatchedNMS(cfg, stream=None)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 9.0, 9.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        device="cuda",
    )
    scores = torch.zeros((3, 3), device="cuda")
    scores[:, 2] = torch.tensor([0.9, 0.8, 0.7], device="cuda")

    num_det, _out_boxes, _out_scores, out_classes = nms._infer(boxes, scores)
    torch.cuda.synchronize()
    count = int(num_det[0, 0])

    assert count == 2
    assert out_classes.dtype == nms._output_dtypes["nmsed_classes"]
    assert out_classes[0, :count].to(torch.long).tolist() == [2, 2]


def should_default_tensor_rt_models_to_fp16() -> None:
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())

    defaults = parser.parse_args([])
    fp32 = parser.parse_args(["--no-detector-trt-fp16", "--no-pose-trt-fp16"])

    assert defaults.detector_trt_fp16 is True
    assert defaults.pose_trt_fp16 is True
    assert defaults.pose_trt_batch_size == 32
    assert fp32.detector_trt_fp16 is False
    assert fp32.pose_trt_fp16 is False


def should_reuse_one_static_pose_engine_across_player_counts(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import pose_factory_plugin

    compiled_modules = []

    def _compile(*_args, **_kwargs):
        compiled = torch.nn.Identity()
        compiled_modules.append(compiled)
        return compiled

    monkeypatch.setattr(pose_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Model(_PoseModelFixture):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)

    runner = pose_factory_plugin._TrtPoseRunner(_Model(), engine_path=str(tmp_path / "pose.engine"))
    batch_one = torch.randn(1, 3, 16, 16)
    batch_two = torch.randn(2, 3, 16, 16)

    runner._ensure_trt_engine(batch_one)
    first_module = runner._trt_module
    runner._ensure_trt_engine(batch_two)
    runner._ensure_trt_engine(batch_one)

    assert len(compiled_modules) == 1
    assert runner._trt_module is first_module


def should_chunk_and_trim_static_pose_batches(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import pose_factory_plugin

    monkeypatch.setattr(
        pose_factory_plugin,
        "compile_torch_tensorrt",
        lambda *_args, **_kwargs: torch.nn.Identity(),
    )

    class _Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def predict(self, feats, data_samples, test_cfg):
            self.batch_sizes.append(feats.shape[0])
            return list(data_samples)

    class _Model(_PoseModelFixture):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Identity()
            self.head = _Head()

    model = _Model()
    runner = pose_factory_plugin._TrtPoseRunner(
        model,
        engine_path=str(tmp_path / "pose.engine"),
        batch_size=2,
    )
    data_samples = [object() for _ in range(5)]

    result = runner.forward(
        {
            "inputs": torch.randn(5, 3, 16, 16),
            "data_samples": data_samples,
        }
    )

    assert result == data_samples
    assert model.head.batch_sizes == [2, 2, 1]
    assert len(runner._trt_modules) == 1


def should_cache_detector_engines_per_static_shape(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import detector_factory_plugin

    compiled_modules = []

    def _compile(*_args, **_kwargs):
        compiled = torch.nn.Identity()
        compiled_modules.append(compiled)
        return compiled

    monkeypatch.setattr(detector_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.neck = torch.nn.Identity()
            self.bbox_head = torch.nn.Identity()

    runner = detector_factory_plugin._TrtDetectorWrapper(
        _Model(),
        engine_path=str(tmp_path / "detector.engine"),
        nms_backend="torchvision",
    )
    size_16 = torch.randn(1, 3, 16, 16)
    size_24 = torch.randn(1, 3, 24, 24)

    runner._ensure_trt_engine(size_16)
    first_module = runner._trt_module
    runner._ensure_trt_engine(size_24)
    runner._ensure_trt_engine(size_16)

    assert len(compiled_modules) == 2
    assert runner._trt_module is first_module


def should_run_detector_backbone_and_head_as_one_batch(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import detector_factory_plugin

    compiled_shapes = []

    def _compile(_module, inputs, **_kwargs):
        compiled_shapes.append(tuple(inputs[0].shape))
        return torch.nn.Identity()

    monkeypatch.setattr(detector_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0
            self.predict_calls = 0

        def forward(self, feats):
            self.forward_calls += 1
            return feats[0], feats[1], feats[2]

        def predict_by_feat(self, *, batch_img_metas, **_kwargs):
            self.predict_calls += 1
            results = []
            for _meta in batch_img_metas:
                inst = detector_factory_plugin.InstanceData()
                inst.bboxes = torch.zeros((1, 4))
                inst.scores = torch.ones((1,))
                inst.labels = torch.zeros((1,), dtype=torch.long)
                results.append(inst)
            return results

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.neck = torch.nn.Identity()
            self.bbox_head = _Head()

    class _Nms:
        def __init__(self) -> None:
            self.batch_sizes = []

        def run_batch(self, instances, img_metas):
            self.batch_sizes.append((len(instances), len(img_metas)))
            return list(instances)

    model = _Model()
    runner = detector_factory_plugin._TrtDetectorWrapper(
        model,
        engine_path=str(tmp_path / "detector.engine"),
        nms_backend="torchvision",
    )
    nms = _Nms()
    runner._nms = nms
    data_samples = [SimpleNamespace(metainfo={"sample": index}) for index in range(3)]

    results = runner.predict(torch.randn(3, 3, 16, 16), data_samples)

    assert len(results) == 3
    assert compiled_shapes == [(3, 3, 16, 16)]
    assert model.bbox_head.forward_calls == 1
    assert model.bbox_head.predict_calls == 1
    assert nms.batch_sizes == [(3, 3)]


def should_resolve_rocm_detector_backends_to_head(monkeypatch) -> None:
    from hmlib.aspen.plugins import detector_factory_plugin

    monkeypatch.setattr(detector_factory_plugin, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(detector_factory_plugin, "_runtime_supports_tensorrt", lambda: False)
    monkeypatch.setattr(detector_factory_plugin, "_runtime_supports_torchvision_nms", lambda: False)

    assert (
        detector_factory_plugin._resolve_detector_nms_backend("trt", owner="test detector wrapper")
        == "head"
    )
    assert (
        detector_factory_plugin._resolve_detector_nms_backend(
            "torchvision", owner="test detector wrapper"
        )
        == "head"
    )
    assert (
        detector_factory_plugin._resolve_detector_nms_backend("head", owner="test detector wrapper")
        == "head"
    )


def should_refuse_requested_trt_detector_backend_on_rocm(monkeypatch) -> None:
    from hmlib.aspen.plugins import detector_factory_plugin

    monkeypatch.setattr(detector_factory_plugin, "_is_rocm_runtime", lambda: True)
    monkeypatch.setattr(detector_factory_plugin, "_runtime_supports_tensorrt", lambda: False)
    plugin = detector_factory_plugin.DetectorFactoryPlugin(trt={"enable": True})
    plugin._model = torch.nn.Identity()

    with pytest.raises(RuntimeError, match="TensorRT detector backend is unavailable on ROCm"):
        plugin.forward({})
    assert plugin._runtime_detector is None


def should_surface_pose_compile_failure_without_switching_backend(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import pose_factory_plugin

    def _compile(*_args, **_kwargs):
        raise RuntimeError("unsupported TensorRT operator")

    monkeypatch.setattr(pose_factory_plugin, "compile_torch_tensorrt", _compile)

    class _Head(torch.nn.Module):
        def predict(self, feats, data_samples, test_cfg):
            return [(feats, data_samples)]

    class _Model(_PoseModelFixture):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.head = _Head()

    runner = pose_factory_plugin._TrtPoseRunner(_Model(), engine_path=str(tmp_path / "pose.engine"))
    with pytest.raises(RuntimeError, match="TensorRT pose compilation failed") as raised:
        runner.forward(
            {
                "inputs": torch.randn(1, 3, 16, 16),
                "data_samples": [object()],
            }
        )
    assert "unsupported TensorRT operator" in str(raised.value.__cause__)
    assert runner._trt_module is None


def should_surface_pose_decode_failure_without_switching_backend(monkeypatch, tmp_path) -> None:
    from hmlib.aspen.plugins import pose_factory_plugin

    monkeypatch.setattr(
        pose_factory_plugin,
        "compile_torch_tensorrt",
        lambda *_args, **_kwargs: torch.nn.Identity(),
    )

    class _Head(torch.nn.Module):
        def predict(self, _feats, _data_samples, test_cfg):
            raise ValueError("malformed pose metadata")

    class _Model(_PoseModelFixture):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Identity()
            self.head = _Head()

    runner = pose_factory_plugin._TrtPoseRunner(
        _Model(),
        engine_path=str(tmp_path / "pose.engine"),
        batch_size=2,
    )
    fallback_result = [object()]
    runner.set_fallback_test_step(lambda _inputs: fallback_result)

    with pytest.raises(RuntimeError, match="TensorRT pose inference failed") as raised:
        runner.forward(
            {
                "inputs": torch.randn(1, 3, 16, 16),
                "data_samples": [object()],
            }
        )
    assert "malformed pose metadata" in str(raised.value.__cause__)
    assert runner._trt_module is not None
