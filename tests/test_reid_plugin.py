"""Appearance embeddings reach the actual tracker payload without changing boxes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hmlib.aspen.plugins.reid_plugin import (
    ReidPlugin,
    configure_reid,
    player_crops,
    validate_reid_session,
)
from hmlib.utils.gpu import unwrap_tensor


class Session:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="images", type="tensor(float)", shape=["N", 3, 256, 128])]

    def get_outputs(self):
        return [SimpleNamespace(shape=["N", 256])]

    def run(self, outputs, inputs):
        self.calls.append(inputs["images"])
        return [np.ones((len(inputs["images"]), 256), dtype=np.float32)]


def should_apply_native_rgb_preprocessing_without_channel_swaps():
    image = torch.tensor([123.675, 116.28, 103.53])[:, None, None].expand(3, 10, 20)
    crop = player_crops(image, torch.tensor([[0, 0, 20, 10]]))
    assert crop.shape == (1, 3, 256, 128)
    torch.testing.assert_close(crop, torch.zeros_like(crop), atol=1e-5, rtol=0)


def should_attach_normalized_features_only_to_valid_detections():
    plugin = ReidPlugin("unused.onnx", batch_size=1)
    plugin.session = Session()
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 1.0, 8.0, 9.0], [0.0, 0.0, 1.0, 1.0]])
    instances = SimpleNamespace(bboxes=boxes, scores=torch.ones(3), num_detections=2)
    sample = SimpleNamespace(video_data_samples=[SimpleNamespace(pred_instances=instances)])
    context = {"original_images": torch.ones(1, 3, 10, 10), "data_samples": [sample]}
    result = plugin.forward(context)
    features = unwrap_tensor(instances.reid_features)
    assert len(plugin.session.calls) == 2
    torch.testing.assert_close(features[:2].norm(dim=1), torch.ones(2))
    assert features[2].count_nonzero() == 0
    assert instances.bboxes is boxes
    assert result["data_samples"] is context["data_samples"]


@pytest.mark.parametrize("channels_last", [False, True])
def should_convert_pipeline_bgr_frames_to_rgb_before_model_inference(channels_last):
    plugin = ReidPlugin("unused.onnx")
    plugin.session = Session()
    bgr = torch.tensor([30.0, 70.0, 210.0])[:, None, None].expand(3, 10, 20)
    image = bgr.permute(1, 2, 0) if channels_last else bgr
    instances = SimpleNamespace(bboxes=torch.tensor([[0.0, 0.0, 20.0, 10.0]]), scores=torch.ones(1))
    sample = SimpleNamespace(video_data_samples=[SimpleNamespace(pred_instances=instances)])
    plugin.forward({"original_images": image[None], "data_samples": [sample]})
    actual = torch.from_numpy(plugin.session.calls[0])
    expected = torch.tensor([210.0, 70.0, 30.0]) - torch.tensor([123.675, 116.28, 103.53])
    expected = (expected * 0.01735207357279195).reshape(1, 3, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual, expected)


def should_keep_disabled_graph_unchanged_and_insert_reid_before_tracking(tmp_path):
    config = {"aspen": {"plugins": {"tracker": {"depends": ["mask"], "params": {}}}}}
    configure_reid(config, tmp_path)
    assert list(config["aspen"]["plugins"]) == ["tracker"]
    path = tmp_path / "appearance.onnx"
    path.touch()
    config["pipeline"] = {"tracker": {"reid-enable": True}}
    config["hm_ui"] = {"run": {"reid_onnx": str(path)}}
    configure_reid(config, tmp_path)
    plugins = config["aspen"]["plugins"]
    assert plugins["player_reid"]["depends"] == ["mask"]
    assert plugins["tracker"]["depends"] == ["player_reid"]
    assert plugins["tracker"]["params"]["tracker_class"].endswith("HmDcfTrackerCudaStatic")
    assert plugins["tracker"]["params"]["tracker_kwargs"]["reid_feature_dim"] == 256


def should_reject_models_with_incompatible_feature_dimensions():
    session = Session()
    session.get_outputs = lambda: [SimpleNamespace(shape=["N", 512])]
    with pytest.raises(ValueError, match="contract"):
        validate_reid_session(session)


def should_use_the_same_embedding_and_pixel_lightness_for_nvidia_team_clustering():
    from hmlib.team.backends import TaoBackend

    backend = TaoBackend("unused.onnx")
    backend._runner.session = Session()
    pixels = torch.stack([torch.full((3, 256, 128), 40.0), torch.full((3, 256, 128), 210.0)])
    result = backend.extract(pixels)
    assert result.features.shape == (2, 256)
    torch.testing.assert_close(result.features.norm(dim=1), torch.ones(2))
    assert result.lightness[0] < result.lightness[1]


def should_enable_nvidia_team_embeddings_without_changing_tracker(tmp_path):
    path = tmp_path / "appearance.onnx"
    path.touch()
    config = {
        "team": {"enabled": True, "backend": "reidentificationnet"},
        "hm_ui": {"run": {"reid_onnx": str(path)}},
        "aspen": {
            "plugins": {
                "tracker": {"depends": ["detector"], "params": {}},
                "team": {
                    "class": "hmlib.aspen.plugins.team_plugin.TeamAffiliationPlugin",
                    "params": {},
                },
            }
        },
    }
    configure_reid(config, tmp_path)
    assert config["aspen"]["plugins"]["tracker"]["params"] == {}
    assert config["aspen"]["plugins"]["team"]["params"]["reid_onnx_path"] == str(path)


def should_exchange_bounded_native_appearance_batches_and_reap_helper(tmp_path, monkeypatch):
    import os
    import sys

    from hmlib.models.appearance import TensorRtAppearanceSession

    helper = tmp_path / "native-infer"
    helper.write_text(f"#!{sys.executable}\n" + """import os, struct, sys
import numpy as np
assert os.environ['HSTREAM_UI_PARENT_PID'] == str(os.getppid())
def read(n):
    out=b''
    while len(out)<n:
        part=sys.stdin.buffer.read(n-len(out))
        if not part: raise RuntimeError('partial input')
        out+=part
    return out
while header := sys.stdin.buffer.read(4):
    n=struct.unpack('<I',header)[0]
    if n==0: break
    assert 0<n<=32
    crops=np.frombuffer(read(n*3*256*128*4),dtype='<f4').reshape(n,-1)
    values=np.repeat(crops.mean(axis=1)[:,None],256,axis=1).astype('<f4')
    sys.stdout.buffer.write(struct.pack('<I',n)+values.tobytes())
    sys.stdout.buffer.flush()
""")
    helper.chmod(0o755)
    monkeypatch.setenv("HM_HSTREAM_MODEL_INFER", str(helper))
    runner = TensorRtAppearanceSession(tmp_path / "exact-model.engine")
    pid = runner.process.pid
    # Model initialization can precede video-reader forks. An inherited writer
    # must not keep the helper alive after the owning pipeline finalizes.
    inherited_writer = os.dup(runner.process.stdin.fileno())
    try:
        for count in (1, 32):
            result = runner.run(None, {"images": np.full((count, 3, 256, 128), 7.0, np.float32)})[0]
            assert result.shape == (count, 256)
            np.testing.assert_array_equal(result, 7.0)
        with pytest.raises(ValueError, match="at most 32"):
            runner.run(None, {"images": np.zeros((33, 3, 256, 128), np.float32)})
    finally:
        try:
            runner.close()
        finally:
            os.close(inherited_writer)
    assert runner.process.returncode == 0
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def should_surface_native_appearance_runtime_errors_without_substitution(tmp_path, monkeypatch):
    import sys

    from hmlib.models.appearance import TensorRtAppearanceSession

    helper = tmp_path / "failed-infer"
    helper.write_text(
        f'#!{sys.executable}\nimport sys\nprint("exact engine runtime failure", file=sys.stderr)\nsys.exit(7)\n'
    )
    helper.chmod(0o755)
    monkeypatch.setenv("HM_HSTREAM_MODEL_INFER", str(helper))
    runner = TensorRtAppearanceSession(tmp_path / "selected.engine")
    try:
        with pytest.raises(RuntimeError, match="exact engine runtime failure"):
            runner.run(None, {"images": np.zeros((1, 3, 256, 128), np.float32)})
    finally:
        with pytest.raises(RuntimeError, match="exact engine runtime failure"):
            runner.close()


def should_use_native_custom_normalization_defaults_consistently(tmp_path):
    import yaml

    from hmlib.models.appearance import read_tracker_model

    (tmp_path / "model.engine").write_bytes(b"prepared-engine")
    config = tmp_path / "tracker.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "ReID": {
                    "inferDims": [3, 256, 128],
                    "reidFeatureSize": 256,
                    "modelEngineFile": "model.engine",
                }
            }
        )
    )
    engine, settings = read_tracker_model(config)
    assert engine == tmp_path / "model.engine"
    pixels = torch.full((3, 10, 10), 7.0)
    actual = player_crops(pixels, torch.tensor([[0.0, 0.0, 10.0, 10.0]]), settings=settings)
    torch.testing.assert_close(actual, torch.full_like(actual, 7.0))
