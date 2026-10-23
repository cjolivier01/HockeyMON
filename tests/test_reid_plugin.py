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
