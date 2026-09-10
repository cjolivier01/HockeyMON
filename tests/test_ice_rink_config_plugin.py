from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

if torch is not None:
    TESTS_DIR = Path(__file__).resolve().parent
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))

    from aspen_plugin_harness import make_track_data_sample
    from hmlib.aspen.plugins.ice_rink_boundaries_plugins import IceRinkSegmConfigPlugin
    from hmlib.utils.gpu import wrap_tensor
else:
    make_track_data_sample = None  # type: ignore[assignment]
    IceRinkSegmConfigPlugin = None  # type: ignore[assignment]
    wrap_tensor = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


@requires_torch
def should_prefer_stitched_frame_shape_over_detector_input_shape(monkeypatch) -> None:
    captured = {}

    def _fake_configure(**kwargs):
        captured.update(kwargs)
        return {
            "shape": tuple(kwargs["expected_shape"]),
            "combined_mask": torch.ones(tuple(kwargs["expected_shape"]), dtype=torch.bool),
        }

    monkeypatch.setattr(
        "hmlib.segm.ice_rink.configure_ice_rink_mask",
        _fake_configure,
    )

    plugin = IceRinkSegmConfigPlugin(require_geometry_provenance=True)
    result = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1, ori_shape=(20, 30)),
            "original_images": wrap_tensor(torch.zeros((1, 20, 30, 3), dtype=torch.float32)),
            "img": wrap_tensor(torch.zeros((1, 10, 15, 3), dtype=torch.float32)),
            "camera_input_geometry": {"stitched_geometry_revision": "calibration-1"},
            "inputs": torch.zeros((1, 3, 736, 1984), dtype=torch.float32),
            "shared": {"game_id": "game-1"},
        }
    )

    assert result["rink_profile"]["shape"] == (20, 30)
    assert tuple(captured["expected_shape"]) == (20, 30)
    assert isinstance(captured["image"], torch.Tensor)
    assert tuple(captured["image"].shape) == (20, 30, 3)

    assert result["rink_profile"]["coordinate_space"] == "original_stitched_pixels"
    assert result["rink_profile"]["frame_size"] == [30, 20]
    assert len(result["rink_profile"]["geometry_revision"]) == 64
    assert captured["force"] is True and captured["persist"] is False


@requires_torch
def should_regenerate_for_same_size_geometry_change_without_overwriting_masks(monkeypatch):
    calls = []

    def configure(**kwargs):
        calls.append(kwargs)
        return {"combined_mask": torch.ones(kwargs["expected_shape"], dtype=torch.bool)}

    monkeypatch.setattr("hmlib.segm.ice_rink.configure_ice_rink_mask", configure)
    plugin = IceRinkSegmConfigPlugin(require_geometry_provenance=True)
    context = {
        "data_samples": make_track_data_sample(num_frames=1, ori_shape=(20, 30)),
        "original_images": torch.zeros(1, 3, 20, 30),
        "game_id": "same-game",
        "camera_input_geometry": {"stitched_geometry_revision": "first-calibration"},
    }
    first = plugin.forward(context)["rink_profile"]
    assert plugin.forward(context)["rink_profile"] is first
    assert len(calls) == 1
    context["camera_input_geometry"] = {"stitched_geometry_revision": "second-calibration"}
    second = plugin.forward(context)["rink_profile"]
    assert len(calls) == 2
    assert first["geometry_revision"] != second["geometry_revision"]
    assert all(call["force"] and not call["persist"] for call in calls)
    # A legacy saved mask with the same dimensions must not acquire grid provenance.
    legacy = IceRinkSegmConfigPlugin()
    assert "coordinate_space" not in legacy.forward(context)["rink_profile"]
    assert not calls[-1]["force"] and calls[-1]["persist"]
    context.pop("camera_input_geometry")
    with pytest.raises(ValueError, match="requires a stitched geometry revision"):
        plugin.forward(context)


@requires_torch
def should_reuse_static_rink_embedding_and_reset_camera_history(tmp_path, monkeypatch):
    from aspen_plugin_harness import make_instance_data
    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin
    from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, pack_gpt_checkpoint
    from hmlib.camera.camera_transformer import CameraNorm
    from hmlib.utils.gpu import unwrap_tensor

    cfg = CameraGPTConfig(
        d_in=12,
        d_out=4,
        feature_mode="base_prev_y",
        include_rink=True,
        rink_input="grid",
        rink_grid_height=8,
        rink_grid_width=16,
        d_model=16,
        nhead=4,
        nlayers=1,
        dim_feedforward=32,
        dropout=0,
    )
    model = CameraPanZoomGPT(cfg)
    path = tmp_path / "camera.pt"
    torch.save(pack_gpt_checkpoint(model, CameraNorm(200, 100, 22), 4, cfg), path)
    controller = CameraControllerPlugin(controller="gpt", model_path=str(path))
    monkeypatch.setattr(controller, "_ensure_cluster_man", lambda: None)
    calls = []
    controller._gpt_model.rink_encoder.register_forward_hook(lambda *args: calls.append(1))
    profile = {
        "combined_mask": torch.ones((100, 200), dtype=torch.bool),
        "combined_bbox": [0, 0, 200, 100],
        "coordinate_space": "original_stitched_pixels",
        "frame_size": [200, 100],
        "geometry_revision": "v1",
    }

    def context():
        tracks = [
            make_instance_data(
                bboxes=torch.tensor([[40.0, 30.0, 50.0, 50.0], [110.0, 30.0, 120.0, 50.0]]),
                instances_id=torch.tensor([1, 2]),
                labels=torch.zeros(2, dtype=torch.long),
                scores=torch.ones(2),
            )
            for _ in range(2)
        ]
        return {
            "data_samples": make_track_data_sample(
                num_frames=2, ori_shape=(100, 200), pred_track_instances=tracks
            ),
            "rink_profile": profile,
            "game_id": "test",
            "device": torch.device("cpu"),
        }

    out = controller(context())
    assert unwrap_tensor(out["camera_boxes"]).shape == (2, 4)
    assert len(calls) == 1 and len(controller._feat_buf) == 2
    controller(context())
    assert len(calls) == 1 and len(controller._feat_buf) == 4
    profile["geometry_revision"] = "v2"
    controller(context())
    assert len(calls) == 2 and len(controller._feat_buf) == 2
    for missing in (True, False):
        invalid = context()
        invalid.pop("rink_profile")
        for sample in invalid["data_samples"]:
            if missing:
                del sample.pred_track_instances
            else:
                sample.pred_track_instances = sample.pred_track_instances[:0]
        with pytest.raises(ValueError, match="requires a profile"):
            controller(invalid)
