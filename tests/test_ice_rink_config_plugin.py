from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
def should_configure_ice_rink_mask_from_numpy_image(tmp_path, monkeypatch) -> None:
    from hmlib.segm import ice_rink

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    captured = {}

    monkeypatch.setattr(
        ice_rink,
        "get_model_config",
        lambda game_id, model_name: ("config.py", "checkpoint.pth"),
    )
    monkeypatch.setattr(ice_rink, "get_game_dir", lambda game_id, assert_exists=True: str(tmp_path))
    monkeypatch.setattr(ice_rink, "prepend_root_dir", lambda path: path)

    def find_masks(**kwargs):
        captured.update(kwargs)
        return {"combined_mask": torch.ones((20, 30), dtype=torch.bool)}

    monkeypatch.setattr(ice_rink, "find_ice_rink_masks", find_masks)

    result = ice_rink.configure_ice_rink_mask(
        game_id="game-1",
        expected_shape=torch.Size((20, 30)),
        device=torch.device("cpu"),
        force=True,
        image=image,
        persist=False,
    )

    assert result["combined_mask"].shape == (20, 30)
    assert captured["image"] is image
    assert captured["device"] == torch.device("cpu")


@requires_torch
def should_reuse_only_the_geometry_keyed_rink_mask(tmp_path, monkeypatch) -> None:
    from hmlib.segm import ice_rink

    game_config = {"rink": {}}
    monkeypatch.setattr(ice_rink, "get_game_config_private", lambda game_id: game_config)
    monkeypatch.setattr(ice_rink, "get_game_dir", lambda game_id, assert_exists=True: str(tmp_path))
    monkeypatch.setattr(
        ice_rink,
        "save_private_config",
        lambda game_id, data, verbose=True: None,
    )

    profile = {
        "masks": [torch.ones((20, 30), dtype=torch.bool)],
        "centroid": torch.tensor([15.0, 10.0]),
        "combined_bbox": [0.0, 0.0, 30.0, 20.0],
    }
    ice_rink.save_rink_profile_config(
        game_id="game-1", rink_profile=profile, geometry_revision="geometry-v1"
    )
    assert game_config["rink"]["ice_contours_geometry_revision"] == "geometry-v1"
    assert list(tmp_path.glob("rink_mask_*_0.png"))

    monkeypatch.setattr(
        ice_rink,
        "get_model_config",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model must not load")),
    )
    cached = ice_rink.configure_ice_rink_mask(
        game_id="game-1",
        expected_shape=torch.Size((20, 30)),
        geometry_revision="geometry-v1",
    )
    assert torch.equal(cached["combined_mask"], profile["masks"][0])

    monkeypatch.setattr(
        ice_rink,
        "get_model_config",
        lambda **kwargs: ("config.py", "checkpoint.pth"),
    )
    monkeypatch.setattr(ice_rink, "prepend_root_dir", lambda path: path)
    monkeypatch.setattr(
        ice_rink,
        "find_ice_rink_masks",
        lambda **kwargs: {
            "combined_mask": torch.zeros((20, 30), dtype=torch.bool),
            "masks": [torch.zeros((20, 30), dtype=torch.bool)],
            "centroid": torch.tensor([15.0, 10.0]),
            "combined_bbox": [0.0, 0.0, 30.0, 20.0],
        },
    )
    regenerated = ice_rink.configure_ice_rink_mask(
        game_id="game-1",
        expected_shape=torch.Size((20, 30)),
        geometry_revision="geometry-v2",
        image=np.zeros((20, 30, 3), dtype=np.uint8),
    )
    assert not torch.equal(regenerated["combined_mask"], profile["masks"][0])
    assert game_config["rink"]["ice_contours_geometry_revision"] == "geometry-v2"


@requires_torch
def should_clear_the_revision_when_saving_without_one(tmp_path, monkeypatch) -> None:
    """A revisionless save must not strand the previous revision on the config."""
    from hmlib.segm import ice_rink

    game_config = {"rink": {}}
    monkeypatch.setattr(ice_rink, "get_game_config_private", lambda game_id: game_config)
    monkeypatch.setattr(ice_rink, "get_game_dir", lambda game_id, assert_exists=True: str(tmp_path))
    monkeypatch.setattr(ice_rink, "save_private_config", lambda game_id, data, verbose=True: None)

    profile = {
        "masks": [torch.ones((20, 30), dtype=torch.bool)],
        "centroid": torch.tensor([15.0, 10.0]),
        "combined_bbox": [0.0, 0.0, 30.0, 20.0],
    }
    ice_rink.save_rink_profile_config(
        game_id="game-1", rink_profile=profile, geometry_revision="geometry-v1"
    )
    assert ice_rink.load_rink_combined_mask(game_id="game-1") is not None

    # Legacy callers pass no revision. The masks they write live under the bare
    # prefix, so a leftover revision would point the loader at the wrong files.
    ice_rink.save_rink_profile_config(game_id="game-1", rink_profile=profile)
    assert "ice_contours_geometry_revision" not in game_config["rink"]
    reloaded = ice_rink.load_rink_combined_mask(game_id="game-1")
    assert reloaded is not None
    assert reloaded["geometry_revision"] is None
    assert torch.equal(reloaded["combined_mask"], profile["masks"][0])


@requires_torch
def should_not_accumulate_masks_for_superseded_revisions(tmp_path, monkeypatch) -> None:
    """A game with no durable geometry identity gets a new revision every run."""
    from hmlib.segm import ice_rink

    game_config = {"rink": {}}
    monkeypatch.setattr(ice_rink, "get_game_config_private", lambda game_id: game_config)
    monkeypatch.setattr(ice_rink, "get_game_dir", lambda game_id, assert_exists=True: str(tmp_path))
    monkeypatch.setattr(ice_rink, "save_private_config", lambda game_id, data, verbose=True: None)

    profile = {
        "masks": [torch.ones((20, 30), dtype=torch.bool)],
        "centroid": torch.tensor([15.0, 10.0]),
        "combined_bbox": [0.0, 0.0, 30.0, 20.0],
    }
    for run in range(5):
        ice_rink.save_rink_profile_config(
            game_id="game-1", rink_profile=profile, geometry_revision=f"run-{run}"
        )
        # Only the current revision plus the bare pointer ever remain.
        assert len(list(tmp_path.glob("rink_mask_*.png"))) == 2
        assert ice_rink.load_rink_combined_mask(game_id="game-1") is not None

    # A run snapshot is not a revision-scoped mask and must not be pruned.
    snapshot = tmp_path / "rink_mask_0-17.png"
    snapshot.write_bytes(b"run snapshot")
    ice_rink.save_rink_profile_config(
        game_id="game-1", rink_profile=profile, geometry_revision="run-final"
    )
    assert snapshot.exists()


@requires_torch
def should_rebuild_rather_than_raise_on_a_truncated_mask(tmp_path, monkeypatch) -> None:
    from hmlib.segm import ice_rink

    game_config = {"rink": {}}
    monkeypatch.setattr(ice_rink, "get_game_config_private", lambda game_id: game_config)
    monkeypatch.setattr(ice_rink, "get_game_dir", lambda game_id, assert_exists=True: str(tmp_path))
    monkeypatch.setattr(ice_rink, "save_private_config", lambda game_id, data, verbose=True: None)

    ice_rink.save_rink_profile_config(
        game_id="game-1",
        rink_profile={
            "masks": [torch.ones((20, 30), dtype=torch.bool)],
            "centroid": torch.tensor([15.0, 10.0]),
            "combined_bbox": [0.0, 0.0, 30.0, 20.0],
        },
        geometry_revision="geometry-v1",
    )
    cached = next(iter(tmp_path.glob("rink_mask_*_0.png")))
    cached.write_bytes(cached.read_bytes()[:12])
    assert ice_rink.load_rink_combined_mask(game_id="game-1") is None


@requires_torch
def should_point_the_bare_mask_name_at_the_current_revision(tmp_path, monkeypatch) -> None:
    """Consumers that address rink_mask_0.png directly must see the newest mask."""
    from hmlib.segm import ice_rink
    from hmlib.segm.ice_rink import load_png_as_boolean_tensor
    from hmlib.stitching.configure_stitching import _calibration_masks

    game_config = {"rink": {}}
    monkeypatch.setattr(ice_rink, "get_game_config_private", lambda game_id: game_config)
    monkeypatch.setattr(ice_rink, "get_game_dir", lambda game_id, assert_exists=True: str(tmp_path))
    monkeypatch.setattr(ice_rink, "save_private_config", lambda game_id, data, verbose=True: None)

    def save(mask: torch.Tensor, revision: str) -> None:
        ice_rink.save_rink_profile_config(
            game_id="game-1",
            rink_profile={
                "masks": [mask],
                "centroid": torch.tensor([15.0, 10.0]),
                "combined_bbox": [0.0, 0.0, 30.0, 20.0],
            },
            geometry_revision=revision,
        )

    first = torch.ones((20, 30), dtype=torch.bool)
    save(first, "geometry-v1")
    bare = tmp_path / "rink_mask_0.png"
    assert torch.equal(load_png_as_boolean_tensor(str(bare)), first)

    second = torch.zeros((20, 30), dtype=torch.bool)
    second[0, 0] = True
    save(second, "geometry-v2")
    assert torch.equal(load_png_as_boolean_tensor(str(bare)), second)

    # A run snapshot is not calibration cache; every revision-scoped copy is,
    # so calibration invalidation must not leave them behind.
    snapshot = tmp_path / "rink_mask_0-17.png"
    snapshot.write_bytes(b"run snapshot")
    stale = {path.name for path in _calibration_masks(tmp_path)}
    assert "rink_mask_0.png" in stale
    assert snapshot.name not in stale
    # The superseded geometry-v1 copy was pruned, so only geometry-v2 remains.
    assert len([name for name in stale if name != "rink_mask_0.png"]) == 1


@requires_torch
def should_skip_ice_rink_mask_when_game_id_is_missing(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "hmlib.segm.ice_rink.configure_ice_rink_mask",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("rink mask should not be configured without a game_id")
        ),
    )

    plugin = IceRinkSegmConfigPlugin()
    context = {
        "data_samples": make_track_data_sample(num_frames=1, ori_shape=(20, 30)),
        "original_images": wrap_tensor(torch.zeros((1, 20, 30, 3), dtype=torch.float32)),
    }

    with caplog.at_level("WARNING"):
        out = plugin.forward(context)

    assert out == {}
    assert "No game_id is available" in caplog.text


@requires_torch
def should_skip_ice_rink_mask_without_game_id_even_with_telemetry_geometry(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        "hmlib.segm.ice_rink.configure_ice_rink_mask",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("rink mask should not be configured without a game_id")
        ),
    )

    plugin = IceRinkSegmConfigPlugin()
    context = {
        "data_samples": make_track_data_sample(num_frames=1, ori_shape=(20, 30)),
        "original_images": wrap_tensor(torch.zeros((1, 20, 30, 3), dtype=torch.float32)),
        "camera_input_geometry": {"stitched_geometry_revision": "calibration-1"},
        "telemetry_batch": object(),
    }

    with caplog.at_level("WARNING"):
        out = plugin.forward(context)

    assert out == {}
    assert "No game_id is available" in caplog.text


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
    assert captured["force"] is False and captured["persist"] is True
    assert captured["geometry_revision"] == "calibration-1"


@requires_torch
def should_regenerate_for_same_size_geometry_change_keyed_by_revision(monkeypatch):
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
    assert all(
        not call["force"]
        and call["persist"]
        and call["geometry_revision"]
        in {
            "first-calibration",
            "second-calibration",
        }
        for call in calls
    )
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


@requires_torch
def should_snapshot_loaded_mask_once_for_run_publication(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    mask = torch.zeros((20, 30), dtype=torch.bool)
    mask[2:8, 4:12] = True
    calls = []

    def configure(**kwargs):
        calls.append(kwargs)
        return {"combined_mask": mask}

    monkeypatch.setattr("hmlib.segm.ice_rink.configure_ice_rink_mask", configure)
    plugin = IceRinkSegmConfigPlugin()
    context = {
        "data_samples": make_track_data_sample(num_frames=1, ori_shape=(20, 30)),
        "original_images": torch.zeros(1, 3, 20, 30),
        "shared": {"game_id": "snapshot-game", "work_dir": str(tmp_path)},
    }
    plugin.forward(context)
    snapshot = tmp_path / "rink_mask_0.png"
    original = snapshot.read_bytes()
    assert np.array_equal(np.asarray(Image.open(snapshot)) > 0, mask.numpy())
    plugin.forward(context)
    assert len(calls) == 1 and snapshot.read_bytes() == original
    assert list(tmp_path.iterdir()) == [snapshot]


@requires_torch
def should_reject_changed_mask_in_a_single_published_run(tmp_path, monkeypatch):
    calls = []

    def configure(**kwargs):
        calls.append(kwargs)
        return {
            "combined_mask": torch.full(kwargs["expected_shape"], len(calls) == 1, dtype=torch.bool)
        }

    monkeypatch.setattr("hmlib.segm.ice_rink.configure_ice_rink_mask", configure)
    plugin = IceRinkSegmConfigPlugin(require_geometry_provenance=True)
    context = {
        "data_samples": make_track_data_sample(num_frames=1, ori_shape=(20, 30)),
        "original_images": torch.zeros(1, 3, 20, 30),
        "camera_input_geometry": {"stitched_geometry_revision": "first"},
        "shared": {"game_id": "snapshot-game", "work_dir": str(tmp_path)},
    }
    plugin.forward(context)
    original = (tmp_path / "rink_mask_0.png").read_bytes()
    context["camera_input_geometry"] = {"stitched_geometry_revision": "second"}
    with pytest.raises(ValueError, match="Rink mask changed"):
        plugin.forward(context)
    assert (tmp_path / "rink_mask_0.png").read_bytes() == original
