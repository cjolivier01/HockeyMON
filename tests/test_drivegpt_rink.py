import json

import cv2
import numpy as np
import pytest
import torch

from hmlib.camera.camera_gpt import (
    CameraGPTConfig,
    CameraPanZoomGPT,
    pack_gpt_checkpoint,
    unpack_gpt_checkpoint,
)
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.camera.rink_context import (
    RINK_CONTEXT_SCHEMA,
    StaticRinkEncoderCache,
    file_sha256,
    load_rink_grid,
    mask_to_grid,
    read_rink_context,
    rink_context_path,
)
from hmlib.cli.camgpt_train import TrainingRollout, _predict_batch


def _model():
    return CameraPanZoomGPT(
        CameraGPTConfig(
            d_in=12,
            feature_mode="base_prev_y",
            d_out=4,
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
    )


def _binding(tmp_path):
    tracking = tmp_path / "tracking-3.csv"
    tracking.write_text("1,1,90,50,10,20,0.9,0,1,{}\n")
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    path = tmp_path / "rink_mask_0.png"
    assert cv2.imwrite(str(path), mask)
    context = {
        "schema": RINK_CONTEXT_SCHEMA,
        "coordinate_space": "original_stitched_pixels",
        "frame_size": [200, 100],
        "tracking": {"file": tracking.name, "sha256": file_sha256(tracking)},
        "masks": [
            {
                "file": path.name,
                "sha256": file_sha256(path),
                "mask_to_tracking": [[2, 0, 80], [0, 2, 40]],
            }
        ],
        "evidence": "Synthetic mask coordinates explicitly transformed into the tracking canvas.",
    }
    rink_context_path(str(tracking)).write_text(json.dumps(context))
    return tracking, context


def should_align_transformed_rink_with_tracking_and_live_mask(tmp_path):
    tracking, _ = _binding(tmp_path)
    norm = CameraNorm(200, 100, 22)
    grid = load_rink_grid(str(tracking), norm, 10, 20)
    expected = np.zeros((1, 10, 20), dtype=np.float32)
    expected[:, 5:7, 9:11] = 1
    np.testing.assert_allclose(grid, expected)
    live = np.zeros((100, 200), dtype=np.uint8)
    live[50:70, 90:110] = 255
    np.testing.assert_allclose(mask_to_grid(live, norm, 10, 20), grid)
    # A larger shared world must pad, not stretch each rink independently.
    padded = mask_to_grid(live, CameraNorm(400, 200, 22), 20, 40)
    np.testing.assert_allclose(padded[:, :10, :20], grid)
    assert not padded[:, 10:].any() and not padded[:, :, 20:].any()


def should_reject_unbound_changed_and_singular_rink_context(tmp_path):
    tracking, context = _binding(tmp_path)
    assert read_rink_context(str(tracking))["frame_size"] == [200, 100]
    tracking.write_text(tracking.read_text() + "2,1,90,50,10,20,0.9,0,1,{}\n")
    with pytest.raises(ValueError, match="tracking checksum"):
        read_rink_context(str(tracking))
    context["tracking"]["sha256"] = file_sha256(tracking)
    context["masks"][0]["mask_to_tracking"] = [[0, 0, 0], [0, 0, 0]]
    rink_context_path(str(tracking)).write_text(json.dumps(context))
    with pytest.raises(ValueError, match="affine"):
        read_rink_context(str(tracking))
    with pytest.raises(FileNotFoundError):
        read_rink_context(str(tmp_path / "tracking-2.csv"))


@pytest.mark.parametrize("overlap", [0, 2])
def should_match_live_union_at_component_seams_and_fractional_grid_edges(tmp_path, overlap):
    tracking, context = _binding(tmp_path)
    context["frame_size"] = [8, 8]
    context["masks"] = []
    union = np.zeros((8, 8), dtype=np.uint8)
    for index, columns in enumerate((slice(0, 4 + overlap), slice(4, 8))):
        mask = np.zeros_like(union)
        mask[:, columns] = 255
        union |= mask
        path = tmp_path / f"component-{index}.png"
        assert cv2.imwrite(str(path), mask)
        context["masks"].append(
            {
                "file": path.name,
                "sha256": file_sha256(path),
                "mask_to_tracking": [[1, 0, 0], [0, 1, 0]],
            }
        )
    rink_context_path(str(tracking)).write_text(json.dumps(context))
    norm = CameraNorm(8.5, 8.5, 22)
    offline = load_rink_grid(str(tracking), norm, 4, 4)
    live = mask_to_grid(union, norm, 4, 4, frame_size=(8, 8))
    np.testing.assert_array_equal(offline, live)
    assert offline[0, -1, -1] < 1


@pytest.mark.parametrize("failure", ["corrupt", "empty", "outside"])
def should_reject_unusable_rink_occupancy(tmp_path, failure):
    tracking, context = _binding(tmp_path)
    path = tmp_path / context["masks"][0]["file"]
    if failure == "corrupt":
        path.write_bytes(b"not a PNG")
    elif failure == "empty":
        assert cv2.imwrite(str(path), np.zeros((20, 40), dtype=np.uint8))
    else:
        context["masks"][0]["mask_to_tracking"] = [[1, 0, 1000], [0, 1, 1000]]
    context["masks"][0]["sha256"] = file_sha256(path)
    rink_context_path(str(tracking)).write_text(json.dumps(context))
    with pytest.raises(ValueError, match="decode|nonempty|no occupancy"):
        load_rink_grid(str(tracking), CameraNorm(200, 100, 22), 8, 16)


def should_encode_rink_once_per_rollout_with_gradients_and_checkpoint_schema():
    torch.set_num_threads(1)
    model = _model()
    calls = []
    handle = model.rink_encoder.register_forward_hook(lambda *args: calls.append(1))
    batch = {
        "base": torch.rand(2, 8, 8),
        "prev0": torch.rand(2, 4),
        "y": torch.rand(2, 8, 4),
        "rink": torch.rand(2, 1, 8, 16),
    }
    rollout = TrainingRollout(model, None, 4)
    output = rollout(batch, 1.0)
    output.square().mean().backward()
    assert len(calls) == 1
    assert sum(float(p.grad.abs().sum()) for p in model.rink_encoder.parameters()) > 0
    model.eval()
    with torch.no_grad():
        _predict_batch(
            model,
            batch,
            torch.device("cpu"),
            free_run=True,
            runtime_slow_aspect_norm=None,
            context_window=4,
        )
    assert len(calls) == 2
    handle.remove()
    checkpoint = pack_gpt_checkpoint(model, CameraNorm(200, 100, 22), 4, model.cfg)
    state, norm, window, cfg = unpack_gpt_checkpoint(checkpoint)
    restored = CameraPanZoomGPT(cfg)
    restored.load_state_dict(state)
    assert cfg.rink_input == "grid" and cfg.rink_grid_width == 16 and window == 4
    with pytest.raises(ValueError, match="requires one rink embedding"):
        restored(torch.rand(2, 4, 12))


def should_cache_live_rink_and_invalidate_geometry_model_and_game():
    model = _model().eval()
    cache = StaticRinkEncoderCache()
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 30:170] = 255
    norm = CameraNorm(200, 100, 22)
    first, changed = cache.get(model, mask, norm, (200, 100), game_id="a", revision="v1")
    assert changed
    same, changed = cache.get(model, mask, norm, (200, 100), game_id="a", revision="v1")
    assert not changed and same is first
    _, changed = cache.get(model, mask, norm, (200, 100), game_id="b", revision="v1")
    assert changed
    with torch.no_grad():
        next(model.rink_encoder.parameters()).add_(0.01)
    updated, changed = cache.get(model, mask, norm, (200, 100), game_id="b", revision="v1")
    assert changed and not torch.equal(updated, first)
    mask[:] = 0
    mask[5:40, 100:180] = 255
    _, changed = cache.get(model, mask, norm, (200, 100), game_id="b", revision="v2")
    assert changed
    with pytest.raises(ValueError, match="dimensions differ"):
        cache.get(model, mask, norm, (100, 50), game_id="b", revision="v2")
    with pytest.raises(ValueError, match="revision"):
        cache.get(model, mask, norm, (200, 100), game_id="b", revision=None)


def should_preserve_legacy_rink_checkpoint_semantics():
    cfg = CameraGPTConfig(
        d_in=19, d_out=4, include_rink=True, d_model=16, nhead=4, nlayers=1, dim_feedforward=32
    )
    model = CameraPanZoomGPT(cfg)
    ckpt = pack_gpt_checkpoint(model, CameraNorm(200, 100, 22), 4, cfg)
    for key in ("rink_input", "rink_grid_height", "rink_grid_width", "rink_encoder_version"):
        ckpt["model"].pop(key)
    state, _, _, loaded_cfg = unpack_gpt_checkpoint(ckpt)
    restored = CameraPanZoomGPT(loaded_cfg)
    restored.load_state_dict(state)
    assert loaded_cfg.rink_input == "stats" and restored.rink_encoder is None
    assert restored(torch.rand(2, 4, 19)).shape == (2, 4, 4)
