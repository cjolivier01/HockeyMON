import tempfile
from pathlib import Path

import pytest


def _torch():
    return pytest.importorskip("torch")


def should_sample_gpt_dataset_and_run_model_forward(monkeypatch):
    torch = _torch()

    from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT
    from hmlib.camera.camera_gpt_dataset import CameraPanZoomGPTIterableDataset, GameCsvPaths
    from hmlib.camera.camera_transformer import CameraNorm
    from hmlib.cli.camgpt_train import _validate_validation_scale, main

    with tempfile.TemporaryDirectory(prefix="hm_camgpt_") as td:
        td_path = Path(td)
        tracking_csv = td_path / "tracking.csv"
        camera_csv = td_path / "camera.csv"
        camera_fast_csv = td_path / "camera_fast.csv"
        pose_csv = td_path / "pose.csv"

        # tracking.csv (no header): Frame,ID,BBox_X,BBox_Y,BBox_W,BBox_H,Scores,Labels,Visibility,JerseyInfo
        tracking_lines = []
        for frame in range(1, 21):
            for tid in (1, 2):
                x = 10.0 * tid + frame
                y = 5.0 * tid
                w = 10.0
                h = 20.0
                tracking_lines.append(f"{frame},{tid},{x},{y},{w},{h},0.9,0,1,\n")
        tracking_csv.write_text("".join(tracking_lines))

        # camera.csv (no header): Frame,BBox_X,BBox_Y,BBox_W,BBox_H
        cam_lines = []
        for frame in range(1, 21):
            cam_lines.append(f"{frame},0,0,100,50\n")
        camera_csv.write_text("".join(cam_lines))

        # camera_fast.csv (no header): Frame,BBox_X,BBox_Y,BBox_W,BBox_H
        cam_fast_lines = []
        for frame in range(1, 21):
            cam_fast_lines.append(f"{frame},10,5,80,40\n")
        camera_fast_csv.write_text("".join(cam_fast_lines))

        # pose.csv (no header): Frame,PoseJSON
        pose_lines = []
        pose_json = '{"predictions":[{"bboxes":[[0,0,10,10]],"keypoint_scores":[[0.9,0.8]]}]}'
        pose_json_csv = pose_json.replace('"', '""')
        for frame in range(1, 21):
            pose_lines.append(f'{frame},"{pose_json_csv}"\n')
        pose_csv.write_text("".join(pose_lines))

        paths = GameCsvPaths(
            game_id="test",
            tracking_csv=str(tracking_csv),
            camera_csv=str(camera_csv),
            camera_fast_csv=str(camera_fast_csv),
            pose_csv=str(pose_csv),
        )
        norm = CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22)
        ds = CameraPanZoomGPTIterableDataset(
            games=[paths],
            norm=norm,
            seq_len=8,
            target_mode="slow_fast_tlwh",
            feature_mode="base_prev_y",
            include_pose=True,
            seed=123,
        )
        sample = next(iter(ds))
        base = sample["base"]
        prev0 = sample["prev0"]
        y = sample["y"]
        assert base.shape == (8, 16)
        assert prev0.shape == (8,)
        assert y.shape == (8, 8)

        # Teacher forcing: prev_y[t] = prev0 for t=0 else y[t-1]
        prev_y = torch.cat([prev0.unsqueeze(0), y[:-1]], dim=0)
        x = torch.cat([base, prev_y], dim=-1)
        assert x.shape == (8, 24)

        cfg = CameraGPTConfig(
            d_in=24,
            d_out=8,
            feature_mode="base_prev_y",
            include_pose=True,
            d_model=32,
            nhead=4,
            nlayers=2,
            dropout=0.1,
        )
        model = CameraPanZoomGPT(cfg)
        out = model(x.unsqueeze(0))
        assert out.shape == (1, 8, 8)
        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

        ds_players = CameraPanZoomGPTIterableDataset(
            games=[paths],
            norm=norm,
            seq_len=8,
            target_mode="slow_fast_tlwh",
            feature_mode="players_prev_y",
            include_pose=True,
            seed=123,
        )
        sample_players = next(iter(ds_players))
        assert sample_players["base"].shape == (8, 104)
        assert sample_players["prev0"].shape == (8,)
        assert sample_players["y"].shape == (8, 8)

        ds_cold_start = CameraPanZoomGPTIterableDataset(
            games=[paths],
            norm=norm,
            seq_len=20,
            target_mode="slow_fast_tlwh",
            feature_mode="base_prev_y",
            include_pose=False,
            seed=123,
        )
        sample_cold_start = next(iter(ds_cold_start))
        assert torch.equal(
            sample_cold_start["prev0"],
            torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        )

        with pytest.raises(SystemExit, match="Validation extents exceed"):
            _validate_validation_scale(
                [paths],
                CameraNorm(scale_x=50.0, scale_y=25.0, max_players=22),
            )

        game_list = td_path / "games.lst"
        game_list.write_text(f"{td_path}\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "camgpt_train",
                "--model-kind=gpt",
                f"--file-list={game_list}",
                "--seq-len=8",
                "--target-mode=slow_fast_tlwh",
                "--val-split=0",
                "--target-iou=0.9",
                "--steps=1",
            ],
        )
        with pytest.raises(SystemExit, match="--target-iou requires validation"):
            main()


def should_keep_drivegpt_windows_inside_contiguous_frame_runs():
    torch = _torch()

    from hmlib.camera.camera_gpt_dataset import CameraPanZoomGPTIterableDataset, GameCsvPaths
    from hmlib.camera.camera_transformer import CameraNorm

    with tempfile.TemporaryDirectory(prefix="hm_camgpt_gaps_") as td:
        td_path = Path(td)
        tracking_csv = td_path / "tracking.csv"
        camera_csv = td_path / "camera.csv"
        frames = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
        tracking_csv.write_text("".join(f"{frame},1,10,5,10,20,0.9,0,1,\n" for frame in frames))
        camera_csv.write_text("".join(f"{frame},{frame},0,10,5\n" for frame in frames))

        dataset = CameraPanZoomGPTIterableDataset(
            games=[
                GameCsvPaths(
                    game_id="gapped",
                    tracking_csv=str(tracking_csv),
                    camera_csv=str(camera_csv),
                )
            ],
            norm=CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22),
            seq_len=2,
            target_mode="slow_tlwh",
            feature_mode="base_prev_y",
            include_pose=False,
            seed=7,
        )
        iterator = iter(dataset)
        saw_run_start = False
        saw_mid_run = False
        for _ in range(100):
            sample = next(iterator)
            frame_ids = torch.round(sample["y"][:, 0] * 100).to(dtype=torch.int64)
            assert torch.equal(frame_ids[1:] - frame_ids[:-1], torch.ones(1, dtype=torch.int64))

            first_frame = int(frame_ids[0].item())
            previous_frame = int(round(float(sample["prev0"][0].item()) * 100))
            if first_frame in {1, 6, 11}:
                saw_run_start = True
                assert torch.equal(sample["prev0"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
            else:
                saw_mid_run = True
                assert previous_frame == first_frame - 1

        assert saw_run_start
        assert saw_mid_run

        unusable = CameraPanZoomGPTIterableDataset(
            games=dataset._games,
            norm=dataset.norm,
            seq_len=5,
            target_mode="slow_tlwh",
            feature_mode="base_prev_y",
            include_pose=False,
        )
        with pytest.raises(RuntimeError, match="longest contiguous run has 4 frames"):
            next(iter(unusable))


def should_resolve_only_complete_matching_csv_generations():
    from hmlib.camera.camera_gpt_dataset import resolve_csv_paths

    with tempfile.TemporaryDirectory(prefix="hm_camgpt_generation_") as td:
        td_path = Path(td)
        for filename in (
            "tracking.csv",
            "camera.csv",
            "tracking-1.csv",
            "camera-1.csv",
            "camera_fast-1.csv",
            "camera-2.csv",
            "camera_fast-2.csv",
            "tracking-3.csv",
        ):
            (td_path / filename).write_text("1\n")

        paths = resolve_csv_paths("generation", td)
        assert paths is not None
        assert Path(paths.tracking_csv).name == "tracking-1.csv"
        assert Path(paths.camera_csv).name == "camera-1.csv"
        assert paths.camera_fast_csv is not None
        assert Path(paths.camera_fast_csv).name == "camera_fast-1.csv"


def should_ignore_pending_hstream_generation_after_tracking_commit_failure():
    import json

    from hmlib.camera.camera_gpt_dataset import resolve_csv_paths

    with tempfile.TemporaryDirectory(prefix="hm_camgpt_publication_failure_") as td:
        td_path = Path(td)
        for suffix in ("-1", "-2"):
            for stem in ("tracking", "camera", "camera_fast"):
                (td_path / f"{stem}{suffix}.csv").write_text("1\n")

        def write_manifest(suffix: str, *, committed: bool) -> None:
            suffix_part = f"-{suffix}"
            (td_path / f"hstream_telemetry{suffix_part}.json").write_text(
                json.dumps(
                    {
                        "schema": "hstream-playtracker-telemetry-v1",
                        "publication_state": "committed" if committed else "pending",
                        "completed": committed,
                        "eligible_for_training": committed,
                        "hm_compatibility": {
                            "tracking_csv": {"file": f"tracking{suffix_part}.csv"},
                            "camera_csv": {"file": f"camera{suffix_part}.csv"},
                            "camera_fast_csv": {"file": f"camera_fast{suffix_part}.csv"},
                        },
                    }
                )
            )

        write_manifest("1", committed=True)
        # Mirrors hstream's injected tracking-directory-fsync failure: all CSV
        # links may be visible, but the durable manifest remains pending.
        write_manifest("2", committed=False)

        paths = resolve_csv_paths("publication-failure", td)
        assert paths is not None
        assert Path(paths.tracking_csv).name == "tracking-1.csv"
        assert Path(paths.camera_csv).name == "camera-1.csv"
        assert paths.camera_fast_csv is not None
        assert Path(paths.camera_fast_csv).name == "camera_fast-1.csv"


@pytest.mark.parametrize("manifest", ["[]", "null", '"invalid"', "42"])
def should_ignore_non_object_hstream_manifest(tmp_path, manifest):
    from hmlib.camera.camera_gpt_dataset import resolve_csv_paths

    for stem in ("tracking", "camera", "camera_fast"):
        (tmp_path / f"{stem}.csv").write_text("1\n")
    (tmp_path / "hstream_telemetry.json").write_text(manifest)

    assert resolve_csv_paths("malformed-manifest", str(tmp_path)) is None


def should_free_run_legacy_prev_slow_by_feeding_predictions():
    torch = _torch()

    from hmlib.cli.camgpt_train import (
        _compute_losses,
        _predict_batch,
        _runtime_aspect_slow_tlwh,
        _runtime_feedback_target,
    )

    class AddToPrev(torch.nn.Module):
        def forward(self, x):
            out = x[..., 8:11].clone()
            out[..., 0] = torch.clamp(out[..., 0] + 0.1, 0.0, 1.0)
            return out

    model = AddToPrev()
    x = torch.zeros((1, 3, 11), dtype=torch.float32)
    x[:, :, 8:11] = torch.tensor([0.2, 0.3, 0.4])
    # Ground-truth previous state for later frames differs from the model feedback.
    x[:, 1:, 8:11] = torch.tensor([0.8, 0.8, 0.8])
    batch = {"x": x, "y": torch.zeros((1, 3, 3), dtype=torch.float32)}

    teacher_pred, _ = _predict_batch(
        model,
        batch,
        torch.device("cpu"),
        free_run=False,
        runtime_slow_aspect_norm=None,
    )
    free_pred, _ = _predict_batch(
        model,
        batch,
        torch.device("cpu"),
        free_run=True,
        runtime_slow_aspect_norm=None,
    )

    assert torch.allclose(teacher_pred[0, :, 0], torch.tensor([0.3, 0.9, 0.9]))
    assert torch.allclose(free_pred[0, :, 0], torch.tensor([0.3, 0.4, 0.5]))

    fitted = _runtime_aspect_slow_tlwh(
        torch.tensor([[[0.0, 0.0, 0.2, 1.0]]], dtype=torch.float32),
        aspect_norm=2.0,
    )
    assert torch.allclose(fitted[0, 0], torch.tensor([0.0, 0.25, 1.0, 0.5]))
    feedback = _runtime_feedback_target(
        torch.tensor([[[0.5, 0.5, 1.0]]], dtype=torch.float32),
        runtime_slow_aspect_norm=2.0,
    )
    assert torch.allclose(feedback[0, 0], torch.tensor([0.5, 0.5, 0.5]))

    _, metrics = _compute_losses(
        pred=torch.tensor([[[0.5, 0.5, 0.5]]], dtype=torch.float32),
        y=torch.tensor([[[0.5, 0.5, 0.5]]], dtype=torch.float32),
        w_l1=1.0,
        w_iou=1.0,
        w_vel=0.0,
        w_acc=0.0,
        fast_mult=1.0,
        runtime_slow_aspect_norm=0.5,
    )
    assert metrics["iou"] == pytest.approx(1.0)

    _, fast_metrics = _compute_losses(
        pred=torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.9, 0.9, 0.5, 0.5]]],
            dtype=torch.float32,
        ),
        y=torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.9, 0.9, 0.1, 0.1]]],
            dtype=torch.float32,
        ),
        w_l1=1.0,
        w_iou=1.0,
        w_vel=0.0,
        w_acc=0.0,
        fast_mult=1.0,
        runtime_slow_aspect_norm=None,
    )
    assert fast_metrics["iou_fast"] == pytest.approx(1.0)


def should_pack_drivegpt_metadata_and_init_from_opendrive_planning():
    torch = _torch()

    from hmlib.camera.camera_gpt import (
        CameraGPTConfig,
        CameraPanZoomGPT,
        init_from_opendrive_uniad_planning,
        pack_gpt_checkpoint,
        unpack_gpt_checkpoint,
    )
    from hmlib.camera.camera_transformer import CameraNorm

    cfg = CameraGPTConfig(
        d_in=4,
        d_out=8,
        model_kind="drivegpt",
        feature_mode="base_prev_y",
        include_pose=False,
        include_rink=True,
        source_model_id="OpenDriveLab/UniAD2.0_R101_nuScenes",
        source_checkpoint="/tmp/fake_uniad.pth",
        source_init="opendrive-uniad-planning",
        residual_prev_y=True,
        residual_scale=0.05,
        d_model=256,
        nhead=8,
        nlayers=3,
        dim_feedforward=512,
        dropout=0.0,
    )
    model = CameraPanZoomGPT(cfg)
    target_sd = model.state_dict()
    source_sd = {}
    mapping = {
        "self_attn.in_proj_weight": "self_attn.in_proj_weight",
        "self_attn.in_proj_bias": "self_attn.in_proj_bias",
        "self_attn.out_proj.weight": "self_attn.out_proj.weight",
        "self_attn.out_proj.bias": "self_attn.out_proj.bias",
        "multihead_attn.in_proj_weight": "multihead_attn.in_proj_weight",
        "multihead_attn.in_proj_bias": "multihead_attn.in_proj_bias",
        "multihead_attn.out_proj.weight": "multihead_attn.out_proj.weight",
        "multihead_attn.out_proj.bias": "multihead_attn.out_proj.bias",
        "linear1.weight": "linear1.weight",
        "linear1.bias": "linear1.bias",
        "linear2.weight": "linear2.weight",
        "linear2.bias": "linear2.bias",
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "norm3.weight": "norm3.weight",
        "norm3.bias": "norm3.bias",
    }
    for layer_idx in range(3):
        for idx, (source_suffix, target_suffix) in enumerate(mapping.items(), start=1):
            target_key = f"decoder.layers.{layer_idx}.{target_suffix}"
            source_key = f"planning_head.attn_module.layers.{layer_idx}.{source_suffix}"
            source_sd[source_key] = torch.full_like(
                target_sd[target_key], float((layer_idx * len(mapping)) + idx) / 100.0
            )

    with tempfile.TemporaryDirectory(prefix="hm_drivegpt_") as td:
        ckpt_path = Path(td) / "uniad_fake.pth"
        torch.save({"state_dict": source_sd}, ckpt_path)
        before = model.state_dict()["decoder.layers.0.linear1.bias"].clone()
        report = init_from_opendrive_uniad_planning(model, str(ckpt_path))

    after = model.state_dict()["decoder.layers.0.linear1.bias"]
    assert report["source"] == "opendrive-uniad-planning"
    assert report["copied_tensors"] == len(mapping) * 3
    assert report["initialized_layers"] == 3
    assert not torch.equal(before, after)

    packed = pack_gpt_checkpoint(
        model, norm=CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22), window=8, cfg=cfg
    )
    _, norm, window, unpacked = unpack_gpt_checkpoint(packed)
    assert norm.scale_x == 100.0
    assert window == 8
    assert unpacked.model_kind == "drivegpt"
    assert unpacked.source_model_id == "OpenDriveLab/UniAD2.0_R101_nuScenes"
    assert unpacked.source_init == "opendrive-uniad-planning"
    assert unpacked.residual_prev_y is True
    assert unpacked.residual_scale == 0.05
    assert unpacked.dim_feedforward == 512

    bad_cfg = CameraGPTConfig(
        d_in=4,
        d_out=8,
        model_kind="drivegpt",
        d_model=256,
        nhead=4,
        nlayers=3,
        dim_feedforward=512,
    )
    with tempfile.TemporaryDirectory(prefix="hm_drivegpt_bad_") as td:
        ckpt_path = Path(td) / "uniad_fake.pth"
        torch.save({"state_dict": source_sd}, ckpt_path)
        with pytest.raises(ValueError, match="requires d_model=256, nhead=8"):
            init_from_opendrive_uniad_planning(CameraPanZoomGPT(bad_cfg), str(ckpt_path))

    partial_cfg = CameraGPTConfig(
        d_in=4,
        d_out=8,
        model_kind="drivegpt",
        d_model=256,
        nhead=8,
        nlayers=1,
        dim_feedforward=512,
    )
    with tempfile.TemporaryDirectory(prefix="hm_drivegpt_partial_") as td:
        ckpt_path = Path(td) / "uniad_fake.pth"
        torch.save({"state_dict": source_sd}, ckpt_path)
        with pytest.raises(ValueError, match="requires nlayers=3"):
            init_from_opendrive_uniad_planning(CameraPanZoomGPT(partial_cfg), str(ckpt_path))


def should_load_drivegpt_controller_as_gpt_backend(monkeypatch):
    torch = _torch()

    import sys
    import types

    class FakeClusterMan:
        def __init__(self, *args, **kwargs):
            pass

    fake_clusters = types.ModuleType("hmlib.camera.clusters")
    fake_clusters.ClusterMan = FakeClusterMan
    monkeypatch.setitem(sys.modules, "hmlib.camera.clusters", fake_clusters)

    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin
    from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, pack_gpt_checkpoint
    from hmlib.camera.camera_transformer import CameraNorm
    from hmlib.utils.gpu import unwrap_tensor

    cfg = CameraGPTConfig(
        d_in=4,
        d_out=4,
        model_kind="drivegpt",
        feature_mode="base_prev_y",
        d_model=32,
        nhead=4,
        nlayers=1,
    )
    model = CameraPanZoomGPT(cfg)
    ckpt = pack_gpt_checkpoint(
        model, norm=CameraNorm(scale_x=400.0, scale_y=400.0, max_players=22), window=8, cfg=cfg
    )

    monkeypatch.setattr(torch, "load", lambda path, map_location=None: ckpt)

    plugin = CameraControllerPlugin(controller="drivegpt", model_path="/tmp/fake_drivegpt.pt")

    assert plugin._controller == "gpt"
    assert plugin._gpt_model is not None
    assert plugin._gpt_cfg is not None
    assert plugin._gpt_cfg.model_kind == "drivegpt"
    pose_feat = plugin._pose_features(
        [
            {
                "predictions": [
                    {
                        "bboxes": [[0.0, 0.0, 10.0, 10.0]],
                        "keypoint_scores": [[0.9, 0.8]],
                    }
                ]
            }
        ],
        0,
        torch.device("cpu"),
    )
    assert pose_feat.shape == (8,)
    assert torch.count_nonzero(pose_feat).item() > 0
    default_h = 400.0 * 0.8
    default_w = default_h * (16.0 / 9.0)
    fitted = plugin._fit_box_inside_bounds(
        torch.tensor([200.0 - default_w * 0.5, 40.0, 200.0 + default_w * 0.5, 360.0]),
        torch.tensor([100.0, 100.0, 300.0, 300.0]),
    )
    assert torch.allclose(fitted, torch.tensor([100.0, 143.75, 300.0, 256.25]), atol=1e-3)

    class Sample:
        metainfo = {"ori_shape": (400, 400)}

    out = plugin.forward(
        {
            "data_samples": [[Sample()]],
            "arena": torch.tensor([100.0, 100.0, 300.0, 300.0]),
            "shared": {"device": torch.device("cpu")},
        }
    )
    emitted = unwrap_tensor(out["camera_boxes"])[0]
    assert torch.allclose(emitted, fitted, atol=1e-3)
    assert plugin._prev_y is not None
    assert torch.allclose(
        plugin._prev_y,
        torch.tensor([0.25, 0.3592, 0.5, 0.2817]),
        atol=1e-3,
    )

    legacy_cfg = CameraGPTConfig(d_in=4, d_out=4, model_kind="gpt", d_model=32, nhead=4, nlayers=1)
    legacy_model = CameraPanZoomGPT(legacy_cfg)
    legacy_ckpt = pack_gpt_checkpoint(
        legacy_model,
        norm=CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22),
        window=8,
        cfg=legacy_cfg,
    )
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: legacy_ckpt)
    with pytest.raises(RuntimeError, match="model_kind='drivegpt'"):
        CameraControllerPlugin(controller="drivegpt", model_path="/tmp/fake_legacy_gpt.pt")
    with pytest.raises(ValueError, match="requires model_path"):
        CameraControllerPlugin(controller="drivegpt", model_path=None)


def should_reject_drivegpt_play_tracker_without_model():
    from hmlib.camera.camera_controller_modes import normalize_play_tracker_camera_controller

    assert normalize_play_tracker_camera_controller("drivegpt", "/tmp/model.pt") == "gpt"
    with pytest.raises(RuntimeError, match="requires rink.camera.camera_model"):
        normalize_play_tracker_camera_controller("drivegpt", None)
