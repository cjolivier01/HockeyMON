import argparse

import cv2
import numpy as np
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from hmlib.camera.camera_gpt import CameraPanZoomGPT, unpack_gpt_checkpoint
from hmlib.camera.camera_gpt_dataset import CameraPanZoomGPTIterableDataset, GameCsvPaths
from hmlib.camera.camera_training_config import catalog_split, expand_training_config
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.camera.rink_context import RINK_CONTEXT_SCHEMA, file_sha256, rink_context_path
from hmlib.cli.camgpt_train import TrainingRollout, _maybe_resume, _target_met
from hmlib.cli.drivegpt_dataset import choose_generation, publish_dataset


def _game(directory: Path, suffix: str = "", frames: int = 12, offset: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    part = f"-{suffix}" if suffix else ""
    (directory / f"tracking{part}.csv").write_text(
        "".join(f"{i},1,{i + offset},2,10,12,0.9,0,1,{{}}\n" for i in range(1, frames + 1))
    )
    (directory / f"camera{part}.csv").write_text(
        "".join(f"{i},0,0,64,36\n" for i in range(1, frames + 1))
    )
    (directory / f"camera_fast{part}.csv").write_text(
        "".join(f"{i},4,2,50,30\n" for i in range(1, frames + 1))
    )

    mask_path = directory / "rink_mask_0.png"
    mask = np.zeros((36, 64), dtype=np.uint8)
    mask[4:34, 3:61] = 255
    assert cv2.imwrite(str(mask_path), mask)
    tracking = directory / f"tracking{part}.csv"
    rink_context_path(str(tracking)).write_text(
        json.dumps(
            {
                "schema": RINK_CONTEXT_SCHEMA,
                "coordinate_space": "original_stitched_pixels",
                "frame_size": [64, 36],
                "tracking": {"file": tracking.name, "sha256": file_sha256(tracking)},
                "masks": [
                    {
                        "file": mask_path.name,
                        "sha256": file_sha256(mask_path),
                        "mask_to_tracking": [[1, 0, 0], [0, 1, 0]],
                    }
                ],
                "evidence": "Synthetic fixture generated in the same64x36 tracking canvas.",
            }
        )
    )


def should_select_most_complete_matching_triple(tmp_path):
    _game(tmp_path, "1", frames=12)
    _game(tmp_path, "2", frames=8)
    _game(tmp_path, "3", frames=14)
    (tmp_path / "camera_fast-3.csv").unlink()
    selected, rejected = choose_generation(tmp_path, min_frames=4)
    assert selected["generation"] == "1"
    assert selected["aligned_frames"] == 12
    assert {r["generation"] for r in rejected} == {"2", "3"}


def should_publish_provenance_and_exclude_duplicate_tracking(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "dataset"
    _game(source / "season/a", "1")
    _game(source / "season/b", "1")
    policy = source / "season/a/camera_policy-1.csv"
    policy.write_text(
        '1,"{""schema"":""hm-camera-policy-v1"",""kind"":""startup"",""policy"":{}}"\n'
    )
    catalog = publish_dataset(source, destination, min_frames=4)
    assert len(catalog["games"]) == 2
    assert catalog["games"][1]["duplicate_of"] == "season/a"
    assert (destination / "games/season/a/camera_policy-1.csv").read_bytes() == policy.read_bytes()
    artifact = catalog["games"][0]["files"]["tracking"]
    assert artifact["source"] == str(source / "season/a/tracking-1.csv")
    assert len(artifact["sha256"]) == 64
    config = yaml.safe_load((destination / "dataset.yaml").read_text())
    config["split"] = {"validation_fraction": 0}
    (destination / "dataset.yaml").write_text(yaml.safe_dump(config))
    train, val, identity = catalog_split(str(destination / "dataset.yaml"))
    assert [g.game_id for g in train] == ["season/a"]
    assert not val
    assert identity["catalog_sha256"]
    config["exclude"] = []
    (destination / "dataset.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Duplicate game/tracking identity"):
        catalog_split(str(destination / "dataset.yaml"))
    with pytest.raises(ValueError, match="nonempty"):
        publish_dataset(source, destination, min_frames=4)


def should_validate_yaml_and_allow_cli_overrides(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--game-id", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--pose", dest="include_pose", action=argparse.BooleanOptionalAction)
    path = tmp_path / "train.yaml"
    path.write_text(
        "schema: hockey-drivegpt-training-v1\nmodel:\n  d_model: 256\nfeatures:\n  include_pose: false\n"
    )
    argv = expand_training_config(parser, ["--config", str(path), "--d-model=16", "--pose"])
    args = parser.parse_args(argv)
    assert args.d_model == 16
    assert args.include_pose is True
    path.write_text(
        "schema: hockey-drivegpt-training-v1\nsampling:\n  game_id: [yaml-game]\noptimization:\n  steps: 100\ncheckpoint:\n  no_resume: true\n"
    )
    args = parser.parse_args(
        expand_training_config(
            parser, ["--config", str(path), "--max-iters=7", "--game-id=cli-game", "--resume"]
        )
    )
    assert args.steps == 10 and args.max_iters == 7
    assert args.game_id == ["cli-game"]
    assert args.resume and not args.no_resume
    path.write_text("schema: hockey-drivegpt-training-v1\nmodel:\n  d_modle: 256\n")
    with pytest.raises(ValueError, match="Unknown or repeated"):
        expand_training_config(parser, ["--config", str(path)])


def should_keep_source_game_groups_together_and_verify_artifacts(tmp_path):
    source, dataset = tmp_path / "source", tmp_path / "dataset"
    for i in range(3):
        _game(source / f"game-{i}", offset=i)
    publish_dataset(source, dataset, min_frames=4)
    config = yaml.safe_load((dataset / "dataset.yaml").read_text())
    config["split"] = {
        "validation_games": ["game-1"],
        "groups": {"same-match": ["game-1", "game-2"]},
    }
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(config))
    train, val, _ = catalog_split(str(dataset / "dataset.yaml"))
    assert [g.game_id for g in train] == ["game-0"]
    assert [g.game_id for g in val] == ["game-1", "game-2"]
    with pytest.raises(ValueError, match="no contiguous run"):
        catalog_split(str(dataset / "dataset.yaml"), min_val_frames=16)
    artifact = dataset / "games/game-0/tracking.csv"
    artifact.write_text(artifact.read_text().replace("0.9", "0.8"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        catalog_split(str(dataset / "dataset.yaml"))


def should_use_distinct_rank_sample_streams(tmp_path):
    _game(tmp_path, frames=40)
    paths = GameCsvPaths(
        "a",
        str(tmp_path / "tracking.csv"),
        str(tmp_path / "camera.csv"),
        str(tmp_path / "camera_fast.csv"),
    )
    kwargs = dict(
        games=[paths],
        norm=CameraNorm(64, 36, 22),
        seq_len=4,
        target_mode="slow_fast_tlwh",
        include_pose=False,
        world_size=2,
    )
    samples = [
        next(iter(CameraPanZoomGPTIterableDataset(**kwargs, rank=rank)))["base"]
        for rank in range(2)
    ]
    assert not torch.equal(*samples)


def should_require_both_box_metrics_to_meet_target():
    assert not _target_met({"iou_slow": 0.98, "iou_fast": 0.969}, 0.97)
    assert not _target_met({"iou_slow": 0.969, "iou_fast": 0.99}, 0.97)
    assert _target_met({"iou_slow": 0.97, "iou_fast": 0.97}, 0.97)


@pytest.mark.parametrize("mode", ["legacy", "teacher", "feedback"])
def should_bound_training_context_and_preserve_feedback(mode):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.01))
            self.inputs = []

        def forward(self, features):
            self.inputs.append(features.detach().clone())
            return features[..., -8:] + self.weight

    model = RecordingModel()
    rollout = TrainingRollout(model, aspect=None, context_window=4)
    base = torch.arange(8.0).reshape(1, 8, 1)
    y = torch.full((1, 8, 8), 0.2)
    previous = torch.full((1, 8), 0.1)
    if mode == "legacy":
        batch = {"x": torch.cat([base, y], dim=-1)}
    else:
        batch = {"base": base, "prev0": previous, "y": y}
    output = rollout(batch, probability=float(mode == "feedback"))
    assert output.shape == y.shape
    assert [x.shape[1] for x in model.inputs] == [1, 2, 3, 4, 4, 4, 4, 4]
    assert torch.equal(model.inputs[-1][0, :, 0], torch.arange(4.0, 8.0))
    if mode == "feedback":
        assert torch.allclose(output[0, :, 0], 0.11 + torch.arange(8.0) * 0.01)
        # Prediction feedback is detached: the direct weight gradient is one,
        # independent of rollout length.
        output.mean().backward()
        assert torch.allclose(model.weight.grad, torch.tensor(1.0))
    elif mode == "teacher":
        assert torch.allclose(output[:, 0], previous + 0.01)
        assert torch.allclose(output[:, 1:], y[:, 1:] + 0.01)
    else:
        assert torch.allclose(output, y + 0.01)


@pytest.mark.parametrize("target_mode,output_dim", [("slow_tlwh", 4), ("slow_fast_tlwh", 8)])
def should_train_stop_and_resume_with_two_cpu_ranks(tmp_path, target_mode, output_dim):
    source, dataset = tmp_path / "source", tmp_path / "dataset"
    for i in range(3):
        _game(source / f"game-{i}", offset=i)
    publish_dataset(source, dataset, min_frames=4)
    config = yaml.safe_load((dataset / "dataset.yaml").read_text())
    config["split"] = {"validation_games": ["game-2"]}
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(config))
    output = tmp_path / "run/drivegpt_best.pt"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "hmlib.cli.camgpt_train",
        "--model-kind=drivegpt",
        "--drivegpt-init=none",
        f"--target-mode={target_mode}",
        "--dataset-config",
        str(dataset / "dataset.yaml"),
        "--no-pose",
        "--include-rink" if target_mode == "slow_tlwh" else "--no-include-rink",
        "--rink-input=grid",
        "--rink-grid-height=8",
        "--rink-grid-width=16",
        "--d-model=16",
        "--nhead=4",
        "--nlayers=1",
        "--dim-feedforward=32",
        "--seq-len=4",
        "--rollout-len=8",
        "--val-seq-len=8",
        "--batch-size=2",
        "--val-batch-size=2",
        "--frames=64",
        "--steps=2",
        "--lr=0.00001",
        "--ss-prob-start=0.5",
        "--ss-prob-end=0.5",
        "--val-steps=1",
        "--eval-every=1",
        "--checkpoint-every=1",
        "--target-iou=0.01",
        "--device=cpu",
        "--ddp-backend=gloo",
        "--cpu-threads=1",
        "--data-workers=1",
        "--out",
        str(output),
    ]
    environment = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["window"] == 4
    assert checkpoint["model"]["d_out"] == output_dim
    state = checkpoint["training_state"]
    assert state["target_met"]
    assert state["evaluation"]["world_size"] == 2
    assert state["data_identity"]["validation_games"] == ["game-2"]
    assert not any(k.startswith("module.") for k in checkpoint["state_dict"])
    lines = [
        json.loads(line) for line in output.with_suffix(".metrics.jsonl").read_text().splitlines()
    ]
    assert [line["kind"] for line in lines] == ["run", "train", "validation", "finished"]
    assert lines[0]["step_budget"] == 2
    # A newer best checkpoint must win over an older numbered checkpoint.
    checkpoint["step"] = 5
    torch.save(checkpoint, output)
    _, norm, _, cfg = unpack_gpt_checkpoint(checkpoint)
    model = CameraPanZoomGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    restored = {"data_identity": state["data_identity"], "evaluation": state["evaluation"]}
    step = _maybe_resume(
        best_path=output,
        prefix="drivegpt",
        ext=".pt",
        resume_mode="force",
        device=torch.device("cpu"),
        model=model,
        opt=optimizer,
        cfg=cfg,
        norm=norm,
        training_state=restored,
    )
    assert step == 6
    assert restored["best_val"] == state["best_val"]
    # The step-5 weights have only step-1 validation: resuming must not certify them.
    result = subprocess.run(
        command + ["--resume", "--batch-size=1", "--rollout-len=10"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout
    resumed_lines = [
        json.loads(line) for line in output.with_suffix(".metrics.jsonl").read_text().splitlines()
    ]
    run = next(line for line in reversed(resumed_lines) if line["kind"] == "run")
    assert run["args"]["batch_size"] == 1 and run["args"]["rollout_len"] == 10
    assert run["evaluation"] == state["evaluation"]
    final_line = resumed_lines[-1]
    assert final_line["kind"] == "finished" and not final_line["target_met"]


def should_abort_all_ranks_when_one_host_cannot_build_rink_grid(tmp_path):
    source, dataset = tmp_path / "source", tmp_path / "dataset"
    for i in range(3):
        _game(source / f"game-{i}", offset=i)
    publish_dataset(source, dataset, min_frames=4)
    config_path = dataset / "dataset.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["split"] = {"validation_games": ["game-2"]}
    config_path.write_text(yaml.safe_dump(config))
    runner = tmp_path / "rank_failure.py"
    runner.write_text("""import os
from hmlib.cli import camgpt_train
original = camgpt_train.load_rink_grid
def load(*args, **kwargs):
    if os.environ['RANK'] == '1':
        raise ValueError('rank-local invalid mask')
    return original(*args, **kwargs)
camgpt_train.load_rink_grid = load
camgpt_train.main()
""")
    output = tmp_path / "model.pt"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(runner),
        "--dataset-config",
        str(config_path),
        "--model-kind=drivegpt",
        "--drivegpt-init=none",
        "--no-pose",
        "--include-rink",
        "--rink-input=grid",
        "--target-mode=slow_tlwh",
        "--seq-len=4",
        "--rollout-len=8",
        "--val-seq-len=8",
        "--steps=1",
        "--device=cpu",
        "--ddp-backend=gloo",
        "--cpu-threads=1",
        "--out",
        str(output),
    ]
    environment = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    result = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert (
        "[rank0]: RuntimeError: Dataset geometry preflight failed: rank 1: ValueError: rank-local invalid mask"
        in result.stdout
    )
    assert (
        "[rank1]: RuntimeError: Dataset geometry preflight failed: rank 1: ValueError: rank-local invalid mask"
        in result.stdout
    )
    assert not output.exists()
