"""Prepare and validate reusable model artifacts from actual game frames."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from hmlib.config import ROOT_DIR
from hmlib.ui.workspace.files import read_json, write_json

REID_MODEL = "reidentificationnet-deployable-v1.2"
REID_SHA256 = "0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b"
REID_URL = (
    "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/reidentificationnet/"
    "deployable_v1.2/files?redirect=true&path=resnet50_market1501_aicity156.onnx"
)


def _effective(game: Path) -> dict:
    from hmlib.ui.calibration_workspace import CalibrationWorkspace

    return CalibrationWorkspace(game)._effective()


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _identity(config: dict, target: str) -> str:
    if target == "reid":
        return REID_SHA256[:16]
    run = config.get("hm_ui", {}).get("run", {})
    graph = "tracking_pose" if target == "pose" else "tracking"
    graph_path = Path(ROOT_DIR) / "config" / "aspen" / f"{graph}.yaml"
    payload = {
        "target": target,
        "detector": run.get("detector", "saved"),
        "graph": _digest(graph_path),
        "aspen": config.get("aspen", {}),
        "model": config.get("model", {}),
        "analytics": run.get("analytics", "saved") if target == "pose" else None,
        "analytics_flags": (
            config.get("pipeline", {}).get("player-analytics", {}) if target == "pose" else None
        ),
        "pose_selection": (
            {key: run.get(key) for key in ("pose_model", "pose_config", "pose_checkpoint")}
            if target == "pose"
            else None
        ),
    }
    # A local architecture or weights update must not reuse a previous cache.
    for key in ("detector", "pose"):
        payload[key + "_settings"] = config.get(key, {})
    for path in sorted((Path(ROOT_DIR) / "config" / "aspen" / "models").glob("*.yaml")):
        payload[str(path.name)] = _digest(path)
    if target == "pose":
        from hmlib.ui.analytics_graph import (
            NATIVE_POSE_CHECKPOINT,
            NATIVE_POSE_CONFIG,
            NATIVE_POSE_SHA256,
        )

        payload["native_pose"] = [
            NATIVE_POSE_CHECKPOINT,
            NATIVE_POSE_SHA256,
            _digest(Path(ROOT_DIR) / NATIVE_POSE_CONFIG),
        ]

    def weight_stamps(value):
        if isinstance(value, dict):
            return {key: weight_stamps(item) for key, item in value.items()}
        if isinstance(value, list):
            return [weight_stamps(item) for item in value]
        if isinstance(value, str) and value.endswith((".pth", ".pt", ".onnx", ".tar", ".py")):
            path = Path(value).expanduser()
            if not path.is_absolute() and not path.is_file():
                path = Path(ROOT_DIR) / path
            if path.is_file():
                if path.suffix == ".py":
                    return [str(path.resolve()), _digest(path)]
                stat = path.stat()
                return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
        return None

    payload["weight_stamps"] = weight_stamps(config)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cache_directory(game: Path, target: str, config: dict | None = None) -> Path:
    if target not in {"detector", "pose", "reid"}:
        raise ValueError("Model target must be detector, pose, or reid")
    return game / ".hm-ui" / "models" / target / _identity(config or _effective(game), target)


def model_status(game: Path, target: str, *, verify_digest: bool = False) -> dict:
    directory = cache_directory(game, target)
    manifest = directory / "ready.json"
    status = {
        "target": target,
        "state": "missing",
        "path": "",
        "message": "Prepare this model before selecting it for a run",
        "runtime_available": importlib.util.find_spec("onnxruntime") is not None,
    }
    if not manifest.is_file():
        return status
    receipt = read_json(manifest)
    path = directory / receipt["file"]
    if path.parent != directory or not path.is_file():
        return dict(status, state="invalid", message="Prepared artifact is missing")
    info = path.stat()
    if info.st_size != receipt["size"] or info.st_mtime_ns != receipt["mtime_ns"]:
        return dict(status, state="invalid", message="Prepared artifact changed; prepare it again")
    if verify_digest and _digest(path) != receipt["sha256"]:
        return dict(status, state="invalid", message="Prepared artifact failed its digest check")
    return dict(
        status,
        state="ready",
        path=str(path),
        message="Validated ONNX artifact available",
        frames=receipt.get("frames", 0),
        sha256=receipt["sha256"],
    )


def readiness(game: Path) -> dict:
    return {"models": [model_status(game, target) for target in ("detector", "pose", "reid")]}


def prepared_onnx_arguments(game: Path, target: str) -> list[str]:
    status = model_status(game, target, verify_digest=True)
    if status["state"] != "ready":
        raise ValueError(f"{target.capitalize()} INT8 preparation is required: {status['message']}")
    return [f"--{target}-onnx-enable", f"--{target}-onnx", status["path"]]


def preparation_command(game: Path, target: str, frames: int = 200) -> list[str]:
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 10000:
        raise ValueError("Calibration frame count must be an integer between 1 and 10000")
    cache_directory(game, target)
    return [
        sys.executable,
        "-m",
        "hmlib.ui.model_preparation",
        "--game",
        str(game),
        "--target",
        target,
        "--frames",
        str(frames),
    ]


def _validate(path: Path, target: str) -> None:
    import onnx
    import onnxruntime as ort

    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"{target} preparation did not produce an ONNX artifact: {path}")
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    if target != "reid" and not any(
        node.op_type in {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "ConvInteger"}
        for node in model.graph.node
    ):
        raise RuntimeError(f"{target} preparation produced no INT8 quantization operators")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if target == "reid":
        from hmlib.aspen.plugins.reid_plugin import validate_reid_session

        validate_reid_session(session)


def prepare(game: Path, target: str, frames: int = 200) -> dict:
    preparation_command(game, target, frames)
    config = _effective(game)
    directory = cache_directory(game, target, config)
    directory.mkdir(parents=True, exist_ok=True)
    # Stage every attempt; a failed refresh never replaces a usable artifact.
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as temporary:
        staging = Path(temporary)
        if target == "reid":
            import torch

            artifact = staging / "reid.onnx"
            print(f"Downloading {REID_MODEL} from NVIDIA NGC", flush=True)
            torch.hub.download_url_to_file(REID_URL, str(artifact), hash_prefix=REID_SHA256)
            if _digest(artifact) != REID_SHA256:
                raise ValueError("ReID weights digest does not match the published model")
        else:
            import yaml

            run = config.get("hm_ui", {}).get("run", {})
            graph = "tracking_pose" if target == "pose" else "tracking"
            detector = run.get("detector", "saved")
            overlay = staging / "preparation.yaml"
            overlay.write_text(
                yaml.safe_dump(
                    {
                        "hm_ui": {
                            "run": {
                                "precision": "onnx" if target == "detector" else "pytorch",
                                "pose_precision": "onnx" if target == "pose" else "pytorch",
                            }
                        },
                        "pipeline": {
                            "tracker": {"reid-enable": False},
                            "player-analytics": {
                                "pose": {"rate-hz": "fps", "continuous-drawing": False}
                            },
                        },
                        "team": {"enabled": False},
                    }
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-m",
                "hmlib.cli.hmtrack",
                "--game-id",
                game.name,
                "--config",
                str(Path(ROOT_DIR) / "config" / "aspen" / f"{graph}.yaml"),
                "--config",
                str(overlay),
                "--detector-model",
                "deployed" if detector == "saved" else detector,
                "--camera-ui=0",
                "--no-save-video",
                "--max-frames",
                str(frames),
                f"--{target}-onnx-enable",
                f"--{target}-onnx",
                str(staging / f"{target}.onnx"),
                f"--{target}-onnx-quantize-int8",
                f"--{target}-onnx-calib-frames={frames}",
            ]
            videos = config.get("game", {}).get("videos", {})
            sources = videos.get("source", [])
            if sources:
                if not isinstance(sources, list) or len(sources) != 1:
                    raise ValueError("Concatenate source chapters before model calibration")
                source = (game / sources[0]).resolve(strict=True)
                command += ["--input-video", str(source)]
            environment = dict(os.environ, HM_GAME_DIR=str(game.parent))
            print(f"Calibrating {target} INT8 from {frames} source frames", flush=True)
            subprocess.run(command, cwd=staging, env=environment, check=True)
            artifacts = list(staging.glob(f"{target}*.int8.onnx"))
            if len(artifacts) != 1:
                raise RuntimeError(
                    f"{target} calibration did not produce exactly one INT8 model; "
                    "check the run log, frame count, and availability of detected players"
                )
            artifact = artifacts[0]
        _validate(artifact, target)
        destination = directory / artifact.name
        artifact.replace(destination)
        info = destination.stat()
        write_json(
            directory / "ready.json",
            {
                "schema": 1,
                "target": target,
                "file": destination.name,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "sha256": _digest(destination),
                "frames": frames if target != "reid" else 0,
            },
        )
    result = model_status(game, target, verify_digest=True)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--target", choices=["detector", "pose", "reid"], required=True)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    prepare(args.game.resolve(strict=True), args.target, args.frames)


if __name__ == "__main__":
    main()
