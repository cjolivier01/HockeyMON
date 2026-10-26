"""Prepare HockeyMON INT8 models from the shared native recording sampler."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from hmlib.ui import model_preparation
from hmlib.ui.desktop import executable
from hmlib.ui.hstream_launch import get_path, merge, set_path
from hmlib.ui.workspace.files import identifier, read_json, write_json

PROFILES = {"default": "hockeyref", "distilled-s": "distilled", "hockey-roles": "hockey-roles"}


def _configuration(game: Path, profile: str) -> tuple[dict, Path, tuple[int, int]]:
    from hmlib.config import ROOT_DIR
    from hmlib.models.detector_selection import PROFILE_INPUT_SCALE

    graph = Path(ROOT_DIR) / "config" / "aspen" / "tracking.yaml"
    config = merge(model_preparation._effective(game), yaml.safe_load(graph.read_text()))
    set_path(config, "hm_ui.run.detector", profile)
    set_path(config, "hm_ui.run.precision", "onnx")
    set_path(config, "hm_ui.run.pose_precision", "pytorch")
    set_path(config, "hm_ui.run.analytics", "tracking")
    set_path(config, "hm_ui.run.save_video", False)
    set_path(config, "video_out.show_image", False)
    set_path(config, "video_out.show_scaled", None)
    set_path(config, "pipeline.tracker.reid-enable", False)
    set_path(config, "team.enabled", False)
    for output in (
        "archive_program",
        "archive_program_4k",
        "archive_stitched",
        "show_youtube",
        "rtsp_enabled",
    ):
        set_path(config, "video_out." + output, False)
    for name, plugin in config["aspen"]["plugins"].items():
        # The sample reel already has the native panorama geometry. Only the
        # detector needs to run: masks, camera control, telemetry and output
        # would interpret this reduced image as a full game canvas.
        # Join nodes remain structural even when their upstream work is
        # disabled: replacing one with a no-op makes its fan-in invalid.
        if (
            name not in {"image_prep", "detector_factory", "detector_join", "detector"}
            and plugin["class"] != "hmlib.aspen.plugins.join_plugin.JoinPlugin"
        ):
            plugin["enabled"] = False
    shape = PROFILE_INPUT_SCALE[profile]
    if shape is None:
        shape = next(
            (
                transform["img_scale"]
                for transform in config["aspen"]["inference_pipeline"]
                if transform["type"] == "HmResize"
            ),
            None,
        )
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096
            for value in shape
        )
    ):
        raise ValueError("Detector graph must provide bounded integer input dimensions")
    return config, graph, tuple(shape)


def _native_config(runner: Path) -> Path:
    root = os.environ.get("HM_CONFIG_ROOT")
    candidates = (
        [Path(root) / "ds_hockey_app_config.yaml"]
        if root
        else [
            runner.parent.parent / "configs" / "ds_hockey_app_config.yaml",
            Path(str(runner) + ".runfiles") / "kstream" / "configs" / "ds_hockey_app_config.yaml",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "The matching native application config is unavailable; set HM_CONFIG_ROOT"
    )


def _kill_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return  # The complete preparation group has already exited.


def _run(command: list[str], environment: dict[str, str], *, cwd: Path | None = None) -> None:
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, stdin=subprocess.DEVNULL, start_new_session=True
    )
    completed = False
    try:
        code = process.wait()
        completed = True
        if code:
            raise RuntimeError(f"Detector preparation step failed ({code}): {command[0]}")
    finally:
        if not completed:
            _kill_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _kill_group(process, signal.SIGKILL)
            # A surviving grandchild must not outlive the cancelled dialog.
            _kill_group(process, signal.SIGKILL)
            process.wait()


@contextlib.contextmanager
def _cancellation():
    def cancel(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError("Detector preparation cancelled")

    previous = {signum: signal.signal(signum, cancel) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _samples(directory: Path, count: int, shape: tuple[int, int]) -> tuple[dict, list[str]]:
    from PIL import Image

    def integer(value):
        # The native YAML emitter writes every scalar in samples.json as a
        # quoted JSON string. Accept its decimal integers and ordinary JSON
        # integer values without accepting booleans or lossy float coercion.
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return int(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError("Native sample manifest requires decimal integer values")

    manifest = read_json(directory / "samples.json")
    if (
        integer(manifest.get("schema")) != 1
        or (integer(manifest.get("width")), integer(manifest.get("height"))) != shape
        or not isinstance(manifest.get("samples"), list)
        or len(manifest["samples"]) != count
    ):
        raise ValueError(
            "Native sample manifest does not match the requested detector dimensions/count"
        )
    previous = -1
    digests = []
    for index, sample in enumerate(manifest["samples"]):
        name = sample.get("image")
        timestamp = integer(sample.get("timeline_ns"))
        if name != f"frame-{index:04d}.png" or timestamp <= previous:
            raise ValueError(
                "Native samples require ordered image names and distinct increasing timestamps"
            )
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= 64 * 1024 * 1024
        ):
            raise ValueError("Native sample image is missing or exceeds the preparation limit")
        with Image.open(path) as image:
            if image.size != shape or image.format != "PNG":
                raise ValueError("Native sample image does not match the detector dimensions")
            image.verify()
        digests.append(model_preparation._digest(path))
        previous = timestamp
    return manifest, digests


def validated_detector_arguments(game: Path, config: dict) -> list[str]:
    """Select precisely the prepared model the operator accepted, without export."""
    selection = get_path(config, "hstream_ui.hockeymon.detector", {})
    if not isinstance(selection, dict):
        raise TypeError("Shared detector selection must be a mapping")
    if selection.get("precision") != "int8":
        return []
    model = selection.get("model")
    if model not in PROFILES:
        raise ValueError("Prepared detector selection has an unknown model")
    engine_value, manifest_value = selection.get("engine"), selection.get("manifest")
    if not isinstance(engine_value, str) or not isinstance(manifest_value, str):
        raise TypeError("Prepare this detector before selecting INT8")
    engine, manifest_path = Path(engine_value), Path(manifest_value)
    for path in (engine, manifest_path):
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not path.stat().st_size
        ):
            raise ValueError(f"Prepared detector artifact is unavailable: {path}")
    manifest = read_json(manifest_path)
    info = engine.stat()
    if (
        manifest.get("schema") != 1
        or manifest.get("target") != "detector"
        or manifest.get("model") != model
        or manifest.get("profile") != PROFILES[model]
        or manifest.get("file") != engine.name
        or engine.parent != manifest_path.parent
        or not engine.name.endswith(".int8.onnx")
        or manifest.get("size") != info.st_size
        or manifest.get("mtime_ns") != info.st_mtime_ns
        or manifest.get("sha256") != model_preparation._digest(engine)
    ):
        raise ValueError(
            f"Prepared detector does not match the selected model or changed on disk: {game.name}"
        )
    return ["--detector-onnx-enable", "--detector-onnx", str(engine)]


def prepare_detector(game_id: str, samples: int, detector_model: str) -> dict[str, str]:
    """Publish an immutable validated ONNX selection and the shared ready record."""
    identifier(game_id, "game ID")
    if isinstance(samples, bool) or not isinstance(samples, int) or not 16 <= samples <= 256:
        raise ValueError("Detector calibration samples must be an integer between 16 and 256")
    if detector_model not in PROFILES:
        raise ValueError(f"Unknown shared detector model: {detector_model}")
    root = Path(os.environ.get("HM_GAME_DIR", str(Path.home() / "Videos"))).expanduser().resolve()
    game = root / game_id
    if not (game / "config.yaml").is_file():
        raise FileNotFoundError(f"Game configuration is unavailable: {game}")
    profile = PROFILES[detector_model]
    config, graph, shape = _configuration(game, profile)
    from hmlib.ui.native_tools import native_runner

    runner = native_runner(os.environ.get("HSTREAM_NATIVE_RUNNER"))
    app_config = _native_config(runner)
    ffmpeg = executable(os.environ.get("HM_FFMPEG"), "ffmpeg")
    cache = model_preparation.cache_directory(game, "detector", config)
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")]
        ),
        HM_GAME_DIR=str(root),
        HSTREAM_UI_PARENT_PID=str(os.getpid()),
    )
    for key in (
        "HM_UI_SESSION_DIR",
        "HM_UI_INITIAL_LAUNCH",
        "HM_SHARED_NATIVE_CALIBRATION",
        "HSTREAM_CALIBRATION_PENDING",
        "HSTREAM_CALIBRATION_START_STAGE",
        "HSTREAM_CALIBRATION_INVALIDATION_ID",
        "HSTREAM_RINK_LEVELING_FLOW",
        "HSTREAM_PROJECTION_CROP_FLOW",
    ):
        environment.pop(key, None)
    with _cancellation(), tempfile.TemporaryDirectory(prefix=".preparing-", dir=cache) as temporary:
        staging = Path(temporary)
        captures = staging / "samples"
        print("\nHSTREAM_INT8_PREPARATION stage=sampling", flush=True)
        _run(
            [
                str(runner),
                "-c",
                str(app_config),
                f"--game-id={game_id}",
                "--enable-sources=URI-MULTIPLE",
                "--enable-sinks=FAKE",
                f"--int8-sample-output={captures}",
                f"--int8-sample-count={samples}",
                f"--int8-sample-width={shape[0]}",
                f"--int8-sample-height={shape[1]}",
            ],
            environment,
        )
        capture, hashes = _samples(captures, samples, shape)
        reel = staging / "samples.mp4"
        _run(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-y",
                "-framerate",
                "1",
                "-i",
                str(captures / "frame-%04d.png"),
                "-frames:v",
                str(samples),
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                str(reel),
            ],
            environment,
        )
        if not reel.is_file() or not reel.stat().st_size:
            raise RuntimeError("Could not prepare the private detector sample recording")
        print("\nHSTREAM_INT8_PREPARATION stage=quantizing", flush=True)
        overlay = staging / "preparation.yaml"
        overlay.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        _run(
            [
                sys.executable,
                "-u",
                "-m",
                "hmlib.cli.hmtrack",
                "--game-id",
                game_id,
                "--config",
                str(graph),
                "--config",
                str(overlay),
                "--input-video",
                str(reel),
                "--detector-model",
                profile,
                "--camera-ui=0",
                "--no-save-video",
                "--max-frames",
                str(samples),
                "--detector-onnx-enable",
                "--detector-onnx",
                str(staging / "detector.onnx"),
                "--detector-onnx-quantize-int8",
                f"--detector-onnx-calib-frames={samples}",
            ],
            environment,
            cwd=staging,
        )
        artifacts = list(staging.glob("detector*.int8.onnx"))
        if len(artifacts) != 1:
            raise RuntimeError("Detector preparation did not produce exactly one INT8 ONNX model")
        artifact = artifacts[0]
        model_preparation._validate(artifact, "detector")
        _run([str(runner), "--int8-sample-verify", str(captures)], environment)
        destination = Path(tempfile.mkdtemp(prefix="prepared-", dir=cache))
        final = destination / artifact.name
        artifact.replace(final)
        info = final.stat()
        manifest = destination / "ready.json"
        write_json(
            manifest,
            {
                "schema": 1,
                "target": "detector",
                "model": detector_model,
                "profile": profile,
                "file": final.name,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "sha256": model_preparation._digest(final),
                "frames": samples,
                "capture": capture,
                "image_sha256": hashes,
            },
        )
        result = {"engine": str(final), "manifest": str(manifest)}
        # Progress renderers can leave an ANSI cursor sequence without a line
        # break. The shared dialog requires this receipt at the start of a line.
        print("\nHSTREAM_INT8_READY " + json.dumps(result), flush=True)
        return result
