#!/usr/bin/env python3
"""Capture an isolated, resumable matrix of HM stitching projection previews."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from hmlib.stitching.artifact_validation import bounded_file, validate_artifact_generation
from hmlib.stitching.artifacts import stitching_lock
from hmlib.stitching.configure_stitching import _file_provenance, _image_content_provenance
from hmlib.stitching.projections import read_panorama_geometry
from hmlib.stitching.seam import read_png_layout
from hmlib.stitching.settings import PROJECTIONS, read_stitching_settings

MARKER = ".hm-projection-capture"
MARKER_CONTENT = "HM projection capture v1\n"


def read_yaml(path: Path) -> dict[str, Any]:
    bounded_file(path, 4 * 1024**2)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def atomic_json(path: Path, data: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def expand_cases(matrix: dict[str, Any], source_config: dict[str, Any] | None = None) -> list[dict]:
    """Use HM's effective settings parser for all projection/FOV validation."""
    if matrix.get("version") != 1 or not isinstance(matrix.get("projections"), list):
        raise ValueError("Matrix requires version: 1 and a projections list")
    defaults = matrix.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    cases, labels = [], set()
    for entry in matrix["projections"]:
        if not isinstance(entry, dict) or entry.get("name") not in PROJECTIONS:
            raise ValueError("Each projection must have a supported canonical name")
        projection = entry["name"]
        if not isinstance(entry.get("variants"), list) or not entry["variants"]:
            raise ValueError(f"{projection} requires a nonempty variants list")
        for index, variant in enumerate(entry["variants"]):
            if not isinstance(variant, dict):
                raise ValueError(f"{projection} variants must be mappings")
            values = {**defaults, **variant}
            label = values.get("label", f"case-{index + 1}")
            if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
                raise ValueError(
                    "Case labels must contain only letters, digits, dots, underscores or dashes"
                )
            label = f"{projection}--{label}"
            if label in labels:
                raise ValueError(f"Duplicate case: {label}")
            labels.add(label)
            effective = copy.deepcopy(source_config or {})
            stitch = effective.get("stitching")
            if stitch is None:
                stitch = {}
                effective["stitching"] = stitch
            if not isinstance(stitch, dict):
                raise ValueError("source stitching configuration must be a mapping")
            backend = values.get("mapping_backend", "nona")
            width = values.get("max_output_width", 4096)
            if not isinstance(width, bool) and isinstance(width, (int, float)) and width == 0:
                width = None
            if "camera_config" in values:
                stitch["camera_config"] = values["camera_config"]
                # A matrix preset replaces inherited explicit FOV overrides.
                stitch.pop("camera_fov", None)
            fov = stitch.setdefault("camera_fov", {})
            if not isinstance(fov, dict):
                raise ValueError("source camera_fov must be a mapping")
            for axis in ("horizontal", "vertical"):
                if f"camera_{axis}_fov" in values:
                    fov[f"{axis}_fov"] = values[f"camera_{axis}_fov"]
            stitch.update(
                {
                    "projection": projection,
                    "projection_parameters": {projection: values.get("parameters", [])},
                    "mapping_backend": backend,
                    "run_autooptimizer": backend == "nona",
                    "max_control_points": values.get("control_points", 900),
                    "calibration_frame_count": values.get("frame_count", 4),
                    "max_output_width": width,
                    "stitch_frame_time": str(values.get("stitch_frame_time", "00:00:00")),
                    "projection_framing": {
                        "auto_fov": values.get("auto_fov", False),
                        "horizontal_fov": values.get("horizontal_fov", 180),
                        "auto_canvas": values.get("auto_canvas", True),
                        "auto_crop": values.get("auto_crop", False),
                        "rotation_degrees": values.get("rotation_degrees", [0, 0, 0]),
                        "crop": values.get("crop", [0, 1, 0, 1]),
                    },
                }
            )
            if "control_point_matcher" in values:
                stitch["control_point_matcher"] = values["control_point_matcher"]
            settings = read_stitching_settings(effective)
            # Save resolved camera values as well as the requested projection framing.
            stitch["control_point_matcher"] = settings.control_point_matcher
            stitch["camera_config"] = settings.camera_config
            stitch["camera_fov"] = {
                "horizontal_fov": settings.horizontal_fov,
                "vertical_fov": settings.vertical_fov,
            }
            cases.append(
                {"label": label, "effective_config": effective, "settings": settings.manifest()}
            )
    if not cases:
        raise ValueError("The matrix does not contain any cases")
    return cases


def source_inputs(source: Path, config: dict, mode: str) -> list[str]:
    if mode == "images":
        images = [str(source / "left.png"), str(source / "right.png")]
        _image_content_provenance(images)
        return images
    videos = config.get("game", {}).get("videos", {})
    result = []
    for side in ("left", "right"):
        entries = videos.get(side)
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], str):
            raise ValueError(
                f"Video capture requires one game.videos.{side} input; combine split recordings first"
            )
        path = Path(entries[0]).expanduser()
        path = path if path.is_absolute() else source / path
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(str(path.resolve()))
    return result


def source_identity(
    source: Path, inputs: list[str], mode: str, reuse_points: bool
) -> dict[str, str]:
    identity = {
        "inputs": (
            _image_content_provenance(inputs) if mode == "images" else _file_provenance(inputs)
        )
    }
    for name in (
        "config.yaml",
        "left_calibration.json",
        *(["hm_project.pto"] if reuse_points else []),
    ):
        path = source / name
        if path.exists():
            bounded_file(path, 32 * 1024**2)
            identity[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif name == "hm_project.pto":
            raise FileNotFoundError(path)
    return identity


def validate_capture(case_dir: Path) -> dict[str, str]:
    game = case_dir / "game"
    width, height = validate_artifact_generation(game, project_name="hm_project.pto")
    read_panorama_geometry(game / "autooptimiser_out.pto")
    preview = read_png_layout(game / "s.png")
    if (preview.width, preview.height) != (width, height):
        raise ValueError("Captured preview dimensions do not match the mapping canvas")
    bounded_file(game / ".stitching_artifacts.json", 1024**2)
    evidence = {}
    for path in sorted(game.iterdir()):
        if path.suffix not in (".png", ".pto", ".tif") and path.name != ".stitching_artifacts.json":
            continue
        bounded_file(path, 2 * 1024**3)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        evidence[path.name] = digest.hexdigest()
    return evidence


def owns_directory(path: Path) -> bool:
    marker = path / MARKER
    return (
        not path.is_symlink()
        and path.is_dir()
        and not marker.is_symlink()
        and marker.is_file()
        and marker.stat().st_size == len(MARKER_CONTENT)
        and marker.read_text() == MARKER_CONTENT
    )


def run_subprocess(plan_path: Path, log_path: Path, timeout: float) -> int:
    """Terminate the whole calibration process group when a case times out."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["HM_GAME_DIR"] = str(plan_path.parent)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).absolute()), "--worker", str(plan_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=plan_path.parent,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        finally:
            # Always address the group: its leader can exit while children live.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    log.write(f"Process group {process.pid} already exited during cleanup\n")
                if sig == signal.SIGTERM:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        log.write("Process did not stop after SIGTERM; sending SIGKILL\n")
            process.wait(timeout=5)


def run_worker(plan_path: Path) -> int:
    from hmlib.stitching.configure_stitching import (
        build_stitching_project,
        configure_video_stitching,
    )
    from hmlib.stitching.hugin import load_pto_file, parse_hugin_control_points
    from hmlib.stitching.control_points import select_evenly_spaced
    from hmlib.stitching.synchronize import synchronize_by_audio
    from hmlib.video.ffmpeg import BasicVideoInfo
    from hmlib.video.video_stream import time_to_frame

    bounded_file(plan_path, 8 * 1024**2)
    plan = json.loads(plan_path.read_text())
    config, inputs = plan["effective_config"], plan["inputs"]
    settings = read_stitching_settings(config)
    game = plan_path.parent / "game"
    game.mkdir()
    (game / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    source = Path(plan["source_game_dir"])
    profile = source / "left_calibration.json"
    if profile.exists():
        bounded_file(profile, 1024**2)
        shutil.copyfile(profile, game / profile.name)
    if plan["source_mode"] == "images":
        points = None
        if plan["reuse_control_points"]:
            project = source / "hm_project.pto"
            bounded_file(project, 32 * 1024**2)
            lines = load_pto_file(str(project))
            for line in lines:
                if line.startswith("c ") and not all(
                    token in line.split() for token in ("n0", "N1", "t0")
                ):
                    raise ValueError(
                        "Reusable control points must be ordinary image-0/image-1 correspondences"
                    )
            parsed = parse_hugin_control_points(lines)
            if parsed is None or len(parsed[0]) < 4:
                raise ValueError("Source PTO does not contain at least four reusable points")
            selected = select_evenly_spaced(parsed[0], settings.max_control_points)
            points = {"m_kpts0": parsed[0][selected], "m_kpts1": parsed[1][selected]}
        build_stitching_project(
            str(game / "hm_project.pto"),
            inputs,
            settings.max_control_points,
            settings=settings,
            force=True,
            control_points=points,
            game_config=config,
        )
    else:
        offsets = config.get("stitching", {}).get("frame_offsets")
        if (
            isinstance(offsets, dict)
            and offsets.get("left") is not None
            and offsets.get("right") is not None
        ):
            left_offset, right_offset = float(offsets["left"]), float(offsets["right"])
        else:
            left_offset, right_offset = synchronize_by_audio(inputs[0], inputs[1])
        timestamp = config["stitching"]["stitch_frame_time"]
        base_frame = time_to_frame(timestamp, BasicVideoInfo(inputs[0]).fps)
        configure_video_stitching(
            str(game),
            *inputs,
            settings.max_control_points,
            left_frame_offset=left_offset,
            right_frame_offset=right_offset,
            base_frame_offset=base_frame,
            stitch_frame_time=timestamp,
            settings=settings,
            game_config=config,
            force=True,
            ignore_private_config=True,
        )
    validate_capture(plan_path.parent)
    return 0


def prepare_output(output: Path, source: Path) -> None:
    if (
        output.is_symlink()
        or source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError(
            "Capture output must be separate from the source game and cannot be a symlink"
        )
    if output.exists():
        if not owns_directory(output):
            raise ValueError(f"Refusing unrecognized capture output directory: {output}")
    else:
        output.mkdir(parents=True)
        (output / MARKER).write_text(MARKER_CONTENT)


def capture(
    matrix: dict,
    cases: list[dict],
    source: Path,
    output: Path,
    *,
    mode: str,
    reuse_points: bool = False,
    start_at: int = 1,
    limit: int | None = None,
    force: bool = False,
    runner=run_subprocess,
) -> int:
    if mode not in ("images", "videos") or not isinstance(reuse_points, bool):
        raise ValueError("source_mode must be videos or images and reuse_control_points a boolean")
    if start_at < 1 or (limit is not None and limit < 1):
        raise ValueError("start_at and limit must be positive")
    timeout = matrix.get("pipeline", {}).get("timeout_seconds", 360)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("pipeline.timeout_seconds must be positive and finite")
    if reuse_points and mode != "images":
        raise ValueError("reuse_control_points requires image mode")
    config = read_yaml(source / "config.yaml")
    inputs = source_inputs(source, config, mode)
    identity = source_identity(source, inputs, mode, reuse_points)
    prepare_output(output, source)
    selected = list(enumerate(cases, 1))[start_at - 1 :]
    selected = selected if limit is None else selected[:limit]
    if not selected:
        raise ValueError("No projection cases selected")
    failures = 0
    with stitching_lock(output, blocking=False):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            bounded_file(manifest_path, 8 * 1024**2)
            rows = json.loads(manifest_path.read_text())
            if not isinstance(rows, dict):
                raise ValueError("Capture manifest must be a mapping")
        else:
            rows = {}
        for sequence, case in selected:
            name = f"{sequence:03d}__{case['label']}"
            case_dir = output / name
            plan = {
                **case,
                "inputs": inputs,
                "source_identity": identity,
                "source_game_dir": str(source),
                "source_mode": mode,
                "reuse_control_points": reuse_points,
            }
            signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            previous = rows.get(name, {})
            if (
                not force
                and previous.get("outcome") == "pass"
                and previous.get("signature") == signature
            ):
                try:
                    if not owns_directory(case_dir):
                        raise ValueError("Unrecognized case directory")
                    bounded_file(case_dir / "plan.json", 8 * 1024**2)
                    if json.loads((case_dir / "plan.json").read_text()) != plan:
                        raise ValueError("Saved plan differs from the requested case")
                    if read_yaml(case_dir / "effective-config.yaml") != case["effective_config"]:
                        raise ValueError("Saved configuration differs from the requested case")
                    if validate_capture(case_dir) != previous.get("evidence"):
                        raise ValueError("Captured artifacts changed since completion")
                    print(f"SKIP {name}", flush=True)
                    continue
                except (OSError, ValueError) as error:
                    print(f"RETRY {name}: invalid saved capture: {error}", flush=True)
            rows[name] = {"outcome": "in_progress", "signature": signature}
            atomic_json(manifest_path, rows)
            started = time.monotonic()
            try:
                if case_dir.exists() or case_dir.is_symlink():
                    if not owns_directory(case_dir):
                        raise ValueError(f"Refusing unrecognized case directory: {case_dir}")
                    shutil.rmtree(case_dir)
                case_dir.mkdir()
                (case_dir / MARKER).write_text(MARKER_CONTENT)
                atomic_json(case_dir / "plan.json", plan)
                (case_dir / "effective-config.yaml").write_text(
                    yaml.safe_dump(case["effective_config"], sort_keys=False)
                )
                code = runner(case_dir / "plan.json", case_dir / "calibration.log", timeout)
                rows[name]["return_code"] = code
                if code != 0:
                    raise RuntimeError(
                        f"Calibration subprocess exited with status {code}; see {case_dir / 'calibration.log'}"
                    )
                evidence = validate_capture(case_dir)
                if source_identity(source, inputs, mode, reuse_points) != identity:
                    raise RuntimeError("Source inputs changed during capture")
                rows[name].update(
                    outcome="pass",
                    evidence=evidence,
                    return_code=code,
                    png=f"{name}/game/s.png",
                    pto=f"{name}/game/autooptimiser_out.pto",
                )
                print(f"PASS {name}", flush=True)
            except Exception as error:
                failures += 1
                rows[name].update(outcome="fail", error=str(error))
                print(f"FAIL {name}: {error}", file=sys.stderr, flush=True)
            rows[name]["duration_seconds"] = round(time.monotonic() - started, 3)
            atomic_json(manifest_path, rows)
    print(f"{len(selected) - failures}/{len(selected)} cases passed or reused; results: {output}")
    return int(failures > 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "hmlib/config/stitching_projection_frames.yaml"
    )
    parser.add_argument("--source-game-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-mode", choices=("videos", "images"))
    parser.add_argument(
        "--reuse-control-points", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        return run_worker(args.worker.resolve())
    if args.start_at < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--start-at and --limit must be positive")
    matrix = read_yaml(args.config.resolve())
    source_value = args.source_game_dir or matrix.get("source_game_dir")
    source = None if source_value is None else Path(source_value).expanduser()
    if source is not None:
        source = (
            source
            if source.is_absolute()
            else (Path.cwd() if args.source_game_dir else args.config.resolve().parent) / source
        ).resolve()
    config = read_yaml(source / "config.yaml") if source is not None else {}
    cases = expand_cases(matrix, config)
    if args.dry_run:
        selected = cases[args.start_at - 1 :]
        selected = selected if args.limit is None else selected[: args.limit]
        if not selected:
            raise ValueError("No projection cases selected")
        for case in selected:
            print(case["label"])
        print(f"Dry run: {len(selected)} of {len(cases)} cases")
        return 0
    if source is None:
        parser.error("Supply --source-game-dir or source_game_dir in the matrix YAML")
    output_value = args.output_dir or matrix.get("output_dir")
    if not output_value:
        parser.error("Supply --output-dir or output_dir in the matrix YAML")
    output = Path(output_value).expanduser()
    output = Path(
        os.path.abspath(
            output
            if output.is_absolute()
            else (Path.cwd() if args.output_dir else args.config.resolve().parent) / output
        )
    )
    # Reject a symlinked output before resolving its other path components.
    if output.is_symlink():
        raise ValueError("Capture output cannot be a symlink")
    output = output.resolve()
    mode = args.source_mode or matrix.get("source_mode", "videos")
    if mode not in ("videos", "images"):
        raise ValueError("source_mode must be videos or images")
    reuse = (
        args.reuse_control_points
        if args.reuse_control_points is not None
        else matrix.get("reuse_control_points", False)
    )
    if not isinstance(reuse, bool):
        raise ValueError("reuse_control_points must be a boolean")
    return capture(
        matrix,
        cases,
        source,
        output,
        mode=mode,
        reuse_points=reuse,
        start_at=args.start_at,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
