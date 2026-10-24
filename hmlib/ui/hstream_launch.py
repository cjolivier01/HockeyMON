"""Translate the shared desktop's versioned launch contract to a Python run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROTOCOL = {"protocol": "hstream-ui", "version": 1, "backend": "hockeymon", "preview": "jpeg-files"}
SINKS = {
    "FAKE",
    "RENDER",
    "ENCODE_FILE",
    "ENCODE_STITCHED_FILE",
    "ENCODE_PROGRAM_4K_FILE",
    "RTMP",
    "RTSP",
}
# These settings describe the native mux/preview transport, which the adapter
# replaces with synchronized Python source loaders and the JPEG preview mailbox.
NATIVE_TRANSPORT = {
    "pipeline.streammux.batch-size": 2,
    "pipeline.streammux.sync-inputs": 0,
    "pipeline.streammux.batched-push-timeout": 2147483647,
    "pipeline.streammux.frame-num-reset-on-stream-reset": 0,
    "pipeline.streammux.frame-num-reset-on-eos": 0,
    "pipeline.hmstitcher.show": 0,
}
RINK_CONTROLS = {
    "rink-camera-enable": "enabled",
    "rink-camera-anticipate": "anticipate",
    "rink-camera-frame-goals": "frame_goals",
    "rink-camera-adaptive-zoom": "adaptive_zoom",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="HockeyMON processing backend for the shared HStream desktop",
        allow_abbrev=False,
    )
    result.add_argument("--ui-backend-info", action="store_true")
    result.add_argument("-g", "--game-id")
    result.add_argument("-c", "--config", action="append", default=[])
    result.add_argument("--options", action="append", default=[])
    result.add_argument("--enable-sources", default="URI-MULTIPLE")
    result.add_argument("--enable-sinks", default="RENDER")
    result.add_argument("--start-time", default="00:00:00")
    result.add_argument("--stitch-frame-time")
    result.add_argument("-t", "--time-limit", type=float)
    result.add_argument("--render-realtime", action="store_true")
    result.add_argument("--show", action="store_true")
    result.add_argument("--show-scaled", type=float)
    result.add_argument("--force-reconfigure", action="store_true")
    result.add_argument("--clean-expected-invalidation-id")
    result.add_argument("--stitching-calibration-only", action="store_true")
    result.add_argument("--ui-preview-windows")
    result.add_argument("--ui-preview-active", default="program")
    result.add_argument("--export-job", action="store_true")
    result.add_argument("--prepare-detector", action="store_true")
    result.add_argument("--samples", type=int, default=128)
    result.add_argument("--detector-model", default="default")
    result.add_argument("--game-dir", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--force", action="store_true")
    result.add_argument("--encode-job", action="store_true")
    result.add_argument("--sbatch", action="append", default=[])
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the translated launch without starting inference",
    )
    return result


def set_path(config: dict, key: str, value) -> None:
    parts = key.split(".")
    if not all(part and part.replace("_", "").replace("-", "").isalnum() for part in parts):
        raise ValueError(f"Invalid setting path: {key}")
    node = config
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise TypeError(f"Cannot set {key} beneath non-mapping {part}")
        node = node[part]
    node[parts[-1]] = value


def get_path(config: dict, key: str, default=None):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def read_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return value


def native_application_config(runner: Path) -> Path:
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
    raise FileNotFoundError("Matching native application config is unavailable; set HM_CONFIG_ROOT")


def merge(left: dict, right: dict) -> dict:
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def seconds(value: str) -> float:
    pieces = value.split(":")
    if len(pieces) > 3:
        raise ValueError(f"Invalid playback time: {value}")
    result = 0.0
    for piece in pieces:
        number = float(piece)
        if number < 0 or not math.isfinite(number):
            raise ValueError("Playback time must be finite and nonnegative")
        result = result * 60 + number
    return result


@dataclass
class LaunchPlan:
    game: Path
    config: dict
    arguments: list[str]
    explicit_arguments: list[str]
    cwd: Path
    sources: dict[str, list[Path]]
    sinks: set[str]
    start_seconds: float
    graph: dict
    start_frame: int = 0
    limit_seconds: float | None = None
    limit_frames: int | None = None
    initial_settings: dict = field(default_factory=dict)

    def launch_receipt(self) -> dict:
        """Tell the desktop which outputs the initial Python invocation resolved."""
        return {
            "version": 1,
            "game_id": self.game.name,
            "sinks": sorted(self.sinks),
            "telemetry": bool(self.config["hm_shared_ui"]["telemetry"]),
        }

    def write(self, session: Path, *, seek_seconds: float = 0) -> list[str]:
        config_path = session / "launch.yaml"
        set_path(
            self.config, "launch.start-time-ns", round((self.start_seconds + seek_seconds) * 1e9)
        )
        config_path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        # Explicit initial Python arguments retain their argparse meaning. Only
        # transport ownership and a supervised seek are owned by this adapter.
        arguments = [*self.arguments, "--config", str(config_path), *self.explicit_arguments]
        if seek_seconds:
            arguments = strip_options(
                arguments, {"--start-time", "--start-frame-time", "-s", "--start-frame"}
            )
            arguments.append(f"--start-time={self.start_seconds + seek_seconds:.9f}")
            if self.limit_seconds is not None:
                remaining = self.limit_seconds - seek_seconds
                if remaining <= 0:
                    raise ValueError("Seek leaves no recording time to process")
                arguments = strip_options(arguments, {"--max-time", "-t", "--max-frames"})
                arguments.append(f"--max-time={remaining:.9f}")
        return [
            sys.executable,
            "-u",
            "-m",
            "hmlib.cli.hmtrack",
            *arguments,
            "--camera-ui=1",
        ]


def strip_options(arguments: list[str], names: set[str]) -> list[str]:
    result, index = [], 0
    while index < len(arguments):
        token = arguments[index]
        if token.split("=", 1)[0] in names:
            if "=" not in token and index + 1 == len(arguments):
                raise ValueError(f"Missing value for {token}")
            index += 1 if "=" in token else 2
        else:
            result.append(token)
            index += 1
    return result


def option_assignments(value: str) -> list[str]:
    """Split native grouped assignments, retaining YAML containers and quoted commas."""
    entries, start, depth, quote, escaped = [], 0, 0, None, False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
        elif quote:
            if char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == "," and depth == 0 and re.match(r"[\w.-]+=", value[index + 1 :]):
            entries.append(value[start:index])
            start = index + 1
    if depth or quote:
        raise ValueError(f"Malformed launch options: {value}")
    entries.append(value[start:])
    return entries


def _initial_output_routes(settings: dict, sinks: set[str]) -> None:
    """Apply only explicitly supplied initial settings to the desktop routes."""
    missing = object()
    if get_path(settings, "video_out.output_video_path"):
        sinks.add("ENCODE_FILE")
    for key, sink in (
        ("hm_ui.run.save_video", "ENCODE_FILE"),
        ("video_out.archive_program", "ENCODE_FILE"),
        ("video_out.archive_stitched", "ENCODE_STITCHED_FILE"),
        ("video_out.archive_program_4k", "ENCODE_PROGRAM_4K_FILE"),
        ("video_out.rtsp_enabled", "RTSP"),
    ):
        value = get_path(settings, key, missing)
        if value is missing or value is None:
            continue
        if not isinstance(value, (bool, int)) or value not in (False, True, 0, 1):
            raise ValueError(f"Initial output setting {key} must be boolean")
        if value:
            sinks.add(sink)
        else:
            sinks.discard(sink)


def make_plan(args: argparse.Namespace, *, initial: Path | None = None) -> LaunchPlan:
    from hmlib.config import get_config
    from hmlib.ui.analytics_graph import graph_paths, selected_analysis
    from hmlib.ui.runtime_setup import (
        apply_engine_preferences,
        launch_arguments,
        prefer_explicit_launch_arguments,
    )

    if not args.game_id or Path(args.game_id).name != args.game_id or args.game_id in {".", ".."}:
        raise ValueError("A simple --game-id is required")
    root = Path(os.environ.get("HM_GAME_DIR", str(Path.home() / "Videos"))).expanduser().resolve()
    game = root / args.game_id
    if not game.is_dir():
        raise FileNotFoundError(game)
    if args.enable_sources != "URI-MULTIPLE":
        raise ValueError("The HockeyMON desktop backend requires recording sources (URI-MULTIPLE)")
    sinks = set(args.enable_sinks.split(","))
    if not sinks or sinks - SINKS:
        raise ValueError(f"Unknown output routes: {sorted(sinks - SINKS)}")
    explicit, cwd = [], Path.cwd()
    initial_args = None
    if initial:
        if initial.stat().st_size > 1024 * 1024:
            raise ValueError("Initial launch envelope is too large")
        request = json.loads(initial.read_text())
        if (
            request.get("version") != 1
            or request.get("module") != "hmlib.cli.hmtrack"
            or request.get("game_id") != args.game_id
        ):
            raise ValueError("Initial launch envelope does not match this Program run")
        explicit = request["arguments"]
        if not isinstance(explicit, list) or not all(
            isinstance(value, str) and "\0" not in value for value in explicit
        ):
            raise ValueError("Initial launch arguments must be strings")
        cwd = Path(request["cwd"]).resolve()
        if not cwd.is_dir():
            raise ValueError("Initial launch working directory is unavailable")
        from hmlib.cli.hmtrack import make_parser
        from hmlib.hm_opts import hm_opts

        python_parser = make_parser()
        initial_args = python_parser.parse_args(explicit)
    config = get_config(
        game_id=args.game_id,
        root_dir=str((cwd / initial_args.root_dir).resolve()) if initial_args else None,
        rink=initial_args.rink if initial_args else None,
        camera=initial_args.camera_name if initial_args else None,
        ignore_private_config=bool(initial_args.ignore_private_config) if initial_args else False,
        resolve_globals=False,
    )
    # HStream user preferences precede game settings, including native aliases.
    user = Path.home() / ".hstream" / "hstream.yaml"
    if user.is_file():
        config = merge(config, read_mapping(user))
        if initial_args is None or not initial_args.ignore_private_config:
            private_config = game / "config.yaml"
            if private_config.is_file():
                config = merge(config, read_mapping(private_config))
    for path in args.config:
        native = Path(path)
        if native.name != "ds_hockey_app_config.yaml":
            raise ValueError(
                f"Unsupported structural native graph: {native}; select the shared Program graph"
            )
        if not native.is_file():
            raise FileNotFoundError(native)
    options = {}
    for assignment in args.options:
        # Native UI groups structural pipeline aliases with commas, while all
        # ordinary settings are passed as individual --options values.
        for entry in option_assignments(assignment):
            key, separator, text = entry.partition("=")
            if not separator:
                raise ValueError(f"Expected --options=key=value: {entry}")
            options[key] = yaml.safe_load(text)
    for key, value in options.items():
        if key in NATIVE_TRANSPORT:
            if value != NATIVE_TRANSPORT[key]:
                raise ValueError(f"Unsupported native transport override {key}={value}")
            continue
        if key.startswith("pipeline.ds-playtracker.private-properties.rink-camera-"):
            name = key.rsplit(".", 1)[-1]
            if name not in RINK_CONTROLS or value not in (True, False, 0, 1):
                raise ValueError(f"Invalid rink framing control {key}={value}")
            set_path(config, "rink.camera.rink_framing." + RINK_CONTROLS[name], bool(value))
            continue
        if not key.startswith(
            (
                "stitching.",
                "rink.",
                "plot.",
                "camera.",
                "team.",
                "pipeline.player-analytics.",
                "pipeline.tracker.",
                "hstream_ui.camera_controls.",
                "hstream_ui.hockeymon.detector.",
                "runtime.gpu_memory_profile",
            )
        ) and key not in {
            "pipeline.hmstitcher.properties.high-bit-depth",
            "pipeline.hmaudio.enable",
            "pipeline.primary-gie.config-file",
            "pipeline.primary-gie.model-engine-file",
            "pipeline.ds-playtracker.private-properties.telemetry-db-dir",
            "pipeline.ds-playtracker.private-properties.telemetry-game-id",
            "pipeline.hmstitcher.calibration-frame-count",
        }:
            raise ValueError(
                f"The HockeyMON backend does not implement native launch setting {key}"
            )
        set_path(config, key, value)
    _translate_native_settings(config, sinks)
    run = config.setdefault("hm_ui", {}).setdefault("run", {})
    run["save_video"] = "ENCODE_FILE" in sinks
    set_path(config, "video_out.archive_program", "ENCODE_FILE" in sinks)
    set_path(config, "video_out.archive_stitched", "ENCODE_STITCHED_FILE" in sinks)
    set_path(config, "video_out.archive_program_4k", "ENCODE_PROGRAM_4K_FILE" in sinks)
    set_path(config, "video_out.show_youtube", "RTMP" in sinks)
    # The shared desktop has one RTMP choice. A legacy spare publisher must
    # never bypass that selection; an explicit initial CLI config can request
    # it below, where it participates in the same network/seek receipt.
    set_path(config, "video_out.spare_rtmp_enabled", False)
    set_path(config, "video_out.rtsp_enabled", "RTSP" in sinks)
    initial_settings = {}
    if initial_args is not None:
        for filename in initial_args.config or []:
            path = Path(filename)
            layer = read_mapping(path if path.is_absolute() else cwd / path)
            _initial_output_routes(layer, sinks)
            initial_settings = merge(initial_settings, layer)
            config = merge(config, layer)
        explicit_names = {
            python_parser._option_string_actions[token.split("=", 1)[0]].dest
            for token in explicit
            if token.split("=", 1)[0] in python_parser._option_string_actions
        }
        hm_opts.apply_arg_config_overrides(
            config, initial_args, parser=python_parser, explicit_arg_names=explicit_names
        )
        hm_opts.apply_config_overrides(config, initial_args.config_overrides)
        explicit_settings = {}
        hm_opts.apply_arg_config_overrides(
            explicit_settings, initial_args, parser=python_parser, explicit_arg_names=explicit_names
        )
        hm_opts.apply_config_overrides(explicit_settings, initial_args.config_overrides)
        initial_settings = merge(initial_settings, explicit_settings)
        _initial_output_routes(explicit_settings, sinks)
        if initial_args.output_video or initial_args.output_file:
            sinks.add("ENCODE_FILE")
        if initial_args.no_save_video:
            sinks.discard("ENCODE_FILE")
        if initial_args.no_audio:
            set_path(config, "pipeline.hmaudio.enable", False)
            set_path(initial_settings, "pipeline.hmaudio.enable", False)
        rtmp_enabled = False
        for key in ("show_youtube", "spare_rtmp_enabled"):
            value = get_path(config, "video_out." + key, False)
            if value is None:
                value = key == "show_youtube" and "RTMP" in sinks
            if not isinstance(value, (bool, int)) or value not in (False, True, 0, 1):
                raise ValueError(f"Initial output setting video_out.{key} must be boolean")
            set_path(config, "video_out." + key, bool(value))
            rtmp_enabled |= bool(value)
        if rtmp_enabled:
            sinks.add("RTMP")
        else:
            sinks.discard("RTMP")
        if {"lfo", "rfo"} & explicit_names:
            if initial_args.lfo is None or initial_args.rfo is None:
                raise ValueError("Manual synchronization requires both --lfo and --rfo")
            for layer in (config, initial_settings):
                set_path(layer, "stitching.sync_method", "manual")
                set_path(
                    layer,
                    "game.stitching.frame_offsets",
                    {"left": initial_args.lfo, "right": initial_args.rfo},
                )
        # Original CLI/config intent precedes generated flags. Freeze the same
        # routes for worker construction, seek admission and the UI receipt.
        set_path(config, "hm_ui.run.save_video", "ENCODE_FILE" in sinks)
        for key, sink in (
            ("archive_program", "ENCODE_FILE"),
            ("archive_stitched", "ENCODE_STITCHED_FILE"),
            ("archive_program_4k", "ENCODE_PROGRAM_4K_FILE"),
            ("rtsp_enabled", "RTSP"),
        ):
            set_path(config, "video_out." + key, sink in sinks)
        if not sinks:
            sinks.add("FAKE")
    paths = graph_paths(selected_analysis(config))
    graph_config = {}
    for path in paths:
        graph_config = merge(graph_config, read_mapping(Path(path)))
    config = merge(graph_config, config)
    apply_engine_preferences(config)
    prepared = []
    if get_path(config, "hstream_ui.hockeymon.detector.precision") == "int8":
        from hmlib.ui.hstream_detector import validated_detector_arguments

        prepared = validated_detector_arguments(game, config)
    generated = launch_arguments(
        game, "track", config=config, explicit_arguments=[*prepared, *explicit]
    )
    generated += prepared
    generated = prefer_explicit_launch_arguments(generated, explicit, "track")
    arguments = ["--game-id", args.game_id, *generated, f"--start-time={args.start_time}"]
    if not get_path(config, "pipeline.hmaudio.enable", True):
        arguments.append("--no-audio")
    if initial_args is not None and initial_args.start_frame:
        arguments = strip_options(arguments, {"--start-time"})
    if args.time_limit is not None:
        if args.time_limit <= 0:
            raise ValueError("Time limit must be positive")
        arguments += [f"--max-time={args.time_limit}"]
    # Desktop runs publish even a deliberately short preview; their receipt
    # determines which generation the shared archive browser can expose.
    if not any(token.split("=", 1)[0] == "--deploy-dir" for token in explicit):
        arguments += ["--deploy-dir", str(game)]
    telemetry = "pipeline.ds-playtracker.private-properties.telemetry-db-dir" in options
    if not telemetry:
        for name, plugin in config.get("aspen", {}).get("plugins", {}).items():
            if str(plugin.get("class", "")).endswith(
                ("SaveCameraPlugin", "SaveTrackingPlugin", "SaveDetectionsPlugin", "SavePosePlugin")
            ):
                plugin["enabled"] = False
    config.setdefault("hm_shared_ui", {})["telemetry"] = telemetry
    videos = get_path(config, "game.videos", {})
    if initial_args is not None and initial_args.input_video:
        selected = [(cwd / path).resolve() for path in initial_args.input_video.split(",")]
        if len(selected) == 1 and selected[0].is_dir():
            if selected[0] != game:
                raise ValueError("The initial input directory must match the selected game")
        elif len(selected) in {1, 2}:
            videos = dict(
                zip(
                    ("source",) if len(selected) == 1 else ("left", "right"),
                    [[str(path)] for path in selected],
                )
            )
            set_path(config, "game.videos", videos)
        else:
            raise ValueError("Program input requires one recording or a left/right pair")
    if not isinstance(videos, dict) or not videos:
        raise ValueError("The selected game has no recording sources")
    sources = {}
    for name, files in videos.items():
        if files is None or files == []:
            continue
        if isinstance(files, str):
            files = [files]
        sources[name] = [Path(file) if Path(file).is_absolute() else game / file for file in files]
        for file in sources[name]:
            if not file.is_file():
                raise FileNotFoundError(file)
    if set(sources) not in ({"source"}, {"left", "right"}):
        raise ValueError("Program requires a source recording or a complete left/right pair")
    if set(sources) == {"source"} and not any(
        token.split("=", 1)[0] == "--input-video" for token in explicit
    ):
        arguments += ["--input-video", str(sources["source"][0])]
    if set(sources) == {"left", "right"}:
        arguments.append("--aspen-stitching")
    arguments = prefer_explicit_launch_arguments(arguments, explicit, "track")
    return LaunchPlan(
        game,
        config,
        arguments,
        explicit,
        cwd,
        sources,
        sinks,
        (
            seconds(initial_args.start_frame_time or args.start_time)
            if initial_args
            else seconds(args.start_time)
        ),
        config.get("aspen", {}),
        initial_args.start_frame if initial_args else 0,
        (
            seconds(initial_args.max_time)
            if initial_args and initial_args.max_time
            else args.time_limit
        ),
        initial_args.max_frames if initial_args else None,
        initial_settings,
    )


def _translate_native_settings(config: dict, sinks: set[str]) -> None:
    memory_profile = get_path(config, "runtime.gpu_memory_profile")
    if memory_profile is not None:
        profiles = {"auto": "automatic", "standard": "normal", "low": "low"}
        if memory_profile not in profiles:
            raise ValueError(f"Unknown GPU memory profile: {memory_profile}")
        set_path(config, "hm_ui.run.memory_profile", profiles[memory_profile])
    controls = get_path(config, "hstream_ui.camera_controls", {})
    for name, key, divisor in (
        ("Bring_Up_Shadows", "shadow_lift", 1),
        ("Lift_Shadow_Black_Point", "shadow_lift_black_point", 1),
        ("Exposure_x100", "exposure_ev", 200),
    ):
        if name in controls:
            value = (
                bool(controls[name])
                if key.endswith("black_point")
                else float(controls[name]) / divisor
            )
            set_path(config, f"rink.camera.color.{key}", value)
    high_bit = str(
        get_path(config, "pipeline.hmstitcher.properties.high-bit-depth", "auto")
    ).lower()
    if high_bit not in {"auto", "0", "1", "false", "true"}:
        raise ValueError("High-bit-depth must be auto, 0 or 1")
    if high_bit in {"1", "true"}:
        set_path(config, "hm_ui.run.stitch_precision", "float32")
    elif high_bit in {"0", "false"}:
        set_path(config, "hm_ui.run.stitch_precision", "uint8")
    shared_detector = get_path(config, "hstream_ui.hockeymon.detector", {})
    selected = get_path(config, "pipeline.primary-gie.config-file")
    if shared_detector:
        families = {
            "default": "config_infer_yolov8_hockey",
            "distilled-s": "config_infer_yolov8s_hockey",
        }
        model = shared_detector.get("model", "default")
        if model not in families:
            raise ValueError(f"Unknown shared detector model: {model}")
        precision = shared_detector.get("precision", "fp32")
        selected = families[model] + ("" if precision == "fp32" else "_" + precision)
    if selected:
        name = Path(selected).stem
        families = {
            "config_infer_yolov8_hockey": "hockeyref",
            "config_infer_yolov8s_hockey": "distilled",
        }
        family = next(
            (prefix for prefix in families if name == prefix or name.startswith(prefix + "_")), None
        )
        if family is None:
            raise ValueError(f"Unknown shared detector model: {selected}")
        precision = name.removeprefix(family).lstrip("_") or "fp32"
        engines = {
            "fp32": "pytorch",
            "fp16": "pytorch-fp16",
            "bf16": "pytorch-bf16",
            "int8": "onnx-int8",
        }
        if precision not in engines:
            raise ValueError(f"Unsupported detector precision: {precision}")
        set_path(config, "hm_ui.run.detector", families[family])
        set_path(config, "hm_ui.run.precision", engines[precision])
    if "RTMP" in sinks:
        url = (
            get_path(config, "pipeline.sink3.output-file")
            or get_path(config, "sink3.output-file")
            or get_path(config, "video_out.youtube_stream_url")
        )
        if url:
            set_path(config, "video_out.youtube_stream_url", url)
    if "RTSP" in sinks:
        port = get_path(config, "pipeline.sink3.rtsp-port") or get_path(config, "sink3.rtsp-port")
        if port:
            set_path(config, "video_out.rtsp_port", port)
        # The shared native server publishes this endpoint on every interface.
        set_path(config, "video_out.rtsp_program_mount", "/ds-test")
        set_path(config, "video_out.rtsp_host", get_path(config, "video_out.rtsp_host", "0.0.0.0"))


def export_job(args: argparse.Namespace) -> Path:
    if args.game_dir is None or args.output is None:
        raise ValueError("Job export requires --game-dir and --output")
    game, output = args.game_dir.resolve(), args.output.resolve()
    if not (game / "config.yaml").is_file():
        raise ValueError("Job export requires a configured game directory")
    if any("\n" in line or "\r" in line or not line.startswith("--") for line in args.sbatch):
        raise ValueError("Each scheduler directive must be one --option line")
    config = read_mapping(game / "config.yaml")
    recipe = get_path(config, "hstream_ui.job.arguments", [])
    if not isinstance(recipe, list) or not all(isinstance(value, str) for value in recipe):
        raise ValueError("Saved job arguments must be a list of strings")
    saved = parser().parse_args(["--game-id", game.name, *recipe])
    encoders = set(saved.enable_sinks.split(",")) & {
        "ENCODE_FILE",
        "ENCODE_STITCHED_FILE",
        "ENCODE_PROGRAM_4K_FILE",
    }
    if not encoders:
        encoders = {"ENCODE_PROGRAM_4K_FILE"}
    if saved.stitching_calibration_only and encoders != {"ENCODE_STITCHED_FILE"}:
        raise ValueError("Calibration jobs must select only Archive Stitched")
    recipe = strip_options(
        recipe,
        {
            "--enable-sinks",
            "--ui-preview-windows",
            "--ui-preview-active",
            "--show-scaled",
            "--game-id",
            "-g",
            "-c",
            "--config",
        },
    )
    recipe = [value for value in recipe if value not in {"--show", "--render-realtime"}]
    recipe = [
        value
        for value in recipe
        if not value.startswith("--options=pipeline.ds-playtracker.private-properties.telemetry-")
    ]
    command = ["--game-id", game.name, *recipe, "--enable-sinks=" + ",".join(sorted(encoders))]
    if get_path(config, "hstream_ui.job.capture_telemetry", False):
        command += [
            f"--options=pipeline.ds-playtracker.private-properties.telemetry-db-dir={game}",
            f"--options=pipeline.ds-playtracker.private-properties.telemetry-game-id={game.name}",
        ]
    lines = [
        "#!/bin/sh",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:1",
        *(f"#SBATCH {line}" for line in args.sbatch),
        "set -eu",
        "unset HSTREAM_UI_PARENT_PID HSTREAM_UI_BACKEND_SESSION HM_UI_INITIAL_LAUNCH",
        f"export HM_GAME_DIR={shlex.quote(str(game.parent))}",
    ]
    if output_root := os.environ.get("HM_OUTPUT_WORK_DIR"):
        lines.append(f"export HM_OUTPUT_WORK_DIR={shlex.quote(output_root)}")
    lines.append('exec "${HM_CLI_BIN:-hm-cli}" ' + shlex.join(command))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.force else "x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    output.chmod(0o755)
    return output
