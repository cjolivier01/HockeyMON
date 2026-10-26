"""Persistent operator settings and the concrete Python launch options they own."""

from __future__ import annotations

from pathlib import Path

from hmlib.ui.analytics_graph import apply_analytics_preferences, graph_paths, selected_analysis
from hmlib.ui.calibration_workspace import CalibrationWorkspace, _get


def _field(key, label, kind, default=None, **extra):
    return dict(key=key, label=label, kind=kind, default=default, apply_mode="next-run", **extra)


RUN_FIELDS = [
    _field(
        "hm_ui.run.detector",
        "Detector model",
        "choice",
        "saved",
        choices=["saved", "deployed", "trained", "distilled", "hockeyref", "hockey-roles"],
    ),
    _field(
        "hm_ui.run.precision",
        "Detector engine",
        "choice",
        "saved",
        choices=[
            "saved",
            "pytorch",
            "pytorch-fp16",
            "pytorch-bf16",
            "trt-fp32",
            "trt-fp16",
            "onnx",
            "onnx-int8",
        ],
    ),
    _field(
        "hm_ui.run.analytics",
        "Player analysis",
        "choice",
        "saved",
        choices=[
            "saved",
            "tracking",
            "jersey",
            "pose",
            "pose-jersey",
            "pose-actions",
            "pose-jersey-actions",
            "custom",
        ],
    ),
    _field("pipeline.player-analytics.pose.enable", "Estimate pose", "bool", False),
    _field("pipeline.player-analytics.jersey.enable", "Read jersey numbers", "bool", False),
    _field(
        "pipeline.player-analytics.action.enable",
        "Recognize activities",
        "bool",
        False,
    ),
    _field(
        "hm_ui.run.pose_model",
        "Pose model",
        "choice",
        "saved",
        choices=["saved", "native", "harpet", "coco", "custom"],
    ),
    _field("hm_ui.run.pose_config", "Custom pose model config", "text", ""),
    _field("hm_ui.run.pose_checkpoint", "Custom pose checkpoint or URL", "text", ""),
    _field(
        "hm_ui.run.jersey_model", "Jersey model", "choice", "native", choices=["native", "custom"]
    ),
    _field("hm_ui.run.jersey_model_path", "Custom hockey PARSeq ONNX model", "text", ""),
    _field(
        "hm_ui.run.action_model", "Activity model", "choice", "native", choices=["native", "custom"]
    ),
    _field("hm_ui.run.action_model_path", "Custom COCO17 STGCN++ ONNX model", "text", ""),
    _field(
        "pipeline.player-analytics.jersey.rate-hz",
        "Jersey updates per second",
        "number",
        2,
        min=0.1,
        max=120,
    ),
    _field(
        "pipeline.player-analytics.jersey.confidence-threshold",
        "Jersey confidence",
        "number",
        0.8,
        min=0,
        max=1,
    ),
    _field(
        "pipeline.player-analytics.action.rate-hz",
        "Activity updates per second",
        "number",
        1,
        min=0.1,
        max=10,
    ),
    _field(
        "pipeline.player-analytics.action.confidence-threshold",
        "Activity confidence",
        "number",
        0.5,
        min=0,
        max=1,
    ),
    _field(
        "pipeline.player-analytics.jersey.roi-mode",
        "Jersey crop",
        "choice",
        "bbox",
        choices=["bbox", "pose"],
    ),
    _field("plot.plot_pose", "Draw pose skeletons", "bool", False),
    _field("plot.plot_jersey_numbers", "Draw jersey numbers", "bool", False),
    _field("plot.plot_actions", "Draw activity labels", "bool", False),
    _field("plot.plot_teams", "Draw team labels", "bool", False),
    _field("plot.plot_individual_player_tracking", "Draw player boxes", "bool", False),
    _field(
        "hm_ui.run.pose_precision",
        "Pose engine",
        "choice",
        "saved",
        choices=["saved", "pytorch", "trt-fp32", "trt-fp16", "onnx", "onnx-int8"],
    ),
    _field("hm_ui.run.save_video", "Record Program video", "bool", True),
    _field("hm_ui.run.output_fps", "Recording FPS (0 = source)", "number", 0, min=0, max=240),
    _field("hm_ui.run.batch_size", "Input batch size", "int", 1, min=1, max=64),
    _field(
        "hm_ui.run.memory_profile",
        "GPU memory profile",
        "choice",
        "automatic",
        choices=["automatic", "normal", "low"],
    ),
    _field(
        "hm_ui.run.stitch_precision",
        "Stitching precision",
        "choice",
        "saved",
        choices=["saved", "uint8", "float16", "float32"],
    ),
    _field(
        "rink.scoreboard.position",
        "Scoreboard placement",
        "choice",
        "upper-left",
        choices=["upper-left", "upper-center", "upper-right"],
    ),
    _field("rink.scoreboard.enabled", "Show scoreboard", "bool", True),
    _field("rink.scoreboard.scoreboard_scale", "Scoreboard scale", "number", 1, min=0.1, max=4),
    _field(
        "rink.scoreboard.resize_filter",
        "Scoreboard resize filter",
        "choice",
        "bilinear",
        choices=["nearest", "bilinear", "bicubic"],
    ),
    _field("rink.scoreboard.sharpen_amount", "Scoreboard sharpening", "number", 0, min=0, max=4),
    _field(
        "rink.camera.rink_framing.map_size",
        "Mini-map size (desktop only)",
        "choice",
        "standard",
        choices=["icon", "compact", "standard", "large"],
    ),
    _field(
        "rink.camera.rink_framing.position",
        "Mini-map placement (desktop only)",
        "choice",
        "upper-right",
        choices=["upper-left", "upper-right", "lower-left"],
    ),
    _field("aspen.pipeline.threaded", "Threaded pipeline", "bool"),
    _field("aspen.pipeline.queue_size", "Pipeline queue size", "int", min=1, max=64),
    _field("aspen.pipeline.max_concurrent", "Concurrent pipeline stages", "int", min=1, max=64),
    _field("team.enabled", "Identify teams", "bool"),
    _field("pipeline.tracker.reid-enable", "Match player appearance", "bool", False),
    _field(
        "pipeline.tracker.reid-model",
        "Player appearance model",
        "choice",
        "reidentificationnet-deployable-v1.2",
        choices=["reidentificationnet-deployable-v1.2"],
    ),
    _field("hm_ui.run.reid_onnx", "Custom player appearance ONNX file", "text", ""),
    _field(
        "team.backend",
        "Team appearance model",
        "choice",
        choices=["prtreid", "lab", "reidentificationnet"],
    ),
    _field("game.home_is_dark", "Home wears dark sweaters", "bool"),
    _field("team.warmup_samples", "Team warmup samples", "int", min=2, max=100000),
    _field("team.refit_interval", "Team refit samples", "int", min=1, max=100000),
    _field("team.min_margin", "Team confidence margin", "number", min=0, max=1),
    _field("detector.max_referees", "Maximum officials", "int", min=0, max=20),
    _field("pipeline.player-analytics.pose.rate-hz", "Pose cadence (Hz or fps)", "text", "10"),
    _field(
        "pipeline.player-analytics.pose.continuous-drawing",
        "Hold pose between inference frames",
        "bool",
        True,
    ),
    _field("video_out.archive_stitched", "Record Stitched archive", "bool", False),
    _field("video_out.archive_program_4k", "Record Program 4K archive", "bool", False),
    _field("video_out.spare_rtmp_enabled", "Publish spare RTMP stream", "bool", False),
    _field("video_out.spare_rtmp_url", "Spare RTMP publish URL", "text", ""),
    _field("video_out.rtsp_enabled", "Serve local RTSP", "bool", False),
    _field("video_out.rtsp_host", "RTSP listen address", "text", "127.0.0.1"),
    _field("video_out.rtsp_port", "RTSP port", "int", 8554, min=1, max=65535),
    _field("video_out.rtsp_program_mount", "Program RTSP mount", "text", "/program"),
    _field("video_out.rtsp_stitched_mount", "Stitched RTSP mount", "text", "/stitched"),
    _field(
        "video_out.encoder_backend",
        "Recording muxer",
        "choice",
        choices=["auto", "pyav", "ffmpeg", "raw"],
    ),
    _field("video_out.output_width", "Recording width (auto or pixels)", "text", "auto"),
    _field("video_out.output_height", "Recording height (empty = preserve aspect)", "text", ""),
    _field("video_out.bit_rate", "Recording bitrate (empty = automatic)", "text", ""),
    _field("video_out.show_youtube", "Publish RTMP stream", "bool"),
    _field(
        "video_out.youtube_stream_url", "RTMP ingest URL (key from HM_YOUTUBE_STREAM_KEY)", "text"
    ),
    _field("video_out.always_stream", "Keep browser preview available", "bool"),
    _field("video_out.headless_preview_host", "Browser preview host", "text"),
    _field(
        "video_out.headless_preview_port",
        "Browser preview port (0 = automatic)",
        "int",
        min=0,
        max=65535,
    ),
]


def camera_fields(config: dict) -> list[dict]:
    """Expose the shipped camera policy without creating a second defaults table."""
    from hmlib.config import ROOT_DIR, baseline_config

    fields = []

    def visit(mapping, prefix="rink.camera"):
        for name, default in mapping.items():
            key = f"{prefix}.{name}"
            if isinstance(default, dict):
                visit(default, key)
                continue
            label = (
                key.removeprefix("rink.camera.").replace("_", " ").replace(".", " · ").capitalize()
            )
            value = _get(config, key, default)
            if key.endswith(".controller"):
                fields.append(
                    _field(
                        key,
                        label,
                        "choice",
                        default,
                        choices=["rule", "transformer", "gpt", "drivegpt"],
                    )
                )
            elif isinstance(default, bool):
                fields.append(_field(key, label, "bool", default))
            elif isinstance(value, (list, tuple)):
                fields.append(_field(key, label, "array", default, length=len(value)))
            elif isinstance(default, (int, float)):
                minimum, maximum = 0, 100000
                if name == "fixed_edge_rotation_angle":
                    minimum, maximum = -90, 90
                elif name == "exposure_ev":
                    minimum, maximum = -10, 10
                elif name == "zoom_in_aggressiveness":
                    maximum = 100
                elif name == "camera_window":
                    minimum = 1
                elif name in {"brightness", "contrast", "gamma"}:
                    minimum, maximum = 0.01, 10
                elif name in {
                    "follower_box_scale_width",
                    "follower_box_scale_height",
                    "follower_box_min_height_ratio",
                    "sticky_size_ratio_to_frame_width",
                    "sticky_translation_gaussian_mult",
                    "unsticky_translation_size_ratio",
                }:
                    minimum, maximum = 0.001, 100
                fields.append(
                    _field(
                        key,
                        label,
                        (
                            "int"
                            if isinstance(default, int) and name != "fixed_edge_rotation_angle"
                            else "number"
                        ),
                        default,
                        min=minimum,
                        max=maximum,
                    )
                )
            elif default is None or isinstance(default, str):
                fields.append(_field(key, label, "text", default))

    visit(baseline_config(ROOT_DIR)["rink"]["camera"])
    return fields


class OperatorWorkspace(CalibrationWorkspace):
    """One revision and transaction cover calibration and run preferences."""

    def _catalog(self, config):
        fields = [*super()._catalog(config), *RUN_FIELDS]
        present = {spec["key"] for spec in fields}
        return [*fields, *(spec for spec in camera_fields(config) if spec["key"] not in present)]

    def _settings(self, config, catalog):
        values = super()._settings(config, catalog)
        for spec in [*RUN_FIELDS, *camera_fields(config)]:
            value = _get(config, spec["key"], spec["default"])
            if value is None:
                value = spec["default"]
            if spec["kind"] == "text":
                value = "" if value is None else str(value)
            values[spec["key"]] = value
        analysis = selected_analysis(config)
        if _get(config, "hm_ui.run.analytics", "saved") != "custom":
            for target in ("pose", "jersey", "action"):
                values[f"pipeline.player-analytics.{target}.enable"] = target in analysis
        return values

    def stage_settings(self, changes, *, expected_revision=None):
        changes = dict(changes)
        enable_keys = {
            f"pipeline.player-analytics.{target}.enable": target
            for target in ("pose", "jersey", "action")
        }
        if enable_keys.keys() & changes.keys() and "hm_ui.run.analytics" not in changes:
            current = self._effective()
            if _get(current, "hm_ui.run.analytics", "saved") != "custom":
                analysis = selected_analysis(current)
                for key, target in enable_keys.items():
                    changes.setdefault(key, target in analysis)
            changes["hm_ui.run.analytics"] = "custom"
        for spec in [*RUN_FIELDS, *camera_fields(self._effective())]:
            key = spec["key"]
            if key not in changes or spec["kind"] != "text":
                continue
            value = changes[key]
            if key == "pipeline.player-analytics.pose.rate-hz":
                if value == "fps":
                    continue
                try:
                    cadence = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError("Pose cadence must be a positive rate or fps") from error
                if isinstance(value, bool) or not 0 < cadence <= 240:
                    raise ValueError("Pose cadence must be in (0, 240] or fps")
                changes[key] = cadence
                continue
            if value is None and key in {
                "video_out.output_height",
                "video_out.output_width",
                "video_out.bit_rate",
            }:
                value = ""
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and key in {"video_out.output_width", "video_out.output_height"}
            ):
                value = str(value)
            if not isinstance(value, str) or len(value) > 4096 or "\x00" in value:
                raise ValueError(f"{spec['label']} must be bounded text")
            if (
                key
                in {
                    "hm_ui.run.pose_config",
                    "hm_ui.run.pose_checkpoint",
                    "hm_ui.run.jersey_model_path",
                    "hm_ui.run.action_model_path",
                    "hm_ui.run.reid_onnx",
                }
                and value
                and not value.startswith(("https://", "http://"))
            ):
                path = Path(value).expanduser()
                changes[key] = str((path if path.is_absolute() else self.game_dir / path).resolve())
            if key in {"video_out.output_width", "video_out.output_height"}:
                if value.lower() == "auto" and key.endswith("width"):
                    value = "auto"
                elif not value.strip():
                    value = None
                else:
                    try:
                        value = int(value)
                    except ValueError as exc:
                        raise ValueError("Recording dimensions must be integer pixels") from exc
                    if not 2 <= value <= 16384 or value % 2:
                        raise ValueError(
                            "Recording dimensions must be even and between 2 and 16384"
                        )
                changes[key] = value
            elif key == "video_out.bit_rate":
                import re

                if value and not re.fullmatch(r"[1-9][0-9]*(?:[kKmM])?", value):
                    raise ValueError("Bitrate must be positive, optionally followed by k or M")
                changes[key] = value or None
            elif key == "video_out.youtube_stream_url" and not value.startswith(
                ("rtmp://", "rtmps://")
            ):
                raise ValueError("RTMP ingest URL must start with rtmp:// or rtmps://")
            elif (
                key == "video_out.spare_rtmp_url"
                and value
                and not value.startswith(("rtmp://", "rtmps://"))
            ):
                raise ValueError("Spare RTMP URL must start with rtmp:// or rtmps://")
            elif key in {"video_out.rtsp_program_mount", "video_out.rtsp_stitched_mount"}:
                if not value.startswith("/") or any(char in value for char in " \r\n?#"):
                    raise ValueError(
                        "RTSP mounts must begin with / and contain no whitespace, query or fragment"
                    )
        staged = super().stage_settings(changes, expected_revision=expected_revision)
        from hmlib.ui.calibration_workspace import _set

        candidate = self._effective()
        for key, value in staged["changes"].items():
            _set(candidate, key, value)
        selected_analysis(candidate)
        run = _get(candidate, "hm_ui.run", {})
        if run.get("pose_model") == "custom" and not all(
            run.get(key) for key in ("pose_config", "pose_checkpoint")
        ):
            raise ValueError("Custom pose requires a model config and checkpoint")
        for target in ("jersey", "action"):
            if run.get(f"{target}_model") == "custom" and not run.get(f"{target}_model_path"):
                raise ValueError(f"Custom {target} requires a compatible ONNX model path")
        return staged


def launch_configuration(game: Path, explicit_arguments: list[str] | None = None) -> dict:
    """Resolve launch inputs before translating saved choices into CLI arguments."""
    import argparse

    from hmlib.config import (
        ROOT_DIR,
        get_config,
        load_config_file,
        normalize_runtime_config,
        recursive_update,
        resolve_global_refs,
    )
    from hmlib.ui.workspace.files import read_config

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--root-dir", default=ROOT_DIR)
    parser.add_argument("--ignore-private-config", "--ignore_private_config", type=int, default=0)
    options, _ = parser.parse_known_args(explicit_arguments or [])
    game_config = load_config_file(
        root_dir=options.root_dir, config_type="games", config_name=game.name
    )
    if not options.ignore_private_config:
        game_config = recursive_update(game_config, read_config(game))
    camera = game_config.get("camera")
    if isinstance(camera, dict):
        camera = camera.get("name")
    config = get_config(
        root_dir=options.root_dir,
        camera=camera,
        rink=_get(game_config, "game.rink"),
        resolve_globals=False,
    )
    return resolve_global_refs(normalize_runtime_config(recursive_update(config, game_config)))


def has_explicit_backend(arguments: list[str] | None, target: str) -> bool:
    """An original backend choice replaces the complete saved engine selection."""
    options = {value.split("=", 1)[0] for value in (arguments or [])}
    return bool(
        options.intersection(
            {
                f"--{target}-onnx",
                f"--{target}-onnx-enable",
                f"--{target}-onnx-quantize-int8",
                f"--{target}-trt-enable",
                f"--{target}-trt-engine",
            }
        )
    )


def launch_arguments(
    game: Path,
    kind: str,
    *,
    explicit_arguments: list[str] | None = None,
    config: dict | None = None,
) -> list[str]:
    """Translate saved run choices; explicit invocation arguments follow these."""
    if config is None:
        config = launch_configuration(game, explicit_arguments)
    run = _get(config, "hm_ui.run", {})
    arguments = []
    if run.get("stitch_precision", "saved") != "saved":
        if run["stitch_precision"] not in {"uint8", "float16", "float32"}:
            raise ValueError("Unknown saved stitching precision")
        arguments += ["--config-override", f"stitching.dtype={run['stitch_precision']}"]
    if kind == "stitch" and run.get("memory_profile") == "low":
        arguments += ["--cache-size", "0", "--fp16-stitch", "--max-output-dimension", "1920"]
    if run.get("save_video") is False:
        arguments.append("--no-save-video")
    if "output_fps" in run and run["output_fps"] is not None:
        arguments += ["--output-fps", str(run["output_fps"])]
    if kind != "track":
        return arguments
    if "batch_size" in run:
        arguments += ["--batch-size", str(run["batch_size"])]
    if run.get("detector", "saved") != "saved":
        arguments += ["--detector-model", run["detector"]]
    analysis = selected_analysis(config)
    if analysis != "saved":
        for path in graph_paths(analysis):
            arguments += ["--config", str(path)]
        if "jersey" in analysis:
            arguments.append("--detect-jersey-numbers")
    for target, setting in (("detector", "precision"), ("pose", "pose_precision")):
        if has_explicit_backend(explicit_arguments, target):
            # An explicit backend replaces the whole saved engine choice. Do
            # this before validating a prepared INT8 cache, since that cache
            # is irrelevant when the caller supplies a different engine.
            continue
        precision = run.get(setting, "saved")
        if precision == "saved":
            continue
        if precision not in {
            "pytorch",
            "pytorch-fp16",
            "pytorch-bf16",
            "trt-fp32",
            "trt-fp16",
            "onnx",
            "onnx-int8",
        }:
            raise ValueError(f"Unknown saved {target} engine: {precision}")
        if precision.startswith("trt"):
            arguments += [
                f"--{target}-trt-enable",
                f"--{'no-' if precision == 'trt-fp32' else ''}{target}-trt-fp16",
            ]
        elif precision.startswith("onnx"):
            if precision == "onnx-int8":
                from hmlib.ui.model_preparation import prepared_onnx_arguments

                arguments += prepared_onnx_arguments(game, target, config=config)
                if target == "detector" and run.get("detector", "saved") == "saved":
                    arguments += ["--detector-model", "deployed"]
            else:
                arguments.append(f"--{target}-onnx-enable")
    return arguments


def prefer_explicit_launch_arguments(
    generated: list[str], explicit: list[str], kind: str
) -> list[str]:
    """Remove generated switches overridden by the invocation, including aliases.

    Merely appending the caller's flags fails for argparse mutually exclusive
    pairs such as --detector-trt-fp16/--no-detector-trt-fp16. Use their shared destinations so
    the original command retains the same meaning as a direct CLI invocation.
    """
    if not generated or not explicit:
        return generated
    if kind == "track":
        from hmlib.cli.hmtrack import make_parser

        parser = make_parser()
    elif kind == "stitch":
        from hmlib.cli.stitch import make_parser
        from hmlib.hm_opts import hm_opts

        parser = hm_opts.parser(parser=make_parser())
    else:
        raise ValueError("Pipeline kind must be track or stitch")
    options = parser._option_string_actions
    overridden = {
        options[token.split("=", 1)[0]].dest
        for token in explicit
        if token.split("=", 1)[0] in options
    }
    # These entries target distinct configuration keys, not one scalar option.
    # Keep all entries in order: unrelated keys coexist and later duplicate keys
    # retain the CLI override semantics of hm_opts.apply_config_overrides.
    overridden.discard("config_overrides")
    result, index = [], 0
    while index < len(generated):
        token = generated[index]
        action = options[token.split("=", 1)[0]]
        # Generated choices use only scalar values and ordinary boolean flags.
        if action.nargs not in (None, 0):
            raise ValueError(f"Unsupported generated launch argument: {token}")
        count = 1 if "=" in token or action.nargs == 0 else 2
        if action.dest not in overridden:
            result.extend(generated[index : index + count])
        index += count
    return result


PIPELINE_PREFERENCES = ("threaded", "queue_size", "max_concurrent")


def saved_pipeline_preferences(
    game_id: str | None,
    *,
    root_dir: str | Path | None = None,
    ignore_private_config: bool = False,
) -> dict:
    """Read explicit game preferences without mistaking baseline defaults for choices."""
    from hmlib.config import get_game_config_private, load_config_file, recursive_update

    if game_id is None:
        return {}
    config = load_config_file(root_dir=root_dir, config_type="games", config_name=game_id)
    if not ignore_private_config:
        config = recursive_update(config or {}, get_game_config_private(game_id=game_id) or {})
    pipeline = _get(config or {}, "aspen.pipeline", {})
    return {key: pipeline[key] for key in PIPELINE_PREFERENCES if key in pipeline}


def apply_pipeline_preferences(config: dict, preferences: dict) -> None:
    """Restore operator choices after loading a graph, before applying CLI overrides."""
    pipeline = config.setdefault("aspen", {}).setdefault("pipeline", {})
    for key in PIPELINE_PREFERENCES:
        if key in preferences:
            pipeline[key] = preferences[key]


def apply_engine_preferences(config: dict) -> None:
    """Clear saved graph engines before CLI applies the selected engine options."""
    apply_analytics_preferences(config)
    run = _get(config, "hm_ui.run", {})
    plugins = config.get("aspen", {}).get("plugins", {})
    for target, setting in (("detector", "precision"), ("pose", "pose_precision")):
        if run.get(setting, "saved") == "saved":
            continue
        factory = plugins.get(f"{target}_factory")
        if factory is None:
            continue
        params = factory.setdefault("params", {})
        for engine in ("trt", "onnx"):
            params.setdefault(engine, {})["enable"] = False


def inspect_pipeline(game: Path) -> dict:
    """Resolve the configured Aspen graph without loading any model weights."""
    from hmlib.config import load_yaml_files_ordered, recursive_update, resolve_global_refs

    config = OperatorWorkspace(game)._effective()
    from hmlib.ui.model_preparation import model_status

    run = _get(config, "hm_ui.run", {})
    graphs = [str(path) for path in graph_paths(selected_analysis(config))]
    from hmlib.config import ROOT_DIR, load_config_file
    from hmlib.ui.workspace.files import read_config

    saved = recursive_update(
        load_config_file(root_dir=ROOT_DIR, config_type="games", config_name=game.name) or {},
        read_config(game),
    )
    config = recursive_update(config, load_yaml_files_ordered(graphs))
    apply_pipeline_preferences(config, _get(saved, "aspen.pipeline", {}))
    config = resolve_global_refs(config)
    apply_engine_preferences(config)
    from hmlib.aspen.plugins.reid_plugin import configure_reid

    configure_reid(config, game, validate_weights=False)
    aspen = config["aspen"]
    for target, setting in (("detector", "precision"), ("pose", "pose_precision")):
        precision = run.get(setting, "saved")
        factory = aspen["plugins"].get(f"{target}_factory")
        if factory is None or precision == "saved":
            continue
        params = factory.setdefault("params", {})
        if precision.startswith("trt"):
            params.setdefault("trt", {}).update(enable=True, fp16=precision == "trt-fp16")
        elif precision.startswith("onnx"):
            params.setdefault("onnx", {})["enable"] = True
            if precision == "onnx-int8":
                status = model_status(game, target)
                params["onnx"].update(path=status.get("path"), preparation=status["state"])
        params["selected_precision"] = precision
    if run.get("detector", "saved") != "saved" and "detector_factory" in aspen["plugins"]:
        aspen["plugins"]["detector_factory"].setdefault("params", {})["selected_model"] = run[
            "detector"
        ]
    nodes = []

    def public(value):
        if isinstance(value, dict):
            return {
                key: (
                    "<configured>"
                    if any(
                        word in key.lower()
                        for word in ("password", "stream_key", "token", "secret")
                    )
                    else public(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [public(item) for item in value]
        return value

    for name, node in aspen["plugins"].items():
        nodes.append(
            {
                "name": name,
                "class": node["class"],
                "enabled": node.get("enabled", True),
                "depends": node.get("depends", []),
                "params": public(node.get("params", {})),
            }
        )
    pipeline = aspen.get("pipeline", {})
    return {
        "nodes": nodes,
        "graph": " + ".join(graphs),
        "mode": "threaded" if pipeline.get("threaded") else "sequential",
        "queue_size": pipeline.get("queue_size"),
        "max_concurrent": pipeline.get("max_concurrent"),
    }
