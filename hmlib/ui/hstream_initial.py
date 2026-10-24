"""Translate an initial hmstitch invocation into the shared native workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from hmlib.ui.hstream_launch import native_application_config, seconds, set_path, strip_options


def calibration_arguments(
    arguments: list[str], envelope: Path, runner: Path
) -> tuple[list[str], Path]:
    from hmlib.cli.stitch import make_parser
    from hmlib.config import get_config
    from hmlib.hm_opts import hm_opts

    if envelope.stat().st_size > 1024 * 1024:
        raise ValueError("Initial launch envelope is too large")
    request = json.loads(envelope.read_text())
    if request.get("version") != 1 or request.get("module") != "hmlib.cli.stitch":
        raise ValueError("Calibration launch requires an hmstitch envelope")
    original = request["arguments"]
    if not isinstance(original, list) or not all(
        isinstance(value, str) and "\0" not in value for value in original
    ):
        raise ValueError("Initial calibration arguments must be strings")
    cwd = Path(request["cwd"]).resolve(strict=True)
    parser = hm_opts.parser(make_parser())
    opts = parser.parse_args(original)
    names = {
        parser._option_string_actions[value.split("=", 1)[0]].dest
        for value in original
        if value.split("=", 1)[0] in parser._option_string_actions
    }
    game_id = request["game_id"]
    if opts.game_id and opts.game_id != game_id:
        raise ValueError("Initial calibration game does not match the desktop")
    transport = {
        "camera_ui",
        "game_id",
        "input_video",
        "root_dir",
        "ignore_private_config",
        "config_overrides",
        "start_frame_time",
        "max_time",
        "lfo",
        "rfo",
        "force",
        "clean",
        "configure_only",
        "output_video",
        "no_save_video",
        "no_audio",
        "show_image",
        "show_scaled",
    }
    translated = {
        name
        for name, paths in hm_opts.ARG_TO_CONFIG_MAP.items()
        if all(
            path.startswith(("stitching.", "rink.", "camera.", "video_out."))
            for path in ([paths] if isinstance(paths, str) else paths)
        )
    }
    unsupported = names - transport - translated
    if unsupported:
        raise ValueError(
            "These Python processing options do not apply to shared native calibration: "
            + ", ".join(sorted(unsupported))
            + ". Use --camera-ui=0 for that Python-only invocation."
        )
    config = get_config(
        game_id=game_id,
        ignore_private_config=bool(opts.ignore_private_config),
        resolve_globals=False,
    )
    hm_opts.apply_arg_config_overrides(config, opts, parser=parser, explicit_arg_names=names)
    hm_opts.apply_config_overrides(config, opts.config_overrides)
    if opts.input_video:
        paths = [(cwd / value).resolve(strict=True) for value in opts.input_video.split(",")]
        if len(paths) == 2 and all(path.is_file() for path in paths):
            set_path(
                config,
                "game.videos",
                {name: [str(path)] for name, path in zip(("left", "right"), paths)},
            )
        elif len(paths) != 1 or not paths[0].is_dir():
            raise ValueError(
                "Native calibration requires a left/right source pair or configured game directory"
            )
    if opts.lfo is not None or opts.rfo is not None:
        if opts.lfo is None or opts.rfo is None:
            raise ValueError("Manual synchronization requires both --lfo and --rfo")
        set_path(config, "game.stitching.frame_offsets", {"left": opts.lfo, "right": opts.rfo})
        set_path(config, "stitching.sync_method", "manual")
    if opts.output_video:
        set_path(config, "pipeline.sink5.output-file", str((cwd / opts.output_video).resolve()))
        arguments = strip_options(arguments, {"--enable-sinks"}) + [
            "--enable-sinks=ENCODE_STITCHED_FILE"
        ]
    if opts.no_save_video:
        from hmlib.ui.hstream_launch import parser as native_parser

        sinks = set(native_parser().parse_args(arguments).enable_sinks.split(","))
        sinks -= {"ENCODE_FILE", "ENCODE_STITCHED_FILE", "ENCODE_PROGRAM_4K_FILE"}
        arguments = strip_options(arguments, {"--enable-sinks"}) + [
            "--enable-sinks=" + ",".join(sorted(sinks or {"FAKE"}))
        ]
    if opts.no_audio:
        set_path(config, "pipeline.hmaudio.enable", False)
    if opts.start_frame_time:
        arguments = strip_options(arguments, {"--start-time", "-s"}) + [
            "--start-time=" + opts.start_frame_time
        ]
    if opts.max_time:
        duration = seconds(opts.max_time)
        if not duration.is_integer() or duration <= 0:
            raise ValueError("Native calibration time limits must be positive whole seconds")
        arguments = strip_options(arguments, {"--time-limit", "-t"}) + [
            "--time-limit=" + str(math.floor(duration))
        ]
    if opts.force:
        arguments.append("--force-reconfigure")
    if opts.clean:
        arguments.append("--clean")
    overlay = envelope.with_name("native-initial.yaml")
    overlay.write_text(yaml.safe_dump(config, sort_keys=False))
    if not any(value.split("=", 1)[0] in {"-c", "--config"} for value in arguments):
        arguments += ["-c", str(native_application_config(runner))]
    arguments += ["--settings-overlay=" + str(overlay)]
    return arguments, cwd
