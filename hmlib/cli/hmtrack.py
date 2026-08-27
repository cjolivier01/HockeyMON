import argparse
import copy
import datetime
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml

# from mmdet.apis import init_track_model
# from mmengine.config import Config
# from torch.nn.parallel import DistributedDataParallel as DDP
import hmlib
from hmlib.config import (
    get_clip_box,
    get_config,
    get_game_config_private,
    get_game_dir,
    get_nested_value,
    load_config_file,
    normalize_runtime_config,
    resolve_global_refs,
    set_nested_value,
)
from hmlib.hm_opts import _get_baseline_runtime_config, copy_opts, hm_opts
from hmlib.log import get_root_logger, logger
from hmlib.utils.path import (
    add_game_id_prefix_to_filename,
    add_prefix_to_filename,
    add_suffix_to_filename,
)
from hmlib.utils.pipeline import get_pipeline_item, update_pipeline_item

ROOT_DIR = os.path.dirname(os.path.abspath(hmlib.__file__))


def make_parser(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser("HockeyMON Tracking")
    parser = hm_opts.parser(parser)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Validate hmtrack CLI startup, report the active torch backend, and optionally "
            "resolve the game directory, then exit."
        ),
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help="Root directory")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--no-wide-start",
        default=False,
        action="store_true",
        help="Don't start with a tracking box of the entire input frame. Immediately track to player detections.",
    )
    parser.add_argument(
        "--no-rink-rotation",
        default=False,
        action="store_true",
        help="Don't do rink rotation.",
    )
    parser.add_argument(
        "--no-play-tracking",
        default=False,
        action="store_true",
        help="Don't do any postprocessing (i.e. play tracking) after basic player tracking.",
    )
    # Output video flag moved to hm_opts.parser
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # cam args
    parser.add_argument(
        "--adjust-exposure",
        default=None,
        type=float,
        help="Adjust overall exposure of all input images",
    )
    parser.add_argument(
        "--cam-ignore-largest",
        default=False,
        action="store_true",
        help="Remove the largest tracking box from the camera set (i.e. at Vallco, a ref is "
        "often right in front of the camera, but not enough of the ref is "
        "visible to note it as a ref)",
    )
    parser.add_argument(
        "--rink",
        default=None,
        type=str,
        help="rink name",
    )
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--track_thresh_low",
        type=float,
        default=0.1,
        help="tracking confidence threshold lower bound",
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--cvat-output",
        action="store_true",
        help="generate dataset data importable by cvat",
    )
    parser.add_argument(
        "--no-stitch",
        "--no-force-stitching",
        "--no_force_stitching",
        dest="no_force_stitching",
        action="store_true",
        help="force video stitching",
    )
    # Plotting, Profiling, and Camera Controller options moved to hm_opts.parser
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Additional YAML config file(s) to merge in order. "
            "Repeat --config to provide multiple files; later ones override earlier ones."
        ),
    )
    parser.add_argument(
        "--experiment-config",
        dest="experiment_config",
        type=str,
        default=None,
        help=(
            "YAML file describing multiple hmtrack variants to run on the same clip "
            "and combine (concat/tile) into a single output."
        ),
    )
    parser.add_argument(
        "--experiment-output",
        dest="experiment_output",
        type=str,
        default=None,
        help="Override the combined experiment output video path.",
    )
    parser.add_argument(
        "--experiment-mode",
        dest="experiment_mode",
        type=str,
        choices=["concat", "tile", "tiles", "grid"],
        default=None,
        help="Override the experiment combine mode (concat or tile/grid).",
    )
    parser.add_argument(
        "--test-size", type=str, default=None, help="WxH of test box size (format WxH)"
    )
    parser.add_argument(
        "--overlay-text",
        dest="overlay_text",
        type=str,
        default=None,
        help="Overlay custom text on the output video (applied via HmImageOverlays).",
    )
    parser.add_argument(
        "--overlay-text-origin",
        dest="overlay_text_origin",
        type=str,
        default=None,
        help="Overlay text origin as 'x,y' (pixels).",
    )
    parser.add_argument(
        "--overlay-text-color",
        dest="overlay_text_color",
        type=str,
        default=None,
        help="Overlay text color as 'r,g,b'.",
    )
    parser.add_argument(
        "--overlay-text-scale",
        dest="overlay_text_scale",
        type=float,
        default=None,
        help="Overlay text scale (overrides auto scale).",
    )
    parser.add_argument(
        "--overlay-text-thickness",
        dest="overlay_text_thickness",
        type=int,
        default=None,
        help="Overlay text thickness.",
    )
    parser.add_argument(
        "--overlay-text-max-lines",
        dest="overlay_text_max_lines",
        type=int,
        default=None,
        help="Maximum number of overlay text lines to draw.",
    )
    # Save frame dir moved to hm_opts.parser
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        type=str,
        default="tracking",
        help="Comma-separated task list (tracking)",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--min-box-area", type=float, default=100, help="filter out tiny boxes")
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    # Data I/O flags moved to hm_opts.parser
    # ONNX detector export/inference options moved to hm_opts.parser
    # TensorRT detector options moved to hm_opts.parser
    # ONNX detector options moved to hm_opts.parser
    # ONNX pose export/inference options moved to hm_opts.parser
    # TensorRT pose options moved to hm_opts.parser
    # ONNX pose options moved to hm_opts.parser
    # Audio-only and output-video moved to hm_opts.parser
    parser.add_argument("--checkpoint", type=str, default=None, help="Tracking checkpoint file")
    parser.add_argument("--detector", help="det checkpoint file")
    parser.add_argument("--reid", help="reid checkpoint file")

    # Pose args
    parser.add_argument("--pose-config", type=str, default=None, help="Pose config file")
    parser.add_argument("--pose-checkpoint", type=str, default=None, help="Pose checkpoint file")
    # Pose visualization args moved to hm_opts.parser
    parser.add_argument(
        "--debug-play-tracker", action="store_true", help="Print per-frame play boxes and counts"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply a temporal filter to smooth the pose estimation results. "
        "See also --smooth-filter-cfg.",
    )
    return hm_opts.finalize_parser(parser)


def _torch_backend_label() -> str:
    if getattr(torch.version, "hip", None):
        return "rocm:" + str(torch.version.hip)
    if getattr(torch.version, "cuda", None):
        return "cuda:" + str(torch.version.cuda)
    return "cpu"


def _exercise_preview_smoke(args: argparse.Namespace) -> list[str]:
    if not args.show_image and not args.show_youtube:
        return []

    from hmlib.ui.headless_preview import mask_stream_url
    from hmlib.ui.shower import Shower

    messages: list[str] = []
    shower = Shower(
        label="hmtrack-smoke-preview",
        show_scaled=args.show_scaled,
        max_size=2,
        enable_local_display=args.show_image,
        show_youtube=args.show_youtube,
        youtube_stream_url=args.youtube_stream_url,
        youtube_stream_key=args.youtube_stream_key,
        headless_preview_host=args.headless_preview_host or "0.0.0.0",
        headless_preview_port=int(args.headless_preview_port or 0),
        always_stream=bool(args.always_stream),
    )
    try:
        frame = torch.full((32, 48, 3), 127, dtype=torch.uint8)
        frame_count = 24 if args.show_youtube else 3
        for _ in range(frame_count):
            shower.show(frame)
            time.sleep(0.05)
        time.sleep(1.0 if args.show_youtube else 0.2)
        if args.show_image:
            if shower._headless_preview is not None:
                messages.append(
                    f"Headless preview OK. url=http://127.0.0.1:{shower._headless_preview.port}/"
                )
            else:
                messages.append("Local preview OK.")
        if args.show_youtube and shower._youtube_stream_url is not None:
            messages.append(
                "YouTube preview publish OK. " f"url={mask_stream_url(shower._youtube_stream_url)}"
            )
    finally:
        shower.close()
    return messages


def _run_smoke_test(args: argparse.Namespace) -> int:
    game_dir = None
    if args.game_id:
        try:
            game_dir = get_game_dir(args.game_id, assert_exists=False)
        except Exception:
            game_dir = None

    for message in _exercise_preview_smoke(args):
        print(message)
    print(
        "Smoke test OK. "
        f"backend={_torch_backend_label()} "
        f"cuda_available={torch.cuda.is_available()} "
        f"game_id={args.game_id} "
        f"game_dir={game_dir}"
    )
    return 0


def _arg_was_explicit(args: argparse.Namespace, name: str) -> bool:
    explicit = getattr(args, "explicit_arg_names", None)
    if explicit is None:
        return False
    return name in set(explicit)


_CONFIG_MISSING = object()


def _config_value(config: Dict[str, Any], path: str) -> Any:
    return get_nested_value(config, path, _CONFIG_MISSING)


def _config_value_is_default_or_missing(
    config: Dict[str, Any], baseline_config: Dict[str, Any], path: str
) -> bool:
    current = _config_value(config, path)
    if current is _CONFIG_MISSING:
        return True
    baseline = _config_value(baseline_config, path)
    return baseline is not _CONFIG_MISSING and current == baseline


def _plugin_value_follows_source_or_missing(
    config: Dict[str, Any], plugin_path: str, source_path: str
) -> bool:
    plugin_value = _config_value(config, plugin_path)
    if plugin_value is _CONFIG_MISSING:
        return True
    source_value = _config_value(config, source_path)
    return (source_value is not _CONFIG_MISSING and plugin_value == source_value) or (
        plugin_value == f"GLOBAL.{source_path}"
    )


def _config_override_was_explicit(args: argparse.Namespace, *config_keys: str) -> bool:
    overrides = args.config_overrides or []
    wanted = {str(key).strip() for key in config_keys if key}
    if not wanted:
        return False
    for override in overrides:
        if not isinstance(override, str):
            continue
        key = override.split("=", 1)[0].strip()
        if key in wanted:
            return True
    return False


def _override_value_is_global_link(value: Any, source_path: str) -> bool:
    return isinstance(value, str) and value.strip() == f"GLOBAL.{source_path}"


def _plugin_config_override_was_explicit(
    args: argparse.Namespace, plugin_path: str, source_path: str
) -> bool:
    overrides = args.config_overrides or []
    for override in overrides:
        if not isinstance(override, str):
            continue
        key, sep, raw_value = override.partition("=")
        if key.strip() != plugin_path:
            continue
        if not sep:
            return True
        if not _override_value_is_global_link(raw_value, source_path):
            return True
    return False


def _game_or_private_config_was_explicit(args: argparse.Namespace, *config_keys: str) -> bool:
    if args.game_id is None:
        return False

    game_config = load_config_file(config_type="games", config_name=str(args.game_id))
    private_config = {}
    if not bool(args.ignore_private_config):
        private_config = get_game_config_private(game_id=str(args.game_id))

    for config_key in config_keys:
        if not config_key:
            continue
        if _config_value(game_config, config_key) is not _CONFIG_MISSING:
            return True
        if _config_value(private_config, config_key) is not _CONFIG_MISSING:
            return True
    return False


def _game_or_private_plugin_config_was_explicit(
    args: argparse.Namespace, plugin_path: str, source_path: str
) -> bool:
    if args.game_id is None:
        return False

    game_config = load_config_file(config_type="games", config_name=str(args.game_id))
    private_config = {}
    if not bool(args.ignore_private_config):
        private_config = get_game_config_private(game_id=str(args.game_id))

    for config in (game_config, private_config):
        plugin_value = _config_value(config, plugin_path)
        if plugin_value is _CONFIG_MISSING:
            continue
        if not _override_value_is_global_link(plugin_value, source_path):
            return True
    return False


def _apply_single_lowmem_gpu_overrides(
    args: argparse.Namespace, game_config: Optional[Dict[str, Any]]
) -> None:
    print("Adjusting configuration for a single low-memory GPU environment...")
    args.cache_size = 0
    if not isinstance(game_config, dict):
        return

    baseline_config = _get_baseline_runtime_config()
    explicit_arg_names = set(getattr(args, "explicit_arg_names", []) or [])
    lowmem_max_output_width = 1920

    if (
        "fp16_stitch" not in explicit_arg_names
        and not _config_override_was_explicit(args, "stitching.dtype")
        and not _plugin_config_override_was_explicit(
            args, "aspen.plugins.stitching.params.dtype", "stitching.dtype"
        )
    ):
        can_override_dtype = (
            not _game_or_private_config_was_explicit(args, "stitching.dtype")
            and not _game_or_private_plugin_config_was_explicit(
                args, "aspen.plugins.stitching.params.dtype", "stitching.dtype"
            )
            and _config_value_is_default_or_missing(game_config, baseline_config, "stitching.dtype")
            and _plugin_value_follows_source_or_missing(
                game_config,
                "aspen.plugins.stitching.params.dtype",
                "stitching.dtype",
            )
        )
        if can_override_dtype:
            args.fp16_stitch = True
            set_nested_value(game_config, "stitching.dtype", "float16")
            set_nested_value(game_config, "aspen.plugins.stitching.params.dtype", "float16")

    if (
        "max_blend_levels" not in explicit_arg_names
        and not _config_override_was_explicit(args, "stitching.max_blend_levels")
        and not _plugin_config_override_was_explicit(
            args,
            "aspen.plugins.stitching.params.max_blend_levels",
            "stitching.max_blend_levels",
        )
    ):
        can_override_max_blend_levels = (
            not _game_or_private_config_was_explicit(args, "stitching.max_blend_levels")
            and not _game_or_private_plugin_config_was_explicit(
                args,
                "aspen.plugins.stitching.params.max_blend_levels",
                "stitching.max_blend_levels",
            )
            and _config_value_is_default_or_missing(
                game_config, baseline_config, "stitching.max_blend_levels"
            )
            and _plugin_value_follows_source_or_missing(
                game_config,
                "aspen.plugins.stitching.params.max_blend_levels",
                "stitching.max_blend_levels",
            )
        )
        if can_override_max_blend_levels:
            args.max_blend_levels = 5
            set_nested_value(game_config, "stitching.max_blend_levels", 5)
            set_nested_value(game_config, "aspen.plugins.stitching.params.max_blend_levels", 5)

    if (
        "minimize_blend" not in explicit_arg_names
        and "no_minimize_blend" not in explicit_arg_names
        and not _config_override_was_explicit(args, "stitching.minimize_blend")
        and not _plugin_config_override_was_explicit(
            args,
            "aspen.plugins.stitching.params.minimize_blend",
            "stitching.minimize_blend",
        )
    ):
        can_override_minimize_blend = (
            not _game_or_private_config_was_explicit(args, "stitching.minimize_blend")
            and not _game_or_private_plugin_config_was_explicit(
                args,
                "aspen.plugins.stitching.params.minimize_blend",
                "stitching.minimize_blend",
            )
            and _config_value_is_default_or_missing(
                game_config, baseline_config, "stitching.minimize_blend"
            )
            and _plugin_value_follows_source_or_missing(
                game_config,
                "aspen.plugins.stitching.params.minimize_blend",
                "stitching.minimize_blend",
            )
        )
        if can_override_minimize_blend:
            args.minimize_blend = 1
            set_nested_value(game_config, "stitching.minimize_blend", True)
            set_nested_value(game_config, "aspen.plugins.stitching.params.minimize_blend", True)

    if (
        "output_width" not in explicit_arg_names
        and "output_height" not in explicit_arg_names
        and not _config_override_was_explicit(
            args,
            "video_out.output_width",
            "video_out.output_height",
            "stitching.max_output_width",
        )
        and not _plugin_config_override_was_explicit(
            args,
            "aspen.plugins.stitching.params.max_output_width",
            "stitching.max_output_width",
        )
        and not _plugin_config_override_was_explicit(
            args,
            "aspen.plugins.video_out_prep.params.output_width",
            "video_out.output_width",
        )
    ):
        can_override_output_width = (
            not _game_or_private_config_was_explicit(
                args,
                "video_out.output_width",
                "video_out.output_height",
                "stitching.max_output_width",
            )
            and not _game_or_private_plugin_config_was_explicit(
                args,
                "aspen.plugins.stitching.params.max_output_width",
                "stitching.max_output_width",
            )
            and not _game_or_private_plugin_config_was_explicit(
                args,
                "aspen.plugins.video_out_prep.params.output_width",
                "video_out.output_width",
            )
            and _config_value_is_default_or_missing(
                game_config, baseline_config, "video_out.output_width"
            )
            and _config_value_is_default_or_missing(
                game_config, baseline_config, "video_out.output_height"
            )
            and _config_value_is_default_or_missing(
                game_config, baseline_config, "stitching.max_output_width"
            )
            and _plugin_value_follows_source_or_missing(
                game_config,
                "aspen.plugins.stitching.params.max_output_width",
                "stitching.max_output_width",
            )
            and _plugin_value_follows_source_or_missing(
                game_config,
                "aspen.plugins.video_out_prep.params.output_width",
                "video_out.output_width",
            )
        )
        if can_override_output_width:
            args.output_width = lowmem_max_output_width
            set_nested_value(game_config, "video_out.output_width", lowmem_max_output_width)
            set_nested_value(game_config, "stitching.max_output_width", lowmem_max_output_width)
            set_nested_value(
                game_config,
                "aspen.plugins.stitching.params.max_output_width",
                lowmem_max_output_width,
            )
            set_nested_value(
                game_config,
                "aspen.plugins.video_out_prep.params.output_width",
                lowmem_max_output_width,
            )

    if "aspen_max_concurrent" not in explicit_arg_names and not _config_override_was_explicit(
        args, "aspen.pipeline.max_concurrent"
    ):
        can_override_max_concurrent = not _game_or_private_config_was_explicit(
            args, "aspen.pipeline.max_concurrent"
        ) and _config_value_is_default_or_missing(
            game_config, baseline_config, "aspen.pipeline.max_concurrent"
        )
        if can_override_max_concurrent:
            args.aspen_max_concurrent = 1
            set_nested_value(game_config, "aspen.pipeline.max_concurrent", 1)

    if not _config_override_was_explicit(
        args, "stitching.cache_rotation_grid"
    ) and not _plugin_config_override_was_explicit(
        args,
        "aspen.plugins.stitching.params.cache_rotation_grid",
        "stitching.cache_rotation_grid",
    ):
        can_override_rotation_grid_cache = (
            not _game_or_private_config_was_explicit(args, "stitching.cache_rotation_grid")
            and not _game_or_private_plugin_config_was_explicit(
                args,
                "aspen.plugins.stitching.params.cache_rotation_grid",
                "stitching.cache_rotation_grid",
            )
            and _config_value_is_default_or_missing(
                game_config, baseline_config, "stitching.cache_rotation_grid"
            )
            and _plugin_value_follows_source_or_missing(
                game_config,
                "aspen.plugins.stitching.params.cache_rotation_grid",
                "stitching.cache_rotation_grid",
            )
        )
        if can_override_rotation_grid_cache:
            set_nested_value(game_config, "stitching.cache_rotation_grid", False)
            set_nested_value(
                game_config,
                "aspen.plugins.stitching.params.cache_rotation_grid",
                False,
            )


def _slugify_label(value: str) -> str:
    if value is None:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "variant"


def _parse_int_tuple(value: Any, length: int) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == length:
        return tuple(int(float(v)) for v in value)
    if isinstance(value, str):
        parts = [p for p in re.split(r"[x, ]+", value.strip()) if p]
        if len(parts) == length:
            return tuple(int(float(p)) for p in parts)
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _flatten_config_for_label(config: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for key, value in (config or {}).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(_flatten_config_for_label(value, path))
        else:
            items.append((path, value))
    return items


def _normalize_experiment_mode(mode: Optional[str]) -> str:
    if not mode:
        return "concat"
    mode_key = str(mode).strip().lower()
    if mode_key in ("tile", "tiles", "grid"):
        return "tile"
    return "concat"


def _build_overlay_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overlay_cfg: Dict[str, Any] = {}
    if not args.overlay_text:
        return overlay_cfg
    overlay_cfg["overlay_text"] = args.overlay_text
    origin = _parse_int_tuple(args.overlay_text_origin, 2)
    if origin is not None:
        overlay_cfg["overlay_text_origin"] = origin
    color = _parse_int_tuple(args.overlay_text_color, 3)
    if color is not None:
        overlay_cfg["overlay_text_color"] = color
    if args.overlay_text_scale is not None:
        overlay_cfg["overlay_text_scale"] = float(args.overlay_text_scale)
    if args.overlay_text_thickness is not None:
        overlay_cfg["overlay_text_thickness"] = int(args.overlay_text_thickness)
    if args.overlay_text_max_lines is not None:
        overlay_cfg["overlay_text_max_lines"] = int(args.overlay_text_max_lines)
    return overlay_cfg


def _enable_load_tracking_plugin(
    game_config: Dict[str, Any],
    prefer_detector: bool = True,
    disable_detector: bool = False,
) -> None:
    aspen = game_config.setdefault("aspen", {}) or {}
    plugins = aspen.setdefault("plugins", {}) or {}

    load_tracking = plugins.get("load_tracking")
    if not isinstance(load_tracking, dict):
        load_tracking = {}
    load_tracking["class"] = "hmlib.aspen.plugins.load_plugins.LoadTrackingPlugin"
    load_tracking["enabled"] = True
    depends = []
    if prefer_detector and isinstance(plugins.get("detector"), dict) and not disable_detector:
        depends = ["detector"]
    elif isinstance(plugins.get("image_prep"), dict):
        depends = ["image_prep"]
    if depends:
        load_tracking["depends"] = depends
    load_tracking.setdefault("params", {})
    plugins["load_tracking"] = load_tracking

    tracker = plugins.get("tracker")
    if isinstance(tracker, dict):
        tracker["enabled"] = False
        plugins["tracker"] = tracker

    save_tracking = plugins.get("save_tracking")
    if isinstance(save_tracking, dict):
        save_tracking["enabled"] = False
        plugins["save_tracking"] = save_tracking

    save_detections = plugins.get("save_detections")
    if isinstance(save_detections, dict):
        save_detections["enabled"] = False
        plugins["save_detections"] = save_detections

    if disable_detector:
        for key in (
            "detector",
            "detector_factory",
            "detector_join",
            "save_detections",
            "ice_boundaries_join",
            "ice_boundaries",
        ):
            plugin = plugins.get(key)
            if isinstance(plugin, dict):
                plugin["enabled"] = False
                plugins[key] = plugin

    camera_controller = plugins.get("camera_controller")
    if isinstance(camera_controller, dict):
        camera_controller["depends"] = ["load_tracking"]
        plugins["camera_controller"] = camera_controller

    aspen["plugins"] = plugins
    game_config["aspen"] = aspen


def _configure_play_tracker_cluster_cache(
    game_config: Dict[str, Any], centroids_path: Optional[str], save_centroids: bool = True
) -> None:
    if not centroids_path:
        return
    aspen = game_config.setdefault("aspen", {}) or {}
    plugins = aspen.setdefault("plugins", {}) or {}
    play_tracker = plugins.get("play_tracker")
    if not isinstance(play_tracker, dict):
        return
    params = play_tracker.setdefault("params", {}) or {}
    params["cluster_centroids_path"] = centroids_path
    params["save_cluster_centroids"] = bool(save_centroids)
    play_tracker["params"] = params
    plugins["play_tracker"] = play_tracker
    aspen["plugins"] = plugins
    game_config["aspen"] = aspen


def _inject_overlay_text(video_out_pipeline: Any, overlay_cfg: Dict[str, Any]) -> Any:
    if not overlay_cfg or not overlay_cfg.get("overlay_text"):
        return video_out_pipeline
    if not isinstance(video_out_pipeline, list):
        return video_out_pipeline
    for stage in video_out_pipeline:
        if isinstance(stage, dict) and stage.get("type") == "HmImageOverlays":
            stage.update(overlay_cfg)
            return video_out_pipeline
    overlay_stage = {"type": "HmImageOverlays", **overlay_cfg}
    for idx, stage in enumerate(video_out_pipeline):
        if isinstance(stage, dict) and stage.get("type") == "HmMakeVisibleImage":
            video_out_pipeline.insert(idx, overlay_stage)
            return video_out_pipeline
    video_out_pipeline.append(overlay_stage)
    return video_out_pipeline


def _parse_experiment_overlay_spec(
    overlay_spec: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not overlay_spec:
        return {}, {"show_params": True, "prefix": None}
    if overlay_spec.get("enabled") is False:
        return {}, {"show_params": False, "prefix": None}
    overlay_cfg: Dict[str, Any] = {}
    label_cfg: Dict[str, Any] = {
        "show_params": overlay_spec.get("show_params", True),
        "prefix": overlay_spec.get("prefix"),
    }
    origin = _parse_int_tuple(overlay_spec.get("origin") or overlay_spec.get("position"), 2)
    if origin is not None:
        overlay_cfg["overlay_text_origin"] = origin
    color = _parse_int_tuple(overlay_spec.get("color"), 3)
    if color is not None:
        overlay_cfg["overlay_text_color"] = color
    scale = _parse_float(overlay_spec.get("scale"))
    if scale is not None:
        overlay_cfg["overlay_text_scale"] = scale
    thickness = overlay_spec.get("thickness")
    if thickness is not None:
        overlay_cfg["overlay_text_thickness"] = int(thickness)
    max_lines = overlay_spec.get("max_lines")
    if max_lines is not None:
        overlay_cfg["overlay_text_max_lines"] = int(max_lines)
    return overlay_cfg, label_cfg


def _build_variant_label_text(
    name: str,
    variant_cfg: Optional[Dict[str, Any]],
    label_cfg: Dict[str, Any],
    explicit_text: Optional[str] = None,
) -> str:
    if explicit_text:
        return str(explicit_text)
    lines: List[str] = []
    prefix = label_cfg.get("prefix")
    if prefix is not None:
        lines.append(f"{prefix}{name}")
    else:
        lines.append(str(name))
    if label_cfg.get("show_params", True) and variant_cfg:
        for key, value in _flatten_config_for_label(variant_cfg):
            lines.append(f"{key}={value}")
    return "\n".join(lines)


def _apply_clip_cfg(args: argparse.Namespace, clip_cfg: Optional[Dict[str, Any]]) -> None:
    if not clip_cfg:
        return
    start_time = clip_cfg.get("start_time") or clip_cfg.get("start_frame_time")
    if start_time:
        args.start_frame_time = str(start_time)
    if "start_frame" in clip_cfg and clip_cfg["start_frame"] is not None:
        args.start_frame = int(clip_cfg["start_frame"])
    duration = clip_cfg.get("duration") or clip_cfg.get("max_time")
    if duration:
        args.max_time = str(duration)
    if "max_frames" in clip_cfg and clip_cfg["max_frames"] is not None:
        args.max_frames = int(clip_cfg["max_frames"])


def _resolve_base_output_path(game_config: Dict[str, Any], work_dir: str) -> str:
    out_path = get_nested_value(
        game_config, "aspen.plugins.video_out.params.output_video_path", default_value=None
    )
    if not out_path:
        out_path = get_nested_value(game_config, "video_out.output_video_path", None)
    if not out_path:
        out_path = get_nested_value(game_config, "aspen.video_out.output_video_path", None)
    if not out_path or "/" not in str(out_path):
        out_path = os.path.join(work_dir, out_path or "tracking_output.mkv")
    return str(out_path)


def _with_audio_output_path(path: str) -> str:
    """Return a sibling output path that makes it obvious audio is present.

    Historically hmtrack produced a video-only MKV under output_workdirs and then
    wrote an MP4-with-audio artifact. When muxing audio directly, prefer an MP4
    output for compatibility and faststart behavior.
    """
    if not path:
        return path
    p = Path(str(path))
    if not p.stem.endswith("-with-audio"):
        p = Path(add_suffix_to_filename(p, "-with-audio"))
    if p.suffix.lower() == ".mkv":
        p = p.with_suffix(".mp4")
    return str(p)


def _resolve_mux_audio_source(
    input_av_files: Any,
) -> Tuple[str, Optional[Any]]:
    """Select an audio source file suitable for ffmpeg muxing.

    Returns (audio_source_path, temp_audio_file_handle). The temp handle must
    be kept alive until muxing completes if present.
    """
    from hmlib.audio import concatenate_audio
    from hmlib.stitching.synchronize import synchronize_by_audio

    temp_audio_file = None
    audio_source = None
    if isinstance(input_av_files, dict) or (
        isinstance(input_av_files, list) and len(input_av_files) == 2
    ):
        if isinstance(input_av_files, dict):
            left_files = list(input_av_files.get("left") or [])
            right_files = list(input_av_files.get("right") or [])
            if not left_files or not right_files:
                raise ValueError("Expected input_av_files dict with non-empty 'left' and 'right'.")
            left_sync = left_files[0]
            right_sync = right_files[0]
        else:
            left_files = [str(input_av_files[0])]
            right_files = [str(input_av_files[1])]
            left_sync = left_files[0]
            right_sync = right_files[0]

        lfo, rfo = synchronize_by_audio(left_sync, right_sync)
        if lfo == 0:
            chosen = left_files
        elif rfo == 0:
            chosen = right_files
        else:
            raise RuntimeError(f"Expected one frame offset to be zero; got {lfo=} {rfo=}.")

        if len(chosen) > 1:
            temp_audio_file = concatenate_audio(chosen)
            if temp_audio_file is None:
                raise RuntimeError("Failed to concatenate audio for muxing.")
            audio_source = temp_audio_file.name
        else:
            audio_source = chosen[0]
    else:
        if isinstance(input_av_files, list):
            if len(input_av_files) != 1:
                raise ValueError(
                    f"Expected a single input AV file when not stitching; got {len(input_av_files)}."
                )
            audio_source = str(input_av_files[0])
        else:
            audio_source = str(input_av_files)

    if not audio_source:
        raise RuntimeError("Could not resolve an audio source for muxing.")
    return str(audio_source), temp_audio_file


def _ensure_three_channel(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3:
        raise AssertionError(f"Expected frame tensor HxWxC, got shape {frame.shape}")
    if frame.shape[-1] == 3:
        return frame
    if frame.shape[-1] == 4:
        return frame[:, :, :3]
    if frame.shape[-1] == 1:
        return frame.repeat(1, 1, 3)
    raise AssertionError(f"Unsupported channel count: {frame.shape[-1]}")


def _frame_to_tensor(frame: Any) -> torch.Tensor:
    from hmlib.utils.image import make_channels_last

    if isinstance(frame, torch.Tensor):
        tensor = frame
    else:
        tensor = torch.from_numpy(frame)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
        tensor = make_channels_last(tensor)
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)
    return _ensure_three_channel(tensor)


def _resize_letterbox(frame: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
    from hmlib.utils.image import (
        image_height,
        image_width,
        make_channels_first,
        make_channels_last,
        resize_image,
    )

    h = int(image_height(frame))
    w = int(image_width(frame))
    if h == target_h and w == target_w:
        return frame
    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = resize_image(frame, new_width=new_w, new_height=new_h)
    if resized.ndim == 3 and resized.shape[0] in (1, 3, 4):
        resized = make_channels_last(resized)
    pad_w = max(0, target_w - new_w)
    pad_h = max(0, target_h - new_h)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if pad_w == 0 and pad_h == 0:
        return resized
    cf = make_channels_first(resized)
    cf = F.pad(cf, [pad_left, pad_right, pad_top, pad_bottom], "constant", 0)
    return make_channels_last(cf)


def _default_codec_for_output(path: str) -> str:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".mp4":
        return "mp4v"
    return "XVID"


def _combine_videos_concat(
    video_paths: List[str],
    output_path: str,
    codec: Optional[str],
    bit_rate: Optional[int],
) -> None:
    from hmlib.video.ffmpeg import BasicVideoInfo
    from hmlib.video.video_stream import VideoStreamReader, create_output_video_stream

    if not video_paths:
        raise ValueError("No videos provided for concatenation.")
    infos = [BasicVideoInfo(p) for p in video_paths]
    fps = infos[0].fps
    width = infos[0].width
    height = infos[0].height
    for info in infos[1:]:
        if abs(info.fps - fps) > 0.01:
            logger.warning("Concat FPS mismatch: %.2f vs %.2f", fps, info.fps)
    output_codec = codec or _default_codec_for_output(output_path)
    writer = create_output_video_stream(
        filename=output_path,
        fps=fps,
        width=width,
        height=height,
        codec=output_codec,
        device=torch.device("cpu"),
        bit_rate=int(bit_rate) if bit_rate else int(55e6),
    )
    try:
        for path, info in zip(video_paths, infos):
            reader = VideoStreamReader(path, type="cv2", batch_size=1, device=torch.device("cpu"))
            iterator = iter(reader)
            while True:
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                frame_tensor = _frame_to_tensor(frame)
                if info.width != width or info.height != height:
                    frame_tensor = _resize_letterbox(frame_tensor, width, height)
                writer.write(frame_tensor)
            reader.close()
    finally:
        writer.close()


def _combine_videos_tile(
    video_paths: List[str],
    output_path: str,
    codec: Optional[str],
    bit_rate: Optional[int],
    rows: int,
    cols: int,
    padding: int = 0,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    from hmlib.video.ffmpeg import BasicVideoInfo
    from hmlib.video.video_stream import VideoStreamReader, create_output_video_stream

    if not video_paths:
        raise ValueError("No videos provided for tiling.")
    infos = [BasicVideoInfo(p) for p in video_paths]
    fps = infos[0].fps
    cell_w = max(info.width for info in infos)
    cell_h = max(info.height for info in infos)
    for info in infos[1:]:
        if abs(info.fps - fps) > 0.01:
            logger.warning("Tile FPS mismatch: %.2f vs %.2f", fps, info.fps)
    out_w = cols * cell_w + max(0, cols - 1) * padding
    out_h = rows * cell_h + max(0, rows - 1) * padding
    output_codec = codec or _default_codec_for_output(output_path)
    writer = create_output_video_stream(
        filename=output_path,
        fps=fps,
        width=out_w,
        height=out_h,
        codec=output_codec,
        device=torch.device("cpu"),
        bit_rate=int(bit_rate) if bit_rate else int(55e6),
    )
    readers = [
        VideoStreamReader(path, type="cv2", batch_size=1, device=torch.device("cpu"))
        for path in video_paths
    ]
    iters = [iter(r) for r in readers]
    done = [False] * len(readers)
    bg_color = torch.tensor(background, dtype=torch.uint8).view(1, 1, 3)
    try:
        while True:
            if all(done):
                break
            frames: List[Optional[torch.Tensor]] = []
            any_frame = False
            for idx, it in enumerate(iters):
                if done[idx]:
                    frames.append(None)
                    continue
                try:
                    frame = next(it)
                except StopIteration:
                    done[idx] = True
                    frames.append(None)
                    continue
                any_frame = True
                frames.append(_frame_to_tensor(frame))
            if not any_frame:
                break
            canvas = bg_color.repeat(out_h, out_w, 1)
            for idx in range(rows * cols):
                r = idx // cols
                c = idx % cols
                y0 = r * (cell_h + padding)
                x0 = c * (cell_w + padding)
                if idx >= len(frames) or frames[idx] is None:
                    continue
                tile = _resize_letterbox(frames[idx], cell_w, cell_h)
                canvas[y0 : y0 + cell_h, x0 : x0 + cell_w, :] = tile
            writer.write(canvas)
    finally:
        for reader in readers:
            reader.close()
        writer.close()


def _run_experiment(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    base_game_config: Dict[str, Any],
) -> int:
    if not args.experiment_config:
        raise ValueError("experiment_config path missing")
    with open(args.experiment_config, "r", encoding="utf-8") as f:
        raw_spec = yaml.safe_load(f) or {}
    if isinstance(raw_spec, dict) and "experiment" in raw_spec:
        spec = raw_spec.get("experiment") or {}
    else:
        spec = raw_spec
    if isinstance(spec, list):
        spec = {"variants": spec}
    variants = spec.get("variants") if isinstance(spec, dict) else None
    if not variants:
        raise ValueError("Experiment config must define a non-empty 'variants' list.")

    exp_name = spec.get("name") if isinstance(spec, dict) else None
    if not exp_name:
        exp_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    clip_cfg = spec.get("clip") if isinstance(spec, dict) else None
    output_cfg = spec.get("output") if isinstance(spec, dict) else {}
    overlay_spec = spec.get("overlay") if isinstance(spec, dict) else None
    overlay_cfg_base, label_cfg = _parse_experiment_overlay_spec(overlay_spec)
    reuse_tracking_default = len(variants) > 1
    reuse_tracking_cfg = (
        spec.get("reuse_tracking", reuse_tracking_default) if isinstance(spec, dict) else False
    )
    reuse_tracking = False
    skip_detector = True
    reuse_clusters = False
    cluster_centroids_path: Optional[str] = None
    if isinstance(reuse_tracking_cfg, dict):
        reuse_tracking = bool(reuse_tracking_cfg.get("enabled", True))
        skip_detector = bool(reuse_tracking_cfg.get("skip_detector", True))
        reuse_clusters = bool(reuse_tracking_cfg.get("cluster_centroids", reuse_tracking))
        cluster_centroids_path = reuse_tracking_cfg.get("cluster_centroids_path")
    else:
        reuse_tracking = bool(reuse_tracking_cfg)
        skip_detector = True
        reuse_clusters = bool(reuse_tracking)

    exp_mode = _normalize_experiment_mode(args.experiment_mode or output_cfg.get("mode"))
    exp_dir = os.path.join(".", "output_workdirs", args.game_id, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    if reuse_clusters and not cluster_centroids_path:
        cluster_centroids_path = os.path.join(exp_dir, "cluster_centroids.csv")
    combined_output = args.experiment_output or output_cfg.get("path")
    if not combined_output:
        combined_output = os.path.join(exp_dir, f"{exp_name}_{exp_mode}.mkv")
    combined_output = str(combined_output)

    # Save resolved experiment spec for reference.
    try:
        exp_yaml_path = os.path.join(exp_dir, f"{exp_name}.yaml")
        with open(exp_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
    except Exception:
        pass

    variant_outputs: List[str] = []
    first_tracking_csv: Optional[str] = None

    for idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            variant = {"name": f"variant_{idx+1}", "config": {}}
        name = variant.get("name") or f"variant_{idx+1}"
        safe_label = _slugify_label(name)

        variant_args = copy_opts(src=args, dest=argparse.Namespace(), parser=parser)
        variant_explicit_arg_names = set(getattr(args, "explicit_arg_names", []) or [])
        variant_args.experiment_config = None
        variant_args.experiment_output = None
        variant_args.experiment_mode = None

        # Apply clip settings (global + per-variant overrides).
        _apply_clip_cfg(variant_args, clip_cfg)
        _apply_clip_cfg(variant_args, variant.get("clip"))

        # Apply any direct arg overrides if provided.
        arg_overrides = variant.get("args") or {}
        if isinstance(arg_overrides, dict):
            for key, value in arg_overrides.items():
                if hasattr(variant_args, key):
                    setattr(variant_args, key, value)
                    variant_explicit_arg_names.add(str(key))
                else:
                    logger.warning("Unknown hmtrack arg in experiment: %s", key)

        # Build per-variant config.
        from hmlib.config import recursive_update

        variant_config = copy.deepcopy(base_game_config)
        variant_cfg_patch = variant.get("config") or {}
        if isinstance(variant_cfg_patch, dict):
            recursive_update(variant_config, variant_cfg_patch)
        if reuse_clusters and cluster_centroids_path:
            _configure_play_tracker_cluster_cache(
                variant_config, cluster_centroids_path, save_centroids=True
            )

        # Overlay text
        overlay_cfg = dict(overlay_cfg_base)
        overlay_text = _build_variant_label_text(
            name=name,
            variant_cfg=variant_cfg_patch if isinstance(variant_cfg_patch, dict) else None,
            label_cfg=label_cfg,
            explicit_text=variant.get("overlay_text"),
        )
        if overlay_text:
            overlay_cfg["overlay_text"] = overlay_text
            variant_args.overlay_text = overlay_text
            if "overlay_text_origin" in overlay_cfg:
                variant_args.overlay_text_origin = overlay_cfg["overlay_text_origin"]
            if "overlay_text_color" in overlay_cfg:
                variant_args.overlay_text_color = overlay_cfg["overlay_text_color"]
            if "overlay_text_scale" in overlay_cfg:
                variant_args.overlay_text_scale = overlay_cfg["overlay_text_scale"]
            if "overlay_text_thickness" in overlay_cfg:
                variant_args.overlay_text_thickness = overlay_cfg["overlay_text_thickness"]
            if "overlay_text_max_lines" in overlay_cfg:
                variant_args.overlay_text_max_lines = overlay_cfg["overlay_text_max_lines"]

        if reuse_tracking and idx > 0:
            if first_tracking_csv and not getattr(variant_args, "input_tracking_data", None):
                variant_args.input_tracking_data = first_tracking_csv
            if getattr(variant_args, "input_tracking_data", None):
                _enable_load_tracking_plugin(
                    variant_config,
                    prefer_detector=not skip_detector,
                    disable_detector=skip_detector,
                )

        normalize_runtime_config(variant_config)
        variant_args.explicit_arg_names = variant_explicit_arg_names
        hm_opts.apply_arg_config_overrides(
            variant_config,
            variant_args,
            parser=parser,
            explicit_arg_names=variant_explicit_arg_names,
        )
        variant_config = resolve_global_refs(variant_config)

        variant_args.output_label = safe_label
        variant_config["initial_args"] = vars(variant_args)
        variant_args.game_config = variant_config

        # Set up task flags
        variant_args.tracking = False
        tokens = variant_args.tasks.split(",")
        for t in tokens:
            setattr(variant_args, t, True)

        variant_args = configure_model(config=variant_args.game_config, args=variant_args)

        if getattr(variant_args, "smoke_test", False):
            logger.info("Smoke test requested; skipping experiment execution.")
            return 0

        if variant_args.game_id:
            num_gpus = 1
        else:
            if isinstance(variant_args.gpus, str):
                variant_args.gpus = [int(g) for g in variant_args.gpus.split(",")]
            num_gpus = len(variant_args.gpus) if variant_args.gpus else 0
            num_gpus = min(num_gpus, torch.cuda.device_count())

        _main(variant_args, num_gpus)

        work_dir = os.path.join(".", "output_workdirs", variant_args.game_id)
        base_out = _resolve_base_output_path(variant_config, work_dir)
        out_path = str(add_prefix_to_filename(base_out, safe_label))
        if os.path.exists(out_path):
            variant_outputs.append(out_path)
        else:
            logger.warning("Expected output not found for variant '%s': %s", name, out_path)

        if reuse_tracking and idx == 0:
            tracking_csv = os.path.join(work_dir, f"{safe_label}_tracking.csv")
            if os.path.exists(tracking_csv):
                first_tracking_csv = tracking_csv
            else:
                logger.warning("Expected tracking CSV not found: %s", tracking_csv)

        if reuse_tracking and idx == 0 and not first_tracking_csv:
            logger.warning(
                "reuse_tracking requested but no tracking CSV was produced; subsequent runs will not reuse tracks."
            )

    if not variant_outputs:
        raise ValueError("No variant outputs were produced; cannot combine.")

    if exp_mode == "concat":
        _combine_videos_concat(
            variant_outputs,
            combined_output,
            codec=output_cfg.get("codec"),
            bit_rate=output_cfg.get("bit_rate"),
        )
    else:
        tile_cfg = output_cfg.get("tile") or {}
        n = len(variant_outputs)
        cols = int(tile_cfg.get("cols") or math.ceil(math.sqrt(n)))
        rows = int(tile_cfg.get("rows") or math.ceil(n / cols))
        padding = int(tile_cfg.get("padding") or 0)
        bg_tuple = _parse_int_tuple(tile_cfg.get("background"), 3) or (0, 0, 0)
        _combine_videos_tile(
            variant_outputs,
            combined_output,
            codec=output_cfg.get("codec"),
            bit_rate=output_cfg.get("bit_rate"),
            rows=rows,
            cols=cols,
            padding=padding,
            background=bg_tuple,
        )

    logger.info("Experiment combined output saved to %s", combined_output)
    return 0


def set_torch_multiprocessing_use_filesystem():
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")


def to_32bit_mul(val):
    return int((val + 31)) & ~31


MAP_ARGS_TO_YAML = {
    "tracker": "model.tracker.type",
}


def update_from_args(args, arg_name, config, noset_value: any = None):
    if not hasattr(args, arg_name):
        return
    set_nested_value(
        dct=config,
        key_str="model.tracker.type",
        set_to=args.tracker,
        noset_value=noset_value,
    )


def configure_model(config: dict, args: argparse.Namespace):
    update_from_args(args, "tracker", config)
    return args


def find_stitched_file(dir_name: str, game_id: str):
    # exts = ["mp4", "mkv", "avi"]
    # basenames = [
    #     # "stitched_output",
    #     "stitched_output-with-audio",
    #     # "stitched_output-" + game_id,
    #     # "stitched_output-with-audio-" + game_id,
    # ]
    # for basename in basenames:
    #     for ext in exts:
    #         path = os.path.join(dir_name, basename + "." + ext)
    #         if os.path.exists(path):
    #             return path
    return None


def is_stitching(input_video: str) -> bool:
    if not input_video:
        raise AttributeError("No valid input video specified")
    input_video_files = input_video.split(",")
    return len(input_video_files) == 2 or os.path.isdir(input_video)


class _StitchRotationController:
    def __init__(self, game_config: Optional[Dict[str, Any]] = None) -> None:
        self._config = game_config
        self._value: Optional[float] = None

    def get_post_stitch_rotate_degrees(self) -> Optional[float]:
        if isinstance(self._config, dict):
            try:
                val = get_nested_value(self._config, "stitching.post_stitch_rotate_degrees", None)
                if val is not None:
                    return float(val)
            except Exception:
                pass
        return self._value

    def set_post_stitch_rotate_degrees(self, degrees: Optional[float]) -> None:
        self._value = degrees
        if isinstance(self._config, dict):
            try:
                set_nested_value(self._config, "stitching.post_stitch_rotate_degrees", degrees)
            except Exception:
                pass


def _main(args, num_gpu):
    from mmcv.transforms import Compose

    import hmlib.hm_transforms  # noqa: F401
    import hmlib.tracking_utils.segm_boundaries  # noqa: F401
    import hmlib.transforms  # noqa: F401
    from hmlib.camera.camera import should_unsharp_mask_camera
    from hmlib.datasets.dataframe import find_latest_dataframe_file
    from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
    from hmlib.datasets.dataset.multi_dataset import MultiDatasetWrapper
    from hmlib.datasets.dataset.stitching_dataloader2 import MultiDataLoaderWrapper, StitchDataset
    from hmlib.orientation import configure_game_videos
    from hmlib.stitching.configure_stitching import configure_video_stitching
    from hmlib.tasks.tracking import run_mmtrack
    from hmlib.utils.gpu import select_gpus
    from hmlib.utils.progress_bar import ProgressBar, ScrollOutput
    from hmlib.video.ffmpeg import BasicVideoInfo
    from hmlib.video.video_stream import time_to_frame

    dataloader = None
    postprocessor = None
    mux_audio_temp_file = None
    opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    try:

        if args.gpus and isinstance(args.gpus, str):
            args.gpus = [int(i) for i in args.gpus.split(",")]

        # set environment variables for distributed training
        cudnn.benchmark = True

        game_config = args.game_config

        dataloader = MultiDatasetWrapper(forgive_missing_attributes=["fps"])

        if args.output_fps is None:
            args.output_fps = get_nested_value(game_config, "camera.output-fps")

        if args.lfo is None and args.rfo is None:
            offsets = get_nested_value(game_config, "stitching.offsets")
            if offsets:
                args.lfo = offsets[0]
                if len(offsets) == 1:
                    args.rfo = 0.0
                else:
                    assert len(offsets) == 2
                    args.rfo = offsets[1]
                if args.lfo < 0:
                    args.rfo += -args.lfo
                    args.lfo = 0.0
                assert args.lfo >= 0 and args.rfo >= 0

        model = None

        # cmdline overrides
        if args.camera_name:
            set_nested_value(game_config, "camera.name", args.camera_name)
        else:
            args.camera_name = get_nested_value(game_config, "camera.name")
        if args.unsharp_mask is None and should_unsharp_mask_camera(args.camera_name):
            args.unsharp_mask = 1

        # Derived camera args (former DefaultArguments)
        # Crop output image unless explicitly disabled via CLI.
        # Prefer rink.tracking.cam_ignore_largest when CLI did not override.
        if not getattr(args, "cam_ignore_largest", False):
            args.cam_ignore_largest = get_nested_value(
                game_config, "rink.tracking.cam_ignore_largest", True
            )
        # Map plotting convenience flag to per-frame tracking overlays.
        args.plot_individual_player_tracking = bool(getattr(args, "plot_tracking", False))
        if args.plot_individual_player_tracking:
            args.plot_boundaries = True

        # See if gameid is in videos
        if not args.input_video and args.game_id:
            game_video_dir = get_game_dir(args.game_id)
            if game_video_dir:
                # TODO: also look for avi and mp4 files
                if not args.no_force_stitching:
                    args.input_video = game_video_dir
                else:
                    pre_stitched_file_name = find_stitched_file(
                        dir_name=game_video_dir, game_id=args.game_id
                    )
                    if pre_stitched_file_name and os.path.exists(pre_stitched_file_name):
                        args.input_video = pre_stitched_file_name
                    else:
                        args.input_video = game_video_dir

        results_folder = os.path.join(".", "output_workdirs", args.game_id)
        os.makedirs(results_folder, exist_ok=True)
        args.work_dir = results_folder
        try:
            args.game_dir = get_game_dir(args.game_id, assert_exists=False)
        except Exception:
            args.game_dir = None

        tracking_data_path = getattr(
            args, "input_tracking_data", None
        ) or find_latest_dataframe_file(args.game_dir, "tracking")
        detection_data_path = getattr(
            args, "input_detection_data", None
        ) or find_latest_dataframe_file(args.game_dir, "detections")
        pose_data_path = getattr(args, "input_pose_data", None) or find_latest_dataframe_file(
            args.game_dir, "pose"
        )
        action_data_path = find_latest_dataframe_file(args.game_dir, "actions")
        args.tracking_data_path = tracking_data_path
        args.detection_data_path = detection_data_path
        args.pose_data_path = pose_data_path
        args.action_data_path = action_data_path

        # Initialize lightweight profiler and attach to args for downstream use
        try:
            from hmlib.utils.profiler import build_profiler_from_args

            default_prof_dir = os.path.join(results_folder, "profiler")
            profiler = build_profiler_from_args(args, save_dir_fallback=default_prof_dir)
        except Exception:
            profiler = None
        setattr(args, "profiler", profiler)

        using_precalculated_tracking = bool(tracking_data_path)
        using_precalculated_detections = bool(detection_data_path)
        # using_precalculated_pose = bool(pose_data_path)

        actual_device_count = torch.cuda.device_count()
        if not actual_device_count:
            raise Exception("At leats one GPU is required for this application")
        while len(args.gpus) > actual_device_count:
            del args.gpus[-1]

        gpus, is_single_lowmem_gpu, gpu_allocator = select_gpus(
            allowed_gpus=args.gpus,
            is_stitching=is_stitching(args.input_video),
            # is_multipose=args.multi_pose,
            is_detecting=not using_precalculated_tracking and not using_precalculated_detections,
            stitch_with_fastest=not args.detect_jersey_numbers,
        )
        # Expose per-role devices to downstream components (Aspen plugins, postprocessor)
        args.camera_device = gpus.get("camera")
        args.encoder_device = gpus.get("encoder")
        if is_single_lowmem_gpu:
            _apply_single_lowmem_gpu_overrides(args, game_config)

        # This would be way too slow on CPU
        assert torch.cuda.is_available()
        main_device = torch.device("cuda")
        for name in ["detection", "stitching", "encoder"]:
            if name in gpus:
                main_device = gpus[name]
                torch.cuda.set_device(main_device)
                break

        data_pipeline = None

        # Prefer unified Aspen config (namespaced under 'aspen') for model + pipeline
        aspen_cfg_for_pipeline = game_config.get("aspen") if isinstance(game_config, dict) else None
        # Expose to downstream run_mmtrack() via args dict
        args.aspen = aspen_cfg_for_pipeline

        # If ONNX/TRT detector flags are provided, thread them into Aspen plugins.detector_factory.params
        if args.aspen and isinstance(args.aspen, dict):
            trunks_cfg = args.aspen.setdefault("plugins", {}) or {}
            # When loading precomputed tracking or detection CSVs, skip writing
            # detections to CSV to avoid unnecessary work and potential dtype issues.
            if getattr(args, "input_tracking_data", None) or getattr(
                args, "input_detection_data", None
            ):
                save_det = trunks_cfg.get("save_detections")
                if isinstance(save_det, dict):
                    save_det["enabled"] = False
            try:
                onnx_enable = bool(
                    args.detector_onnx_enable
                    or args.detector_onnx_path
                    or args.detector_onnx_quantize_int8
                )
                # Optional static detection outputs (fixed-shape top-k)
                static_det_enable = bool(getattr(args, "detector_static_detections", False))
                static_det_max = int(getattr(args, "detector_static_max_detections", 0) or 0)
                # Configure static-shape detection outputs by default whenever the
                # detector supports them (e.g., YOLOXHead). The CLI flag controls
                # additional overrides such as max_detections.
                if "detector" in trunks_cfg:
                    df = trunks_cfg.setdefault(
                        "detector_factory",
                        {
                            "class": "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    df_params = df.setdefault("params", {}) or {}
                    static_cfg = df_params.setdefault("static_detections", {}) or {}
                    # Always enable static detections when supported; allow CLI
                    # to override max_detections when provided.
                    static_cfg.setdefault("enable", True)
                    if static_det_enable and static_det_max > 0:
                        static_cfg["max_detections"] = static_det_max
                    df_params["static_detections"] = static_cfg
                    df["params"] = df_params
                    trunks_cfg["detector_factory"] = df
                if onnx_enable and "detector" in trunks_cfg:
                    df = trunks_cfg.setdefault(
                        "detector_factory",
                        {
                            "class": "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    df_params = df.setdefault("params", {}) or {}
                    onnx_cfg = df_params.setdefault("onnx", {}) or {}
                    # Determine if ONNX should be enabled
                    onnx_cfg["enable"] = True
                    # Default path under results folder if not provided
                    default_onnx_path = os.path.join(results_folder, "detector.onnx")
                    onnx_cfg["path"] = args.detector_onnx_path or default_onnx_path
                    onnx_cfg["force_export"] = bool(args.detector_onnx_force_export)
                    onnx_cfg["quantize_int8"] = bool(args.detector_onnx_quantize_int8)
                    onnx_cfg["calib_frames"] = int(args.detector_onnx_calib_frames or 0)
                    # Mirror NMS configuration for ONNX-backed detectors so the
                    # same DetectorNMS path can be used.
                    onnx_cfg["nms_backend"] = args.detector_nms_backend
                    onnx_cfg["nms_test"] = bool(args.detector_nms_test)
                    onnx_cfg["nms_plugin"] = args.detector_trt_nms_plugin
                    df_params["onnx"] = onnx_cfg
                    df["params"] = df_params
                    trunks_cfg["detector_factory"] = df
                # TensorRT detector integration
                trt_enable = bool(args.detector_trt_enable or args.detector_trt_engine)
                if trt_enable and "detector" in trunks_cfg:
                    df = trunks_cfg.setdefault(
                        "detector_factory",
                        {
                            "class": "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    df_params = df.setdefault("params", {}) or {}
                    trt_cfg = df_params.setdefault("trt", {}) or {}
                    trt_cfg["enable"] = True
                    default_engine_path = os.path.join(results_folder, "detector.engine")
                    trt_cfg["engine"] = args.detector_trt_engine or default_engine_path
                    trt_cfg["force_build"] = bool(args.detector_trt_force_build)
                    trt_cfg["fp16"] = bool(args.detector_trt_fp16)
                    # INT8 options
                    trt_cfg["int8"] = bool(args.detector_trt_int8)
                    trt_cfg["calib_frames"] = int(args.detector_trt_calib_frames or 0)
                    # NMS backend selection for TensorRT detector
                    trt_cfg["nms_backend"] = args.detector_nms_backend
                    trt_cfg["nms_test"] = bool(args.detector_nms_test)
                    trt_cfg["nms_plugin"] = args.detector_trt_nms_plugin
                    df_params["trt"] = trt_cfg
                    df["params"] = df_params
                    trunks_cfg["detector_factory"] = df
                # Pose ONNX integration (pose_factory)
                pose_onnx_enable = bool(
                    args.pose_onnx_enable or args.pose_onnx_path or args.pose_onnx_quantize_int8
                )
                if pose_onnx_enable and "pose" in trunks_cfg:
                    pf = trunks_cfg.setdefault(
                        "pose_factory",
                        {
                            "class": "hmlib.aspen.plugins.pose_factory_plugin.PoseInferencerFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    pf_params = pf.setdefault("params", {}) or {}
                    ponnx_cfg = pf_params.setdefault("onnx", {}) or {}
                    ponnx_cfg["enable"] = True
                    default_pose_onnx = os.path.join(results_folder, "pose.onnx")
                    ponnx_cfg["path"] = args.pose_onnx_path or default_pose_onnx
                    ponnx_cfg["force_export"] = bool(args.pose_onnx_force_export)
                    ponnx_cfg["quantize_int8"] = bool(args.pose_onnx_quantize_int8)
                    ponnx_cfg["calib_frames"] = int(args.pose_onnx_calib_frames or 0)
                    pf_params["onnx"] = ponnx_cfg
                    pf["params"] = pf_params
                    trunks_cfg["pose_factory"] = pf
                # Pose TensorRT integration (pose_factory)
                pose_trt_enable = bool(args.pose_trt_enable or args.pose_trt_engine)
                if pose_trt_enable and "pose" in trunks_cfg:
                    pf = trunks_cfg.setdefault(
                        "pose_factory",
                        {
                            "class": "hmlib.aspen.plugins.pose_factory_plugin.PoseInferencerFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    pf_params = pf.setdefault("params", {}) or {}
                    ptrt_cfg = pf_params.setdefault("trt", {}) or {}
                    ptrt_cfg["enable"] = True
                    default_pose_engine = os.path.join(results_folder, "pose.engine")
                    ptrt_cfg["engine"] = args.pose_trt_engine or default_pose_engine
                    ptrt_cfg["force_build"] = bool(args.pose_trt_force_build)
                    ptrt_cfg["fp16"] = bool(args.pose_trt_fp16)
                    # INT8 options
                    ptrt_cfg["int8"] = bool(args.pose_trt_int8)
                    ptrt_cfg["calib_frames"] = int(args.pose_trt_calib_frames or 0)
                    ptrt_cfg["batch_size"] = int(args.pose_trt_batch_size)
                    pf_params["trt"] = ptrt_cfg
                    pf["params"] = pf_params
                    trunks_cfg["pose_factory"] = pf
                # Tracker backend selection (HmTracker vs static CUDA ByteTrack)
                tracker_backend = getattr(args, "tracker_backend", None)
                if tracker_backend is not None and "tracker" in trunks_cfg:
                    tracker_cfg = trunks_cfg.setdefault(
                        "tracker",
                        {
                            "class": "hmlib.aspen.plugins.tracker_plugin.TrackerPlugin",
                            "depends": [
                                "detector",
                                "ice_boundaries",
                                "model_factory",
                                "boundaries",
                            ],
                            "params": {},
                        },
                    )
                    tracker_params = tracker_cfg.setdefault("params", {}) or {}
                    if tracker_backend == "hm":
                        # Default HmTracker backend; clear any explicit overrides.
                        tracker_params.pop("tracker_class", None)
                        tracker_params.pop("tracker_kwargs", None)
                    elif tracker_backend == "static_bytetrack":
                        tracker_params["tracker_class"] = (
                            "hmlib.tracking_utils.bytetrack.HmByteTrackerCudaStatic"
                        )
                        tracker_kwargs = tracker_params.setdefault("tracker_kwargs", {}) or {}
                        max_det = getattr(args, "tracker_max_detections", 256)
                        max_tracks = getattr(args, "tracker_max_tracks", 256)
                        if max_det is not None:
                            tracker_kwargs["max_detections"] = int(max_det)
                        if max_tracks is not None:
                            tracker_kwargs["max_tracks"] = int(max_tracks)
                        tracker_device = getattr(args, "tracker_device", None)
                        if tracker_device:
                            tracker_kwargs["device"] = tracker_device
                        tracker_params["tracker_kwargs"] = tracker_kwargs
                    tracker_cfg["params"] = tracker_params
                    trunks_cfg["tracker"] = tracker_cfg
                args.aspen["plugins"] = trunks_cfg
            except Exception:
                traceback.print_exc()

        if args.tracking:
            model = None  # Built by Aspen ModelFactoryPlugin

            # Build inference pipeline from Aspen YAML if provided
            pipeline = None
            if aspen_cfg_for_pipeline and "inference_pipeline" in aspen_cfg_for_pipeline:
                pipeline = aspen_cfg_for_pipeline["inference_pipeline"]
                # first transform should be HmLoadImageFromWebcam in streaming
                if pipeline and isinstance(pipeline[0], dict):
                    pipeline[0]["type"] = "HmLoadImageFromWebcam"
                # Coerce types not representable in YAML (e.g., tuple for meta_keys)
                for step in pipeline:
                    if not isinstance(step, dict):
                        continue
                    t = step.get("type")
                    if t in ("mmdet.PackTrackInputs", "PackTrackInputs"):
                        mk = step.get("meta_keys")
                        if isinstance(mk, list):
                            step["meta_keys"] = tuple(mk)
                    update_pipeline_item(
                        pipeline,
                        "IceRinkSegmConfig",
                        dict(
                            game_id=args.game_id,
                            ice_rink_inference_scale=getattr(
                                args, "ice_rink_inference_scale", None
                            ),
                        ),
                    )
                # Apply clip box if present
                orig_clip_box = get_clip_box(game_id=args.game_id, root_dir=args.root_dir)
                if orig_clip_box:
                    hm_crop = get_pipeline_item(pipeline, "HmCrop")
                    if hm_crop is not None:
                        hm_crop["rectangle"] = orig_clip_box
                from mmcv.transforms import Compose

                data_pipeline = Compose(pipeline)
            else:
                data_pipeline = None

            #
            # post-detection pipeline updates
            #
            # For Aspen-built model, boundaries will be applied by BoundariesPlugin.
            # Put boundary inputs into config dict so run_mmtrack can pass to Aspen shared.
            # Recompute tuned boundary lines from game_config (legacy behavior of DefaultArguments).
            game_bound_cfg = (
                game_config.get("game", {}).get("boundaries", {})
                if isinstance(game_config, dict)
                else {}
            )
            top_border_lines = game_bound_cfg.get("upper", []) or []
            bottom_border_lines = game_bound_cfg.get("lower", []) or []
            upper_tune_position = game_bound_cfg.get("upper_tune_position", []) or []
            lower_tune_position = game_bound_cfg.get("lower_tune_position", []) or []
            boundary_scale_width = game_bound_cfg.get("scale_width", 1.0)
            boundary_scale_height = game_bound_cfg.get("scale_height", 1.0)

            def _tune_lines(lines, tune_pos):
                if not lines or not tune_pos:
                    return lines
                tuned = []
                for x1, y1, x2, y2 in lines:
                    if boundary_scale_width:
                        x1 *= boundary_scale_width
                        x2 *= boundary_scale_width
                    if boundary_scale_height:
                        y2 *= boundary_scale_height
                        y1 *= boundary_scale_height
                    x1 += tune_pos[0]
                    x2 += tune_pos[0]
                    y1 += tune_pos[1]
                    y2 += tune_pos[1]
                    tuned.append([x1, y1, x2, y2])
                return tuned

            top_border_lines = _tune_lines(top_border_lines, upper_tune_position)
            bottom_border_lines = _tune_lines(bottom_border_lines, lower_tune_position)

            args.initial_args = vars(args)
            args.initial_args["top_border_lines"] = top_border_lines
            args.initial_args["bottom_border_lines"] = bottom_border_lines
            args.initial_args["original_clip_box"] = get_clip_box(
                game_id=args.game_id, root_dir=args.root_dir
            )
            # Keep a copy under game_config for Aspen plugins that read from game_config.initial_args
            if hasattr(args, "game_config") and isinstance(args.game_config, dict):
                args.game_config["initial_args"] = args.initial_args

        if args.max_frames or args.max_time:
            if not args.no_audio:
                print("Disabling audio extraction due to max-frames/max-time limit")
                args.no_audio = True

        postprocessor = None
        if args.input_video:
            input_video_files = args.input_video.split(",")
            aspen_stitching_cli = getattr(args, "aspen_stitching", None)
            if aspen_stitching_cli is None:
                use_aspen_stitching = bool(
                    get_nested_value(game_config, "stitching.aspen-stitching", False)
                )
            else:
                use_aspen_stitching = bool(aspen_stitching_cli)
            if is_stitching(args.input_video):
                if args.camera_ui and not use_aspen_stitching:
                    raise ValueError(
                        "--camera-ui requires Aspen stitching for multi-camera input; "
                        "remove --no-aspen-stitching"
                    )
                project_file_name = "hm_project.pto"

                game_videos = {}

                if len(input_video_files) == 2:
                    vl = input_video_files[0]
                    vr = input_video_files[1]
                    dir_name = os.path.dirname(vl)
                    assert dir_name == os.path.dirname(vr)
                elif os.path.isdir(args.input_video):
                    game_videos = configure_game_videos(
                        game_id=args.game_id,
                        inference_scale=getattr(args, "ice_rink_inference_scale", None),
                    )
                    dir_name = args.input_video
                    assert dir_name
                    input_video_files = game_videos

                left_vid = BasicVideoInfo(",".join(game_videos["left"]))
                right_vid = BasicVideoInfo(",".join(game_videos["right"]))

                total_frames = min(left_vid.frame_count, right_vid.frame_count)
                print(f"Total possible stitched video frames: {total_frames}")

                stitch_frame_time = args.stitch_frame_time
                if args.start_frame_time:
                    stitch_time_is_zero = stitch_frame_time is None
                    if not stitch_time_is_zero and stitch_frame_time:
                        try:
                            stitch_time_is_zero = (
                                time_to_frame(time_str=stitch_frame_time, fps=left_vid.fps) <= 0
                            )
                        except Exception:
                            stitch_time_is_zero = False
                    if stitch_time_is_zero:
                        stitch_frame_time = args.start_frame_time
                stitch_frame_number = 0
                if stitch_frame_time:
                    stitch_frame_number = time_to_frame(
                        time_str=stitch_frame_time, fps=left_vid.fps
                    )

                assert not args.start_frame or not args.start_frame_time
                if not args.start_frame and args.start_frame_time:
                    args.start_frame = time_to_frame(
                        time_str=args.start_frame_time, fps=left_vid.fps
                    )

                assert not args.max_frames or not args.max_time
                if not args.max_frames and args.max_time:
                    args.max_frames = time_to_frame(time_str=args.max_time, fps=left_vid.fps)

                pto_project_file, lfo, rfo = configure_video_stitching(
                    dir_name=dir_name,
                    video_left=str(game_videos["left"][0]),
                    video_right=str(game_videos["right"][0]),
                    max_control_points=args.max_control_points,
                    project_file_name=project_file_name,
                    left_frame_offset=args.lfo,
                    right_frame_offset=args.rfo,
                    base_frame_offset=stitch_frame_number,
                    game_id=args.game_id,
                    stitch_frame_time=stitch_frame_time,
                    ignore_private_config=bool(args.ignore_private_config),
                    game_config=args.game_config,
                )
                stitch_videos = {
                    "left": {
                        "files": game_videos["left"],
                        "frame_offset": lfo,
                    },
                    "right": {
                        "files": game_videos["right"],
                        "frame_offset": rfo,
                    },
                }

                def _set_runtime_arg(name: str, value: Any) -> None:
                    setattr(args, name, value)
                    if hasattr(args, "initial_args") and isinstance(args.initial_args, dict):
                        args.initial_args[name] = value
                    if isinstance(args.game_config, dict):
                        init_args = args.game_config.get("initial_args")
                        if isinstance(init_args, dict):
                            init_args[name] = value

                _set_runtime_arg("stitch_pto_project_file", str(pto_project_file))
                args.stitch_data_pipeline = data_pipeline

                if use_aspen_stitching:
                    # Enable the UI slider without a StitchDataset instance.
                    args.stitch_rotation_controller = _StitchRotationController(args.game_config)

                    frame_step_left = 1
                    frame_step_right = 1
                    if left_vid.fps > right_vid.fps:
                        int_ratio = int(left_vid.fps // right_vid.fps)
                        float_ratio = float(left_vid.fps / right_vid.fps)
                        if math.isclose(float(int_ratio), float_ratio) and int_ratio != 1:
                            frame_step_left = int_ratio
                    elif right_vid.fps > left_vid.fps:
                        int_ratio = int(right_vid.fps // left_vid.fps)
                        float_ratio = float(right_vid.fps / left_vid.fps)
                        if math.isclose(float(int_ratio), float_ratio) and int_ratio != 1:
                            frame_step_right = int_ratio

                    game_id = os.path.basename(str(dir_name))
                    left_loader = MOTLoadVideoWithOrig(
                        path=game_videos["left"],
                        game_id=game_id,
                        max_frames=args.max_frames,
                        batch_size=args.batch_size,
                        start_frame_number=args.start_frame + lfo,
                        original_image_only=True,
                        dtype=torch.uint8,
                        device=gpus["stitching"],
                        decoder_device=(
                            torch.device(args.decoder_device) if args.decoder_device else None
                        ),
                        decoder_type=args.video_stream_decode_method,
                        frame_step=frame_step_left,
                        no_cuda_streams=args.no_cuda_streams,
                        checkerboard_input=args.checkerboard_input,
                        prefetch_batches=args.dataset_prefetch_batches,
                    )
                    right_loader = MOTLoadVideoWithOrig(
                        path=game_videos["right"],
                        game_id=game_id,
                        max_frames=args.max_frames,
                        batch_size=args.batch_size,
                        start_frame_number=args.start_frame + rfo,
                        original_image_only=True,
                        dtype=torch.uint8,
                        device=gpus["stitching"],
                        decoder_device=(
                            torch.device(args.decoder_device) if args.decoder_device else None
                        ),
                        decoder_type=args.video_stream_decode_method,
                        frame_step=frame_step_right,
                        no_cuda_streams=args.no_cuda_streams,
                        checkerboard_input=args.checkerboard_input,
                        prefetch_batches=args.dataset_prefetch_batches,
                    )
                    stitch_inputs = MultiDataLoaderWrapper(
                        dataloaders=[left_loader, right_loader],
                    )
                    dataloader.append_dataset("stitch_inputs", stitch_inputs)
                else:
                    if isinstance(aspen_cfg_for_pipeline, dict):
                        stitching_cfg = aspen_cfg_for_pipeline.get("stitching")
                        if isinstance(stitching_cfg, dict):
                            stitching_cfg["enabled"] = False
                        plugins_cfg = aspen_cfg_for_pipeline.get("plugins")
                        if isinstance(plugins_cfg, dict):
                            stitching_plugin = plugins_cfg.get("stitching")
                            if isinstance(stitching_plugin, dict):
                                stitching_plugin["enabled"] = False
                    stitch_cfg = get_nested_value(args.game_config, "stitching", {}) or {}
                    left_stitch_pipeline_cfg = stitch_cfg.get("left_stitch_pipeline")
                    right_stitch_pipeline_cfg = stitch_cfg.get("right_stitch_pipeline")
                    stitch_dtype_cfg = str(stitch_cfg.get("dtype") or "float32").lower()
                    stitch_dtype = torch.half if "16" in stitch_dtype_cfg else torch.float
                    stitched_dataset = StitchDataset(
                        videos=stitch_videos,
                        pto_project_file=pto_project_file,
                        start_frame_number=args.start_frame,
                        max_frames=args.max_frames,
                        image_roi=None,
                        batch_size=args.batch_size,
                        remapping_device=gpus["stitching"],
                        decoder_device=(
                            torch.device(args.decoder_device) if args.decoder_device else None
                        ),
                        decoder_type=args.video_stream_decode_method,
                        blend_mode=str(stitch_cfg.get("blend_mode") or opts.blend_mode),
                        dtype=stitch_dtype,
                        python_blender=bool(stitch_cfg.get("python_blender", args.python_blender)),
                        minimize_blend=bool(stitch_cfg.get("minimize_blend", True)),
                        no_cuda_streams=bool(
                            stitch_cfg.get("no_cuda_streams", args.no_cuda_streams)
                        ),
                        post_stitch_rotate_degrees=stitch_cfg.get(
                            "post_stitch_rotate_degrees",
                            getattr(args, "stitch_rotate_degrees", None),
                        ),
                        profiler=getattr(args, "profiler", None),
                        config_ref=args.game_config,
                        left_color_pipeline=left_stitch_pipeline_cfg,
                        right_color_pipeline=right_stitch_pipeline_cfg,
                        capture_rgb_stats=bool(
                            stitch_cfg.get(
                                "capture_rgb_stats", getattr(args, "checkerboard_input", False)
                            )
                        ),
                        checkerboard_input=args.checkerboard_input,
                        prefetch_batches=args.dataset_prefetch_batches,
                    )
                    # Expose the StitchDataset instance so PlayTracker can control
                    # post-stitch rotation via the UI slider.
                    args.stitch_rotation_controller = stitched_dataset
                    # Create the MOT video data loader, passing it the
                    # stitching data loader as its image source
                    mot_dataloader = MOTLoadVideoWithOrig(
                        path=None,
                        game_id=dir_name,
                        start_frame_number=args.start_frame,
                        batch_size=1,  # This batch will contain one batch of whatever the stitcher's batch size is
                        embedded_data_loader=stitched_dataset,
                        data_pipeline=data_pipeline,
                        dtype=torch.float if not args.fp16 else torch.half,
                        device=gpus["stitching"],
                        original_image_only=False,
                        adjust_exposure=args.adjust_exposure,
                        no_cuda_streams=args.no_cuda_streams,
                        checkerboard_input=args.checkerboard_input,
                        prefetch_batches=args.dataset_prefetch_batches,
                    )
                    try:
                        mot_dataloader.set_profiler(getattr(args, "profiler", None))
                    except Exception:
                        pass
                    dataloader.append_dataset("pano", mot_dataloader)
            else:
                assert len(input_video_files) == 1
                if isinstance(aspen_cfg_for_pipeline, dict):
                    stitching_cfg = aspen_cfg_for_pipeline.get("stitching")
                    if isinstance(stitching_cfg, dict):
                        stitching_cfg["enabled"] = False
                    plugins_cfg = aspen_cfg_for_pipeline.get("plugins")
                    if isinstance(plugins_cfg, dict):
                        stitching_plugin = plugins_cfg.get("stitching")
                        if isinstance(stitching_plugin, dict):
                            stitching_plugin["enabled"] = False
                if os.path.isdir(input_video_files[0]):
                    dir_name = input_video_files[0]
                else:
                    dir_name = Path(input_video_files[0]).parent
                assert not args.start_frame or not args.start_frame_time
                if not args.start_frame and args.start_frame_time:
                    vid_info = BasicVideoInfo(input_video_files[0])
                    args.start_frame = time_to_frame(
                        time_str=args.start_frame_time, fps=vid_info.fps
                    )

                assert not args.max_frames or not args.max_time
                if not args.max_frames and args.max_time:
                    vid_info = BasicVideoInfo(input_video_files[0])
                    args.max_frames = time_to_frame(time_str=args.max_time, fps=vid_info.fps)
                pano_dataloader = MOTLoadVideoWithOrig(
                    path=input_video_files[0],
                    start_frame_number=args.start_frame,
                    batch_size=args.batch_size,
                    max_frames=args.max_frames,
                    device=main_device,
                    decoder_device=(
                        torch.device(args.decoder_device) if args.decoder_device else None
                    ),
                    decoder_type=args.video_stream_decode_method,
                    data_pipeline=data_pipeline,
                    dtype=torch.float if not args.fp16 else torch.half,
                    # When a data_pipeline is provided, we must deliver both
                    # the preprocessed pano and original_images; disable the
                    # original_image_only fast path in this mode.
                    original_image_only=False,
                    adjust_exposure=args.adjust_exposure,
                    no_cuda_streams=args.no_cuda_streams,
                    async_mode=not args.no_async_dataset,
                    checkerboard_input=bool(getattr(args, "checkerboard_input", False)),
                    prefetch_batches=args.dataset_prefetch_batches,
                )
                try:
                    pano_dataloader.set_profiler(getattr(args, "profiler", None))
                except Exception:
                    pass
                dataloader.append_dataset("pano", pano_dataloader)

            if args.end_zones:
                # Try far_left and far_right videos if they exist
                other_videos: List[Tuple[str, str]] = [
                    ("far_left", os.path.join(dir_name, "far_left.mp4")),
                    ("far_right", os.path.join(dir_name, "far_right.mp4")),
                ]
                ez_count = 0
                for vid_name, vid_path in other_videos:
                    if os.path.exists(vid_path):
                        extra_dataloader = MOTLoadVideoWithOrig(
                            path=vid_path,
                            start_frame_number=args.start_frame,
                            batch_size=args.batch_size,
                            dtype=torch.float if not args.fp16 else torch.half,
                            device=gpus["encoder"],
                            decoder_type=args.video_stream_decode_method,
                            original_image_only=True,
                            no_cuda_streams=args.no_cuda_streams,
                            async_mode=args.no_async_dataset,
                            checkerboard_input=bool(getattr(args, "checkerboard_input", False)),
                            prefetch_batches=args.dataset_prefetch_batches,
                        )
                    try:
                        extra_dataloader.set_profiler(getattr(args, "profiler", None))
                    except Exception:
                        pass
                    dataloader.append_dataset(vid_name, extra_dataloader)
                    ez_count += 1
                if not ez_count:
                    raise ValueError("--end-zones specified, but no end-zone videos found")

        if dataloader is None:
            raise ValueError("Dataloader could not be constructed")

        if not args.no_progress_bar:
            table_map = OrderedDict()
            if is_stitching(args.input_video):
                table_map["Stitching"] = "ENABLED"

            batch_size_hint = max(1, int(getattr(dataloader, "batch_size", args.batch_size) or 1))
            progress_bar = ProgressBar(
                total=len(dataloader),
                scroll_output=ScrollOutput(lines=args.progress_bar_lines).register_logger(logger),
                update_rate=args.print_interval,
                table_map=table_map,
                title=args.game_id,
                use_curses=args.curses_progress,
                units_per_iter=batch_size_hint,
            )
        else:
            progress_bar = None

        is_truncated_run = bool(args.max_time or args.max_frames)

        output_video_path = None
        if not args.no_save_video:
            output_video_path = _resolve_base_output_path(game_config, results_folder)

        output_label = args.output_label or args.label
        if output_label and output_video_path:
            try:
                output_video_path = str(
                    add_prefix_to_filename(output_video_path, str(output_label))
                )
            except Exception:
                pass
        if output_label and args.output_video:
            try:
                args.output_video = str(
                    add_prefix_to_filename(args.output_video, str(output_label))
                )
            except Exception:
                pass

        mux_audio_file = args.mux_audio_file
        if (
            (not args.audio_only)
            and (not is_truncated_run)
            and (not args.no_audio)
            and output_video_path
        ):
            if not mux_audio_file:
                try:
                    mux_audio_file, mux_audio_temp_file = _resolve_mux_audio_source(
                        input_video_files
                    )
                except Exception:
                    logger.exception(
                        "Failed to resolve mux audio source; continuing without audio."
                    )
                    mux_audio_file = None
            if mux_audio_file:
                output_video_path = _with_audio_output_path(output_video_path)

        args.mux_audio_file = mux_audio_file
        args.output_video_path = output_video_path
        if output_video_path:
            try:
                set_nested_value(game_config, "video_out.output_video_path", output_video_path)
            except Exception:
                pass

        if not args.audio_only:

            if not args.output_video_bit_rate:
                args.output_video_bit_rate = dataloader.get_max_attribute("bit_rate")

            if not args.no_play_tracking:

                #
                # Video output pipeline
                #
                video_out_pipeline = None
                if model is not None and hasattr(model, "cfg"):
                    video_out_pipeline = getattr(model.cfg, "video_out_pipeline")
                else:
                    video_out_pipeline = game_config.get("video_out_pipeline")
                    if video_out_pipeline is None and aspen_cfg_for_pipeline:
                        video_out_pipeline = aspen_cfg_for_pipeline.get("video_out_pipeline")
                if video_out_pipeline:
                    video_out_pipeline = copy.deepcopy(video_out_pipeline)
                overlay_cfg = _build_overlay_cfg_from_args(args)
                if overlay_cfg:
                    video_out_pipeline = _inject_overlay_text(video_out_pipeline, overlay_cfg)
                # Make video_out_pipeline available to Aspen plugins via args
                args.video_out_pipeline = video_out_pipeline
            postprocessor = None

            other_kwargs = {
                "dataloader": dataloader,
                "postprocessor": postprocessor,
            }

            run_mmtrack(
                model=model,
                config=vars(args),
                device=main_device,
                fp16=args.fp16,
                input_cache_size=args.cache_size,
                progress_bar=progress_bar,
                no_cuda_streams=args.no_cuda_streams,
                profiler=getattr(args, "profiler", None),
                **other_kwargs,
            )

        #
        # Deploy output video and CSV artifacts to --deploy-dir (explicit) or the
        # game directory (full run).
        #
        dest_path = None
        deploy_dir = args.deploy_dir
        target_deploy_dir = None
        if deploy_dir:
            target_deploy_dir = deploy_dir
        elif not is_truncated_run:
            target_deploy_dir = (
                args.game_dir if args.game_dir and os.path.isdir(args.game_dir) else None
            )

        if output_video_path and os.path.exists(output_video_path):
            try:
                if args.output_video:
                    output_av_path = str(args.output_video)
                    parent = os.path.dirname(output_av_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    if os.path.abspath(output_video_path) != os.path.abspath(output_av_path):
                        shutil.copy2(output_video_path, output_av_path)
                    dest_path = Path(output_av_path)
                elif target_deploy_dir:
                    os.makedirs(target_deploy_dir, exist_ok=True)
                    file_name = os.path.basename(output_video_path)
                    file_name = str(
                        add_game_id_prefix_to_filename(file_name, args.game_id, sep="-")
                    )
                    base_name, extension = os.path.splitext(file_name)
                    output_av_path = None
                    for i in range(1000):
                        if i:
                            fname = f"{base_name}-{i}{extension}"
                        else:
                            fname = f"{base_name}{extension}"
                        candidate = os.path.join(target_deploy_dir, fname)
                        if not os.path.exists(candidate):
                            output_av_path = candidate
                            break
                    if output_av_path is None:
                        raise RuntimeError("Could not find a free deploy filename for output video")
                    if os.path.abspath(output_video_path) != os.path.abspath(output_av_path):
                        shutil.copy2(output_video_path, output_av_path)
                    dest_path = Path(output_av_path)
            except Exception:
                logger.exception("Failed to deploy output video; continuing.")
                dest_path = None

        if target_deploy_dir:
            os.makedirs(target_deploy_dir, exist_ok=True)
            csv_names = []
            try:
                for name in os.listdir(results_folder):
                    if not name.endswith(".csv"):
                        continue
                    src_path = os.path.join(results_folder, name)
                    if os.path.isfile(src_path):
                        csv_names.append(name)
            except Exception:
                traceback.print_exc()
                csv_names = []

            def extract_suffix_num(path: Optional[os.PathLike | str]) -> Optional[int]:
                if not path:
                    return None
                base = os.path.splitext(os.path.basename(str(path)))[0]
                dash_idx = base.rfind("-")
                if dash_idx == -1:
                    return None
                tail = base[dash_idx + 1 :]
                if tail.isdigit():
                    return int(tail)
                return None

            def with_index(name: str, suffix_num: int) -> str:
                root, ext = os.path.splitext(name)
                if suffix_num <= 0:
                    return f"{root}{ext}"
                return f"{root}-{suffix_num}{ext}"

            def choose_free_suffix(names: List[str]) -> int:
                for i in range(0, 1000):
                    collision = False
                    for name in names:
                        if os.path.exists(os.path.join(target_deploy_dir, with_index(name, i))):
                            collision = True
                            break
                    if not collision:
                        return i
                raise RuntimeError("Could not find a free suffix for CSV deployment")

            suffix_num = extract_suffix_num(dest_path)
            if suffix_num is not None and csv_names:
                for name in csv_names:
                    if os.path.exists(
                        os.path.join(target_deploy_dir, with_index(name, suffix_num))
                    ):
                        suffix_num = None
                        break
            if suffix_num is None and csv_names:
                suffix_num = choose_free_suffix(csv_names)

            for name in csv_names:
                src_path = os.path.join(results_folder, name)
                dst_path = os.path.join(target_deploy_dir, with_index(name, int(suffix_num or 0)))
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception:
                    traceback.print_exc()
        logger.info("Completed")
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        try:
            if postprocessor is not None:
                try:
                    postprocessor.stop()
                except Exception:
                    traceback.print_exc()
            if dataloader is not None and hasattr(dataloader, "close"):
                try:
                    dataloader.close()
                except Exception:
                    traceback.print_exc()
            if mux_audio_temp_file is not None:
                try:
                    mux_audio_temp_file.close()
                except Exception:
                    traceback.print_exc()
        except Exception as ex:
            print(f"Exception while shutting down: {ex}")


def setup_logging():
    root_logger = get_root_logger()
    root_logger.setLevel(20)


def main():
    setup_logging()

    # Prefer CUDA, but don't hard-fail if the runtime isn't detected so CPU-only
    # runs can still proceed (albeit slowly).
    if not torch.cuda.is_available():
        logger.warning("CUDA not detected; running hmtrack on CPU will be very slow.")
    elif not torch.backends.cudnn.is_available():
        logger.warning("cuDNN not detected; performance may be degraded.")

    parser = make_parser()
    args = parser.parse_args()
    args.explicit_arg_names = hm_opts.collect_explicit_arg_names(parser)
    if args.smoke_test:
        return _run_smoke_test(args)

    import hmlib.hm_transforms  # noqa: F401 (register custom MMEngine transforms)

    if getattr(args, "smoke_test", False):
        return _run_smoke_test(args)

    game_config = get_config(
        game_id=args.game_id,
        rink=args.rink,
        camera=args.camera_name,
        root_dir=args.root_dir,
        ignore_private_config=bool(args.ignore_private_config),
    )

    # Merge user-provided YAML configs in order (--config can be repeated).
    # Later files override earlier values.
    from hmlib.config import load_yaml_files_ordered, recursive_update

    def _split_and_strip(items):
        paths = []
        if not items:
            return paths
        for it in items:
            if not it:
                continue
            parts = [p.strip() for p in str(it).split(",") if p.strip()]
            paths.extend(parts)
        return paths

    additional_cfg_paths = _split_and_strip(args.config)
    if not additional_cfg_paths:
        default_aspen = os.path.join(ROOT_DIR, "config", "aspen", "tracking.yaml")
        if os.path.exists(default_aspen):
            additional_cfg_paths.append(default_aspen)
    if additional_cfg_paths:
        merged_extra = load_yaml_files_ordered(additional_cfg_paths)
        if merged_extra:
            game_config = recursive_update(game_config, merged_extra)
    normalize_runtime_config(game_config)

    # Apply CLI-driven config overrides before resolving GLOBAL.* references so
    # Aspen plugins see the updated values via GLOBAL.*.
    hm_opts.apply_arg_config_overrides(
        game_config,
        args,
        parser=parser,
        explicit_arg_names=getattr(args, "explicit_arg_names", None),
    )

    # Let hm_opts apply --config-override before resolving GLOBAL.* refs.
    args.game_config = game_config
    args = hm_opts.init(args, parser)
    hm_opts.persist_private_config_overrides(
        args,
        parser=parser,
        config=args.game_config,
        explicit_arg_names=getattr(args, "explicit_arg_names", None),
    )
    game_config = resolve_global_refs(args.game_config)
    args.game_config = game_config

    if args.experiment_config:
        if getattr(args, "py_trace_out", None):
            logger.warning("--py-trace-out is ignored when running experiment sweeps.")
        return _run_experiment(args, parser, game_config)

    # Set up the task flags
    args.tracking = False
    tokens = args.tasks.split(",")
    for t in tokens:
        setattr(args, t, True)

    game_config["initial_args"] = vars(args)
    args.game_config = game_config

    args = configure_model(config=args.game_config, args=args)

    if args.game_id:
        num_gpus = 1
    else:
        if isinstance(args.gpus, str):
            args.gpus = [int(g) for g in args.gpus.split(",")]
        num_gpus = len(args.gpus) if args.gpus else 0
        num_gpus = min(num_gpus, torch.cuda.device_count())

    # Optional Python cProfile
    if getattr(args, "py_trace_out", None):
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        try:
            _main(args, num_gpus)
        finally:
            pr.disable()
            out_path = args.py_trace_out
            try:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            except Exception:
                pass
            if out_path.endswith(".txt"):
                with open(out_path, "w") as f:
                    ps = pstats.Stats(pr, stream=f)
                    ps.sort_stats("cumulative").print_stats()
            else:
                pr.dump_stats(out_path)
    else:
        _main(args, num_gpus)
    print("Done.")


if __name__ == "__main__":
    try:
        # Skip the eager CUDA stream bootstrap for smoke tests so CPU/ROCm checks can run
        # without touching full tracker startup.
        if "--smoke-test" not in sys.argv and torch.cuda.is_available():
            with torch.cuda.stream(torch.cuda.Stream(torch.device("cuda"))):
                main()
        else:
            main()
    except Exception as e:
        print(f"Exception during processing: {e}")
        traceback.print_exc()
        # Debug: list live threads to help diagnose hangs where an error
        # has been raised but the process does not exit promptly.
        try:
            import threading

            print("Live threads after exception:")
            for t in threading.enumerate():
                try:
                    print(f" - {t.name} (daemon={t.daemon})")
                except Exception:
                    print(f" - {t}")
        except Exception:
            pass
        raise SystemExit(1)
