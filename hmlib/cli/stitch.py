"""
Experiments in stitching
"""

import argparse
import contextlib
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hmlib.config import (
    get_game_config_private,
    get_nested_value,
    load_config_file,
    set_nested_value,
)
from hmlib.hm_opts import _get_baseline_runtime_config, hm_opts, preferred_arg
from hmlib.log import get_root_logger
from hmlib.orientation import configure_game_videos
from hmlib.stitching.configure_stitching import clean_stitch_game_artifacts
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.path import add_prefix_to_filename

ROOT_DIR = os.getcwd()

logger = get_root_logger()
_CONFIG_MISSING = object()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    parser.add_argument("--force", action="store_true", help="Force all recalcs (clean then run)")
    parser.add_argument(
        "--clean",
        "--clean-only",
        dest="clean",
        action="store_true",
        help="Delete rebuildable stitching artifacts and cached stitch config, then exit",
    )
    parser.add_argument(
        "--configure-only", action="store_true", help="Run stitching configuration only"
    )
    parser.add_argument(
        "--single-file",
        default=0,
        type=int,
        help="Only use a single video file from each perspective",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use multiple GPUs (probably slower, but if memory issues)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Validate stitch CLI startup, report the active torch backend, "
            "optionally exercise preview output setup, and exit."
        ),
    )
    return parser


def convert_seconds_to_hms(total_seconds):
    hours = int(total_seconds // 3600)  # Calculate the number of hours
    minutes = int((total_seconds % 3600) // 60)  # Calculate the remaining minutes
    seconds = int(total_seconds % 60)  # Calculate the remaining seconds

    # Format the time in "HH:MM:SS" format
    return f"{hours:02}:{minutes:02}:{seconds:02}"


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
        label="stitch-smoke-preview",
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
    for message in _exercise_preview_smoke(args):
        print(message)
    print(
        "Smoke test OK. "
        f"backend={_torch_backend_label()} "
        f"cuda_available={torch.cuda.is_available()} "
        f"game_id={args.game_id} "
        f"video_dir={args.video_dir}"
    )
    return 0


def _arg_was_explicit(args: Optional[argparse.Namespace], name: str) -> bool:
    explicit = getattr(args, "explicit_arg_names", None)
    if explicit is None:
        return False
    return name in set(explicit)


def _config_override_was_explicit(args: Optional[argparse.Namespace], *config_keys: str) -> bool:
    if args is None:
        return False
    overrides = getattr(args, "config_overrides", None) or []
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
    args: Optional[argparse.Namespace], plugin_path: str, source_path: str
) -> bool:
    if args is None:
        return False
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


def _game_or_private_config_was_explicit(
    args: Optional[argparse.Namespace], *config_keys: str
) -> bool:
    if args is None or args.game_id is None:
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
    args: Optional[argparse.Namespace], plugin_path: str, source_path: str
) -> bool:
    if args is None or args.game_id is None:
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


def _apply_stitch_buffering_defaults(
    aspen_cfg_all: Dict[str, Any], args: Optional[argparse.Namespace]
) -> None:
    """Clamp stitch pipeline buffering so panoramas do not queue multiple frames by default."""
    aspen_cfg = aspen_cfg_all.setdefault("aspen", {})
    if not isinstance(aspen_cfg, dict):
        return
    pipeline_cfg = aspen_cfg.setdefault("pipeline", {})
    if not isinstance(pipeline_cfg, dict):
        return

    baseline_config = _get_baseline_runtime_config()
    applied: list[str] = []

    if not _arg_was_explicit(args, "aspen_max_concurrent") and not _config_override_was_explicit(
        args, "aspen.pipeline.max_concurrent"
    ):
        if not _game_or_private_config_was_explicit(args, "aspen.pipeline.max_concurrent") and (
            _config_value_is_default_or_missing(
                aspen_cfg_all, baseline_config, "aspen.pipeline.max_concurrent"
            )
        ):
            pipeline_cfg["max_concurrent"] = 1
            applied.append("aspen.pipeline.max_concurrent=1")

    if not _arg_was_explicit(args, "aspen_thread_queue_size") and not _config_override_was_explicit(
        args, "aspen.pipeline.queue_size"
    ):
        if not _game_or_private_config_was_explicit(args, "aspen.pipeline.queue_size") and (
            _config_value_is_default_or_missing(
                aspen_cfg_all, baseline_config, "aspen.pipeline.queue_size"
            )
        ):
            pipeline_cfg["queue_size"] = 1
            applied.append("aspen.pipeline.queue_size=1")

    if not applied:
        return

    logger.info(
        "Using conservative stitch buffering defaults to limit peak memory: %s",
        ", ".join(applied),
    )


def _apply_single_lowmem_gpu_overrides(
    args: argparse.Namespace, aspen_cfg_all: Optional[Dict[str, Any]]
) -> bool:
    print("Adjusting stitch configuration for a single low-memory GPU environment...")
    explicit = set(getattr(args, "explicit_arg_names", None) or [])
    use_half_dtype = False
    if not isinstance(aspen_cfg_all, dict):
        return use_half_dtype

    baseline_config = _get_baseline_runtime_config()

    if (
        "fp16_stitch" not in explicit
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
            and _config_value_is_default_or_missing(
                aspen_cfg_all, baseline_config, "stitching.dtype"
            )
            and _plugin_value_follows_source_or_missing(
                aspen_cfg_all,
                "aspen.plugins.stitching.params.dtype",
                "stitching.dtype",
            )
        )
        if can_override_dtype:
            args.fp16_stitch = True
            set_nested_value(aspen_cfg_all, "stitching.dtype", "float16")
            set_nested_value(aspen_cfg_all, "aspen.plugins.stitching.params.dtype", "float16")
            use_half_dtype = True

    if (
        "max_blend_levels" not in explicit
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
                aspen_cfg_all, baseline_config, "stitching.max_blend_levels"
            )
            and _plugin_value_follows_source_or_missing(
                aspen_cfg_all,
                "aspen.plugins.stitching.params.max_blend_levels",
                "stitching.max_blend_levels",
            )
        )
        if can_override_max_blend_levels:
            args.max_blend_levels = 5
            set_nested_value(aspen_cfg_all, "stitching.max_blend_levels", 5)
            set_nested_value(aspen_cfg_all, "aspen.plugins.stitching.params.max_blend_levels", 5)

    if (
        "minimize_blend" not in explicit
        and "no_minimize_blend" not in explicit
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
                aspen_cfg_all, baseline_config, "stitching.minimize_blend"
            )
            and _plugin_value_follows_source_or_missing(
                aspen_cfg_all,
                "aspen.plugins.stitching.params.minimize_blend",
                "stitching.minimize_blend",
            )
        )
        if can_override_minimize_blend:
            args.minimize_blend = 1
            args.no_minimize_blend = False
            set_nested_value(aspen_cfg_all, "stitching.minimize_blend", True)
            set_nested_value(aspen_cfg_all, "aspen.plugins.stitching.params.minimize_blend", True)

    if (
        "output_width" not in explicit
        and "output_height" not in explicit
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
                aspen_cfg_all, baseline_config, "video_out.output_width"
            )
            and _config_value_is_default_or_missing(
                aspen_cfg_all, baseline_config, "video_out.output_height"
            )
            and _config_value_is_default_or_missing(
                aspen_cfg_all, baseline_config, "stitching.max_output_width"
            )
            and _plugin_value_follows_source_or_missing(
                aspen_cfg_all,
                "aspen.plugins.stitching.params.max_output_width",
                "stitching.max_output_width",
            )
            and _plugin_value_follows_source_or_missing(
                aspen_cfg_all,
                "aspen.plugins.video_out_prep.params.output_width",
                "video_out.output_width",
            )
        )
        if can_override_output_width:
            args.output_width = 1920
            set_nested_value(aspen_cfg_all, "video_out.output_width", 1920)
            set_nested_value(aspen_cfg_all, "stitching.max_output_width", 1920)
            set_nested_value(aspen_cfg_all, "aspen.plugins.stitching.params.max_output_width", 1920)
            set_nested_value(
                aspen_cfg_all, "aspen.plugins.video_out_prep.params.output_width", 1920
            )

    if "aspen_max_concurrent" not in explicit and not _config_override_was_explicit(
        args, "aspen.pipeline.max_concurrent"
    ):
        can_override_max_concurrent = not _game_or_private_config_was_explicit(
            args, "aspen.pipeline.max_concurrent"
        ) and _config_value_is_default_or_missing(
            aspen_cfg_all, baseline_config, "aspen.pipeline.max_concurrent"
        )
        if can_override_max_concurrent:
            args.aspen_max_concurrent = 1
            set_nested_value(aspen_cfg_all, "aspen.pipeline.max_concurrent", 1)

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
                aspen_cfg_all, baseline_config, "stitching.cache_rotation_grid"
            )
            and _plugin_value_follows_source_or_missing(
                aspen_cfg_all,
                "aspen.plugins.stitching.params.cache_rotation_grid",
                "stitching.cache_rotation_grid",
            )
        )
        if can_override_rotation_grid_cache:
            set_nested_value(aspen_cfg_all, "stitching.cache_rotation_grid", False)
            set_nested_value(
                aspen_cfg_all,
                "aspen.plugins.stitching.params.cache_rotation_grid",
                False,
            )

    return use_half_dtype


def _resolve_stitch_tensor_dtype(
    default_dtype: torch.dtype, stitch_cfg: Dict[str, Any]
) -> torch.dtype:
    dtype_value = stitch_cfg.get("dtype")
    if dtype_value is None:
        return default_dtype
    if isinstance(dtype_value, torch.dtype):
        return dtype_value

    dtype_name = str(dtype_value).strip().lower()
    if dtype_name in ("float16", "fp16", "half"):
        return torch.float16
    if dtype_name in ("float32", "float", "fp32"):
        return torch.float32
    if dtype_name in ("uint8", "u8"):
        return torch.uint8
    raise ValueError(f"Unsupported stitch dtype: {dtype_value!r}")


def stitch_videos(
    dir_name: str,
    videos: Dict[str, List[Path]],
    max_control_points: int,
    lfo: int = None,
    rfo: int = None,
    game_id: str = None,
    project_file_name: str = "hm_project.pto",
    blend_mode: str = "multiblend",
    start_frame_number: int = 0,
    max_frames: int = None,
    batch_size: int = 1,
    show: bool = False,
    show_scaled: Optional[float] = None,
    output_stitched_video_file: str = os.path.join(".", "stitched_output.mkv"),
    decoder_device: Optional[torch.device] = None,
    remapping_device: torch.device = torch.device("cuda", 0),
    encoder_device: torch.device = torch.device("cpu"),
    ignore_clip_box: bool = True,
    cache_size: int = 4,
    dtype: torch.dtype = torch.float,
    start_frame_time: Optional[str] = None,
    stitch_frame_time: Optional[str] = None,
    force: Optional[bool] = False,
    minimize_blend: bool = True,
    python_blender: bool = False,
    configure_only: bool = False,
    lowmem: bool = False,
    post_stitch_rotate_degrees: Optional[float] = None,
    camera_ui: int = 0,
    args: Optional[argparse.Namespace] = None,
):
    from hmlib.config import (
        get_clip_box,
        get_config,
        load_yaml_files_ordered,
        normalize_runtime_config,
        resolve_global_refs,
    )
    from hmlib.tracking_utils.timer import Timer
    from hmlib.ui import Shower
    from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
    from hmlib.utils.image import image_height, image_width, resize_image
    from hmlib.utils.progress_bar import ProgressBar, ScrollOutput, convert_hms_to_seconds
    from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH

    AspenNet = globals().get("AspenNet")
    if AspenNet is None:
        from hmlib.aspen import AspenNet as ImportedAspenNet

        AspenNet = ImportedAspenNet
        import hmlib.hm_transforms  # noqa: F401
        import hmlib.transforms  # noqa: F401
    else:
        try:
            import hmlib.hm_transforms  # noqa: F401
            import hmlib.transforms  # noqa: F401
        except ModuleNotFoundError:
            pass

    configure_video_stitching = globals().get("configure_video_stitching")
    if configure_video_stitching is None:
        from hmlib.stitching.configure_stitching import (
            configure_video_stitching as ImportedConfigureVideoStitching,
        )

        configure_video_stitching = ImportedConfigureVideoStitching

    BasicVideoInfo = globals().get("BasicVideoInfo")
    if BasicVideoInfo is None:
        from hmlib.video.ffmpeg import BasicVideoInfo as ImportedBasicVideoInfo

        BasicVideoInfo = ImportedBasicVideoInfo

    cuda_stream = torch.cuda.Stream(remapping_device)
    torch.cuda.synchronize()
    with torch.cuda.stream(cuda_stream):
        if configure_only:
            cache_size = 0
        decoder_type = (
            getattr(args, "video_stream_decode_method", None) if args is not None else None
        )
        if dir_name is None and game_id:
            dir_name = os.path.join(os.environ["HOME"], "Videos", game_id)
        left_vid = BasicVideoInfo(",".join(videos["left"]))
        right_vid = BasicVideoInfo(",".join(videos["right"]))
        total_frames = min(left_vid.frame_count, right_vid.frame_count)
        print(f"Total possible stitched video frames: {total_frames}")

        ignore_private_config = bool(getattr(args, "ignore_private_config", 0))
        base_cfg = get_config(
            game_id=game_id,
            resolve_globals=False,
            ignore_private_config=ignore_private_config,
        )
        aspen_cfg_all: Dict[str, Any] = load_yaml_files_ordered(
            ["config/aspen/stitching.yaml"], base=base_cfg
        )
        normalize_runtime_config(aspen_cfg_all)
        override_parser = hm_opts.parser(parser=make_parser())
        if args is not None:
            hm_opts.apply_arg_config_overrides(
                aspen_cfg_all,
                args,
                parser=override_parser,
                explicit_arg_names=getattr(args, "explicit_arg_names", None),
            )
            hm_opts.apply_config_overrides(aspen_cfg_all, getattr(args, "config_overrides", None))
            args.game_config = aspen_cfg_all
            hm_opts.persist_private_config_overrides(
                args,
                parser=override_parser,
                config=aspen_cfg_all,
                explicit_arg_names=getattr(args, "explicit_arg_names", None),
            )
            if lowmem:
                if _apply_single_lowmem_gpu_overrides(args, aspen_cfg_all):
                    dtype = torch.float16
        resolve_global_refs(aspen_cfg_all)

        stitch_cfg = get_nested_value(aspen_cfg_all, "stitching", {}) or {}
        config_stitch_frame_time = stitch_cfg.get("stitch_frame_time")
        stitch_frame_time = preferred_arg(stitch_frame_time, config_stitch_frame_time)
        blend_mode = str(stitch_cfg.get("blend_mode") or blend_mode)
        control_point_matcher = str(
            stitch_cfg.get("control_point_matcher") or "superpoint-lightglue"
        )
        mapping_backend = str(stitch_cfg.get("mapping_backend") or "nona")
        max_output_dimension = stitch_cfg.get("max_output_dimension")
        if max_output_dimension is not None:
            max_output_dimension = int(max_output_dimension)
        minimize_blend = bool(stitch_cfg.get("minimize_blend", minimize_blend))
        python_blender = bool(stitch_cfg.get("python_blender", python_blender))
        dtype = _resolve_stitch_tensor_dtype(dtype, stitch_cfg)
        post_stitch_rotate_degrees = stitch_cfg.get(
            "post_stitch_rotate_degrees", post_stitch_rotate_degrees
        )
        if start_frame_time:
            stitch_time_is_zero = stitch_frame_time is None
            if not stitch_time_is_zero and stitch_frame_time:
                try:
                    stitch_time_is_zero = convert_hms_to_seconds(str(stitch_frame_time)) <= 0
                except Exception:
                    stitch_time_is_zero = False
            if stitch_time_is_zero:
                stitch_frame_time = start_frame_time
        stitch_frame_number = 0
        if stitch_frame_time:
            seconds = convert_hms_to_seconds(str(stitch_frame_time))
            if seconds > 0:
                stitch_frame_number = int(round(seconds * left_vid.fps))
        if start_frame_time:
            assert not start_frame_number
            seconds = convert_hms_to_seconds(start_frame_time)
            if seconds > 0:
                start_frame_number = int(round(seconds * left_vid.fps))

        pto_project_file, lfo, rfo = configure_video_stitching(
            dir_name,
            video_left=str(videos["left"][0]),
            video_right=str(videos["right"][0]),
            project_file_name=project_file_name,
            left_frame_offset=lfo,
            right_frame_offset=rfo,
            base_frame_offset=stitch_frame_number,
            max_control_points=max_control_points,
            force=force,
            game_id=game_id,
            stitch_frame_time=stitch_frame_time,
            ignore_private_config=ignore_private_config,
            game_config=aspen_cfg_all,
            control_point_matcher=control_point_matcher,
            mapping_backend=mapping_backend,
            max_output_dimension=max_output_dimension,
        )

        stitch_videos = {
            "left": {
                "files": videos["left"],
                "frame_offset": lfo,
            },
            "right": {
                "files": videos["right"],
                "frame_offset": rfo,
            },
        }

        profiler = getattr(args, "profiler", None)
        if args is not None:
            setattr(args, "stitch_pto_project_file", str(pto_project_file))

        # Keep the dataloader path aligned with the actual Aspen plugin enablement.
        # The stitching graph stores this under `aspen.plugins.stitching.enabled`,
        # not `aspen.stitching.enabled`.
        use_aspen_stitching = bool(
            get_nested_value(aspen_cfg_all, "aspen.plugins.stitching.enabled", False)
        )
        if args is not None and getattr(args, "aspen_stitching", None) is not None:
            use_aspen_stitching = bool(getattr(args, "aspen_stitching"))
        if camera_ui and not use_aspen_stitching:
            raise ValueError("--camera-ui requires Aspen stitching; remove --no-aspen-stitching")
        if use_aspen_stitching and lowmem:
            _apply_stitch_buffering_defaults(aspen_cfg_all, args)

        if use_aspen_stitching:
            MOTLoadVideoWithOrig = globals().get("MOTLoadVideoWithOrig")
            if MOTLoadVideoWithOrig is None:
                from hmlib.datasets.dataset.mot_video import (
                    MOTLoadVideoWithOrig as ImportedMOTLoadVideoWithOrig,
                )

                MOTLoadVideoWithOrig = ImportedMOTLoadVideoWithOrig

            MultiDataLoaderWrapper = globals().get("MultiDataLoaderWrapper")
            if MultiDataLoaderWrapper is None:
                from hmlib.datasets.dataset.stitching_dataloader2 import (
                    MultiDataLoaderWrapper as ImportedMultiDataLoaderWrapper,
                )

                MultiDataLoaderWrapper = ImportedMultiDataLoaderWrapper

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

            game_id_name = os.path.basename(str(dir_name))
            left_loader = MOTLoadVideoWithOrig(
                path=stitch_videos["left"]["files"],
                game_id=game_id_name,
                max_frames=max_frames,
                batch_size=batch_size,
                start_frame_number=start_frame_number + lfo,
                original_image_only=True,
                dtype=torch.uint8,
                device=remapping_device,
                decoder_device=decoder_device,
                decoder_type=decoder_type,
                frame_step=frame_step_left,
                no_cuda_streams=args.no_cuda_streams,
                checkerboard_input=args.checkerboard_input,
                async_mode=not args.serial,
                prefetch_batches=args.dataset_prefetch_batches,
            )
            right_loader = MOTLoadVideoWithOrig(
                path=stitch_videos["right"]["files"],
                game_id=game_id_name,
                max_frames=max_frames,
                batch_size=batch_size,
                start_frame_number=start_frame_number + rfo,
                original_image_only=True,
                dtype=torch.uint8,
                device=remapping_device,
                decoder_device=decoder_device,
                decoder_type=decoder_type,
                frame_step=frame_step_right,
                no_cuda_streams=args.no_cuda_streams,
                checkerboard_input=args.checkerboard_input,
                async_mode=not args.serial,
                prefetch_batches=args.dataset_prefetch_batches,
            )
            data_loader = MultiDataLoaderWrapper(
                dataloaders=[left_loader, right_loader],
            )
        else:
            StitchDataset = globals().get("StitchDataset")
            if StitchDataset is None:
                from hmlib.datasets.dataset.stitching_dataloader2 import (
                    StitchDataset as ImportedStitchDataset,
                )

                StitchDataset = ImportedStitchDataset

            data_loader = StitchDataset(
                pto_project_file=pto_project_file,
                videos=stitch_videos,
                start_frame_number=start_frame_number,
                max_frames=max_frames,
                batch_size=batch_size,
                image_roi=(
                    get_clip_box(game_id=game_id, root_dir=ROOT_DIR)
                    if not ignore_clip_box
                    else None
                ),
                decoder_device=decoder_device,
                decoder_type=decoder_type,
                blend_mode=blend_mode,
                remapping_device=remapping_device,
                dtype=dtype,
                minimize_blend=preferred_arg(getattr(args, "minimize_blend", None), minimize_blend),
                python_blender=python_blender,
                post_stitch_rotate_degrees=post_stitch_rotate_degrees,
                profiler=profiler,
                no_cuda_streams=args.no_cuda_streams,
                max_blend_levels=args.max_blend_levels,
                prefetch_batches=args.dataset_prefetch_batches,
            )

        data_loader_iter = CachedIterator(iterator=iter(data_loader), cache_size=cache_size)

        frame_count = 0
        dataset_delivery_fps = 0.0

        use_progress_bar: bool = not args.no_progress_bar
        progress_bar: Optional[ProgressBar] = None
        scroll_output: Optional[ScrollOutput] = None

        shower = None
        if (show or args.show_youtube) and not use_aspen_stitching:
            shower = Shower(
                label="stitched_image",
                show_scaled=show_scaled,
                cache_on_cpu=lowmem,
                max_size=0,
                enable_local_display=bool(show),
                show_youtube=bool(args.show_youtube),
                youtube_stream_url=args.youtube_stream_url,
                youtube_stream_key=args.youtube_stream_key,
                headless_preview_host=args.headless_preview_host or "0.0.0.0",
                headless_preview_port=int(args.headless_preview_port or 0),
            )

        if use_progress_bar and not configure_only:
            total_batches = len(data_loader)
            batch_size_hint = max(1, int(getattr(data_loader, "batch_size", batch_size) or 1))
            # Prefer the underlying dataset length (if available) for an accurate
            # frame count; fall back to estimating from batches.
            total_frames = 0
            dataset = getattr(data_loader, "dataset", None)
            if dataset is not None:
                try:
                    total_frames = int(len(dataset))
                except TypeError:
                    total_frames = int(total_batches) * int(batch_size_hint) if total_batches else 0
            else:
                total_frames = int(total_batches) * int(batch_size_hint) if total_batches else 0

            def _table_callback(table_map: OrderedDict):
                processed = frame_count
                remaining = max(0, total_frames - processed)
                if total_frames:
                    table_map["Frames"] = f"{processed}/{total_frames}"
                else:
                    table_map["Frames"] = str(processed)
                if dataset_delivery_fps > 0:
                    remaining_secs = remaining / dataset_delivery_fps
                    eta = convert_seconds_to_hms(remaining_secs)
                    table_map["Stitch FPS"] = f"{dataset_delivery_fps:.2f}"
                    table_map["ETA"] = eta
                else:
                    table_map["Stitch FPS"] = "warming up"
                    table_map["ETA"] = "--:--:--"
                if shower is not None:
                    shower.update_progress_table(table_map)

            scroll_output = ScrollOutput(lines=args.progress_bar_lines)

            scroll_output.register_logger(logger)

            progress_bar = ProgressBar(
                total=total_batches,
                iterator=data_loader_iter,
                scroll_output=scroll_output,
                update_rate=args.print_interval,
                table_callback=_table_callback,
                title=game_id,
                use_curses=args.curses_progress,
                units_per_iter=batch_size_hint,
            )
            data_loader_iter = progress_bar

        # Build AspenNet-based video-output pipeline for stitching.
        # Load the stitching Aspen graph from YAML and wire in CLI-specific
        # parameters (output path, skip-final-save, frame dumping).
        aspen_graph_cfg: Dict[str, Any] = aspen_cfg_all.get("aspen", {}) or {}
        plugins_cfg: Dict[str, Any] = aspen_graph_cfg.get("plugins", {}) or {}
        if not camera_ui:
            # Avoid a worker/queue handoff for a host-side UI sink that cannot
            # do useful work when the camera UI is disabled.
            plugins_cfg.pop("stitch_ui", None)
        video_out_prep_spec: Dict[str, Any] = plugins_cfg.get("video_out_prep", {}) or {}
        video_out_prep_params: Dict[str, Any] = video_out_prep_spec.get("params", {}) or {}
        video_out_spec: Dict[str, Any] = plugins_cfg.get("video_out", {}) or {}
        video_out_params: Dict[str, Any] = video_out_spec.get("params", {}) or {}
        output_label = None
        if args is not None:
            output_label = getattr(args, "label", None) or getattr(args, "output_label", None)
        output_path = output_stitched_video_file or "stitched_output.mkv"
        if output_label:
            try:
                output_path = str(add_prefix_to_filename(output_path, str(output_label)))
            except Exception:
                pass
        video_out_params.setdefault("output_video_path", output_path)
        video_out_prep_spec["params"] = video_out_prep_params
        plugins_cfg["video_out_prep"] = video_out_prep_spec
        video_out_spec["params"] = video_out_params
        plugins_cfg["video_out"] = video_out_spec
        aspen_graph_cfg["plugins"] = plugins_cfg

        # For stitching we want to preserve the full panorama resolution, so
        # disable cropping in the camera pipeline by default.
        apply_camera_spec = plugins_cfg.get("apply_camera", {}) or {}
        apply_camera_params = apply_camera_spec.get("params", {}) or {}
        apply_camera_params.setdefault("crop_output_image", False)
        apply_camera_params.setdefault("crop_play_box", False)
        apply_camera_params.setdefault("end_zones", False)
        apply_camera_spec["params"] = apply_camera_params
        plugins_cfg["apply_camera"] = apply_camera_spec

        work_dir = os.path.join(".", "output_workdirs", (game_id or "stitch"))
        if args is not None and hasattr(args, "work_dir") and args.work_dir:
            work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)

        aspen_shared: Dict[str, Any] = {
            "device": encoder_device,
            "work_dir": work_dir,
            "progress_bar": progress_bar,
        }
        if profiler is not None:
            aspen_shared["profiler"] = profiler
        if args is not None:
            aspen_shared["game_id"] = getattr(args, "game_id", None)
            aspen_shared["game_config"] = getattr(args, "game_config", None)
            aspen_shared["game_dir"] = dir_name
            aspen_shared["output_label"] = output_label
            aspen_shared["camera_ui"] = int(camera_ui)
        aspen_name = game_id or "stitch"
        aspen_net = AspenNet(aspen_name, aspen_graph_cfg, shared=aspen_shared)
        aspen_net = aspen_net.to(encoder_device)

        try:
            start = None

            dataset_timer = Timer()
            with (
                (
                    profiler
                    if (profiler is not None and profiler.enabled)
                    else contextlib.nullcontext()
                ),
                torch.no_grad(),
            ):
                for i, batch in enumerate(data_loader_iter):
                    if configure_only:
                        break
                    if use_aspen_stitching:
                        stitch_inputs = batch
                        left_item = (
                            stitch_inputs[0] if isinstance(stitch_inputs, (list, tuple)) else None
                        )
                        frame_ids = None
                        if isinstance(left_item, dict):
                            frame_ids = left_item.get("frame_ids")
                            if frame_ids is None:
                                frame_ids = left_item.get("ids")
                        if frame_ids is None:
                            frame_ids = torch.arange(i * batch_size, (i + 1) * batch_size)
                        batch_size = (
                            int(frame_ids.shape[0])
                            if isinstance(frame_ids, torch.Tensor)
                            else batch_size
                        )

                        context = {
                            "stitch_inputs": stitch_inputs,
                            "stitch_fps": data_loader.fps,
                            "fps": data_loader.fps,
                            "game_id": game_id,
                            "work_dir": work_dir,
                        }
                        aspen_net(context)
                    else:
                        stitched_image = unwrap_tensor(batch)

                        if not args.skip_final_video_save:
                            # Downscale oversized panoramas to stay within encoder
                            # limits while preserving aspect ratio.
                            width = int(image_width(stitched_image))
                            height = int(image_height(stitched_image))
                            if width > MAX_NEVC_VIDEO_WIDTH:
                                scale = float(MAX_NEVC_VIDEO_WIDTH) / float(width)
                                new_w = MAX_NEVC_VIDEO_WIDTH
                                new_h = int(height * scale)
                                # Ensure even dimensions for encoders
                                if new_w % 2 != 0:
                                    new_w -= 1
                                if new_h % 2 != 0:
                                    new_h -= 1
                                stitched_image = resize_image(
                                    stitched_image, new_width=new_w, new_height=new_h
                                )

                        if shower is not None:
                            shower.show(wrap_tensor(stitched_image), clone=False)

                        # Execute the Aspen graph to handle camera cropping and
                        # video encoding via VideoOutPlugin.
                        context = {
                            "img": wrap_tensor(stitched_image),
                            "frame_ids": torch.arange(i * batch_size, (i + 1) * batch_size),
                            "fps": data_loader.fps,
                            "game_id": game_id,
                            "work_dir": work_dir,
                        }
                        aspen_net(context)

                    # Per-iteration profiler step for gated profiling windows
                    if profiler is not None and getattr(profiler, "enabled", False):
                        profiler.step()

                    if i > 1:
                        dataset_timer.toc()
                    if (i + 1) % 50 == 0:
                        if not use_aspen_stitching:
                            assert stitched_image.ndim == 4
                        dataset_delivery_fps = batch_size / max(1e-5, dataset_timer.average_time)
                        logger.info(
                            "Dataset frame {} ({:.2f} fps)".format(
                                i * batch_size,
                                batch_size / max(1e-5, dataset_timer.average_time),
                            )
                        )
                        if i % 100 == 0:
                            dataset_timer = Timer()

                    frame_count += batch_size

                    if i == 1:
                        start = time.time()
                    dataset_timer.tic()

                    if not use_aspen_stitching:
                        del stitched_image

                if start is not None:
                    duration = time.time() - start
                    print(
                        f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
                    )
        except StopIteration:
            pass
        finally:
            data_loader.close()
            if shower is not None:
                shower.close()
            try:
                aspen_net.finalize()
            except Exception:
                pass
    return lfo, rfo


def _main(args) -> None:
    from hmlib.segm.ice_rink import main as ice_rink_main
    from hmlib.utils.gpu import GpuAllocator
    from hmlib.utils.progress_bar import convert_hms_to_seconds
    from hmlib.video.ffmpeg import BasicVideoInfo

    # `--force` implies starting from a clean stitch state.
    if args.force or args.clean:
        try:
            game_dir = (
                Path(args.video_dir)
                if args.video_dir
                else Path(os.environ["HOME"]) / "Videos" / str(args.game_id)
            )
            if args.game_id and game_dir.exists():
                clean_stitch_game_artifacts(game_id=args.game_id, game_dir=game_dir)
        except Exception as ex:
            logger.warning("Failed to clean stitch artifacts: %s", ex)
    if args.clean:
        return

    game_videos = configure_game_videos(
        game_id=args.game_id,
        write_results=not args.single_file,
        force=args.force,
        inference_scale=getattr(args, "ice_rink_inference_scale", None),
    )

    HalfFloatType = torch.float16

    if args.fp16:
        torch.set_default_dtype(HalfFloatType)

    if args.single_file or args.configure_only:
        if "left" in game_videos and game_videos["left"]:
            game_videos["left"] = game_videos["left"][:1]
        if "right" in game_videos and game_videos["right"]:
            game_videos["right"] = game_videos["right"][:1]

    # If user specified max processing time (-t/--max-time), convert to frames
    # once FPS is known from input videos. Prefer explicit --max-frames when set.
    try:
        if (getattr(args, "max_frames", None) in (None, 0)) and getattr(args, "max_time", None):
            # Use left video FPS as reference for stitched stream
            left_vid = BasicVideoInfo(",".join(game_videos["left"]))
            seconds = convert_hms_to_seconds(args.max_time)
            if seconds > 0 and left_vid.fps > 0:
                args.max_frames = int(seconds * left_vid.fps)
                logger.info(
                    "Limiting processing to %s seconds -> %d frames (fps=%.3f)",
                    args.max_time,
                    args.max_frames,
                    left_vid.fps,
                )
    except Exception as e:
        logger.warning("Failed converting max-time to frames: %s", e)
    # Initialize lightweight profiler and attach to args for downstream use (same pattern as hmtrack.py)
    profiler = None
    try:
        from hmlib.utils.profiler import build_profiler_from_args

        # Use a per-game profiler directory under output_workdirs/<game_id>/profiler
        results_folder = os.path.join(".", "output_workdirs", args.game_id or "stitch")
        os.makedirs(results_folder, exist_ok=True)
        args.work_dir = results_folder
        default_prof_dir = os.path.join(results_folder, "profiler")
        profiler = build_profiler_from_args(args, save_dir_fallback=default_prof_dir)
    except Exception:
        profiler = None
    setattr(args, "profiler", profiler)

    gpu_allocator = GpuAllocator(gpus=args.gpus.split(","))
    assert not args.start_frame_offset
    remapping_device = torch.device("cuda", gpu_allocator.allocate_fast())
    if args.multi_gpu:
        encoder_device = torch.device("cuda", gpu_allocator.allocate_modern())
        decoder_device = (
            torch.device(args.decoder_device) if args.decoder_device else remapping_device
        )
    else:
        encoder_device, decoder_device = remapping_device, remapping_device
    if args.encoder_device:
        encoder_device = torch.device(args.encoder_device)
    if args.decoder_device:
        decoder_device = torch.device(args.decoder_device)
    with torch.no_grad():
        stitch_videos(
            args.video_dir,
            videos=game_videos,
            lfo=args.lfo,
            rfo=args.rfo,
            start_frame_time=args.start_frame_time,
            stitch_frame_time=args.stitch_frame_time,
            batch_size=args.batch_size,
            project_file_name=args.project_file,
            game_id=args.game_id,
            show=args.show_image,
            show_scaled=args.show_scaled,
            max_frames=args.max_frames,
            output_stitched_video_file=args.output_file,
            blend_mode=args.blend_mode,
            ignore_clip_box=True,
            cache_size=0,
            remapping_device=remapping_device,
            decoder_device=decoder_device,
            encoder_device=encoder_device,
            dtype=HalfFloatType if args.fp16 else torch.float,
            force=args.force,
            minimize_blend=not args.no_minimize_blend,
            python_blender=args.python_blender,
            max_control_points=args.max_control_points,
            configure_only=args.configure_only,
            lowmem=gpu_allocator.is_single_lowmem_gpu(),
            post_stitch_rotate_degrees=getattr(args, "stitch_rotate_degrees", None),
            camera_ui=int(args.camera_ui or 0),
            args=args,
        )

    if args.configure_only:
        # Configure the rink mask as well
        ice_rink_main(
            args,
            device=(
                decoder_device if not gpu_allocator.is_single_lowmem_gpu() else torch.device("cpu")
            ),
        )


def main() -> None:
    parser = hm_opts.parser(parser=make_parser())
    args = parser.parse_args()
    args.explicit_arg_names = hm_opts.collect_explicit_arg_names(parser)
    if args.smoke_test:
        return _run_smoke_test(args)
    args = hm_opts.init(args, parser=parser)
    _main(args)


if __name__ == "__main__":
    main()
    print("Done.")
