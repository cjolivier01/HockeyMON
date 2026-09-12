import contextlib
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# AspenNet graph runner
from hmlib.aspen import AspenNet
from hmlib.config import get_game_dir, get_nested_value
from hmlib.datasets.dataframe import find_latest_dataframe_file
from hmlib.log import logger
from hmlib.telemetry.recorder import TelemetryRecorder
from hmlib.tracking_utils.timer import Timer
from hmlib.utils import MeanTracker
from hmlib.utils.finalization import finalize_resources
from hmlib.utils.gpu import cuda_stream_scope
from hmlib.utils.image import make_channels_first
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.path import add_prefix_to_filename
from hmlib.utils.progress_bar import (
    ProgressBar,
    build_aspen_graph_renderable,
    convert_seconds_to_hms,
)


def run_mmtrack(
    model,
    config: Dict[str, Any],
    dataloader,
    postprocessor,
    progress_bar: Optional[ProgressBar] = None,
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
    no_cuda_streams: bool = False,
    track_mean_mode: Optional[str] = None,
    profiler: Any = None,
    pose_inferencer: Any = None,
    source_video_paths: Optional[List[str]] = None,
):
    mean_tracker: Optional[MeanTracker] = None
    aspen_net: Optional[AspenNet] = None
    work_dir: Optional[str] = None
    telemetry = None
    if config is None:
        config = {}
    try:
        if track_mean_mode:
            mean_tracker = MeanTracker(
                file_path="pre_detect.txt",
                mode=track_mean_mode,
            )
        cuda_stream = torch.cuda.Stream(device) if not no_cuda_streams else None
        with cuda_stream_scope(cuda_stream):
            dataloader_iterator = CachedIterator(
                iterator=iter(dataloader), cache_size=input_cache_size
            )
            # print("WARNING: Not cacheing data loader")

            #
            # Calculate some dataset stats for our progress display
            #
            batch_size = dataloader.batch_size
            fps: Optional[float] = getattr(dataloader, "fps", None)
            total_batches_in_video = len(dataloader)
            total_frames_in_video = batch_size * total_batches_in_video
            number_of_batches_processed = 0
            total_duration_str = convert_seconds_to_hms(total_frames_in_video / dataloader.fps)
            nr_tracks = 0

            if model is not None:
                model.eval()

            wraparound_timer = None
            detect_timer = None
            last_frame_id = None
            max_tracking_id = 0

            display_opt = config.get("display_plugin_profile")
            if display_opt is None:
                display_opt = get_nested_value(
                    config, "aspen.pipeline.display_plugin_profile", None
                )
            display_plugin_profile = bool(display_opt)
            graph_opt = config.get("display_aspen_graph")
            if graph_opt is None:
                graph_opt = get_nested_value(config, "aspen.pipeline.display_graph", None)
            display_aspen_graph = bool(graph_opt)

            last_aspen_timing: Optional[Dict[str, Any]] = None
            last_dataloader_time: Optional[float] = None
            plugin_names: List[str] = []
            plugin_display_names: List[str] = []

            def _format_top_timing_summary(
                timing: Optional[Dict[str, Any]],
                dataloader_time: Optional[float],
                limit: int = 8,
            ) -> Optional[str]:
                if not timing:
                    return None
                plugin_times = timing.get("plugins", {})
                if not isinstance(plugin_times, dict) or not plugin_times:
                    return None
                total_time = float(timing.get("total", 0.0) or 0.0)
                if dataloader_time is not None:
                    total_time += float(dataloader_time)
                if total_time <= 0.0:
                    return None
                entries: List[Tuple[str, float]] = []
                if dataloader_time is not None and dataloader_time > 0.0:
                    entries.append(("dataloader", float(dataloader_time)))
                entries.extend((str(name), float(value)) for name, value in plugin_times.items())
                entries.sort(key=lambda item: item[1], reverse=True)
                parts = [
                    f"{name}={100.0 * value / total_time:.1f}%"
                    for name, value in entries[:limit]
                    if value > 0.0
                ]
                return ", ".join(parts) if parts else None

            if progress_bar is not None:
                dataloader_iterator = progress_bar.set_iterator(dataloader_iterator)

                #
                # The progress table
                #
                def _table_callback(table_map: OrderedDict[Any, Any]):
                    for key in list(table_map.keys()):
                        if str(key).startswith("Pct "):
                            del table_map[key]
                    duration_processed_in_seconds = (
                        number_of_batches_processed * batch_size / dataloader.fps
                    )
                    remaining_frames_to_process = total_frames_in_video - (
                        number_of_batches_processed * batch_size
                    )
                    remaining_seconds_to_process = remaining_frames_to_process / dataloader.fps

                    if wraparound_timer is not None:
                        processing_fps = batch_size / max(1e-5, wraparound_timer.average_time)

                        table_map["HMTrack FPS"] = "{:.2f}".format(processing_fps)
                        table_map["Dataset length"] = total_duration_str
                        table_map["Processed"] = convert_seconds_to_hms(
                            duration_processed_in_seconds
                        )
                        table_map["Remaining"] = convert_seconds_to_hms(
                            remaining_seconds_to_process
                        )
                        table_map["ETA"] = convert_seconds_to_hms(
                            remaining_frames_to_process / processing_fps
                        )
                        table_map["Track count"] = str(nr_tracks)
                        table_map["Track IDs"] = str(int(max_tracking_id))
                    if display_plugin_profile and last_aspen_timing is not None:
                        plugin_times = last_aspen_timing.get("plugins", {})
                        total_time = float(last_aspen_timing.get("total", 0.0) or 0.0)
                        if last_dataloader_time is not None:
                            total_time += float(last_dataloader_time)
                        if total_time > 0.0:
                            dl_pct = (
                                100.0 * float(last_dataloader_time) / total_time
                                if last_dataloader_time is not None
                                else None
                            )
                            if dl_pct is not None and dl_pct >= 1.0:
                                table_map["Pct dataloader"] = f"{dl_pct:.1f}%"
                            ordered_plugins = (
                                plugin_display_names or plugin_names or list(plugin_times.keys())
                            )
                            for name in ordered_plugins:
                                if name not in plugin_times:
                                    continue
                                pct = 100.0 * float(plugin_times[name]) / total_time
                                if pct < 1.0:
                                    continue
                                table_map[f"Pct {name}"] = f"{pct:.1f}%"

                # Add that table-maker to the progress bar
                progress_bar.add_table_callback(_table_callback)

            game_dir = config.get("game_dir")
            if not game_dir and config.get("game_id"):
                try:
                    game_dir = get_game_dir(config.get("game_id"), assert_exists=False)
                except Exception:
                    game_dir = None
            work_dir = config.get("work_dir") or config.get("results_folder")
            tracking_data_path = (
                config.get("tracking_data_path")
                or config.get("input_tracking_data")
                or find_latest_dataframe_file(game_dir, "tracking")
            )
            detection_data_path = (
                config.get("detection_data_path")
                or config.get("input_detection_data")
                or find_latest_dataframe_file(game_dir, "detections")
            )
            pose_data_path = (
                config.get("pose_data_path")
                or config.get("input_pose_data")
                or find_latest_dataframe_file(game_dir, "pose")
            )
            action_data_path = (
                config.get("action_data_path")
                or config.get("input_action_data")
                or find_latest_dataframe_file(game_dir, "actions")
            )

            # using_precalculated_tracking = bool(tracking_data_path)
            # using_precalculated_detection = bool(detection_data_path)
            # using_precalculated_pose = bool(pose_data_path)

            # Build AspenNet if a config is provided under config['aspen']
            aspen_cfg: Optional[Dict[str, Any]] = None
            initial_args = config.get("initial_args", {}) or {}
            cfg_aspen = config.get("aspen")
            if isinstance(cfg_aspen, dict):
                aspen_cfg = dict(config.get("aspen") or {})
            if aspen_cfg:
                trunks_cfg = aspen_cfg.get("plugins", {}) or {}

                aspen_cfg["plugins"] = trunks_cfg

                # Ensure stitching plugin mapping directory defaults to the current game dir.
                # This is required for both mapping file discovery and any debug outputs
                # written relative to dir_name.
                if game_dir and "stitching" in trunks_cfg:
                    stitching_spec = trunks_cfg.get("stitching")
                    if isinstance(stitching_spec, dict):
                        stitching_params = stitching_spec.setdefault("params", {}) or {}
                        stitching_params["dir_name"] = game_dir
                        stitching_spec["params"] = stitching_params
                        trunks_cfg["stitching"] = stitching_spec

                # Apply camera controller CLI overrides if present
                if "camera_controller" in trunks_cfg:
                    cc = trunks_cfg["camera_controller"]
                    cc_params = cc.setdefault("params", {}) or {}
                    ctrl = initial_args.get("camera_controller")
                    if ctrl:
                        cc_params["controller"] = ctrl
                    model_path = initial_args.get("camera_model")
                    if model_path:
                        cc_params["model_path"] = model_path
                    win = initial_args.get("camera_window")
                    if win:
                        try:
                            cc_params["window"] = int(win)
                        except Exception:
                            pass
                    # If CLI overrides not provided, pull from rink.camera config
                    if not cc_params.get("controller"):
                        cam_ctrl = (config.get("rink") or {}).get("camera", {}).get("controller")
                        if cam_ctrl:
                            cc_params["controller"] = cam_ctrl
                    if not cc_params.get("model_path"):
                        cam_model = (config.get("rink") or {}).get("camera", {}).get("camera_model")
                        if cam_model:
                            cc_params["model_path"] = cam_model
                    if not cc_params.get("window"):
                        cam_win = (config.get("rink") or {}).get("camera", {}).get("camera_window")
                        if cam_win:
                            try:
                                cc_params["window"] = int(cam_win)
                            except Exception:
                                pass
                    cc["params"] = cc_params
                # Apply jersey trunk CLI overrides if present
                if "jersey_numbers" in trunks_cfg:
                    j = trunks_cfg["jersey_numbers"]
                    j_params = j.setdefault("params", {}) or {}

                    def set_if(name_cfg: str, name_arg: str, allow_false: bool = False):
                        val = config.get(name_arg)
                        if val is None:
                            return
                        if isinstance(val, bool) and not val and not allow_false:
                            # Only set booleans when True unless explicitly allowed
                            return
                        j_params[name_cfg] = val

                    # ROI / SAM
                    set_if("roi_mode", "jersey_roi_mode")
                    set_if("sam_enabled", "jersey_sam_enabled")
                    set_if("sam_checkpoint", "jersey_sam_checkpoint")
                    set_if("sam_model_type", "jersey_sam_model_type")
                    set_if("sam_device", "jersey_sam_device")
                    # STR
                    set_if("str_backend", "jersey_str_backend")
                    set_if("parseq_weights", "jersey_parseq_weights")
                    set_if("parseq_device", "jersey_parseq_device")
                    # Legibility
                    set_if("legibility_enabled", "jersey_legibility_enabled")
                    set_if("legibility_weights", "jersey_legibility_weights")
                    set_if("legibility_threshold", "jersey_legibility_threshold", allow_false=True)
                    # ReID
                    set_if("reid_enabled", "jersey_reid_enabled")
                    set_if("reid_backend", "jersey_reid_backend")
                    set_if("reid_backbone", "jersey_reid_backbone")
                    set_if("reid_threshold", "jersey_reid_threshold", allow_false=True)
                    set_if("centroid_reid_path", "jersey_centroid_reid_path")
                    set_if("centroid_reid_device", "jersey_centroid_reid_device")
                    j["params"] = j_params
                # Disable postprocess trunk when CLI requests no play tracking.
                if config.get("no_play_tracking") and "postprocess" in trunks_cfg:
                    pp = trunks_cfg["postprocess"]
                    if isinstance(pp, dict):
                        pp["enabled"] = False
                output_label = (
                    config.get("label")
                    or config.get("output_label")
                    or initial_args.get("label")
                    or initial_args.get("output_label")
                )
                shared = dict(
                    model=model,
                    postprocessor=postprocessor,
                    pose_inferencer=pose_inferencer,
                    fp16=fp16,
                    device=device,
                    # using_precalculated_tracking=using_precalculated_tracking,
                    # using_precalculated_detection=using_precalculated_detection,
                    plot_pose=bool(config.get("plot_pose", False)),
                    # Propagate CLI flag to BoundariesPlugin -> IceRinkSegmBoundaries(draw)
                    plot_ice_mask=bool(config.get("plot_ice_mask", False)),
                    # Boundary + identity context for plugins
                    game_id=config.get("game_id"),
                    game_dir=game_dir,
                    work_dir=work_dir,
                    tracking_data_path=tracking_data_path,
                    detection_data_path=detection_data_path,
                    pose_data_path=pose_data_path,
                    action_data_path=action_data_path,
                    original_clip_box=config.get("original_clip_box"),
                    top_border_lines=config.get("top_border_lines"),
                    bottom_border_lines=config.get("bottom_border_lines"),
                    # Full game config and CLI-derived initial args for plugins
                    game_config=config.get("game_config"),
                    source_video_paths=source_video_paths or [],
                    initial_args=config.get("initial_args"),
                    # Runtime camera UI toggle for PlayTrackerPlugin
                    camera_ui=int(initial_args.get("camera_ui") or config.get("camera_ui") or 0),
                    # Optional stitching rotation controller (e.g., StitchDataset instance)
                    stitch_rotation_controller=config.get("stitch_rotation_controller"),
                    output_label=output_label,
                    progress_bar=progress_bar,
                    mux_audio_file=config.get("mux_audio_file"),
                    mux_audio_stream=config.get("mux_audio_stream"),
                    mux_audio_offset_seconds=config.get("mux_audio_offset_seconds"),
                    mux_audio_aac_bitrate=config.get("mux_audio_aac_bitrate"),
                )
                # Optional per-plugin audit hook for debugging CUDA stream correctness.
                audit_dir = initial_args.get("audit_dir")
                if audit_dir:
                    try:
                        from hmlib.aspen.audit import AspenAuditConfig, AspenAuditHook

                        audit_plugins = initial_args.get("audit_plugins")
                        plugins = None
                        if audit_plugins:
                            plugins = [
                                p.strip() for p in str(audit_plugins).split(",") if p.strip()
                            ]
                        ref_dir = initial_args.get("audit_reference_dir")
                        ref_path = Path(ref_dir) if ref_dir else None
                        audit_cfg = AspenAuditConfig(
                            out_dir=Path(audit_dir),
                            reference_dir=ref_path,
                            plugins=plugins,
                            dump_images=bool(initial_args.get("audit_dump_images", False)),
                            fail_fast=bool(int(initial_args.get("audit_fail_fast", 1) or 0)),
                        )
                        shared["_aspen_audit"] = AspenAuditHook(audit_cfg)
                        logger.info(
                            "Aspen audit enabled: out_dir=%s ref_dir=%s",
                            audit_cfg.out_dir,
                            ref_path,
                        )
                    except Exception:
                        logger.exception("Failed to initialize Aspen audit hook")
                if profiler is not None:
                    shared["profiler"] = profiler
                aspen_name = aspen_cfg.get("name") or config.get("game_id") or "aspen"
                aspen_net = AspenNet(aspen_name, aspen_cfg, shared=shared)
                aspen_net = aspen_net.to(device)
                stages = [
                    node.module.telemetry_kind
                    for node in (aspen_net.exec_order if work_dir else [])
                    if node.module.enabled and hasattr(node.module, "telemetry_kind")
                ]
                if len(stages) != len(set(stages)):
                    raise ValueError(
                        "Telemetry requires at most one save plugin per observation kind"
                    )
                if work_dir and stages:
                    telemetry = TelemetryRecorder(
                        work_dir,
                        config.get("game_id") or Path(work_dir).name,
                        {
                            "game_config": config.get("game_config"),
                            "aspen": aspen_cfg,
                            "source_video_paths": source_video_paths or [],
                            "timestamp_source": "source frame / source FPS unless explicit pts_ns",
                            "native_replay": "unavailable in HM producer",
                        },
                        stages,
                    )
                if display_plugin_profile:
                    plugin_names = [node.name for node in aspen_net.exec_order]
                    plugin_display_names = []
                    for node in aspen_net.exec_order:
                        module = node.module
                        if getattr(module, "enabled", True) is False:
                            continue
                        if module.__class__.__name__ == "_NoOpPlugin":
                            continue
                        plugin_display_names.append(node.name)
                if display_aspen_graph and progress_bar is not None:
                    aspen_net.enable_progress_graph()

                    def _aspen_graph_panel():
                        snapshot = aspen_net.get_progress_snapshot()
                        if snapshot is None:
                            return None
                        return build_aspen_graph_renderable(snapshot)

                    progress_bar.set_extra_panel_callback(_aspen_graph_panel, title="AspenNet")
            # Optional torch profiler context spanning the run
            prof_ctx = profiler if getattr(profiler, "enabled", False) else contextlib.nullcontext()
            with prof_ctx:

                def _extract_stitch_ids(stitch_inputs: Any) -> Tuple[Optional[torch.Tensor], int]:
                    if isinstance(stitch_inputs, dict):
                        left = stitch_inputs.get("left")
                    else:
                        left = stitch_inputs[0] if stitch_inputs else None
                    if left is None:
                        return None, 0
                    ids_val = left.get("frame_ids")
                    if ids_val is None:
                        ids_val = left.get("ids")
                    if ids_val is None:
                        ids_val = left.get("img_id")
                    if isinstance(ids_val, (list, tuple)):
                        ids_val = torch.tensor(ids_val, dtype=torch.int64)
                    if isinstance(ids_val, torch.Tensor) and ids_val.is_cuda:
                        ids_val = ids_val.detach().cpu()
                    if isinstance(ids_val, torch.Tensor):
                        return ids_val, int(ids_val.shape[0])
                    if ids_val is not None:
                        try:
                            return torch.tensor(list(ids_val), dtype=torch.int64), len(ids_val)
                        except Exception:
                            pass
                    img = left.get("img")
                    if isinstance(img, torch.Tensor):
                        return None, int(img.shape[0]) if img.ndim == 4 else 1
                    return None, 0

                cur_iter = 0
                while True:
                    dataloader_start = time.perf_counter() if display_plugin_profile else None
                    try:
                        dataset_results = next(dataloader_iterator)
                    except StopIteration:
                        break
                    if display_plugin_profile:
                        last_dataloader_time = (
                            time.perf_counter() - dataloader_start
                            if dataloader_start is not None
                            else None
                        )
                    data_item = None
                    original_images = None
                    info_imgs = None
                    ids = None
                    frame_id = None
                    batch_size = 0
                    stitch_inputs = None

                    if "pano" in dataset_results:
                        data_item = dataset_results.pop("pano")
                        original_images = data_item.pop("original_images")
                        info_imgs = data_item["img_info"]
                        ids = data_item["ids"]
                        if fps:
                            data_item["fps"] = fps
                        with torch.no_grad():
                            frame_id = info_imgs["frame_id"]
                        batch_size = original_images.shape[0]
                    elif "stitch_inputs" in dataset_results:
                        stitch_inputs = dataset_results.pop("stitch_inputs")
                        ids, batch_size = _extract_stitch_ids(stitch_inputs)
                        if isinstance(ids, torch.Tensor) and ids.numel():
                            frame_id = int(ids[0].item())
                        elif ids is not None:
                            try:
                                frame_id = int(ids[0])
                            except Exception:
                                frame_id = None
                    else:
                        raise RuntimeError(
                            "Dataset results missing expected 'pano' or 'stitch_inputs'"
                        )

                    if frame_id is not None and batch_size:
                        if last_frame_id is None:
                            last_frame_id = int(frame_id)
                        else:
                            assert int(frame_id) == last_frame_id + batch_size
                            last_frame_id = int(frame_id)

                    if detect_timer is None:
                        detect_timer = Timer()

                    if aspen_net is not None:
                        # Execute the configured DAG
                        # Prepare per-iteration context
                        iter_context: Dict[str, Any] = dict(
                            device=device,
                            cuda_stream=cuda_stream,
                            detect_timer=detect_timer,
                            mean_tracker=mean_tracker,
                        )
                        if display_plugin_profile:
                            iter_context["_aspen_timing_enabled"] = True
                        if stitch_inputs is not None:
                            iter_context.update(
                                stitch_inputs=stitch_inputs,
                                stitch_data_pipeline=config.get("stitch_data_pipeline"),
                                stitch_fps=fps,
                            )
                            if fps:
                                iter_context["fps"] = float(fps)
                        else:
                            # In pano mode, upstream stitching is disabled and the dataloader
                            # already provides model inputs + data_samples. Surface those as
                            # first-class context keys rather than nesting under `data`.
                            inputs = None if data_item is None else data_item.get("img")
                            data_samples = (
                                None if data_item is None else data_item.get("data_samples")
                            )
                            hm_real_time_fps = (
                                None if data_item is None else data_item.get("hm_real_time_fps")
                            )
                            debug_rgb_stats = (
                                None if data_item is None else data_item.get("debug_rgb_stats")
                            )
                            iter_context.update(
                                original_images=make_channels_first(original_images),
                                inputs=inputs,
                                data_samples=data_samples,
                                hm_real_time_fps=hm_real_time_fps,
                                debug_rgb_stats=debug_rgb_stats,
                                ids=ids,
                                info_imgs=info_imgs,
                                frame_id=int(frame_id),
                            )
                            if fps:
                                iter_context["fps"] = float(fps)
                        if frame_id is not None and "frame_id" not in iter_context:
                            iter_context["frame_id"] = int(frame_id)
                        # Merge shared into context for plugins convenience
                        iter_context.update(aspen_net.shared)
                        if telemetry is not None:
                            iter_context["telemetry_batch"] = telemetry.new_batch()
                        if dataset_results:
                            iter_context["dataset_results"] = dataset_results

                        if getattr(profiler, "enabled", False):
                            with profiler.rf("aspen.forward"):
                                out_context = aspen_net(iter_context)
                        else:
                            out_context = aspen_net(iter_context)

                        # Async AspenNet returns None from forward()
                        if out_context is not None:
                            if display_plugin_profile:
                                last_aspen_timing = aspen_net.get_last_timing()
                            # Update stats for progress bar
                            nr_tracks = out_context.get("nr_tracks", 0)
                            if isinstance(nr_tracks, torch.Tensor):
                                try:
                                    nr_tracks = int(nr_tracks.reshape(-1)[0].item())
                                except Exception:
                                    nr_tracks = 0
                            else:
                                nr_tracks = int(nr_tracks)
                            max_tracking_id = out_context.get("max_tracking_id", 0)
                            if isinstance(max_tracking_id, torch.Tensor):
                                try:
                                    max_tracking_id = int(max_tracking_id.reshape(-1)[0].item())
                                except Exception:
                                    max_tracking_id = 0
                        elif display_plugin_profile:
                            last_aspen_timing = aspen_net.get_last_timing()
                        elif not isinstance(max_tracking_id, (int, float)):
                            try:
                                max_tracking_id = int(max_tracking_id)
                            except Exception:
                                max_tracking_id = 0
                    else:
                        # Legacy MMTracking path has been removed. An Aspen config is required.
                        raise RuntimeError(
                            "AspenNet config is required. Legacy non-Aspen pipeline has been removed."
                        )

                    if detect_timer is not None and cur_iter % 50 == 0:
                        # print(
                        logger.info(
                            "Detector time, frame {} ({:.2f} fps)".format(
                                frame_id,
                                batch_size * 1.0 / max(1e-5, detect_timer.average_time),
                            )
                        )

                    if cur_iter % 200 == 0:
                        if wraparound_timer is not None:
                            wraparound_timer.toc()
                            logger.info(
                                "Full wraparound time, frame {} ({:.2f} fps)".format(
                                    frame_id,
                                    batch_size * 1.0 / max(1e-5, wraparound_timer.average_time),
                                )
                            )
                        wraparound_timer = Timer()
                    elif wraparound_timer is None:
                        wraparound_timer = Timer()
                    else:
                        wraparound_timer.toc()
                    if display_plugin_profile and cur_iter % 200 == 0:
                        timing_summary = _format_top_timing_summary(
                            last_aspen_timing, last_dataloader_time
                        )
                        if timing_summary:
                            logger.info(
                                "Aspen timing breakdown, frame %s: %s",
                                frame_id,
                                timing_summary,
                            )

                    number_of_batches_processed += 1
                    cur_iter += 1
                    # Per-iteration profiler step for per-iter export if enabled
                    if getattr(profiler, "enabled", False):
                        profiler.step()
                    wraparound_timer.tic()
    except StopIteration:
        print("run_mmtrack reached end of dataset")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        actions = []
        if config.get("save_pose_data") and work_dir:

            def finalize_pose_file():
                Path(work_dir).mkdir(parents=True, exist_ok=True)
                pose_name = "pose.csv"
                label = config.get("label") or config.get("output_label")
                if label:
                    pose_name = str(add_prefix_to_filename(pose_name, str(label)))
                (Path(work_dir) / pose_name).touch(exist_ok=True)

            actions.append(("pose output", finalize_pose_file))
        if aspen_net is not None:
            actions.append(("Aspen pipeline", aspen_net.finalize))
            audit_hook = aspen_net.shared.get("_aspen_audit")
            close_fn = getattr(audit_hook, "close", None)
            if callable(close_fn):
                actions.append(("Aspen audit output", close_fn))
        if mean_tracker is not None:
            actions.append(("mean tracker", mean_tracker.close))
        if telemetry is not None:
            actions.append(("telemetry database", telemetry.close))
        finalize_resources(actions, primary_error=sys.exc_info()[1])
    return telemetry.path if telemetry is not None else None
