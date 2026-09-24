"""Shared detector presets for hmtrack and game comparisons."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("deployed", "trained", "distilled")


def add_detector_arguments(parser):
    parser.add_argument("--detector-model", choices=PROFILES, default=None)
    parser.add_argument(
        "--detector",
        "--detector-checkpoint",
        dest="detector",
        default=None,
        help="Checkpoint override for the selected detector (default profile: deployed).",
    )
    parser.add_argument(
        "--trained-checkpoint", default=None, help="CrowdHuman YOLOv8-M checkpoint."
    )
    parser.add_argument(
        "--distilled-checkpoint",
        default=None,
        help="YOLOv8-S checkpoint; accepts exported students or full distillation checkpoints.",
    )
    parser.add_argument(
        "--compare-detectors",
        metavar="OUTPUT_DIR",
        default=None,
        help="Compare deployed and trained detectors on identical hmtrack inputs.",
    )
    parser.add_argument("--compare-distilled", action="store_true")
    parser.add_argument("--detector-annotations", default=None, help="COCO ground-truth JSON.")
    parser.add_argument("--detector-sample-every", type=int, default=30)
    parser.add_argument("--detector-preview-frames", type=int, default=20)
    parser.add_argument("--detector-review-threshold", type=float, default=0.25)


def _local_checkpoint(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {candidate}")
    return str(candidate.resolve())


def default_checkpoint(profile: str) -> str:
    """Pick a validated checkpoint, never silently substitute the last epoch."""
    if profile == "trained":
        directory = ROOT / "openmm/work_dirs/hm_crowdhuman_yolov8_m_1408_544"
        paths = list(directory.glob("best_coco_bbox_mAP_epoch_*.pth"))
        if len(paths) == 1:
            return _local_checkpoint(paths[0])
        raise ValueError(f"Expected one best M checkpoint in {directory}; use --trained-checkpoint")
    if profile != "distilled":
        raise ValueError(f"No automatic checkpoint for {profile}")
    directory = ROOT / "openmm/work_dirs/hm_yolov8_s_distilled_crowdhuman_300e"
    ranked = []
    for path in directory.glob("topk_coco_bbox_mAP_epoch_*_score_*.pth"):
        match = re.fullmatch(r"topk_coco_bbox_mAP_epoch_(\d+)_score_(\d+\.\d+)\.pth", path.name)
        if match:
            ranked.append((float(match[2]), int(match[1]), path))
    if ranked:
        return _local_checkpoint(max(ranked)[2])
    raise ValueError(f"No ranked S checkpoints in {directory}; use --distilled-checkpoint")


def resolve_detector(profile: str, deployed_params: dict, checkpoint: str | None = None) -> dict:
    """Return the exact architecture/checkpoint consumed by DetectorFactoryPlugin."""
    import yaml

    if profile not in PROFILES:
        raise ValueError(f"Unknown detector profile: {profile}")
    if deployed_params.get("detector") is not None:
        model = copy.deepcopy(deployed_params["detector"])
    else:
        from hmlib.config import prepend_root_dir

        path = prepend_root_dir(deployed_params["detector_yaml"])
        with open(path, encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        model = copy.deepcopy(raw.get("detector", raw))

    if profile != "deployed":
        # The new checkpoints use the MMYOLO M/S architectures, independent
        # of any per-game deployment override.
        with open(
            ROOT / "hmlib/config/aspen/models/hm_crowdhuman_yolov8_m_1984_736.detector.yaml",
            encoding="utf-8",
        ) as stream:
            model = yaml.safe_load(stream)
        model["bbox_head"]["head_module"]["num_classes"] = 1
        model["train_cfg"]["assigner"]["num_classes"] = 1
        checkpoint = checkpoint or default_checkpoint(profile)
        if profile == "distilled":
            model["backbone"].update(
                deepen_factor=0.33, widen_factor=0.5, last_stage_out_channels=1024
            )
            model["neck"].update(
                deepen_factor=0.33,
                widen_factor=0.5,
                in_channels=[256, 512, 1024],
                out_channels=[256, 512, 1024],
            )
            model["bbox_head"]["head_module"].update(widen_factor=0.5, in_channels=[256, 512, 1024])
    spec = {"detector": model}
    checkpoint = checkpoint or deployed_params.get("checkpoint")
    if checkpoint:
        spec["checkpoint"] = _local_checkpoint(checkpoint)
        init_cfg = model.get("init_cfg")
        prefix = init_cfg.get("prefix") if isinstance(init_cfg, dict) else None
        if profile == "deployed":
            prefix = deployed_params.get("checkpoint_prefix") or prefix
        if prefix:
            spec["checkpoint_prefix"] = prefix
        model["init_cfg"] = None
    return spec


def detector_identity(spec: dict) -> str:
    """Separate compiled-engine caches when weights or architecture change."""
    import json

    payload = json.dumps(spec, sort_keys=True)
    checkpoint = spec.get("checkpoint") or (spec["detector"].get("init_cfg") or {}).get(
        "checkpoint"
    )
    if checkpoint and Path(checkpoint).is_file():
        stat = Path(checkpoint).stat()
        payload += f":{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def configure_detector_selection(args, game_config: dict) -> None:
    """Apply CLI selection after backend overrides, before Aspen is built."""
    if not (args.detector_model or args.detector or args.compare_detectors):
        return
    plugins = game_config.get("aspen", {}).get("plugins", {})
    if "detector_factory" not in plugins or "detector" not in plugins:
        raise ValueError("Detector selection requires an Aspen detector_factory and detector stage")
    if args.input_detection_data or args.input_tracking_data:
        raise ValueError(
            "Detector selection/comparison requires live detections; remove input CSV flags"
        )
    for name, plugin in plugins.items():
        replay = str(plugin.get("class", "")).endswith(
            ("LoadDetectionsPlugin", "LoadTrackingPlugin")
        )
        if replay and plugin.get("enabled", True):
            raise ValueError(f"Disable Aspen {name} when selecting/comparing detectors")
    params = plugins["detector_factory"].setdefault("params", {})
    plugins["detector_factory"]["enabled"] = True
    plugins["detector"]["enabled"] = True
    deployed = copy.deepcopy(params)
    selected = args.detector_model or "deployed"
    overrides = {"trained": args.trained_checkpoint, "distilled": args.distilled_checkpoint}
    spec = resolve_detector(selected, deployed, args.detector or overrides.get(selected))
    params.pop("detector_yaml", None)
    params.pop("checkpoint", None)
    params.pop("checkpoint_prefix", None)
    params.update(spec)
    identity = detector_identity(spec)
    for backend, field, extension in (("trt", "engine", "engine"), ("onnx", "path", "onnx")):
        options = params.setdefault(backend, {})
        path = Path(options.get(field) or f"output_workdirs/{args.game_id}/detector.{extension}")
        options[field] = str(path.with_name(f"{path.stem}-{identity}{path.suffix}"))
    if not args.compare_detectors:
        return
    models = {
        name: spec if name == selected else resolve_detector(name, deployed, overrides.get(name))
        for name in PROFILES
        if name != "distilled" or args.compare_distilled or selected == name
    }
    # All models use the same PyTorch head NMS and precision for this comparison.
    params["trt"]["enable"] = False
    params["onnx"]["enable"] = False
    params["nms_backend"] = "head"
    params["static_detections"] = {"enable": False}
    plugins["detector"][
        "class"
    ] = "hmlib.aspen.plugins.detector_compare_plugin.DetectorComparePlugin"
    plugins["detector"]["params"] = dict(
        models=models,
        selected=selected,
        output_dir=args.compare_detectors,
        annotations=args.detector_annotations,
        sample_every=args.detector_sample_every,
        preview_frames=args.detector_preview_frames,
        threshold=args.detector_review_threshold,
    )
    game_config["aspen"].setdefault("pipeline", {}).update(
        threaded=False, graph=False, cuda_graph=False
    )
