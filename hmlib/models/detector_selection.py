"""Shared detector presets for hmtrack and game comparisons."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from hmlib.log import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("deployed", "trained", "distilled", "hockeyref", "hockey-roles")

#: Published `hockeyref` weights, pinned by digest.
#:
#: Unlike the other trained profiles this one does not look in
#: `openmm/work_dirs/`: those weights only exist on the machine that ran the
#: fine-tune, so a profile resolved from them is unusable anywhere else. The
#: published file is the same checkpoint with the optimizer, EMA copy and
#: training logs removed -- 104 MB rather than 210 MB, and byte-identical in
#: the only part inference reads. The digest is verified on download and on
#: every reuse of the cached copy, because a truncated download is otherwise
#: indistinguishable from a working one until the weights load as noise.
HOCKEYREF_WEIGHTS_URL = (
    "https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/"
    "hm_hockeyref_yolov8_m_1984_736-1431d6e713364e14.pth"
)
HOCKEYREF_WEIGHTS_SHA256 = "1431d6e713364e14d7f2330485ac9a4d5d607c2982a35267fd8a241a2db4e74d"

HOCKEY_ROLES_WEIGHTS_URL = (
    "https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/"
    "hm_hockey_roles_yolov8_m_1408_544_epoch10-0463621f21d9e847.pth"
)
HOCKEY_ROLES_WEIGHTS_SHA256 = "0463621f21d9e847bead9f3a8392c8da6a70ec076a8a03fd9ff833daa3d50a25"
PROFILE_GOALIE_LABEL = {"hockey-roles": 1}

#: Class index -> name for each profile's detector head. Everything except
#: the role profiles is a single-class person detector. Role profiles carry
#: their own class order so overlay and camera filtering cannot drift.
PROFILE_CLASSES = {
    "deployed": ("person",),
    "trained": ("person",),
    "distilled": ("person",),
    "hockeyref": ("player", "referee"),
    "hockey-roles": ("person", "goalie", "official"),
}

#: Detector class index that denotes a match official, per profile. None means
#: the profile cannot distinguish one.
PROFILE_REFEREE_LABEL = {
    "deployed": None,
    "trained": None,
    "distilled": None,
    "hockeyref": 1,
    "hockey-roles": 2,
}

#: Input size a profile must be served at, when it is sensitive to one. None
#: leaves the graph's own `HmResize` alone.
#:
#: `hockeyref` pins 1984x736 because that is what it was fine-tuned at and the
#: player/referee split does not survive the graph's faster default. Measured
#: on one tv-14-2-p1 frame: at 1984x736 the model finds 9 players and the 2
#: real officials; at the graph's 1632x608 the same weights report 7 players
#: and 4 officials, badging navy skaters as referees. Stripes are fine,
#: high-frequency texture -- an 18% smaller input is where that signal goes.
PROFILE_INPUT_SCALE = {
    "deployed": None,
    "trained": None,
    "distilled": None,
    "hockeyref": (1984, 736),
    "hockey-roles": (1408, 544),
}


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pinned_checkpoint(url: str, sha256: str, filename: str) -> str:
    """Return a cached copy of `url`, downloading it once and verifying it.

    A cached file whose digest does not match is removed and fetched again: a
    download interrupted partway leaves a plausible-looking file behind, and
    silently loading half a checkpoint produces a model that detects nothing
    rather than an error anyone can act on.
    """
    import torch

    target = Path(torch.hub.get_dir()) / "checkpoints" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if _file_sha256(target) == sha256:
            return str(target)
        logger.warning("Cached detector checkpoint %s failed its digest; downloading again", target)
        target.unlink()
    logger.info("Downloading detector checkpoint %s -> %s", url, target)
    torch.hub.download_url_to_file(url, str(target), progress=True)
    actual = _file_sha256(target)
    if actual != sha256:
        target.unlink(missing_ok=True)
        raise ValueError(f"Detector checkpoint {url} has digest {actual}, expected {sha256}")
    return str(target)


def default_checkpoint(profile: str) -> str:
    """Pick a validated checkpoint, never silently substitute the last epoch."""
    if profile == "hockey-roles":
        return download_pinned_checkpoint(
            HOCKEY_ROLES_WEIGHTS_URL,
            HOCKEY_ROLES_WEIGHTS_SHA256,
            Path(HOCKEY_ROLES_WEIGHTS_URL).name,
        )
    if profile == "hockeyref":
        return download_pinned_checkpoint(
            HOCKEYREF_WEIGHTS_URL,
            HOCKEYREF_WEIGHTS_SHA256,
            Path(HOCKEYREF_WEIGHTS_URL).name,
        )
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
        # Role profiles have a wider head; class order travels with the preset.
        head_classes = len(PROFILE_CLASSES[profile])
        model["bbox_head"]["head_module"]["num_classes"] = head_classes
        model["train_cfg"]["assigner"]["num_classes"] = head_classes
        if head_classes > 1:
            # One skater is either a player or an official, never both, so each
            # box must take its argmax class. `multi_label` lets a single box
            # emit a detection per class above threshold -- harmless on the
            # 80-class COCO head where only `person` ever fired, but with two
            # person-like classes it produces a duplicate box for the same
            # skater under both labels, and the tracker then badges whichever
            # duplicate it happened to keep.
            test_cfg = model.setdefault("test_cfg", {})
            test_cfg["multi_label"] = False
            # Class-aware NMS still lets two *different* anchors claim the same
            # skater under different labels, which `multi_label` alone does not
            # catch. Suppressing across classes removes that last duplicate;
            # the cost is that a player and an official overlapping above the
            # 0.7 IoU threshold collapse to one box, which is rarer than the
            # ghost tracks it prevents.
            test_cfg.setdefault("nms", {})["class_agnostic"] = True
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


def _apply_input_scale(game_config: dict, profile: str) -> None:
    """Serve a profile at the input size it was trained for.

    The graph's `HmResize` is a throughput choice, not a property of any one
    model, so a profile that is sensitive to input size has to say so and have
    it enforced here. Silently serving a detector below its training
    resolution degrades it in ways that look like a bad model rather than a
    bad config.
    """
    scale = PROFILE_INPUT_SCALE.get(profile)
    if scale is None:
        return
    pipeline = (game_config.get("aspen") or {}).get("inference_pipeline")
    if not isinstance(pipeline, list):
        # A malformed graph, not a caller passing the wrong type: the message is
        # what the operator has to act on, so ValueError is the right class.
        raise ValueError(  # noqa: TRY004
            f"Detector profile {profile!r} must be served at {scale[0]}x{scale[1]}, "
            "but the graph has no aspen.inference_pipeline to pin"
        )
    resizes = [t for t in pipeline if isinstance(t, dict) and t.get("type") == "HmResize"]
    if not resizes:
        raise ValueError(
            f"Detector profile {profile!r} must be served at {scale[0]}x{scale[1]}, "
            "but the graph's inference_pipeline has no HmResize stage"
        )
    for transform in resizes:
        transform["img_scale"] = list(scale)

    # HmPad is paired with the resize and must grow with it, or padding to the
    # old smaller canvas asserts. Its `size` is H-W ordered, the reverse of
    # img_scale's W-H.
    width, height = scale
    for transform in pipeline:
        if (
            isinstance(transform, dict)
            and transform.get("type") == "HmPad"
            and transform.get("size") is not None
        ):
            transform["size"] = [height, width]


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

    # Tell the overlay which detector class index means "official". Only the
    # role profiles have one, and the index is a property of the trained
    # head, so it travels with the selection rather than being configured
    # independently and drifting out of step with it.
    #
    # Written straight onto the trunk params, not into `plot.*`: hmtrack
    # resolves GLOBAL.* references in main() before _main() calls this, so a
    # `GLOBAL.plot.referee_label` in the graph would already have resolved to
    # null and the badge would silently never draw.
    referee_label = PROFILE_REFEREE_LABEL.get(selected)
    for plugin in plugins.values():
        if not isinstance(plugin, dict):
            continue
        cls = str(plugin.get("class", ""))
        # The overlay badges officials, the detector trunk caps how many of
        # them a frame may contain, and the team trunk keeps them out of the
        # clustering -- they are on neither side, and with this head they are
        # a fifth of the crops. All three need the index, and none can read it
        # from `plot.*` because GLOBAL.* is resolved before this runs.
        if (
            "OverlayPlugin" in cls
            or "DetectorInferencePlugin" in cls
            or "TeamAffiliationPlugin" in cls
        ):
            plugin.setdefault("params", {})["referee_label"] = referee_label

        if "OverlayPlugin" in cls:
            plugin.setdefault("params", {})["goalie_label"] = PROFILE_GOALIE_LABEL.get(selected)
    game_config.setdefault("detector", {})["role_labels"] = {
        "referee": referee_label,
        "goalie": PROFILE_GOALIE_LABEL.get(selected),
    }

    _apply_input_scale(game_config, selected)

    identity = detector_identity(spec)
    for backend, field, extension in (("trt", "engine", "engine"), ("onnx", "path", "onnx")):
        options = params.setdefault(backend, {})
        path = Path(options.get(field) or f"output_workdirs/{args.game_id}/detector.{extension}")
        # A prepared artifact already carries this identity. Preserve it when
        # reopening that exact model, including its calibrated INT8 variant.
        if path.stem.endswith((f"-{identity}", f"-{identity}.int8")):
            options[field] = str(path)
        elif path.stem.endswith(".int8") and (path.parent / "ready.json").is_file():
            raise ValueError(
                "The prepared INT8 detector does not match the selected model; "
                "prepare the selected model before running it"
            )
        else:
            options[field] = str(path.with_name(f"{path.stem}-{identity}{path.suffix}"))
    if not args.compare_detectors:
        return
    # `distilled` and `hockeyref` are opt-in: both need a locally trained
    # checkpoint, so sweeping them into every comparison would fail on a
    # checkout that has not trained them.
    opt_in = {"distilled": args.compare_distilled, "hockeyref": False, "hockey-roles": False}
    models = {
        name: spec if name == selected else resolve_detector(name, deployed, overrides.get(name))
        for name in PROFILES
        if name not in opt_in or opt_in[name] or selected == name
    }
    # All models use the same PyTorch head NMS and precision for this comparison.
    params["trt"]["enable"] = False
    params["onnx"]["enable"] = False
    params["nms_backend"] = "head"
    params["static_detections"] = {"enable": False}
    plugins["detector"][
        "class"
    ] = "hmlib.aspen.plugins.detector_compare_plugin.DetectorComparePlugin"
    plugins["detector"]["params"] = {
        "models": models,
        "selected": selected,
        "output_dir": args.compare_detectors,
        "annotations": args.detector_annotations,
        "sample_every": args.detector_sample_every,
        "preview_frames": args.detector_preview_frames,
        "threshold": args.detector_review_threshold,
    }
    game_config["aspen"].setdefault("pipeline", {}).update(
        threaded=False, graph=False, cuda_graph=False
    )
