"""Resolve desktop player-analysis presets to durable, composable Aspen graphs."""

from pathlib import Path

from hmlib.config import ROOT_DIR

BASE_GRAPHS = {
    "saved": "tracking",
    "tracking": "tracking",
    "jersey": "tracking",
    "pose": "tracking_pose",
    "pose-jersey": "tracking_pose_jersey",
    "pose-actions": "tracking_pose",
    "pose-jersey-actions": "tracking_pose_jersey",
}

NATIVE_POSE_CONFIG = "config/models/body_2d_keypoint/rtmpose/coco/rtmpose-m_coco17-256x192.py"
NATIVE_POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
NATIVE_POSE_SHA256 = "5e55be2a03f6e5dcd14d088afc4ae5afe94a4f9de93c22e5deb725ad0eee899d"


def graph_paths(analysis: str) -> list[Path]:
    if analysis not in BASE_GRAPHS:
        raise ValueError(f"Unknown saved player analysis: {analysis}")
    directory = Path(ROOT_DIR) / "config" / "aspen"
    paths = [directory / f"{BASE_GRAPHS[analysis]}.yaml"]
    if analysis == "jersey":
        paths.append(directory / "player_jerseys.yaml")
    if "actions" in analysis:
        paths.append(directory / "player_actions.yaml")
    return paths


def selected_analysis(config: dict) -> str:
    """Keep presets, while custom mode independently selects the three models."""
    run = config.get("hm_ui", {}).get("run", {})
    analysis = run.get("analytics", "saved")
    analytics = config.get("pipeline", {}).get("player-analytics", {})
    if analysis == "custom" or (
        analysis == "saved"
        and any("enable" in analytics.get(name, {}) for name in ("pose", "jersey", "action"))
    ):
        enabled = {
            name: analytics.get(name, {}).get("enable", False)
            for name in ("pose", "jersey", "action")
        }
        if enabled["action"] and not enabled["pose"]:
            raise ValueError("Activity recognition requires pose estimation")
        names = [name for name in ("pose", "jersey") if enabled[name]]
        if enabled["action"]:
            names.append("actions")
        analysis = "-".join(names) or "tracking"
    if analysis not in BASE_GRAPHS:
        raise ValueError(f"Unknown saved player analysis: {analysis}")
    if (
        "jersey" in analysis
        and analytics.get("jersey", {}).get("roi-mode", "bbox") == "pose"
        and "pose" not in analysis
    ):
        raise ValueError("Fresh-pose jersey crops require pose estimation")
    needs_coco = "actions" in analysis or (
        "jersey" in analysis and analytics.get("jersey", {}).get("roi-mode", "bbox") == "pose"
    )
    if needs_coco and run.get("pose_model") == "harpet":
        raise ValueError(
            "Activity recognition and fresh-pose jersey crops require COCO17 pose; select coco or a compatible custom model"
        )
    if "actions" in analysis:
        cadence = analytics.get("pose", {}).get("rate-hz", 10)
        if cadence != "fps" and (isinstance(cadence, bool) or float(cadence) < 10):
            raise ValueError("Activity recognition requires pose at 10 Hz or faster")
    return analysis


def apply_analytics_preferences(config: dict) -> None:
    """Apply the same model and crop choices to preparation, inspection and runs."""
    run = config.get("hm_ui", {}).get("run", {})
    plugins = config.get("aspen", {}).get("plugins", {})
    pose = plugins.get("pose_factory")
    selection = run.get("pose_model", "saved")
    analysis = selected_analysis(config)
    roi_mode = (
        config.get("pipeline", {})
        .get("player-analytics", {})
        .get("jersey", {})
        .get("roi-mode", "bbox")
    )
    if (
        "actions" in analysis or ("jersey" in analysis and roi_mode == "pose")
    ) and selection == "saved":
        selection = "native"
    if pose is not None and selection != "saved":
        if selection == "native":
            values = (NATIVE_POSE_CONFIG, NATIVE_POSE_CHECKPOINT)
        elif selection in {"harpet", "coco"}:
            model = config["model"][f"pose_{selection}"]
            values = (model["config"], model["checkpoint"])
        elif selection == "custom":
            values = (run.get("pose_config"), run.get("pose_checkpoint"))
            if not all(values):
                raise ValueError("Custom pose requires a model config and checkpoint")
        else:
            raise ValueError(f"Unknown pose model: {selection}")
        pose.setdefault("params", {}).update(pose_config=values[0], pose_checkpoint=values[1])
        pose["params"]["pose_checkpoint_sha256"] = (
            NATIVE_POSE_SHA256 if selection == "native" else None
        )
    jerseys = plugins.get("jersey_numbers")
    if jerseys is not None:
        mode = (
            config.get("pipeline", {})
            .get("player-analytics", {})
            .get("jersey", {})
            .get("roi-mode", "bbox")
        )
        if mode not in {"bbox", "pose"}:
            raise ValueError("Jersey crop mode must be bbox or pose")
        jerseys.setdefault("params", {})["roi_mode"] = mode
        if "save_tracking" in plugins:
            plugins["jersey_save_join"] = {
                "class": "hmlib.aspen.plugins.join_plugin.JoinPlugin",
                "depends": ["tracker", "jersey_numbers"],
                "params": {
                    "required_plugins": ["tracker", "jersey_numbers"],
                    "output_key": "jersey_tracking_joined",
                },
            }
            plugins["save_tracking"]["depends"] = ["jersey_save_join"]
        if "play_tracker" in plugins:
            plugins["play_tracker"].setdefault("params", {})["plot_jersey_numbers"] = False
    for target, node_name, class_name in (
        ("jersey", "jersey_numbers", "ParseqJerseyPlugin"),
        ("action", "actions", "TemporalActionPlugin"),
    ):
        node = plugins.get(node_name)
        if node is None:
            continue
        selection = run.get(f"{target}_model", "native")
        if selection not in {"native", "custom"}:
            raise ValueError(f"Unknown {target} model: {selection}")
        path = run.get(f"{target}_model_path") if selection == "custom" else None
        if selection == "custom" and not path:
            raise ValueError(f"Custom {target} requires a compatible ONNX model path")
        node["class"] = f"hmlib.aspen.plugins.native_analytics_plugin.{class_name}"
        node.setdefault("params", {})["model_path"] = path
