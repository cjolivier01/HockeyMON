"""Bind the shared desktop's canonical analytics choices to Python graph models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hmlib.models.player_analytics import model_contract


def translate_shared_analytics(
    config: dict,
    game: Path,
    *,
    initial_settings: dict | None = None,
    config_directory: Path | None = None,
) -> None:
    """Mark shared choices before graph selection; native preparation supplies bundles later."""
    initial_settings = initial_settings or {}
    initial_run = initial_settings.get("hm_ui", {}).get("run", {})
    initial_plugins = initial_settings.get("aspen", {}).get("plugins", {})
    pipeline = config.setdefault("pipeline", {})
    analytics = pipeline.setdefault("player-analytics", {})
    run = config.setdefault("hm_ui", {}).setdefault("run", {})
    if "analytics" not in initial_run and not any(
        name in initial_plugins for name in ("pose_factory", "jersey_numbers", "actions")
    ):
        # The shared desktop's independent compute toggles replace an older
        # Rust-side preset; an original Python graph/preset still owns its run.
        run["analytics"] = "custom"
    selected = {}
    for feature, node in (
        ("pose", "pose_factory"),
        ("jersey", "jersey_numbers"),
        ("action", "actions"),
        ("team", "team_affiliation"),
    ):
        settings = analytics.get(feature, {})
        if not settings.get("enable", False):
            continue
        if any(key.startswith(feature + "_") for key in initial_run) or node in initial_plugins:
            continue
        if feature == "team" and any(
            key in initial_settings.get("team", {}) for key in ("enabled", "backend")
        ):
            continue
        model = settings.get("model") or model_contract(feature)["model_id"]
        if model not in {model_contract(feature)["model_id"], "custom"}:
            raise ValueError(f"Unsupported shared {feature} model: {model}")
        settings["model"] = model
        if model == "custom":
            if not settings.get("bundle"):
                raise ValueError(f"Custom {feature} requires its prepared model bundle")
            path = Path(settings["bundle"]).expanduser()
            settings["bundle"] = str(
                path.resolve()
                if path.is_absolute()
                else ((config_directory or game) / path).resolve()
            )
        selected[feature] = model
        if feature == "team":
            team = config.setdefault("team", {})
            team.update(enabled=True, backend="reidentificationnet")
            for canonical in (
                "warmup_samples",
                "refit_interval",
                "min_margin",
                "min_lightness_gap",
                "min_cluster_share",
                "acquisition_evidence",
                "switch_hysteresis",
                "min_track_height",
                "rate_hz",
            ):
                native = canonical.replace("_", "-")
                if native in settings:
                    team[canonical] = settings[native]
                elif canonical in team:
                    settings[native] = team[canonical]
                else:
                    raise ValueError(
                        f"Shared team setting {canonical} is missing from the baseline"
                    )
        else:
            run[f"{feature}_model"] = "shared"
        if feature == "pose":
            run["pose_precision"] = "saved"
    tracker = pipeline.get("tracker", {})
    if tracker.get("reid-enable", False) and not initial_run.get("reid_onnx"):
        model = tracker.get("reid-model") or (
            "custom" if tracker.get("reid-config-file") else "deepstream"
        )
        if model not in {"deepstream", "reidentificationnet-deployable-v1.2", "custom"}:
            raise ValueError(f"Unsupported shared appearance model: {model}")
        tracker["reid-model"] = model
        if model == "custom":
            if not tracker.get("reid-config-file"):
                raise ValueError("Custom appearance requires its prepared tracker configuration")
            path = Path(tracker["reid-config-file"]).expanduser()
            tracker["reid-config-file"] = str(
                path.resolve()
                if path.is_absolute()
                else ((config_directory or game) / path).resolve()
            )
        selected["reid"] = model
    config["hm_ui"]["shared_analytics"] = selected


def shared_pose_parameters(config: dict) -> dict:
    from hmlib.ui.analytics_graph import NATIVE_POSE_CONFIG

    settings = config["pipeline"]["player-analytics"]["pose"]
    return {
        "pose_config": NATIVE_POSE_CONFIG,
        "pose_checkpoint": None,
        "pose_checkpoint_sha256": None,
        "shared_model": {"model": settings["model"], "bundle": settings.get("bundle")},
        "onnx": {"enable": False},
        "trt": {"enable": False},
    }


def apply_prepared_shared_analytics(config: dict, prepared: dict) -> None:
    """Rebind accepted assets after preparation or a seek rebuild, without replacing controls."""
    selected = config.get("hm_ui", {}).get("shared_analytics", {})
    pipeline = prepared.get("pipeline", prepared)
    for feature, model in selected.items():
        destination = config["pipeline"].setdefault(
            "tracker" if feature == "reid" else "player-analytics", {}
        )
        source = (
            pipeline.get("tracker", {})
            if feature == "reid"
            else pipeline.get("player-analytics", {}).get(feature, {})
        )
        if feature != "reid":
            destination = destination.setdefault(feature, {})
        model_key = "reid-model" if feature == "reid" else "model"
        path_key = "reid-config-file" if feature == "reid" else "bundle"
        if source.get(model_key) != model or not source.get(path_key):
            raise ValueError(f"Prepared {feature} assets do not match selected model {model}")
        if (
            model == "custom"
            and Path(destination[path_key]).resolve() != Path(source[path_key]).resolve()
        ):
            raise ValueError(f"Prepared {feature} bundle changed during the run")
        if feature == "reid":
            from hmlib.models.appearance import validate_appearance_gpu

            # The native preparer resolves device inheritance together with the
            # engine. Reuse that device after seeks as well as the initial run.
            destination["gpu-id"] = validate_appearance_gpu(source.get("gpu-id"))
        destination[path_key] = deepcopy(source[path_key])
    from hmlib.ui.runtime_setup import apply_engine_preferences

    apply_engine_preferences(config)
