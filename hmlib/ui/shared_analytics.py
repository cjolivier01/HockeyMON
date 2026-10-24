"""Bind the shared desktop's canonical analytics choices to Python graph models."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from hmlib.models.player_analytics import model_contract, validate_model_gpu

_ACCEPTED_MODELS = "accepted-models.json"
_MODEL_FEATURES = {"pose", "jersey", "action", "team", "reid"}


def _model_settings(pipeline: dict, feature: str) -> tuple[dict, dict, str, str]:
    owner = pipeline.setdefault("tracker" if feature == "reid" else "player-analytics", {})
    settings = owner if feature == "reid" else owner.setdefault(feature, {})
    return (
        owner,
        settings,
        "reid-model" if feature == "reid" else "model",
        "reid-config-file" if feature == "reid" else "bundle",
    )


def write_accepted_shared_analytics(
    session: Path, game_id: str, config: dict, prepared: dict, initial_settings: dict
) -> None:
    """Freeze accepted model bindings outside YAML files the worker reloads."""
    from hmlib.ui.hstream_runtime import write_document

    selected = config.get("hm_ui", {}).get("shared_analytics", {})
    models = {}
    for feature, model in selected.items():
        owner, settings, model_key, path_key = _model_settings(prepared["pipeline"], feature)
        path = Path(settings[path_key])
        if settings[model_key] != model or not path.is_absolute():
            raise ValueError(f"Prepared {feature} binding is not resolved")
        entry = {"model": model, "path": str(path), "gpu": validate_model_gpu(owner.get("gpu-id"))}
        _, original, _, _ = _model_settings(deepcopy(initial_settings.get("pipeline", {})), feature)
        for key, name in ((model_key, "original_model"), (path_key, "original_path")):
            if key in original:
                entry[name] = original[key]
        models[feature] = entry
    write_document(
        session / _ACCEPTED_MODELS,
        {
            "version": 1,
            "session": str(session.resolve()),
            "game_id": game_id,
            "selected": selected,
            "models": models,
        },
    )


def restore_worker_shared_analytics(config: dict, game_id: str) -> None:
    """Restore only accepted model bindings after original worker YAML/CLI overrides."""
    if os.environ.get("HM_SHARED_UI_BACKEND") != "1":
        return
    session = Path(os.environ["HM_UI_SESSION_DIR"]).resolve()
    path = session / _ACCEPTED_MODELS
    selected = config.get("hm_ui", {}).get("shared_analytics", {})
    if not path.exists() and not selected:
        return
    if not path.is_file():
        raise ValueError("Missing or oversized accepted shared model snapshot")
    with path.open("rb") as stream:
        payload = stream.read(64 * 1024 + 1)
    if len(payload) > 64 * 1024:
        raise ValueError("Missing or oversized accepted shared model snapshot")
    snapshot = json.loads(payload)
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("version") != 1
        or snapshot.get("session") != str(session)
        or snapshot.get("game_id") != game_id
        or snapshot.get("selected") != selected
        or not isinstance(selected, dict)
        or not set(selected) <= _MODEL_FEATURES
        or not isinstance(snapshot.get("models"), dict)
        or set(snapshot["models"]) != set(selected)
    ):
        raise ValueError(
            "Accepted shared model snapshot does not match this session/game/selections"
        )
    for feature, model in selected.items():
        entry = snapshot["models"][feature]
        if not isinstance(entry, dict) or entry.get("model") != model:
            raise ValueError(f"Accepted {feature} model identity changed")
        accepted_path = entry.get("path")
        if not isinstance(accepted_path, str) or not Path(accepted_path).is_absolute():
            raise ValueError(f"Accepted {feature} asset path is not resolved")
        owner, destination, model_key, path_key = _model_settings(config["pipeline"], feature)
        current_model = destination.get(model_key)
        if current_model != model and not (
            "original_model" in entry and current_model == entry["original_model"]
        ):
            raise ValueError(f"Shared {feature} model changed after preparation")
        if model == "custom":
            current_path = destination.get(path_key)
            if not isinstance(current_path, str) or not (
                current_path == entry.get("original_path")
                or Path(current_path).expanduser().resolve() == Path(accepted_path).resolve()
            ):
                raise ValueError(f"Prepared {feature} bundle changed during the run")
        destination[model_key], destination[path_key] = model, accepted_path
        owner["gpu-id"] = validate_model_gpu(entry.get("gpu"))
    # Generic engine preferences ran before CLI overrides. Repeating them here
    # would erase unrelated explicit Python engine settings; update only the
    # graph parameters owned by the accepted shared selections instead.
    plugins = config.get("aspen", {}).get("plugins", {})
    if "pose" in selected and "pose_factory" in plugins:
        plugins["pose_factory"].setdefault("params", {}).update(shared_pose_parameters(config))
    for feature, node in (("jersey", "jersey_numbers"), ("action", "actions")):
        if feature in selected and node in plugins:
            analytics = config["pipeline"]["player-analytics"]
            plugins[node].setdefault("params", {}).update(
                model_path=None,
                model_bundle=analytics[feature]["bundle"],
                gpu=analytics["gpu-id"],
            )


def translate_shared_analytics(
    config: dict,
    game: Path,
    *,
    initial_settings: dict | None = None,
    config_directory: Path | None = None,
    initial_model_overrides: set[str] | None = None,
) -> None:
    """Mark shared choices before graph selection; native preparation supplies bundles later."""
    initial_settings = initial_settings or {}
    initial_model_overrides = initial_model_overrides or set()
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
        if (
            feature in initial_model_overrides
            or any(key.startswith(feature + "_") for key in initial_run)
            or node in initial_plugins
        ):
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

    analytics = config["pipeline"]["player-analytics"]
    settings = analytics["pose"]
    return {
        "pose_config": NATIVE_POSE_CONFIG,
        "pose_checkpoint": None,
        "pose_checkpoint_sha256": None,
        "shared_model": {
            "model": settings["model"],
            "bundle": settings.get("bundle"),
            "gpu": analytics.get("gpu-id"),
        },
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
        # Native preparation resolves device inheritance together with assets.
        # Preserve that accepted device when operator config is reloaded on seek.
        owner = "tracker" if feature == "reid" else "player-analytics"
        config["pipeline"][owner]["gpu-id"] = validate_model_gpu(pipeline[owner].get("gpu-id"))
        destination[path_key] = deepcopy(source[path_key])
    from hmlib.ui.runtime_setup import apply_engine_preferences

    apply_engine_preferences(config)
