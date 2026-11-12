"""Baseline-only startup and compatibility with old UI presets."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hmlib.config import recursive_update
from hmlib.runtime_controls import apply_motion_overrides, normalize_control_layer


def should_apply_baseline_motion_without_a_ui_or_sidecar():
    baseline = yaml.safe_load(
        (Path(__file__).parents[1] / "hmlib/config/baseline.yaml").read_text()
    )
    camera = baseline["rink"]["camera"]
    camera["motion_overrides"].update(max_speed_x=12.3, max_accel_y=4.5)
    fast = SimpleNamespace(max_speed_x=20, max_accel_y=2)
    follower = SimpleNamespace(max_speed_x=10, max_accel_y=1)
    apply_motion_overrides(camera, fast, follower)
    assert vars(fast) == {"max_speed_x": 20, "max_accel_y": 2}
    assert vars(follower) == {"max_speed_x": 12.3, "max_accel_y": 4.5}
    camera["motion_overrides"]["apply_to_fast_box"] = True
    apply_motion_overrides(camera, fast, follower)
    assert vars(fast) == vars(follower)


def should_migrate_each_layer_without_masking_legacy_game_overrides():
    user = {"video_out": {"tone": {"exposure": 0.4}}}
    game = {
        "hstream_ui": {
            "camera_controls": {"Exposure_x100": 73},
            "playback_start_time": "01:02:03.456",
        }
    }
    before = deepcopy(game)
    result = recursive_update(user, game)
    assert result["video_out"]["tone"]["exposure"] == 0.73
    assert result["playback"]["start_seconds"] == 3723.456
    assert "hstream_ui" not in result
    assert game == before
    game["video_out"] = {"tone": {"exposure": 0}}
    assert normalize_control_layer(game)["video_out"]["tone"]["exposure"] == 0


@pytest.mark.parametrize("value", ["00:60:00", "00:00:-1", ".nan", "1x"])
def should_reject_malformed_legacy_playback(value):
    with pytest.raises(ValueError):
        normalize_control_layer({"hstream_ui": {"playback_start_time": value}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def should_reject_invalid_canonical_tone(value):
    with pytest.raises((ValueError, TypeError)):
        normalize_control_layer({"video_out": {"tone": {"exposure": value}}})


def should_preserve_python_color_overlays_and_canonical_same_layer_precedence():
    base = normalize_control_layer({"video_out": {"tone": {"exposure": 0.2}}})
    result = recursive_update(base, {"rink": {"camera": {"color": {"exposure_ev": 0.4}}}})
    assert result["video_out"]["tone"]["exposure"] == 0.8
    assert result["rink"]["camera"]["color"]["exposure_ev"] == 0.4
    result = recursive_update(
        result,
        {
            "video_out": {"tone": {"exposure": 0.6, "lift_shadow_black_point": "false"}},
            "rink": {"camera": {"color": {"exposure_ev": 0.1}}},
        },
    )
    assert result["rink"]["camera"]["color"]["exposure_ev"] == 0.3
    assert result["rink"]["camera"]["color"]["shadow_lift_black_point"] is False
    saved = normalize_control_layer(result, derive_python=False)
    assert "color" not in saved.get("rink", {}).get("camera", {})
    assert normalize_control_layer({"rink": {"camera": {"color": {"exposure_ev": None}}}}) == {}


@pytest.mark.parametrize(
    "explicit,start,frame,expected",
    [
        ([], None, 0, "42.0"),
        (["start_frame_time"], "0", 0, "0"),
        (["start_frame"], None, 0, None),
        (["start_frame"], None, 30, None),
    ],
)
def should_resolve_direct_worker_start_without_overwriting_explicit_zero(
    explicit, start, frame, expected
):
    from hmlib.runtime_controls import apply_worker_runtime_preferences

    config = {
        "playback": {"start_seconds": 42, "realtime": False},
        "telemetry": {"enabled": False},
        "aspen": {"plugins": {"capture": {"class": "module.SaveTrackingPlugin", "enabled": True}}},
    }
    args = SimpleNamespace(explicit_arg_names=explicit, start_frame_time=start, start_frame=frame)
    apply_worker_runtime_preferences(config, args)
    assert args.start_frame_time == expected
    assert config["aspen"]["plugins"]["capture"]["enabled"] is False


def should_load_canonical_tone_for_direct_python_consumers_and_save_only_canonical(
    tmp_path, monkeypatch
):
    import hmlib.config as configuration

    game = tmp_path / "test-game"
    game.mkdir()
    monkeypatch.setitem(
        configuration.save_private_config.__globals__, "GAME_DIR_BASE", str(tmp_path)
    )
    configuration.save_private_config(
        "test-game",
        {"rink": {"camera": {"color": {"exposure_ev": 0.4, "shadow_lift_black_point": False}}}},
        verbose=False,
    )
    saved = yaml.safe_load((game / "config.yaml").read_text())
    assert saved == {"video_out": {"tone": {"exposure": 0.8, "lift_shadow_black_point": False}}}
    loaded = configuration.get_game_config_private("test-game")
    assert loaded["rink"]["camera"]["color"]["exposure_ev"] == 0.4


def should_preserve_explicit_shared_job_capture_over_baseline_telemetry():
    from hmlib.runtime_controls import apply_worker_runtime_preferences

    config = {
        "playback": {"start_seconds": 0, "realtime": False},
        "telemetry": {"enabled": False},
        "hm_shared_ui": {"telemetry": True},
        "aspen": {"plugins": {"capture": {"class": "module.SaveTrackingPlugin", "enabled": True}}},
    }
    args = SimpleNamespace(explicit_arg_names=[], start_frame_time=None, start_frame=0)
    apply_worker_runtime_preferences(config, args)
    assert config["aspen"]["plugins"]["capture"]["enabled"] is True
