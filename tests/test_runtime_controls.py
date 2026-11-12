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
