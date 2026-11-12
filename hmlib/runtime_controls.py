"""Canonical shared runtime controls and compatibility for saved UI layers."""

from __future__ import annotations

import copy
import math

CONTROL_KEYS = {
    "Apply_To_Fast_Box": ("rink.camera.motion_overrides.apply_to_fast_box", 1, True),
    "Apply_To_Follower_Box": ("rink.camera.motion_overrides.apply_to_follower_box", 1, True),
    "Max_Speed_X_x10": ("rink.camera.motion_overrides.max_speed_x", 10, False),
    "Max_Speed_Y_x10": ("rink.camera.motion_overrides.max_speed_y", 10, False),
    "Max_Accel_X_x10": ("rink.camera.motion_overrides.max_accel_x", 10, False),
    "Max_Accel_Y_x10": ("rink.camera.motion_overrides.max_accel_y", 10, False),
    "Bring_Up_Shadows": ("video_out.tone.shadow_lift_percent", 1, False),
    "Lift_Shadow_Black_Point": ("video_out.tone.lift_shadow_black_point", 1, True),
    "Exposure_x100": ("video_out.tone.exposure", 100, False),
}
_MISSING = object()


def _get(node, path):
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _set(node, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _remove(node, path):
    key, dot, rest = path.partition(".")
    if not dot:
        node.pop(key, None)
    elif isinstance(node.get(key), dict):
        _remove(node[key], rest)
        if not node[key]:
            del node[key]


def _boolean(value):
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value in ("0", "1", "true", "false"):
        return value in ("1", "true")
    raise ValueError("Runtime control must be boolean")


def _number(value):
    if isinstance(value, bool):
        raise TypeError("Runtime numeric control cannot be boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Runtime control must be finite")
    return number


def _seconds(value):
    parts = str(value).split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError("Invalid playback start time")
    result = 0.0
    for index, part in enumerate(parts):
        number = _number(part)
        if (
            number < 0
            or (index and number >= 60)
            or (index + 1 < len(parts) and not number.is_integer())
        ):
            raise ValueError("Invalid playback start time")
        result = result * 60 + number
    return result


def normalize_control_layer(layer):
    """Normalize before merging. Explicit canonical keys win within a layer."""
    result = copy.deepcopy(layer)
    if not isinstance(layer, dict):
        return result

    def move(legacy, canonical, convert):
        value = _get(layer, legacy)
        if value is _MISSING:
            return
        if _get(layer, canonical) is _MISSING:
            _set(result, canonical, convert(value))
        _remove(result, legacy)

    for key in ("shadow_lift_percent", "lift_shadow_black_point", "exposure"):
        move(f"runtime.amd.output.{key}", f"video_out.tone.{key}", copy.deepcopy)
    for identifier, (path, divisor, boolean) in CONTROL_KEYS.items():
        move(
            "hstream_ui.camera_controls." + identifier,
            path,
            _boolean if boolean else lambda value, divisor=divisor: _number(value) / divisor,
        )
    for legacy, canonical, convert in (
        ("runtime.amd.playback_start_seconds", "playback.start_seconds", _number),
        ("hstream_ui.playback_start_time", "playback.start_seconds", _seconds),
        ("runtime.amd.output.realtime", "playback.realtime", _boolean),
        ("hstream_ui.playback_realtime", "playback.realtime", _boolean),
        ("hstream_ui.drivegpt_database.enabled", "telemetry.enabled", _boolean),
    ):
        move(legacy, canonical, convert)
    return result


def apply_motion_overrides(camera, fast, follower):
    """Apply the same absolute native-unit translation overrides as HStream."""
    motion = camera["motion_overrides"]
    limits = {}
    for key in ("max_speed_x", "max_speed_y", "max_accel_x", "max_accel_y"):
        value = _number(motion[key])
        if value < 0:
            raise ValueError("Motion override must be nonnegative: " + key)
        limits[key] = value
    for key, box in (("apply_to_fast_box", fast), ("apply_to_follower_box", follower)):
        if _boolean(motion[key]):
            for name, value in limits.items():
                if value > 0:
                    setattr(box, name, value)
