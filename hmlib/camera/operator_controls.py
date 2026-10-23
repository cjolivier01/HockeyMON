"""Runtime controls shared with the desktop operator workflow.

These settings act on the existing tracker; changing a player filter must not
reconstruct the camera and discard its velocity or braking history.
"""

from __future__ import annotations

from typing import Any

from hmlib.config import get_nested_value

_FILTER_WINDOW = "Player Filtering"
_OVERLAY_WINDOW = "Program Overlays"
_SCOREBOARD_WINDOW = "Scoreboard"
_SCOREBOARD = {
    "Show_Scoreboard": ("enabled", 1, True, 1, []),
    "Scoreboard_Position": (
        "position",
        2,
        "upper-left",
        1,
        ["upper-left", "upper-center", "upper-right"],
    ),
    "Scoreboard_Scale_x100": ("scoreboard_scale", 390, 1.0, 100, []),
    "Scoreboard_Sharpen_x100": ("sharpen_amount", 400, 0.0, 100, []),
    "Scoreboard_Resize_Filter": (
        "resize_filter",
        2,
        "bilinear",
        1,
        ["nearest", "bilinear", "bicubic"],
    ),
}
_FILTER_PATHS = {
    ("rink", "tracking", key)
    for key in (
        "cam_ignore_largest",
        "cam_ignore_largest_count",
        "cam_ignore_oversized",
        "cam_oversized_percent",
    )
}
_OVERLAYS = {
    "Player_Boxes": ("plot_individual_player_tracking", "_plot_individual_player_tracking"),
    "Play_Tracking": ("plot_moving_boxes", "_plot_moving_boxes"),
    "Tracking_Circles": ("plot_tracking_circles", "_plot_tracking_circles"),
    "Player_Trajectories": ("plot_trajectories", "_plot_trajectories"),
    "Player_Speed": ("plot_speed", "_plot_speed"),
    "Rink_Boundaries": ("plot_boundaries", "_plot_boundaries"),
}
OVERLAY_OWNER_PARAMS = {
    "plot_individual_player_tracking": ("plot_tracking", False),
    "plot_tracking_circles": ("draw_tracking_circles", True),
    "plot_trajectories": ("plot_trajectories", False),
}


def overlay_owner_values(config: dict) -> dict[str, bool]:
    """Find the late overlay owner's actual constructor values in this graph."""
    owners = [
        spec
        for spec in config.get("aspen", {}).get("plugins", {}).values()
        if str(spec.get("class", "")).endswith(".OverlayPlugin") and spec.get("enabled", True)
    ]
    if len(owners) > 1:
        raise ValueError("Live overlay controls require a single OverlayPlugin owner")
    if not owners:
        return {}
    params = owners[0].get("params") or {}
    values = {}
    for key, (parameter, default) in OVERLAY_OWNER_PARAMS.items():
        value = params.get(parameter, default)
        if isinstance(value, str) and value.startswith("GLOBAL."):
            value = get_nested_value(config, value.removeprefix("GLOBAL."), default)
        values[key] = bool(value)
    return values


class OperatorControls:
    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        self.process = tracker._hm_ui_process
        self.blend = None
        self.overlay_values = overlay_owner_values(tracker._game_config)
        self._last_filters = self._filter_values(tracker._game_config)
        # The command line may override the persisted enable flag.
        self._last_filters["Ignore_Largest_Count"] = (
            tracker._cam_ignore_largest_count if tracker._cam_ignore_largest else 0
        )
        system = self._filter_values(tracker._system_game_config)
        self.process.add_window(_FILTER_WINDOW)
        for name, maximum, label in (
            ("Ignore_Largest_Count", 100, "Ignore largest players (0 = off)"),
            ("Ignore_Oversized_Players", 1, "Ignore oversized players"),
            ("Oversized_Player_Percent_x10", 10000, "Larger than other players' average"),
        ):
            percentage = name == "Oversized_Player_Percent_x10"
            self.process.add_slider(
                _FILTER_WINDOW,
                name,
                max(maximum, self._last_filters[name], system[name]),
                self._last_filters[name],
                label=label,
                divisor=10 if percentage else 1,
                unit="%" if percentage else "",
                description="Applies live to camera framing; player detection is unchanged.",
            )
        self.process.set_system_defaults({_FILTER_WINDOW: system})
        self._last_overlays = {}
        self.process.add_window(_OVERLAY_WINDOW)
        defaults = {}
        for name, (key, attribute) in _OVERLAYS.items():
            value = int(self.overlay_values.get(key, getattr(tracker, attribute)))
            if key in self.overlay_values:
                setattr(tracker, attribute, False)
            self._last_overlays[name] = value
            self.process.add_slider(
                _OVERLAY_WINDOW,
                name,
                1,
                value,
                label=name.replace("_", " "),
                description="Applies live to previews and recorded output.",
            )
            defaults[name] = int(bool(get_nested_value(tracker._system_game_config, f"plot.{key}")))
        self.process.set_system_defaults({_OVERLAY_WINDOW: defaults})
        self._last_scoreboard = self._scoreboard_values(tracker._game_config)
        self.process.add_window(_SCOREBOARD_WINDOW)
        for name, (_, maximum, _, divisor, choices) in _SCOREBOARD.items():
            self.process.add_slider(
                _SCOREBOARD_WINDOW,
                name,
                maximum,
                self._last_scoreboard[name],
                choices=choices or None,
                divisor=divisor,
                offset=0.1 if name == "Scoreboard_Scale_x100" else 0,
                description="Applies live to the Program preview and recording. Select the scoreboard region in Geometry.",
            )
        self.process.set_system_defaults(
            {_SCOREBOARD_WINDOW: self._scoreboard_values(tracker._system_game_config)}
        )
        if tracker._stitch_rotation_controller is not None:
            # Reuse the blend vocabulary, unsupported-entry handling and
            # quantization rules from hmstitch, with a single Save/reset owner.
            from hmlib.aspen.plugins.stitch_ui_plugin import StitchUiPlugin

            self.blend = StitchUiPlugin(blend_only=True)
            self.blend.attach_controls(
                self.process, tracker._game_config, tracker._system_game_config, tracker._game_id
            )

    @staticmethod
    def _scoreboard_values(config):
        values = {}
        for name, (key, _, default, divisor, choices) in _SCOREBOARD.items():
            value = get_nested_value(config, f"rink.scoreboard.{key}", default)
            if choices:
                if value not in choices:
                    raise ValueError(f"Unknown scoreboard {key}: {value}")
                values[name] = choices.index(value)
            else:
                values[name] = round(
                    (float(value) - (0.1 if key == "scoreboard_scale" else 0)) * divisor
                )
        return values

    @staticmethod
    def _filter_values(config: dict) -> dict[str, int]:
        tracking = get_nested_value(config, "rink.tracking", {}) or {}
        return {
            "Ignore_Largest_Count": (
                int(tracking.get("cam_ignore_largest_count", 1))
                if tracking.get("cam_ignore_largest", False)
                else 0
            ),
            "Ignore_Oversized_Players": int(bool(tracking.get("cam_ignore_oversized", False))),
            "Oversized_Player_Percent_x10": round(
                float(tracking.get("cam_oversized_percent", 100)) * 10
            ),
        }

    def managed_paths(self) -> set[tuple[str, ...]]:
        paths = _FILTER_PATHS | {("plot", key) for key, _ in _OVERLAYS.values()}
        paths |= {("rink", "scoreboard", key) for key, *_ in _SCOREBOARD.values()}
        if self.blend is not None:
            paths |= self.blend._managed_paths()
        return paths

    def apply(self) -> None:
        tracker = self.tracker
        values = {
            name: self.process.get_value(_FILTER_WINDOW, name, poll=False)
            for name in self._last_filters
        }
        if values != self._last_filters:
            count = values["Ignore_Largest_Count"]
            oversized = bool(values["Ignore_Oversized_Players"])
            percent = (
                values["Oversized_Player_Percent_x10"] / 10
                if values["Oversized_Player_Percent_x10"]
                != self._last_filters["Oversized_Player_Percent_x10"]
                else tracker._cam_oversized_percent
            )
            if tracker._playtracker is not None:
                tracker._playtracker.set_player_size_filter(count, oversized, percent)
            tracker._cam_ignore_largest = count > 0
            tracker._cam_ignore_largest_count = count
            tracker._cam_ignore_oversized = oversized
            tracker._cam_oversized_percent = percent
            for key, value in (
                ("cam_ignore_largest", count > 0),
                ("cam_ignore_largest_count", count),
                ("cam_ignore_oversized", oversized),
                ("cam_oversized_percent", percent),
            ):
                tracker._set_ui_config_value(("rink", "tracking", key), value)
            self._last_filters = values
        for name, (key, attribute) in _OVERLAYS.items():
            value = self.process.get_value(_OVERLAY_WINDOW, name, poll=False)
            if value != self._last_overlays[name]:
                if key in self.overlay_values:
                    self.overlay_values[key] = bool(value)
                else:
                    setattr(tracker, attribute, bool(value))
                tracker._set_ui_config_value(("plot", key), bool(value))
                self._last_overlays[name] = value
        if self.blend is not None:
            self.blend._apply_blend_controls()
        for name, (key, _, _, divisor, choices) in _SCOREBOARD.items():
            value = self.process.get_value(_SCOREBOARD_WINDOW, name, poll=False)
            if value != self._last_scoreboard[name]:
                result = (
                    choices[value]
                    if choices
                    else (bool(value) if name == "Show_Scoreboard" else value / divisor)
                )
                if key == "scoreboard_scale":
                    result += 0.1
                tracker._set_ui_config_value(("rink", "scoreboard", key), result)
                self._last_scoreboard[name] = value

    def restore(self, source: dict) -> None:
        """Restore exact values after the UI's quantized reset snapshot was applied."""
        tracker = self.tracker
        tracking = get_nested_value(source, "rink.tracking", {}) or {}
        tracker._cam_ignore_largest = bool(tracking.get("cam_ignore_largest", False))
        tracker._cam_ignore_largest_count = int(tracking.get("cam_ignore_largest_count", 1))
        tracker._cam_ignore_oversized = bool(tracking.get("cam_ignore_oversized", False))
        tracker._cam_oversized_percent = float(tracking.get("cam_oversized_percent", 100))
        if tracker._playtracker is not None:
            tracker._playtracker.set_player_size_filter(
                tracker._cam_ignore_largest_count if tracker._cam_ignore_largest else 0,
                tracker._cam_ignore_oversized,
                tracker._cam_oversized_percent,
            )
        self._last_filters = self._filter_values(source)
        self._last_scoreboard = self._scoreboard_values(source)
        for name, (key, attribute) in _OVERLAYS.items():
            value = bool(get_nested_value(source, f"plot.{key}", key == "plot_tracking_circles"))
            if key in self.overlay_values:
                self.overlay_values[key] = value
            else:
                setattr(tracker, attribute, value)
            self._last_overlays[name] = int(value)
        if self.blend is not None:
            self.blend._restore_managed_config(source)
