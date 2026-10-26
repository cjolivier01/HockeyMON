"""Rust runtime controls for the stitching-only Aspen workflow."""

from __future__ import annotations

import copy
import time
from typing import Any

from hmlib.builder import HM
from hmlib.camera.hm_ui_bridge import HmUiProcess
from hmlib.config import (
    get_config,
    get_game_config_private,
    get_nested_value,
    normalize_runtime_config,
    save_private_config,
)
from hmlib.log import logger
from hmlib.stitching.blend import (
    BLEND_MODES,
    DEFAULT_BLEND_MODE,
    DEFAULT_FEATHER_FRACTION,
    GPU_BLEND_MODES,
    MAX_FEATHER_FRACTION,
    PYTHON_BLEND_MODES,
    config_blend_mode,
    normalize_feather_fraction,
    normalize_max_blend_levels,
    resolve_blend_settings,
)
from hmlib.utils.shadow_lift import shadow_lift_settings

from .base import Plugin

_MISSING = object()
_EXPOSURE_EV_X10_MIN = -40
_EXPOSURE_EV_X10_MAX = 40
_EXPOSURE_EV_X10_SLIDER_MAX = _EXPOSURE_EV_X10_MAX - _EXPOSURE_EV_X10_MIN
_EXPOSURE_EV_X10_SLIDER_ZERO = -_EXPOSURE_EV_X10_MIN
_UI_ACTION_RETRY_INITIAL_SECONDS = 0.5
_UI_ACTION_RETRY_MAX_SECONDS = 5.0

_BLEND_WINDOW = "Stitch Blend"
_BLEND_MODE_CONTROL = "Seam_Blend_Mode"
_BLEND_FEATHER_CONTROL = "Seam_Feather_Percent"
# Combo labels for every mode this repository knows. A mode the configured
# blender cannot render still gets an entry (see `_resolve_blend_choices`), but
# it is marked so the operator can tell it apart from a real choice.
_BLEND_MODE_LABELS = {
    "laplacian": "Laplacian (multi-band)",
    "alpha": "Alpha (feathered seam)",
    "gpu-hard-seam": "Hard seam (no blending)",
    "multiblend": "Multiblend (CPU)",
}
# The feather is a fraction; the UI carries integer controls, so it travels as
# percent and converts at the boundary.
_FEATHER_PERCENT_MAX = round(MAX_FEATHER_FRACTION * 100.0)
_NEXT_RUN_NOTE = "Applies on the next stitch run."


@HM.register_module()
class StitchUiPlugin(Plugin):
    """Expose only controls that affect the stitched image."""

    disable_in_cuda_graph_pipeline = True

    def __init__(self, enabled: bool = True, blend_only: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._blend_only = blend_only
        self._process: HmUiProcess | None = None
        self._game_config: dict[str, Any] = {}
        self._open_config: dict[str, Any] = {}
        self._system_config: dict[str, Any] = {}
        self._game_id: str | None = None
        self._dirty_paths: set[tuple[str, ...]] = set()
        self._save_count = 0
        self._blend_choices: tuple[str, ...] = ()
        self._blend_runnable: tuple[str, ...] = ()
        self._shared: dict[str, Any] | None = None
        self._active = False
        self._action_retry_after_monotonic = 0.0
        self._action_retry_delay_seconds = 0.0

    @staticmethod
    def _stitch_deg_to_slider(degrees: Any) -> int:
        value = max(-90.0, min(90.0, float(degrees or 0.0)))
        return int(max(0, min(180, round(90.0 - value))))

    @staticmethod
    def _slider_to_stitch_deg(position: int) -> float:
        return float(90 - int(position))

    @staticmethod
    def _exposure_ev_to_slider(value: Any) -> int:
        try:
            ev_x10 = round(float(value) * 10.0)
        except (TypeError, ValueError):
            ev_x10 = 0
        ev_x10 = max(_EXPOSURE_EV_X10_MIN, min(_EXPOSURE_EV_X10_MAX, ev_x10))
        return ev_x10 - _EXPOSURE_EV_X10_MIN

    @staticmethod
    def _slider_to_exposure_ev(position: int) -> float:
        position = max(0, min(_EXPOSURE_EV_X10_SLIDER_MAX, int(position)))
        return float(position + _EXPOSURE_EV_X10_MIN) / 10.0

    @staticmethod
    def _config_blend_entry(config: dict[str, Any]) -> str | None:
        """The combo entry this config names, or None when it names nothing.

        A mode that will not normalize is kept, folded, so it can be its own
        entry the same way a mode this path cannot render is - two spellings of
        the same unknown mode must not become two entries. Silent by design:
        this runs on every apply, and the one warning is logged at open time.
        """
        return config_blend_mode(get_nested_value(config, "stitching.blend_mode", None))

    @staticmethod
    def _config_feather_percent(config: dict[str, Any]) -> int | None:
        """The slider position this config asks for, or None when it asks for one
        the slider cannot hold.

        None means "the slider is authoritative", so the next apply repairs the
        config rather than leaving a value that would fail the next run.
        """
        raw = get_nested_value(config, "stitching.blend_feather_fraction", None)
        if raw is None:
            return round(DEFAULT_FEATHER_FRACTION * 100.0)
        try:
            return round(normalize_feather_fraction(raw) * 100.0)
        except ValueError:
            return None

    def _resolve_blend_choices(self) -> tuple[str, ...]:
        """Renderable modes, plus the game's own mode when this path cannot run it.

        Showing an unrunnable mode as its own entry is what makes picking a real
        one an index change, so the choice reaches the config instead of looking
        already-selected and firing nothing. Resolved once at open time: the
        combo the operator sees cannot change under them, and applying a mode
        rewrites the config this reads from.

        Which modes are runnable depends on the blender this game is configured
        for, so the combo never offers one the next run would refuse to build.
        """
        # The plugin param is what actually selects the blender; the shared key is
        # only its default, so an override of one must not leave the combo
        # offering modes the other cannot run.
        python_blender = get_nested_value(
            self._game_config, "aspen.plugins.stitching.params.python_blender", None
        )
        if python_blender is None:
            python_blender = get_nested_value(self._game_config, "stitching.python_blender", False)
        self._blend_runnable = PYTHON_BLEND_MODES if bool(python_blender) else GPU_BLEND_MODES
        modes = list(self._blend_runnable)
        for config in (self._game_config, self._system_config):
            mode = self._config_blend_entry(config)
            if mode is not None and mode not in modes:
                modes.append(mode)
        return tuple(modes)

    def _blend_mode_labels(self) -> list[str]:
        return [
            (
                _BLEND_MODE_LABELS.get(mode, mode)
                if mode in self._blend_runnable
                else f"{_BLEND_MODE_LABELS.get(mode, mode)} - not supported here"
            )
            for mode in self._blend_choices
        ]

    def _blend_mode_index(self, config: dict[str, Any]) -> int:
        mode = self._config_blend_entry(config) or DEFAULT_BLEND_MODE
        choices = self._blend_choices
        return choices.index(mode) if mode in choices else choices.index(DEFAULT_BLEND_MODE)

    def _blend_feather_position(self, config: dict[str, Any]) -> int:
        percent = self._config_feather_percent(config)
        return round(DEFAULT_FEATHER_FRACTION * 100.0) if percent is None else percent

    @classmethod
    def _color_defaults(cls, config: dict[str, Any], path: str) -> dict[str, int]:
        color_cfg = get_nested_value(config, path, {}) or {}
        if not isinstance(color_cfg, dict):
            color_cfg = {}
        shadow, black_point = shadow_lift_settings(
            color_cfg.get("shadow_lift"), color_cfg.get("shadow_lift_black_point")
        )
        defaults = {
            "White_Balance_Kelvin_Enable": 0,
            "White_Balance_Kelvin_Temperature": 6500,
            "White_Balance_Red_Gain_x100": 100,
            "White_Balance_Green_Gain_x100": 100,
            "White_Balance_Blue_Gain_x100": 100,
            "Brightness_Multiplier_x100": 100,
            "Exposure_EV_x10": _EXPOSURE_EV_X10_SLIDER_ZERO,
            "Shadow_Lift_Percent": round(shadow),
            "Shadow_Lift_Black_Point": int(black_point),
            "Contrast_Multiplier_x100": 100,
            "Gamma_Multiplier_x100": 100,
        }
        temperature = color_cfg.get("white_balance_temp")
        gains = color_cfg.get("white_balance")
        if temperature is not None:
            try:
                defaults["White_Balance_Kelvin_Enable"] = 1
                defaults["White_Balance_Kelvin_Temperature"] = int(
                    max(1000, min(15000, float(str(temperature).lower().removesuffix("k"))))
                )
            except (TypeError, ValueError):
                logger.warning("Invalid white balance temperature in %s: %r", path, temperature)
        elif isinstance(gains, (list, tuple)) and len(gains) == 3:
            try:
                blue, green, red = gains
                defaults["White_Balance_Red_Gain_x100"] = int(float(red) * 100.0)
                defaults["White_Balance_Green_Gain_x100"] = int(float(green) * 100.0)
                defaults["White_Balance_Blue_Gain_x100"] = int(float(blue) * 100.0)
            except (TypeError, ValueError):
                logger.warning("Invalid white balance gains in %s: %r", path, gains)
        defaults["Exposure_EV_x10"] = cls._exposure_ev_to_slider(color_cfg.get("exposure_ev", 0.0))
        for key, control in (
            ("brightness", "Brightness_Multiplier_x100"),
            ("contrast", "Contrast_Multiplier_x100"),
            ("gamma", "Gamma_Multiplier_x100"),
        ):
            try:
                defaults[control] = int(float(color_cfg.get(key, 1.0)) * 100.0)
            except (TypeError, ValueError):
                logger.warning("Invalid %s in %s: %r", key, path, color_cfg.get(key))
        return defaults

    def _add_color_window(self, window_name: str, defaults: dict[str, int]) -> None:
        assert self._process is not None
        self._process.add_window(window_name)
        maximums = {
            "White_Balance_Kelvin_Enable": 1,
            "White_Balance_Kelvin_Temperature": 15000,
            "White_Balance_Red_Gain_x100": 300,
            "White_Balance_Green_Gain_x100": 300,
            "White_Balance_Blue_Gain_x100": 300,
            "Brightness_Multiplier_x100": 300,
            "Exposure_EV_x10": _EXPOSURE_EV_X10_SLIDER_MAX,
            "Shadow_Lift_Percent": 100,
            "Shadow_Lift_Black_Point": 1,
            "Contrast_Multiplier_x100": 300,
            "Gamma_Multiplier_x100": 300,
        }
        for name, value in defaults.items():
            self._process.add_slider(window_name, name, maximums[name], value)

    def _ensure_initialized(self, context: dict[str, Any]) -> None:
        if self._process is not None or self._active:
            return
        shared = context.get("shared", {})
        if not isinstance(shared, dict):
            return
        self._active = bool(shared.get("camera_ui"))
        if not self._active:
            return

        self._shared = shared
        self._game_config = shared.get("game_config") or {}
        self._open_config = copy.deepcopy(self._game_config)
        self._game_id = shared.get("game_id")
        self._system_config = copy.deepcopy(shared.get("hm_ui_system_config", self._game_config))
        if self._game_id and "hm_ui_system_config" not in shared:
            try:
                self._system_config = get_config(
                    game_id=self._game_id,
                    ignore_private_config=True,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as ex:
                logger.warning(
                    "Could not load system stitch UI defaults for %s; using open-time values: %s",
                    self._game_id,
                    ex,
                )

        if self._blend_only:
            title = f"HM Stitch Blend Repair - {self._game_id}"
        else:
            title = f"HM Stitch UI - {self._game_id}" if self._game_id else "HM Stitch UI"
        self._process = shared.get("hm_ui_existing_process") or HmUiProcess(
            title=title,
            preview_names=(
                ("Stitched",) if self._blend_only else ("Stitched", "Camera 1", "Camera 2")
            ),
        )
        if not self._blend_only:
            self._process.enable_playback()
        alignment_window = "Stitch Alignment"
        if not self._blend_only:
            self._process.add_window(alignment_window)
            rotation = get_nested_value(
                self._game_config,
                "stitching.post_stitch_rotate_degrees",
                0.0,
            )
            self._process.add_slider(
                alignment_window,
                "Stitch_Rotate_Degrees",
                180,
                self._stitch_deg_to_slider(rotation),
            )

        self._blend_choices = self._resolve_blend_choices()
        # The accessors stay silent because they run on every apply; say it once.
        blend_entry = self._config_blend_entry(self._game_config)
        if blend_entry is not None and blend_entry not in BLEND_MODES:
            logger.warning(
                "stitching.blend_mode %r is not a mode this build knows; the stitch UI keeps it "
                "selectable but the next run will reject it",
                blend_entry,
            )
        elif blend_entry is not None and blend_entry not in self._blend_runnable:
            logger.warning(
                "stitching.blend_mode %r cannot be rendered by the configured blender; the stitch "
                "UI keeps it selectable but the next run will reject it",
                blend_entry,
            )
        if self._config_feather_percent(self._game_config) is None:
            logger.warning(
                "stitching.blend_feather_fraction %r is out of range; %s",
                get_nested_value(self._game_config, "stitching.blend_feather_fraction", None),
                (
                    "the stitch UI opens at the default and saving repairs it"
                    if "alpha" in self._blend_runnable
                    # No feather control is published where alpha cannot be
                    # selected, so nothing here can write a usable value.
                    else "no control here can repair it; fix the game config"
                ),
            )
        self._process.add_window(_BLEND_WINDOW)
        self._process.add_slider(
            _BLEND_WINDOW,
            _BLEND_MODE_CONTROL,
            len(self._blend_choices) - 1,
            self._blend_mode_index(self._game_config),
            choices=self._blend_mode_labels(),
            description=(
                "Laplacian mixes the cameras across every scale in the overlap; alpha crossfades "
                f"only at the seam. {_NEXT_RUN_NOTE}"
            ),
        )
        if "alpha" in self._blend_runnable:
            self._process.add_slider(
                _BLEND_WINDOW,
                _BLEND_FEATHER_CONTROL,
                _FEATHER_PERCENT_MAX,
                self._blend_feather_position(self._game_config),
                description=(
                    "Alpha crossfade width, as a percent of the narrowest camera image. Ignored "
                    f"by the other modes. {_NEXT_RUN_NOTE}"
                ),
            )

        color_windows = {
            "Tracker Controls (Stitched Color)": "rink.camera.color",
            "Tracker Controls (Left Color)": "stitching.left.color",
            "Tracker Controls (Right Color)": "stitching.right.color",
        }
        system_defaults: dict[str, dict[str, int]] = {
            _BLEND_WINDOW: {
                _BLEND_MODE_CONTROL: self._blend_mode_index(self._system_config),
            }
        }
        if not self._blend_only:
            system_defaults[alignment_window] = {
                "Stitch_Rotate_Degrees": self._stitch_deg_to_slider(
                    get_nested_value(
                        self._system_config,
                        "stitching.post_stitch_rotate_degrees",
                        0.0,
                    )
                )
            }
        if "alpha" in self._blend_runnable:
            system_defaults[_BLEND_WINDOW][_BLEND_FEATHER_CONTROL] = self._blend_feather_position(
                self._system_config
            )
        if not self._blend_only:
            for window_name, path in color_windows.items():
                defaults = self._color_defaults(self._game_config, path)
                system_defaults[window_name] = self._color_defaults(self._system_config, path)
                self._add_color_window(window_name, defaults)
        self._process.set_system_defaults(system_defaults)
        shared["hm_ui_process"] = self._process
        shared["hm_ui_preview_active"] = True

    def attach_controls(
        self,
        process: HmUiProcess,
        game_config: dict[str, Any],
        system_config: dict[str, Any],
        game_id: str | None,
    ) -> None:
        """Mount blend controls in the tracker's sidecar; the tracker owns Save/reset."""
        if not self._blend_only:
            raise ValueError("Only blend controls can be attached to a tracking UI")
        self._ensure_initialized(
            {
                "shared": {
                    "camera_ui": 1,
                    "game_config": game_config,
                    "game_id": game_id,
                    "hm_ui_existing_process": process,
                    "hm_ui_system_config": system_config,
                }
            }
        )

    @staticmethod
    def _values_equal(left: Any, right: Any, *, path: tuple[str, ...] = ()) -> bool:
        if left is _MISSING or right is _MISSING:
            return left is right
        if path == ("stitching", "blend_mode"):
            # The combo writes the canonical spelling; a system config holding an
            # alias of the same mode is not a value to override, and saving one
            # would shadow every later change to the system value.
            return config_blend_mode(left) == config_blend_mode(right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) < 1e-6
        return left == right

    @staticmethod
    def _path_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return _MISSING
            current = current[key]
        return current

    @classmethod
    def _set_path(cls, config: dict[str, Any], path: tuple[str, ...], value: Any) -> bool:
        current: Any = config
        parents = []
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                if value is _MISSING:
                    return False
                current[key] = {}
            parents.append((current, key))
            current = current[key]
        leaf = path[-1]
        previous = current.get(leaf, _MISSING)
        if value is _MISSING:
            if leaf not in current:
                return False
            del current[leaf]
            while parents:
                parent, key = parents.pop()
                if isinstance(parent.get(key), dict) and not parent[key]:
                    del parent[key]
                else:
                    break
            return True
        if cls._values_equal(previous, value, path=path):
            return False
        current[leaf] = copy.deepcopy(value)
        return True

    def _set_runtime_path(self, path: tuple[str, ...], value: Any) -> None:
        if self._set_path(self._game_config, path, value):
            self._dirty_paths.add(path)

    def _apply_color_window(self, window_name: str, prefix: tuple[str, ...]) -> None:
        assert self._process is not None

        def value(name: str) -> int:
            assert self._process is not None
            return self._process.get_value(window_name, name, poll=False)

        if value("White_Balance_Kelvin_Enable"):
            kelvin = max(1000, min(15000, value("White_Balance_Kelvin_Temperature")))
            self._set_runtime_path(prefix + ("white_balance_temp",), f"{kelvin}k")
            self._set_runtime_path(prefix + ("white_balance",), _MISSING)
        else:
            gains = [
                max(1, value("White_Balance_Blue_Gain_x100")) / 100.0,
                max(1, value("White_Balance_Green_Gain_x100")) / 100.0,
                max(1, value("White_Balance_Red_Gain_x100")) / 100.0,
            ]
            self._set_runtime_path(prefix + ("white_balance",), gains)
            self._set_runtime_path(prefix + ("white_balance_temp",), _MISSING)
        self._set_runtime_path(
            prefix + ("brightness",),
            max(1, value("Brightness_Multiplier_x100")) / 100.0,
        )
        self._set_runtime_path(
            prefix + ("exposure_ev",),
            self._slider_to_exposure_ev(value("Exposure_EV_x10")),
        )
        for key, setting in (
            ("shadow_lift", float(value("Shadow_Lift_Percent"))),
            ("shadow_lift_black_point", bool(value("Shadow_Lift_Black_Point"))),
        ):
            if setting or self._path_value(self._game_config, prefix + (key,)) is not _MISSING:
                self._set_runtime_path(prefix + (key,), setting)
        self._set_runtime_path(
            prefix + ("contrast",),
            max(1, value("Contrast_Multiplier_x100")) / 100.0,
        )
        self._set_runtime_path(
            prefix + ("gamma",),
            max(1, value("Gamma_Multiplier_x100")) / 100.0,
        )

    def _apply_blend_controls(self) -> None:
        """Write back only the blend controls the operator actually moved.

        Both controls are lossy views of the config: the combo canonicalizes the
        spelling and the slider quantizes the fraction to whole percent. Writing
        unconditionally would rewrite `gpu_hard_seam` as `gpu-hard-seam` and
        0.125 as 0.12 on an apply triggered by some unrelated control, and each
        rewrite then reads as an override the save can never clear. Comparing
        against what the config already means keeps an untouched control silent.
        """
        assert self._process is not None
        choices = self._blend_choices
        index = self._process.get_value(_BLEND_WINDOW, _BLEND_MODE_CONTROL, poll=False)
        # The bridge clamps to the control's range; clamp again rather than
        # substitute a different mode if that ever stops holding.
        mode = choices[min(max(index, 0), len(choices) - 1)]
        # A marked entry stands for the value the config already holds, not for a
        # choice: selecting it must leave that value alone rather than write a
        # mode the next graph build would refuse. Only a runnable mode is a real
        # selection, and an absent or null mode means "inherit the default",
        # which is what the combo already shows.
        if mode in self._blend_runnable and mode != (
            self._config_blend_entry(self._game_config) or DEFAULT_BLEND_MODE
        ):
            self._set_runtime_path(("stitching", "blend_mode"), mode)
        if "alpha" not in self._blend_runnable:
            # No feather control exists where alpha cannot be selected, so there
            # is nothing to write and nothing that could act on it.
            return
        percent = max(
            0, int(self._process.get_value(_BLEND_WINDOW, _BLEND_FEATHER_CONTROL, poll=False))
        )
        # A config width the slider cannot hold reads as None, so the slider wins
        # and the unusable value is repaired rather than left for the next run.
        if percent != self._config_feather_percent(self._game_config):
            self._set_runtime_path(
                ("stitching", "blend_feather_fraction"),
                min(MAX_FEATHER_FRACTION, percent / 100.0),
            )

    def _apply_controls(self) -> None:
        assert self._process is not None
        if self._blend_only:
            self._apply_blend_controls()
            return
        rotation = self._slider_to_stitch_deg(
            self._process.get_value("Stitch Alignment", "Stitch_Rotate_Degrees", poll=False)
        )
        self._set_runtime_path(("stitching", "post_stitch_rotate_degrees"), rotation)
        self._apply_blend_controls()
        self._apply_color_window(
            "Tracker Controls (Stitched Color)",
            ("rink", "camera", "color"),
        )
        self._apply_color_window(
            "Tracker Controls (Left Color)",
            ("stitching", "left", "color"),
        )
        self._apply_color_window(
            "Tracker Controls (Right Color)",
            ("stitching", "right", "color"),
        )

    def _save(self) -> None:
        if not self._game_id:
            return
        private_config = get_game_config_private(game_id=self._game_id) or {}
        normalize_runtime_config(private_config)
        rotation_path = ("stitching", "post_stitch_rotate_degrees")
        if rotation_path in self._dirty_paths:
            for path in (
                ("rink", "ice_contours_mask_count"),
                ("rink", "ice_contours_mask_centroid"),
                ("rink", "ice_contours_combined_bbox"),
                ("rink", "ice_contours_geometry_revision"),
                ("rink", "scoreboard", "perspective_polygon"),
            ):
                self._set_path(private_config, path, _MISSING)

        changed = False
        for path in self._managed_paths() | self._dirty_paths:
            current = self._path_value(self._game_config, path)
            system = self._path_value(self._system_config, path)
            value = (
                _MISSING
                if current is _MISSING or self._values_equal(current, system, path=path)
                else current
            )
            changed |= self._set_path(private_config, path, value)
        if changed:
            save_private_config(self._game_id, private_config, verbose=True)
        self._dirty_paths.clear()
        self._save_count += 1

    @property
    def save_count(self) -> int:
        return self._save_count

    def _managed_paths(self) -> set[tuple[str, ...]]:
        rotation_path = ("stitching", "post_stitch_rotate_degrees")
        color_keys = {
            "white_balance",
            "white_balance_temp",
            "brightness",
            "exposure_ev",
            "shadow_lift",
            "shadow_lift_black_point",
            "contrast",
            "gamma",
        }
        managed_paths: set[tuple[str, ...]] = {("stitching", "blend_mode")}
        if "alpha" in self._blend_runnable:
            managed_paths.add(("stitching", "blend_feather_fraction"))
        if self._blend_only:
            return managed_paths
        managed_paths.add(rotation_path)
        for prefix in (
            ("rink", "camera", "color"),
            ("stitching", "left", "color"),
            ("stitching", "right", "color"),
        ):
            managed_paths.update(prefix + (key,) for key in color_keys)
        return managed_paths

    def _restore_managed_config(self, source: dict[str, Any]) -> None:
        for path in self._managed_paths():
            self._set_runtime_path(path, self._path_value(source, path))
        # The controls already hold the reset values, and restoring the source
        # verbatim would undo the repair of a mode or width the combo and slider
        # cannot represent - leaving what the operator sees and what the next run
        # gets disagreeing, decided by which button was pressed last.
        if self._process is not None:
            self._apply_blend_controls()

    def _restore_final_reset_config(self, events: list[Any], final_values: dict[str, Any]) -> None:
        for event in reversed(events):
            event_values = final_values if event.values is None else event.values
            if event_values != final_values:
                continue
            if event.kind == "reset-system":
                self._restore_managed_config(self._system_config)
                return
            if event.kind == "reset-open":
                self._restore_managed_config(self._open_config)
                return

    def _disable_ui(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None
        if self._shared is not None:
            self._shared["hm_ui_process"] = None
            self._shared["hm_ui_preview_active"] = False

    def input_keys(self) -> set[str]:
        return {
            "img",
            "shared",
            "frame_ids",
            "fps",
            "source_frame_ids",
            "decoded_pts_ns",
            "decoded_timing",
            "decoded_source_role",
        }

    def output_keys(self) -> set[str]:
        return {"img"}

    def _poll_controls(self) -> None:
        if self._process is None:
            return
        self._process.poll()
        controls_changed = self._process.last_poll_values_changed
        if self._process.closed:
            self._disable_ui()
            return
        final_values = self._process.control_values()
        events = self._process.consume_action_events(poll=False)
        runtime_values = None
        retry_now = time.monotonic()
        retry_deferred = bool(events) and retry_now < self._action_retry_after_monotonic
        if not events:
            self._action_retry_after_monotonic = 0.0
            self._action_retry_delay_seconds = 0.0
        try:
            if retry_deferred:
                if controls_changed:
                    self._apply_controls()
            else:
                for event in events:
                    if self._process.apply_playback_action(event.kind):
                        continue
                    self._process.apply_control_values(
                        final_values if event.values is None else event.values
                    )
                    event_values = self._process.control_values()
                    if event.kind == "reset-system":
                        self._apply_controls()
                        self._restore_managed_config(self._system_config)
                        runtime_values = event_values
                    elif event.kind == "reset-open":
                        self._apply_controls()
                        self._restore_managed_config(self._open_config)
                        runtime_values = event_values
                    elif event.kind == "save":
                        if event_values != runtime_values:
                            self._apply_controls()
                            runtime_values = event_values
                        self._save()
                self._process.apply_control_values(final_values, publish=bool(events))
                if (controls_changed or events) and final_values != runtime_values:
                    self._apply_controls()
                if events:
                    self._process.acknowledge_action_events(
                        max(event.seq for event in events),
                        message=(
                            "Saved game settings"
                            if any(e.kind == "save" for e in events)
                            else "Applied"
                        ),
                    )
                    self._action_retry_after_monotonic = 0.0
                    self._action_retry_delay_seconds = 0.0
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            self._process.report_action_error(str(error))
            delay = (
                _UI_ACTION_RETRY_INITIAL_SECONDS
                if self._action_retry_delay_seconds <= 0.0
                else min(
                    _UI_ACTION_RETRY_MAX_SECONDS,
                    self._action_retry_delay_seconds * 2.0,
                )
            )
            self._action_retry_delay_seconds = delay
            self._action_retry_after_monotonic = retry_now + delay
            logger.exception(
                "Stitch camera UI action processing failed; retrying pending actions in %.1fs",
                delay,
            )
            try:
                self._process.apply_control_values(final_values, publish=bool(events))
                if final_values != runtime_values:
                    self._apply_controls()
                    self._restore_final_reset_config(events, final_values)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as restore_ex:
                logger.error("Failed to restore final stitch camera UI values: %s", restore_ex)

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {"img": context.get("img")}
        img = context.get("img")
        try:
            self._ensure_initialized(context)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as ex:
            self._disable_ui()
            raise RuntimeError("Failed to initialize stitch camera UI") from ex
        process = self._process
        if process is None:
            return {"img": img}
        process.wait_for_playback(self._poll_controls)
        if self._process is None:
            return {"img": img}
        frame_ids = context.get("frame_ids")
        if frame_ids is not None:
            from hmlib.utils.gpu import unwrap_tensor

            frame_ids = unwrap_tensor(frame_ids)
        frame = int(frame_ids[-1]) if frame_ids is not None and len(frame_ids) else None
        process.publish_runtime(frame=frame, fps=context.get("fps"))
        if img is not None:
            from hmlib.ui.preview_overlays import source_metadata

            self._process.publish_preview(img, name="Stitched", metadata=source_metadata(context))
        return {"img": img}

    def finalize(self) -> None:
        self._disable_ui()


def repair_blend_before_stitch(
    game_config: dict[str, Any],
    game_id: str | None,
    *,
    raw_graph_params: dict[str, Any] | None = None,
    ignore_private_config: bool = False,
    run_blend_overrides: tuple[str, ...] = (),
) -> bool:
    """Let the operator save a runnable blend before a broken stitch graph starts.

    Returns true when the UI saved a repair. The caller must end this run: the
    graph's stitcher parameters were resolved from the old config already.
    """
    stitching = get_nested_value(game_config, "stitching", {}) or {}
    if not isinstance(stitching, dict):
        raise ValueError("stitching config must be a mapping")  # noqa: TRY004
    graph_params = get_nested_value(game_config, "aspen.plugins.stitching.params", None)
    if graph_params is not None and not isinstance(graph_params, dict):
        raise ValueError("aspen.plugins.stitching.params must be a mapping")
    mode_is_shared = graph_params is None or (
        raw_graph_params is not None
        and raw_graph_params.get("blend_mode") == "GLOBAL.stitching.blend_mode"
    )
    feather_is_shared = graph_params is None or (
        raw_graph_params is not None
        and raw_graph_params.get("blend_feather_fraction")
        == "GLOBAL.stitching.blend_feather_fraction"
    )
    levels = (
        graph_params.get("max_blend_levels")
        if graph_params is not None
        else stitching.get("max_blend_levels")
    )
    if levels is not None:
        # The blend UI has no pyramid-depth control, so do not wait for an edit
        # that cannot repair this value.
        normalize_max_blend_levels(levels)
    python_blender = bool(
        graph_params.get("python_blender", False)
        if graph_params is not None
        else stitching.get("python_blender", False)
    )

    def mode_value() -> Any:
        if mode_is_shared:
            return stitching.get("blend_mode")
        return graph_params.get("blend_mode", DEFAULT_BLEND_MODE)

    def feather_value() -> Any:
        if feather_is_shared:
            return stitching.get("blend_feather_fraction")
        return graph_params.get("blend_feather_fraction")

    def validate() -> None:
        blend = resolve_blend_settings(
            blend_mode=mode_value(),
            blend_feather_fraction=feather_value(),
            max_blend_levels=levels,
        )
        blend.require_python_mode() if python_blender else blend.require_gpu_mode()

    try:
        validate()
    except ValueError as error:
        if graph_params is not None and not mode_is_shared:
            try:
                explicit_mode = resolve_blend_settings(
                    blend_mode=graph_params.get("blend_mode", DEFAULT_BLEND_MODE)
                )
                (
                    explicit_mode.require_python_mode()
                    if python_blender
                    else explicit_mode.require_gpu_mode()
                )
            except ValueError as explicit_error:
                raise ValueError(
                    f"{explicit_error}; edit aspen.plugins.stitching.params.blend_mode directly"
                ) from explicit_error
        if graph_params is not None and not feather_is_shared:
            raw_feather = graph_params.get("blend_feather_fraction")
            if raw_feather is not None:
                try:
                    normalize_feather_fraction(raw_feather)
                except ValueError as explicit_error:
                    raise ValueError(
                        f"{explicit_error}; edit "
                        "aspen.plugins.stitching.params.blend_feather_fraction directly"
                    ) from explicit_error
        if python_blender and feather_value() is not None:
            # The Python path publishes no feather control.
            normalize_feather_fraction(feather_value())
        if run_blend_overrides:
            raise ValueError(
                f"{error}; this command overrides {', '.join(run_blend_overrides)} on every "
                "run, so change or remove that override before saving a stitch UI repair"
            ) from error
        if ignore_private_config:
            raise ValueError(
                f"{error}; --ignore-private-config would discard a stitch UI repair"
            ) from error
        if not game_id:
            raise ValueError(f"{error}; the stitch UI needs --game-id to save a repair") from error
        logger.warning("%s. Open the stitch UI, choose a supported blend, and Save.", error)
    else:
        return False

    plugin = StitchUiPlugin(blend_only=True)
    shared = {"camera_ui": 1, "game_id": game_id, "game_config": game_config}
    context = {"img": None, "shared": shared}
    checked_saves = 0
    try:
        while True:
            plugin.forward(context)
            if shared.get("hm_ui_process") is None:
                raise ValueError("Stitch UI closed without saving a runnable blend mode")
            if plugin.save_count > checked_saves:
                checked_saves = plugin.save_count
                try:
                    validate()
                except ValueError as error:
                    logger.warning("Blend configuration still cannot stitch: %s", error)
                else:
                    logger.info("Saved a runnable blend. Run hmstitch again to use it.")
                    return True
            time.sleep(0.05)
    finally:
        plugin.finalize()


__all__ = ["StitchUiPlugin", "repair_blend_before_stitch"]
