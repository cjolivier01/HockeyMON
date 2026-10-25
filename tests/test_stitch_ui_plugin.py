from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import ClassVar

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="requires torch")

if torch is not None:
    from hmlib.aspen.plugins import stitch_ui_plugin as stitch_ui_module
    from hmlib.aspen.plugins import video_preview_plugin as video_preview_module
    from hmlib.aspen.plugins.stitch_ui_plugin import StitchUiPlugin
    from hmlib.aspen.plugins.video_preview_plugin import VideoPreviewPlugin
else:
    stitch_ui_module = None  # type: ignore[assignment]
    video_preview_module = None  # type: ignore[assignment]
    StitchUiPlugin = None  # type: ignore[assignment,misc]
    VideoPreviewPlugin = None  # type: ignore[assignment,misc]


def _color() -> dict:
    return {
        "white_balance": [1.0, 1.0, 1.0],
        "brightness": 1.0,
        "exposure_ev": 0.0,
        "contrast": 1.0,
        "gamma": 1.0,
    }


def _config(rotation: float) -> dict:
    return {
        "stitching": {
            "post_stitch_rotate_degrees": rotation,
            "left": {"color": _color()},
            "right": {"color": _color()},
        },
        "rink": {"camera": {"color": _color()}},
    }


class _FakeHmUiProcess:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.values = {}
        self.open_defaults = {}
        self.choices = {}
        self.descriptions = {}
        self.system_defaults = {}
        self.changed = False
        self.last_poll_values_changed = False
        self.actions = []
        self.next_action_seq = 1
        self.acknowledged_seq = 0
        self.previews = []
        self.closed = False
        self.instances.append(self)

    def enable_playback(self):
        self.playback_enabled = True

    def apply_playback_action(self, kind):
        return False

    def report_action_error(self, error):
        self.error = error

    def wait_for_playback(self, callback):
        callback()

    def publish_runtime(self, **kwargs):
        self.runtime = kwargs

    def add_window(self, name: str) -> None:
        self.values.setdefault(name, {})
        self.open_defaults.setdefault(name, {})

    def add_slider(
        self,
        window: str,
        name: str,
        _maximum: int,
        value: int,
        *,
        choices=None,
        description: str = "",
    ) -> None:
        self.values[window][name] = value
        self.open_defaults[window][name] = value
        self.choices.setdefault(window, {})[name] = list(choices) if choices else []
        self.descriptions.setdefault(window, {})[name] = description

    def set_system_defaults(self, defaults) -> None:
        self.system_defaults = copy.deepcopy(defaults)

    def get_value(self, window: str, name: str, *, poll: bool = True) -> int:
        del poll
        return self.values[window][name]

    def poll(self) -> bool:
        changed, self.changed = self.changed, False
        self.last_poll_values_changed = changed
        return changed or bool(self.actions)

    def consume_action_events(self, *, poll: bool = True):
        del poll
        actions, self.actions = self.actions, []
        return actions

    def control_values(self):
        return copy.deepcopy(self.values)

    def apply_control_values(self, values, *, publish: bool = False) -> bool:
        del publish
        changed = values != self.values
        for window, controls in values.items():
            self.values.setdefault(window, {}).update(copy.deepcopy(controls))
        return changed

    def queue_action(self, kind: str) -> None:
        self.actions.append(
            _FakeAction(
                seq=self.next_action_seq,
                kind=kind,
                values=self.control_values(),
            )
        )
        self.next_action_seq += 1

    def acknowledge_action_events(self, through_seq: int, *, message: str = "") -> None:
        self.acknowledged_seq = through_seq

    def queue_reset(self, *, system: bool) -> None:
        self.values = copy.deepcopy(self.system_defaults if system else self.open_defaults)
        self.changed = True
        self.queue_action("reset-system" if system else "reset-open")

    def publish_preview(self, img, *, name: str) -> None:
        self.previews.append((img, name))

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _FakeAction:
    seq: int
    kind: str
    values: dict


class _FakeShower:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.calls = []
        self.instances.append(self)

    def show(self, img, *, clone: bool) -> None:
        self.calls.append((img, clone))

    def close(self) -> None:
        self.closed = True

    def update_progress_table(self, _table) -> None:
        return None


def should_apply_and_save_stitch_only_rust_controls(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    current_config = _config(rotation=5.0)
    system_config = _config(rotation=0.5)
    private_config = {"rink": {"camera": {"color": {"contrast": 1.0}}}}
    saved = {}

    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", _FakeHmUiProcess)
    monkeypatch.setattr(
        stitch_ui_module,
        "get_config",
        lambda **_kwargs: copy.deepcopy(system_config),
    )
    monkeypatch.setattr(
        stitch_ui_module,
        "get_game_config_private",
        lambda **_kwargs: copy.deepcopy(private_config),
    )

    def save_private(_game_id, data, verbose=True):
        del verbose
        saved.clear()
        saved.update(copy.deepcopy(data))

    monkeypatch.setattr(stitch_ui_module, "save_private_config", save_private)

    shared = {
        "camera_ui": 1,
        "game_id": "game-1",
        "game_config": current_config,
    }
    plugin = StitchUiPlugin()
    image = object()
    plugin.forward({"img": image, "shared": shared})

    process = _FakeHmUiProcess.instances[0]
    assert process.kwargs["preview_names"] == ("Stitched", "Camera 1", "Camera 2")
    assert shared["hm_ui_process"] is process
    assert process.previews[-1] == (image, "Stitched")

    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 125
    process.changed = True
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert current_config["rink"]["camera"]["color"]["brightness"] == 1.25

    process.queue_reset(system=True)
    process.queue_action("save")
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 0.5
    assert current_config["rink"]["camera"]["color"]["brightness"] == 1.0
    assert "post_stitch_rotate_degrees" not in saved.get("stitching", {})

    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 125
    process.changed = True
    plugin.forward({"img": image, "shared": shared})

    process.queue_action("save")
    plugin.forward({"img": image, "shared": shared})

    assert saved["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.25}

    # Save must use its own snapshot even if a later reset is in the same poll.
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 75
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 150
    process.queue_action("save")
    process.queue_reset(system=True)
    plugin.forward({"img": image, "shared": shared})

    assert saved["stitching"]["post_stitch_rotate_degrees"] == 15.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.5}
    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 0.5

    # An edit after reset must be applied before a following Save snapshot.
    process.queue_reset(system=True)
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.values["Tracker Controls (Stitched Color)"]["Brightness_Multiplier_x100"] = 125
    process.queue_action("save")
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["stitching"]["post_stitch_rotate_degrees"] == 10.0
    assert saved["rink"]["camera"]["color"] == {"brightness": 1.25}

    # Multiple resets retain click order and restore the exact source value.
    process.queue_reset(system=True)
    process.queue_reset(system=False)
    plugin.forward({"img": image, "shared": shared})

    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 5.0

    plugin.finalize()
    assert process.closed is True
    assert shared["hm_ui_process"] is None


def should_propagate_stitch_ui_initialization_failure(monkeypatch):
    class FailingHmUiProcess(_FakeHmUiProcess):
        def add_window(self, name: str) -> None:
            del name
            raise OSError("cannot start hm-ui")

    FailingHmUiProcess.instances.clear()
    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", FailingHmUiProcess)
    plugin = StitchUiPlugin()

    with pytest.raises(RuntimeError, match="Failed to initialize stitch camera UI") as exc_info:
        plugin.forward(
            {
                "img": object(),
                "shared": {
                    "camera_ui": 1,
                    "game_config": _config(rotation=0.0),
                },
            }
        )

    assert str(exc_info.value.__cause__) == "cannot start hm-ui"
    assert FailingHmUiProcess.instances[0].closed is True


def should_retry_stitch_ui_actions_and_restore_final_values_after_failure(monkeypatch):
    class RetryingHmUiProcess(_FakeHmUiProcess):
        def consume_action_events(self, *, poll: bool = True):
            del poll
            return list(self.actions)

        def acknowledge_action_events(self, through_seq: int, *, message: str = "") -> None:
            super().acknowledge_action_events(through_seq)
            self.actions = [action for action in self.actions if action.seq > through_seq]

    RetryingHmUiProcess.instances.clear()
    current_config = _config(rotation=5.0)
    save_attempts = []
    clock = [100.0]

    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", RetryingHmUiProcess)
    monkeypatch.setattr(stitch_ui_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        stitch_ui_module,
        "get_config",
        lambda **_kwargs: _config(rotation=0.5),
    )
    monkeypatch.setattr(
        stitch_ui_module,
        "get_game_config_private",
        lambda **_kwargs: {},
    )

    def save_private(_game_id, _data, verbose=True):
        del verbose
        save_attempts.append(_game_id)
        if len(save_attempts) == 1:
            raise OSError("temporary config write failure")

    monkeypatch.setattr(stitch_ui_module, "save_private_config", save_private)

    plugin = StitchUiPlugin()
    context = {
        "img": object(),
        "shared": {
            "camera_ui": 1,
            "game_id": "game-1",
            "game_config": current_config,
        },
    }
    plugin.forward(context)

    process = RetryingHmUiProcess.instances[0]
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.queue_action("save")
    process.queue_reset(system=True)
    final_values = process.control_values()

    plugin.forward(context)
    assert process.closed is False
    assert process.acknowledged_seq == 0
    assert len(process.actions) == 2
    assert process.control_values() == final_values
    assert current_config["stitching"]["post_stitch_rotate_degrees"] == 0.5
    assert save_attempts == ["game-1"]

    # Persistent failures are not retried or logged on every video frame.
    plugin.forward(context)
    assert process.acknowledged_seq == 0
    assert save_attempts == ["game-1"]

    clock[0] += 0.5
    plugin.forward(context)
    assert process.closed is False
    assert process.acknowledged_seq == 2
    assert process.actions == []
    assert process.control_values() == final_values
    assert save_attempts == ["game-1", "game-1"]


def should_suppress_local_preview_when_rust_camera_ui_owns_it(monkeypatch):
    _FakeShower.instances.clear()
    monkeypatch.setattr(video_preview_module, "Shower", _FakeShower)
    plugin = VideoPreviewPlugin()
    context = {
        "img": torch.zeros((1, 8, 8, 3), dtype=torch.uint8),
        "fps": 30.0,
        "shared": {"game_config": {"video_out": {"show_image": True}}},
    }

    plugin(context)
    assert len(_FakeShower.instances) == 1
    local_shower = _FakeShower.instances[0]

    context["shared"]["camera_ui"] = 1
    plugin(context)
    assert local_shower.closed is True
    assert plugin._shower is None

    _FakeShower.instances.clear()
    plugin = VideoPreviewPlugin()
    plugin(context)
    assert _FakeShower.instances == []


def should_save_and_reset_shadow_lift_controls(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    config = _config(rotation=0)
    system = copy.deepcopy(config)
    config["stitching"]["left"]["color"].update(shadow_lift=25, shadow_lift_black_point=True)
    saved = {}
    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", _FakeHmUiProcess)
    monkeypatch.setattr(stitch_ui_module, "get_config", lambda **_kwargs: copy.deepcopy(system))
    monkeypatch.setattr(stitch_ui_module, "get_game_config_private", lambda **_kwargs: {})
    monkeypatch.setattr(
        stitch_ui_module,
        "save_private_config",
        lambda _game_id, data, **_kwargs: saved.update(copy.deepcopy(data)),
    )
    plugin = StitchUiPlugin()
    context = {"shared": {"camera_ui": True, "game_id": "game", "game_config": config}}
    plugin.forward(context)
    process = _FakeHmUiProcess.instances[0]
    controls = process.values["Tracker Controls (Left Color)"]
    assert controls["Shadow_Lift_Percent"] == 25
    assert controls["Shadow_Lift_Black_Point"] == 1
    controls["Shadow_Lift_Percent"] = 80
    controls["Shadow_Lift_Black_Point"] = 0
    process.queue_action("save")
    plugin.forward(context)
    assert saved["stitching"]["left"]["color"] == {
        "shadow_lift": 80.0,
        "shadow_lift_black_point": False,
    }
    process.queue_reset(system=True)
    plugin.forward(context)
    assert "shadow_lift" not in config["stitching"]["left"]["color"]
    assert "shadow_lift_black_point" not in config["stitching"]["left"]["color"]
    plugin.finalize()


def _blend_plugin(monkeypatch, *, game: dict, system: dict, saved: dict):
    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", _FakeHmUiProcess)
    monkeypatch.setattr(stitch_ui_module, "get_config", lambda **_k: copy.deepcopy(system))
    # `saved` stands in for the private config on disk, so a later save sees what
    # an earlier one wrote and can remove keys that no longer override anything.
    monkeypatch.setattr(
        stitch_ui_module, "get_game_config_private", lambda **_k: copy.deepcopy(saved)
    )

    def save_private(_game_id, data, verbose=True):
        del verbose
        saved.clear()
        saved.update(copy.deepcopy(data))

    monkeypatch.setattr(stitch_ui_module, "save_private_config", save_private)
    shared = {"camera_ui": 1, "game_id": "game-1", "game_config": game}
    plugin = StitchUiPlugin()
    plugin.forward({"img": object(), "shared": shared})
    return plugin, shared, _FakeHmUiProcess.instances[-1]


def should_offer_seam_blend_controls_opened_on_the_game_value(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"].update(blend_mode="alpha", blend_feather_fraction=0.12)
    system = _config(rotation=0.0)
    system["stitching"].update(blend_mode="laplacian", blend_feather_fraction=0.05)

    _plugin, _shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved={})

    assert process.choices["Stitch Blend"]["Seam_Blend_Mode"] == [
        "Laplacian (multi-band)",
        "Alpha (feathered seam)",
        "Hard seam (no blending)",
    ]
    assert process.values["Stitch Blend"]["Seam_Blend_Mode"] == 1
    assert process.values["Stitch Blend"]["Seam_Feather_Percent"] == 12
    # The stitcher is built once, so both controls take effect on the next run.
    for name in ("Seam_Blend_Mode", "Seam_Feather_Percent"):
        assert "next stitch run" in process.descriptions["Stitch Blend"][name]
    assert process.system_defaults["Stitch Blend"] == {
        "Seam_Blend_Mode": 0,
        "Seam_Feather_Percent": 5,
    }


def should_apply_save_and_reset_the_seam_blend(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"].update(blend_mode="laplacian", blend_feather_fraction=0.05)
    system = copy.deepcopy(game)
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    process.values["Stitch Blend"]["Seam_Blend_Mode"] = 1
    process.values["Stitch Blend"]["Seam_Feather_Percent"] = 20
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})

    assert game["stitching"]["blend_mode"] == "alpha"
    assert game["stitching"]["blend_feather_fraction"] == pytest.approx(0.2)

    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})
    assert saved["stitching"]["blend_mode"] == "alpha"
    assert saved["stitching"]["blend_feather_fraction"] == pytest.approx(0.2)

    # Resetting to system defaults drops both overrides from the private config.
    process.queue_reset(system=True)
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})
    assert game["stitching"]["blend_mode"] == "laplacian"
    assert "blend_mode" not in saved.get("stitching", {})
    assert "blend_feather_fraction" not in saved.get("stitching", {})


def should_keep_an_unrenderable_game_blend_mode_selectable(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    # multiblend is HockeyMON's CPU-only mode; the stitch UI drives the GPU path.
    game["stitching"].update(blend_mode="multiblend")
    system = _config(rotation=0.0)
    system["stitching"].update(blend_mode="laplacian")
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    labels = process.choices["Stitch Blend"]["Seam_Blend_Mode"]
    assert labels[-1] == "Multiblend (CPU) - not supported here"
    # Opening on its own entry is what makes picking a real mode an index change.
    assert process.values["Stitch Blend"]["Seam_Blend_Mode"] == len(labels) - 1

    process.values["Stitch Blend"]["Seam_Blend_Mode"] = 1
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    assert game["stitching"]["blend_mode"] == "alpha"


def should_survive_a_malformed_blend_config_without_losing_its_values(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"].update(blend_mode="pyramid", blend_feather_fraction=9.0)
    system = _config(rotation=0.0)

    _plugin, _shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved={})

    # The UI opens rather than failing, the three renderable modes are offered,
    # and the mode this build cannot parse is its own entry instead of being
    # silently presented as Laplacian.
    assert process.choices["Stitch Blend"]["Seam_Blend_Mode"] == [
        "Laplacian (multi-band)",
        "Alpha (feathered seam)",
        "Hard seam (no blending)",
        "pyramid - not supported here",
    ]
    assert process.values["Stitch Blend"]["Seam_Blend_Mode"] == 3
    # A width the slider cannot hold opens at the default.
    assert process.values["Stitch Blend"]["Seam_Feather_Percent"] == 5


def should_not_rewrite_blend_values_the_operator_never_touched(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=5.0)
    # An underscore spelling HStream writes, and a width the whole-percent
    # slider cannot represent.
    game["stitching"].update(blend_mode="gpu_hard_seam", blend_feather_fraction=0.125)
    system = copy.deepcopy(game)
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    # Move an unrelated control, which applies every control.
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})

    assert game["stitching"]["blend_mode"] == "gpu_hard_seam"
    assert game["stitching"]["blend_feather_fraction"] == 0.125
    # Neither may appear as a private override: they still match the system config.
    assert "blend_mode" not in saved.get("stitching", {})
    assert "blend_feather_fraction" not in saved.get("stitching", {})

    # Moving the control itself still writes, canonicalized.
    process.values["Stitch Blend"]["Seam_Blend_Mode"] = 0
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    assert game["stitching"]["blend_mode"] == "laplacian"


def should_offer_only_the_modes_the_configured_blender_can_render(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"].update(blend_mode="multiblend", python_blender=True)
    system = copy.deepcopy(game)

    _plugin, _shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved={})

    labels = process.choices["Stitch Blend"]["Seam_Blend_Mode"]
    # Alpha is a GPU kernel with no Python equivalent, so it must not be offered
    # here. multiblend is the game's own mode and no video path can render it, so
    # it keeps a marked entry rather than being offered as a choice.
    assert labels == [
        "Laplacian (multi-band)",
        "Hard seam (no blending)",
        "Multiblend (CPU) - not supported here",
    ]
    assert process.values["Stitch Blend"]["Seam_Blend_Mode"] == len(labels) - 1
    # No feather control at all: alpha cannot be selected, so a width here could
    # never act on anything.
    assert "Seam_Feather_Percent" not in process.values["Stitch Blend"]


@pytest.mark.parametrize("configured", [None, "feathered-alpha-v2"])
def should_not_invent_a_blend_mode_for_a_config_that_names_none(monkeypatch, configured):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=5.0)
    if configured is None:
        game["stitching"].pop("blend_mode", None)
    else:
        # A mode a newer HStream could write, which this build cannot parse.
        game["stitching"]["blend_mode"] = configured
    system = copy.deepcopy(game)
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    # An unparseable mode keeps its own entry rather than showing as Laplacian.
    labels = process.choices["Stitch Blend"]["Seam_Blend_Mode"]
    if configured is not None:
        assert labels[-1] == f"{configured} - not supported here"
        assert process.values["Stitch Blend"]["Seam_Blend_Mode"] == len(labels) - 1
    else:
        assert len(labels) == 3

    # Moving an unrelated control must not write a blend override.
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})

    assert game["stitching"].get("blend_mode") == configured
    assert "blend_mode" not in saved.get("stitching", {})


def should_repair_a_feather_width_the_slider_cannot_hold(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=5.0)
    game["stitching"]["blend_feather_fraction"] = 9.0
    system = _config(rotation=5.0)
    system["stitching"]["blend_feather_fraction"] = 0.05
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    # The slider opens at the default, and the next apply writes that back rather
    # than leaving a value the next run would reject.
    assert process.values["Stitch Blend"]["Seam_Feather_Percent"] == 5
    process.values["Stitch Alignment"]["Stitch_Rotate_Degrees"] = 80
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    assert game["stitching"]["blend_feather_fraction"] == pytest.approx(0.05)


def should_repair_rather_than_restore_a_system_value_the_controls_cannot_hold(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"].update(blend_mode="laplacian", blend_feather_fraction=0.05)
    system = _config(rotation=0.0)
    system["stitching"].update(blend_mode="laplacian", blend_feather_fraction=9.0)
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    process.queue_reset(system=True)
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})

    # The slider cannot hold 9.0, so a reset must leave the config matching what
    # the operator sees, not restore a value the next run would reject.
    assert process.values["Stitch Blend"]["Seam_Feather_Percent"] == 5
    assert game["stitching"]["blend_feather_fraction"] == pytest.approx(0.05)


def should_not_save_an_override_that_only_respells_the_system_mode(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"]["blend_mode"] = "gpu_hard_seam"
    system = copy.deepcopy(game)
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    labels = process.choices["Stitch Blend"]["Seam_Blend_Mode"]
    hard_seam = labels.index("Hard seam (no blending)")
    # A round trip through the combo writes the canonical spelling, which means
    # the same thing as the system config's alias and must not shadow it.
    for index in (0, hard_seam):
        process.values["Stitch Blend"]["Seam_Blend_Mode"] = index
        process.changed = True
        plugin.forward({"img": object(), "shared": shared})
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})

    assert game["stitching"]["blend_mode"] == "gpu-hard-seam"
    assert "blend_mode" not in saved.get("stitching", {})


def should_recognize_an_alias_spelling_of_a_runnable_mode(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    # An alias HStream's ParseBlendMode accepts. It names a mode this path runs,
    # so it must not become a second entry for the same seam.
    game["stitching"]["blend_mode"] = "hard-seam"
    system = copy.deepcopy(game)
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    assert process.choices["Stitch Blend"]["Seam_Blend_Mode"] == [
        "Laplacian (multi-band)",
        "Alpha (feathered seam)",
        "Hard seam (no blending)",
    ]
    assert process.values["Stitch Blend"]["Seam_Blend_Mode"] == 2

    # Re-picking the same seam is not an override of the system's spelling.
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})
    assert "blend_mode" not in saved.get("stitching", {})


def should_not_write_a_mode_no_renderer_can_run(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"]["blend_mode"] = "multiblend"
    system = _config(rotation=0.0)
    system["stitching"]["blend_mode"] = "laplacian"
    saved: dict = {}

    plugin, shared, process = _blend_plugin(monkeypatch, game=game, system=system, saved=saved)

    labels = process.choices["Stitch Blend"]["Seam_Blend_Mode"]
    unrunnable = labels.index("Multiblend (CPU) - not supported here")

    # Picking a real mode works.
    process.values["Stitch Blend"]["Seam_Blend_Mode"] = 0
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    assert game["stitching"]["blend_mode"] == "laplacian"

    # Picking the marked entry back must not write a mode the next graph build
    # would refuse, leaving the game unlaunchable with no UI left to fix it.
    process.values["Stitch Blend"]["Seam_Blend_Mode"] = unrunnable
    process.changed = True
    plugin.forward({"img": object(), "shared": shared})
    process.queue_action("save")
    plugin.forward({"img": object(), "shared": shared})
    assert game["stitching"]["blend_mode"] == "laplacian"
    assert "blend_mode" not in saved.get("stitching", {})


def should_save_a_blend_repair_before_the_stitch_graph_starts(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"].update(blend_mode="multiblend", python_blender=False)
    game["aspen"] = {
        "plugins": {
            "stitching": {
                "params": {
                    "blend_mode": "multiblend",
                    "blend_feather_fraction": 0.05,
                    "max_blend_levels": 11,
                    "python_blender": False,
                }
            }
        }
    }
    system = _config(rotation=0.0)
    system["stitching"].update(blend_mode="laplacian", python_blender=False)
    saved: dict = {}

    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", _FakeHmUiProcess)
    monkeypatch.setattr(stitch_ui_module, "get_config", lambda **_k: copy.deepcopy(system))
    monkeypatch.setattr(
        stitch_ui_module, "get_game_config_private", lambda **_k: copy.deepcopy(saved)
    )

    def save_private(_game_id, data, verbose=True):
        del verbose
        saved.clear()
        saved.update(copy.deepcopy(data))

    monkeypatch.setattr(stitch_ui_module, "save_private_config", save_private)

    def select_and_save(_seconds):
        process = _FakeHmUiProcess.instances[-1]
        process.values["Stitch Blend"]["Seam_Blend_Mode"] = 1
        process.changed = True
        process.queue_action("save")

    monkeypatch.setattr(stitch_ui_module.time, "sleep", select_and_save)

    assert (
        stitch_ui_module.repair_blend_before_stitch(
            game,
            "game-1",
            raw_graph_params={
                "blend_mode": "GLOBAL.stitching.blend_mode",
                "blend_feather_fraction": "GLOBAL.stitching.blend_feather_fraction",
                "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                "python_blender": "GLOBAL.stitching.python_blender",
            },
        )
        is True
    )
    assert game["stitching"]["blend_mode"] == "alpha"
    assert saved["stitching"]["blend_mode"] == "alpha"
    # The already-resolved graph copy is stale; the caller ends this run so the
    # next graph can resolve the saved shared value.
    assert game["aspen"]["plugins"]["stitching"]["params"]["blend_mode"] == "multiblend"
    process = _FakeHmUiProcess.instances[-1]
    assert list(process.values) == ["Stitch Blend"]
    assert list(saved) == ["stitching"]
    assert process.closed is True


def should_fail_clearly_when_the_repair_ui_closes_without_saving(monkeypatch):
    _FakeHmUiProcess.instances.clear()
    game = _config(rotation=0.0)
    game["stitching"]["blend_mode"] = "multiblend"
    monkeypatch.setattr(stitch_ui_module, "HmUiProcess", _FakeHmUiProcess)

    def close_ui(_seconds):
        _FakeHmUiProcess.instances[-1].closed = True

    monkeypatch.setattr(stitch_ui_module.time, "sleep", close_ui)

    with pytest.raises(ValueError, match="closed without saving"):
        stitch_ui_module.repair_blend_before_stitch(game, "game-1")


def should_report_an_unrepairable_aspen_blend_override() -> None:
    game = _config(rotation=0.0)
    game["stitching"]["blend_mode"] = "laplacian"
    game["aspen"] = {"plugins": {"stitching": {"params": {"blend_mode": "multiblend"}}}}

    with pytest.raises(ValueError, match="edit aspen.plugins.stitching.params.blend_mode"):
        stitch_ui_module.repair_blend_before_stitch(
            game, "game-1", raw_graph_params={"blend_mode": "multiblend"}
        )


def should_not_claim_to_repair_a_private_config_that_will_be_ignored() -> None:
    game = _config(rotation=0.0)
    game["stitching"]["blend_mode"] = "multiblend"

    with pytest.raises(ValueError, match="--ignore-private-config would discard"):
        stitch_ui_module.repair_blend_before_stitch(game, "game-1", ignore_private_config=True)


def should_not_save_a_repair_that_a_cli_override_will_replace() -> None:
    game = _config(rotation=0.0)
    game["stitching"]["blend_mode"] = "multiblend"

    with pytest.raises(ValueError, match="change or remove that override"):
        stitch_ui_module.repair_blend_before_stitch(
            game, "game-1", run_blend_overrides=("blend_mode",)
        )
