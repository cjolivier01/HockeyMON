"""HStream desktop protocol over HockeyMON's frame-boundary control mailbox.

This module is used by the supervising ``hm-cli`` process. It does not import
Torch, create a tracker, or claim that a control was applied when it was merely
queued. The pipeline owns mutation and acknowledges its request after applying
the complete batch. Each request is bound to that pipeline's session identity.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_INFO = {
    "protocol": "hstream-ui",
    "version": 1,
    "backend": "hockeymon",
    "preview": "jpeg-files",
}
PREVIEW_NAMES = {
    "program": "Final",
    "main": "Final",
    "stitched": "Stitched",
    "source0": "Camera 1",
    "source1": "Camera 2",
    "none": "none",
}
_TUNING_CONTROLS = {
    "stop-translation-on-dir-change-delay": ("Stop_Direction_Change_Delay_Frames", 1),
    "zoom-in-aggressiveness": ("Zoom_In_Aggressiveness", 1),
    "ignore-largest-bbox-count": ("Ignore_Largest_Count", 1),
    "oversized-bbox-percent": ("Oversized_Player_Percent_x10", 10),
    "ignore-oversized-bboxes": ("Ignore_Oversized_Players", 1),
    "cancel-stop-on-opposite-dir": ("Cancel_Stop_On_Opposite_Direction", 1),
    "cancel-stop-hysteresis-frames": ("Stop_Cancel_Hysteresis_Frames", 1),
    "stop-delay-cooldown-frames": ("Stop_Delay_Cooldown_Frames", 1),
    "time-to-dest-speed-limit-frames": ("Time_To_Dest_Speed_Limit_Frames", 1),
    "post-nonstop-stop-delay-count": ("Post_Nonstop_Stop_Delay_Frames", 1),
    "overshoot-stop-delay-count": ("Overshoot_Stop_Delay_Frames", 1),
    "overshoot-scale-speed-ratio": ("Overshoot_Speed_Ratio_x100", 100),
    "max-speed-x": ("Max_Speed_X_x10", 10),
    "max-speed-y": ("Max_Speed_Y_x10", 10),
    "max-accel-x": ("Max_Accel_X_x10", 10),
    "max-accel-y": ("Max_Accel_Y_x10", 10),
}
_MAX_DOCUMENT_BYTES = 1024 * 1024


def read_document(path: Path, *, yaml_document: bool = False) -> dict:
    """Read one bounded, nonsymlinked regular protocol/config file."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_DOCUMENT_BYTES:
            raise ValueError(f"Protocol document must be a bounded regular file: {path}")
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        data = stream.read(_MAX_DOCUMENT_BYTES + 1)
    if len(data) > _MAX_DOCUMENT_BYTES:
        raise ValueError(f"Protocol document is too large: {path}")
    value = yaml.safe_load(data) if yaml_document else json.loads(data)
    if not isinstance(value, dict):
        raise TypeError(f"Protocol document must contain a mapping: {path}")
    return value


def write_document(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(value: str, *, minimum: int = 0) -> int:
    if not re.fullmatch(r"-?[0-9]+", value):
        raise ValueError("Expected an integer")
    result = int(value)
    if result < minimum or result > (1 << 63) - 1:
        raise ValueError("Integer is outside the supported range")
    return result


def _scaled_integer(value: Any, scale: float = 1) -> int:
    if not isinstance(value, (int, float, bool, str)):
        raise TypeError("Control value must be numeric")
    result = float(value) * scale
    if not math.isfinite(result) or abs(result - round(result)) > 1e-6:
        raise ValueError("Control value cannot be represented exactly by this runtime")
    return round(result)


@dataclass
class _Pending:
    deadline: float
    success: list[str]
    failure: Callable[[str], str]
    payload: dict


def shared_playback_fps(left_fps: float, right_fps: float) -> float:
    """Resolve the delivered rate supported by integer source-frame decimation."""
    low, high = sorted((float(left_fps), float(right_fps)))
    if low <= 0 or not all(math.isfinite(rate) for rate in (low, high)):
        raise ValueError("Shared camera frame rates must be finite and positive")
    ratio = high / low
    if not math.isclose(ratio, round(ratio)):
        raise ValueError(
            "Shared HockeyMON camera frame rates must match or have an integer ratio; "
            f"received {left_fps} and {right_fps} FPS"
        )
    return low


class RuntimeProtocol:
    """Translate native desktop messages and report acknowledged Python state.

    Call ``handle_line`` for stdin messages and ``poll`` from the supervisor's
    event loop. ``seek`` starts a supervised replacement at the requested offset;
    call ``seek_completed`` only after that replacement produces its first frame.
    Hooks for external owners (audio, scoreboard and preview layers) must either
    complete their operation or raise; an absent hook fails explicitly.
    """

    def __init__(
        self,
        session: Path,
        emit: Callable[[str], None],
        *,
        game_dir: Path | None = None,
        seek: Callable[[int, int], None] | None = None,
        mute_audio: Callable[[bool], None] | None = None,
        process_paused: Callable[[bool], None] | None = None,
        scoreboard: Callable[[], None] | None = None,
        preview_overlays: Callable[[bool, bool, bool], None] | None = None,
        graph: dict | None = None,
        total_ns: int | None = None,
        start_ns: int = 0,
        initial_preview: str = "program",
        request_timeout: float = 5.0,
    ) -> None:
        self.session = Path(session)
        self.game_dir = Path(game_dir) if game_dir is not None else None
        self._scoreboard_session = None
        self.emit = emit
        self.seek = seek
        self.mute_audio = mute_audio
        self.process_paused = process_paused
        self.scoreboard = scoreboard
        self.preview_overlays = preview_overlays
        self.graph = graph or {}
        self.total_ns = total_ns
        self.start_ns = start_ns
        self.frame_offset = 0
        self.request_timeout = request_timeout
        self._pending: dict[str, _Pending] = {}
        self._session_id = ""
        self._generation = 0
        self._pause_generation = 0
        self._progress_generation = 0
        if initial_preview not in PREVIEW_NAMES or initial_preview == "main":
            raise ValueError(f"Unknown initial preview channel: {initial_preview}")
        self._preview_generation = 1
        self._preview_requested_generation = 1
        self._active_channel = initial_preview
        self._preview_ready_identity: tuple | None = None
        self._runtime_mtime: int | None = None
        self._last_position_ns = 0
        self._child_offset_ns = 0
        self._clock_origin = time.monotonic()
        self._clock_position_ns = 0
        self._replay_values: dict[str, dict[str, int]] = {}
        self._overlay_layers = [
            name in os.environ.get("HSTREAM_UI_PREVIEW_OVERLAYS", "").split(",")
            for name in ("players", "play", "rink")
        ]

    def _spec(self) -> dict:
        spec = read_document(self.session / "spec.json")
        identity = spec.get("session_id")
        if not isinstance(identity, str) or not identity or not spec.get("windows"):
            raise ValueError("Python runtime controls are not ready")
        if self._session_id and identity != self._session_id:
            raise ValueError("Python runtime session changed; supervisor must reset it")
        if not self._session_id:
            self._session_id = identity
            self._generation += 1
            self._inspect_emit("session", 0, status="ok")
        return spec

    def _queue(
        self,
        payload: dict,
        success: list[str],
        failure: Callable[[str], str] | None = None,
    ) -> None:
        self._spec()
        if len(self._pending) >= 32:
            raise ValueError("Runtime request queue is full")
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.request_timeout
        payload = {**payload, "session_id": self._session_id, "deadline": deadline}
        self._pending[request_id] = _Pending(
            deadline,
            success,
            failure or (lambda reason: f"runtime command failed: {reason}"),
            payload,
        )
        try:
            write_document(self.session / "runtime-requests" / f"{request_id}.json", payload)
        except (OSError, ValueError, TypeError):
            del self._pending[request_id]
            raise

    def _control(self, values: dict, name: str, value: Any, *, window: str | None = None) -> None:
        found = [
            (item["name"], control)
            for item in self._spec()["windows"]
            for control in item["controls"]
            if control["name"] == name and (window is None or item["name"] == window)
        ]
        if len(found) != 1:
            raise ValueError(f"Runtime control is unavailable or ambiguous: {name}")
        owner, control = found[0]
        value = _scaled_integer(value)
        if not control.get("min_value", 0) <= value <= control["max_value"]:
            raise ValueError(f"Runtime control {name} is outside its supported range")
        previous = values.setdefault(owner, {}).get(name)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting assignments for runtime control {name}")
        values[owner][name] = value

    def _translate_property(self, element: str, prop: str, value: str, values: dict) -> None:
        if element == "dsplaytracker0" and prop == "runtime-tuning-config-file":
            config = read_document(Path(value), yaml_document=True)
            tracker = config["play-tracker"]
            tuning = tracker["hstream-runtime-tuning"]
            if not isinstance(tuning, dict):
                raise ValueError("Runtime tuning must be a mapping")
            unknown = tuning.keys() - _TUNING_CONTROLS.keys()
            if unknown:
                raise ValueError(f"Unknown runtime tuning fields: {sorted(unknown)}")
            for key, setting in tuning.items():
                name, scale = _TUNING_CONTROLS[key]
                self._control(values, name, _scaled_integer(setting, scale))
            for key, name in (
                ("hstream-apply-to-fast-box", "Apply_To_Fast_Box"),
                ("hstream-apply-to-follower-box", "Apply_To_Follower_Box"),
            ):
                selected = tracker[key]
                if not isinstance(selected, bool):
                    raise TypeError(f"{key} must be a boolean")
                self._control(values, name, int(selected))
            return
        if element in {"dsplaytracker0", "playcropper0"} and prop.startswith(
            "fixed-edge-rotation-angle"
        ):
            angle = _scaled_integer(value, 10)
            if prop == "fixed-edge-rotation-angle":
                self._control(values, "Link_Fixed_Edge_Rotation_Left_Right", 1)
                self._control(values, "Left_Fixed_Edge_Rotation_Angle_x10", angle)
                self._control(values, "Right_Fixed_Edge_Rotation_Angle_x10", angle)
            elif prop in {"fixed-edge-rotation-angle-left", "fixed-edge-rotation-angle-right"}:
                self._control(values, "Link_Fixed_Edge_Rotation_Left_Right", 0)
                side = "Left" if prop.endswith("left") else "Right"
                self._control(values, f"{side}_Fixed_Edge_Rotation_Angle_x10", angle)
            else:
                raise ValueError(f"Unknown rotation property: {prop}")
            return
        if element == "playcropper0" and prop == "scoreboard-position":
            choices = ["upper-left", "upper-center", "upper-right"]
            if value not in choices:
                raise ValueError("Invalid scoreboard position")
            self._control(values, "Scoreboard_Position", choices.index(value))
            return
        if element in {"playcropper0", "hmstitcher0"} and prop in {
            "shadow-lift",
            "shadow-lift-black-point",
        }:
            name = "Shadow_Lift_Percent" if prop == "shadow-lift" else "Shadow_Lift_Black_Point"
            self._control(values, name, value, window="Tracker Controls (Color)")
            return
        if element in {"playcropper0", "hmstitcher0"} and prop == "exposure":
            # HStream's reference curve is 2^(setting / 2), not a linear gain.
            # Keep its 0.01 setting steps instead of rounding Python's EV slider.
            self._control(
                values,
                "HStream_Exposure_x100",
                _scaled_integer(value, 100),
                window="Tracker Controls (Color)",
            )
            return
        raise ValueError(f"No Python runtime owner for {element}.{prop}")

    def _properties(self, arguments: str, *, batch: bool) -> None:
        assignments = arguments.split(";") if batch else [arguments]
        if not 1 <= len(assignments) <= 64:
            raise ValueError("Runtime property batch is empty or too large")
        parsed = []
        values: dict[str, dict[str, int]] = {}
        for assignment in assignments:
            match = re.fullmatch(r"([A-Za-z0-9_-]+) ([A-Za-z0-9_-]+)=(.*)", assignment)
            if match is None:
                raise ValueError("Malformed runtime property assignment")
            parsed.append(match.groups())
        if parsed == [("playcropper0", "scoreboard-reselect", "1")]:
            if self.scoreboard is not None:
                self.scoreboard()
            elif self.game_dir is not None:
                from hmlib.ui.hstream_scoreboard import ScoreboardSession

                self._spec()
                if self._scoreboard_session is not None:
                    self._scoreboard_session.close()
                self._scoreboard_session = ScoreboardSession(
                    self.game_dir, self.session, self._session_id, self.emit
                )
                self._scoreboard_session.start()
            else:
                raise ValueError("Scoreboard selection is not connected to this run")
            self.emit("runtime property playcropper0 scoreboard-reselect=1")
            return
        epoch = None
        for element, prop, value in parsed:
            if element == "hmstitcher0" and prop == "stitched-output-epoch":
                if epoch is not None:
                    raise ValueError("A runtime batch can contain only one output epoch")
                epoch = value
            else:
                self._translate_property(element, prop, value, values)
        success = [f"runtime property {element} {prop}={value}" for element, prop, value in parsed]
        names = ",".join(f"{element}.{prop}" for element, prop, _ in parsed)
        self._queue(
            (
                {
                    "kind": "operation",
                    "operation": {
                        "kind": "stitched-output-epoch",
                        "serialized": epoch,
                        "values": values,
                    },
                }
                if epoch is not None
                else {"kind": "controls", "values": values}
            ),
            success,
            lambda reason: f"runtime command failed: {names}: {reason}",
        )

    def handle_line(self, line: str) -> bool:
        """Consume one command; failures are visible and do not mutate the child."""
        line = line.rstrip("\r\n")
        try:
            if len(line.encode("utf-8")) > 65536 or "\0" in line or "\n" in line:
                raise ValueError("Runtime command is oversized or contains control characters")
            if line in {"q", "p", "r"}:
                action = {"q": "stop", "p": "pause", "r": "resume"}[line]
                self._queue({"kind": "playback", "action": action}, [])
                return True
            if not line.startswith("@"):
                raise ValueError("Unknown runtime command")
            command, _, arguments = line[1:].partition(" ")
            if command in {"set-property", "set-properties"}:
                self._properties(arguments, batch=command == "set-properties")
            elif command in {"process-pause", "process-resume"}:
                generation = _integer(arguments, minimum=1)
                if command == "process-pause":
                    if generation <= self._pause_generation:
                        raise ValueError("Stale process pause generation")
                    if self.process_paused is not None:
                        self.process_paused(True)
                    self._pause_generation = generation
                    status = "prepared"
                else:
                    if generation != self._pause_generation:
                        raise ValueError("Stale process resume generation")
                    # Before the first frame there is no pacing clock to reset.
                    if (self.session / "spec.json").is_file():
                        self._queue({"kind": "clock-reset"}, [])
                    if self.process_paused is not None:
                        self.process_paused(False)
                    self._clock_origin = time.monotonic()
                    self._clock_position_ns = self._last_position_ns
                    status = "resumed"
                self.emit(f"HSTREAM_PROCESS_PAUSE status={status} generation={generation}")
            elif command == "reset-progress-rate":
                self._progress_generation = _integer(arguments, minimum=1)
                self._clock_origin = time.monotonic()
                self._clock_position_ns = self._last_position_ns
                self.emit(f"HSTREAM_PROGRESS status=reset generation={self._progress_generation}")
            elif command in {"seek", "seek-relative"}:
                position, generation = arguments.split()
                target = _integer(
                    position, minimum=-(1 << 63) if command.endswith("relative") else 0
                )
                generation = _integer(generation, minimum=1)
                if command.endswith("relative"):
                    target = max(0, self._last_position_ns + target)
                if self.seek is None:
                    raise ValueError("Seeking is not available for this run")
                if self.total_ns is not None and target >= self.total_ns:
                    raise ValueError("Seek position is outside the recording")
                self.seek(target, generation)
            elif command == "set-render-audio-muted":
                muted = _integer(arguments)
                if muted not in {0, 1} or self.mute_audio is None:
                    raise ValueError("Local audio monitor is unavailable or mute value is invalid")
                self.mute_audio(bool(muted))
            elif command == "set-preview-active":
                channel, generation_text = arguments.split()
                generation = _integer(generation_text, minimum=1)
                if (
                    channel not in PREVIEW_NAMES
                    or channel == "main"
                    or generation <= self._preview_requested_generation
                ):
                    raise ValueError("Invalid or stale preview request")
                name = PREVIEW_NAMES[channel]
                self._queue(
                    {"kind": "preview", "name": name, "channel": channel, "generation": generation},
                    [
                        f"HSTREAM_PREVIEW channel={channel} status=activated generation={generation} message=selected"
                    ],
                    lambda reason: f"HSTREAM_PREVIEW channel={channel} status=failed generation={generation} message={reason}",
                )
                self._preview_requested_generation = generation
            elif command == "set-preview-overlays":
                generation, *layers = arguments.split()
                generation = _integer(generation, minimum=1)
                if len(layers) != 3 or any(value not in {"0", "1"} for value in layers):
                    raise ValueError("Preview overlay request requires three booleans")
                layer_text = (
                    f"generation={generation} players={layers[0]} play={layers[1]} rink={layers[2]}"
                )
                if self.preview_overlays is None:
                    self._queue(
                        {
                            "kind": "operation",
                            "operation": {
                                "kind": "preview-overlays",
                                "layers": [value == "1" for value in layers],
                            },
                        },
                        [f"HSTREAM_PREVIEW_OVERLAYS status=applied {layer_text}"],
                        lambda reason: f"HSTREAM_PREVIEW_OVERLAYS status=failed {layer_text} reason={reason.replace(' ', '-')}",
                    )
                else:
                    self.preview_overlays(*(value == "1" for value in layers))
                    self.emit(f"HSTREAM_PREVIEW_OVERLAYS status=applied {layer_text}")
            elif command == "capture-preview-frame":
                channel, destination = arguments.split(" ", 1)
                preview = self._preview_path(channel)
                target = Path(destination)
                if not target.is_absolute() or target.is_symlink():
                    raise ValueError("Capture destination must be an absolute nonsymlink path")
                if not preview.is_file() or preview.is_symlink():
                    raise ValueError("Preview has no frame available")
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                try:
                    from PIL import Image

                    output_format = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}.get(
                        target.suffix.lower()
                    )
                    if output_format is None:
                        raise ValueError("Capture destination must name a PNG or JPEG image")
                    with Image.open(preview) as captured:
                        captured.save(temporary, format=output_format)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                self.emit(f"runtime preview frame channel={channel} path={target}")
            elif command.startswith("inspect-"):
                self._inspect(command, arguments)
            else:
                raise ValueError(f"Unknown runtime command: {command}")
            return True
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            tokens = line.split()
            reason = str(error).replace("\n", " ").replace("\r", " ")
            if (
                tokens
                and tokens[0] in {"@seek", "@seek-relative"}
                and len(tokens) >= 3
                and tokens[2].isdigit()
            ):
                self.emit(
                    f"HSTREAM_SEEK status=failed generation={tokens[2]} reason={reason.replace(' ', '-')}"
                )
            elif tokens and tokens[0] == "@set-preview-active" and len(tokens) == 3:
                self.emit(
                    f"HSTREAM_PREVIEW channel={tokens[1]} status=failed generation={tokens[2]} message={reason}"
                )
            elif tokens and tokens[0] == "@set-preview-overlays" and len(tokens) == 5:
                self.emit(
                    f"HSTREAM_PREVIEW_OVERLAYS status=failed generation={tokens[1]} players={tokens[2]} play={tokens[3]} rink={tokens[4]} reason={reason.replace(' ', '-')}"
                )
            self.emit(f"runtime command failed: {line.split(' ', 1)[0]}: {reason}")
            return False

    def _preview_path(self, channel: str) -> Path:
        name = PREVIEW_NAMES.get(channel)
        if name is None or name == "none":
            raise ValueError("Unknown preview channel")
        slug = name.lower().replace(" ", "-")
        return self.session / f"preview-{slug}.jpg"

    def _inspect_emit(self, kind: str, request_id: int, **fields: Any) -> None:
        self.emit(self._inspect_message(kind, request_id, **fields))

    def _inspect_message(self, kind: str, request_id: int, **fields: Any) -> str:
        return "HSTREAM_PIPELINE_INSPECTOR " + json.dumps(
            {
                "version": 1,
                "kind": kind,
                "requestId": request_id,
                "stage": 0,
                "generation": self._generation,
                **fields,
            },
            separators=(",", ":"),
            allow_nan=False,
        )

    def _inspect(self, command: str, arguments: str) -> None:
        tokens = arguments.split()
        kind = {
            "inspect-pipeline": "graph",
            "inspect-properties": "properties",
            "inspect-set-property": "set-result",
        }.get(command)
        if kind is None or len(tokens) < 3:
            raise ValueError("Malformed inspector request")
        request_id = _integer(tokens[0], minimum=1)
        try:
            self._spec()
            if _integer(tokens[1]) != 0 or _integer(tokens[2]) != self._generation:
                raise ValueError("Stale pipeline inspector session")
            plugins = self.graph.get("plugins", self.graph.get("aspen", {}).get("plugins", {}))
            if not plugins:
                raise ValueError("Live Aspen graph is unavailable")
            if command == "inspect-pipeline":
                if len(tokens) != 3:
                    raise ValueError("Malformed graph request")
                nodes = []
                edges = []
                for name, definition in plugins.items():
                    nodes.append(
                        {
                            "id": name,
                            "appIndex": 0,
                            "path": name,
                            "parentId": "",
                            "name": name,
                            "factory": definition["class"],
                            "type": "Aspen trunk",
                            "state": "running" if definition.get("enabled", True) else "disabled",
                            "bin": False,
                        }
                    )
                    edges.extend(
                        {
                            "source": dependency,
                            "sourcePad": "context",
                            "sink": name,
                            "sinkPad": "context",
                        }
                        for dependency in definition.get("depends", [])
                    )
                self._inspect_emit(kind, request_id, status="ok", nodes=nodes, edges=edges)
                return
            if len(tokens) not in {5, 7} or _integer(tokens[3]) != 0:
                raise ValueError("Malformed property request")
            name = base64.b64decode(tokens[4], validate=True).decode("utf-8")
            if name not in plugins:
                raise ValueError("Unknown Aspen trunk")
            controls = {}
            if "PlayTracker" in plugins[name]["class"]:
                for window in self._spec()["windows"]:
                    for control in window["controls"]:
                        if control["name"] in {"Stitch_Rotate_Degrees", "Exposure_EV_x10"}:
                            continue
                        controls[f"runtime/{window['name']}/{control['name']}"] = (
                            window["name"],
                            control,
                        )
            if command == "inspect-set-property":
                if len(tokens) != 7:
                    raise ValueError("Malformed property assignment")
                property_name, value = [
                    base64.b64decode(token, validate=True).decode("utf-8") for token in tokens[5:]
                ]
                if property_name not in controls:
                    raise ValueError(
                        "Aspen constructor parameters are read-only; select a runtime property"
                    )
                window, control = controls[property_name]
                values = {}
                self._control(
                    values, control["name"], _integer(value, minimum=-(1 << 63)), window=window
                )
                self._queue(
                    {"kind": "controls", "values": values},
                    [
                        self._inspect_message(
                            kind,
                            request_id,
                            status="ok",
                            nodeId=name,
                            property=property_name,
                            value=value,
                        )
                    ],
                    lambda reason: self._inspect_message(
                        kind, request_id, status="error", message=reason
                    ),
                )
                return
            if len(tokens) != 5:
                raise ValueError("Malformed property request")
            properties = [
                {
                    "name": key,
                    "value": json.dumps(value, allow_nan=False),
                    "type": type(value).__name__,
                    "readable": True,
                    "writable": False,
                    "editable": False,
                    "description": "Resolved Aspen constructor parameter; applies on next start.",
                }
                for key, value in plugins[name].get("params", {}).items()
            ]
            properties.extend(
                {
                    "name": property_name,
                    "value": str(control["value"]),
                    "type": "int",
                    "kind": "integer",
                    "readable": True,
                    "writable": True,
                    "editable": True,
                    "minimum": control.get("min_value", 0),
                    "maximum": control["max_value"],
                    "description": control.get("description")
                    or "Live camera control; applied atomically at a frame boundary.",
                }
                for property_name, (_, control) in controls.items()
            )
            self._inspect_emit(kind, request_id, status="ok", nodeId=name, properties=properties)
        except (OSError, ValueError, TypeError, KeyError, binascii.Error) as error:
            self._inspect_emit(kind, request_id, status="error", message=str(error))

    def timeline_seconds(self, runtime: dict) -> float | None:
        """Map delivered frame IDs to the synchronized recording's time origin.

        The native dual-input loader retains the left synchronization offset in
        frame IDs. Its reported FPS already accounts for source frame stepping.
        Both desktop progress and the audio monitor must consume this clock.
        """
        frame, fps = runtime.get("frame"), runtime.get("fps")
        if frame is None or fps is None:
            return None
        rate = float(fps)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Runtime frame rate must be finite and positive")
        # The legacy loader labels a zero start as frame 1, but nonzero
        # starts retain their exact integer frame ID. Reset this bias on seeks.
        offset = int(self.frame_offset)
        zero_start_bias = int(offset == 0 and self.start_ns + self._child_offset_ns == 0)
        return max(0.0, (int(frame) - offset - zero_start_bias) / rate)

    def poll(self) -> None:
        """Drain receipts, then expose the last completed frame and preview."""
        now = time.monotonic()
        for request_id, pending in list(self._pending.items()):
            result_path = self.session / "runtime-results" / f"{request_id}.json"
            try:
                result = read_document(result_path)
            except FileNotFoundError:
                if now < pending.deadline:
                    continue
                result = {"ok": False, "error": "runtime acknowledgement timed out"}
            except (OSError, ValueError, TypeError) as error:
                result = {"ok": False, "error": str(error)}
            del self._pending[request_id]
            (self.session / "runtime-requests" / f"{request_id}.json").unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            if result.get("ok") is True and result.get("session_id") != self._session_id:
                result = {"ok": False, "error": "stale runtime acknowledgement"}
            if result.get("ok") is True:
                if pending.payload["kind"] == "preview":
                    self._preview_generation = pending.payload["generation"]
                    self._active_channel = pending.payload["channel"]
                    self._preview_ready_identity = None
                operation = pending.payload.get("operation", {})
                if operation.get("kind") == "preview-overlays":
                    self._overlay_layers = operation["layers"]
                values = pending.payload.get("values", operation.get("values", {}))
                for window, controls in values.items():
                    self._replay_values.setdefault(window, {}).update(controls)
                for message in pending.success:
                    self.emit(message)
            else:
                self.emit(pending.failure(str(result.get("error", "runtime request failed"))))
        try:
            runtime_path = self.session / "runtime.json"
            modified = runtime_path.stat().st_mtime_ns
            if modified != self._runtime_mtime:
                runtime = read_document(runtime_path)
                self._runtime_mtime = modified
                self._spec()
                fps = runtime.get("fps")
                timeline = self.timeline_seconds(runtime)
                if timeline is not None:
                    position = max(0, round(timeline * 1e9) - self.start_ns)
                    self._last_position_ns = position
                    speed = (
                        (position - self._clock_position_ns)
                        / max(now - self._clock_origin, 1e-6)
                        / 1e9
                    )
                    fields = {
                        "instance": "aggregate",
                        "stage": 0,
                        "instances": 1,
                        "generation": self._progress_generation,
                        "processed_ns": position,
                        "fps": fps,
                        "fps_avg": fps,
                        "speed_x": max(0, speed),
                    }
                    if self.total_ns:
                        remaining = max(0, self.total_ns - position)
                        fields.update(
                            total_ns=self.total_ns,
                            remaining_ns=remaining,
                            fraction=min(1, position / self.total_ns),
                            eta_ns=round(remaining / speed) if speed > 0 else "unknown",
                        )
                    self.emit(
                        "HSTREAM_PROGRESS "
                        + " ".join(f"{key}={value}" for key, value in fields.items())
                    )
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.emit(f"runtime status failed: {error}")
            return
        if self._active_channel == "none":
            # Qt needs transport readiness before it may activate a view. A
            # render-off launch has no image to announce, but its frame owner
            # and authenticated command mailbox are already available.
            identity = (self._session_id, self._preview_generation, "none")
            if self._session_id and identity != self._preview_ready_identity:
                self._preview_ready_identity = identity
                self.emit(
                    f"HSTREAM_PREVIEW_RUNTIME status=ready channel=none generation={self._preview_generation}"
                )
            return
        try:
            path = self._preview_path(self._active_channel)
            metadata = read_document(path.with_suffix(".json"))
            if not path.is_file():
                return
            identity = (self._session_id, self._preview_generation, self._active_channel)
            if identity != self._preview_ready_identity:
                self._preview_ready_identity = identity
                self.emit(
                    f"HSTREAM_PREVIEW_RUNTIME status=ready channel={self._active_channel} generation={self._preview_generation}"
                )
                self.emit(
                    f"HSTREAM_PREVIEW channel={self._active_channel} status=ready generation={self._preview_generation} message=frame-ready resolution={metadata['width']}x{metadata['height']}"
                )
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.emit(f"runtime preview failed: {error}")

    def reset_child(self, offset_ns: int = 0) -> None:
        """Invalidate old receipts before launching a replacement Python child."""
        for request_id, pending in self._pending.items():
            (self.session / "runtime-requests" / f"{request_id}.json").unlink(missing_ok=True)
            self.emit(pending.failure("pipeline was replaced before acknowledgement"))
        self._pending.clear()
        if self._scoreboard_session is not None:
            self._scoreboard_session.close()
            self._scoreboard_session = None
        self._session_id = ""
        self._runtime_mtime = None
        self._preview_ready_identity = None
        self._child_offset_ns = offset_ns
        self._clock_origin = time.monotonic()
        self._clock_position_ns = offset_ns

    def seek_completed(self, target_ns: int, generation: int) -> None:
        self.emit(f"HSTREAM_SEEK status=ok generation={generation} position_ns={target_ns}")

    def stopped(self, reason: str = "pipeline stopped") -> None:
        if self._scoreboard_session is not None:
            self._scoreboard_session.close()
            self._scoreboard_session = None
        for request_id, pending in self._pending.items():
            (self.session / "runtime-requests" / f"{request_id}.json").unlink(missing_ok=True)
            self.emit(pending.failure(reason))
        self._pending.clear()

    @property
    def replay_values(self) -> dict[str, dict[str, int]]:
        """Acknowledged camera values the supervisor must retain across seeking."""
        return {window: dict(values) for window, values in self._replay_values.items()}

    @property
    def replay_state(self) -> dict:
        """Acknowledged visual preferences retained across a seek replacement."""
        return {
            "values": self.replay_values,
            "preview": PREVIEW_NAMES[self._active_channel],
            "overlays": list(self._overlay_layers),
        }


def publish_precision_status(frame, emit=print, *, environment=None) -> None:
    """Publish the selected decoder's actual grading precision on its first frame."""
    environment = os.environ if environment is None else environment
    mode = environment["HM_SHARED_HIGH_BIT_DEPTH_MODE"]
    enabled = environment["HM_SHARED_HIGH_BIT_DEPTH"]
    depths = json.loads(environment["HM_SHARED_SOURCE_BIT_DEPTHS"])
    if mode not in {"auto", "0", "1"} or enabled not in {"0", "1"}:
        raise ValueError("Shared decoder precision settings are invalid")
    if (
        not isinstance(depths, list)
        or not depths
        or any(
            value is not None and (type(value) is not int or not 1 <= value <= 64)
            for value in depths
        )
    ):
        raise ValueError("Shared decoder source bit depths are invalid")
    if enabled == "1" and not frame.dtype.is_floating_point:
        raise ValueError("High-bit-depth decoder lost source precision before grading")
    known = [value for value in depths if value is not None]
    emit(
        f"HSTREAM_HIGH_BIT_DEPTH mode={mode} enabled={enabled} sources={len(depths)} "
        f"unknown={len(depths) - len(known)} minimum-source-bit-depth={min(known) if known else 0}"
    )
