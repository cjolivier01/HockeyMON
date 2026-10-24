"""Bridge between PlayTracker camera controls and the Rust hm-ui sidecar."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hmlib.log import logger
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hmlib.utils.image import image_height, image_width, make_visible_image, resize_image

# Bumped whenever the spec gains a field the sidecar has to understand. The
# sidecar echoes it into the state file, so a binary too old to render a control
# is reported rather than quietly drawing it as something else.
# 2: enumerated controls (`choices`) and per-control `description`.
_SPEC_VERSION = 3


@dataclass
class _Control:
    name: str
    max_value: int
    value: int
    default_value: int
    system_default_value: int
    group: str
    view: str
    value_revision: int
    # Labels for an enumerated control, indexed by value. Empty means a slider
    # (or a checkbox at max_value 1); the value stays an integer either way.
    choices: list[str] = field(default_factory=list)
    description: str = ""
    label: str = ""
    min_value: int = 0
    divisor: float | None = None
    offset: float = 0.0
    unit: str = ""


@dataclass
class _PreviewJob:
    img: Any
    show_scaled: float | None
    max_width: int
    metadata: dict | None = None
    overlays: tuple[bool, bool, bool] = (False, False, False)


@dataclass(frozen=True)
class HmUiAction:
    seq: int
    kind: str
    values: dict[str, dict[str, int]] | None


class RuntimeRollbackError(RuntimeError):
    """The frame owner could not restore state after a rejected runtime edit."""


class HmUiProcess:
    """Owns the Rust hm-ui process and its JSON spec/state files."""

    def __init__(
        self,
        *,
        title: str = "HM UI",
        tmpdir: Path | None = None,
        preview_names: Iterable[str] = ("Stitched", "Final"),
    ) -> None:
        self.title = title
        self.session_id = uuid.uuid4().hex
        attached_dir = os.environ.get("HM_UI_SESSION_DIR") if tmpdir is None else None
        self._attached = attached_dir is not None
        self._selected_preview_only = os.environ.get("HM_SHARED_UI_BACKEND") == "1"
        self._tmpdir = (
            Path(attached_dir)
            if attached_dir
            else Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="hm-ui-"))
        )
        self.spec_path = self._tmpdir / "spec.json"
        self.state_path = self._tmpdir / "state.json"
        self.action_ack_path = self._tmpdir / "action-ack.json"
        self.runtime_path = self._tmpdir / "runtime.json"
        self.paused = False
        self.stop_requested = False
        self._step_batches = 0
        self._paused_replay_pending = False
        self._runtime_frame: int | None = None
        self._runtime_fps: float | None = None
        self._runtime_enabled = False
        self._runtime_last_write = 0.0
        self._runtime_message = ""
        self.playback_rate = 1.0
        self._last_batch_time = None
        self._last_batch_frames = 1
        self._frame_clock_position = None
        self._frame_clock_wall = None
        normalized_preview_names = [
            str(name).strip() for name in preview_names if str(name).strip()
        ]
        if not normalized_preview_names:
            raise ValueError("hm-ui requires at least one preview name")
        self.preview_paths: dict[str, Path] = {
            name: self._tmpdir / f"preview-{self._slug(name)}.jpg"
            for name in normalized_preview_names
        }
        self._selected_preview_name = next(iter(self.preview_paths))
        # Compatibility for callers/tests that predate named preview streams.
        self.preview_path = next(iter(self.preview_paths.values()))
        self._windows: dict[str, list[_Control]] = {}
        self._process: subprocess.Popen | None = None
        self.stderr_path = self._tmpdir / "hm-ui.stderr.log"
        self._last_state_mtime_ns: int | None = None
        self._last_poll_values_changed = False
        self._last_preview_write_monotonic: dict[str, float] = {
            name: 0.0 for name in self.preview_paths
        }
        self._preview_condition = threading.Condition()
        self._pending_preview_jobs: OrderedDict[str, _PreviewJob] = OrderedDict()
        self._preview_worker: threading.Thread | None = None
        self._preview_worker_processing = False
        self._preview_worker_stop = False
        self._preview_batch_warned = False
        self._last_action_seq = 0
        self._pending_actions: list[HmUiAction] = []
        self._closed = False
        self._replay_controls_restored = False
        self._spec_version_warned = False
        requested_layers = set(os.environ.get("HSTREAM_UI_PREVIEW_OVERLAYS", "").split(","))
        self.preview_overlay_layers = tuple(
            name in requested_layers for name in ("players", "play", "rink")
        )
        self._preview_overlay_renderer = None

    def add_window(self, name: str) -> None:
        self._windows.setdefault(name, [])
        self._write_spec()
        self.ensure_started()

    def add_slider(
        self,
        window_name: str,
        name: str,
        max_value: int,
        initial_value: int,
        *,
        choices: Iterable[str] | None = None,
        description: str = "",
        label: str = "",
        min_value: int = 0,
        divisor: float | None = None,
        offset: float = 0.0,
        unit: str = "",
    ) -> None:
        """Declare an integer control, rendered as a combo box when `choices` is given."""
        labels = [str(choice) for choice in choices] if choices is not None else []
        if choices is not None:
            if len(labels) < 2:
                raise ValueError(f"hm-ui control {name} needs at least two choices")
            # The value is the choice index, so the range follows the label count.
            max_value = len(labels) - 1
        controls = self._windows.setdefault(window_name, [])
        view, group = self._control_location(window_name, name)
        if min_value != 0:
            raise ValueError("hm-ui bridge currently requires a zero raw minimum")
        if (divisor is not None and (not np.isfinite(divisor) or divisor == 0)) or not np.isfinite(
            offset
        ):
            raise ValueError("hm-ui numeric conversion must be finite with a nonzero divisor")
        for control in controls:
            if control.name == name:
                control.max_value = max(1, int(max_value))
                control.value = self._clamp(initial_value, control.max_value)
                control.default_value = control.value
                control.system_default_value = control.value
                control.group = group
                control.view = view
                control.choices = labels
                control.description = description
                control.label = label
                control.divisor = divisor
                control.offset = offset
                control.unit = unit
                control.value_revision += 1
                break
        else:
            max_i = max(1, int(max_value))
            value_i = self._clamp(initial_value, max_i)
            controls.append(
                _Control(
                    name=name,
                    max_value=max_i,
                    value=value_i,
                    default_value=value_i,
                    system_default_value=value_i,
                    group=group,
                    view=view,
                    value_revision=0,
                    choices=labels,
                    description=description,
                    label=label,
                    min_value=min_value,
                    divisor=divisor,
                    offset=offset,
                    unit=unit,
                )
            )
        self._write_spec()

    def set_system_defaults(self, defaults: dict[str, dict[str, int]]) -> None:
        """Publish system defaults separately from the values captured at UI open time."""
        for window_name, values in defaults.items():
            for control_name, value in values.items():
                try:
                    control = self._find_control(window_name, control_name)
                except KeyError:
                    logger.warning(
                        "Ignoring system default for unknown hm-ui control %s.%s",
                        window_name,
                        control_name,
                    )
                    continue
                control.system_default_value = self._clamp(value, control.max_value)
        self._write_spec()

    def get_value(self, window_name: str, control_name: str, *, poll: bool = True) -> int:
        if poll:
            self.poll()
        control = self._find_control(window_name, control_name)
        return control.value

    def set_value(
        self, window_name: str, control_name: str, value: int, *, notify: bool = True
    ) -> bool:
        control = self._find_control(window_name, control_name)
        new_value = self._clamp(value, control.max_value)
        if not notify:
            control.default_value = new_value
        if new_value == control.value:
            if not notify:
                if self._process is None:
                    self._write_state()
                self._write_spec()
            return False
        control.value = new_value
        control.value_revision += 1
        if self._process is None:
            self._write_state()
        self._write_spec()
        return bool(notify)

    def poll(self) -> bool:
        self._last_poll_values_changed = False
        self._poll_operator_actions()
        if self._process is not None and self._process.poll() is not None:
            logger.warning(
                "hm-ui exited with status %s; disabling Rust camera UI. stderr log: %s",
                self._process.returncode,
                self.stderr_path,
            )
            self._process = None
            self._closed = True
            return True
        if not self.state_path.exists():
            return False
        try:
            mtime_ns = self.state_path.stat().st_mtime_ns
        except OSError:
            return False
        if mtime_ns == self._last_state_mtime_ns:
            return False
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as ex:
            logger.warning("Failed to read hm-ui state: %s", ex)
            return False
        self._last_state_mtime_ns = mtime_ns
        self._check_spec_version(state)
        selected_preview = state.get("selected_preview")
        if (
            isinstance(selected_preview, str)
            and selected_preview in self.preview_paths
            and selected_preview != self._selected_preview_name
        ):
            self._selected_preview_name = selected_preview
            self._last_preview_write_monotonic[selected_preview] = 0.0
        changed = self._apply_state_values(state)
        self._last_poll_values_changed = changed
        actions = state.get("actions")
        if not isinstance(actions, list):
            actions = [state.get("last_action")]
        for action in actions:
            if not isinstance(action, dict):
                continue
            seq = int(action.get("seq") or 0)
            kind = str(action.get("kind") or "")
            if seq > self._last_action_seq and kind:
                self._last_action_seq = seq
                self._pending_actions.append(
                    HmUiAction(
                        seq=seq,
                        kind=kind,
                        values=self._normalize_control_values(action.get("windows")),
                    )
                )
                changed = True
        return changed

    def _apply_runtime_controls(self, values: dict, apply_controls: Callable[[], bool]) -> None:
        if not isinstance(values, dict):
            raise TypeError("Runtime control batch must be a mapping")
        for window, controls in values.items():
            if not isinstance(controls, dict):
                raise TypeError("Runtime window controls must be a mapping")
            for name, value in controls.items():
                control = self._find_control(window, name)
                if type(value) is not int or not control.min_value <= value <= control.max_value:
                    raise ValueError(f"Runtime control is outside its range: {name}")
        previous = self.control_values()
        try:
            self.apply_control_values(values, publish=True)
            if not apply_controls():
                raise ValueError("Runtime owner rejected the control batch")
            self._write_state()
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            try:
                self.apply_control_values(previous, publish=True)
                if not apply_controls():
                    raise RuntimeError("Runtime owner rejected restored controls")
                self._write_state()
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as rollback_error:
                raise RuntimeRollbackError(
                    "Runtime control rollback failed; pipeline cannot continue"
                ) from rollback_error
            raise

    def apply_runtime_requests(
        self,
        apply_controls: Callable[[], bool],
        operation: Callable[[dict], None] | None = None,
    ) -> None:
        """Validate and apply whole desktop requests on the frame-owning thread."""
        from hmlib.ui.hstream_runtime import read_document

        if not self._replay_controls_restored:
            replay = self._tmpdir / "replay-controls.json"
            try:
                saved = read_document(replay)
            except FileNotFoundError:
                saved = None
            if saved is not None:
                self._apply_runtime_controls(saved["values"], apply_controls)
                if "preview" in saved:
                    name = saved["preview"]
                    if name != "none" and name not in self.preview_paths:
                        raise ValueError("Restored preview is unavailable")
                    self._selected_preview_name = name
                if "overlays" in saved:
                    layers = saved["overlays"]
                    if (
                        not isinstance(layers, list)
                        or len(layers) != 3
                        or any(type(v) is not bool for v in layers)
                    ):
                        raise ValueError("Restored preview overlays require three booleans")
                    self.preview_overlay_layers = tuple(layers)
                if "paused" in saved:
                    if type(saved["paused"]) is not bool:
                        raise ValueError("Restored playback pause must be a boolean")
                    if saved["paused"]:
                        self.apply_playback_action("pause")
                        # A seek needs its first complete batch to publish the
                        # new preview. Keep the paused state/audio while that
                        # batch runs, then hold all subsequent source frames.
                        self._paused_replay_pending = True
                self._write_state()
                replay.unlink()
            self._replay_controls_restored = True
        directory = self._tmpdir / "runtime-requests"
        if not directory.is_dir():
            return
        pending = []
        for path in directory.glob("*.json"):
            try:
                pending.append((path.stat().st_mtime_ns, path))
            except FileNotFoundError:
                continue
        for _, path in sorted(pending)[:8]:
            result_path = self._tmpdir / "runtime-results" / path.name
            if result_path.exists():
                path.unlink(missing_ok=True)
                continue
            try:
                request = read_document(path)
                if request.get("session_id") != self.session_id:
                    raise ValueError("Runtime request belongs to a different pipeline session")
                deadline = request["deadline"]
                if (
                    not isinstance(deadline, (int, float))
                    or not math.isfinite(deadline)
                    or deadline < time.monotonic()
                ):
                    raise ValueError("Runtime request expired before application")
                kind = request["kind"]
                if kind == "controls":
                    self._apply_runtime_controls(request["values"], apply_controls)
                elif kind == "preview":
                    name = request["name"]
                    if name != "none" and name not in self.preview_paths:
                        raise ValueError(f"Preview source is unavailable: {name}")
                    self._selected_preview_name = name
                    self._write_state()
                elif kind == "clock-reset":
                    self._frame_clock_position = None
                    self._frame_clock_wall = None
                elif kind == "playback":
                    if not self.apply_playback_action(request["action"]):
                        raise ValueError("Unknown playback request")
                elif kind == "operation" and operation is not None:
                    operation(request["operation"])
                else:
                    raise ValueError(f"Unknown runtime request kind: {kind}")
                result = {"ok": True, "session_id": self.session_id}
            except RuntimeRollbackError:
                raise
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                result = {"ok": False, "error": str(error), "session_id": self.session_id}
                logger.warning("Shared desktop runtime request failed: %s", error)
            self._write_json_atomic(result_path, result)
            path.unlink(missing_ok=True)

    def _check_spec_version(self, state: dict) -> None:
        """Warn once when the sidecar does not understand the spec we wrote.

        An older binary ignores `choices` and renders an enumerated control as a
        bare 0..N slider with no labels, which an operator can easily read as a
        magnitude. It cannot be fixed from this side, but it can be reported.
        """
        if self._spec_version_warned:
            return
        if not any(control.choices for controls in self._windows.values() for control in controls):
            # Nothing published here needs a v2 sidecar, so an older one renders
            # this session identically and there is nothing to report.
            return
        try:
            reported = int(state.get("spec_version") or 0)
        except (TypeError, ValueError):
            reported = 0
        if reported >= _SPEC_VERSION:
            return
        self._spec_version_warned = True
        logger.warning(
            "hm-ui reports control-spec version %s but this build writes version %s; "
            "controls with named choices will render as an unlabelled slider, or as an "
            "on/off checkbox when there are exactly two choices. "
            "Set HM_UI_BIN, or rebuild hm-ui (bazelisk build //hm-ui:hm-ui) and make sure "
            "the rebuilt binary is the one on PATH.",
            reported,
            _SPEC_VERSION,
        )

    @property
    def last_poll_values_changed(self) -> bool:
        return self._last_poll_values_changed

    def consume_action_events(self, *, poll: bool = True) -> list[HmUiAction]:
        if poll:
            self.poll()
        return list(self._pending_actions)

    def acknowledge_action_events(self, through_seq: int, *, message: str = "") -> None:
        through_seq = int(through_seq)
        payload = {"seq": through_seq}
        if message:
            payload["message"] = message
        self._write_json_atomic(self.action_ack_path, payload)
        self._pending_actions = [
            action for action in self._pending_actions if action.seq > through_seq
        ]

    def report_action_error(self, error: str) -> None:
        """Expose a failed action without acknowledging it; the owner can retry."""
        self._write_json_atomic(self.action_ack_path, {"seq": 0, "error": error})

    def enable_playback(self) -> None:
        self._runtime_enabled = True
        self.publish_runtime(force=True)

    def apply_playback_action(self, kind: str) -> bool:
        if kind not in {"pause", "resume", "step", "stop"} and not kind.startswith("rate:"):
            return False
        if not self._runtime_enabled:
            raise ValueError("Playback controls are unavailable in this UI")
        self._last_batch_time = None
        self._frame_clock_position = None
        self._frame_clock_wall = None
        self._paused_replay_pending = False
        if kind.startswith("rate:"):
            rate = float(kind.split(":", 1)[1])
            if rate not in {0, 0.25, 0.5, 1, 2, 4}:
                raise ValueError("Playback speed must be 0.25, 0.5, 1, 2, 4 or 0 (maximum)")
            self.playback_rate = rate
        elif kind == "stop":
            self.stop_requested = True
            self.paused = False
            self._step_batches = 0
        elif not self.stop_requested:
            self.paused = kind != "resume"
            self._step_batches = self._step_batches + 1 if kind == "step" else 0
        self.publish_runtime(force=True)
        return True

    def _poll_operator_actions(self) -> None:
        """Supervisor requests have separate ingress from Rust's control state."""
        directory = self._tmpdir / "operator-actions"
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json"))[:8]:
            result_path = self._tmpdir / "operator-results" / path.name
            if result_path.exists():
                path.unlink()
                continue
            try:
                if path.is_symlink() or path.stat().st_size > 65536:
                    raise ValueError("Operator request must be a bounded regular JSON file")
                request = json.loads(path.read_text())
                if not self.apply_playback_action(request["kind"]):
                    raise ValueError(f"Unknown operator action: {request['kind']}")
                result = {"ok": True, "kind": request["kind"]}
            except (OSError, ValueError, TypeError, KeyError) as error:
                result = {"ok": False, "error": str(error)}
                logger.warning("Operator request %s failed: %s", path.name, error)
            self._write_json_atomic(result_path, result)
            path.unlink()

    def publish_runtime(
        self, *, frame: int | None = None, fps: float | None = None, force: bool = False
    ) -> None:
        if frame is not None:
            if self._runtime_frame is not None:
                self._last_batch_frames = max(1, int(frame) - self._runtime_frame)
            self._runtime_frame = int(frame)
        if fps is not None:
            self._runtime_fps = float(fps)
        now = time.monotonic()
        if not force and now - self._runtime_last_write < 0.25:
            return
        self._runtime_last_write = now
        self._write_json_atomic(
            self.runtime_path,
            {
                "state": (
                    "stopping" if self.stop_requested else "paused" if self.paused else "running"
                ),
                "frame": self._runtime_frame,
                "fps": self._runtime_fps,
                "rate": self.playback_rate,
                "seek_ready": not self._paused_replay_pending,
                "message": (
                    "Finishing queued batches" if self.stop_requested else self._runtime_message
                ),
                "updated_ms": int(time.time() * 1000),
                "capabilities": (
                    ["pause", "resume", "step", "stop", "rate"] if self._runtime_enabled else []
                ),
            },
        )

    def wait_for_shared_frame(
        self, frame: int, fps: float, poll_controls: Callable[[], None]
    ) -> None:
        """Pace source frames while servicing the frame owner's control mailbox."""
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Real-time playback requires a positive source frame rate")
        position = frame / fps
        while not self.stop_requested and not self.closed:
            poll_controls()
            if self._paused_replay_pending:
                break
            if self.paused and self._step_batches:
                self._step_batches -= 1
                break
            now = time.monotonic()
            if self._frame_clock_position is None:
                self._frame_clock_position = position
                self._frame_clock_wall = now
            delay = (
                self._frame_clock_wall
                + (position - self._frame_clock_position) / self.playback_rate
                - now
                if os.environ.get("HM_SHARED_RENDER_REALTIME") == "1" and self.playback_rate
                else 0
            )
            if not self.paused and delay <= 0:
                break
            self.publish_runtime()
            time.sleep(0.02 if self.paused else min(0.02, delay))
        self.publish_runtime(frame=frame, fps=fps)

    def complete_shared_batch(self, *, frame: int | None, fps: float | None) -> None:
        """Mark the replacement preview's first batch ready without resuming it."""
        replay_completed = self._paused_replay_pending
        self._paused_replay_pending = False
        self.publish_runtime(frame=frame, fps=fps, force=replay_completed)

    def wait_for_playback(self, poll_controls: Callable[[], None]) -> None:
        """Gate one batch on its owning worker while keeping Save/reset responsive."""
        poll_controls()
        while not self.stop_requested and not self.closed:
            if self.paused and self._step_batches:
                self._step_batches -= 1
                break
            delay = (
                self._last_batch_time
                + self._last_batch_frames / self._runtime_fps / self.playback_rate
                - time.monotonic()
                if self._last_batch_time is not None and self._runtime_fps and self.playback_rate
                else 0
            )
            if not self.paused and delay <= 0:
                break
            self.publish_runtime()
            time.sleep(0.05 if self.paused else min(0.05, delay))
            poll_controls()
        self._last_batch_time = time.monotonic()

    def consume_actions(self, *, poll: bool = True) -> list[str]:
        events = self.consume_action_events(poll=poll)
        if events:
            self.acknowledge_action_events(max(event.seq for event in events))
        return [event.kind for event in events]

    def publish_preview(
        self,
        img,
        *,
        name: str = "Stitched",
        show_scaled: float | None = None,
        max_width: int = 1280,
        min_interval_seconds: float | None = None,
        metadata: dict | list[dict] | None = None,
        snapshot: bool = False,
    ) -> None:
        if self._closed:
            return
        if name not in self.preview_paths:
            return
        if self._selected_preview_name == "none":
            return
        if self._selected_preview_only and name != self._selected_preview_name:
            return
        if min_interval_seconds is None:
            min_interval_seconds = 1.0 / 15.0 if name == self._selected_preview_name else 1.0
        else:
            min_interval_seconds = max(0.0, float(min_interval_seconds))
        now = time.monotonic()
        with self._preview_condition:
            if self._preview_worker_stop:
                return
            if now - self._last_preview_write_monotonic[name] < min_interval_seconds:
                return
            if snapshot:
                # Source tabs must retain decoded pixels before the processing
                # graph can grade/draw on them. Own only a bounded frame, and
                # keep tensor resizing on its GPU before the worker readback.
                frame = unwrap_tensor(img)
                if frame.ndim == 4:
                    frame = frame[-1]
                width = image_width(frame)
                if max_width > 0 and width > max_width:
                    height = max(1, round(image_height(frame) * max_width / width))
                    frame = (
                        cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
                        if isinstance(frame, np.ndarray)
                        else resize_image(frame, new_width=max_width, new_height=height)
                    )
                else:
                    frame = frame.copy() if isinstance(frame, np.ndarray) else frame.clone()
                img = frame if isinstance(frame, np.ndarray) else wrap_tensor(frame)
            # Keep one pending frame per stream. Replacing a queued frame makes
            # preview publication lossy and bounded instead of slowing tracking.
            self._pending_preview_jobs[name] = _PreviewJob(
                img=img,
                show_scaled=show_scaled,
                max_width=int(max_width),
                metadata=(metadata[-1] if isinstance(metadata, list) and metadata else metadata),
                overlays=self.preview_overlay_layers,
            )
            self._last_preview_write_monotonic[name] = now
            if self._preview_worker is None:
                self._preview_worker = threading.Thread(
                    target=self._preview_worker_main,
                    name="hm-ui-preview",
                    daemon=True,
                )
                self._preview_worker.start()
            self._preview_condition.notify()

    def flush_previews(self, timeout: float = 5.0) -> bool:
        """Wait for queued preview work, primarily for orderly shutdown and tests."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._preview_condition:
            while self._pending_preview_jobs or self._preview_worker_processing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._preview_condition.wait(timeout=remaining)
        return True

    def _preview_worker_main(self) -> None:
        while True:
            with self._preview_condition:
                while not self._pending_preview_jobs and not self._preview_worker_stop:
                    self._preview_condition.wait()
                if self._preview_worker_stop:
                    self._pending_preview_jobs.clear()
                    self._preview_condition.notify_all()
                    return
                name, job = self._pending_preview_jobs.popitem(last=False)
                self._preview_worker_processing = True
            try:
                self._encode_preview(name, job)
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                AssertionError,
                cv2.error,
            ) as ex:
                logger.warning("Failed to encode hm-ui %s preview frame: %s", name, ex)
            finally:
                with self._preview_condition:
                    self._preview_worker_processing = False
                    if not self._pending_preview_jobs:
                        self._preview_condition.notify_all()

    def _encode_preview(self, name: str, job: _PreviewJob) -> None:
        frame = unwrap_tensor(job.img)
        if frame.ndim == 4:
            if frame.shape[0] > 1 and not self._preview_batch_warned:
                logger.warning(
                    "hm-ui preview received a batch with %s frames; publishing the latest frame",
                    frame.shape[0],
                )
                self._preview_batch_warned = True
            frame = frame[-1]
        source_width = int(image_width(frame))
        source_height = int(image_height(frame))
        # Resize before a CUDA-to-host transfer when the source is a GPU tensor.
        frame = make_visible_image(
            frame,
            enable_resizing=job.show_scaled,
            force_numpy=False,
        )
        if job.max_width > 0 and image_width(frame) > job.max_width:
            scale = float(job.max_width) / float(image_width(frame))
            new_height = max(1, round(frame.shape[-3] * scale))
            if isinstance(frame, np.ndarray):
                frame = cv2.resize(
                    frame,
                    (job.max_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                frame = resize_image(
                    frame,
                    new_width=job.max_width,
                    new_height=new_height,
                )
        frame = make_visible_image(frame, force_numpy=True)
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            raise ValueError(f"hm-ui preview expected HxWxC image, got shape={frame.shape}")
        frame = np.ascontiguousarray(frame)
        if job.metadata is not None and any(job.overlays):
            if self._preview_overlay_renderer is None:
                from hmlib.ui.preview_overlays import PreviewOverlayRenderer

                self._preview_overlay_renderer = PreviewOverlayRenderer()
            frame = self._preview_overlay_renderer.draw(
                frame, job.metadata, (source_width, source_height), job.overlays
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode hm-ui preview frame")
        preview_path = self.preview_paths[name]
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = preview_path.with_suffix(preview_path.suffix + ".tmp")
        tmp.write_bytes(encoded.tobytes())
        os.replace(tmp, preview_path)
        # Publish separately from the control spec: the preview worker must not
        # race the tracking thread's control/default updates to spec.json.
        self._write_json_atomic(
            preview_path.with_suffix(".json"),
            {"width": source_width, "height": source_height},
        )

    def ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("hm-ui was closed")
        if self._attached:
            return
        if self._process is not None and self._process.poll() is None:
            return
        cmd = self._resolve_command()
        if cmd is None:
            raise RuntimeError(
                "hm-ui binary not found. Build it with `bazelisk build //hm-ui:hm-ui` "
                "or `cargo build --manifest-path hm-ui/Cargo.toml`, or set HM_UI_BIN=/path/to/hm-ui."
            )
        self._write_spec()
        self._write_state()
        full_cmd = [
            *cmd,
            "--spec",
            str(self.spec_path),
            "--state",
            str(self.state_path),
            "--title",
            self.title,
        ]
        self._process = subprocess.Popen(
            full_cmd,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr_path.open("ab"),
        )
        time.sleep(0.05)
        if self._process.poll() is not None:
            stderr_tail = self._read_stderr_tail()
            raise RuntimeError(
                f"hm-ui exited during startup with status {self._process.returncode}. "
                f"stderr log: {self.stderr_path}. {stderr_tail}"
            )
        self._last_action_seq = 0

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        with self._preview_condition:
            self._preview_worker_stop = True
            self._pending_preview_jobs.clear()
            self._preview_condition.notify_all()
        if self._preview_worker is not None:
            self._preview_worker.join(timeout=5.0)
        if self._preview_worker is not None and self._preview_worker.is_alive():
            logger.error(
                "hm-ui preview worker did not stop; preserving temporary files at %s",
                self._tmpdir,
            )
        elif not self._attached:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        if self._attached:
            self._write_json_atomic(
                self.runtime_path,
                {
                    "state": "finished",
                    "frame": self._runtime_frame,
                    "fps": self._runtime_fps,
                    "message": "Run finished",
                    "updated_ms": int(time.time() * 1000),
                    "capabilities": [],
                },
            )
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def control_values(self) -> dict[str, dict[str, int]]:
        return {
            window_name: {control.name: control.value for control in controls}
            for window_name, controls in self._windows.items()
        }

    def apply_control_values(
        self,
        values: dict[str, dict[str, int]],
        *,
        publish: bool = False,
    ) -> bool:
        normalized = self._normalize_control_values(values)
        if normalized is None:
            return False
        changed = self._apply_normalized_control_values(normalized)
        if publish:
            for window_name, controls in normalized.items():
                for control_name in controls:
                    self._find_control(window_name, control_name).value_revision += 1
            self._write_spec()
        return changed

    def _normalize_control_values(self, windows: Any) -> dict[str, dict[str, int]] | None:
        if not isinstance(windows, dict):
            return None
        normalized: dict[str, dict[str, int]] = {}
        for window_name, controls in self._windows.items():
            raw_values = windows.get(window_name)
            if not isinstance(raw_values, dict):
                continue
            values: dict[str, int] = {}
            for control in controls:
                if control.name not in raw_values:
                    continue
                try:
                    values[control.name] = self._clamp(
                        int(raw_values[control.name]), control.max_value
                    )
                except (TypeError, ValueError):
                    continue
            normalized[window_name] = values
        return normalized

    def _normalize_control_revisions(self, revisions: Any) -> dict[str, dict[str, int]] | None:
        if not isinstance(revisions, dict):
            return None
        normalized: dict[str, dict[str, int]] = {}
        for window_name, controls in self._windows.items():
            raw_revisions = revisions.get(window_name)
            if not isinstance(raw_revisions, dict):
                normalized[window_name] = {}
                continue
            values: dict[str, int] = {}
            for control in controls:
                if control.name not in raw_revisions:
                    continue
                try:
                    values[control.name] = max(0, int(raw_revisions[control.name]))
                except (TypeError, ValueError):
                    continue
            normalized[window_name] = values
        return normalized

    def _apply_state_values(self, state: dict) -> bool:
        windows = self._normalize_control_values(state.get("windows"))
        if windows is None:
            return False
        revisions = self._normalize_control_revisions(state.get("control_revisions"))
        return self._apply_normalized_control_values(windows, revisions=revisions)

    def _apply_normalized_control_values(
        self,
        windows: dict[str, dict[str, int]],
        *,
        revisions: dict[str, dict[str, int]] | None = None,
    ) -> bool:
        changed = False
        for window_name, values in windows.items():
            controls = self._windows.get(window_name, [])
            by_name = {control.name: control for control in controls}
            for control_name, new_value in values.items():
                control = by_name.get(control_name)
                if control is None:
                    continue
                if revisions is not None:
                    revision = revisions.get(window_name, {}).get(control_name)
                    if revision is None or revision < control.value_revision:
                        continue
                    control.value_revision = revision
                if new_value != control.value:
                    control.value = new_value
                    changed = True
        return changed

    def _write_spec(self) -> None:
        payload = {
            "version": _SPEC_VERSION,
            "session_id": self.session_id,
            "title": self.title,
            "subtitle": "Runtime tracking, stitch, and camera controls",
            "preview_path": str(self.preview_path),
            "action_ack_path": str(self.action_ack_path),
            "runtime_path": str(self.runtime_path),
            "previews": [
                {
                    "name": name,
                    "path": str(path),
                    "metadata_path": str(path.with_suffix(".json")),
                }
                for name, path in self.preview_paths.items()
            ],
            "windows": [
                {
                    "name": window_name,
                    "controls": [
                        {
                            "name": control.name,
                            "max_value": control.max_value,
                            "value": control.value,
                            "default_value": control.default_value,
                            "system_default_value": control.system_default_value,
                            "group": control.group,
                            "view": control.view,
                            "value_revision": control.value_revision,
                            "choices": list(control.choices),
                            "description": control.description,
                            "label": control.label,
                            "min_value": control.min_value,
                            "divisor": control.divisor,
                            "offset": control.offset,
                            "unit": control.unit,
                        }
                        for control in controls
                    ],
                }
                for window_name, controls in self._windows.items()
            ],
        }
        self._write_json_atomic(self.spec_path, payload)

    def _write_state(self) -> None:
        payload = {
            "version": 1,
            # This file is the bootstrap the sidecar reads before it writes its
            # own; claiming the version we wrote keeps the compatibility check
            # from firing on our own output before the sidecar has started.
            "spec_version": _SPEC_VERSION,
            "updated_ms": int(time.time() * 1000),
            "windows": {
                window_name: {control.name: control.value for control in controls}
                for window_name, controls in self._windows.items()
            },
            "control_revisions": {
                window_name: {control.name: control.value_revision for control in controls}
                for window_name, controls in self._windows.items()
            },
            "selected_preview": self._selected_preview_name,
            "actions": [],
            "last_action": None,
        }
        self._write_json_atomic(self.state_path, payload)

    def _read_stderr_tail(self, max_chars: int = 1200) -> str:
        try:
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        text = text.strip()
        if not text:
            return ""
        return text[-max_chars:]

    def _find_control(self, window_name: str, control_name: str) -> _Control:
        for control in self._windows[window_name]:
            if control.name == control_name:
                return control
        raise KeyError(control_name)

    @staticmethod
    def _control_location(window_name: str, control_name: str) -> tuple[str, str]:
        """Keep controls that affect different rendered views in separate tabs."""
        lower_window = window_name.lower()
        if "scoreboard" in lower_window:
            return "Final", "Scoreboard"
        if "overlay" in lower_window:
            return "Final", "Overlays"
        if "left" in lower_window and "color" in lower_window:
            return "Stitched", "Left Input Color"
        if "right" in lower_window and "color" in lower_window:
            return "Stitched", "Right Input Color"
        if "color" in lower_window:
            view = "Stitched" if "stitched" in lower_window else "Final"
            return view, f"{view} Color"
        if "blend" in lower_window:
            return "Stitched", "Seam Blend"
        if control_name == "Stitch_Rotate_Degrees" or "stitch" in lower_window:
            return "Stitched", "Alignment"
        if control_name.startswith(("Overshoot_", "Post_Nonstop_")):
            return "Final", "Breakaway Tracking"
        if "Fixed_Edge_Rotation" in control_name:
            return "Final", "Perspective Rotation"
        if control_name.startswith(("Max_Speed_", "Max_Accel_", "Apply_To_")):
            return "Final", "Motion Limits"
        return "Final", "Play Tracking"

    @staticmethod
    def _slug(value: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
        return slug or "view"

    @staticmethod
    def _clamp(value: int, max_value: int) -> int:
        return max(0, min(int(max_value), int(value)))

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def _resolve_command() -> list[str] | None:
        env_bin = os.environ.get("HM_UI_BIN")
        if env_bin:
            return [env_bin]
        exe = shutil.which("hm-ui")
        if exe:
            return [exe]
        repo_root = Path(__file__).resolve().parents[2]
        hmlib_root = Path(__file__).resolve().parents[1]
        runfiles_dir = os.environ.get("RUNFILES_DIR")
        if runfiles_dir:
            for candidate in (
                Path(runfiles_dir) / "hockeymon" / "hmlib" / "bin" / "hm-ui",
                Path(runfiles_dir) / "hmlib" / "bin" / "hm-ui",
                Path(runfiles_dir) / "hockeymon" / "hm-ui" / "hm-ui-bin",
                Path(runfiles_dir) / "hm-ui" / "hm-ui-bin",
            ):
                if candidate.exists() and os.access(candidate, os.X_OK):
                    return [str(candidate)]
        for candidate in (
            hmlib_root / "bin" / "hm-ui",
            repo_root / "bazel-bin" / "hmlib" / "bin" / "hm-ui",
            repo_root / "bazel-bin" / "hm-ui" / "hm-ui-bin",
            repo_root / "hm-ui" / "target" / "release" / "hm-ui",
            repo_root / "hm-ui" / "target" / "debug" / "hm-ui",
        ):
            if candidate.exists() and os.access(candidate, os.X_OK):
                return [str(candidate)]
        if os.environ.get("HM_UI_ALLOW_CARGO_RUN") == "1" and shutil.which("cargo"):
            return [
                "cargo",
                "run",
                "--locked",
                "--manifest-path",
                str(repo_root / "hm-ui" / "Cargo.toml"),
                "--",
            ]
        maybe_bazel_runfile = Path(sys.argv[0]).resolve().parent / "hm-ui" / "hm-ui-bin"
        if maybe_bazel_runfile.exists() and os.access(maybe_bazel_runfile, os.X_OK):
            return [str(maybe_bazel_runfile)]
        return None


class HmUiDialog:
    """Handle for one control group backed by the hm-ui sidecar."""

    def __init__(
        self,
        manager: HmUiProcess,
        window_name: str,
        *,
        on_change: Callable[[int], None] | None = None,
        initial_size: tuple[int, int] = (900, 640),
        position: tuple[int, int] | None = None,
    ) -> None:
        del initial_size, position
        self.window_name = window_name
        self._manager = manager
        self._on_change = on_change

    def open(self) -> None:
        self._manager.add_window(self.window_name)

    def add_slider(self, name: str, max_value: int, initial_value: int) -> None:
        self._manager.add_slider(self.window_name, name, max_value, initial_value)

    def get_value(self, name: str) -> int:
        return self._manager.get_value(self.window_name, name, poll=False)

    def set_value(self, name: str, value: int, *, notify: bool = True) -> None:
        changed = self._manager.set_value(self.window_name, name, value, notify=notify)
        if changed and self._on_change is not None:
            self._on_change(value)

    def show(self) -> None:
        self._manager.poll()
        if self._manager.last_poll_values_changed and self._on_change is not None:
            self._on_change(0)
        if self._manager.closed:
            raise RuntimeError("hm-ui was closed")

    def consume_actions(self) -> list[str]:
        return self._manager.consume_actions()
