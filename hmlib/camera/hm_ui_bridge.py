"""Bridge between PlayTracker camera controls and the Rust hm-ui sidecar."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from hmlib.log import logger
from hmlib.utils.gpu import unwrap_tensor
from hmlib.utils.image import image_width, make_visible_image, resize_image


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


@dataclass
class _PreviewJob:
    img: Any
    show_scaled: Optional[float]
    max_width: int


@dataclass(frozen=True)
class HmUiAction:
    seq: int
    kind: str
    values: Optional[Dict[str, Dict[str, int]]]


class HmUiProcess:
    """Owns the Rust hm-ui process and its JSON spec/state files."""

    def __init__(
        self,
        *,
        title: str = "HM UI",
        tmpdir: Optional[Path] = None,
        preview_names: Iterable[str] = ("Stitched", "Final"),
    ) -> None:
        self.title = title
        self._tmpdir = (
            Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="hm-ui-"))
        )
        self.spec_path = self._tmpdir / "spec.json"
        self.state_path = self._tmpdir / "state.json"
        self.action_ack_path = self._tmpdir / "action-ack.json"
        normalized_preview_names = [
            str(name).strip() for name in preview_names if str(name).strip()
        ]
        if not normalized_preview_names:
            raise ValueError("hm-ui requires at least one preview name")
        self.preview_paths: Dict[str, Path] = {
            name: self._tmpdir / f"preview-{self._slug(name)}.jpg"
            for name in normalized_preview_names
        }
        self._selected_preview_name = next(iter(self.preview_paths))
        # Compatibility for callers/tests that predate named preview streams.
        self.preview_path = next(iter(self.preview_paths.values()))
        self._windows: Dict[str, List[_Control]] = {}
        self._process: Optional[subprocess.Popen] = None
        self.stderr_path = self._tmpdir / "hm-ui.stderr.log"
        self._last_state_mtime_ns: Optional[int] = None
        self._last_poll_values_changed = False
        self._last_preview_write_monotonic: Dict[str, float] = {
            name: 0.0 for name in self.preview_paths
        }
        self._preview_condition = threading.Condition()
        self._pending_preview_jobs: OrderedDict[str, _PreviewJob] = OrderedDict()
        self._preview_worker: Optional[threading.Thread] = None
        self._preview_worker_processing = False
        self._preview_worker_stop = False
        self._preview_batch_warned = False
        self._last_action_seq = 0
        self._pending_actions: List[HmUiAction] = []
        self._closed = False

    def add_window(self, name: str) -> None:
        self._windows.setdefault(name, [])
        self._write_spec()
        self.ensure_started()

    def add_slider(self, window_name: str, name: str, max_value: int, initial_value: int) -> None:
        controls = self._windows.setdefault(window_name, [])
        view, group = self._control_location(window_name, name)
        for control in controls:
            if control.name == name:
                control.max_value = max(1, int(max_value))
                control.value = self._clamp(initial_value, control.max_value)
                control.default_value = control.value
                control.system_default_value = control.value
                control.group = group
                control.view = view
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
                )
            )
        self._write_spec()

    def set_system_defaults(self, defaults: Dict[str, Dict[str, int]]) -> None:
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
        if notify:
            return True
        return False

    def poll(self) -> bool:
        self._last_poll_values_changed = False
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

    @property
    def last_poll_values_changed(self) -> bool:
        return self._last_poll_values_changed

    def consume_action_events(self, *, poll: bool = True) -> List[HmUiAction]:
        if poll:
            self.poll()
        return list(self._pending_actions)

    def acknowledge_action_events(self, through_seq: int) -> None:
        through_seq = int(through_seq)
        self._write_json_atomic(self.action_ack_path, {"seq": through_seq})
        self._pending_actions = [
            action for action in self._pending_actions if action.seq > through_seq
        ]

    def consume_actions(self, *, poll: bool = True) -> List[str]:
        events = self.consume_action_events(poll=poll)
        if events:
            self.acknowledge_action_events(max(event.seq for event in events))
        return [event.kind for event in events]

    def publish_preview(
        self,
        img,
        *,
        name: str = "Stitched",
        show_scaled: Optional[float] = None,
        max_width: int = 1280,
        min_interval_seconds: Optional[float] = None,
    ) -> None:
        if self._closed:
            return
        if name not in self.preview_paths:
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
            # Keep one pending frame per stream. Replacing a queued frame makes
            # preview publication lossy and bounded instead of slowing tracking.
            self._pending_preview_jobs[name] = _PreviewJob(
                img=img,
                show_scaled=show_scaled,
                max_width=int(max_width),
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
        # Resize before a CUDA-to-host transfer when the source is a GPU tensor.
        frame = make_visible_image(
            frame,
            enable_resizing=job.show_scaled,
            force_numpy=False,
        )
        if job.max_width > 0 and image_width(frame) > job.max_width:
            scale = float(job.max_width) / float(image_width(frame))
            new_height = max(1, int(round(frame.shape[-3] * scale)))
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

    def ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("hm-ui was closed")
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
        else:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def control_values(self) -> Dict[str, Dict[str, int]]:
        return {
            window_name: {control.name: control.value for control in controls}
            for window_name, controls in self._windows.items()
        }

    def apply_control_values(
        self,
        values: Dict[str, Dict[str, int]],
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

    def _normalize_control_values(self, windows: Any) -> Optional[Dict[str, Dict[str, int]]]:
        if not isinstance(windows, dict):
            return None
        normalized: Dict[str, Dict[str, int]] = {}
        for window_name, controls in self._windows.items():
            raw_values = windows.get(window_name)
            if not isinstance(raw_values, dict):
                continue
            values: Dict[str, int] = {}
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

    def _normalize_control_revisions(self, revisions: Any) -> Optional[Dict[str, Dict[str, int]]]:
        if not isinstance(revisions, dict):
            return None
        normalized: Dict[str, Dict[str, int]] = {}
        for window_name, controls in self._windows.items():
            raw_revisions = revisions.get(window_name)
            if not isinstance(raw_revisions, dict):
                normalized[window_name] = {}
                continue
            values: Dict[str, int] = {}
            for control in controls:
                if control.name not in raw_revisions:
                    continue
                try:
                    values[control.name] = max(0, int(raw_revisions[control.name]))
                except (TypeError, ValueError):
                    continue
            normalized[window_name] = values
        return normalized

    def _apply_state_values(self, state: Dict) -> bool:
        windows = self._normalize_control_values(state.get("windows"))
        if windows is None:
            return False
        revisions = self._normalize_control_revisions(state.get("control_revisions"))
        return self._apply_normalized_control_values(windows, revisions=revisions)

    def _apply_normalized_control_values(
        self,
        windows: Dict[str, Dict[str, int]],
        *,
        revisions: Optional[Dict[str, Dict[str, int]]] = None,
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
            "version": 1,
            "title": self.title,
            "subtitle": "Runtime tracking, stitch, and camera controls",
            "preview_path": str(self.preview_path),
            "action_ack_path": str(self.action_ack_path),
            "previews": [
                {"name": name, "path": str(path)} for name, path in self.preview_paths.items()
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
    def _control_location(window_name: str, control_name: str) -> Tuple[str, str]:
        """Keep controls that affect different rendered views in separate tabs."""
        lower_window = window_name.lower()
        if "left" in lower_window and "color" in lower_window:
            return "Stitched", "Left Input Color"
        if "right" in lower_window and "color" in lower_window:
            return "Stitched", "Right Input Color"
        if "color" in lower_window:
            view = "Stitched" if "stitched" in lower_window else "Final"
            return view, f"{view} Color"
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
    def _write_json_atomic(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def _resolve_command() -> Optional[List[str]]:
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
        on_change: Optional[Callable[[int], None]] = None,
        initial_size: Tuple[int, int] = (900, 640),
        position: Optional[Tuple[int, int]] = None,
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

    def consume_actions(self) -> List[str]:
        return self._manager.consume_actions()
