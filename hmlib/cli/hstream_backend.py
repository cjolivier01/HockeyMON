"""HockeyMON process adapter for the shared HStream desktop protocol."""

from __future__ import annotations

import errno
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path


def emit(message: str) -> None:
    # Child progress renderers can finish with an ANSI cursor sequence without
    # a newline. Protocol records must still start at a fresh logical line.
    print("\n" + message, flush=True)


def _media_info(paths: list[Path]) -> tuple[float, float, bool, list[int | None], list[float]]:
    from fractions import Fraction

    duration, fps, audio = 0.0, 0.0, False
    depths = []
    durations = []
    for path in paths:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        info = json.loads(result.stdout)
        video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
        rate = float(Fraction(video["avg_frame_rate"]))
        if rate <= 0:
            raise ValueError(f"Invalid source frame rate: {path}")
        if fps and abs(fps - rate) > 0.001:
            raise ValueError("Chapter frame rates must agree")
        fps = rate
        durations.append(float(info["format"]["duration"]))
        duration += durations[-1]
        audio |= any(stream["codec_type"] == "audio" for stream in info["streams"])
        depth = int(video.get("bits_per_raw_sample", 0) or 0)
        pixel_format = video.get("pix_fmt", "")
        if not depth:
            match = re.search(r"(?:p|gray)(9|10|12|14|16)(?:le|be)?$", pixel_format)
            if match:
                depth = int(match.group(1))
            elif pixel_format in {
                "yuv420p",
                "yuv422p",
                "yuv444p",
                "yuvj420p",
                "nv12",
                "bgr24",
                "rgb24",
            }:
                depth = 8
        depths.append(depth or None)
    return duration, fps, audio, depths, durations


def _native_max_output_width(config: dict) -> int:
    from hmlib.ui.hstream_launch import get_path

    # The complete effective config is a native settings overlay. Its canonical
    # key takes precedence over legacy property aliases at the same layer.
    value = get_path(config, "stitching.max_output_width", None)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stitching.max_output_width must be null or a nonnegative integer")
    return value


class Supervisor:
    """Keep native preparation and Python processing within the UI-owned group."""

    def __init__(self, args, arguments: list[str], session: Path, *, web=None) -> None:
        from hmlib.ui.hstream_launch import make_plan
        from hmlib.ui.hstream_runtime import RuntimeProtocol, shared_playback_fps

        self.args, self.arguments, self.session = args, arguments, session
        self.web = web
        self.web_cancelled = False
        launch = os.environ.get("HM_UI_INITIAL_LAUNCH")
        self.initial = Path(launch) if launch else None
        self.plan = make_plan(args, initial=self.initial)
        if self.web is not None:
            self.web.validate_plan(self.plan)
            self.web.write_admission()
        self.environment = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                [str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")]
            ),
            HM_UI_SESSION_DIR=str(session),
            HSTREAM_UI_BACKEND_SESSION=str(session),
            HM_SHARED_UI_BACKEND="1",
            HM_SHARED_RENDER_REALTIME="1" if args.render_realtime else "0",
            HM_BACKEND_PUBLICATION_FILE=str(session / "publication.json"),
        )
        output_root = Path(
            self.environment.get("HM_OUTPUT_WORK_DIR", str(Path.home() / "hstream_output"))
        ).expanduser()
        if not output_root.is_absolute():
            output_root = self.plan.cwd / output_root
        runs = output_root.resolve() / self.plan.game.name / "runs"
        self.environment["HM_OUTPUT_WORK_DIR"] = str(output_root.resolve())
        runs.mkdir(parents=True, exist_ok=True)
        self.working_directory = Path(tempfile.mkdtemp(prefix="hockeymon-", dir=runs))
        if self.web is not None:
            self.web.working_directory = self.working_directory
        identity = self.working_directory.stat()
        self._working_identity = (identity.st_dev, identity.st_ino)
        self.worker_cwd = self.plan.cwd if self.initial else self.working_directory
        self.environment["HM_SHARED_WORK_DIR"] = str(self.working_directory)
        source_info = [_media_info(paths) for paths in self.plan.sources.values()]
        self.source_info = dict(zip(self.plan.sources, source_info))
        from hmlib.ui.hstream_launch import get_path, set_path

        depths = [depth for info in source_info for depth in info[3]]
        high_bit = str(
            get_path(self.plan.config, "pipeline.hmstitcher.properties.high-bit-depth", "auto")
        ).lower()
        high_bit = {"true": "1", "false": "0"}.get(high_bit, high_bit)
        enabled = high_bit == "1" or (
            high_bit == "auto" and all(depth is not None and depth > 8 for depth in depths)
        )
        self.environment.update(
            HM_SHARED_HIGH_BIT_DEPTH_MODE=high_bit,
            HM_SHARED_HIGH_BIT_DEPTH="1" if enabled else "0",
            HM_SHARED_SOURCE_BIT_DEPTHS=json.dumps(depths),
        )
        if enabled:
            set_path(self.plan.config, "stitching.dtype", "float32")
            emit(
                "HSTREAM_STARTUP stage=decoder message=High-bit-depth grading uses FFmpeg BGR48 software decode and floating processing"
            )
        self.fps = (
            shared_playback_fps(self.source_info["left"][1], self.source_info["right"][1])
            if set(self.plan.sources) == {"left", "right"}
            else source_info[0][1]
        )
        if self.plan.start_frame:
            self.plan.start_seconds = self.plan.start_frame / self.fps
        self.duration = max(0.0, min(info[0] for info in source_info) - self.plan.start_seconds)
        if self.plan.limit_seconds is not None:
            self.duration = min(self.duration, self.plan.limit_seconds)
        if self.plan.limit_frames is not None:
            self.duration = min(self.duration, self.plan.limit_frames / self.fps)
        self.plan.limit_seconds = self.duration
        self.environment["HM_SHARED_TELEMETRY"] = (
            "1" if self.plan.config["hm_shared_ui"]["telemetry"] else "0"
        )
        self.environment.pop("HM_SHARED_SOURCE_SCOREBOARD_POLYGON", None)
        if set(self.plan.sources) == {"source"}:
            self.environment["HM_SHARED_SOURCE_CHAPTERS"] = json.dumps(
                [str(path) for path in self.plan.sources["source"]]
            )
            # Initial source-specific corners are explicit operator intent,
            # even when their coordinates equal a saved pair's old selection.
            scoreboard = self.plan.initial_settings.get("rink", {}).get("scoreboard", {})
            if "perspective_polygon" in scoreboard:
                self.environment["HM_SHARED_SOURCE_SCOREBOARD_POLYGON"] = json.dumps(
                    scoreboard["perspective_polygon"], allow_nan=False
                )
        protocol_type = RuntimeProtocol
        protocol_options = {}
        if self.web is not None:
            from hmlib.ui.hstream_web_runtime import WebRuntimeProtocol

            protocol_type = WebRuntimeProtocol
            protocol_options = {
                "allow_seek": not (self.plan.sinks - {"RENDER", "FAKE"})
                and self.environment["HM_SHARED_TELEMETRY"] == "0",
                "stop_applied": self._web_stop_applied,
                "realtime": args.render_realtime,
            }
        self.protocol = protocol_type(
            session,
            emit,
            game_dir=self.plan.game,
            seek=self.seek,
            mute_audio=self.mute_audio,
            process_paused=self.pause_audio,
            graph=self.plan.graph,
            total_ns=round(self.duration * 1e9),
            start_ns=round(self.plan.start_seconds * 1e9),
            initial_preview=args.ui_preview_active,
            web_binding=(
                {key: self.web.document[key] for key in ("run_id", "request_digest")}
                if self.web is not None
                else None
            ),
            **protocol_options,
        )
        self.child = None
        self.phase = "preparing"
        self.stop_requested = False
        self.stop_deadline = None
        self.pending_seek = None
        self.seek_generation = None
        self.last_seek = 0
        self.selector = selectors.DefaultSelector()
        self.stdin_buffer = b""
        self.native = self.environment.get("HSTREAM_NATIVE_RUNNER")
        self.audio = None
        self.audio_paused = False
        self.audio_seeking = False
        self.snapshot = None
        self.prepared_analytics = None
        self.audio_muted = (
            self.environment.get("HSTREAM_RENDER_AUDIO_MUTED", "0") == "1"
            or "RENDER" not in self.plan.sinks
        )
        self._old_handlers = {}
        for kind in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[kind] = signal.signal(kind, self._stop_signal)
        if parent := self.environment.get("HSTREAM_UI_PARENT_PID"):
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            if not parent.isdecimal() or int(parent) != os.getppid():
                raise ValueError("HockeyMON backend has a stale or invalid HSTREAM_UI_PARENT_PID")
        try:
            self.selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "stdin")
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
            emit(
                "HSTREAM_STARTUP stage=input message=Standard input is not pollable; runtime commands are disabled"
            )

    def _stop_signal(self, signum, frame) -> None:
        self.stop_requested = True
        self.web_cancelled = True

    def _spawn(self, command: list[str], *, phase: str, cwd: Path | None = None) -> None:
        self.phase = phase
        if self.web is not None:
            self.web.check()
        label = {
            "calibration": "shared native calibration",
            "models": "shared model preparation",
            "python": "HockeyMON Python pipeline",
        }[phase]
        emit(f"HSTREAM_STARTUP stage={phase} message=Starting {label}")
        from hmlib.ui.process_guard import guarded_command

        self.child = subprocess.Popen(
            guarded_command(command),
            cwd=cwd or self.plan.cwd,
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self.selector.register(self.child.stdout.fileno(), selectors.EVENT_READ, "child")

    def _terminate_child(self) -> None:
        """Ask the guard to reap the worker and its decoder descendants."""
        if self.child is None:
            return
        try:
            if self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=6)
                except subprocess.TimeoutExpired as error:
                    # Keep the subreaper alive if a kernel-blocked descendant
                    # cannot exit yet; killing its owner would orphan it.
                    raise RuntimeError(
                        "Subprocess guard is still reaping descendants after shutdown timeout"
                    ) from error
        finally:
            if not self.child.stdout.closed:
                try:
                    self.selector.unregister(self.child.stdout.fileno())
                except KeyError:
                    pass
                self.child.stdout.close()
            self.child.stdin.close()

    def _native_runner(self) -> Path:
        from hmlib.ui.native_tools import native_runner

        self.native = str(native_runner(self.native))
        self.environment["HSTREAM_NATIVE_RUNNER"] = self.native
        return Path(self.native)

    def _prepare_models(self) -> None:
        selected = self.plan.config.get("hm_ui", {}).get("shared_analytics", {})
        if not selected or self.stop_requested:
            return
        import yaml

        from hmlib.ui.hstream_launch import merge, native_application_config, read_mapping
        from hmlib.ui.shared_analytics import (
            apply_prepared_shared_analytics,
            write_accepted_shared_analytics,
        )

        runner = self._native_runner()
        application = (
            Path(self.args.config[-1]).resolve()
            if self.args.config
            else native_application_config(runner)
        )
        # Native application files contain structural nodes at their root;
        # the resolved full configuration stores those nodes under pipeline.
        config = merge({}, self.plan.config)
        config["pipeline"] = merge(read_mapping(application), config.get("pipeline", {}))
        for feature in ("pose", "jersey", "action", "team"):
            if feature not in selected:
                config["pipeline"].setdefault("player-analytics", {}).setdefault(feature, {})[
                    "enable"
                ] = False
        if "reid" not in selected:
            config["pipeline"].setdefault("tracker", {})["reid-enable"] = False
        explicit = self.environment.get("HM_HSTREAM_MODEL_PREPARE")
        candidates = (
            [Path(explicit)]
            if explicit
            else [
                runner.parent / "hstream-model-prepare",
                runner.parent.parent / "hstream-model-prepare" / "hstream-model-prepare",
                Path(str(runner) + ".runfiles")
                / "kstream/src/apps/hstream-model-prepare/hstream-model-prepare",
            ]
        )
        helper = next(
            (path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None
        )
        if helper is None:
            raise FileNotFoundError(
                "Matching native model preparation helper is unavailable; set HM_HSTREAM_MODEL_PREPARE"
            )
        source = self.session / "models-input.yaml"
        output = self.session / "models-resolved.yaml"
        receipt = self.session / "models-receipt.json"
        source.write_text(yaml.safe_dump(config, sort_keys=False))
        self._spawn(
            [
                str(helper),
                "--config",
                str(source),
                "--config-directory",
                str(application.parent),
                "--output",
                str(output),
                "--receipt",
                str(receipt),
            ],
            phase="models",
        )
        code = self._pump()
        if self.stop_requested:
            return
        if code != 0:
            raise RuntimeError(f"Shared model preparation stopped with status {code}")
        result = json.loads(receipt.read_text())
        if (
            result.get("version") != 1
            or result.get("status") != "complete"
            or Path(result.get("output", "")).resolve() != output.resolve()
        ):
            raise ValueError("Shared model preparation did not commit the requested output")
        prepared = read_mapping(output)
        apply_prepared_shared_analytics(self.plan.config, prepared)
        write_accepted_shared_analytics(
            self.session,
            self.plan.game.name,
            self.plan.config,
            prepared,
            self.plan.initial_settings,
        )
        self.prepared_analytics = prepared
        self.plan.graph = self.plan.config.get("aspen", {})
        self.protocol.graph = self.plan.graph

    def _prepare(self) -> None:
        self._prepare_models()
        if self.stop_requested or set(self.plan.sources) != {"left", "right"}:
            return
        if self.web is not None:
            self.web.check()
            self.snapshot = self.web.calibration
            if self.snapshot is None:
                raise ValueError("Prepare stitching before starting paired HockeyMON Program")
            self._accept_calibration()
            return
        self._native_runner()
        # Native owns validity, recovery, synchronization and all editor dialogs.
        # One decoded pair is sufficient after the calibration transaction commits.
        import yaml

        from hmlib.ui.hstream_launch import native_application_config, strip_options

        effective_config = self.session / "native-launch.yaml"
        effective_config.write_text(yaml.safe_dump(self.plan.config, sort_keys=False))

        args = strip_options(
            self.arguments,
            {
                "--enable-sinks",
                "--ui-preview-windows",
                "--ui-preview-active",
                "--time-limit",
                "-t",
                "--start-time",
                "-s",
                "--show-scaled",
            },
        )
        args = [arg for arg in args if arg not in {"--show", "--render-realtime", "--dry-run"}]
        if not self.args.config:
            args += ["-c", str(native_application_config(Path(self.native)))]
        args += [
            "--enable-sources=" + self.args.enable_sources,
            "--settings-overlay=" + str(effective_config),
            "--stitching-calibration-only",
            "--stitching-calibration-with-ice-mask",
            "--enable-sinks=FAKE",
            "--time-limit=1",
            f"--start-time={self.plan.start_seconds:.9f}",
        ]
        from hmlib.ui.hstream_initial import prefer_initial_settings

        args = prefer_initial_settings(args, self.plan.initial_settings)
        self._spawn([self.native, *args], phase="calibration")
        code = self._pump()
        if self.stop_requested:
            return
        if code != 0:
            raise RuntimeError(f"Shared calibration stopped with status {code}")
        from hmlib.stitching.native_calibration import prepare_native_calibration

        self.snapshot = prepare_native_calibration(
            self.plan.game,
            self.session / "calibration",
            effective_config=effective_config,
            max_output_width=_native_max_output_width(self.plan.config),
        )
        self._accept_calibration()

    def _accept_calibration(self) -> None:
        self.environment["HM_SHARED_NATIVE_CALIBRATION"] = str(self.snapshot.directory)
        self.protocol.frame_offset = self.snapshot.frame_offsets[0]
        available = (
            min(
                self.source_info[name][0] - offset / self.source_info[name][1]
                for name, offset in zip(("left", "right"), self.snapshot.frame_offsets)
            )
            - self.plan.start_seconds
        )
        self.duration = max(0.0, min(self.duration, available))
        self.plan.limit_seconds = self.duration
        self.protocol.total_ns = round(self.duration * 1e9)

    def _prepare_audio(self) -> None:
        if self.web is not None:
            return  # Browser audio requires its separate supervised transport.
        from hmlib.ui.hstream_audio import AudioMonitor
        from hmlib.ui.hstream_launch import get_path

        if not get_path(self.plan.config, "pipeline.hmaudio.enable", True):
            return
        candidates = [name for name, info in self.source_info.items() if info[2]]
        if not candidates:
            emit("HSTREAM_STARTUP stage=audio message=Sources contain no audio stream")
            return
        offsets = dict(zip(("left", "right"), self.snapshot.frame_offsets)) if self.snapshot else {}
        source = min(candidates, key=lambda name: offsets.get(name, 0))
        self.audio = AudioMonitor(
            self.plan.sources[source],
            durations=self.source_info[source][4],
            offset=float(offsets.get(source, 0)) / self.source_info[source][1],
            muted=self.audio_muted,
        )

    def _python(self, offset_ns: int = 0) -> None:
        from hmlib.ui.hstream_runtime import write_document

        self.protocol.reset_child(offset_ns)
        for name in (
            "spec.json",
            "runtime.json",
            "state.json",
            "action-ack.json",
            "publication.json",
        ):
            (self.session / name).unlink(missing_ok=True)
        for path in self.session.glob("preview-*"):
            if path.suffix in {".jpg", ".json"}:
                path.unlink()
        for directory in (
            "operator-actions",
            "operator-results",
            "runtime-requests",
            "runtime-results",
        ):
            for path in (self.session / directory).glob("*.json"):
                path.unlink()
        write_document(self.session / "replay-controls.json", self.protocol.replay_state)
        command = self.plan.write(self.session, seek_seconds=offset_ns / 1e9)
        command[command.index("hmlib.cli.hmtrack")] = "hmlib.cli.hstream_worker"
        self._spawn(command, phase="python", cwd=self.worker_cwd)
        self.stop_deadline = None

    def seek(self, target_ns: int, generation: int) -> None:
        if self.phase != "python" or self.pending_seek is not None or self.stop_requested:
            raise ValueError("Seeking is unavailable while another operation is active")
        if (
            self.plan.sinks - {"RENDER", "FAKE"}
            or self.environment.get("HM_SHARED_TELEMETRY", "1") == "1"
        ):
            raise ValueError("Seeking requires recording and telemetry to be disabled")
        if not 0 <= target_ns < round(self.duration * 1e9):
            raise ValueError("Seek position lies outside the recording")
        self.pending_seek = (target_ns, generation)
        self.audio_seeking = True
        self.pause_audio(self.audio_paused)
        self._request_stop()

    def _request_stop(self) -> None:
        from hmlib.ui.hstream_runtime import write_document

        if self.child is None or self.child.poll() is not None or self.stop_deadline is not None:
            return
        if self.phase == "python":
            write_document(
                self.session / "operator-actions" / "supervisor-stop.json", {"kind": "stop"}
            )
        else:
            self.child.stdin.write(b"q\n")
            self.child.stdin.flush()
        self.stop_deadline = time.monotonic() + 20

    def mute_audio(self, muted: bool) -> None:
        self.audio_muted = muted
        if self.audio is not None:
            self.audio.set_muted(muted)

    def pause_audio(self, paused: bool) -> None:
        self.audio_paused = paused
        if self.audio is not None:
            self.audio.set_paused(paused or self.audio_seeking)

    def _refresh_calibration(self) -> None:
        """Reload native edits without moving the run's original playback origin."""
        if self.web is not None:
            self.web.check()
            self.snapshot = self.web.calibration
            if self.snapshot is None:
                raise ValueError("Web Program lost its completed native calibration")
            return
        import yaml

        from hmlib.stitching.native_calibration import prepare_native_calibration
        from hmlib.ui.hstream_launch import make_plan, set_path

        plan = make_plan(self.args, initial=self.initial)
        if self.prepared_analytics is not None:
            from hmlib.ui.shared_analytics import apply_prepared_shared_analytics

            apply_prepared_shared_analytics(plan.config, self.prepared_analytics)
            plan.graph = plan.config.get("aspen", {})
        if plan.sources != self.plan.sources:
            raise ValueError("Recording sources changed during playback; start a new run")
        plan.start_seconds = self.plan.start_seconds
        plan.limit_seconds = self.duration
        if self.environment["HM_SHARED_HIGH_BIT_DEPTH"] == "1":
            set_path(plan.config, "stitching.dtype", "float32")
        effective_config = self.session / "native-launch.yaml"
        effective_config.write_text(yaml.safe_dump(plan.config, sort_keys=False))
        self.snapshot = prepare_native_calibration(
            plan.game,
            self.session / "calibration",
            effective_config=effective_config,
            max_output_width=_native_max_output_width(plan.config),
        )
        self.environment["HM_SHARED_NATIVE_CALIBRATION"] = str(self.snapshot.directory)
        self.protocol.frame_offset = self.snapshot.frame_offsets[0]
        self.plan = plan
        self.protocol.graph = plan.graph

    def _line(self, line: str) -> None:
        if self.web is not None:
            self.protocol.handle_line(line)
            return
        if line in {"q", "quit"}:
            self.stop_requested = True
        elif self.phase == "calibration":
            if self.child and self.child.poll() is None:
                self.child.stdin.write((line + "\n").encode())
                self.child.stdin.flush()
        else:
            self.protocol.handle_line(line)

    def _web_stop_applied(self) -> None:
        self.stop_requested = True

    def _pump(self) -> int:
        while True:
            if self.stop_requested:
                self._request_stop()
            if self.stop_deadline is not None and time.monotonic() >= self.stop_deadline:
                self._terminate_child()
                raise TimeoutError(
                    "Pipeline did not complete its stop request; working outputs are retained"
                )
            for key, _ in self.selector.select(0.05):
                data = os.read(key.fd, 65536)
                if not data:
                    self.selector.unregister(key.fd)
                    continue
                if key.data == "child":
                    # Preserve native statuses and Python diagnostics verbatim.
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                else:
                    self.stdin_buffer += data
                    if len(self.stdin_buffer) > 1024 * 1024:
                        raise ValueError("Runtime input exceeds the protocol limit")
                    while b"\n" in self.stdin_buffer:
                        line, self.stdin_buffer = self.stdin_buffer.split(b"\n", 1)
                        self._line(line.decode("utf-8").strip())
                    if self.stdin_buffer in {b"q", b"p", b"r"}:
                        self._line(self.stdin_buffer.decode())
                        self.stdin_buffer = b""
            if self.phase == "python":
                self.protocol.poll()
                if self.audio is not None and (self.session / "runtime.json").is_file():
                    state = json.loads((self.session / "runtime.json").read_text())
                    timeline = self.protocol.timeline_seconds(state)
                    if timeline is not None:
                        self.audio.update(
                            timeline,
                            playing=state.get("state") == "running",
                        )
                if self.seek_generation is not None:
                    path = self.session / "runtime.json"
                    state = json.loads(path.read_text()) if path.is_file() else {}
                    if (
                        state.get("frame") is not None
                        and state.get("seek_ready", True)
                        and self.protocol.seek_completed(self.last_seek, self.seek_generation)
                        is not False
                    ):
                        self.seek_generation = None
                        self.audio_seeking = False
                        self.pause_audio(self.audio_paused)
            code = self.child.poll()
            if code is not None:
                # Drain trailing publication/errors before testing the exit code.
                # The guard normally closes every inherited writer. A broken
                # helper must never make shutdown block on an orphaned pipe.
                os.set_blocking(self.child.stdout.fileno(), False)
                remainder = self.child.stdout.read()
                if remainder:
                    sys.stdout.buffer.write(remainder)
                    sys.stdout.buffer.flush()
                try:
                    self.selector.unregister(self.child.stdout.fileno())
                except KeyError:
                    pass
                self.child.stdout.close()
                self.child.stdin.close()
                return code

    def _published_artifacts(self) -> list[dict]:
        receipt = self.session / "publication.json"
        if self.web is not None:
            from hmlib.ui.hstream_web import read_publications

            records = read_publications(self.web.root)
        else:
            records = json.loads(receipt.read_text()) if receipt.is_file() else []
        if not isinstance(records, list):
            raise TypeError("Publication receipt must contain an artifact list")
        if self.web is not None:
            return records  # The confined web receipt validates all metadata too.
        required = {
            kind
            for sink, kind in (
                ("ENCODE_FILE", "program"),
                ("ENCODE_STITCHED_FILE", "stitched"),
                ("ENCODE_PROGRAM_4K_FILE", "program-4k"),
            )
            if sink in self.plan.sinks
        }
        if self.environment["HM_SHARED_TELEMETRY"] == "1":
            required.add("telemetry")
        artifacts = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise TypeError("Published artifact must identify its durable file")
            path = Path(record["path"])
            if not path.is_absolute() or path.resolve().is_relative_to(self.working_directory):
                raise ValueError(f"Published artifact is not outside working storage: {path}")
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                raise ValueError(f"Published artifact is empty or not a regular file: {path}")
            kind = record.get("kind")
            if kind is None:
                if path.suffix.lower() in {".mp4", ".mkv", ".mov", ".db"}:
                    raise ValueError("Published media artifact has no logical kind")
                continue
            if kind not in {"program", "stitched", "program-4k", "telemetry"}:
                raise ValueError(f"Unknown published artifact kind: {kind}")
            required.discard(kind)
            artifacts.append(record)
        if required:
            raise ValueError(
                "Required publication artifacts are missing: " + ", ".join(sorted(required))
            )
        return artifacts

    def _cleanup_working_directory(self) -> None:
        info = self.working_directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._working_identity:
            raise RuntimeError("Working directory ownership changed; refusing cleanup")
        # rmtree does not traverse symlinks; only this supervisor's unique run
        # directory is owned. Original CLI cwd and published files remain intact.
        shutil.rmtree(self.working_directory)

    def _close_resources(self) -> None:
        # Every resource gets cleanup even if an earlier owner raises.
        with ExitStack() as cleanup:
            for kind, handler in self._old_handlers.items():
                cleanup.callback(signal.signal, kind, handler)
            cleanup.callback(self.selector.close)
            cleanup.callback(self.protocol.stopped)
            if self.audio is not None:
                cleanup.callback(self.audio.close)
            cleanup.callback(self._terminate_child)

    def run(self) -> int:
        closed = False
        try:
            (self.session / "publication.json").unlink(missing_ok=True)
            emit("HM_BACKEND_LAUNCH " + json.dumps(self.plan.launch_receipt()))
            if self.web is not None:
                emit(
                    "HM_WEB_PROGRAM_ADMITTED "
                    + json.dumps(
                        {
                            "version": 1,
                            "run_id": self.web.document["run_id"],
                            "request_digest": self.web.document["request_digest"],
                            "deployment_id": self.web.document["deployment_id"],
                        }
                    )
                )
            self._prepare()
            if self.stop_requested:
                if self.web is not None:
                    closed = True
                    self._close_resources()
                    self._cleanup_working_directory()
                    self._web_publication("cancelled", [])
                    return 0
                emit(
                    "HM_BACKEND_PUBLICATION "
                    + json.dumps(
                        {
                            "version": 1,
                            "status": "cancelled",
                            "stage": "preparing",
                            "game_id": self.plan.game.name,
                            "artifacts": [],
                        }
                    )
                )
                return 0
            self._prepare_audio()
            self._python()
            while True:
                code = self._pump()
                if self.pending_seek and not self.stop_requested:
                    if code:
                        raise RuntimeError("The previous pipeline failed while preparing a seek")
                    self.last_seek, self.seek_generation = self.pending_seek
                    self.pending_seek = None
                    if self.snapshot is not None:
                        self._refresh_calibration()
                    self._python(self.last_seek)
                    continue
                break
            if code:
                emit(
                    "HM_BACKEND_PUBLICATION "
                    + json.dumps(
                        {
                            "version": 1,
                            "status": "failed",
                            "game_id": self.plan.game.name,
                            "error": f"Python pipeline exited with status {code}",
                        }
                    )
                )
                return code
            cancelled = self.web is not None and (
                self.web_cancelled or (self.web.root / "cancel").exists()
            )
            artifacts = [] if cancelled else self._published_artifacts()
            closed = True
            self._close_resources()
            self._cleanup_working_directory()
            if self.web is not None:
                self._web_publication("cancelled" if cancelled else "complete", artifacts)
                return 0
            emit(
                "HM_BACKEND_PUBLICATION "
                + json.dumps(
                    {
                        "version": 1,
                        "status": "complete",
                        "game_id": self.plan.game.name,
                        "artifacts": artifacts,
                    }
                )
            )
            return 0
        finally:
            if not closed:
                self._close_resources()

    def _web_publication(self, status: str, artifacts: list[dict]) -> None:
        from hmlib.ui.hstream_web import RECEIPT_NAME, file_content

        self.web.receipt(status, artifacts)
        emit(
            "HM_WEB_PROGRAM_PUBLICATION "
            + json.dumps(
                {
                    "version": 1,
                    "run_id": self.web.document["run_id"],
                    "request_digest": self.web.document["request_digest"],
                    "receipt": RECEIPT_NAME,
                    "content": file_content(self.web.root / RECEIPT_NAME),
                }
            )
        )


def _web_main(request: str) -> int:
    from hmlib.ui.hstream_web import WebProgram

    context = WebProgram(Path(request))
    environment = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(context.environment(environment))
        from hmlib.ui.hstream_launch import parser

        arguments = context.arguments()
        args = parser().parse_args(arguments)
        return Supervisor(args, arguments, context.session, web=context).run()
    finally:
        os.environ.clear()
        os.environ.update(environment)
        context.close()


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    # Handshakes are deliberately independent of torch, OpenCV, CUDA and YAML.
    if arguments == ["--web-backend-info"]:
        from hmlib.ui.hstream_web import capabilities

        emit(json.dumps(capabilities()))
        return 0
    if arguments and arguments[0].split("=", 1)[0] == "--web-program-request":
        try:
            if len(arguments) == 2 and arguments[0] == "--web-program-request":
                return _web_main(arguments[1])
            if len(arguments) == 1 and arguments[0].startswith("--web-program-request="):
                return _web_main(arguments[0].split("=", 1)[1])
            raise ValueError("Web Program requests cannot be combined with desktop arguments")
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            emit(f"HSTREAM_BACKEND_ERROR {type(error).__name__}: {error}")
            return 1
    if arguments == ["--ui-backend-info"]:
        emit(
            json.dumps(
                {
                    "protocol": "hstream-ui",
                    "version": 1,
                    "backend": "hockeymon",
                    "preview": "jpeg-files",
                }
            )
        )
        return 0
    from hmlib.ui.hstream_launch import export_job, make_plan, parser

    args = parser().parse_args(arguments)
    try:
        if args.resolve_initial_sources:
            initial = os.environ.get("HM_UI_INITIAL_LAUNCH")
            if not initial:
                raise ValueError("Initial source resolution requires HM_UI_INITIAL_LAUNCH")
            plan = make_plan(args, initial=Path(initial), validate_models=False)
            emit(
                json.dumps(
                    {
                        "version": 1,
                        "game_id": plan.game.name,
                        "sources": {
                            role: [str(path.resolve()) for path in paths]
                            for role, paths in plan.sources.items()
                        },
                    }
                )
            )
            return 0
        if args.prepare_detector:
            from hmlib.ui.hstream_detector import prepare_detector

            prepare_detector(args.game_id, args.samples, args.detector_model)
            return 0
        if args.export_job:
            emit(str(export_job(args)))
            return 0
        if args.stitching_calibration_only:
            from hmlib.ui.native_tools import native_runner

            native = native_runner(os.environ.get("HSTREAM_NATIVE_RUNNER"))
            if initial := os.environ.get("HM_UI_INITIAL_LAUNCH"):
                from hmlib.ui.hstream_initial import calibration_arguments

                arguments, cwd = calibration_arguments(arguments, Path(initial), native)
                os.chdir(cwd)
            os.execv(str(native), [str(native), *arguments])
        if args.dry_run:
            plan = make_plan(args)
            emit(
                json.dumps(
                    {
                        "game": str(plan.game),
                        "arguments": plan.arguments,
                        "cwd": str(plan.cwd),
                        "sinks": sorted(plan.sinks),
                    },
                    indent=2,
                )
            )
            return 0
        requested_session = os.environ.get("HSTREAM_UI_BACKEND_SESSION")
        if requested_session:
            session = Path(requested_session).resolve(strict=True)
            return Supervisor(args, arguments, session).run()
        with tempfile.TemporaryDirectory(prefix="hm-cli-") as temporary:
            return Supervisor(args, arguments, Path(temporary)).run()
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        emit(f"HSTREAM_BACKEND_ERROR {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
