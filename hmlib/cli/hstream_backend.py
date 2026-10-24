"""HockeyMON process adapter for the shared HStream desktop protocol."""

from __future__ import annotations

import errno
import json
import os
import re
import selectors
import signal
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


class Supervisor:
    """Keep native preparation and Python processing within the UI-owned group."""

    def __init__(self, args, arguments: list[str], session: Path) -> None:
        from hmlib.ui.hstream_launch import make_plan
        from hmlib.ui.hstream_runtime import RuntimeProtocol

        self.args, self.arguments, self.session = args, arguments, session
        launch = os.environ.get("HM_UI_INITIAL_LAUNCH")
        self.initial = Path(launch) if launch else None
        self.plan = make_plan(args, initial=self.initial)
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
        self.fps = source_info[0][1]
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
        if set(self.plan.sources) == {"source"}:
            self.environment["HM_SHARED_SOURCE_CHAPTERS"] = json.dumps(
                [str(path) for path in self.plan.sources["source"]]
            )
        self.protocol = RuntimeProtocol(
            session,
            emit,
            game_dir=self.plan.game,
            seek=self.seek,
            mute_audio=self.mute_audio,
            process_paused=self.pause_audio,
            graph=self.plan.graph,
            total_ns=round(self.duration * 1e9),
            start_ns=round(self.plan.start_seconds * 1e9),
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
        self.audio_muted = (
            self.environment.get("HSTREAM_RENDER_AUDIO_MUTED", "0") == "1"
            or "RENDER" not in self.plan.sinks
        )
        self._old_handlers = {}
        for kind in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[kind] = signal.signal(kind, self._stop_signal)
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

    def _spawn(self, command: list[str], *, phase: str, cwd: Path | None = None) -> None:
        self.phase = phase
        emit(
            f"HSTREAM_STARTUP stage={phase} message=Starting {'shared native calibration' if phase == 'calibration' else 'HockeyMON Python pipeline'}"
        )
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

    def _prepare(self) -> None:
        if set(self.plan.sources) != {"left", "right"}:
            return
        if not self.native:
            from hmlib.ui.desktop import executable

            self.native = str(
                executable(None, "hstream-cli", (Path("/opt/hstream/bin/hstream-cli"),))
            )
            self.environment["HSTREAM_NATIVE_RUNNER"] = self.native
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
        ]
        self._spawn([self.native, *args], phase="calibration")
        code = self._pump()
        if code != 0 or self.stop_requested:
            raise RuntimeError(f"Shared calibration stopped with status {code}")
        from hmlib.stitching.native_calibration import prepare_native_calibration

        self.snapshot = prepare_native_calibration(
            self.plan.game, self.session / "calibration", effective_config=effective_config
        )
        self.environment["HM_SHARED_NATIVE_CALIBRATION"] = str(self.snapshot.directory)
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
        self.protocol.reset_child(offset_ns)
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
        self._spawn(command, phase="python", cwd=self.plan.cwd)
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
        import yaml

        from hmlib.stitching.native_calibration import prepare_native_calibration
        from hmlib.ui.hstream_launch import make_plan, set_path

        plan = make_plan(self.args, initial=self.initial)
        if plan.sources != self.plan.sources:
            raise ValueError("Recording sources changed during playback; start a new run")
        plan.start_seconds = self.plan.start_seconds
        plan.limit_seconds = self.duration
        if self.environment["HM_SHARED_HIGH_BIT_DEPTH"] == "1":
            set_path(plan.config, "stitching.dtype", "float32")
        effective_config = self.session / "native-launch.yaml"
        effective_config.write_text(yaml.safe_dump(plan.config, sort_keys=False))
        self.snapshot = prepare_native_calibration(
            plan.game, self.session / "calibration", effective_config=effective_config
        )
        self.environment["HM_SHARED_NATIVE_CALIBRATION"] = str(self.snapshot.directory)
        self.plan = plan
        self.protocol.graph = plan.graph

    def _line(self, line: str) -> None:
        if self.phase == "calibration":
            if self.child and self.child.poll() is None:
                self.child.stdin.write((line + "\n").encode())
                self.child.stdin.flush()
        elif line in {"q", "quit"}:
            self.stop_requested = True
        else:
            self.protocol.handle_line(line)

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
                    if state.get("frame") is not None and state.get("fps"):
                        self.audio.update(
                            float(state["frame"]) / float(state["fps"]),
                            playing=state.get("state") == "running",
                        )
                if self.seek_generation is not None:
                    path = self.session / "runtime.json"
                    if path.is_file() and json.loads(path.read_text()).get("frame") is not None:
                        self.protocol.seek_completed(self.last_seek, self.seek_generation)
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

    def run(self) -> int:
        try:
            (self.session / "publication.json").unlink(missing_ok=True)
            self._prepare()
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
            receipt = self.session / "publication.json"
            records = json.loads(receipt.read_text()) if receipt.is_file() else []
            artifacts = []
            for record in records:
                path = Path(record["path"])
                if path.suffix == ".db":
                    kind = "telemetry"
                elif path.suffix.lower() in {".mp4", ".mkv", ".mov"}:
                    kind = (
                        "program-4k"
                        if "4k" in path.name.lower()
                        else "stitched" if "stitched" in path.name else "program"
                    )
                else:
                    continue
                artifacts.append({**record, "kind": kind})
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
            # Execute every cleanup even when a worker, audio device or helper
            # fails during shutdown. ExitStack preserves visible failures.
            with ExitStack() as cleanup:
                for kind, handler in self._old_handlers.items():
                    cleanup.callback(signal.signal, kind, handler)
                cleanup.callback(self.selector.close)
                cleanup.callback(self.protocol.stopped)
                if self.audio is not None:
                    cleanup.callback(self.audio.close)
                cleanup.callback(self._terminate_child)


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    # Handshake is deliberately independent of torch, OpenCV, CUDA and YAML.
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
        if args.prepare_detector:
            from hmlib.ui.hstream_detector import prepare_detector

            prepare_detector(args.game_id, args.samples, args.detector_model)
            return 0
        if args.export_job:
            emit(str(export_job(args)))
            return 0
        if args.stitching_calibration_only:
            from hmlib.ui.desktop import executable

            native = executable(
                os.environ.get("HSTREAM_NATIVE_RUNNER"),
                "hstream-cli",
                (Path("/opt/hstream/bin/hstream-cli"),),
            )
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
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        emit(f"HSTREAM_BACKEND_ERROR {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
