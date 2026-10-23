"""Desktop workspace, filesystem IPC, and supervised Python pipelines.

Requests are local operator actions, not an HTTP API. Each command is written
atomically into ``commands/<id>.json``; its durable result is retained under
``results/`` so polling cannot execute a completed action twice.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from typing_extensions import Self

from .files import (
    config_lock,
    file_info,
    identifier,
    read_config,
    read_json,
    revision,
    set_path,
    write_config,
    write_json,
)
from .jobs import ACTIVE_STATES, Jobs
from .media import Media, trajectory

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".mts", ".m2ts"}
PIPELINE_KINDS = {"track", "stitch"}


def _calibration_revision(game: Path) -> str:
    records = []
    for path in sorted(game.iterdir()):
        if path.is_file() and (
            path.suffix.lower() in {".pto", ".tif", ".png"}
            or path.name == ".stitching_artifacts.json"
        ):
            info = path.stat()
            records.append([path.name, info.st_size, info.st_mtime_ns, info.st_ino])
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def _copy_config_path(source: dict, target: dict, dotted: str) -> None:
    parts = dotted.split(".")
    value = source
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            parent = target
            for key in parts[:-1]:
                if not isinstance(parent, dict) or key not in parent:
                    return
                parent = parent[key]
            if isinstance(parent, dict):
                parent.pop(parts[-1], None)
            return
        value = value[part]
    set_path(target, dotted, copy.deepcopy(value))


class Workspace:
    def __init__(self, root_dir: str | Path, session_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(
            self.session_dir / ".owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._lock_fd)
            raise RuntimeError(f"Another desktop owns session {self.session_dir}") from None
        self.commands = self.session_dir / "commands"
        self.results = self.session_dir / "results"
        self.commands.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)
        self.jobs = Jobs(self.session_dir / "jobs")
        self.media = Media(self.session_dir / "media")
        self.current_game: Path | None = None
        self.operations: dict[str, Callable[[dict], Any]] = {}
        self._last_snapshot = ""
        self._closed = False
        self.publish_snapshot()

    def register_operation(self, name: str, callback: Callable[[dict], Any]) -> None:
        self.operations[name] = callback

    def game(self, game_id: str | None = None) -> Path:
        if game_id is None:
            if self.current_game is None:
                raise ValueError("Select a game first")
            return self.current_game
        name = identifier(game_id, "game ID")
        path = self.root_dir / name
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Game is not a directory directly under {self.root_dir}: {name}")
        return path

    def require_idle(self, game: Path | None = None) -> None:
        self._reap()
        for job in self.jobs.records.values():
            if job["status"] in ACTIVE_STATES and (
                game is None or job["game"] == str(game) or job.get("source_game") == str(game)
            ):
                raise RuntimeError(f"Stop job {job['id']} before changing this game")

    def list_games(self) -> list[dict]:
        return [
            {"id": path.name, "path": str(path), "has_config": (path / "config.yaml").is_file()}
            for path in sorted(self.root_dir.iterdir())
            if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
        ]

    def create_game(self, game_id: str) -> dict:
        path = self.root_dir / identifier(game_id, "game ID")
        path.mkdir(exist_ok=False)
        write_config(path, {"game": {"videos": {}}})
        self.current_game = path
        return {"id": path.name, "path": str(path)}

    def select_game(self, game_id: str) -> dict:
        self.current_game = self.game(game_id)
        return {"id": self.current_game.name, "path": str(self.current_game)}

    def set_root(self, value: str) -> dict:
        self.require_idle()
        root = Path(value).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("The workspace root must be a directory")
        self.root_dir = root
        self.current_game = None
        return {"root": str(root), "games": self.list_games()}

    def remove_video(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        self.require_idle(game)
        role = args["role"]
        selected = Path(os.path.abspath(game / args["path"]))
        with config_lock(game):
            config = read_config(game)
            videos = config.get("game", {}).get("videos", {})
            paths = videos.get(role)
            if not isinstance(paths, list):
                raise TypeError(f"No configured source list for role {role}")
            remaining = [
                value for value in paths if Path(os.path.abspath(game / value)) != selected
            ]
            if len(remaining) == len(paths):
                raise ValueError("The selected source is not configured for this role")
            videos[role] = remaining
            write_config(game, config)
        return {"role": role, "paths": remaining, "removed": str(selected), "media_preserved": True}

    def clean_stitching(self, args: dict) -> dict:
        from .clean import clean_stitching

        game = self.game(args.get("game_id"))
        self.require_idle(game)
        return clean_stitching(game)

    def update_outputs(self, args: dict) -> dict:
        from hmlib.ui.output_control import validate_changes

        self._reap()
        active = [
            job
            for job in self.jobs.records.values()
            if job.get("pipeline") and job["status"] in {"running", "paused"}
        ]
        if not active:
            raise ValueError("Start a pipeline before changing live output routes")
        if not (self.session_dir / "live" / "output-state.json").is_file():
            raise ValueError("The pipeline is still preparing its live output routes")
        changes = validate_changes(args["changes"])
        request_id = uuid.uuid4().hex
        write_json(
            self.session_dir / "live" / "output-actions" / f"{request_id}.json",
            {"id": request_id, "changes": changes},
        )
        return {"request_id": request_id, "state": "queued for the next output frame"}

    def import_videos(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        self.require_idle(game)
        role = args.get("role", "source")
        if role not in {"left", "right", "source"} and not re.fullmatch(
            r"cam[1-9][0-9]*", str(role)
        ):
            raise ValueError("Video role must be left, right, source, or camN")
        mode = args.get("mode", "link")
        if mode not in {"link", "copy"}:
            raise ValueError("Import mode must be link or copy")
        paths = args.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("Select at least one video file")
        sources = [Path(path).expanduser().resolve(strict=True) for path in paths]
        if any(not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES for path in sources):
            raise ValueError("Imports require regular video files")
        folder = game / ({"left": "cam1", "right": "cam2"}.get(role, role))
        if folder.is_symlink():
            raise ValueError("The import directory cannot be a symbolic link")
        folder.mkdir(exist_ok=True)
        created: list[Path] = []
        with config_lock(game):
            config = read_config(game)
            try:
                game_config = config.setdefault("game", {})
                videos = game_config.setdefault("videos", {})
                if not isinstance(videos, dict):
                    raise TypeError("game.videos must be a mapping")
                existing = videos.get(role, [])
                if not isinstance(existing, list):
                    raise TypeError(f"game.videos.{role} must be a list")
                values = list(existing)
                for source in sources:
                    destination = folder / source.name
                    counter = 1
                    while os.path.lexists(destination):
                        if destination.resolve() == source:
                            break
                        destination = folder / f"{source.stem}-{counter}{source.suffix}"
                        counter += 1
                    if not os.path.lexists(destination):
                        if mode == "link":
                            destination.symlink_to(source)
                        else:
                            # Exclusive creation prevents a concurrent import from
                            # replacing a previously imported recording.
                            with source.open("rb") as incoming, destination.open("xb") as output:
                                created.append(destination)
                                shutil.copyfileobj(incoming, output, length=1024 * 1024)
                                output.flush()
                                os.fsync(output.fileno())
                        if destination not in created:
                            created.append(destination)
                    relative = str(destination.relative_to(game))
                    if relative not in values:
                        values.append(relative)
                videos[role] = values
                write_config(game, config)
            except BaseException:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
        return {"game": str(game), "role": role, "paths": values}

    def concatenate_videos(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        self.require_idle(game)
        role = args.get("role", "source")
        if role not in {"left", "right", "source"} and not re.fullmatch(
            r"cam[1-9][0-9]*", str(role)
        ):
            raise ValueError("Video role must be left, right, source, or camN")
        with config_lock(game):
            config = read_config(game)
            paths = config.get("game", {}).get("videos", {}).get(role, [])
            if not isinstance(paths, list) or len(paths) < 2:
                raise ValueError("Import at least two chapters for this role before concatenating")
            request = {
                "game": str(game),
                "role": role,
                "paths": [str((game / path).resolve(strict=True)) for path in paths],
                "source_revision": revision(game),
            }
        work = self.session_dir / "concatenate-runs" / uuid.uuid4().hex
        work.mkdir(parents=True)
        write_json(work / "request.json", request)
        return self.jobs.start(
            [
                sys.executable,
                "-m",
                "hmlib.ui.workspace.worker",
                "concatenate",
                str(work / "request.json"),
            ],
            game=game,
            kind="concatenate",
            cwd=work,
            environment=self._environment(game),
            extra={"receipt": str(work / "result.json")},
        )

    def _environment(self, game: Path) -> dict[str, str]:
        environment = dict(os.environ)
        environment["HM_GAME_DIR"] = str(game.parent)
        environment["HM_UI_SESSION_DIR"] = str(self.session_dir / "live")
        environment["HM_UI_PIPELINE_CHILD"] = "1"
        # Subprocess cwd is isolated, while a source checkout remains importable.
        package_parent = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (package_parent, environment.get("PYTHONPATH", "")) if value
        )
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def pipeline_command(self, game: Path, kind: str, arguments: list[str]) -> list[str]:
        if kind not in PIPELINE_KINDS:
            raise ValueError("Pipeline kind must be track or stitch")
        if not isinstance(arguments, list) or not all(
            isinstance(value, str) for value in arguments
        ):
            raise ValueError("arguments must be a list of CLI strings")
        # Identity and IPC are owned by the supervisor; accepting a second game
        # or camera-ui option would make the visible selection inaccurate.
        for value in arguments:
            option = value.split("=", 1)[0]
            if option in {"--game-id", "--camera-ui", "--deploy-dir"}:
                raise ValueError(f"{option} is managed by the desktop")
        module = "hmlib.cli.hmtrack" if kind == "track" else "hmlib.cli.stitch"
        from hmlib.ui.runtime_setup import launch_arguments

        setup_arguments = launch_arguments(game, kind)
        videos = read_config(game).get("game", {}).get("videos", {})
        has_input = any(value.split("=", 1)[0] == "--input-video" for value in arguments)
        if (
            not has_input
            and isinstance(videos, dict)
            and videos.get("source") is not None
            and not videos["source"]
            and not videos.get("left")
            and not videos.get("right")
        ):
            raise ValueError("Import a source video before starting playback")
        if not has_input and isinstance(videos, dict) and videos.get("source"):
            sources = videos["source"]
            if not isinstance(sources, list) or len(sources) != 1:
                raise ValueError(
                    "Select one source video, or concatenate its chapters before playback"
                )
            setup_arguments += ["--input-video", str((game / sources[0]).resolve(strict=True))]
        return [
            sys.executable,
            "-m",
            module,
            "--game-id",
            game.name,
            "--camera-ui=1",
            "--deploy-dir",
            str(game),
            *setup_arguments,
            *arguments,
        ]

    def _prepare_pipeline(self) -> None:
        self._reap()
        if any(
            record["status"] in ACTIVE_STATES and record.get("pipeline", False)
            for record in self.jobs.records.values()
        ):
            raise RuntimeError("Stop the active pipeline before starting another")
        live = self.session_dir / "live"
        if live.is_symlink():
            raise ValueError("Live IPC directory cannot be a symbolic link")
        if live.exists():
            shutil.rmtree(live)
        live.mkdir()

    def start_command(self, command: list[str], game_id: str, kind: str = "track") -> dict:
        game = self.game(game_id)
        self.current_game = game
        # The initial CLI may contain relative input/model paths. Preserve its
        # invocation directory instead of interpreting those paths in a run cache.
        return self._start_pipeline(command, game, kind, cwd=Path.cwd())

    def _start_pipeline(
        self,
        command: list[str],
        game: Path,
        kind: str,
        *,
        extra: dict | None = None,
        cwd: Path | None = None,
    ) -> dict:
        self._prepare_pipeline()
        work = self.session_dir / "runs" / uuid.uuid4().hex
        work.mkdir(parents=True)
        return self.jobs.start(
            command,
            game=game,
            kind=kind,
            cwd=cwd or work,
            environment=self._environment(game),
            extra={"pipeline": True, **(extra or {})},
        )

    def stop_job(self, job_id: str, *, grace_seconds: float = 45) -> dict:
        self.jobs.poll()
        record = self.jobs.get(job_id)
        if record["status"] not in ACTIVE_STATES or record["status"] == "stopping":
            return dict(record)
        runtime_path = self.session_dir / "live" / "runtime.json"
        if record.get("pipeline") and runtime_path.is_file():
            runtime = read_json(runtime_path)
            if "stop" in runtime.get("capabilities", []):
                request_id = uuid.uuid4().hex
                write_json(
                    self.session_dir / "live" / "operator-actions" / f"{request_id}.json",
                    {"id": request_id, "kind": "stop"},
                )
                record["stop_action_id"] = request_id
                return self.jobs.request_stop(job_id, grace_seconds=grace_seconds)
        if record.get("pipeline"):
            record.update(
                forced_stop=True,
                error="The pipeline has not exposed playback controls; forced termination may leave incomplete outputs",
            )
        return self.jobs.stop(job_id)

    def seek_job(self, args: dict) -> dict:
        from .highlights import number

        self._reap()
        record = self.jobs.get(args["job_id"])
        if (
            not record.get("pipeline")
            or record["kind"] != "track"
            or record["status"] not in {"running", "paused"}
        ):
            raise ValueError("Seeking requires an active tracking playback job")
        seconds = number(args["seconds"], "Seek time", 0, 365 * 86400)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        timestamp = f"{int(hours):02d}:{int(minutes):02d}:{seconds_part:06.3f}"
        options = {
            "-s",
            "--start-time",
            "--start-frame-time",
            "--start-frame",
            "--start-frame-offset",
        }
        command, index = [], 0
        original = record["command"]
        while index < len(original):
            value = original[index]
            option = value.split("=", 1)[0]
            if option in options:
                index += 1 if "=" in value else 2
                continue
            if value.startswith("-s") and not value.startswith("--") and len(value) > 2:
                index += 1
                continue
            command.append(value)
            index += 1
        command += ["--start-time", timestamp]
        record["seek_restart"] = {"command": command, "seconds": seconds, "cwd": record["cwd"]}
        self.jobs._save(record)
        self.stop_job(record["id"])
        return {
            "waiting_for": record["id"],
            "seconds": seconds,
            "state": "finishing current recording",
        }

    def restart_job(self, args: dict) -> dict:
        self._reap()
        record = self.jobs.get(args["job_id"])
        if not record.get("pipeline") or record["kind"] not in PIPELINE_KINDS:
            raise ValueError("Restart requires a tracking or stitching pipeline job")
        if record["status"] == "stopping":
            raise ValueError("Wait for the current stop operation before restarting")
        if record["status"] in ACTIVE_STATES:
            record["seek_restart"] = {
                "command": record["command"],
                "cwd": record["cwd"],
                "operation": "restart",
            }
            record.pop("seek_restart_consumed", None)
            self.jobs._save(record)
            self.stop_job(record["id"])
            return {"waiting_for": record["id"], "state": "finishing current recording"}
        return self._restart_pipeline(record, record["command"], Path(record["cwd"]))

    def _restart_pipeline(
        self, record: dict, command: list[str], cwd: Path, *, extra: dict | None = None
    ) -> dict:
        job = self._start_pipeline(
            command,
            Path(record["game"]),
            record["kind"],
            cwd=cwd,
            extra={
                "previous_job": record["id"],
                **(extra or {}),
                **{key: record[key] for key in ("experiment_id", "source_game") if key in record},
            },
        )
        if "experiment_id" in record:
            directory, experiment = self._experiment(
                Path(record["source_game"]), record["experiment_id"]
            )
            experiment.update(status="running", job_id=job["id"])
            write_json(directory / "experiment.json", experiment)
        return job

    def export_job(self, args: dict) -> dict:
        if "job_id" in args:
            record = self.jobs.get(args["job_id"])
            if not record.get("pipeline"):
                raise ValueError("Job scripts export tracking or stitching pipelines")
            command = list(record["command"])
            game = Path(record["game"])
        else:
            game = self.game(args.get("game_id"))
            command = self.pipeline_command(
                game, args.get("kind", "track"), args.get("arguments", [])
            )
        path = Path(args["path"]).expanduser().absolute()
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Job script already exists: {path}")
        if not path.parent.is_dir():
            raise ValueError("Choose an existing directory for the job script")
        # Exported jobs run independently, with previews off and no stale IPC.
        cleaned: list[str] = []
        index = 0
        while index < len(command):
            value = command[index]
            if value == "--camera-ui":
                index += 2
                continue
            if not value.startswith("--camera-ui="):
                cleaned.append(value)
            index += 1
        cleaned.append("--camera-ui=0")
        contents = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "unset HM_UI_SESSION_DIR HM_UI_PIPELINE_CHILD\n"
            f"export HM_GAME_DIR={shlex.quote(str(game.parent))}\n"
            f"export PYTHONPATH={shlex.quote(str(Path(__file__).resolve().parents[3]))}"
            "${PYTHONPATH:+:${PYTHONPATH}}\n"
            + (
                f"cd {shlex.quote(record['cwd'])}\n"
                if "job_id" in args
                else 'job_workdir="$(mktemp -d "${TMPDIR:-/tmp}/hm-render.XXXXXX")"\ncd "${job_workdir}"\n'
            )
            + f"exec {shlex.join(cleaned)}\n"
        )
        with path.open("x") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o700)
        return {"path": str(path), "command": cleaned}

    def catalog(self, game: Path) -> dict:
        archives, recordings, videos = [], [], []
        pattern = re.compile(
            rf"^{re.escape(game.name)}-(tracking|stitched|program_4k)_output"
            r"(?:-full)?(?:-with-audio)?(?:-part\d+)?(?:-(\d+))?\.(mp4|mkv|mov)$",
            re.IGNORECASE,
        )
        for path in sorted(game.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            match = pattern.fullmatch(path.name)
            if match:
                archives.append(
                    {**file_info(path), "kind": match[1], "version": int(match[2] or 0)}
                )
            elif path.suffix.lower() in {".db", ".sqlite"} and "telemetry" in path.stem:
                version_match = re.search(r"-(\d+)$", path.stem)
                recordings.append(
                    {**file_info(path), "version": int(version_match[1]) if version_match else 0}
                )
        configured = read_config(game).get("game", {}).get("videos", {})
        if not isinstance(configured, dict):
            raise TypeError("game.videos must be a mapping")
        for role, paths in configured.items():
            if not isinstance(paths, list):
                raise TypeError(f"game.videos.{role} must be a list")
            for value in paths:
                path = game / value
                videos.append(
                    {"role": role, "path": str(path), "name": path.name, "exists": path.is_file()}
                )
        for archive in archives:
            archive["recordings"] = [
                item["path"] for item in recordings if item["version"] == archive["version"]
            ]
            for name in (
                f"recording_segments-{archive['version']}.json",
                f"{game.name}-recording_segments-{archive['version']}.json",
            ):
                manifest = game / name
                if not manifest.is_file():
                    continue
                from hmlib.utils.output_publication import artifact_name

                for segment in read_json(manifest).get("segments", []):
                    base = segment["filename"]
                    if not base.startswith(game.name + "-"):
                        base = game.name + "-" + base
                    if artifact_name(base, archive["version"]) == archive["name"]:
                        archive.update(
                            timeline_offset_seconds=segment["timeline_offset_seconds"],
                            duration_seconds=segment["duration_seconds"],
                            segment_manifest=str(manifest),
                        )
        return {"archives": archives, "recordings": recordings, "videos": videos}

    def _experiments_dir(self, game: Path) -> Path:
        return game / ".hm-workspace" / "experiments"

    def media_path(self, value: str) -> Path:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Select a media file")
        game = self.game()
        if path.is_relative_to(game) or path.is_relative_to(self.session_dir):
            return path
        configured = self.catalog(game)["videos"]
        imported = {Path(item["path"]).resolve() for item in configured if item["exists"]}
        # Older games discover root-level video aliases without config lists.
        imported.update(
            candidate.resolve() for candidate in game.glob("*.mp4") if candidate.is_file()
        )
        if path not in imported:
            raise ValueError("Media must belong to the selected game or a desktop job")
        return path

    def experiment_telemetry(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        _, experiment = self._experiment(game, args["experiment_id"])
        output = {}
        messages = []
        for label, directory in (("baseline", game), ("trial", Path(experiment["game_directory"]))):
            replay = experiment.get("replay")
            if label == "baseline" and replay:
                output[label] = trajectory(
                    Path(replay["recording_path"]),
                    max_points=args.get("max_points", 2000),
                    run_id=replay["run_id"],
                    start_seconds=replay["start_ms"] / 1000,
                    end_seconds=replay["end_ms"] / 1000,
                )
                continue
            candidates = sorted(
                self.catalog(directory)["recordings"],
                key=lambda row: (row["version"], row["mtime_ns"]),
                reverse=True,
            )
            if not candidates:
                output[label] = None
                messages.append(f"No {label} telemetry recording is available")
                continue
            output[label] = trajectory(
                Path(candidates[0]["path"]), max_points=args.get("max_points", 2000)
            )
            if label == "trial" and replay:
                for points in output[label]["series"].values():
                    for point in points:
                        point["time_seconds"] += replay["start_ms"] / 1000
        output["message"] = "; ".join(messages)
        return output

    def list_experiments(self, game: Path) -> list[dict]:
        return [
            read_json(path)
            for path in sorted(self._experiments_dir(game).glob("*/experiment.json"))
        ]

    def _experiment(self, game: Path, experiment_id: str) -> tuple[Path, dict]:
        directory = self._experiments_dir(game) / identifier(experiment_id, "experiment ID")
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Unknown experiment")
        return directory, read_json(directory / "experiment.json")

    def create_experiment(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        self.require_idle(game)
        kind = args.get("kind", "camera")
        if kind not in {"camera", "stitch"}:
            raise ValueError("Experiment kind must be camera or stitch")
        changes = args.get("changes", {})
        if not isinstance(changes, dict) or not all(isinstance(key, str) for key in changes):
            raise ValueError("Experiment changes must map dotted configuration keys to values")
        allowed = "rink.camera." if kind == "camera" else "stitching."
        if any(not key.startswith(allowed) for key in changes):
            raise ValueError(f"{kind} experiment changes must be under {allowed}")
        experiment_id = uuid.uuid4().hex
        directory = self._experiments_dir(game) / experiment_id
        candidate = directory / "games" / game.name
        candidate.mkdir(parents=True)
        with config_lock(game):
            source_revision = revision(game)
            calibration_revision = _calibration_revision(game)
            config = copy.deepcopy(read_config(game))
            replay = None
            if args.get("replay"):
                if kind != "camera":
                    raise ValueError("Telemetry replay is only available for camera experiments")
                from .replay import prepare_replay

                replay_args = args["replay"]
                catalog = self.catalog(game)
                database = (game / replay_args["recording_path"]).resolve(strict=True)
                if str(database) not in {item["path"] for item in catalog["recordings"]}:
                    raise ValueError("Select a telemetry recording from this game")
                replay, config = prepare_replay(
                    database, game, directory, replay_args, catalog, self.media
                )
            # Copy calibration data and link media. Output recordings and old
            # experiments never become a candidate's mutable input tree.
            for source in game.iterdir():
                if source.name.startswith(".") or source.name == "config.yaml":
                    continue
                if source.is_dir() and re.fullmatch(r"cam\d+|source", source.name):
                    target_dir = candidate / source.name
                    target_dir.mkdir()
                    for video in source.iterdir():
                        if video.is_file() and video.suffix.lower() in VIDEO_SUFFIXES:
                            (target_dir / video.name).symlink_to(video.resolve())
                elif source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
                    if "_output" not in source.stem:
                        (candidate / source.name).symlink_to(source.resolve())
                elif source.is_file() and source.suffix.lower() in {
                    ".pto",
                    ".tif",
                    ".png",
                    ".json",
                }:
                    shutil.copy2(source, candidate / source.name)
            artifact_manifest = game / ".stitching_artifacts.json"
            if artifact_manifest.is_file():
                shutil.copy2(artifact_manifest, candidate / artifact_manifest.name)
            if replay:
                from .replay import restore_replay_mask

                restore_replay_mask(replay, candidate, config)
            # Keep input aliases stable even for explicitly selected paths
            # outside the conventional camera subdirectories.
            videos = config.get("game", {}).get("videos", {})
            if isinstance(videos, dict):
                for role, paths in videos.items():
                    if not isinstance(paths, list):
                        raise TypeError(f"game.videos.{role} must be a list")
                    videos[role] = [str((game / path).resolve(strict=True)) for path in paths]
            for key, value in changes.items():
                set_path(config, key, value)
            optics_notes = []
            if kind == "stitch":
                from hmlib.stitching.settings import read_stitching_settings
                from hmlib.stitching.video_lens import resolve_video_lens

                settings = read_stitching_settings(config)
                sources = (
                    [str((game / item).resolve()) for paths in videos.values() for item in paths]
                    if isinstance(videos, dict)
                    else []
                )
                if not sources:
                    sources = [
                        str(path.resolve())
                        for folder in (game / "cam1", game / "cam2")
                        for path in sorted(folder.glob("*"))
                        if path.suffix.lower() in VIDEO_SUFFIXES
                    ]
                if settings.camera_auto_fov or settings.camera_auto_lens:
                    if not sources:
                        raise ValueError(
                            "Import source recordings before freezing automatic experiment optics"
                        )
                    settings, optics_notes = resolve_video_lens(config, settings, sources)
                    set_path(
                        config,
                        "stitching.camera_fov",
                        {
                            "horizontal_fov": settings.horizontal_fov,
                            "vertical_fov": settings.vertical_fov,
                        },
                    )
                    set_path(config, "stitching.input_lens", asdict(settings.input_lens))
                    set_path(
                        config,
                        "stitching.camera_automation",
                        {"fov": settings.camera_auto_fov, "lens": settings.camera_auto_lens},
                    )
                    set_path(config, "stitching.auto_input_lens_frozen", True)
            write_config(candidate, config)
        record = {
            "id": experiment_id,
            "kind": kind,
            "name": str(args.get("name", experiment_id)),
            "source_game": str(game),
            "game_directory": str(candidate),
            "status": "created",
            "source_revision": source_revision,
            "calibration_revision": calibration_revision,
            "optics_notes": optics_notes,
            "changes": changes,
            "replay": replay,
            "created_at": time.time(),
        }
        write_json(directory / "experiment.json", record)
        return record

    def run_experiment(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        directory, record = self._experiment(game, args["experiment_id"])
        self.require_idle(game)
        candidate = Path(record["game_directory"])
        kind = "track" if record["kind"] == "camera" else "stitch"
        arguments = args.get("arguments", [])
        if record.get("replay"):
            replay = record["replay"]
            if Media.identity(Path(replay["source_path"])) != replay["source_identity"]:
                raise RuntimeError("The frozen experiment source changed; create a new experiment")
            managed = {
                "--input-video",
                "--input-tracking-data",
                "--start-frame",
                "--start-time",
                "--start-frame-time",
                "--max-frames",
                "--detector-model",
                "--detector",
                "--compare-detectors",
                "-s",
                "-t",
            }
            if any(value.split("=", 1)[0] in managed for value in arguments):
                raise ValueError(
                    "Replay source, tracking and range arguments are owned by the experiment"
                )
            arguments = [
                "--input-tracking-data",
                replay["input_tracking_data"],
                "--start-frame",
                str(replay["start_frame"]),
                "--max-frames",
                str(replay["max_frames"]),
                *arguments,
            ]
        if kind == "stitch":
            arguments = ["--force", *arguments]
        command = self.pipeline_command(candidate, kind, arguments)
        job = self._start_pipeline(
            command,
            candidate,
            kind,
            extra={"experiment_id": record["id"], "source_game": str(game)},
        )
        record.update(status="running", job_id=job["id"])
        write_json(directory / "experiment.json", record)
        return job

    def promote_experiment(self, args: dict) -> dict:
        game = self.game(args.get("game_id"))
        self.require_idle(game)
        directory, record = self._experiment(game, args["experiment_id"])
        if record["status"] != "completed":
            raise ValueError("Run the experiment successfully before promoting its settings")
        with config_lock(game):
            if revision(game) != record["source_revision"]:
                raise RuntimeError(
                    "Game configuration changed after the experiment was created; create a new candidate"
                )
            original_config = (
                (game / "config.yaml").read_bytes() if (game / "config.yaml").exists() else None
            )
            config = read_config(game)
            for key, value in record["changes"].items():
                set_path(config, key, value)
            if record["kind"] == "stitch":
                from hmlib.stitching.artifact_validation import validate_artifact_generation
                from hmlib.stitching.artifacts import (
                    artifact_stage,
                    publish_artifacts,
                    stitching_lock,
                )
                from hmlib.stitching.pto_sources import rewrite_pto_sources

                candidate = Path(record["game_directory"])
                if _calibration_revision(game) != record.get("calibration_revision"):
                    raise RuntimeError(
                        "Source calibration changed after the experiment was created; create a new candidate"
                    )
                candidate_config = read_config(candidate)
                for key in (
                    "stitching.control_points",
                    "stitching.frame_offsets",
                    "stitching.projection_framing",
                    "stitching.camera_fov",
                    "stitching.input_lens",
                    "stitching.camera_automation",
                    "stitching.auto_input_lens_frozen",
                    "rink.scoreboard.perspective_polygon",
                    "rink.ice_contours_mask_count",
                    "rink.ice_contours_mask_centroid",
                    "rink.ice_contours_combined_bbox",
                    "rink.ice_contours_geometry_revision",
                    "rink.ice_contours_mask_file_prefix",
                ):
                    _copy_config_path(candidate_config, config, key)
                # Rink correspondences are tied to the prior pixel geometry.
                if isinstance(config.get("rink"), dict):
                    config["rink"].pop("workspace_map", None)
                required = ["hm_project.pto", ".stitching_artifacts.json"]
                if any(not (candidate / name).is_file() for name in required):
                    raise ValueError(
                        "Candidate did not produce a complete Python stitching calibration"
                    )
                manifest = read_json(candidate / ".stitching_artifacts.json")
                validate_artifact_generation(candidate, project_name="hm_project.pto")
                with stitching_lock(game), artifact_stage(game) as stage:
                    if _calibration_revision(game) != record.get("calibration_revision"):
                        raise RuntimeError(
                            "Source calibration changed while waiting for its publication lock"
                        )
                    names = []
                    for source in candidate.iterdir():
                        if (
                            source.is_file()
                            and not source.is_symlink()
                            and (
                                source.suffix.lower() in {".pto", ".tif", ".png"}
                                or source.name == ".stitching_artifacts.json"
                            )
                        ):
                            shutil.copy2(source, stage / source.name)
                            names.append(source.name)
                    if not manifest or not any(name.startswith("mapping_") for name in names):
                        raise ValueError("Candidate is missing calibrated mapping artifacts")
                    for name in names:
                        if name.endswith(".pto"):
                            rewrite_pto_sources(
                                stage / name, source_directory=candidate, target_directory=game
                            )
                    validate_artifact_generation(stage, project_name="hm_project.pto")
                    # Config is part of the same rollback-protected publication.
                    write_config(stage, config)
                    names.append("config.yaml")
                    publish_artifacts(
                        game, stage, names, expected_old_contents={"config.yaml": original_config}
                    )
            else:
                write_config(game, config)
        record.update(status="promoted", promoted_at=time.time())
        write_json(directory / "experiment.json", record)
        return record

    def dispatch(self, operation: str, args: dict) -> Any:
        if not isinstance(args, dict):
            raise TypeError("Operation arguments must be an object")
        if operation in self.operations:
            return self.operations[operation](args)
        if operation == "workspace.set_root":
            return self.set_root(args["path"])
        if operation == "outputs.update":
            return self.update_outputs(args)
        if operation == "outputs.result":
            return read_json(
                self.session_dir
                / "live"
                / "output-results"
                / f"{identifier(args['request_id'], 'request ID')}.json"
            )
        if operation == "games.list":
            return self.list_games()
        if operation == "games.create":
            return self.create_game(args["game_id"])
        if operation == "games.select":
            return self.select_game(args["game_id"])
        if operation == "games.import_videos":
            return self.import_videos(args)
        if operation == "games.concatenate":
            return self.concatenate_videos(args)
        if operation == "games.remove_video":
            return self.remove_video(args)
        if operation == "games.clean_stitching":
            return self.clean_stitching(args)
        if operation == "catalog.list":
            return self.catalog(self.game(args.get("game_id")))
        if operation == "media.info":
            return self.media.info(self.media_path(args["path"]))
        if operation == "media.frame":
            return self.media.frame(
                self.media_path(args["path"]), args.get("time_ms", 0), args.get("width", 1280)
            )
        if operation == "jobs.start":
            game = self.game(args.get("game_id"))
            kind = args.get("kind", "track")
            return self._start_pipeline(
                self.pipeline_command(game, kind, args.get("arguments", [])), game, kind
            )
        if operation == "jobs.stop":
            return self.stop_job(args["job_id"])
        if operation == "jobs.seek":
            return self.seek_job(args)
        if operation == "jobs.restart":
            return self.restart_job(args)
        if operation in {"jobs.pause", "jobs.resume"}:
            return self.jobs.pause(args["job_id"], operation == "jobs.pause")
        if operation == "jobs.log":
            return self.jobs.log(
                args["job_id"], int(args.get("offset", 0)), int(args.get("limit", 65536))
            )
        if operation == "jobs.export":
            return self.export_job(args)
        if operation == "experiments.list":
            return self.list_experiments(self.game(args.get("game_id")))
        if operation == "experiments.telemetry":
            return self.experiment_telemetry(args)
        if operation == "experiments.replay_sources":
            from .replay import replay_sources

            game = self.game(args.get("game_id"))
            catalog = self.catalog(game)
            database = (game / args["recording_path"]).resolve(strict=True)
            if str(database) not in {item["path"] for item in catalog["recordings"]}:
                raise ValueError("Select a telemetry recording from this game")
            return replay_sources(database, game, catalog)
        if operation == "experiments.create":
            return self.create_experiment(args)
        if operation == "experiments.run":
            return self.run_experiment(args)
        if operation == "experiments.promote":
            return self.promote_experiment(args)
        if operation.startswith("highlights."):
            return self._highlights(operation, args)
        raise ValueError(f"Unknown workspace operation: {operation}")

    def _highlights(self, operation: str, args: dict) -> dict:
        from .highlights import normalize_plan

        game = self.game(args.get("game_id"))
        path = game / "highlights.json"
        if operation in {"highlights.track_choices", "highlights.capture_track"}:
            from hmlib.telemetry.database import database_stem

            from .tracks import capture_tracks

            selected = (game / args["archive_path"]).expanduser().resolve(strict=True)
            archive = next(
                (
                    item
                    for item in self.catalog(game)["archives"]
                    if Path(item["path"]).resolve() == selected
                ),
                None,
            )
            if archive is None:
                raise ValueError(
                    "Track capture requires a published archive from the selected game"
                )
            databases = [Path(value) for value in archive["recordings"]]
            if args.get("database"):
                databases = [
                    value
                    for value in databases
                    if value.resolve() == Path(args["database"]).resolve()
                ]
            else:
                canonical = [
                    value
                    for value in databases
                    if value.name.startswith(database_stem(game.name) + "-")
                ]
                databases = canonical or databases
            if len(databases) != 1:
                raise ValueError(
                    "Select exactly one telemetry database from this archive's publication version"
                )
            return capture_tracks(
                databases[0],
                game.name,
                args,
                route=args.get("route")
                or ("stitched" if archive["kind"] == "stitched" else "program"),
                segment_offset_ms=1000 * archive.get("timeline_offset_seconds", 0),
                capture=operation == "highlights.capture_track",
            )
        if operation == "highlights.load":
            return (
                read_json(path)
                if path.exists()
                else {"schema": 2, "base_name": "highlights", "intervals": []}
            )
        plan = normalize_plan(args["plan"] if "plan" in args else read_json(path), game)
        if operation == "highlights.save":
            with config_lock(game):
                write_json(path, plan)
            return plan
        if operation not in {"highlights.preview", "highlights.render"}:
            raise ValueError(f"Unknown highlights operation: {operation}")
        if not plan["intervals"]:
            raise ValueError("Add a clip or title card before rendering")
        work = self.session_dir / "highlight-runs" / uuid.uuid4().hex
        work.mkdir(parents=True)
        request = {
            "game": str(game),
            "plan": plan,
            "preview": operation == "highlights.preview",
            "output": str(work / "highlights.mp4"),
        }
        write_json(work / "request.json", request)
        command = [
            sys.executable,
            "-m",
            "hmlib.ui.workspace.worker",
            "highlights",
            str(work / "request.json"),
        ]
        return self.jobs.start(
            command,
            game=game,
            kind=operation,
            cwd=work,
            environment=self._environment(game),
            extra={"output": request["output"], "receipt": str(work / "result.json")},
        )

    def snapshot(self) -> dict:
        selected = {"archives": [], "recordings": [], "videos": []}
        experiments, errors = [], []
        if self.current_game:
            try:
                selected = self.catalog(self.current_game)
            except Exception as error:  # noqa: BLE001 - report corrupt catalog
                errors.append(f"Catalog: {type(error).__name__}: {error}")
            try:
                experiments = self.list_experiments(self.current_game)
            except Exception as error:  # noqa: BLE001 - keep configuration repair available
                errors.append(f"Experiments: {type(error).__name__}: {error}")
        output_state = self.session_dir / "live" / "output-state.json"
        outputs = None
        if output_state.is_file():
            try:
                outputs = read_json(output_state)
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"Live outputs: {type(error).__name__}: {error}")
        recording_state = self.session_dir / "live" / "recording-state.json"
        recordings_status = None
        if recording_state.is_file():
            try:
                recordings_status = read_json(recording_state)
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"Live recordings: {type(error).__name__}: {error}")
        return {
            "schema": 1,
            "root": str(self.root_dir),
            "current_game": str(self.current_game) if self.current_game else None,
            "games": self.list_games(),
            "jobs": list(self.jobs.records.values()),
            **selected,
            "experiments": experiments,
            "errors": errors,
            "output_routes": outputs,
            "recording_outputs": recordings_status,
        }

    def publish_snapshot(self) -> None:
        snapshot = self.snapshot()
        encoded = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        if encoded != self._last_snapshot:
            write_json(self.session_dir / "workspace.json", snapshot)
            self._last_snapshot = encoded

    def _reap(self) -> None:
        self.jobs.poll()
        for record in list(self.jobs.records.values()):
            if record["status"] in ACTIVE_STATES:
                continue
            if "experiment_id" in record:
                source = Path(record["source_game"])
                directory, experiment = self._experiment(source, record["experiment_id"])
                if experiment.get("job_id") == record["id"] and experiment["status"] not in {
                    record["status"],
                    "promoted",
                }:
                    experiment.update(
                        status=record["status"], finished_at=record.get("finished_at")
                    )
                    write_json(directory / "experiment.json", experiment)
            if record["status"] == "completed" and "receipt" in record and "result" not in record:
                receipt = Path(record["receipt"])
                if receipt.is_file():
                    record["result"] = read_json(receipt)
                    self.jobs._save(record)
            restart = record.get("seek_restart")
            if restart and not record.get("seek_restart_consumed") and not self._closed:
                record["seek_restart_consumed"] = True
                self.jobs._save(record)
                operation = restart.get("operation", "seek")
                if record["status"] != "completed":
                    record[f"{operation}_error"] = (
                        "Restart was cancelled because the previous recording did not finish successfully"
                    )
                    self.jobs._save(record)
                    continue
                try:
                    job = self._restart_pipeline(
                        record,
                        restart["command"],
                        cwd=Path(restart["cwd"]),
                        extra=(
                            {"seek_seconds": restart["seconds"]} if "seconds" in restart else None
                        ),
                    )
                    record[f"{operation}_job_id"] = job["id"]
                except Exception as error:  # noqa: BLE001 - expose queued restart failures
                    record[f"{operation}_error"] = f"{type(error).__name__}: {error}"
                self.jobs._save(record)

    def poll(self) -> None:
        self._reap()
        if self._closed:
            self.publish_snapshot()
            return
        for path in sorted(self.commands.glob("*.json")):
            reply = self.results / path.name
            if reply.exists():
                continue
            request_id = path.stem
            try:
                identifier(request_id, "command filename")
                request = read_json(path)
                if not isinstance(request, dict):
                    raise TypeError("A workspace command must be an object")
                request_id = identifier(request["id"], "request ID")
                if request_id != path.stem:
                    raise ValueError("Command ID must match its filename")
                result = self.dispatch(request["operation"], request.get("args", {}))
                response = {"id": request_id, "ok": True, "result": result, "error": None}
            # IPC must return every operation failure to the operator.
            except Exception as error:  # noqa: BLE001
                response = {
                    "id": request_id,
                    "ok": False,
                    "result": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            write_json(reply, response)
        self.publish_snapshot()

    def serve(self, interval: float = 0.1) -> None:
        try:
            while not self._closed:
                self.poll()
                time.sleep(interval)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for job_id in list(self.jobs.processes):
                self.stop_job(job_id)
            self.jobs.close()
            self.poll()
        finally:
            os.close(self._lock_fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
