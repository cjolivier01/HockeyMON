"""Opt-in, confined producer contract for an authenticated outer supervisor.

This module is stdlib-only. It does not grant browser access or advertise a
qualified preview/runtime backend. The caller owns admission and all descendants.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.machinery
import json
import os
import re
import stat
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

PROTOCOL = "hstream-web-program"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
READ_BYTES = 256 * 1024
REQUEST_NAME = "hockeymon-program-request.json"
RECEIPT_NAME = "hockeymon-program-receipt.json"
NATIVE_FILES = {
    "runner": "bin/hstream-cli",
    "calibration_export": "bin/hstream-calibration-export",
    "model_prepare": "bin/hstream-model-prepare",
    "model_infer": "bin/hstream-model-infer",
    "application": "configs/ds_hockey_app_config.yaml",
    "baseline": "configs/baseline.yaml",
}
OUTPUT_KINDS = {
    "ENCODE_FILE": "program",
    "ENCODE_STITCHED_FILE": "stitched",
    "ENCODE_PROGRAM_4K_FILE": "program-4k",
}
SINKS = {"FAKE", "RENDER", "RTMP", "RTSP", *OUTPUT_KINDS}


def canonical_bytes(document: Any) -> bytes:
    """Python JSON: sorted keys, UTF-8, no whitespace/NaN, no Unicode normalization."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def request_digest(document: dict) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in document.items() if key != "request_digest"})
    ).hexdigest()


def _object(value: Any, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Invalid {label} fields")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 content identity")
    return value


def _parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        raise ValueError("Invalid confined path")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Invalid confined path")
    return parts


def _open_beneath(root: Path, relative: str, *, directory: bool = False) -> int:
    parts = _parts(relative)
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
            )
            os.close(parent)
            parent = child
        return os.open(
            parts[-1],
            os.O_RDONLY
            | os.O_NOFOLLOW
            | os.O_NONBLOCK
            | os.O_CLOEXEC
            | (os.O_DIRECTORY if directory else 0),
            dir_fd=parent,
        )
    finally:
        os.close(parent)


def local_identity(info: os.stat_result) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in {
            "device": info.st_dev,
            "inode": info.st_ino,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }.items()
    }


class PinnedFile:
    def __init__(
        self, root: Path, relative: str, expected: dict | None = None, *, allow_empty: bool = False
    ):
        self.root, self.relative = root, relative
        self.fd = _open_beneath(root, relative)
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size < (0 if allow_empty else 1):
                raise ValueError("Pinned input must be a nonempty regular file")
            self.identity = local_identity(info)
            self.content = None
            if expected is not None:
                _object(expected, {"size", "sha256"}, "portable file identity")
                _digest(expected["sha256"])
                if type(expected["size"]) is not int or expected["size"] != info.st_size:
                    raise ValueError("Pinned input has changed size")
                digest = hashlib.sha256()
                while block := os.read(self.fd, READ_BYTES):
                    digest.update(block)
                self.content = {"size": info.st_size, "sha256": digest.hexdigest()}
                if self.content != expected:
                    raise ValueError("Pinned input content does not match admission")
                os.lseek(self.fd, 0, os.SEEK_SET)
                self.check()
        except BaseException:
            os.close(self.fd)
            raise

    @property
    def path(self) -> Path:
        return self.root / self.relative

    def check(self) -> None:
        if local_identity(os.fstat(self.fd)) != self.identity:
            raise ValueError("Pinned input changed after admission")
        current = _open_beneath(self.root, self.relative)
        try:
            if local_identity(os.fstat(current)) != self.identity:
                raise ValueError("Pinned input path changed after admission")
        finally:
            os.close(current)

    def read(self, maximum: int) -> bytes:
        self.check()
        size = int(self.identity["size"])
        if size > maximum:
            raise ValueError("Metadata exceeds its byte limit")
        os.lseek(self.fd, 0, os.SEEK_SET)
        contents = bytearray()
        while len(contents) < size:
            block = os.read(self.fd, min(READ_BYTES, size - len(contents)))
            if not block:
                raise ValueError("Pinned metadata was truncated")
            contents.extend(block)
        self.check()
        return bytes(contents)

    def close(self) -> None:
        os.close(self.fd)


def file_content(path: Path) -> dict[str, Any]:
    """Portable file binding; inode/device are checked separately on each host."""
    path = Path(path)
    pinned = PinnedFile(path.parent.resolve(strict=True), path.name, allow_empty=True)
    try:
        digest = hashlib.sha256()
        while block := os.read(pinned.fd, READ_BYTES):
            digest.update(block)
        pinned.check()
        return {"size": int(pinned.identity["size"]), "sha256": digest.hexdigest()}
    finally:
        pinned.close()


def _package_root(name: str) -> Path:
    spec = importlib.machinery.PathFinder.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(f"Installed {name} package is unavailable")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise ValueError(f"Installed {name} package has ambiguous locations")
    return Path(locations[0]).resolve(strict=True)


def runtime_provenance() -> dict[str, Any]:
    """Hash actual installed package bytes, including owned extensions/libraries.

    No imports of hockeymon, Torch, CUDA, OpenCV or YAML occur here. Absolute
    installation paths and host-local inode/device values are not content IDs.
    """
    records = []
    native_extension = False
    for name in ("hmlib", "hockeymon"):
        root = Path(__file__).resolve().parents[1] if name == "hmlib" else _package_root(name)
        for parent, folders, files in os.walk(root, followlinks=False):
            folders[:] = sorted(folder for folder in folders if folder != "__pycache__")
            if any((Path(parent) / folder).is_symlink() for folder in folders):
                raise ValueError("Installed runtime contains a linked package directory")
            for filename in sorted(files):
                if filename.endswith((".pyc", ".pyo")):
                    continue
                path = Path(parent) / filename
                relative = path.relative_to(root).as_posix()
                before = local_identity(path.lstat())
                identity = file_content(path.resolve(strict=True))
                if local_identity(path.lstat()) != before:
                    raise ValueError("Installed runtime link changed during provenance read")
                records.append({"path": name + "/" + relative, **identity})
                native_extension |= (
                    name == "hockeymon"
                    and filename.startswith("_hockeymon.")
                    and filename.endswith(".so")
                )
    records.sort(key=lambda item: item["path"])
    return {
        "content_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "file_count": len(records),
        "native_extension": native_extension,
        "python": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
            **file_content(Path(sys.executable).resolve(strict=True)),
        },
        "baseline": file_content(Path(__file__).resolve().parents[1] / "config/baseline.yaml"),
    }


def capabilities() -> dict[str, Any]:
    from hmlib.ui.hstream_web_control import ACTIONS, MAX_CONTROL_BYTES

    return {
        "protocol": PROTOCOL,
        "version": 1,
        "backend": "hockeymon",
        "features": [
            "confined-request-v1",
            "confined-publication-v1",
            "selected-jpeg-snapshot-v1",
            "typed-runtime-controls-v1",
        ],
        "qualified": False,
        "missing": [
            "authenticated-preview-qualification",
            "browser-audio",
            "runtime-parity-qualification",
            "export-qualification",
        ],
        "maximum_request_bytes": MAX_REQUEST_BYTES,
        "control": {
            "version": 1,
            "maximum_request_bytes": MAX_CONTROL_BYTES,
            "actions": list(ACTIONS),
        },
        "runtime": runtime_provenance(),
    }


def _write_json(path: Path, document: Any, maximum: int = MAX_RECEIPT_BYTES) -> None:
    data = canonical_bytes(document)
    if len(data) > maximum:
        raise ValueError("Producer document exceeds its byte limit")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".web-program-", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


class WebProgram:
    """Pinned request and private workspace owned by one producer supervisor."""

    def __init__(self, request: Path, *, environment: dict[str, str] | None = None):
        environment = os.environ if environment is None else environment
        if environment.get("HM_UI_INITIAL_LAUNCH"):
            raise ValueError("Desktop initial-launch envelopes are forbidden in web Program")
        request = Path(request)
        if request.name != REQUEST_NAME or not request.is_absolute():
            raise ValueError("Web Program requires its absolute fixed request filename")
        self.root = request.parent.resolve(strict=True)
        self.pins: dict[str, PinnedFile] = {}
        self.lock = None
        self.working_directory = None
        self._directories: dict[Path, tuple[int, int]] = {}
        try:
            self.pins["request"] = PinnedFile(self.root, REQUEST_NAME)
            document = json.loads(self.pins["request"].read(MAX_REQUEST_BYTES))
            _object(
                document,
                {
                    "version",
                    "run_id",
                    "deployment_id",
                    "runtime",
                    "request_digest",
                    "native_root",
                    "native",
                    "game_config",
                    "sources",
                    "launch",
                    "calibration",
                },
                "web Program request",
            )
            if document["version"] != 1 or type(document["version"]) is not int:
                raise ValueError("Unsupported web Program request version")
            if (
                not isinstance(document["run_id"], str)
                or str(uuid.UUID(document["run_id"])) != document["run_id"]
            ):
                raise ValueError("Web Program requires a canonical run UUID")
            _digest(document["deployment_id"])
            if _digest(document["request_digest"]) != request_digest(document):
                raise ValueError("Web Program request digest does not match")
            self.document = document
            actual = runtime_provenance()
            if (
                canonical_bytes(document["runtime"]) != canonical_bytes(actual)
                or not actual["native_extension"]
            ):
                raise ValueError("Installed Program runtime does not match its admitted deployment")
            native_root = Path(document["native_root"])
            if not native_root.is_absolute() or native_root.resolve(strict=True) != native_root:
                raise ValueError("Native installation must have a canonical absolute root")
            _object(document["native"], set(NATIVE_FILES), "native tool bindings")
            for name, relative in NATIVE_FILES.items():
                self.pins["native_" + name] = PinnedFile(
                    native_root, relative, document["native"][name]
                )
                if name not in {"application", "baseline"} and not os.access(
                    native_root / relative, os.X_OK
                ):
                    raise ValueError("Pinned native helper is not executable")
            if document["native"]["baseline"] != actual["baseline"]:
                raise ValueError("Native and HockeyMON baselines differ")
            self.pins["game_config"] = PinnedFile(
                self.root, "game/config.yaml", document["game_config"]
            )
            self._validate_launch()
            sources = document["sources"]
            if not isinstance(sources, dict) or set(sources) not in ({"source"}, {"left", "right"}):
                raise ValueError("Program requires one source or one ordered camera pair")
            for role, entries in sources.items():
                if not isinstance(entries, list) or not entries:
                    raise ValueError("Each admitted source requires ordered chapters")
                for index, entry in enumerate(entries):
                    _object(entry, {"alias", "path", "content"}, "source binding")
                    parts = _parts(entry["alias"])
                    if len(parts) != 2 or parts[0] != "game":
                        raise ValueError("Source aliases must be direct game files")
                    path = Path(entry["path"])
                    if not path.is_absolute() or path.resolve(strict=True) != path:
                        raise ValueError("Source recording must use a canonical absolute path")
                    self.pins[f"source_{role}_{index}"] = PinnedFile(
                        path.parent, path.name, entry["content"]
                    )
                    if (self.root / entry["alias"]).resolve(strict=True) != path:
                        raise ValueError("Source alias does not match its admitted recording")
            self.game = self.root / "game"
            self.session = self.root / "work/hockeymon-session"
            self.output = self.root / "work/hockeymon-output"
            self.home = self.root / "home"
            for relative in (
                "game",
                "home",
                "work",
                "work/hockeymon-session",
                "work/hockeymon-output",
            ):
                self._directory(relative)
            if (self.root / RECEIPT_NAME).exists():
                raise ValueError("Web Program already has a terminal receipt")
            self.lock = os.open(
                self.session / "producer.lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                0o600,
            )
            if not stat.S_ISREG(os.fstat(self.lock).st_mode):
                raise ValueError("Producer lock is not a regular file")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.session / "web-admission.json").exists():
                raise ValueError("Web Program request was already admitted")
            self.calibration = self._calibration()
        except BaseException:
            self.close()
            raise

    def _directory(self, relative: str) -> None:
        path = self.root / relative
        path.mkdir(mode=0o700, exist_ok=True)
        descriptor = _open_beneath(self.root, relative, directory=True)
        try:
            info = os.fstat(descriptor)
            self._directories[path] = (info.st_dev, info.st_ino)
        finally:
            os.close(descriptor)

    def _validate_launch(self) -> None:
        launch = _object(
            self.document["launch"],
            {"sinks", "telemetry", "start_seconds", "duration_seconds", "realtime", "preview"},
            "Program launch",
        )
        sinks = launch["sinks"]
        if (
            not isinstance(sinks, list)
            or not sinks
            or not all(isinstance(value, str) for value in sinks)
            or len(set(sinks)) != len(sinks)
            or set(sinks) - SINKS
        ):
            raise ValueError("Invalid Program output selection")
        for name in ("telemetry", "realtime"):
            if type(launch[name]) is not bool:
                raise ValueError("Program flags require booleans")
        for name in ("start_seconds", "duration_seconds"):
            value = launch[name]
            if value is None and name == "duration_seconds":
                continue
            if (
                type(value) not in {int, float}
                or not 0 <= value <= 100000000
                or (name == "duration_seconds" and value == 0)
            ):
                raise ValueError("Invalid Program playback range")
        if launch["preview"] not in {"none", "program", "stitched", "source0", "source1"}:
            raise ValueError("Invalid Program preview route")

    def _calibration(self):
        calibration = self.document["calibration"]
        if set(self.document["sources"]) == {"source"}:
            if calibration is not None:
                raise ValueError("Single-source Program must not inherit a paired calibration")
            return None
        if calibration is None:
            raise ValueError(
                "Prepare stitching before starting HockeyMON Program with paired cameras"
            )
        _object(
            calibration,
            {"directory", "manifest", "generation", "output_generation", "invalidation_id"},
            "completed calibration",
        )
        parts = _parts(calibration["directory"])
        if parts[0] != "calibration" or len(parts) != 2:
            raise ValueError("Completed calibration must be in its private snapshot directory")
        relative = calibration["directory"] + "/native-calibration.yaml"
        self.pins["calibration"] = PinnedFile(self.root, relative, calibration["manifest"])
        from hmlib.stitching.native_calibration import load_native_calibration

        result = load_native_calibration(self.root / calibration["directory"])
        for key in ("generation", "output_generation", "invalidation_id"):
            if (
                not isinstance(calibration[key], str)
                or not calibration[key]
                or getattr(result, key) != calibration[key]
            ):
                raise ValueError("Completed native calibration changed its generation")
        status = result.config.get("hstream_ui", {}).get("stitching_calibration", {}).get("status")
        if status is not None and status != "complete":
            raise ValueError("Prepare stitching before starting a pending native calibration")
        result.verify_sources(
            self.game,
            {
                role: [item["path"] for item in entries]
                for role, entries in self.document["sources"].items()
            },
        )
        import yaml

        manifest = yaml.safe_load(self.pins["calibration"].read(16 * 1024 * 1024))
        for name, expected in manifest["artifacts"].items():
            if len(_parts(name)) != 1:
                raise ValueError("Invalid native calibration artifact")
            self.pins["calibration_" + name] = PinnedFile(
                self.root, calibration["directory"] + "/" + name, expected
            )
        self.pins["calibration"].check()
        return result

    def check(self) -> None:
        for path, expected in self._directories.items():
            descriptor = _open_beneath(
                self.root, path.relative_to(self.root).as_posix(), directory=True
            )
            try:
                info = os.fstat(descriptor)
                if (info.st_dev, info.st_ino) != expected:
                    raise ValueError("Private Program directory was replaced")
            finally:
                os.close(descriptor)
        for pinned in self.pins.values():
            pinned.check()
        for entries in self.document["sources"].values():
            for entry in entries:
                if (self.root / entry["alias"]).resolve(strict=True) != Path(entry["path"]):
                    raise ValueError("Source alias changed after admission")

    def arguments(self) -> list[str]:
        launch = self.document["launch"]
        arguments = [
            "--game-id=game",
            "--enable-sources=URI-MULTIPLE",
            "--enable-sinks=" + ",".join(launch["sinks"]),
            "--start-time=" + str(launch["start_seconds"]),
            "--ui-preview-active=" + launch["preview"],
            "-c",
            str(self.pins["native_application"].path),
        ]
        if launch["duration_seconds"] is not None:
            arguments.append("--time-limit=" + str(launch["duration_seconds"]))
        if launch["realtime"]:
            arguments.append("--render-realtime")
        if launch["telemetry"]:
            arguments.append(
                "--options=pipeline.ds-playtracker.private-properties.telemetry-db-dir="
                + str(self.output)
            )
        return arguments

    def environment(self, inherited: dict[str, str]) -> dict[str, str]:
        self.check()
        import yaml

        config = yaml.safe_load(self.pins["game_config"].read(MAX_REQUEST_BYTES))
        overlays = config.get("preview_overlays") if isinstance(config, dict) else None
        _object(overlays, {"players", "play", "rink"}, "frozen preview_overlays")
        if any(type(value) is not bool for value in overlays.values()):
            raise ValueError("Frozen preview_overlays must contain exactly three booleans")
        allowed = {
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        }
        result = {key: value for key, value in inherited.items() if key in allowed}
        result.update(
            {
                "HOME": str(self.home),
                "XDG_CONFIG_HOME": str(self.home / ".config"),
                "HM_GAME_DIR": str(self.root),
                "HM_OUTPUT_WORK_DIR": str(self.output),
                "HM_CONFIG_ROOT": str(self.pins["native_baseline"].path.parent),
                "HSTREAM_UI_BACKEND_SESSION": str(self.session),
                "HSTREAM_UI_PARENT_PID": str(os.getppid()),
                "HSTREAM_NATIVE_RUNNER": str(self.pins["native_runner"].path),
                "HM_HSTREAM_CALIBRATION_EXPORT": str(self.pins["native_calibration_export"].path),
                "HM_HSTREAM_MODEL_PREPARE": str(self.pins["native_model_prepare"].path),
                "HM_HSTREAM_MODEL_INFER": str(self.pins["native_model_infer"].path),
                "HSTREAM_RENDER_AUDIO_MUTED": "1",
                "HSTREAM_UI_PREVIEW_OVERLAYS": ",".join(
                    name for name in ("players", "play", "rink") if overlays[name]
                ),
                "HM_WEB_PROGRAM_REQUEST": str(self.root / REQUEST_NAME),
                "HM_WEB_PROGRAM_DIGEST": self.document["request_digest"],
            }
        )
        return result

    def validate_plan(self, plan) -> None:
        self.check()
        expected = {
            role: [Path(item["path"]) for item in entries]
            for role, entries in self.document["sources"].items()
        }
        actual_sources = {
            role: [path.resolve(strict=True) for path in paths]
            for role, paths in plan.sources.items()
        }
        if (
            plan.game != self.game
            or actual_sources != expected
            or plan.sinks != set(self.document["launch"]["sinks"])
        ):
            raise ValueError("Resolved Program plan differs from admitted inputs or outputs")
        if bool(plan.config["hm_shared_ui"]["telemetry"]) != self.document["launch"]["telemetry"]:
            raise ValueError("Resolved Program telemetry differs from admission")
        from hmlib.ui.hstream_launch import get_path

        for key in (
            "video_out.output_video_path",
            "aspen.video_out.output_video_path",
            "aspen.plugins.video_out.params.output_video_path",
        ):
            value = get_path(plan.config, key)
            if value and (
                not isinstance(value, str) or "/" in value or "\\" in value or value in {".", ".."}
            ):
                raise ValueError("Web Program working video outputs must be plain filenames")
        if (
            plan.explicit_arguments
            or plan.start_seconds != self.document["launch"]["start_seconds"]
            or plan.limit_seconds != self.document["launch"]["duration_seconds"]
        ):
            raise ValueError("Resolved Program range or explicit arguments differ from admission")
        if self.calibration:
            self.calibration.verify_sources(
                self.game,
                {role: [str(path) for path in paths] for role, paths in plan.sources.items()},
            )

    def write_admission(self) -> None:
        self.check()
        _write_json(
            self.session / "web-admission.json",
            {
                "version": 1,
                "run_id": self.document["run_id"],
                "request_digest": self.document["request_digest"],
                "files": {
                    name: {"path": str(pin.path), **pin.identity} for name, pin in self.pins.items()
                },
            },
        )

    def expected_outputs(self) -> set[str]:
        result = {
            OUTPUT_KINDS[sink] for sink in self.document["launch"]["sinks"] if sink in OUTPUT_KINDS
        }
        if self.document["launch"]["telemetry"]:
            result.add("telemetry")
        return result

    def receipt(self, status: str, records: list[dict]) -> dict:
        self.check()
        if self.working_directory is not None and os.path.lexists(self.working_directory):
            raise ValueError("Producer working storage has not completed cleanup")
        if status not in {"complete", "cancelled"} or (status == "cancelled" and records):
            raise ValueError("Invalid producer terminal outcome")
        artifacts, seen, kinds, versions = [], set(), set(), set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "name",
                "path",
                "version",
                "kind",
                "identity",
            }:
                raise ValueError("Invalid pinned producer publication record")
            path = Path(record["path"])
            if (
                path.parent != self.game
                or path.name in seen
                or type(record["version"]) is not int
                or record["version"] < 1
            ):
                raise ValueError("Publication must contain unique confined game files")
            if not isinstance(record["name"], str) or len(_parts(record["name"])) != 1:
                raise ValueError("Published logical name must be one filename")
            kind = record["kind"]
            if kind not in {*OUTPUT_KINDS.values(), "telemetry", "metadata"}:
                raise ValueError("Invalid producer artifact kind")
            pinned = PinnedFile(self.game, path.name)
            try:
                if pinned.identity != record["identity"]:
                    raise ValueError("Published output changed before terminal receipt")
                os.fsync(pinned.fd)
                artifact = {
                    "path": "game/" + path.name,
                    "name": record["name"],
                    "kind": kind,
                    "version": record["version"],
                    **pinned.identity,
                }
                artifacts.append(artifact)
            finally:
                pinned.close()
            seen.add(path.name)
            versions.add(record["version"])
            if kind != "metadata":
                if kind in kinds:
                    raise ValueError("Publication repeated a requested output kind")
                kinds.add(kind)
        if len(versions) > 1:
            raise ValueError("Published outputs and companions do not share one generation")
        if status == "complete" and kinds != self.expected_outputs():
            raise ValueError("Published output kinds differ from the admitted selection")
        document = {
            "protocol": PROTOCOL,
            "version": 1,
            "run_id": self.document["run_id"],
            "deployment_id": self.document["deployment_id"],
            "runtime": self.document["runtime"],
            "request_digest": self.document["request_digest"],
            "status": status,
            "artifacts": artifacts,
        }
        _write_json(self.root / RECEIPT_NAME, document)
        return document

    def close(self) -> None:
        for pinned in self.pins.values():
            pinned.close()
        self.pins.clear()
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None


def worker_binding() -> tuple[Path, dict]:
    """Recheck compute-local pins before a guarded Python child uses the request."""
    request = Path(os.environ["HM_WEB_PROGRAM_REQUEST"])
    if not request.is_absolute() or request.name != REQUEST_NAME:
        raise ValueError("Invalid worker Program request")
    root = request.parent.resolve(strict=True)
    pinned = PinnedFile(root, REQUEST_NAME)
    try:
        document = json.loads(pinned.read(MAX_REQUEST_BYTES))
        if request_digest(document) != document.get("request_digest") or document[
            "request_digest"
        ] != os.environ.get("HM_WEB_PROGRAM_DIGEST"):
            raise ValueError("Worker Program request digest changed")
        request_identity = pinned.identity
    finally:
        pinned.close()
    admission_file = PinnedFile(root, "work/hockeymon-session/web-admission.json")
    try:
        admission = json.loads(admission_file.read(MAX_RECEIPT_BYTES))
    finally:
        admission_file.close()
    _object(admission, {"version", "run_id", "request_digest", "files"}, "local producer admission")
    if (
        admission["version"] != 1
        or admission["run_id"] != document["run_id"]
        or admission["request_digest"] != document["request_digest"]
    ):
        raise ValueError("Worker belongs to another Program admission")
    for relative in ("game", "home", "work", "work/hockeymon-session", "work/hockeymon-output"):
        descriptor = _open_beneath(root, relative, directory=True)
        os.close(descriptor)
    expected_paths = {"request": root / REQUEST_NAME, "game_config": root / "game/config.yaml"}
    expected_paths.update(
        {
            "native_" + key: Path(document["native_root"]) / value
            for key, value in NATIVE_FILES.items()
        }
    )
    for role, sources in document["sources"].items():
        for index, source in enumerate(sources):
            expected_paths[f"source_{role}_{index}"] = Path(source["path"])
            if (root / source["alias"]).resolve(strict=True) != Path(source["path"]):
                raise ValueError("Worker source alias changed")
    if document["calibration"] is not None:
        directory = root / document["calibration"]["directory"]
        expected_paths["calibration"] = directory / "native-calibration.yaml"
        import yaml

        manifest_pin = PinnedFile(directory, "native-calibration.yaml")
        try:
            manifest = yaml.safe_load(manifest_pin.read(16 * 1024 * 1024))
        finally:
            manifest_pin.close()
        for name in manifest["artifacts"]:
            if len(_parts(name)) != 1:
                raise ValueError("Invalid native calibration artifact")
            expected_paths["calibration_" + name] = directory / name
    if set(admission["files"]) != set(expected_paths):
        raise ValueError("Worker admission contains unexpected file bindings")
    for name, path in expected_paths.items():
        row = admission["files"][name]
        _object(
            row, {"path", "device", "inode", "size", "mtime_ns", "ctime_ns"}, "local file binding"
        )
        if row["path"] != str(path):
            raise ValueError("Worker admission changed a pinned path")
        current = PinnedFile(path.parent, path.name)
        try:
            if current.identity != {key: row[key] for key in current.identity}:
                raise ValueError("Worker input changed after producer admission")
        finally:
            current.close()
    if {key: admission["files"]["request"][key] for key in request_identity} != request_identity:
        raise ValueError("Worker request was replaced during validation")
    return root, document


def verify_worker_environment() -> None:
    root, document = worker_binding()
    if canonical_bytes(runtime_provenance()) != canonical_bytes(document["runtime"]):
        raise ValueError("Worker installed runtime changed after producer admission")
    expected = {
        "HOME": root / "home",
        "HM_GAME_DIR": root,
        "HM_UI_SESSION_DIR": root / "work/hockeymon-session",
        "HSTREAM_UI_BACKEND_SESSION": root / "work/hockeymon-session",
        "HM_OUTPUT_WORK_DIR": root / "work/hockeymon-output",
    }
    if any(os.environ.get(key) != str(path) for key, path in expected.items()) or os.environ.get(
        "HM_UI_INITIAL_LAUNCH"
    ):
        raise ValueError("Worker private environment differs from admission")
    work = Path(os.environ["HM_SHARED_WORK_DIR"])
    if not work.resolve(strict=True).is_relative_to(expected["HM_OUTPUT_WORK_DIR"]):
        raise ValueError("Worker working storage escapes its private output root")


def validate_publication_directory(directory: Path, generation_directories) -> None:
    root, _ = worker_binding()
    expected = root / "game"
    if (
        directory != expected
        or directory.resolve(strict=True) != expected
        or any(Path(path) != expected for path in generation_directories)
    ):
        raise ValueError("Web Program publication must stay in its private game directory")


def publication_records(records: list[dict]) -> list[dict]:
    """Capture exact committed identities after publisher staging links retire."""
    root, _ = worker_binding()
    result = []
    for record in records:
        path = Path(record["path"])
        if path.parent != root / "game":
            raise ValueError("Published artifact escaped its private game directory")
        kind = record.get("kind", "metadata")
        if kind == "metadata" and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".db", ".sqlite"}:
            raise ValueError("Published media requires its producer's logical kind")
        pinned = PinnedFile(root / "game", path.name)
        try:
            os.fsync(pinned.fd)
            result.append({**record, "kind": kind, "identity": pinned.identity})
        finally:
            pinned.close()
    return result


def read_publications(root: Path) -> list[dict]:
    relative = "work/hockeymon-session/publication.json"
    if not os.path.lexists(root / relative):
        return []
    pinned = PinnedFile(root, relative)
    try:
        result = json.loads(pinned.read(MAX_RECEIPT_BYTES))
    finally:
        pinned.close()
    if not isinstance(result, list):
        raise TypeError("Producer publication journal must be a list")
    return result
