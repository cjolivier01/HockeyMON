"""Durable publication and stable reads of a game's stitching file generation."""

import fcntl
import json
import logging
import os
import shutil
import stat
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

logger = logging.getLogger(__name__)
_JOURNAL = ".stitching-publication.json"
_STAGE_PREFIX = ".stitching-stage-"
_locks_guard = threading.Lock()
_locks: dict[Path, threading.RLock] = {}
_held = threading.local()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _identity(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Stitching artifact must be a regular file: {path}")
    return [info.st_dev, info.st_ino]


def _content(path: Path) -> bytes | None:
    """Read a small publication target without following replacement symlinks."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
            raise ValueError(f"Invalid publication comparison target: {path}")
        payload = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Publication comparison target changed while reading: {path}")
    return payload


def _write_journal(directory: Path, journal: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=_JOURNAL + ".", suffix=".tmp", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(journal, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / _JOURNAL)
        _sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read_journal(directory: Path) -> dict | None:
    path = directory / _JOURNAL
    if not path.exists():
        return None
    _identity(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"Oversized stitching recovery journal: {path}")
    journal = json.loads(path.read_text(encoding="utf-8"))
    stage = journal.get("stage", "")
    if (
        journal.get("version") != 1
        or not stage.startswith(_STAGE_PREFIX)
        or Path(stage).name != stage
        or journal.get("phase") not in ("prepared", "committed")
        or not isinstance(journal.get("entries"), list)
    ):
        raise ValueError(f"Invalid stitching recovery journal: {path}")
    names = set()
    for entry in journal["entries"]:
        name = entry.get("name", "")
        if (
            not name
            or Path(name).name != name
            or name in names
            or (name.startswith(".stitching") and name != ".stitching_artifacts.json")
        ):
            raise ValueError(f"Invalid stitching recovery artifact name: {name!r}")
        names.add(name)
        for key in ("old", "new"):
            value = entry.get(key)
            if value is not None and (
                not isinstance(value, list)
                or len(value) != 2
                or not all(type(number) is int and number >= 0 for number in value)
            ):
                raise ValueError(f"Invalid stitching recovery identity: {entry!r}")
        if entry.get("new") is None:
            raise ValueError(f"Missing new stitching artifact identity: {entry!r}")
    for key in ("guarded",):
        values = journal.get(key, [])
        if (
            not isinstance(values, list)
            or len(set(values)) != len(values)
            or any(value not in names for value in values)
        ):
            raise ValueError(f"Invalid stitching recovery {key}: {path}")
        journal[key] = values
    return journal


def recover_artifacts(directory: Path) -> None:
    """Recover an interrupted publication while the caller owns ``stitching_lock``."""
    directory = Path(directory)
    journal = _read_journal(directory)
    if journal is None:
        return
    stage = directory / journal["stage"]
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError(f"Missing or unsafe stitching recovery directory: {stage}")
    previous = stage / "previous"
    if previous.is_symlink() or not previous.is_dir():
        raise ValueError(f"Missing or unsafe stitching recovery backup: {previous}")
    guarded = set(journal.get("guarded", []))
    for entry in journal["entries"]:
        path = directory / entry["name"]
        current = _identity(path)
        if journal["phase"] == "committed":
            if current != entry["new"]:
                if entry["name"] in guarded:
                    continue
                raise RuntimeError(f"Published stitching artifact changed before recovery: {path}")
            continue
        if current == entry["old"]:
            continue
        if current != entry["new"]:
            if entry["name"] in guarded:
                continue
            raise RuntimeError(f"Stitching recovery refuses to replace an unowned artifact: {path}")
        if entry["old"] is None:
            path.unlink()
        else:
            backup = previous / entry["name"]
            if _identity(backup) != entry["old"]:
                raise RuntimeError(f"Stitching recovery backup is missing or changed: {backup}")
            os.replace(backup, path)
    _sync_directory(directory)
    (directory / _JOURNAL).unlink()
    _sync_directory(directory)
    shutil.rmtree(stage)
    logger.warning("Recovered %s stitching publication in %s", journal["phase"], directory)


@contextmanager
def stitching_lock(directory: str | Path, *, blocking: bool = True) -> Iterator[None]:
    """Serialize loaders/builders across threads and processes; allow nested calls."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _locks_guard:
        lock = _locks.setdefault(directory, threading.RLock())
    if not lock.acquire(blocking=blocking):
        raise BlockingIOError(f"Stitching artifacts are busy: {directory}")
    try:
        held = getattr(_held, "directories", None)
        if held is None:
            held = _held.directories = set()
        if directory in held:
            yield
            return
        descriptor = os.open(
            directory / ".stitching.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "a+") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Stitching lock must be a regular file")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            held.add(directory)
            try:
                recover_artifacts(directory)
                yield
            finally:
                held.remove(directory)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        lock.release()


def publish_artifacts(
    directory: Path,
    stage: Path,
    names: list[str],
    *,
    expected_old_contents: Mapping[str, bytes | None] | None = None,
) -> None:
    """Replace a validated generation, restoring prior files after interrupted writes.

    The caller holds ``stitching_lock``. All files are staged on this filesystem;
    the journal and hardlinked backups are durable before the first replacement.
    """
    directory, stage = Path(directory).resolve(), Path(stage).resolve()
    if stage.parent != directory or not stage.name.startswith(_STAGE_PREFIX):
        raise ValueError("Stitching stage must be an owned directory inside the game directory")
    if not names or len(set(names)) != len(names):
        raise ValueError("Stitching publication requires distinct artifact names")
    expected_old_contents = expected_old_contents or {}
    if set(expected_old_contents) - set(names):
        raise ValueError("Expected publication targets must be part of the generation")
    for name, expected in expected_old_contents.items():
        if _content(directory / name) != expected:
            raise RuntimeError(
                f"Publication target changed while the generation was staged: {directory / name}"
            )
    previous = stage / "previous"
    previous.mkdir()
    entries = []
    for name in names:
        if Path(name).name != name or (
            name.startswith(".stitching") and name != ".stitching_artifacts.json"
        ):
            raise ValueError(f"Invalid stitching artifact name: {name!r}")
        source, target = stage / name, directory / name
        new, old = _identity(source), _identity(target)
        if new is None or source.stat().st_size == 0:
            raise ValueError(f"Empty or missing staged stitching artifact: {source}")
        with source.open("rb") as stream:
            os.fsync(stream.fileno())
        if old is not None:
            os.link(target, previous / name)
            with (previous / name).open("rb") as stream:
                os.fsync(stream.fileno())
        entries.append({"name": name, "old": old, "new": new})
    _sync_directory(previous)
    _sync_directory(stage)
    journal = {
        "version": 1,
        "stage": stage.name,
        "phase": "prepared",
        "entries": entries,
        "guarded": sorted(expected_old_contents),
    }
    try:
        _write_journal(directory, journal)
        for name in names:
            if (
                name in expected_old_contents
                and _content(directory / name) != expected_old_contents[name]
            ):
                raise RuntimeError(
                    f"Publication target changed while the generation was staged: {directory / name}"
                )
            os.replace(stage / name, directory / name)
        _sync_directory(directory)
        journal["phase"] = "committed"
        _write_journal(directory, journal)
    except BaseException:
        try:
            recover_artifacts(directory)
        except Exception as recovery_error:
            logger.error(
                "Stitching recovery also failed; journal retained: %s",
                recovery_error,
                exc_info=True,
            )
        raise
    (directory / _JOURNAL).unlink()
    _sync_directory(directory)
    shutil.rmtree(stage)


@contextmanager
def artifact_stage(directory: str | Path) -> Iterator[Path]:
    """Create a private generation and retain it if a journal needs recovery."""
    directory = Path(directory).resolve()
    with stitching_lock(directory):
        stage = Path(tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=directory))
        try:
            yield stage
        finally:
            if stage.exists() and not (directory / _JOURNAL).exists():
                primary_error = sys.exc_info()[1]
                try:
                    shutil.rmtree(stage)
                except Exception:
                    if primary_error is None:
                        raise
                    logger.exception(
                        "Failed to remove stitching stage %s after %s; retaining evidence",
                        stage,
                        primary_error,
                    )
