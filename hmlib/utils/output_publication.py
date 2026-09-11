"""Publish complete output generations without exposing partially copied CSVs."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hmlib.utils.finalization import finalize_resources


@dataclass(frozen=True)
class PublishedArtifacts:
    suffix: int
    files: dict[str, Path]


def artifact_name(name: str, suffix: int) -> str:
    path = Path(name)
    if path.name != name or name in ("", ".", ".."):
        raise ValueError(f"Artifact name must be a filename: {name!r}")
    if suffix < 0:
        raise ValueError("Artifact suffix must be nonnegative")
    return name if suffix == 0 else f"{path.stem}-{suffix}{path.suffix}"


def _is_discovery_marker(name: str) -> bool:
    stem = re.sub(r"-\d+$", "", Path(name).stem)
    return stem == "tracking" or stem.endswith("-tracking")


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def _remove_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = _identity(path)
    except FileNotFoundError:
        return
    if current != identity:
        raise RuntimeError(f"Published artifact changed ownership during cleanup: {path}")
    path.unlink()


def _copy_complete(source: Path, target, source_name: str) -> None:
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as input_file:
        before = os.fstat(input_file.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Output artifact is not a regular file: {source}")
        shutil.copyfileobj(input_file, target, length=1024 * 1024)
        after = os.fstat(input_file.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Output artifact changed during publication: {source_name}")
        os.fchmod(target.fileno(), stat.S_IMODE(before.st_mode))
        target.flush()
        os.fsync(target.fileno())


def publish_artifacts(
    sources: Mapping[str, str | Path],
    directory: str | Path,
    *,
    suffix: int = 0,
    exact: bool = False,
) -> PublishedArtifacts:
    """Copy a generation into destination storage and publish tracking last.

    A persistent directory lock serializes suffix selection with other HM
    publishers. Files are fully copied and synced in hidden staging files on
    the destination filesystem, then linked to final names without replacing
    existing entries. Hard links remain within that filesystem, including NFS;
    published files are independent of the working source files.

    ``exact`` reserves exactly the requested suffix and raises on collisions.
    Otherwise numbering starts at one, above every existing tracking/stitched
    video or companion generation, even when some numbers are missing. On
    failure, only names still owned by this attempt are removed; all source
    artifacts remain available for recovery.
    """
    directory = Path(directory)
    names = list(sources)
    for name in names:
        artifact_name(name, suffix)
    if not names:
        return PublishedArtifacts(suffix, {})
    directory.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(
        directory / ".hm-output-publication.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ValueError("Output publication lock is not a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not exact:
            suffix = max(1, suffix)
            patterns = [
                re.compile(
                    rf"^{re.escape(Path(name).stem)}(?:-(\d+))?{re.escape(Path(name).suffix)}$"
                )
                for name in names
            ]
            patterns.extend(
                [
                    re.compile(
                        r"^(?:.*-)?(?:tracking|stitched)_output(?:-with-audio)?"
                        r"(?:-(\d+))?\.(?:mp4|mkv|mov|m4v|avi)(?:\.hstream-pin)?$",
                        re.IGNORECASE,
                    ),
                    re.compile(
                        r"^(?:tracking|detections|camera|camera_fast|hstream_frame_index|"
                        r"hstream_config_events)(?:-(\d+))?\.csv$"
                    ),
                    re.compile(
                        r"^(?:rink_mask_\d+|hstream_telemetry|hstream_replay)"
                        r"(?:-(\d+))?\.(?:png|json|jsonl)$"
                    ),
                ]
            )
            for existing in directory.iterdir():
                for pattern in patterns:
                    match = pattern.fullmatch(existing.name)
                    if match:
                        suffix = max(suffix, int(match.group(1) or 0) + 1)
        candidates = {name: directory / artifact_name(name, suffix) for name in names}
        while any(os.path.lexists(path) for path in candidates.values()):
            if exact:
                raise FileExistsError(f"Output generation {suffix} already exists in {directory}")
            suffix += 1
            candidates = {name: directory / artifact_name(name, suffix) for name in names}

        staged: dict[str, tuple[Path, tuple[int, int]]] = {}
        published: list[tuple[Path, tuple[int, int]]] = []
        committed = False
        try:
            for name in names:
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix=".hm-publish-",
                    suffix=".partial",
                    dir=directory,
                    delete=False,
                ) as output:
                    path = Path(output.name)
                    staged[name] = (path, _identity(path))
                    _copy_complete(Path(sources[name]), output, name)
            ordered = sorted(names, key=lambda name: (_is_discovery_marker(name), name))
            for name in ordered:
                if _is_discovery_marker(name):
                    _sync_directory(directory)
                stage, identity = staged[name]
                destination = candidates[name]
                try:
                    os.link(stage, destination, follow_symlinks=False)
                except OSError as link_error:
                    # NFS can report an error after committing the link. Only
                    # accept that result when it names this exact staged inode.
                    try:
                        linked_identity = _identity(destination)
                    except FileNotFoundError:
                        raise link_error
                    if linked_identity != identity:
                        raise
                published.append((destination, identity))
            _sync_directory(directory)
            committed = True
        finally:
            actions = []
            if not committed:
                # Discovery markers are linked last, so rollback removes them first.
                actions.extend(
                    (
                        f"rollback {path}",
                        lambda path=path, identity=identity: _remove_owned(path, identity),
                    )
                    for path, identity in reversed(published)
                )
            actions.extend(
                (
                    f"staging cleanup {path}",
                    lambda path=path, identity=identity: _remove_owned(path, identity),
                )
                for path, identity in staged.values()
            )
            actions.append(
                (f"sync publication directory {directory}", lambda: _sync_directory(directory))
            )
            finalize_resources(actions, primary_error=sys.exc_info()[1])
        return PublishedArtifacts(suffix, candidates)
    finally:
        os.close(lock_fd)
