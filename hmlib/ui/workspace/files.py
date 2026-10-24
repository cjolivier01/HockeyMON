"""Small durable file operations shared by desktop workflows."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


def identifier(value: Any, label: str = "identifier") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Invalid {label}: use letters, numbers, dots, underscores or hyphens")
    return value


def atomic_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def read_json(path: Path, maximum: int = 16 * 1024 * 1024) -> Any:
    if path.stat().st_size > maximum:
        raise ValueError(f"JSON document exceeds {maximum} bytes: {path}")
    return json.loads(path.read_text())


def read_config(game: Path) -> dict:
    path = game / "config.yaml"
    if not path.exists():
        return {}
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Game configuration exceeds 16 MiB: {path}")
    # Keep fingerprinted native frame context strings intact when preserving
    # configuration produced by another application.
    from hmlib.config import load_yaml_preserving_player_frame_context

    value = load_yaml_preserving_player_frame_context(path.read_bytes())
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Game configuration must be a mapping: {path}")
    return value


def write_config(game: Path, value: dict) -> None:
    atomic_bytes(game / "config.yaml", yaml.safe_dump(value, sort_keys=False).encode())


def revision(game: Path) -> str:
    path = game / "config.yaml"
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


@contextlib.contextmanager
def config_lock(game: Path) -> Iterator[None]:
    fd = os.open(game / ".hm-workspace-config.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def set_path(config: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid configuration path: {key!r}")
    parent = config
    for part in parts[:-1]:
        if part not in parent:
            parent[part] = {}
        if not isinstance(parent[part], dict):
            raise TypeError(f"Configuration ancestor is not a mapping: {key}")
        parent = parent[part]
    parent[parts[-1]] = value


def file_info(path: Path) -> dict:
    info = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }
