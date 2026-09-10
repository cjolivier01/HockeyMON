"""Camera-policy provenance and training boundaries without changing source frame IDs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

POLICY_SCHEMA = "hm-camera-policy-v1"


def camera_policy_path(camera_csv: str | Path) -> Path:
    """Pair a camera CSV with its policy file, preserving output labels/suffixes."""
    path = Path(camera_csv)
    match = re.fullmatch(r"(.*?)(-\d+)?", path.stem)
    assert match is not None
    return path.with_name(f"{match[1]}_policy{match[2] or ''}.csv")


class CameraPolicyRecorder:
    """Freeze the effective policy immediately before each source frame uses it."""

    def __init__(self) -> None:
        self._previous: Optional[dict[str, Any]] = None

    def record(self, frame: int, policy: dict[str, Any]) -> Optional[dict[str, Any]]:
        # Round-trip before comparison so later in-place UI edits cannot change
        # earlier events; numeric 1 and 1.0 remain equal for no-op controls.
        frozen = json.loads(json.dumps(policy, allow_nan=False, sort_keys=True))
        if frozen == self._previous:
            return None
        event = {
            "frame": int(frame),
            "schema": POLICY_SCHEMA,
            "kind": "startup" if self._previous is None else "change",
            "policy": frozen,
        }
        self._previous = frozen
        return event


def read_camera_policy_boundaries(camera_csv: str | Path, camera_frames: Iterable[int]) -> set[int]:
    """Read an optional companion; malformed/incomplete provenance is an error."""
    path = camera_policy_path(camera_csv)
    try:
        stream = path.open(encoding="utf-8", newline="")
    except FileNotFoundError:
        if path.is_symlink():
            raise ValueError(f"Broken camera policy companion: {path}") from None
        return set()  # Legacy exports have no policy companion.
    boundaries: list[int] = []
    with stream:
        for row in csv.reader(stream):
            if len(row) != 2:
                raise ValueError(f"Invalid camera policy CSV row in {path}")
            try:
                frame = int(row[0])
                event = json.loads(row[1])
                valid = (
                    frame >= 0
                    and isinstance(event, dict)
                    and event.get("schema") == POLICY_SCHEMA
                    and event.get("kind") == ("change" if boundaries else "startup")
                    and isinstance(event.get("policy"), dict)
                    and (not boundaries or frame > boundaries[-1])
                )
                # Reject nonfinite values accepted by Python's JSON decoder.
                json.dumps(event, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid camera policy event in {path}") from error
            if not valid:
                raise ValueError(f"Invalid camera policy event in {path}")
            boundaries.append(frame)
    frames = set(int(frame) for frame in camera_frames)
    if not boundaries or not frames or boundaries[0] != min(frames):
        raise ValueError(f"Camera policy startup does not match camera CSV: {path}")
    if not set(boundaries).issubset(frames):
        raise ValueError(f"Camera policy event has no matching camera frame: {path}")
    return set(boundaries)
