"""Closed, bounded web control vocabulary; safe for the stdlib-only handshake."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import uuid

from hmlib.ui.hstream_web import canonical_bytes

PREFIX = "@web-control "
MAX_CONTROL_BYTES = 16 * 1024
MAX_INTEGER = 2**53 - 1
ACTIONS = (
    "pause",
    "resume",
    "step",
    "stop",
    "rate",
    "seek",
    "preview",
    "preview-overlays",
    "properties",
)
PROPERTY_NAMES = {
    "dsplaytracker0": {
        "fixed-edge-rotation-angle",
        "fixed-edge-rotation-angle-left",
        "fixed-edge-rotation-angle-right",
    },
    "playcropper0": {
        "fixed-edge-rotation-angle",
        "fixed-edge-rotation-angle-left",
        "fixed-edge-rotation-angle-right",
        "scoreboard-position",
        "exposure",
        "shadow-lift",
        "shadow-lift-black-point",
    },
    "hmstitcher0": {"exposure", "shadow-lift", "shadow-lift-black-point"},
}


def _fields(value, fields: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Invalid web control fields")


def decode_request(line: str) -> dict:
    """Decode an exact canonical envelope; malformed identities have no correlatable ack."""
    if not line.startswith(PREFIX) or len(line) > len(PREFIX) + (MAX_CONTROL_BYTES * 4 + 2) // 3:
        raise ValueError("Expected a bounded web control envelope")
    encoded = line[len(PREFIX) :]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
        raise ValueError("Web control must use unpadded base64url")
    try:
        data = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        value = json.loads(data)
    except (binascii.Error, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid web control encoding") from error
    if (
        len(data) > MAX_CONTROL_BYTES
        or base64.urlsafe_b64encode(data).decode().rstrip("=") != encoded
        or canonical_bytes(value) != data
    ):
        raise ValueError("Web control must contain bounded canonical JSON")
    _fields(
        value,
        {"version", "run_id", "request_digest", "session_id", "request_id", "action", "payload"},
    )
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported web control version")
    if not isinstance(value["run_id"], str) or str(uuid.UUID(value["run_id"])) != value["run_id"]:
        raise ValueError("Invalid web control run")
    for name, size in (("request_digest", 64), ("session_id", 32)):
        if not isinstance(value[name], str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", value[name]):
            raise ValueError("Invalid web control binding")
    if type(value["request_id"]) is not int or not 0 < value["request_id"] <= MAX_INTEGER:
        raise ValueError("Invalid web control request identity")
    if not isinstance(value["action"], str) or not re.fullmatch(
        r"[a-z][a-z-]{0,63}", value["action"]
    ):
        raise ValueError("Invalid web control action name")
    return value


def validate_payload(action: str, payload: dict) -> None:
    if action not in ACTIONS:
        raise ValueError("Unsupported web control action")
    if action in {"pause", "resume", "step", "stop"}:
        _fields(payload, set())
    elif action == "rate":
        _fields(payload, {"rate"})
        if type(payload["rate"]) not in {int, float} or payload["rate"] not in {
            0,
            0.25,
            0.5,
            1,
            2,
            4,
        }:
            raise ValueError("Playback rate must be 0, 0.25, 0.5, 1, 2 or 4")
    elif action == "seek":
        _fields(payload, {"position_ns"})
        if (
            type(payload["position_ns"]) is not int
            or not 0 <= payload["position_ns"] <= MAX_INTEGER
        ):
            raise ValueError("Seek position must be an exact nonnegative integer")
    elif action == "preview":
        _fields(payload, {"channel"})
        if not isinstance(payload["channel"], str) or payload["channel"] not in {
            "none",
            "program",
            "stitched",
            "source0",
            "source1",
        }:
            raise ValueError("Unknown web preview channel")
    elif action == "preview-overlays":
        _fields(payload, {"players", "play", "rink"})
        if any(type(value) is not bool for value in payload.values()):
            raise ValueError("Preview overlays require three booleans")
    elif action == "properties":
        _fields(payload, {"assignments"})
        assignments = payload["assignments"]
        if not isinstance(assignments, list) or not 1 <= len(assignments) <= 64:
            raise ValueError("Property batch must contain 1 to 64 assignments")
        seen = set()
        for assignment in assignments:
            _fields(assignment, {"element", "property", "value"})
            element, prop, value = (assignment[key] for key in ("element", "property", "value"))
            if (
                not isinstance(element, str)
                or not isinstance(prop, str)
                or prop not in PROPERTY_NAMES.get(element, set())
            ):
                raise ValueError("Unsupported web runtime property")
            if (element, prop) in seen:
                raise ValueError("Duplicate web runtime property")
            seen.add((element, prop))
            if prop == "scoreboard-position":
                if not isinstance(value, str) or value not in {
                    "upper-left",
                    "upper-center",
                    "upper-right",
                }:
                    raise ValueError("Invalid scoreboard position")
            elif (
                type(value) not in {int, float}
                or abs(value) > MAX_INTEGER
                or not math.isfinite(value)
            ):
                raise ValueError("Web runtime property must be a bounded finite number")
