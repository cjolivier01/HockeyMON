"""Validated local commands for changing network routes during playback."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

NETWORK_KEYS = {
    "show_youtube",
    "youtube_stream_url",
    "youtube_stream_key",
    "spare_rtmp_enabled",
    "spare_rtmp_url",
    "rtsp_enabled",
    "rtsp_host",
    "rtsp_port",
    "rtsp_program_mount",
    "rtsp_stitched_mount",
    "rtsp_mounts",
}
ARCHIVE_KEYS = {"archive_program", "archive_stitched", "archive_program_4k"}


def validate_changes(changes: dict) -> dict:
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Output changes must be a nonempty mapping")
    normalized = {}
    for original, value in changes.items():
        if not isinstance(original, str):
            raise TypeError("Output setting names must be strings")
        key = original.removeprefix("video_out.")
        if key not in NETWORK_KEYS | ARCHIVE_KEYS:
            raise ValueError(f"This setting cannot change live: {original}")
        if key in {"show_youtube", "spare_rtmp_enabled", "rtsp_enabled"} | ARCHIVE_KEYS:
            if not isinstance(value, bool):
                raise TypeError(f"{key} requires a boolean")
        elif key == "rtsp_port":
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError("RTSP port must be an integer between 1 and 65535")
        elif key == "rtsp_mounts":
            if value is not None:
                if not isinstance(value, list) or not 1 <= len(value) <= 16:
                    raise ValueError("RTSP requires between one and sixteen mounts")
                seen = set()
                for item in value:
                    if not isinstance(item, dict):
                        raise TypeError("RTSP mount definitions must be mappings")
                    mount = item.get("mount")
                    if not isinstance(mount, str) or not re.fullmatch(
                        r"/[A-Za-z0-9_/-]{1,100}", mount
                    ):
                        raise ValueError("Invalid RTSP mount path")
                    if mount in seen or item.get("source", "program") not in {
                        "program",
                        "stitched",
                    }:
                        raise ValueError(
                            "RTSP mounts require unique paths and a program or stitched source"
                        )
                    seen.add(mount)
        else:
            if not isinstance(value, str) or "\0" in value or len(value) > 8192:
                raise ValueError(f"{key} requires bounded text without NUL characters")
            if (
                key.endswith("_mount")
                and value
                and not re.fullmatch(r"/[A-Za-z0-9_/-]{1,100}", value)
            ):
                raise ValueError("Invalid RTSP mount path")
            if key in {"youtube_stream_url", "spare_rtmp_url"} and value:
                parsed = urlsplit(value)
                if parsed.scheme not in {"rtmp", "rtmps"} or not parsed.hostname:
                    raise ValueError("Publish URLs must use rtmp:// or rtmps://")
        normalized[key] = value
    return normalized


def public_config(config: dict) -> dict:
    # Ingest URLs and stream keys are credentials; the output status only needs
    # enable switches and the local RTSP endpoints.
    return {
        key: value
        for key, value in config.items()
        if key in NETWORK_KEYS | ARCHIVE_KEYS
        and key not in {"youtube_stream_key", "youtube_stream_url", "spare_rtmp_url"}
    }
