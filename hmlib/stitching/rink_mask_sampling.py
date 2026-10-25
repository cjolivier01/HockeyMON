"""Final-mask frame selection without seeking the active playback streams."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import cv2
import torch
import yaml

from hmlib.video.ffmpeg import BasicVideoInfo

_TIME = re.compile(r"(-?)(\d{1,6})(?::(\d{1,2}))?(?::(\d{1,2}))?(?:\.(\d{1,3}))?\Z")


def parse_mask_frame_time(value: str) -> float | None:
    if value == "auto":
        return None
    if not isinstance(value, str) or (match := _TIME.fullmatch(value)) is None:
        raise ValueError("Rink mask time must be auto or [-]SS, [-]MM:SS, [-]HH:MM:SS[.mmm]")
    sign, first, second, third, fraction = match.groups()
    fields = [int(part) for part in (first, second, third) if part is not None]
    if (len(fields) > 1 and fields[-1] > 59) or (len(fields) == 3 and fields[1] > 59):
        raise ValueError("Rink mask time minutes/seconds after a colon must be below 60")
    seconds = sum(value * 60**index for index, value in enumerate(reversed(fields)))
    seconds += int((fraction or "").ljust(3, "0")) / 1000
    if seconds * 1e9 > 2**63 - 1 or (sign and seconds == 0):
        raise ValueError("Rink mask time is overflowing or negative zero")
    return -seconds if sign else seconds


@lru_cache(maxsize=1)
def _mask_baseline():
    with (Path(__file__).resolve().parents[1] / "config" / "baseline.yaml").open() as stream:
        return yaml.safe_load(stream)["stitching"]


def read_mask_frame_time(config) -> str:
    stitch = config.get("stitching") or {}
    value = stitch.get("rink_mask_frame_time")
    if value is None:
        name = stitch.get("rink_config")
        profile = dict((_mask_baseline().get("rink_configs") or {}).get(name) or {})
        profile.update((stitch.get("rink_configs") or {}).get(name) or {})
        value = profile.get("rink_mask_frame_time")
    if value is None:
        value = stitch.get("rink_mask_frame_time_default")
    if value is None:
        baseline = _mask_baseline()
        value = baseline.get("rink_mask_frame_time_default", "auto")
        if value is None:
            value = "auto"
    seconds = parse_mask_frame_time(value)
    if seconds is None:
        return "auto"
    milliseconds = round(abs(seconds) * 1000)
    whole, fraction = divmod(milliseconds, 1000)
    return (
        ("-" if seconds < 0 else "")
        + f"{whole//3600:02}:{whole//60%60:02}:{whole%60:02}"
        + (f".{fraction:03}" if fraction else "")
    )


def resolve_mask_frame_time(value: str, duration: float) -> float | None:
    offset = parse_mask_frame_time(value)
    if offset is None:
        return None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Mask frame time requires a known nonempty synchronized recording")
    seconds = duration + offset if offset < 0 else offset
    if not 0 <= seconds < duration:
        raise ValueError("Mask frame time is outside the synchronized recording")
    return seconds


def source_playlists(config, game_dir: Path, *, source_video_paths=None, single_source=False):
    configured = (config.get("game") or {}).get("videos") or {}
    roles = (config.get("hstream_ui") or {}).get("video_roles") or {}
    result = {}
    declared = False
    for role in ("left", "right", "stitched", "source"):
        if role in roles or role in configured:
            values = roles[role] if role in roles else configured[role]
            if values is None:
                continue  # The baseline declares unconfigured roles as null.
            declared = True
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"Source playlist {role} must contain paths")
            if values:
                result[role] = [(game_dir / value).resolve(strict=True) for value in values]
    if declared:
        # Empty assignments are removals. Never rediscover recordings the
        # operator removed or resurrect an older configured assignment.
        return result
    if single_source and source_video_paths:
        return {"source": [Path(value).resolve(strict=True) for value in source_video_paths]}
    from hmlib.orientation import get_available_videos

    discovered = get_available_videos(str(game_dir))
    for role in ("left", "right", "stitched", "source"):
        if role in discovered:
            result[role] = [Path(path) for _, path in sorted(discovered[role].items())]
    if result:
        return result
    cameras = [chapters for key, chapters in discovered.items() if key != "stitched" and chapters]
    if single_source and len(cameras) == 1:
        return {"source": [Path(path) for _, path in sorted(cameras[0].items())]}
    if len(cameras) == 2:
        return {
            role: [Path(path) for _, path in sorted(chapters.items())]
            for role, chapters in zip(("left", "right"), cameras, strict=True)
        }
    raise ValueError("Mask sampling needs a stitched playlist or both selected source cameras")


def _playlist_info(paths):
    result = []
    for path in paths:
        info = BasicVideoInfo(str(path))
        fps, count = float(info.fps), int(info.frame_count)
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Cannot determine recording duration: {path}")
        result.append((path, fps, count))
    return result


def _decode_at(playlist, seconds):
    for path, fps, count in playlist:
        duration = count / fps
        if seconds >= duration:
            seconds -= duration
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or not capture.set(
                cv2.CAP_PROP_POS_FRAMES, min(count - 1, math.floor(seconds * fps + 1e-7))
            ):
                raise OSError(f"Cannot seek mask reference frame: {path}")
            valid, frame = capture.read()
            if not valid or frame is None:
                raise OSError(f"Cannot decode mask reference frame: {path}")
            return torch.from_numpy(frame)  # Playback and its color transforms use BGR.
        finally:
            capture.release()
    raise ValueError("Mask reference frame lies beyond the source playlist")


def sample_rink_mask_frame(context, first_frame):
    shared = context.get("shared") or {}
    config = shared.get("game_config") or context.get("game_config")
    if config is None:
        # Replay of telemetry does not need the source videos again; its recorded
        # mask remains bound to the telemetry geometry.
        return first_frame, "auto"
    preference = read_mask_frame_time(config)
    if preference == "auto":
        return first_frame, preference
    from hmlib.config import get_game_dir

    game_id = context.get("game_id") or shared.get("game_id")
    directory = Path(shared.get("game_dir") or get_game_dir(game_id=game_id))
    renderer = context.get("rink_mask_renderer") or shared.get("rink_mask_renderer")
    geometry = context.get("camera_input_geometry") or shared.get("camera_input_geometry") or {}
    single_source = renderer is None and geometry.get("stitched_geometry_revision") is None
    playlists = source_playlists(
        config,
        directory,
        source_video_paths=shared.get("source_video_paths"),
        single_source=single_source,
    )
    if single_source and "source" in playlists:
        roles = ("source",)
    elif "left" in playlists and "right" in playlists:
        roles = ("left", "right")
    elif "stitched" in playlists:
        roles = ("stitched",)
    else:
        raise ValueError("Select both camera playlists for synchronized mask sampling")
    info = {role: _playlist_info(playlists[role]) for role in roles}
    offsets = (config.get("stitching") or {}).get("frame_offsets") or {}
    starts = {
        role: float(offsets.get(role) or 0) / info[role][0][1] if role in ("left", "right") else 0.0
        for role in roles
    }
    if any(not math.isfinite(value) or value < 0 for value in starts.values()):
        raise ValueError("Invalid source frame offsets for mask sampling")
    duration = min(
        sum(count / fps for _, fps, count in info[role]) - starts[role] for role in roles
    )
    seconds = resolve_mask_frame_time(preference, duration)
    frames = [_decode_at(info[role], seconds + starts[role]) for role in roles]
    if len(frames) == 2:
        if renderer is None:
            raise ValueError("Timed rink mask sampling requires the current stitched renderer")
        frame = renderer(frames)
    else:
        frame = frames[0]
    if torch.is_tensor(first_frame) and torch.is_tensor(frame):
        original_size = tuple(
            first_frame.shape[:2] if first_frame.shape[-1] in (3, 4) else first_frame.shape[-2:]
        )
        sampled_size = tuple(frame.shape[:2] if frame.shape[-1] in (3, 4) else frame.shape[-2:])
        if original_size != sampled_size and len(frames) == 1:
            # MOTLoadVideoWithOrig keeps the full frame even when detector HmCrop
            # is active. Some callers supply the cropped plane instead; apply
            # only their explicit crop, never resize an unrelated coordinate plane.
            crop = shared.get("original_clip_box")
            if crop is None:
                crop = context.get("original_clip_box")
            if crop is not None:
                left, top, right, bottom = crop
                frame = frame[top:bottom, left:right]
                sampled_size = tuple(frame.shape[:2])
        if original_size != sampled_size:
            raise ValueError("Timed mask frame does not match the active stitched coordinate plane")
    return frame, preference
