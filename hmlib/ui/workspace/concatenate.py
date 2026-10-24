"""Concatenate imported chapters without modifying the original recordings."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .files import config_lock, read_config, revision, set_path, write_config


def concatenate(request: dict, work: Path) -> dict:
    game = Path(request["game"])
    paths = [Path(value).resolve(strict=True) for value in request["paths"]]
    signatures = []
    for path in paths:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(probe.stdout)["streams"]
        signatures.append(
            [
                {
                    key: stream.get(key)
                    for key in (
                        "codec_type",
                        "codec_name",
                        "profile",
                        "width",
                        "height",
                        "pix_fmt",
                        "sample_rate",
                        "channels",
                        "channel_layout",
                        "time_base",
                        "r_frame_rate",
                        "extradata_size",
                    )
                }
                for stream in streams
                if stream.get("codec_type") in {"video", "audio"}
            ]
        )
    if not signatures[0] or any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            "Chapters must have matching video/audio codecs, dimensions, and frame rates"
        )
    aliases = []
    for index, path in enumerate(paths):
        alias = work / f"chapter-{index:04d}{path.suffix.lower()}"
        alias.symlink_to(path)
        aliases.append(f"file '{alias.name}'\n")
    listing = work / "chapters.txt"
    listing.write_text("".join(aliases))
    output = work / "joined.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(listing),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    role = request["role"]
    folder = game / "joined"
    folder.mkdir(exist_ok=True)
    if folder.is_symlink():
        raise ValueError("Joined video directory cannot be a symbolic link")
    with config_lock(game):
        if revision(game) != request["source_revision"]:
            raise RuntimeError(
                "Game configuration changed during concatenation; the completed file remains in the job directory"
            )
        config = read_config(game)
        # Exclusive creation and the config lock protect independent desktop sessions.
        index = 1
        while True:
            destination = folder / f"{role}-{index}.mp4"
            try:
                stream = destination.open("xb")
                break
            except FileExistsError:
                index += 1
        try:
            with stream, output.open("rb") as source:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
                stream.flush()
                import os

                os.fsync(stream.fileno())
            set_path(config, f"game.videos.{role}", [str(destination.relative_to(game))])
            write_config(game, config)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return {"path": str(destination), "role": role, "chapters": [str(path) for path in paths]}
