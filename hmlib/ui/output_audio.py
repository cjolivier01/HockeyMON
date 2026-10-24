"""Complete requested recording audio even when the CPU writer cannot mux it."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path


def ensure_archive_audio(
    path: Path, *, source: str, stream: int = 0, offset_seconds: float = 0, bitrate: str = "192k"
) -> None:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if json.loads(probe.stdout).get("streams"):
        return
    temporary = path.with_name(f".{path.stem}-audio-{uuid.uuid4().hex}{path.suffix}")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(path)]
    if offset_seconds:
        command += (
            ["-ss", str(-float(offset_seconds))]
            if offset_seconds < 0
            else ["-itsoffset", str(float(offset_seconds))]
        )
    command += [
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        f"1:a:{stream}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        "-ac",
        "2",
        "-ar",
        "48000",
        "-shortest",
    ]
    if path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        command += ["-movflags", "+faststart"]
    command.append(str(temporary))
    try:
        subprocess.run(command, check=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_chapter_audio(paths: list[Path], work_dir: Path, *, stream: int = 0):
    """Select real audio, joining chapters in scratch storage without source edits."""
    import tempfile

    if not paths:
        raise ValueError("Audio selection requires source recordings")
    available = []
    for path in paths:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        count = len(json.loads(result.stdout).get("streams", []))
        available.append(count > stream)
    if not any(available):
        return None, None
    if not all(available):
        raise ValueError(
            "Source chapters disagree on the selected audio stream; select a complete audio source or disable audio"
        )
    if len(paths) == 1:
        return str(paths[0]), None
    temporary = tempfile.TemporaryDirectory(prefix=".source-audio-", dir=work_dir)
    directory = Path(temporary.name)
    try:
        lines = []
        for index, path in enumerate(paths):
            alias = directory / f"chapter-{index:04d}{path.suffix.lower()}"
            alias.symlink_to(path.resolve())
            lines.append(f"file '{alias.name}'\n")
        listing = directory / "chapters.txt"
        listing.write_text("".join(lines))
        target = directory / "source.m4a"
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
                "-vn",
                "-map",
                f"0:a:{stream}",
                "-c:a",
                "aac",
                "-ac",
                "2",
                "-ar",
                "48000",
                str(target),
            ],
            check=True,
        )
        return str(target), temporary
    except BaseException:
        temporary.cleanup()
        raise
