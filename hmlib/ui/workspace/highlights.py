"""Portable highlight plans and an offline FFmpeg/Pillow renderer.

Plans retain source-time cue coordinates. Reordering clips or trimming their
edges therefore does not shift an authored event. Audio is normalized per item
and concatenated with the video; cards and silent source clips receive silence.
"""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from .files import identifier


def number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _color(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Colors must be six-digit hexadecimal values, such as #ffffff")
    return value


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\0" in value:
        raise ValueError(f"{label} must contain at most {maximum} characters")
    return value


def normalize_plan(value: dict, game: Path) -> dict:
    if not isinstance(value, dict) or value.get("schema", 2) not in {1, 2}:
        raise ValueError("Highlights require a schema 1 or 2 plan")
    plan = copy.deepcopy(value)
    plan["schema"] = 2
    plan["base_name"] = identifier(plan.get("base_name", "highlights"), "highlight name")
    rows = plan.get("intervals", [])
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("A highlight plan supports at most 1000 items")
    plan["intervals"] = rows
    plan["archive_offset_ms"] = number(plan.get("archive_offset_ms", 0), "Archive origin", 0, 1e12)
    archive = plan.get("archive_path", "")
    if archive:
        archive_path = (game / archive).expanduser().resolve(strict=True)
        if not archive_path.is_file():
            raise ValueError("The selected highlight archive is not a file")
        plan["archive_path"] = str(archive_path)
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Highlight items must be objects")
        row["label"] = _text(row.get("label", ""), "Clip label")
        if row.get("is_card", False):
            card = row.setdefault("card", {})
            if not isinstance(card, dict):
                raise ValueError("Title card must be an object")
            for key in ("heading", "team_a", "team_b", "date", "logo_a", "logo_b"):
                card[key] = _text(card.get(key, "Game Highlights" if key == "heading" else ""), key)
            card["duration_ms"] = number(card.get("duration_ms", 3000), "Card duration", 1, 600000)
            card["color"] = _color(card.get("color", "#ffffff"))
            card["background"] = _color(card.get("background", "#101b2c"))
            card["size"] = number(card.get("size", 0.075), "Card text size", 0.005, 0.4)
            card["weight"] = number(card.get("weight", 800), "Font weight", 100, 1000)
            if card.get("matchup") and (
                not card["team_a"]
                or not card["team_b"]
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", card["date"])
            ):
                raise ValueError("Matchup cards require both team names and a YYYY-MM-DD date")
            for key in ("logo_a", "logo_b"):
                if card[key]:
                    logo = (game / card[key]).expanduser().resolve(strict=True)
                    if not logo.is_file():
                        raise ValueError(f"Missing card logo: {logo}")
                    card[key] = str(logo)
            total += card["duration_ms"]
            continue
        if not archive:
            raise ValueError("Select a published archive for video clips")
        if row.get("event_mode", False):
            event = number(row.get("event_ms"), "Event time", 0, 1e12)
            duration = number(row.get("duration_ms"), "Clip duration", 1, 86400000)
            row["start_ms"] = max(0, event - 0.75 * duration)
            row["end_ms"] = event + 0.25 * duration
        start = number(row.get("start_ms"), "Clip start", 0, 1e12)
        end = number(row.get("end_ms"), "Clip end", start + 1, start + 86400000)
        if start < plan["archive_offset_ms"]:
            raise ValueError("A clip begins before the selected archive's origin")
        total += end - start
        annotations = row.setdefault("annotations", [])
        if not isinstance(annotations, list) or len(annotations) > 64:
            raise ValueError("A clip supports at most 64 annotations")
        for cue in annotations:
            if not isinstance(cue, dict) or cue.get("kind", "arrow") not in {
                "arrow",
                "text",
                "box",
            }:
                raise ValueError("Annotation kind must be arrow, text, or box")
            cue["kind"] = cue.get("kind", "arrow")
            cue["motion"] = cue.get("motion", "fixed")
            if cue["motion"] not in {"fixed", "keyframes", "track"}:
                raise ValueError("Annotation motion must be fixed, keyframes, or track")
            cue["text"] = _text(cue.get("text", ""), "Annotation text")
            cue["start_ms"] = number(cue.get("start_ms", start), "Cue start", 0, 1e12)
            cue["end_ms"] = number(cue.get("end_ms", end), "Cue end", cue["start_ms"] + 1, 1e12)
            cue["color"] = _color(cue.get("color", "#ffff00"))
            for key, default, low, high in (
                ("x", 0.5, 0, 1),
                ("y", 0.5, 0, 1),
                ("dx", 0.0, -1, 1),
                ("dy", 0.12, -1, 1),
                ("size", 0.055, 0.005, 0.5),
                ("thickness", 0.008, 0.001, 0.1),
                ("weight", 900, 100, 1000),
                ("blink_ms", 250, 16, 60000),
            ):
                cue[key] = number(cue.get(key, default), key, low, high)
            positions = cue.setdefault("positions", [])
            if not isinstance(positions, list) or len(positions) > 4000:
                raise ValueError("An annotation supports at most 4000 recorded positions")
            previous = -1
            for position in positions:
                if not isinstance(position, dict):
                    raise TypeError("Annotation positions must be objects")
                timestamp = number(position.get("time_ms"), "Position time", 0, 1e12)
                if timestamp <= previous:
                    raise ValueError(
                        "Annotation positions must have strictly increasing timestamps"
                    )
                previous = timestamp
                for coordinate in ("x", "y"):
                    minimum, maximum = (-2, 2) if cue["motion"] == "track" else (0, 1)
                    number(position.get(coordinate), "Position coordinate", minimum, maximum)
            if cue["motion"] != "fixed" and not positions:
                raise ValueError("Moving annotations require captured or authored positions")
    if total > 86400000:
        raise ValueError("A highlight reel cannot exceed 24 hours")
    return plan


def probe_media(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    video = next(
        (stream for stream in data["streams"] if stream.get("codec_type") == "video"), None
    )
    if video is None:
        raise ValueError(f"No video stream in {path}")
    duration = data.get("format", {}).get("duration", video.get("duration"))
    if duration is None:
        raise ValueError(f"Cannot determine archive duration: {path}")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration": float(duration),
        "audio": any(stream.get("codec_type") == "audio" for stream in data["streams"]),
    }


def _font(size: int, weight: float):
    from PIL import ImageFont

    family = "DejaVuSans-Bold.ttf" if weight >= 600 else "DejaVuSans.ttf"
    return ImageFont.truetype(family, max(1, size))


def _card_image(card: dict, path: Path, width: int, height: int) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), card["background"])
    draw = ImageDraw.Draw(image)
    font = _font(int(height * card["size"]), card["weight"])
    lines = [card["heading"]]
    if card["team_a"] or card["team_b"]:
        lines.append(f"{card['team_a']}  vs  {card['team_b']}")
    if card["date"]:
        lines.append(card["date"])
    spacing = max(8, int(height * card["size"] * 0.4))
    text = "\n".join(lines)
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    draw.multiline_text(
        ((width - (bounds[2] - bounds[0])) / 2, (height - (bounds[3] - bounds[1])) / 2 - bounds[1]),
        text,
        fill=card["color"],
        font=font,
        spacing=spacing,
        align="center",
    )
    for key, center in (("logo_a", 0.15), ("logo_b", 0.85)):
        if card[key]:
            with Image.open(card[key]) as source:
                logo = source.convert("RGBA")
            logo.thumbnail((int(width * 0.18), int(height * 0.3)))
            image.paste(logo, (int(width * center - logo.width / 2), int(height * 0.18)), logo)
    image.save(path)


def _cue_image(cue: dict, path: Path, width: int, height: int) -> tuple[int, int]:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGBA", (width * 3, height * 3))
    draw = ImageDraw.Draw(canvas)
    x, y = width * 1.5, height * 1.5
    thickness = max(1, int(height * cue["thickness"]))
    color = cue["color"]
    if cue["kind"] == "text":
        draw.text(
            (x, y),
            cue["text"],
            fill=color,
            font=_font(int(height * cue["size"]), cue["weight"]),
            anchor="mm",
        )
    elif cue["kind"] == "box":
        dx, dy = max(abs(cue["dx"]), 0.02) * width, max(abs(cue["dy"]), 0.02) * height
        draw.rectangle(
            (x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2), outline=color, width=thickness
        )
    else:
        dx, dy = cue["dx"] * width, cue["dy"] * height
        end = (x + dx, y + dy)
        draw.line(((x, y), end), fill=color, width=thickness)
        angle = math.atan2(dy, dx)
        arm = max(height * cue["size"], thickness * 3)
        draw.polygon(
            (
                end,
                (end[0] - arm * math.cos(angle - 0.5), end[1] - arm * math.sin(angle - 0.5)),
                (end[0] - arm * math.cos(angle + 0.5), end[1] - arm * math.sin(angle + 0.5)),
            ),
            fill=color,
        )
    bounds = canvas.getbbox()
    if bounds is None:
        raise ValueError("Annotation image is empty")
    canvas.crop(bounds).save(path)
    return int(x - bounds[0]), int(y - bounds[1])


def _position(cue: dict, coordinate: str, start_ms: float) -> str:
    positions = cue["positions"] if cue["motion"] != "fixed" else []
    if not positions:
        return f"{cue[coordinate]:.9f}"

    # Linear interpolation includes constant endpoints. Expressions use the
    # clip's local video time; plan positions remain in absolute source time.
    def segment_tree(begin: int, end: int) -> str:
        # Balanced branches bound FFmpeg's expression recursion to log2(N).
        # A linear if-chain cannot parse realistic captured tracks.
        if end - begin > 1:
            middle = (begin + end) // 2
            boundary = (positions[middle]["time_ms"] - start_ms) / 1000
            return f"if(lt(t,{boundary:.9f}),{segment_tree(begin, middle)},{segment_tree(middle, end)})"
        left, right = positions[begin], positions[end]
        left_time = (left["time_ms"] - start_ms) / 1000
        right_time = (right["time_ms"] - start_ms) / 1000
        return f"({left[coordinate]:.9f}+({right[coordinate] - left[coordinate]:.9f})*(t-{left_time:.9f})/{right_time-left_time:.9f})"

    if len(positions) == 1:
        return f"{positions[0][coordinate]:.9f}"
    last_time = (positions[-1]["time_ms"] - start_ms) / 1000
    expression = f"if(lt(t,{last_time:.9f}),{segment_tree(0, len(positions) - 1)},{positions[-1][coordinate]:.9f})"
    first_time = (positions[0]["time_ms"] - start_ms) / 1000
    return f"if(lt(t,{first_time:.9f}),{positions[0][coordinate]:.9f},{expression})"


def render_plan(plan: dict, game: Path, output: Path, *, preview: bool) -> dict:
    plan = normalize_plan(plan, game)
    archive = Path(plan["archive_path"]) if plan.get("archive_path") else None
    media = probe_media(archive) if archive else {"width": 1920, "height": 1080, "audio": False}
    width = int(number(plan.get("width", min(media["width"], 1920)), "Output width", 16, 4096))
    height = int(
        number(
            plan.get("height", round(width * media["height"] / media["width"])),
            "Output height",
            16,
            4096,
        )
    )
    width -= width % 2
    height -= height % 2
    fps = number(plan.get("fps", 30), "Output FPS", 1, 120)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / "segments"
    temporary.mkdir(exist_ok=False)
    parts = []
    elapsed = 0.0
    budget = 30.0 if preview else 86400.0
    for index, row in enumerate(plan["intervals"]):
        card = bool(row.get("is_card"))
        duration = (
            row["card"]["duration_ms"] / 1000 if card else (row["end_ms"] - row["start_ms"]) / 1000
        )
        duration = min(duration, budget - elapsed)
        if duration <= 0:
            break
        target = temporary / f"segment-{index:04}.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex_threads",
            "1",
        ]
        if card:
            still = temporary / f"card-{index}.png"
            _card_image(row["card"], still, width, height)
            command += ["-loop", "1", "-framerate", str(fps), "-i", str(still)]
        else:
            offset = (row["start_ms"] - plan["archive_offset_ms"]) / 1000
            if offset + duration > media["duration"] + 0.05:
                raise ValueError(f"Clip {index + 1} extends beyond the selected archive")
            command += ["-ss", str(offset), "-i", str(archive)]
        audio = not card and media["audio"]
        if not audio:
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        filters = [
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},setpts=PTS-STARTPTS[v0]"
        ]
        current = "v0"
        input_index = 1 if audio else 2
        for cue_index, cue in enumerate([] if card else row["annotations"]):
            begin = max(0, (cue["start_ms"] - row["start_ms"]) / 1000)
            end = min(duration, (cue["end_ms"] - row["start_ms"]) / 1000)
            if end <= begin:
                continue
            asset = temporary / f"cue-{index}-{cue_index}.png"
            anchor_x, anchor_y = _cue_image(cue, asset, width, height)
            command += ["-loop", "1", "-framerate", str(fps), "-i", str(asset)]
            x = _position(cue, "x", row["start_ms"])
            y = _position(cue, "y", row["start_ms"])
            enable = f"between(t,{begin:.6f},{end:.6f})"
            if cue.get("blink", True):
                blink = cue["blink_ms"] / 1000
                enable += f"*lt(mod(t-{begin:.6f},{blink * 2:.6f}),{blink:.6f})"
            following = f"v{cue_index + 1}"
            filters.append(
                f"[{current}][{input_index}:v]overlay=x='{x}*{width}-{anchor_x}':y='{y}*{height}-{anchor_y}':enable='{enable}':shortest=1[{following}]"
            )
            current = following
            input_index += 1
        audio_source = "0:a" if audio else "1:a"
        filters.append(
            f"[{audio_source}]aresample=48000,apad,atrim=duration={duration},asetpts=PTS-STARTPTS[a]"
        )
        # Captured tracks can exceed the OS per-argument limit even with a
        # balanced expression. Read the graph from a generated local file.
        filter_script = temporary / f"filters-{index}.txt"
        filter_script.write_text(";".join(filters))
        command += [
            "-filter_complex_script",
            str(filter_script),
            "-map",
            f"[{current}]",
            "-map",
            "[a]",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(target),
        ]
        print(f"Rendering highlight item {index + 1}/{len(plan['intervals'])}", flush=True)
        subprocess.run(command, check=True)
        parts.append(target)
        elapsed += duration
    if not parts:
        raise ValueError("The highlight plan has no renderable items")
    concat = temporary / "concat.txt"
    # All segment basenames are generated above; no user strings enter the
    # concat demuxer's directive language.
    concat.write_text("".join(f"file '{part.name}'\n" for part in parts))
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    measured = probe_media(output)
    if abs(measured["duration"] - elapsed) > max(0.25, len(parts) * 0.08):
        raise RuntimeError("Rendered highlight duration does not match the plan")
    result = {
        "path": str(output),
        "duration": measured["duration"],
        "width": width,
        "height": height,
        "preview": preview,
    }
    if not preview:
        from hmlib.utils.output_publication import publish_artifacts

        name = f"{game.name}-{plan['base_name']}.mp4"
        published = publish_artifacts({name: output}, game)
        result.update(path=str(published.files[name]), version=published.suffix)
    return result
