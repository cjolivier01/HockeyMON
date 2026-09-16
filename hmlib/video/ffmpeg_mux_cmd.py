from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import List, Optional


def build_ffmpeg_raw_bitstream_mux_cmd(
    *,
    ffmpeg: str,
    bitstream_path: Path,
    output_path: Path,
    stream_format: str,
    fps: float,
    muxer: Optional[str],
    audio_file: Optional[str] = None,
    audio_stream: int = 0,
    audio_offset_seconds: float = 0.0,
    audio_codec: Optional[str] = None,
    aac_bitrate: str = "192k",
) -> List[str]:
    """Build an ffmpeg command to mux a raw elementary bitstream into a container.

    This is intentionally a pure function so it can be unit-tested without
    importing torch/PyNvVideoCodec (which may not be available in all test
    environments).
    """
    fps_frac = Fraction(float(fps)).limit_denominator(1001)
    fps_str = (
        f"{fps_frac.numerator}/{fps_frac.denominator}"
        if fps_frac.denominator != 1
        else str(fps_frac.numerator)
    )

    cmd: List[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-progress",
        "pipe:2",
        "-nostats",
        "-f",
        str(stream_format),
        "-framerate",
        fps_str,
        "-i",
        str(bitstream_path),
    ]

    if audio_file:
        # Apply an optional offset to the audio timeline (relative to video).
        if abs(float(audio_offset_seconds)) > 1e-9:
            cmd += ["-itsoffset", str(float(audio_offset_seconds))]
        cmd += ["-i", str(audio_file)]
        cmd += ["-map", "0:v:0", "-map", f"1:a:{int(audio_stream)}"]

    cmd += ["-c:v", "copy"]

    if audio_file:
        # If the audio codec is already AAC, stream copy; otherwise re-encode to AAC.
        if (audio_codec or "").lower() == "aac":
            cmd += ["-c:a", "copy"]
        else:
            cmd += [
                "-c:a",
                "aac",
                "-b:a",
                str(aac_bitrate),
                "-ac",
                "2",
                "-ar",
                "48000",
            ]
        cmd.append("-shortest")

    if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        # Make MP4/MOV outputs friendlier for Apple/iPhone playback and progressive download.
        cmd += ["-movflags", "+faststart"]
        if stream_format == "hevc":
            cmd += ["-tag:v", "hvc1"]
        elif stream_format == "av1":
            cmd += ["-tag:v", "av01"]

    if muxer:
        cmd += ["-f", str(muxer)]

    cmd.append(str(output_path))
    return cmd


def build_ffmpeg_live_bitstream_publish_cmd(
    *,
    ffmpeg: str,
    output_url: str,
    stream_format: str,
    fps: float,
    aac_bitrate: str = "128k",
) -> List[str]:
    """Build an ffmpeg command to publish a raw elementary bitstream live.

    This is used for NVENC-backed RTMP(S) publishing where encoded H.264
    packets are produced in-memory and piped directly into ffmpeg for muxing
    and transport.
    """
    fps_frac = Fraction(float(fps)).limit_denominator(1001)
    fps_str = (
        f"{fps_frac.numerator}/{fps_frac.denominator}"
        if fps_frac.denominator != 1
        else str(fps_frac.numerator)
    )
    time_base = Fraction(fps_frac.denominator, fps_frac.numerator)
    time_base_str = f"{time_base.numerator}/{time_base.denominator}"
    setts_bsf = f"setts=pts=N:dts=N:duration=1:time_base={time_base_str}"

    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-progress",
        "pipe:2",
        "-nostats",
        "-nostdin",
        "-f",
        str(stream_format),
        "-framerate",
        fps_str,
        "-i",
        "-",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-bsf:v",
        setts_bsf,
        "-c:a",
        "aac",
        "-b:a",
        str(aac_bitrate),
        "-ac",
        "2",
        "-ar",
        "48000",
        "-shortest",
        "-f",
        "flv",
        str(output_url),
    ]


def build_ffmpeg_live_hls_bitstream_publish_cmd(
    *,
    ffmpeg: str,
    output_manifest_path: Path,
    segment_path_pattern: Path,
    stream_format: str,
    fps: float,
    hls_time: float = 1.0,
    hls_list_size: int = 6,
) -> List[str]:
    """Build an ffmpeg command to mux a raw elementary bitstream into live HLS."""
    fps_frac = Fraction(float(fps)).limit_denominator(1001)
    fps_str = (
        f"{fps_frac.numerator}/{fps_frac.denominator}"
        if fps_frac.denominator != 1
        else str(fps_frac.numerator)
    )
    time_base = Fraction(fps_frac.denominator, fps_frac.numerator)
    time_base_str = f"{time_base.numerator}/{time_base.denominator}"
    setts_bsf = f"setts=pts=N:dts=N:duration=1:time_base={time_base_str}"

    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-progress",
        "pipe:2",
        "-nostats",
        "-nostdin",
        "-f",
        str(stream_format),
        "-framerate",
        fps_str,
        "-i",
        "-",
        "-an",
        "-c:v",
        "copy",
        "-bsf:v",
        setts_bsf,
        "-f",
        "hls",
        "-hls_time",
        str(float(hls_time)),
        "-hls_list_size",
        str(int(hls_list_size)),
        "-hls_allow_cache",
        "0",
        "-hls_flags",
        "delete_segments+append_list+independent_segments+omit_endlist+temp_file",
        "-hls_segment_filename",
        str(segment_path_pattern),
        "-start_number",
        "0",
        str(output_manifest_path),
    ]
