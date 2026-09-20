import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest


@pytest.mark.parametrize("codec", ["h264", "hevc"])
@pytest.mark.parametrize("fps", [Fraction(5), Fraction(30000, 1001)])
@pytest.mark.parametrize("with_audio", [False, True])
def should_remux_raw_bitstream_with_timestamps(codec, fps, with_audio):
    from hmlib.video.py_nv_encoder import PyNvVideoEncoder

    with tempfile.TemporaryDirectory(prefix="hm_ffmpeg_mux_") as td:
        td_path = Path(td)
        raw = td_path / f"test.{codec}"
        out_mkv = td_path / "out.mkv"
        out_mp4 = td_path / "out.mp4"

        # Match NVENC's elementary-stream contract: no packet timestamps or
        # frame reordering. HEVC does not infer PTS from -framerate alone.
        frame_count = 12
        codec_args = (
            ["-x265-params", "pools=1:frame-threads=1:log-level=error"] if codec == "hevc" else []
        )
        subprocess.check_call(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=320x240:r={fps}",
                "-frames:v",
                str(frame_count),
                "-c:v",
                "libx265" if codec == "hevc" else "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-bf",
                "0",
                *codec_args,
                "-f",
                codec,
                str(raw),
            ]
        )
        assert raw.is_file() and raw.stat().st_size > 0

        # Instantiate without calling __init__ (avoids requiring PyNvVideoCodec).
        enc = PyNvVideoEncoder.__new__(PyNvVideoEncoder)
        enc.fps = float(fps)
        enc.codec = codec
        enc._frames_in_current_bitstream = frame_count
        enc._ffmpeg_output_handler = None
        enc._mux_audio_file = None
        enc._mux_audio_stream = 0
        enc._mux_audio_offset_seconds = 0.0
        enc._mux_audio_aac_bitrate = "192k"
        if with_audio:
            audio = td_path / "audio.m4a"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=sample_rate=48000",
                    "-t",
                    "3",
                    "-c:a",
                    "aac",
                    str(audio),
                ],
                check=True,
            )
            enc._mux_audio_file = str(audio)

        enc.output_path = out_mkv
        enc._mux_bitstream_file_with_ffmpeg(raw)
        enc.output_path = out_mp4
        enc._mux_bitstream_file_with_ffmpeg(raw)

        for out in (out_mkv, out_mp4):
            assert out.is_file() and out.stat().st_size > 0
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "packet=pts_time,dts_time:stream=time_base",
                        "-of",
                        "json",
                        str(out),
                    ]
                )
            )
            packets = probe["packets"]
            assert len(packets) == frame_count
            tolerance = float(Fraction(probe["streams"][0]["time_base"]))
            # Matroska shifts all streams to accommodate AAC encoder priming.
            # The video cadence must remain exact after that common offset.
            start = float(packets[0]["pts_time"])
            assert 0 <= start <= (1024 / 48000 + tolerance if with_audio else tolerance)
            for index, packet in enumerate(packets):
                assert float(packet["pts_time"]) == pytest.approx(
                    start + index / float(fps), abs=tolerance
                )
                assert packet["pts_time"] == packet["dts_time"]
            if with_audio:
                audio_codec = subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "a:0",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "default=nw=1:nk=1",
                        str(out),
                    ],
                    text=True,
                ).strip()
                assert audio_codec == "aac"

        # MP4 should be faststart'd (moov before mdat) for iPhone-friendly playback/streaming.
        data = out_mp4.read_bytes()
        offset = 0
        moov_off = None
        mdat_off = None
        while offset + 8 <= len(data) and offset < 1024 * 1024:
            size = int.from_bytes(data[offset : offset + 4], byteorder="big")
            typ = data[offset + 4 : offset + 8]
            header = 8
            if size == 1:
                if offset + 16 > len(data):
                    break
                size = int.from_bytes(data[offset + 8 : offset + 16], byteorder="big")
                header = 16
            elif size == 0:
                size = len(data) - offset
            if typ == b"moov":
                moov_off = offset
            elif typ == b"mdat":
                mdat_off = offset
            if size < header:
                break
            offset += size
        assert moov_off is not None and mdat_off is not None
        assert moov_off < mdat_off
