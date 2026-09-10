import subprocess
import tempfile
from pathlib import Path


def should_remux_raw_bitstream_with_timestamps():
    from hmlib.video.py_nv_encoder import PyNvVideoEncoder

    with tempfile.TemporaryDirectory(prefix="hm_ffmpeg_mux_") as td:
        td_path = Path(td)
        raw = td_path / "test.h264"
        out_mkv = td_path / "out.mkv"
        out_mp4 = td_path / "out.mp4"

        # Create a tiny raw H.264 elementary stream (no container timestamps).
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
                "testsrc2=size=320x240:r=5",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "h264",
                str(raw),
            ]
        )
        assert raw.is_file() and raw.stat().st_size > 0

        # Instantiate without calling __init__ (avoids requiring PyNvVideoCodec).
        enc = PyNvVideoEncoder.__new__(PyNvVideoEncoder)
        enc.fps = 5.0
        enc.codec = "h264"
        enc._frames_in_current_bitstream = 5
        enc._ffmpeg_output_handler = None
        enc._mux_audio_file = None
        enc._mux_audio_stream = 0
        enc._mux_audio_offset_seconds = 0.0
        enc._mux_audio_aac_bitrate = "192k"

        enc.output_path = out_mkv
        enc._mux_bitstream_file_with_ffmpeg(raw)
        enc.output_path = out_mp4
        enc._mux_bitstream_file_with_ffmpeg(raw)

        for out in (out_mkv, out_mp4):
            assert out.is_file() and out.stat().st_size > 0
            probe = subprocess.check_output(
                [
                    "ffprobe",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(out),
                ]
            ).decode("utf-8", errors="ignore")
            duration = float(probe.strip())
            assert duration > 0.5

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
