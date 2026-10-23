"""Network routes verified with independent FFmpeg receiving processes."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from hmlib.ui.output_routes import OutputRoutes
from hmlib.ui.workspace.files import read_json, write_json


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process):
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


def probe_output_cadence(path, expected_frames, expected_fps, expected_duration):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    assert int(video["nb_read_frames"]) == expected_frames
    assert float(Fraction(video["avg_frame_rate"])) == pytest.approx(expected_fps)
    assert float(data["format"]["duration"]) == pytest.approx(expected_duration, abs=0.09)
    assert any(stream["codec_type"] == "audio" for stream in data["streams"])
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    )
    assert np.abs(np.frombuffer(decoded.stdout, dtype=np.int16).astype(float)).mean() > 100
    return video


@pytest.mark.parametrize(("source_fps", "output_fps"), [(60, 30), (30, 60)])
@pytest.mark.parametrize(("archive_width", "archive_height"), [(160, 96), (161, 97)])
def should_resample_recordings_without_changing_source_timing_or_audio(
    tmp_path, source_fps, output_fps, archive_width, archive_height
):
    import torch

    from hmlib.aspen.plugins.video_out_plugin import VideoOutPlugin

    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            str(audio),
        ],
        check=True,
    )
    plugin = VideoOutPlugin(
        output_video_path=str(tmp_path / "program.mkv"),
        preview_via_sink=False,
        video_out_device="cpu",
    )
    shared = {
        "game_config": {"video_out": {"output_fps": output_fps, "archive_stitched": True}},
        "fps": source_fps,
        "mux_audio_file": str(audio),
    }
    try:
        # Non-divisible batch sizes exercise phase continuity across batches.
        for start in range(0, source_fps * 2, 7):
            count = min(7, source_fps * 2 - start)
            frame_ids = torch.arange(start + 1, start + count + 1)
            context = {
                "shared": shared,
                "fps": source_fps,
                "img": torch.full((count, 96, 160, 3), 80, dtype=torch.uint8),
                "original_images": torch.full(
                    (count, 3, archive_height, archive_width), 120, dtype=torch.uint8
                ),
                "frame_ids": frame_ids,
                "video_frame_cfg": {
                    "output_frame_width": 160,
                    "output_frame_height": 96,
                    "output_aspect_ratio": 160 / 96,
                },
            }
            plugin.forward(context)
            assert context["fps"] == source_fps and shared["fps"] == source_fps
            assert torch.equal(context["frame_ids"], frame_ids)
    finally:
        plugin.finalize()
    assert plugin._frames_written == source_fps * 2
    for path in plugin.output_paths():
        video = probe_output_cadence(path, output_fps * 2, output_fps, 2)
        expected_size = (archive_width, archive_height) if "stitched" in path.name else (160, 96)
        assert (video["width"], video["height"]) == expected_size
        if expected_size == (161, 97):
            from hmlib.video.video_stream import VideoStreamReader

            assert video["pix_fmt"] == "yuv444p"
            # Odd-raster preflight must choose CPU decoding before entering a
            # native decoder that rejects this raster, including on CUDA hosts.
            reader = VideoStreamReader(str(path), type="pynvcodec", device=torch.device("cuda"))
            try:
                decoded = list(reader)
                assert len(decoded) == output_fps * 2
                assert all(frame.shape == (1, 3, 97, 161) for frame in decoded)
            finally:
                reader.close()
    assert len(plugin.output_paths()) == 2


def should_resample_live_rtmp_at_recording_fps_and_keep_source_audio_timing(tmp_path):
    audio = tmp_path / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            str(audio),
        ],
        check=True,
    )
    url = f"rtmp://127.0.0.1:{free_port()}/live/test"
    received = tmp_path / "received.flv"
    receiver = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-listen",
            "1",
            "-i",
            url,
            "-c",
            "copy",
            str(received),
        ],
        stderr=subprocess.PIPE,
    )
    routes = OutputRoutes(
        {"output_fps": 30}, fps=60, audio={"file": str(audio)}, session_dir=tmp_path / "live"
    )
    frames = np.full((5, 96, 160, 3), [20, 180, 20], dtype=np.uint8)
    try:
        time.sleep(0.5)
        # Activate after one second of source processing. The audio offset must
        # use 60 source frames, not the 30 frames selected for output.
        for _ in range(12):
            routes.write({"img": frames})
        write_json(
            tmp_path / "live/output-actions/start.json",
            {"changes": {"show_youtube": True, "youtube_stream_url": url}},
        )
        for _ in range(24):
            routes.write({"img": frames})
        routes.close()
        _, error = receiver.communicate(timeout=15)
        assert receiver.returncode == 0, error.decode()
        probe_output_cadence(received, 60, 30, 2)
        assert routes.frames_written == 180
    finally:
        routes.close()
        stop(receiver)


def should_keep_live_resampled_segments_on_source_time_across_recording_toggles(tmp_path):
    import torch

    from hmlib.aspen.plugins.video_out_plugin import VideoOutPlugin

    audio = tmp_path / "source.wav"
    timeline = np.arange(3 * 48000) / 48000
    frequencies = np.where(timeline < 1, 440, np.where(timeline < 2, 880, 1320))
    samples = (12000 * np.sin(2 * np.pi * frequencies * timeline)).astype(np.int16)
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 48000, len(samples), "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())
    flags = {"archive_program": True, "archive_stitched": True}
    shared = {
        "game_config": {"video_out": {**flags, "output_fps": 30}},
        "fps": 60,
        "mux_audio_file": str(audio),
    }
    plugin = VideoOutPlugin(
        output_video_path=str(tmp_path / "program.mkv"),
        preview_via_sink=False,
        video_out_device="cpu",
    )
    frame = torch.full((1, 96, 160, 3), 80, dtype=torch.uint8)
    try:
        for index in range(180):
            # Wide binary stripes survive lossy encoding and reveal which
            # source frame became the first frame of each resumed segment.
            for bit in range(8):
                frame[0, :, bit * 20 : (bit + 1) * 20] = 255 if index & (1 << bit) else 0
            if index in (32, 65, 97, 130):
                shared["game_config"]["video_out"].update(
                    {key: index in (65, 130) for key in flags}
                )
            plugin.forward(
                {
                    "shared": shared,
                    "fps": 60,
                    "img": frame,
                    "original_images": frame,
                    "frame_ids": torch.tensor([index + 1]),
                    "video_frame_cfg": {
                        "output_frame_width": 160,
                        "output_frame_height": 96,
                        "output_aspect_ratio": 160 / 96,
                    },
                }
            )
    finally:
        plugin.finalize()
    manifest = read_json(
        next(path for path in plugin.supplementary_paths() if path.suffix == ".json")
    )
    assert manifest["fps"] == 60
    assert manifest["output_fps"] == 30
    assert len(manifest["segments"]) == 6
    for segment in manifest["segments"]:
        start, end = segment["start_frame"], segment["end_frame"]
        assert (start, end) in {(0, 32), (65, 97), (130, 180)}
        assert segment["timeline_offset_seconds"] == pytest.approx(start / 60)
        assert segment["duration_seconds"] == pytest.approx((end - start) / 60)
        count = 25 if start == 130 else 16
        path = next(path for path in plugin.output_paths() if path.name == segment["filename"])
        probe_output_cadence(path, count, 30, (end - start) / 60)
        pixels = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
        stripes = np.frombuffer(pixels, dtype=np.uint8).reshape(96, 8, 20, 3).mean(axis=(0, 2, 3))
        first_source_frame = sum((stripes[bit] > 128) * (1 << bit) for bit in range(8))
        assert 0 <= (first_source_frame - start) / 60 < 1 / 30
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
            check=True,
        )
        decoded = np.frombuffer(result.stdout, dtype=np.int16).astype(float)[2400:12000]
        peak = np.argmax(np.abs(np.fft.rfft(decoded))) * 48000 / len(decoded)
        expected = {0: 440, 65: 880, 130: 1320}[start]
        assert peak == pytest.approx(expected, abs=5)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def should_activate_redirect_and_disable_live_routes_with_source_audio(tmp_path):
    urls = [f"rtmp://127.0.0.1:{free_port()}/live/test" for _ in range(2)]
    audio = tmp_path / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            str(audio),
        ],
        check=True,
    )
    receivers = [
        subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-listen",
                "1",
                "-i",
                url,
                "-c",
                "copy",
                str(tmp_path / f"received-{index}.flv"),
            ],
            stderr=subprocess.PIPE,
        )
        for index, url in enumerate(urls)
    ]
    routes = OutputRoutes({}, fps=10, audio={"file": str(audio)}, session_dir=tmp_path / "live")
    frame = np.full((1, 96, 160, 3), [20, 180, 20], dtype=np.uint8)

    def change(request_id, changes):
        write_json(
            tmp_path / "live" / "output-actions" / f"{request_id}.json", {"changes": changes}
        )
        routes.write({"img": frame})
        result = read_json(tmp_path / "live" / "output-results" / f"{request_id}.json")
        assert result["ok"], result

    try:
        time.sleep(0.5)
        for _ in range(5):
            routes.write({"img": frame})
        for index, url in enumerate(urls):
            change(str(index), {"show_youtube": True, "youtube_stream_url": url})
            for _ in range(9):
                routes.write({"img": frame})
        change("disabled", {"show_youtube": False})
        assert routes.publishers == []
        for index, receiver in enumerate(receivers):
            _, error = receiver.communicate(timeout=15)
            assert receiver.returncode == 0, error.decode()
            path = tmp_path / f"received-{index}.flv"
            frames = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                capture_output=True,
                check=True,
            ).stdout
            assert len(frames) == 10 * 96 * 160 * 3
            samples = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-"],
                capture_output=True,
                check=True,
            ).stdout
            assert np.abs(np.frombuffer(samples, dtype=np.int16).astype(float)).mean() > 100
        status = read_json(tmp_path / "live" / "output-state.json")
        assert not status["config"]["show_youtube"]
        assert "youtube_stream_url" not in status["config"]
    finally:
        routes.close()
        for receiver in receivers:
            stop(receiver)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def should_deliver_program_frames_to_primary_and_spare_rtmp(tmp_path, device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ports = [free_port(), free_port()]
    urls = [f"rtmp://127.0.0.1:{port}/live/test" for port in ports]
    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            str(audio),
        ],
        check=True,
    )
    receivers = []
    routes = None
    try:
        for index, url in enumerate(urls):
            receivers.append(
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-listen",
                        "1",
                        "-i",
                        url,
                        "-map",
                        "0:v:0",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        str(tmp_path / f"received-{index}.rgb"),
                        "-map",
                        "0:a:0",
                        "-f",
                        "s16le",
                        str(tmp_path / f"received-{index}.pcm"),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            )
        time.sleep(0.5)
        routes = OutputRoutes(
            {
                "show_youtube": True,
                "youtube_stream_url": urls[0],
                "spare_rtmp_enabled": True,
                "spare_rtmp_url": urls[1],
            },
            fps=10,
            audio={"file": str(audio)},
        )
        for _ in range(20):
            frames = np.full((1, 96, 160, 3), [20, 180, 20], dtype=np.uint8)
            routes.write(
                {"img": torch.as_tensor(frames, device=device) if device == "cuda" else frames}
            )
        routes.close()
        routes = None
        for index, receiver in enumerate(receivers):
            _, error = receiver.communicate(timeout=15)
            assert receiver.returncode == 0, error.decode()
            decoded = np.frombuffer(
                (tmp_path / f"received-{index}.rgb").read_bytes(), dtype=np.uint8
            )
            assert len(decoded) == 20 * 96 * 160 * 3
            assert decoded.reshape(-1, 3).mean(axis=0)[1] > 150
            samples = np.frombuffer(
                (tmp_path / f"received-{index}.pcm").read_bytes(), dtype=np.int16
            )
            assert np.abs(samples.astype(float)).mean() > 100
    finally:
        for receiver in receivers:
            stop(receiver)
        if routes:
            routes.close()


@pytest.mark.skipif(
    not Path("/usr/bin/python3").exists() or shutil.which("ffmpeg") is None,
    reason="System Python GI and FFmpeg required",
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def should_serve_distinct_program_and_stitched_rtsp_mounts(tmp_path, device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    probe = subprocess.run(
        ["/usr/bin/python3", "-c", "import gi; gi.require_version('GstRtspServer','1.0')"],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        pytest.skip("System Python lacks GstRtspServer")
    port = free_port()
    routes = OutputRoutes(
        {
            "rtsp_enabled": True,
            "rtsp_port": port,
            "rtsp_program_mount": "/program",
            "rtsp_stitched_mount": "",
        },
        fps=10,
        session_dir=tmp_path / "live",
    )
    receivers = []
    try:
        context = {
            "img": torch.as_tensor(
                np.full((1, 96, 160, 3), [20, 180, 20], dtype=np.uint8), device=device
            ),
            "original_images": torch.as_tensor(
                np.full((1, 120, 200, 3), [20, 20, 180], dtype=np.uint8), device=device
            ),
        }
        write_json(
            tmp_path / "live" / "output-actions" / "mounts.json",
            {
                "changes": {
                    "rtsp_mounts": [
                        {"mount": "/program", "source": "program"},
                        {"mount": "/stitched", "source": "stitched"},
                    ]
                }
            },
        )
        routes.write(context)
        assert read_json(tmp_path / "live" / "output-results" / "mounts.json")["ok"]
        for mount in ("program", "stitched"):
            receivers.append(
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-rtsp_transport",
                        "tcp",
                        "-i",
                        f"rtsp://127.0.0.1:{port}/{mount}",
                        "-frames:v",
                        "3",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        str(tmp_path / f"{mount}.rgb"),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(
            receiver.poll() is None for receiver in receivers
        ):
            routes.write(context)
            time.sleep(0.1)
        for receiver in receivers:
            _, error = receiver.communicate(timeout=2)
            assert receiver.returncode == 0, error.decode()
        program = np.frombuffer((tmp_path / "program.rgb").read_bytes(), dtype=np.uint8).reshape(
            3, 96, 160, 3
        )
        stitched = np.frombuffer((tmp_path / "stitched.rgb").read_bytes(), dtype=np.uint8).reshape(
            3, 120, 200, 3
        )
        assert program.mean(axis=(0, 1, 2))[1] > 150
        assert stitched.mean(axis=(0, 1, 2))[0] > 150
    finally:
        for receiver in receivers:
            stop(receiver)
        routes.close()


def should_preserve_source_audio_on_rtsp(tmp_path):
    probe = subprocess.run(
        ["/usr/bin/python3", "-c", "import gi; gi.require_version('GstRtspServer','1.0')"],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        pytest.skip("System Python lacks GstRtspServer")
    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=20",
            str(audio),
        ],
        check=True,
    )
    port = free_port()
    routes = OutputRoutes(
        {"rtsp_enabled": True, "rtsp_port": port}, fps=10, audio={"file": str(audio)}
    )
    receiver = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            f"rtsp://127.0.0.1:{port}/program",
            "-t",
            "1",
            "-c",
            "copy",
            str(tmp_path / "received.mkv"),
        ],
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and receiver.poll() is None:
            routes.write({"img": np.full((96, 160, 3), 100, dtype=np.uint8)})
            time.sleep(0.1)
        _, error = receiver.communicate(timeout=2)
        assert receiver.returncode == 0, error.decode()
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(tmp_path / "received.mkv"),
                "-map",
                "0:a:0",
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
            check=True,
        )
        assert np.abs(np.frombuffer(result.stdout, dtype=np.int16).astype(float)).mean() > 100
    finally:
        stop(receiver)
        routes.close()


def should_keep_rink_map_drawing_out_of_output_frames():
    import torch

    from hmlib.aspen.plugins.video_preview_plugin import VideoPreviewPlugin

    class Preview:
        def publish_preview(self, image, *, name):
            self.image = image
            self.name = name

    preview = Preview()
    plugin = VideoPreviewPlugin(show_image=False, show_youtube=False)
    frame = torch.zeros((1, 96, 160, 3), dtype=torch.uint8)
    try:
        plugin.forward(
            {
                "img": frame,
                "fps": 10,
                "shared": {"hm_ui_process": preview, "game_config": {}},
                "rink_map_overlay": {"points": [[[10, 5]]], "size": [60, 25]},
            }
        )
        assert not frame.any()
        assert preview.image.any()
        assert preview.name == "Final"
    finally:
        plugin.finalize()


def should_encode_independent_program_stitched_and_4k_archives_with_audio(tmp_path):
    import torch

    from hmlib.aspen.plugins.video_out_plugin import VideoOutPlugin

    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(audio),
        ],
        check=True,
    )
    plugin = VideoOutPlugin(
        output_video_path=str(tmp_path / "program.mkv"),
        preview_via_sink=False,
        video_out_device="cpu",
    )
    shared = {
        "game_config": {"video_out": {"archive_program_4k": True, "archive_stitched": True}},
        "fps": 10,
        "mux_audio_file": str(audio),
    }
    try:
        for index in range(3):
            plugin.forward(
                {
                    "shared": shared,
                    "fps": 10,
                    "img": torch.full((1, 96, 160, 3), 80, dtype=torch.uint8),
                    "original_images": torch.full((1, 3, 120, 200), 120, dtype=torch.uint8),
                    "program_4k_img": torch.full((1, 2160, 3840, 3), 150, dtype=torch.uint8),
                    "frame_ids": torch.tensor([index]),
                    "video_frame_cfg": {
                        "output_frame_width": 160,
                        "output_frame_height": 96,
                        "output_aspect_ratio": 160 / 96,
                    },
                }
            )
    finally:
        plugin.finalize()
    assert {path.name for path in plugin.supplementary_paths()} == {
        "stitched_output.mkv",
        "program_4k_output.mkv",
    }
    paths = {path.name: path for path in plugin.supplementary_paths()}
    paths["program.mkv"] = tmp_path / "program.mkv"
    for name, size in (
        ("program.mkv", (160, 96)),
        ("stitched_output.mkv", (200, 120)),
        ("program_4k_output.mkv", (3840, 2160)),
    ):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(paths[name])],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout)["streams"]
        video = next(stream for stream in streams if stream["codec_type"] == "video")
        assert (video["width"], video["height"]) == size
        assert any(stream["codec_type"] == "audio" for stream in streams)


def should_finalize_and_resume_each_live_recording_without_losing_segments(tmp_path, monkeypatch):
    import torch

    from hmlib.aspen.plugins.video_out_plugin import VideoOutPlugin

    session = tmp_path / "live"
    monkeypatch.setenv("HM_UI_SESSION_DIR", str(session))
    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            str(audio),
        ],
        check=True,
    )
    flags = {"archive_program": True, "archive_stitched": True, "archive_program_4k": True}
    shared = {"game_config": {"video_out": dict(flags)}, "fps": 10, "mux_audio_file": str(audio)}
    plugin = VideoOutPlugin(
        output_video_path=str(tmp_path / "tracking_output.mkv"),
        preview_via_sink=False,
        video_out_device="cpu",
    )
    routes = OutputRoutes(dict(flags), fps=10, session_dir=session)
    try:
        for index in range(7):
            if index in {2, 4, 6}:
                write_json(
                    session / "output-actions" / f"{index}.json",
                    {"changes": {key: index == 4 for key in flags}},
                )
            context = {
                "shared": shared,
                "fps": 10,
                "img": torch.full((1, 96, 160, 3), 80, dtype=torch.uint8),
                "original_images": torch.full((1, 3, 120, 200), 120, dtype=torch.uint8),
                "program_4k_img": torch.full((1, 2160, 3840, 3), 150, dtype=torch.uint8),
                "frame_ids": torch.tensor([index]),
                "video_frame_cfg": {
                    "output_frame_width": 160,
                    "output_frame_height": 96,
                    "output_aspect_ratio": 160 / 96,
                },
            }
            routes.write(context)
            plugin.forward(context)
            if index in {2, 6}:
                state = read_json(session / "recording-state.json")
                assert not any(state["active"].values())
                assert all(item["state"] == "finalized" for item in state["segments"])
    finally:
        plugin.finalize()
        routes.close()
    paths = plugin.output_paths()
    assert len(paths) == 6
    assert len({path.name for path in paths}) == 6
    for path in paths:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = json.loads(result.stdout)["streams"]
        video = next(stream for stream in streams if stream["codec_type"] == "video")
        assert int(video["nb_read_frames"]) == 2
        assert float(video["start_time"]) < 0.1
        assert any(stream["codec_type"] == "audio" for stream in streams)
    state = read_json(session / "recording-state.json")
    assert state["state"] == "finalized"
    assert [(item["start_frame"], item["end_frame"]) for item in state["segments"]] == [
        (0, 2)
    ] * 3 + [(4, 6)] * 3
    from hmlib.ui.workspace import Workspace
    from hmlib.utils.output_publication import publish_artifacts

    manifest = next(path for path in plugin.supplementary_paths() if path.suffix == ".json")
    with Workspace(tmp_path / "games", tmp_path / "workspace") as workspace:
        workspace.create_game("sample")
        publish_artifacts(
            {**{f"sample-{path.name}": path for path in paths}, manifest.name: manifest},
            workspace.game(),
        )
        archives = workspace.catalog(workspace.game())["archives"]
        assert len(archives) == 6
        assert {item["version"] for item in archives} == {1}
        assert sorted(item["timeline_offset_seconds"] for item in archives) == [
            0,
            0,
            0,
            0.4,
            0.4,
            0.4,
        ]
        assert {item["duration_seconds"] for item in archives} == {0.2}
