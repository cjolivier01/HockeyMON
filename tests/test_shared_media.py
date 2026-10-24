"""Real media precision and local audio lifecycle checks."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from hmlib.ui.hstream_audio import AudioMonitor
from hmlib.video.high_bit_reader import HighBitVideoReader


def audio_descendants(player):
    """Wait for the real ffplay process behind its lightweight owner."""
    deadline = time.monotonic() + 3
    children = Path(f"/proc/{player.pid}/task/{player.pid}/children")
    while time.monotonic() < deadline:
        assert player.poll() is None, "Audio owner exited before launching ffplay"
        if children.is_file() and (pids := children.read_text().split()):
            return [int(pid) for pid in pids]
        time.sleep(0.01)
    raise AssertionError("Audio owner did not launch ffplay")


def should_retain_sub_8_bit_samples_and_seek_exactly(tmp_path):
    width, height = 64, 32
    data = np.arange(width * height * 3, dtype=np.uint16).reshape(height, width, 3) * 7
    frames = np.stack([data, data + 31])
    video = tmp_path / "gradient.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr48le",
            "-s",
            f"{width}x{height}",
            "-r",
            "2",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "gbrp16le",
            str(video),
        ],
        input=frames.tobytes(),
        check=True,
    )
    reader = HighBitVideoReader(video, width, height, batch_size=2)
    try:
        batch = next(reader)
        assert batch.dtype == torch.float32
        np.testing.assert_allclose(
            batch.numpy(), frames.transpose(0, 3, 1, 2).astype(np.float32) / 257, atol=1e-5
        )
        assert torch.any(batch != batch.round())
        with pytest.raises(StopIteration):
            next(reader)
    finally:
        reader.close()
    reader = HighBitVideoReader(video, width, height, seconds=0.5)
    try:
        np.testing.assert_allclose(
            next(reader)[0].numpy(), frames[1].transpose(2, 0, 1) / 257, atol=1e-5
        )
    finally:
        reader.close()


def should_surface_decoder_failure_without_partial_frames(tmp_path):
    reader = HighBitVideoReader(tmp_path / "missing.mp4", 64, 32)
    try:
        with pytest.raises(RuntimeError, match="decode failed"):
            next(reader)
    finally:
        reader.close()


def should_monitor_real_audio_and_reap_every_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    audio = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=5",
            str(audio),
        ],
        check=True,
    )
    monitor = AudioMonitor(audio)
    children = []
    descendants = []
    try:
        monitor.update(0.0, playing=True)
        children.append(monitor.player)
        descendants.extend(audio_descendants(monitor.player))
        time.sleep(0.1)
        monitor.update(0.1, playing=True)
        assert monitor.player.poll() is None
        assert os.getpgid(monitor.player.pid) == os.getpgrp()
        monitor.set_muted(True)
        assert children[-1].poll() is not None
        monitor.update(0.2, playing=True)
        assert monitor.player is None
        monitor.set_muted(False)
        monitor.update(1.0, playing=True)
        children.append(monitor.player)
        monitor.set_paused(True)
        assert children[-1].poll() is not None
        monitor.set_paused(False)
        monitor.update(1.1, playing=True)
        children.append(monitor.player)
        monitor.update(3.0, playing=True)
        assert children[-1].poll() is not None
        children.append(monitor.player)
    finally:
        monitor.close()
    assert all(child.poll() is not None for child in children)
    assert all(not Path(f"/proc/{pid}").exists() for pid in descendants)


def should_advance_local_audio_to_the_next_chapter_without_joining_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    chapters = [tmp_path / f"chapter-{index}.wav" for index in range(2)]
    for index, path in enumerate(chapters):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={440 * (index + 1)}:duration=1",
                str(path),
            ],
            check=True,
        )
    monitor = AudioMonitor(chapters, durations=[1.0, 1.0])
    children = []
    descendants = []
    try:
        monitor.update(0.1, playing=True)
        children.append(monitor.player)
        descendants.extend(audio_descendants(monitor.player))
        assert monitor.player.args[-1] == str(chapters[0])
        monitor.update(1.2, playing=True)
        assert children[-1].poll() is not None
        children.append(monitor.player)
        descendants.extend(audio_descendants(monitor.player))
        assert monitor.player.args[-1] == str(chapters[1])
        assert float(monitor.player.args[monitor.player.args.index("-ss") + 1]) == pytest.approx(
            0.2
        )
        time.sleep(0.1)
        monitor.update(1.3, playing=True)
        assert monitor.player is children[-1]
        assert monitor.player.poll() is None
        monitor.update(0.3, playing=True)
        assert children[-1].poll() is not None
        children.append(monitor.player)
        assert monitor.player.args[-1] == str(chapters[0])
    finally:
        monitor.close()
    assert all(child.poll() is not None for child in children)
    assert all(not Path(f"/proc/{pid}").exists() for pid in descendants)
