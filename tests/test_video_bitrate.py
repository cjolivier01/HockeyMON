from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from hmlib.video.bitrate import (
    DEFAULT_OUTPUT_BITRATE,
    resolve_output_bitrate,
    select_source_bitrate_density,
)


def should_select_highest_density_across_cameras_and_chapters():
    metadata = {
        "left-1.mp4": (60_000_000, 3840, 2160),
        "left-2.mp4": (55_000_000, 3840, 2160),
        "right-1.mp4": (30_000_000, 1920, 1080),
    }
    calls = []

    def probe(path):
        calls.append(path)
        rate, width, height = metadata[path]
        return SimpleNamespace(bit_rate=rate, width=width, height=height)

    density = select_source_bitrate_density([*metadata, "left-1.mp4"], probe=probe)
    assert density == Fraction(30_000_000, 1920 * 1080)
    assert calls == list(metadata)
    assert resolve_output_bitrate(None, density, 3840, 2160) == 120_000_000
    assert resolve_output_bitrate(None, density, 960, 540) == 7_500_000


def should_round_fractional_output_bitrate_half_up():
    assert resolve_output_bitrate(None, Fraction(5, 2), 1, 1) == 3
    assert resolve_output_bitrate(None, Fraction(7, 3), 1, 1) == 2


def should_preserve_explicit_bitrate():
    assert resolve_output_bitrate(55_000_000, Fraction(10), 640, 360) == 55_000_000


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), float("inf")])
def should_reject_invalid_explicit_bitrate(value):
    with pytest.raises((ValueError, OverflowError)):
        resolve_output_bitrate(value, None, 640, 360)


def should_warn_when_source_metadata_is_missing_and_use_fallback(caplog):
    density = select_source_bitrate_density(
        ["unknown.mp4"], probe=lambda path: SimpleNamespace(bit_rate=0, width=640, height=360)
    )
    assert density is None
    assert resolve_output_bitrate(None, density, 1920, 1080) == DEFAULT_OUTPUT_BITRATE
    assert "unknown.mp4" in caplog.text
    assert "using 55000000 bps" in caplog.text


def should_propagate_probe_failures():
    def probe(path):
        raise OSError("ffprobe unavailable")

    with pytest.raises(OSError, match="ffprobe unavailable"):
        select_source_bitrate_density(["video.mp4"], probe=probe)


@pytest.mark.parametrize("explicit", [None, 22_000_000])
def should_wire_source_density_through_video_plugin(monkeypatch, explicit):
    from hmlib.aspen.plugins import video_out_plugin as plugin_module

    created = []
    probed = []
    monkeypatch.setattr(
        plugin_module, "VideoOutput", lambda **kwargs: created.append(kwargs) or SimpleNamespace()
    )
    monkeypatch.setattr(
        plugin_module,
        "select_source_bitrate_density",
        lambda paths: probed.append(paths) or Fraction(25),
    )
    plugin = plugin_module.VideoOutPlugin(bit_rate=explicit, preview_via_sink=False)
    paths = ["left-1.mp4", "left-2.mp4", "right-1.mp4"]
    plugin._ensure_initialized(
        {
            "img": torch.zeros((1, 10, 20, 3), dtype=torch.uint8),
            "shared": {"source_video_paths": paths, "game_config": {}},
        }
    )
    assert created[0]["bit_rate"] == explicit
    assert created[0]["source_bitrate_density"] == (Fraction(25) if explicit is None else None)
    assert probed == ([paths] if explicit is None else [])


def should_scale_using_final_encoder_geometry_for_each_output(monkeypatch, tmp_path):
    from hmlib.video import video_out as output_module

    writers = []
    monkeypatch.setattr(
        output_module,
        "create_output_video_stream",
        lambda **kwargs: writers.append(kwargs) or SimpleNamespace(isOpened=lambda: True),
    )
    output = output_module.VideoOutput(
        output_video_path=str(tmp_path / "video.mp4"),
        fps=30,
        device="cpu",
        source_bitrate_density=Fraction(12),
        enable_end_zones=True,
    )
    output.create_output_videos(
        {"video_frame_cfg": {"output_frame_width": 1280, "output_frame_height": 720}}
    )
    assert len(writers) == 2
    assert all(writer["bit_rate"] == 12 * 1280 * 720 for writer in writers)


def should_use_video_stream_bitrate_metadata_from_ffprobe(tmp_path, monkeypatch):
    from hmlib.video import ffmpeg

    class Stream:
        r_frame_rate = "30/1"

        def durationSeconds(self):
            return 1

        def frameSize(self):
            return 1920, 1080

        def bitrate(self):
            return 40_000_000

        def codecTag(self):
            return "HEVC"

    monkeypatch.setattr(ffmpeg, "FFProbe", lambda path: SimpleNamespace(video=[Stream()]))
    assert select_source_bitrate_density([str(tmp_path / "clip.mp4")]) == Fraction(
        40_000_000, 1920 * 1080
    )


def should_pass_actual_tracking_sources_to_the_output_graph(monkeypatch):
    from hmlib.tasks import tracking

    captured = []

    class Loader(list):
        fps = 30
        batch_size = 1

    class Net:
        def __init__(self, name, config, shared):
            self.shared = shared
            captured.append(shared)

        def to(self, device):
            return self

        def finalize(self):
            return None

    monkeypatch.setattr(tracking, "AspenNet", Net)
    monkeypatch.setattr(tracking, "CachedIterator", lambda **kwargs: iter(()))
    paths = ["camera-one/chapter1.mp4", "camera-one/chapter2.mp4", "camera-two/chapter1.mp4"]
    tracking.run_mmtrack(
        model=None,
        config={"aspen": {"plugins": {}}},
        dataloader=Loader(),
        postprocessor=None,
        device="cpu",
        no_cuda_streams=True,
        source_video_paths=paths,
    )
    assert captured[0]["source_video_paths"] == paths
