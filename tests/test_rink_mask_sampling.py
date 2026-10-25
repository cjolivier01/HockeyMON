from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from hmlib.stitching.rink_mask_sampling import (
    parse_mask_frame_time,
    read_mask_frame_time,
    resolve_mask_frame_time,
    sample_rink_mask_frame,
    source_playlists,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", None),
        ("-1", -1),
        ("00:00:00", 0),
        ("01:02.5", 62.5),
        ("-00:00:01.250", -1.25),
        ("123:02:03", 442923),
    ],
)
def should_parse_native_mask_time_forms(value, expected):
    assert parse_mask_frame_time(value) == expected


@pytest.mark.parametrize(
    "value", ["-0", "-00:00:00", "00:00:60", "01:60:00", "1.2345", "+1", "", None]
)
def should_reject_ambiguous_or_out_of_range_mask_times(value):
    with pytest.raises(ValueError):
        parse_mask_frame_time(value)


def should_resolve_venue_inheritance_and_synchronized_recording_end():
    config = {
        "stitching": {
            "rink_config": "rink",
            "rink_configs": {"rink": {"rink_mask_frame_time": "-1"}},
            "rink_mask_frame_time_default": "-3",
        }
    }
    assert read_mask_frame_time(config) == "-00:00:01"
    config["stitching"]["rink_mask_frame_time"] = "auto"
    assert read_mask_frame_time(config) == "auto"
    config["stitching"]["rink_mask_frame_time"] = None
    assert resolve_mask_frame_time(read_mask_frame_time(config), 60) == 59
    for offset, duration in (("-61", 60), ("60", 60), ("-1", 0)):
        with pytest.raises(ValueError):
            resolve_mask_frame_time(offset, duration)


def should_inherit_the_shipped_rink_defaults_and_allow_auto():
    from hmlib.stitching.rink_mask_sampling import _mask_baseline

    baseline = _mask_baseline()
    assert read_mask_frame_time({}) == baseline.get("rink_mask_frame_time_default", "auto")
    for rink in ("tv-dublin", "fremont"):
        assert read_mask_frame_time({"stitching": {"rink_config": rink}}) == "-00:00:01"
        assert read_mask_frame_time(
            {"stitching": {"rink_config": rink, "rink_mask_frame_time_default": "auto"}}
        ) == (baseline.get("rink_configs", {}).get(rink, {}).get("rink_mask_frame_time") or "auto")
    assert read_mask_frame_time({"stitching": {"rink_mask_frame_time": "auto"}}) == "auto"


def should_support_optional_shipped_mask_time_default(monkeypatch):
    from hmlib.stitching import rink_mask_sampling

    monkeypatch.setattr(rink_mask_sampling, "_mask_baseline", dict)
    assert read_mask_frame_time({}) == "auto"
    monkeypatch.setattr(
        rink_mask_sampling, "_mask_baseline", lambda: {"rink_mask_frame_time_default": "-1"}
    )
    assert read_mask_frame_time({}) == "-00:00:01"


def _movie(path, values):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (24, 16))
    assert writer.isOpened()
    try:
        for value in values:
            writer.write(np.full((16, 24, 3), value, np.uint8))
    finally:
        writer.release()


def should_decode_selected_synchronized_frame_across_chapters_without_playback_seek(tmp_path):
    _movie(tmp_path / "left-a.avi", [10] * 10)
    _movie(tmp_path / "left-b.avi", [20] * 10)
    _movie(tmp_path / "right.avi", [30] * 30)
    config = {
        "game": {"videos": {"left": ["left-a.avi", "left-b.avi"], "right": ["right.avi"]}},
        "stitching": {"rink_mask_frame_time": "-0.5", "frame_offsets": {"left": 0, "right": 10}},
    }
    observed = []

    def renderer(frames):
        observed.extend(float(frame.float().mean()) for frame in frames)
        return frames[0].permute(2, 0, 1)

    context = {
        "shared": {"game_config": config, "game_dir": str(tmp_path), "game_id": "game"},
        "rink_mask_renderer": renderer,
    }
    playback = cv2.VideoCapture(str(tmp_path / "left-a.avi"))
    try:
        assert playback.read()[0]
        before = playback.get(cv2.CAP_PROP_POS_FRAMES)
        frame, preference = sample_rink_mask_frame(context, torch.zeros(3, 16, 24))
        assert observed == pytest.approx([20, 30], abs=2)
        assert tuple(frame.shape) == (3, 16, 24) and preference == "-00:00:00.500"
        assert playback.get(cv2.CAP_PROP_POS_FRAMES) == before
    finally:
        playback.release()
    config["stitching"]["rink_mask_frame_time"] = "auto"
    original = torch.zeros(3, 16, 24)
    assert sample_rink_mask_frame(context, original)[0] is original


def should_render_timed_mask_with_separate_stitcher_and_captured_rotation(tmp_path, monkeypatch):
    from hmlib.aspen.plugins.stitching_plugin import StitchingPlugin

    created = []

    class IndependentStitcher:
        def forward(self, inputs):
            return inputs[0]

    def create(self, context, imgs, device):
        created.append(self)
        self._stitcher = IndependentStitcher()

    monkeypatch.setattr(StitchingPlugin, "_create_stitcher", create)
    plugin = StitchingPlugin(dir_name=str(tmp_path))
    plugin._stitcher = object()
    context = {"shared": {"game_config": {}}}
    frames = [torch.zeros(16, 24, 3, dtype=torch.uint8) for _ in range(2)]
    frames[0][:8, :12] = 255
    inputs = [frame.unsqueeze(0) for frame in frames]
    render = plugin._make_mask_renderer(context, inputs, 15)
    output = render(frames)
    assert len(created) == 1 and created[0] is not plugin
    assert tuple(output.shape) == (3, 16, 24)
    assert not torch.equal(output, frames[0].permute(2, 0, 1))


@pytest.mark.parametrize("configured", [True, False])
def should_sample_single_source_with_fresh_default_config(tmp_path, configured):
    from hmlib.config import ROOT_DIR, baseline_config

    _movie(tmp_path / "source.avi", [10] * 10 + [80] * 10)
    config = baseline_config(ROOT_DIR)
    config["stitching"]["enabled"] = False
    if configured:
        config["game"] = {"videos": {"source": ["source.avi"]}}
    context = {
        "shared": {
            "game_config": config,
            "game_dir": str(tmp_path),
            "game_id": "fresh-single-source",
            "source_video_paths": [str(tmp_path / "source.avi")],
            "original_clip_box": [2, 2, 22, 14],
        }
    }
    frame, preference = sample_rink_mask_frame(context, torch.zeros(3, 16, 24))
    assert preference == "-00:00:01"
    assert frame.shape == (16, 24, 3)  # original_images precedes detector HmCrop.
    assert frame.float().mean().item() == pytest.approx(80, abs=2)
    cropped, _ = sample_rink_mask_frame(context, torch.zeros(3, 12, 20))
    assert torch.equal(cropped, frame[2:14, 2:22])


def should_honor_removed_source_roles_without_rediscovery(tmp_path):
    _movie(tmp_path / "source.avi", [10] * 20)
    config = {
        "game": {"videos": {"source": ["source.avi"]}},
        "hstream_ui": {"video_roles": {"source": []}},
    }
    assert (
        source_playlists(
            config, tmp_path, source_video_paths=[str(tmp_path / "source.avi")], single_source=True
        )
        == {}
    )
    with pytest.raises(ValueError, match="Select both camera playlists"):
        sample_rink_mask_frame(
            {"shared": {"game_config": config, "game_dir": str(tmp_path)}},
            torch.zeros(3, 16, 24),
        )


def should_not_treat_flat_stitcher_paths_as_single_source_chapters(tmp_path):
    _movie(tmp_path / "source.avi", [10] * 20)
    context = {
        "shared": {
            "game_config": {"stitching": {"rink_mask_frame_time": "-1"}},
            "game_dir": str(tmp_path),
            "source_video_paths": [str(tmp_path / "source.avi")],
        },
        "camera_input_geometry": {"stitched_geometry_revision": "active"},
    }
    with pytest.raises(ValueError, match="stitched playlist|both camera playlists"):
        sample_rink_mask_frame(context, torch.zeros(3, 16, 24))
