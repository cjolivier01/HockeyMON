from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from PIL import Image

from hmlib.stitching.rink_mask_sampling import read_mask_frame_time
from hmlib.stitching.settings import read_stitching_settings
from hmlib.ui.calibration_workspace import (
    CalibrationWorkspace,
    prepare_rink_map,
    project_goal_polygons,
    project_rink_points,
    unproject_rink_points,
)


@pytest.fixture
def game(tmp_path):
    directory = tmp_path / "game-a"
    directory.mkdir()
    (directory / "config.yaml").write_text("game:\n  name: Original\ncustom:\n  value: keep\n")
    return directory


def _private(game):
    return yaml.safe_load((game / "config.yaml").read_text())


def _generation(game):
    for name in ("s.png", "left.png", "right.png"):
        Image.new("RGB", (400, 200), (100, 140, 180)).save(game / name)
    project = (
        "p f2 w800 h200 v180\n"
        'i w400 h200 f0 v108 n"left.png"\n'
        'i w400 h200 f0 v108 n"right.png"\n'
        "c n0 N1 x300 y40 X100 Y40 t0\n"
        "c n0 N1 x350 y40 X150 Y40 t0\n"
        "c n0 N1 x300 y150 X100 Y150 t0\n"
        "c n0 N1 x350 y150 X150 Y150 t0\n"
    )
    (game / "hm_project.pto").write_text(project)
    (game / "autooptimiser_out.pto").write_text(project)
    settings = read_stitching_settings(CalibrationWorkspace(game)._effective())
    (game / ".stitching_artifacts.json").write_text(json.dumps(settings.manifest()))


def _landmarks():
    return [
        {"image": [0, 0], "rink": [0, 0]},
        {"image": [399, 0], "rink": [60.96, 0]},
        {"image": [399, 199], "rink": [60.96, 25.908]},
        {"image": [0, 199], "rink": [0, 25.908]},
    ]


def should_load_effective_settings_without_materializing_baseline(game):
    before = (game / "config.yaml").read_bytes()
    workspace = CalibrationWorkspace(game)
    snapshot = workspace.snapshot()
    assert snapshot["settings"]["stitching.mapping_backend"] == "nona"
    assert snapshot["settings"]["stitching.max_control_points"] == 1500
    assert (game / "config.yaml").read_bytes() == before
    assert snapshot["scoreboard"]["image"] is None
    assert snapshot["rink_mask_frame_time"] == {
        "selection": None,
        "effective": read_mask_frame_time({}),
        "source": "default",
    }


def should_read_bundled_runfile_symlinks_but_refuse_private_config_symlinks(
    game, tmp_path, monkeypatch
):
    from hmlib.ui import calibration_workspace

    package = tmp_path / "package"
    (package / "config").mkdir(parents=True)
    (package / "config" / "baseline.yaml").symlink_to(
        calibration_workspace.ROOT_DIR + "/config/baseline.yaml"
    )
    monkeypatch.setattr(calibration_workspace, "ROOT_DIR", str(package))
    assert CalibrationWorkspace(game).snapshot()["settings"]["stitching.mapping_backend"] == "nona"
    original = (game / "config.yaml").read_bytes()
    (game / "config.yaml").unlink()
    external = tmp_path / "private.yaml"
    external.write_bytes(original)
    (game / "config.yaml").symlink_to(external)
    with pytest.raises(ValueError, match="regular document"):
        CalibrationWorkspace(game).snapshot()


def should_canonicalize_mask_time_and_preserve_camera_calibration(game):
    _generation(game)
    config = _private(game)
    config["stitching"] = {"control_points": [[1, 2, 3, 4]]}
    config["rink"] = {
        "scoreboard": {"perspective_polygon": [[1, 2]] * 4},
        "ice_contours_geometry_revision": "old-mask",
    }
    (game / "config.yaml").write_text(yaml.safe_dump(config))
    (game / "xor_file.png").write_bytes(b"retained seam")
    (game / "rink_mask_0.png").write_bytes(b"derived mask")
    retained = {
        name: (game / name).read_bytes()
        for name in ("hm_project.pto", "autooptimiser_out.pto", "xor_file.png")
    }
    workspace = CalibrationWorkspace(game)
    draft = workspace.stage_settings({"stitching.rink_mask_frame_time": "-0:02.50"})
    assert draft["apply_mode"] == "next-run"
    assert draft["changes"]["stitching.rink_mask_frame_time"] == "-00:00:02.500"
    result = workspace.apply(draft)
    assert result["rink_mask_frame_time"]["selection"] == "-00:00:02.500"
    assert _private(game)["stitching"]["control_points"] == [[1, 2, 3, 4]]
    assert not (game / "rink_mask_0.png").exists()
    assert "ice_contours_geometry_revision" not in _private(game)["rink"]
    assert all((game / name).read_bytes() == contents for name, contents in retained.items())
    assert workspace.update_settings({"stitching.rink_mask_frame_time": None})[
        "rink_mask_frame_time"
    ] == {"selection": None, "effective": read_mask_frame_time({}), "source": "default"}


def should_stage_and_apply_only_requested_private_leaves(game):
    workspace = CalibrationWorkspace(game)
    before = (game / "config.yaml").read_bytes()
    draft = workspace.stage_settings({"stitching.camera_fov.horizontal_fov": 110.5})
    assert (game / "config.yaml").read_bytes() == before
    workspace.apply(draft)
    assert _private(game) == {
        "game": {"name": "Original"},
        "custom": {"value": "keep"},
        "stitching": {"camera_fov": {"horizontal_fov": 110.5}},
    }


def should_reject_stale_draft_without_overwriting_external_edits(game):
    workspace = CalibrationWorkspace(game)
    draft = workspace.stage_settings({"stitching.camera_fov.horizontal_fov": 110})
    (game / "config.yaml").write_text("external: edit\n")
    with pytest.raises(ValueError, match="changed"):
        workspace.apply(draft)
    assert _private(game) == {"external": "edit"}


@pytest.mark.parametrize(
    "changes",
    [
        {"stitching.fake_feature": True},
        {"stitching.mapping_backend": "nonsense"},
        {"stitching.camera_fov.horizontal_fov": float("nan")},
        {"stitching.calibration_frame_count": 1.5},
        {"stitching.run_autooptimizer": False},
        {"stitching.stitch_frame_time": "00:60:00"},
        {"stitching.projection_framing.crop": [0.7, 0.1, 0, 1]},
    ],
)
def should_reject_invalid_settings_without_mutation(game, changes):
    before = (game / "config.yaml").read_bytes()
    with pytest.raises(ValueError):
        CalibrationWorkspace(game).update_settings(changes)
    assert (game / "config.yaml").read_bytes() == before


def should_leave_unchanged_settings_byte_identical(game):
    before = (game / "config.yaml").read_bytes()
    CalibrationWorkspace(game).update_settings({"stitching.mapping_backend": "nona"})
    assert (game / "config.yaml").read_bytes() == before


def should_preserve_inherited_rotation_when_cropping(game):
    (game / "config.yaml").write_text("stitching:\n  rink_config: tv-dublin\n")
    workspace = CalibrationWorkspace(game)
    expected = workspace.snapshot()["settings"]["stitching.projection_framing.rotation_degrees"]
    workspace.set_crop([0.05, 0.95, 0.1, 0.9])
    assert "rotation_degrees" not in _private(game)["stitching"]["projection_framing"]
    assert (
        workspace.snapshot()["settings"]["stitching.projection_framing.rotation_degrees"]
        == expected
    )


def should_invalidate_geometry_annotations_and_only_derived_masks(game):
    _generation(game)
    private = _private(game)
    private["rink"] = {
        "scoreboard": {"perspective_polygon": [[1, 1]] * 4},
        "ice_contours_mask_count": 1,
    }
    (game / "config.yaml").write_text(yaml.safe_dump(private))
    for name in ("rink_mask_0.png", "rink_mask_0-17.png", "xor_file.png"):
        (game / name).write_bytes(b"cache")
    CalibrationWorkspace(game).set_leveling([0, -2, 3])
    assert "perspective_polygon" not in _private(game)["rink"]["scoreboard"]
    assert not (game / "rink_mask_0.png").exists()
    assert not (game / "xor_file.png").exists()
    assert (game / "rink_mask_0-17.png").read_bytes() == b"cache"
    assert (game / "autooptimiser_out.pto").is_file()


def should_save_bounded_scoreboard_and_explicit_disabled_sentinel(game):
    _generation(game)
    workspace = CalibrationWorkspace(game)
    workspace.set_scoreboard([[50, 30], [10, 10], [10, 30], [50, 10]])
    assert _private(game)["rink"]["scoreboard"]["perspective_polygon"] == [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
    ]
    before = (game / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="outside"):
        workspace.set_scoreboard([[0, 0], [400, 0], [400, 20], [0, 20]])
    assert (game / "config.yaml").read_bytes() == before
    workspace.set_scoreboard([])
    assert _private(game)["rink"]["scoreboard"]["perspective_polygon"] == [[0, 0]] * 4


def should_replace_manual_matches_and_trigger_existing_edited_pto_path(game):
    from hmlib.stitching.configure_stitching import is_older_than
    from hmlib.stitching.hugin import configure_control_points

    _generation(game)
    workspace = CalibrationWorkspace(game)
    before = (game / ".stitching_artifacts.json").read_bytes()
    pairs = [[250, 25, 50, 25], [350, 25, 150, 25], [250, 150, 50, 150], [350, 150, 150, 150]]
    workspace.save_control_points(pairs)
    assert workspace.read_control_points()["pairs"] == pairs
    assert (game / ".stitching_artifacts.json").read_bytes() == before
    assert is_older_than(game / "hm_project.pto", game / "autooptimiser_out.pto")
    consumed = configure_control_points(
        str(game / "hm_project.pto"), str(game / "left.png"), str(game / "right.png"), 10
    )
    assert consumed["m_kpts0"].tolist() == [pair[:2] for pair in pairs]


def should_fit_and_roundtrip_physical_rink_map_and_goals(game):
    _generation(game)
    workspace = CalibrationWorkspace(game)
    goal = [[0, 70], [20, 70], [20, 130], [0, 130]]
    workspace.save_rink_map(_landmarks(), [goal])
    mapping = prepare_rink_map(game)
    assert mapping["coordinate_system"] == "corner-origin-metres"
    assert mapping["reprojection_error"] < 1e-6
    assert np.allclose(project_rink_points([[199.5, 99.5]], mapping), [[30.48, 12.954]])
    assert np.allclose(
        project_rink_points([[99.75, 49.75]], mapping, image_size=[200, 100]), [[30.48, 12.954]]
    )
    assert np.allclose(unproject_rink_points([[30.48, 12.954]], mapping), [[199.5, 99.5]])
    assert len(project_goal_polygons(mapping)[0]) == 4
    private = _private(game)
    private.setdefault("stitching", {})["post_stitch_rotate_degrees"] = 2
    (game / "config.yaml").write_text(yaml.safe_dump(private))
    with pytest.raises(ValueError, match="different calibration"):
        prepare_rink_map(game)


def should_reject_degenerate_map_and_enable_only_reviewed_maps(game):
    _generation(game)
    workspace = CalibrationWorkspace(game)
    before = (game / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="projection"):
        workspace.save_rink_map([{"image": [x, 0], "rink": [x, 0]} for x in range(4)])
    with pytest.raises(ValueError, match="reviewed map"):
        workspace.update_settings({"rink.camera.rink_framing.enabled": True})
    assert (game / "config.yaml").read_bytes() == before
    workspace.save_rink_map(_landmarks())
    workspace.update_settings({"rink.camera.rink_framing.enabled": True})
    assert prepare_rink_map(game) is not None
    with pytest.raises(ValueError, match="at least one"):
        workspace.update_settings(
            {
                "rink.camera.rink_framing.anticipate": False,
                "rink.camera.rink_framing.frame_goals": False,
                "rink.camera.rink_framing.adaptive_zoom": False,
            }
        )


def should_resolve_game_from_explicit_directory_without_global_root(game, monkeypatch):
    import hmlib.config

    other = game.parent / "other"
    other.mkdir()
    monkeypatch.setattr(hmlib.config, "GAME_DIR_BASE", str(other))
    CalibrationWorkspace(game).update_settings({"stitching.camera_fov.horizontal_fov": 111})
    assert _private(game)["stitching"]["camera_fov"]["horizontal_fov"] == 111
    assert not list(other.iterdir())


@pytest.mark.parametrize(
    "preference, expected", [(None, "2k"), ("auto", "2k"), ("1k", "1k"), ("native", "native")]
)
def should_resolve_shared_superpoint_size_in_calibration_provenance(preference, expected):
    settings = read_stitching_settings({"stitching": {"control_point_resolution": preference}})
    assert settings.control_point_resolution == expected
    assert (
        json.loads(settings.manifest()["calibration_settings"])["control_point_resolution"]
        == expected
    )


@pytest.mark.parametrize(
    "resolution, expected", [("2k", (1152, 2048)), ("1k", (768, 1024)), ("native", (480, 640))]
)
def should_restore_superpoint_source_centers_and_discard_padding(resolution, expected):
    import torch

    from hmlib.stitching.superpoint import SuperPoint

    class FakeDetector(SuperPoint):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.shape = None

        def forward(self, data):
            self.shape = tuple(data["image"].shape[-2:])
            height, width = self.shape
            return {
                "keypoints": torch.tensor([[[24.0, 48.0], [float(width - 1), float(height - 1)]]]),
                "keypoint_scores": torch.ones((1, 2)),
                "descriptors": torch.ones((1, 2, 256)),
            }

    extractor = FakeDetector()
    features = extractor.extract(torch.zeros((3, 480, 640)), resolution=resolution)
    assert extractor.shape == expected
    scale = {"2k": 2.4, "1k": 1.6, "native": 1.0}[resolution]
    assert torch.allclose(
        features["keypoints"][0, 0], (torch.tensor([24.0, 48.0]) + 0.5) / scale - 0.5
    )
    assert features["image_size"].tolist() == [[640, 480]]
    if resolution == "2k":
        assert features["keypoints"].shape == (1, 1, 2)
        assert features["descriptors"].shape == (1, 1, 256)


def _gyro_signal(delay=0.0, start=0.0, rate=200):
    time = start + np.arange(15 * rate) / rate
    shifted = time + delay
    magnitude = (
        10
        + np.sin(13 * shifted + 0.4 * shifted**2)
        + 0.7 * np.cos(29 * shifted)
        + 0.2 * np.sin(67 * shifted)
    )
    return np.column_stack((time, magnitude, np.zeros((len(time), 2))))


@pytest.mark.parametrize("delay", [-1.375, 0.0, 2.125])
def should_correlate_gyro_with_different_axes_origins_and_sample_rates(delay):
    from hmlib.stitching.imu_synchronization import estimate_offset

    right = _gyro_signal(delay, 0.035, 400)
    right[:, 3] = -3 * right[:, 1]
    right[:, 1] = 0
    assert estimate_offset(_gyro_signal(), right) == pytest.approx(delay, abs=0.006)


def should_reject_constant_ambiguous_and_invalid_gyro():
    from hmlib.stitching.imu_synchronization import estimate_offset

    constant = _gyro_signal()
    constant[:, 1] = 1
    with pytest.raises(ValueError, match="weak or constant"):
        estimate_offset(constant, constant)
    periodic = _gyro_signal()
    periodic[:, 1] = 2 + np.sin(periodic[:, 0] * 10)
    with pytest.raises(ValueError, match="ambiguous"):
        estimate_offset(periodic, periodic)
    invalid = _gyro_signal()
    invalid[10, 0] = invalid[9, 0]
    with pytest.raises(ValueError, match="non-increasing"):
        estimate_offset(invalid, _gyro_signal())


def should_parse_nested_gopro_gyro_with_local_signed_scales():
    import struct

    from hmlib.stitching.imu_synchronization import parse_gpmf

    def tag(key, kind, width, count, data):
        return struct.pack(">4sBBH", key, kind, width, count) + data + b"\0" * (-len(data) % 4)

    gyro = tag(b"SCAL", ord("l"), 4, 1, struct.pack(">i", 100)) + tag(
        b"GYRO", ord("s"), 6, 2, struct.pack(">hhhhhh", 100, -200, 300, 400, -500, 600)
    )
    unrelated = tag(b"SCAL", ord("f"), 4, 5, b"\0" * 20)
    payload = tag(b"STRM", 0, 1, len(unrelated), unrelated) + tag(b"STRM", 0, 1, len(gyro), gyro)
    packet = tag(b"DEVC", 0, 1, len(payload), payload)
    assert parse_gpmf(packet, 2, 1).tolist() == [[2, 1, -2, 3], [2.5, 4, -5, 6]]
    with pytest.raises(ValueError, match="Truncated"):
        parse_gpmf(packet[:-1], 0, 1)


def _append_insta_gyro(path, raw, delay):
    import struct

    def varint(value):
        result = b""
        while True:
            byte = value & 127
            value >>= 7
            result += bytes([byte | (128 if value else 0)])
            if not value:
                return result

    def field(key, value):
        return varint(key << 3) + varint(value)

    gyro = bytearray()
    for row in _gyro_signal(delay):
        timestamp = round(1000000 + (row[0] + 0.125) * (1000000 if raw else 1000))
        gyro.extend(struct.pack("<Q", timestamp))
        if raw:
            gyro.extend(
                struct.pack("<6H", 32768, 32768, 32768, 32768 + round(row[1] * 100), 32768, 32768)
            )
        else:
            gyro.extend(struct.pack("<6d", 0, 0, 0, row[1], 0, 0))
    trailer = bytes(gyro) + struct.pack("<HI", 0x300, len(gyro))
    info = (
        field(24, 1000000)
        + field(62, int(raw))
        + varint((28 << 3) | 1)
        + struct.pack("<d", 125)
        + field(29, 1)
    )
    trailer += info + struct.pack("<HI", 0x100, len(info)) + b"\0" * 32
    trailer += struct.pack("<II", len(trailer) + 40, 1) + b"8db42d694ccc418790edff439fe026bf"
    with path.open("ab") as output:
        output.write(trailer)


def should_demux_real_insta_trailers_without_decoding_video(game):
    import subprocess

    from hmlib.stitching.imu_synchronization import read_gyro, synchronize_by_imu

    left, right = game / "left.mp4", game / "right.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=30",
            "-t",
            "0.1",
            "-c:v",
            "mpeg4",
            str(left),
        ],
        check=True,
        timeout=30,
    )
    right.write_bytes(left.read_bytes())
    _append_insta_gyro(left, True, 0)
    _append_insta_gyro(right, False, 1.25)
    a, fps = read_gyro(left)
    b, _ = read_gyro(right)
    assert fps == 30
    assert a[0, 0] == pytest.approx(0)
    assert b[0, 0] == pytest.approx(0)
    assert synchronize_by_imu(str(left), str(right)) == pytest.approx((37.5, 0), abs=0.2)


def should_route_imu_and_report_auto_audio_fallback(game, monkeypatch, caplog):
    import hmlib.stitching.imu_synchronization as imu
    import hmlib.stitching.synchronize as sync

    monkeypatch.setattr(sync, "get_game_config_private", lambda game_id: {})
    saved = []
    monkeypatch.setattr(sync, "save_private_config", lambda **kwargs: saved.append(kwargs["data"]))
    monkeypatch.setattr(sync, "synchronize_by_audio", lambda **kwargs: (0, 3))

    def unavailable(left, right):
        raise ValueError("no gyro samples")

    monkeypatch.setattr(imu, "synchronize_by_imu", unavailable)
    with pytest.raises(ValueError, match="no gyro"):
        sync.configure_synchronization("g", "/left.mp4", "/right.mp4", sync_method="imu")
    assert not saved
    assert sync.configure_synchronization("g", "/left.mp4", "/right.mp4", sync_method="auto") == {
        "left": 0,
        "right": 3,
    }
    assert "using audio" in caplog.text


def should_preserve_explicit_manual_offsets_even_when_recalibrating(game, monkeypatch):
    import hmlib.stitching.synchronize as sync

    monkeypatch.setattr(
        sync,
        "get_game_config_private",
        lambda game_id: {"stitching": {"frame_offsets": {"left": 1.25, "right": 0}}},
    )
    assert sync.configure_synchronization(
        "g", "/left.mp4", "/right.mp4", force=True, sync_method="manual"
    ) == {"left": 1.25, "right": 0}


def should_fence_reference_image_changes_and_pending_geometry(game):
    _generation(game)
    workspace = CalibrationWorkspace(game)
    snapshot = workspace.snapshot()
    Image.new("RGB", (400, 200), (1, 2, 3)).save(game / "s.png")
    with pytest.raises(ValueError, match="changed"):
        workspace.set_scoreboard(
            [[10, 10], [100, 10], [100, 30], [10, 30]], expected_revision=snapshot["revision"]
        )
    workspace.update_settings({"stitching.camera_fov.horizontal_fov": 115})
    with pytest.raises(ValueError, match="recalibrate"):
        workspace.save_rink_map(_landmarks())
    with pytest.raises(ValueError, match="recalibrate"):
        workspace.save_control_points(
            [[300, 40, 100, 40], [350, 40, 150, 40], [300, 150, 100, 150], [350, 150, 150, 150]]
        )


def should_expose_actual_lens_filtering_and_independent_automation(game):
    workspace = CalibrationWorkspace(game)
    workspace.update_settings(
        {
            "stitching.camera_config": "gopro-mission-1-factory",
            "stitching.control_point_filtering.epipolar_ransac": False,
        }
    )
    values = workspace.snapshot()["settings"]
    assert values["stitching.input_lens.projection"] == "full-frame-fisheye"
    assert values["stitching.input_lens.optimization"] == "fixed"
    assert not values["stitching.camera_automation.lens"]
    assert not values["stitching.control_point_filtering.epipolar_ransac"]
    workspace.update_settings({"stitching.camera_automation.fov": True})
    assert workspace.snapshot()["settings"]["stitching.camera_automation.fov"]


def should_edit_single_source_scoreboard_and_fence_recording_replacement(game, monkeypatch):
    import subprocess

    from hmlib.scoreboard import proposal

    source = game / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=5",
            "-t",
            "1",
            str(source),
        ],
        check=True,
        timeout=30,
    )
    (game / "config.yaml").write_text(
        yaml.safe_dump({"game": {"videos": {"source": ["source.mp4"]}}})
    )
    editor = CalibrationWorkspace(game)
    snapshot = editor.snapshot()
    assert (snapshot["scoreboard"]["width"], snapshot["scoreboard"]["height"]) == (320, 180)
    assert Image.open(snapshot["scoreboard"]["image"]).size == (320, 180)
    points = [[10, 10], [100, 10], [100, 40], [10, 40]]
    monkeypatch.setattr(proposal, "propose", lambda image, size: points)
    draft = editor.propose_scoreboard(expected_revision=snapshot["revision"])
    assert draft["points"] == points
    assert "rink" not in _private(game)
    editor.set_scoreboard(points, expected_revision=draft["revision"])
    monkeypatch.setattr(
        proposal, "propose", lambda *args: pytest.fail("Saved corners bypass inference")
    )
    assert editor.propose_scoreboard()["points"] == points
    saved_revision = editor.snapshot()["revision"]
    source.touch()
    with pytest.raises(ValueError, match="changed"):
        editor.set_scoreboard(points, expected_revision=saved_revision)
