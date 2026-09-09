from __future__ import annotations

import io
import json
import math
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from hmlib.stitching.artifacts import stitching_lock
from hmlib.stitching.leveling_editor import LevelingSession
from hmlib.stitching.projections import read_panorama_geometry
from hmlib.stitching.rink_leveling import (
    estimate_leveling,
    format_points,
    parse_rays,
    prepare_project,
    rotation_delta,
    rotation_matrix,
)
from hmlib.stitching.settings import read_stitching_settings


def _ray_lines(desired=(0, -18, 7), published=(0, 0, 0), count=7):
    lines = []
    for longitude in np.linspace(-60, 60, count):
        angle = math.radians(longitude)
        lines.append(
            [[math.cos(angle), math.sin(angle), -0.2], [math.cos(angle), math.sin(angle), 0.25]]
        )
    return np.array(lines) @ rotation_matrix(desired) @ rotation_matrix(published).T


def should_fit_absolute_tilt_from_previously_rotated_camera_rays():
    result = estimate_leveling(_ray_lines(published=(27, -11, 12)), (27, -11, 12), 31)
    assert result.rotation_degrees == pytest.approx([31, -18, 7], abs=1e-10)
    assert result.inlier_indices == tuple(range(7))
    assert result.rms_residual_degrees < 1e-10


def should_reject_outliers_and_report_which_posts_were_used():
    lines = _ray_lines()
    lines[-1] = [[1, -0.4, 0.1], [1, 0.4, 0.1]]
    result = estimate_leveling(lines, [0, 0, 0], 12)
    assert result.rotation_degrees == pytest.approx([12, -18, 7])
    assert result.inlier_indices == tuple(range(6))
    assert result.residual_degrees[-1] > 3


def should_reject_repeated_posts_and_short_rays():
    with pytest.raises(ValueError, match="disagree|farther apart"):
        estimate_leveling(np.repeat(_ray_lines()[:1], 3, axis=0), [0, 0, 0], 0)
    with pytest.raises(ValueError, match="taller"):
        estimate_leveling(np.ones((3, 2, 3)), [0, 0, 0], 0)


@pytest.mark.parametrize(
    "published,desired",
    [([15, -25, 17], [-27, 10, -9]), ([0, 0, 0], [30, 90, 22]), ([0, 0, 0], [30, -90, 22])],
)
def should_compose_rotation_delta_including_gimbal_lock(published, desired):
    delta = rotation_matrix(rotation_delta(published, desired))
    np.testing.assert_allclose(
        delta @ rotation_matrix(published), rotation_matrix(desired), atol=1e-12
    )


def should_prepare_private_equirectangular_project_with_quoted_filenames():
    source = 'p f0 w500 h250 v90\ni w640 h480 TrX0 TrY0 TrZ0 n"camera TrX1 left.png"\ni w640 h480 TrX=0 n"right.png"\n'
    result = prepare_project(source)
    assert result.image_sizes == ((640, 480), (640, 480))
    assert result.pto.startswith('p f2 w3600 h1800 v360 n"TIFF_m c:LZW r:CROP"\n')
    assert result.pto.splitlines()[1:] == source.splitlines()[1:]
    for invalid in (source.replace("TrX0", "TrX1"), source.replace("TrX=0", "TrX=2")):
        with pytest.raises(ValueError, match="translation|missing source"):
            prepare_project(invalid)


def should_validate_original_source_coordinates_and_pano_trafo_output():
    posts = [{"image_index": 0, "first": [0, 0], "second": [19, 9]} for _ in range(3)]
    assert len(format_points(posts, [(20, 10)]).splitlines()) == 6
    posts[0]["first"] = [20, 0]
    with pytest.raises(ValueError, match="outside"):
        format_points(posts, [(20, 10)])
    rays = parse_rays("1799.5 899.5\n1799.5 -0.5\n" * 3, 3)
    np.testing.assert_allclose(rays[0], [[1, 0, 0], [0, 0, 1]], atol=1e-15)
    for invalid in ("nan 899.5\n" * 6, "3600 0\n" * 6, "1 2 3\n" * 6):
        with pytest.raises(ValueError):
            parse_rays(invalid, 3)


def _game(tmp_path, *, rotation=None):
    config = {
        "stitching": {
            "mapping_backend": "nona",
            "run_autooptimizer": True,
            "projection": "equirectangular",
            "projection_framing": {"auto_canvas": False, "horizontal_fov": 120},
            "rink_config": "rink",
            "rink_configs": {"rink": {"rotation_degrees": [0, -2, 1]}},
        }
    }
    if rotation is not None:
        config["stitching"]["projection_framing"]["rotation_degrees"] = rotation
    Image.new("RGB", (160, 100), (40, 100, 150)).save(tmp_path / "left camera.png")
    (tmp_path / "autooptimiser_out.pto").write_text(
        'p f2 w240 h120 v120 n"PNG"\nm i0\ni w160 h100 f0 v90 y0 p0 r0 n"left camera.png"\n'
    )
    (tmp_path / ".stitching_artifacts.json").write_text(
        json.dumps(read_stitching_settings(config).manifest())
    )
    (tmp_path / "config.yaml").write_text("unrelated: keep\n")
    return config


def _fake_hugin(command, *, input=None):
    if command[0] == "pano_modify":
        destination, source = Path(command[command.index("-o") + 1]), Path(command[-1])
        text = source.read_text()
        if "--crop=AUTO" in command:
            text = text.replace('v120 n"PNG"', 'v120 S24,216,12,108 n"PNG"')
        destination.write_text(text)
    elif command[0] == "nona":
        output = Path(command[command.index("-o") + 1])
        geometry = read_panorama_geometry(command[-1])
        Image.new("RGB", (geometry.width, geometry.height), (50, 120, 180)).save(output)
    elif command[0] == "pano_trafo":
        assert len(input.splitlines()) == 6
        return "1000 1000\n1000 700\n1800 1000\n1800 700\n2600 1000\n2600 700\n"
    else:
        raise AssertionError(command)
    return ""


def _state(session):
    info = session.info()
    return {key: info[key] for key in ("rotation_degrees", "crop", "auto_crop")} | {"posts": []}


def should_save_previewed_crop_without_materializing_inherited_rotation(tmp_path):
    config = _game(tmp_path)
    session = LevelingSession(tmp_path, lambda: config, run=_fake_hugin)
    try:
        state = _state(session)
        state["crop"] = [0.1, 0.9, 0.2, 0.8]
        preview = session.preview(state)
        assert (preview["width"], preview["height"]) == (240, 120)
        assert preview["crop"] == state["crop"]
        session.save(state, preview["token"])
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert saved["unrelated"] == "keep"
        assert saved["stitching"]["projection_framing"] == {
            "crop": [0.1, 0.9, 0.2, 0.8],
            "auto_crop": False,
        }
        assert (tmp_path / "autooptimiser_out.pto").read_text().startswith("p f2 w240")
        with pytest.raises(ValueError, match="Preview|already saved"):
            session.save(state, preview["token"])
    finally:
        session.close()


@pytest.mark.parametrize(
    "changed",
    [
        "config.yaml",
        "autooptimiser_out.pto",
        ".stitching_artifacts.json",
        "left camera.png",
        "settings",
        "inherited_config",
        "state",
        "token",
    ],
)
def should_refuse_stale_or_unpreviewed_saves(tmp_path, changed):
    config = _game(tmp_path)
    session = LevelingSession(tmp_path, lambda: config, run=_fake_hugin)
    try:
        state = _state(session)
        preview = session.preview(state)
        if changed == "settings":
            config["stitching"]["max_output_dimension"] = 1000
        elif changed == "inherited_config":
            config["stitching"]["frame_offsets"] = {"left": 12, "right": 18}
        elif changed == "state":
            state["rotation_degrees"][1] += 1
        elif changed == "token":
            preview["token"] = "old"
        else:
            with (tmp_path / changed).open("ab") as stream:
                stream.write(b"\n")
        before = (tmp_path / "config.yaml").read_bytes()
        with pytest.raises(ValueError, match="changed|Preview"):
            session.save(state, preview["token"])
        assert (tmp_path / "config.yaml").read_bytes() == before
    finally:
        session.close()


def should_preserve_published_rotation_for_relative_preview_and_save_explicit_zero(tmp_path):
    config = _game(tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return _fake_hugin(command, **kwargs)

    session = LevelingSession(tmp_path, lambda: config, run=run)
    try:
        state = _state(session)
        state["rotation_degrees"] = [0, 0, 0]
        preview = session.preview(state)
        argument = next(arg for arg in commands[0] if arg.startswith("--rotate="))
        assert [float(value) for value in argument.split("=")[1].split(",")] == pytest.approx(
            rotation_delta([0, -2, 1], [0, 0, 0])
        )
        session.save(state, preview["token"])
        assert yaml.safe_load((tmp_path / "config.yaml").read_text())["stitching"][
            "projection_framing"
        ]["rotation_degrees"] == [0, 0, 0]
    finally:
        session.close()


def should_show_auto_crop_on_full_canvas_and_save_auto_mode(tmp_path):
    config = _game(tmp_path)
    session = LevelingSession(tmp_path, lambda: config, run=_fake_hugin)
    try:
        state = _state(session) | {"auto_crop": True}
        preview = session.preview(state)
        assert preview["crop"] == [0.1, 0.9, 0.1, 0.9]
        with Image.open(io.BytesIO(session.preview_image)) as image:
            assert image.size == (240, 120)
        session.save(state, preview["token"])
        assert yaml.safe_load((tmp_path / "config.yaml").read_text())["stitching"][
            "projection_framing"
        ]["auto_crop"]
    finally:
        session.close()


def should_fail_busy_editor_operations_without_waiting_for_calibration(tmp_path):
    config = _game(tmp_path)
    session = LevelingSession(tmp_path, lambda: config, run=_fake_hugin)
    acquired, release = threading.Event(), threading.Event()

    def hold_lock():
        with stitching_lock(tmp_path):
            acquired.set()
            release.wait(10)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    try:
        assert acquired.wait(5)
        with pytest.raises(BlockingIOError):
            session.preview(_state(session))
    finally:
        release.set()
        thread.join()
        session.close()


def should_reject_camera_settings_changed_since_calibration(tmp_path):
    config = _game(tmp_path)
    config["stitching"]["camera_fov"] = {"horizontal_fov": 115}
    with pytest.raises(ValueError, match="recalibrate"):
        LevelingSession(tmp_path, lambda: config, run=_fake_hugin)


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("pano_modify", "pano_trafo", "nona")),
    reason="Hugin tools unavailable",
)
def should_render_and_transform_using_real_hugin_without_changing_published_project(tmp_path):
    config = _game(tmp_path, rotation=[0, 0, 0])
    original = (tmp_path / "autooptimiser_out.pto").read_bytes()
    session = LevelingSession(tmp_path, lambda: config)
    try:
        state = _state(session)
        state["rotation_degrees"] = [4, -8, 3]
        preview = session.preview(state)
        assert preview["width"] <= 1920 and preview["height"] <= 1920
        posts = [{"image_index": 0, "first": [x, 10], "second": [x, 90]} for x in (10, 80, 150)]
        estimate = session.estimate(posts, 7)
        assert estimate["rotation_degrees"] == pytest.approx([7, 0, 0], abs=1e-5)
        assert (tmp_path / "autooptimiser_out.pto").read_bytes() == original
    finally:
        session.close()
