from __future__ import annotations

import copy
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from hmlib.stitching import configure_stitching
from hmlib.stitching.projections import apply_projection, read_panorama_geometry
from hmlib.stitching.settings import (
    PROJECTIONS,
    maximum_projection_fov,
    projection_parameters,
    read_stitching_settings,
)


def _nona(**values):
    return read_stitching_settings(
        {
            "stitching": {
                "mapping_backend": "nona",
                "run_autooptimizer": True,
                **values,
            }
        }
    )


def should_resolve_camera_and_inherit_rink_without_mutating_config():
    config = {
        "stitching": {
            "camera_config": "gopro-mission-1",
            "camera_fov": {"vertical_fov": 98},
            "rink_config": "venue",
            "rink_configs": {"venue": {"rotation_degrees": [0, -23, 4]}},
        }
    }
    original = copy.deepcopy(config)
    settings = read_stitching_settings(config)
    assert (settings.horizontal_fov, settings.vertical_fov) == (127.2, 98)
    assert settings.framing.rotation_degrees == (0, -23, 4)
    assert config == original
    config["stitching"]["projection_framing"] = {"rotation_degrees": [0, 0, 0]}
    assert read_stitching_settings(config).framing.rotation_degrees == (0, 0, 0)
    config["stitching"]["projection_framing"]["rotation_degrees"] = None
    assert read_stitching_settings(config).framing.rotation_degrees == (0, -23, 4)


def should_accept_custom_camera_with_complete_explicit_fov():
    settings = read_stitching_settings(
        {
            "stitching": {
                "camera_config": "custom-camera",
                "camera_fov": {"horizontal_fov": 115, "vertical_fov": 80},
            }
        }
    )
    assert settings.horizontal_fov == 115
    with pytest.raises(ValueError, match="vertical_fov"):
        read_stitching_settings(
            {
                "stitching": {
                    "camera_config": "custom-camera",
                    "camera_fov": {"horizontal_fov": 115},
                }
            }
        )


@pytest.mark.parametrize(
    "values, message",
    [
        ({"mapping_backend": "nona"}, "run_autooptimizer"),
        ({"projection": "general-panini"}, "only rectilinear"),
        ({"run_autooptimizer": "nope"}, "true or false"),
        ({"camera_fov": {"horizontal_fov": float("nan")}}, "finite"),
        ({"rink_config": "missing-rink"}, "Unknown"),
        ({"projection_framing": {"crop": [0.9, 0.1, 0, 1]}}, "crop"),
        ({"projection_framing": {"auto_crop": True, "crop": [0, 0.5, 0, 1]}}, "mutually exclusive"),
        ({"projection_framing": {"rotation_degrees": [0, 181, 0]}}, "-180 and 180"),
        ({"max_output_dimension": 10.5}, "max_output_dimension"),
        ({"max_output_width": True}, "finite number"),
    ],
)
def should_reject_invalid_settings_before_mutating_artifacts(tmp_path, values, message):
    sentinel = tmp_path / "hm_project.pto"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match=message):
        configure_stitching.configure_video_stitching(
            str(tmp_path),
            "missing-left.mp4",
            "missing-right.mp4",
            100,
            game_config={"stitching": values},
            force=True,
        )
    assert sentinel.read_text() == "original"
    assert not (tmp_path / ".stitching.lock").exists()


@pytest.mark.parametrize("projection", PROJECTIONS)
def should_accept_every_hugin_projection_with_auto_fov(projection):
    settings = _nona(projection=projection, projection_framing={"auto_fov": True})
    assert settings.projection == projection


def should_validate_projection_parameters_and_dynamic_fov():
    assert maximum_projection_fov("general-panini", (0, 0, 0)) == pytest.approx(160)
    assert maximum_projection_fov("general-panini", (100, 0, 0)) > 300
    assert maximum_projection_fov("biplane", (45, 0)) == 224
    with pytest.raises(ValueError, match="projection limit"):
        _nona(projection="rectilinear")
    with pytest.raises(ValueError, match="increments"):
        projection_parameters("general-panini", [100.001, 0, 0])
    with pytest.raises(ValueError, match="exactly"):
        projection_parameters("biplane", [45, 0.5])


def should_fingerprint_effective_settings_and_preserve_inactive_tuning():
    config = {"stitching": {"projection_parameters": {"general-panini": [50, 1, 2]}}}
    before = copy.deepcopy(config)
    settings = read_stitching_settings(config)
    assert settings.parameters == ()
    assert config == before
    assert settings.manifest() != replace(settings, horizontal_fov=120).manifest()
    assert (
        json.loads(settings.manifest()["calibration_settings"])["camera_config"]
        == "gopro-mission-1"
    )


def should_invalidate_cache_after_camera_or_framing_change(tmp_path):
    for name in (
        "hm_project.pto",
        "autooptimiser_out.pto",
        "seam_file.png",
        *(f"mapping_{index:04d}{suffix}.tif" for index in range(2) for suffix in ("", "_x", "_y")),
    ):
        (tmp_path / name).touch()
    settings = _nona()
    (tmp_path / ".stitching_artifacts.json").write_text(json.dumps(settings.manifest()))
    args = (tmp_path / "hm_project.pto", tmp_path / "autooptimiser_out.pto")
    assert configure_stitching._stitch_project_is_complete(*args, settings=settings)
    for modified in (
        replace(settings, horizontal_fov=110),
        replace(settings, framing=replace(settings.framing, rotation_degrees=(0, -20, 0))),
        replace(settings, max_output_width=1920),
    ):
        assert not configure_stitching._stitch_project_is_complete(*args, settings=modified)


def _pto(projection=19, fov=180, parameters=' P"100 0 0"', crop=""):
    return f'p f{projection} w1000 h500 v{fov}{parameters}{crop} n"TIFF_m c:LZW r:CROP"\n'


@pytest.mark.parametrize(
    "pto, message",
    [
        (_pto(projection=1), "requested projection"),
        (_pto(fov=170), "clamped"),
        (_pto(parameters=' P"99 0 0"'), "parameters"),
        (_pto(crop=" S-1,1000,0,500"), "crop"),
    ],
)
def should_reject_changed_hugin_output_without_replacing_input(tmp_path, pto, message):
    project = tmp_path / "autooptimiser_out.pto"
    original = _pto(projection=2, parameters="")
    project.write_text(original)

    def run(command):
        Path(command[command.index("-o") + 1]).write_text(pto)

    with pytest.raises(ValueError, match=message):
        apply_projection(project, _nona(), run)
    assert project.read_text() == original
    assert list(tmp_path.iterdir()) == [project]


def should_apply_rotation_and_manual_crop_before_mapping(tmp_path):
    project = tmp_path / "autooptimiser_out.pto"
    project.write_text(_pto(projection=2, parameters=""))
    commands = []
    settings = _nona(
        projection_framing={"crop": [0.1, 0.9, 0.2, 0.8], "rotation_degrees": [0, -25, 2]}
    )

    def run(command):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_text(_pto(crop=" S100,900,100,400"))

    result = apply_projection(project, settings, run)
    assert result.effective_size == (800, 300)
    assert "--rotate=0,-25,2" in commands[0]
    assert "--crop=10,90,20,80%" in commands[0]
    assert "--projection-parameter=100 0 0" in commands[0]


@pytest.mark.skipif(
    not shutil.which("pano_modify") or not shutil.which("pto_gen"),
    reason="Hugin binaries unavailable",
)
def should_apply_projection_and_canvas_cap_with_real_hugin(tmp_path):
    images = [tmp_path / "left.png", tmp_path / "right.png"]
    for image in images:
        assert cv2.imwrite(str(image), np.zeros((240, 320, 3), dtype=np.uint8))
    project = tmp_path / "autooptimiser_out.pto"
    subprocess.run(
        ["pto_gen", "-p", "0", "-f", "108", "-o", str(project), *map(str, images)],
        check=True,
        capture_output=True,
    )
    settings = _nona(
        max_output_width=200,
        projection_framing={
            "auto_fov": True,
            "rotation_degrees": [0, -25, 0],
            "crop": [0.1, 0.9, 0.1, 0.9],
        },
    )
    result = apply_projection(
        project, settings, lambda command: subprocess.run(command, check=True, capture_output=True)
    )
    assert result.width <= 200
    assert result.projection == 19
    assert result.parameters == (100, 0, 0)
    assert read_panorama_geometry(project) == result


def should_forward_effective_config_to_calibration_worker(monkeypatch, tmp_path):
    config = {
        "stitching": {
            "control_point_matcher": "loftr",
            "camera_config": "gopro-hero-11",
            "max_output_width": 1920,
        }
    }
    captured = {}

    def worker(**kwargs):
        captured.update(kwargs)
        return "pto", 0, 0

    monkeypatch.setattr(configure_stitching, "_configure_video_stitching_locked", worker)
    assert configure_stitching.configure_video_stitching(
        str(tmp_path), "left", "right", 1500, game_config=config
    ) == ("pto", 0, 0)
    assert captured["settings"].control_point_matcher == "loftr"
    assert captured["settings"].horizontal_fov == 108
    assert captured["settings"].max_output_width == 1920
