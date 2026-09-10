from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from hmlib.stitching import akaze, configure_stitching
from hmlib.stitching.akaze import load_lens_calibration
from hmlib.stitching.calibration import CalibrationAlignmentError
from hmlib.stitching.control_points import calculate_control_points, normalize_control_point_matcher
from hmlib.stitching.settings import read_stitching_settings


@pytest.fixture
def native_detector(monkeypatch):
    """Run the same production binding without importing the CUDA extension."""
    root = (
        Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        if "TEST_SRCDIR" in os.environ
        else Path(__file__).resolve().parents[1] / "bazel-bin"
    )
    library = Path(
        os.environ.get(
            "HM_AKAZE_TEST_EXTENSION", root / "hockeymon/csrc/stitcher/_akaze_test_native.so"
        )
    )
    if library.is_file():
        spec = importlib.util.spec_from_file_location("_akaze_test_native", library)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(akaze, "_detect_akaze", module.detect_akaze_features)


def _profile():
    camera = {
        "width": 800,
        "height": 480,
        "fx": 500,
        "fy": 490,
        "cx": 400,
        "cy": 240,
        "d": [0.01, -0.02, 0.003, 0],
    }
    return {"left_uniforms": camera.copy(), "right_uniforms": camera.copy()}


def should_match_translated_camera_overlap_without_learned_models(native_detector):
    rng = np.random.default_rng(37)
    image = rng.integers(0, 256, size=(480, 1200, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    assert normalize_control_point_matcher("AKAZE") == "akaze-hamming"
    matches = calculate_control_points(
        image[:, :800], image[:, 400:], 100, matcher="akaze", device=torch.device("cpu")
    )
    assert len(matches["m_kpts0"]) >= 20
    difference = matches["m_kpts0"] - matches["m_kpts1"]
    torch.testing.assert_close(
        difference, torch.tensor([400.0, 0.0]).expand_as(difference), atol=1.0, rtol=0
    )


def should_reject_textureless_akaze_pair_as_alignment_failure(native_detector):
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(CalibrationAlignmentError, match="descriptors"):
        calculate_control_points(blank, blank, 100, matcher="akaze")


def should_load_and_fingerprint_the_exact_paired_profile(tmp_path):
    contents = json.dumps(_profile()).encode()
    (tmp_path / "left_calibration.json").write_bytes(contents)
    calibration = load_lens_calibration(tmp_path)
    assert calibration.fingerprint == hashlib.sha256(contents).hexdigest()
    np.testing.assert_allclose(
        calibration.left.camera_matrix(400, 240), [[250, 0, 200], [0, 245, 120], [0, 0, 1]]
    )
    assert len(calibration.right.native_values()) == 10
    settings = read_stitching_settings(control_point_matcher="akaze")
    assert (
        replace(settings, lens_profile_fingerprint=calibration.fingerprint).manifest()
        != settings.manifest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: profile.pop("right_uniforms"),
        lambda profile: profile["left_uniforms"].update(width=10.5),
        lambda profile: profile["left_uniforms"].update(fx=-1),
        lambda profile: profile["right_uniforms"].update(d=[0, 0, 0]),
        lambda profile: profile["right_uniforms"].update(cy=float("nan")),
    ],
)
def should_fail_closed_on_existing_invalid_lens_profiles(tmp_path, mutation):
    profile = _profile()
    mutation(profile)
    (tmp_path / "left_calibration.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="AKAZE"):
        load_lens_calibration(tmp_path)


def should_reject_nonregular_and_oversized_profiles_before_reading(tmp_path):
    path = tmp_path / "left_calibration.json"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        load_lens_calibration(tmp_path)
    path.unlink()
    with path.open("wb") as stream:
        stream.truncate(1024 * 1024 + 1)
    with pytest.raises(ValueError, match="1 MiB"):
        load_lens_calibration(tmp_path)


def should_allow_missing_profiles_with_explicit_diagnostic(tmp_path, caplog):
    assert load_lens_calibration(tmp_path) is None
    assert "matching original camera images" in caplog.text


def should_reject_calibrated_akaze_with_nona_before_mutation(tmp_path):
    (tmp_path / "left_calibration.json").write_text(json.dumps(_profile()))
    with pytest.raises(ValueError, match="NONA does not consume KB4"):
        configure_stitching.configure_video_stitching(
            str(tmp_path),
            "left.mp4",
            "right.mp4",
            100,
            game_config={
                "stitching": {
                    "control_point_matcher": "akaze",
                    "mapping_backend": "nona",
                    "run_autooptimizer": True,
                }
            },
        )
    assert not (tmp_path / ".stitching.lock").exists()


def should_pin_lens_profile_through_matching_worker(monkeypatch, tmp_path):
    path = tmp_path / "left_calibration.json"
    contents = json.dumps(_profile())
    path.write_text(contents)
    captured = {}

    def worker(**kwargs):
        path.write_text("malformed replacement")
        captured.update(kwargs)
        return "pto", 0, 0

    monkeypatch.setattr(configure_stitching, "_configure_video_stitching_locked", worker)
    configure_stitching.configure_video_stitching(
        str(tmp_path),
        "left",
        "right",
        100,
        game_config={"stitching": {"control_point_matcher": "akaze"}},
    )
    assert (
        captured["settings"].lens_profile_fingerprint
        == hashlib.sha256(contents.encode()).hexdigest()
    )
    assert captured["lens_calibration"].fingerprint == captured["settings"].lens_profile_fingerprint
