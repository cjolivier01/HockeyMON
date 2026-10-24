from __future__ import annotations

import math
import struct
from dataclasses import replace

import numpy as np
import pytest

from hmlib.stitching.geometric_filtering import ControlPointFiltering, geometric_inliers
from hmlib.stitching.input_lens import InputLens, apply_input_lens, read_input_lens
from hmlib.stitching.settings import read_stitching_settings
from hmlib.stitching.video_lens import (
    hugin_fov,
    parse_dji_lens,
    parse_gopro_lens,
    parse_insta_lens,
    read_video_lens,
    resolve_video_lens,
)


def _tag(key, kind, data, width=None):
    width = width or (4 if kind in ("f", "L") else 1)
    return (
        struct.pack(">4sBBH", key.encode(), ord(kind) if kind else 0, width, len(data) // width)
        + data
        + b"\0" * ((-len(data)) % 4)
    )


def _f(*values):
    return struct.pack(">" + "f" * len(values), *values)


def _gopro(*, prescription=True, warp=False, fov=True):
    global_tags = b"".join(
        [
            _tag("MINF", "c", b"MISSION 1 PRO"),
            _tag("PRJT", "F", b"GPRO", 4),
            _tag("EISA", "c", b"HS EIS"),
            _tag("HCTL", "c", b"Off"),
            _tag("ZOOM", "f", _f(1)),
            _tag("VFPS", "L", struct.pack(">II", 60000, 1001)),
            _tag("VRES", "L", struct.pack(">II", 7680, 4320)),
        ]
    )
    lens = b"".join(
        [
            _tag("DVID", "F", b"FOVL", 4),
            _tag("VFOV", "c", b"W"),
            _tag("ABSC", "f", _f(0)),
            _tag("MAPX", "f", _f(2 if warp else 1)),
            _tag("MAPY", "f", _f(1)),
            _tag("ARUW", "f", _f(7680 / 4320)),
            _tag("ARWA", "f", _f(7680 / 4320)),
            _tag("ZMPL", "f", _f(0.476107776165009)),
        ]
    )
    if fov:
        lens += (
            _tag("XFOV", "f", _f(120.000427246094))
            + _tag("YFOV", "f", _f(70.3657608032227))
            + _tag("ZFOV", "f", _f(135.511917114258))
        )
    if prescription:
        lens += _tag(
            "POLY",
            "f",
            _f(
                0,
                2.67053532600403,
                0.22922171652317,
                -2.08499050140381,
                1.64104437828064,
                -7.24889386797406e-13,
                4.47891317133589e-13,
                0,
            ),
            32,
        )
        lens += _tag("PYCF", "c", b"".join(f"r{i}".encode() + b"\0" * 6 for i in range(8)), 8)
    return _tag("DEVC", "", global_tags) + _tag("DEVC", "", lens)


def _atom(name, data):
    return struct.pack(">I4s", len(data) + 8, name.encode()) + data


def _movie(data):
    description = bytearray(100)
    description[12:16] = b"hvc1"
    description[40:44] = struct.pack(">HH", 7680, 4320)
    track = _atom("trak", _atom("mdia", _atom("minf", _atom("stbl", _atom("stsd", description)))))
    return _atom("moov", _atom("udta", _atom("GPMF", data)) + track)


def _varint(value):
    data = b""
    while value >= 128:
        data += bytes([(value & 127) | 128])
        value >>= 7
    return data + bytes([value])


def _field(number, data):
    return _varint((number << 3) | 2) + _varint(len(data)) + data


def _scalar(number, value):
    return _varint(number << 3) + _varint(value)


def should_match_native_mission_factory_fit_and_read_bounded_mp4(tmp_path):
    metadata = parse_gopro_lens(_gopro())
    assert metadata.horizontal_fov == pytest.approx(120.000427246094)
    assert metadata.calibration_fov == pytest.approx(125.12273264898167, abs=1e-7)
    assert metadata.lens.distortion == pytest.approx(
        (-0.004931889213955486, 0.04057437656776939, -0.02182075784509423), abs=1e-8
    )
    path = tmp_path / "camera.mp4"
    path.write_bytes(_movie(_gopro()))
    assert read_video_lens(path) == metadata
    path.write_bytes(_movie(_gopro(prescription=False)))
    assert read_video_lens(path).lens is None
    with pytest.raises(ValueError, match="Truncated"):
        parse_gopro_lens(_gopro()[:-4])


def should_import_axis_fov_without_lens_and_refuse_baked_warp():
    axes = parse_gopro_lens(_gopro(prescription=False))
    assert axes.lens is None and axes.horizontal_fov > 120
    warped = parse_gopro_lens(_gopro(warp=True))
    assert warped.lens is None and warped.horizontal_fov is None
    derived = parse_gopro_lens(_gopro(fov=False))
    assert derived.horizontal_fov == pytest.approx(120.01437255147613)


def should_convert_independent_fov_using_the_selected_radial_normalization():
    lens = parse_gopro_lens(_gopro()).lens
    calibration = hugin_fov(120.000427246094, 7680, 4320, lens)
    assert calibration == pytest.approx(125.10066440223888, abs=1e-7)
    assert hugin_fov(calibration, 7680, 4320, lens, to_physical=True) == pytest.approx(
        120.000427246094
    )
    with pytest.raises(ValueError, match="fold|cover"):
        hugin_fov(120, 7680, 4320, InputLens(distortion=(-0.8, 0.0, 0.0)))


def should_import_dji_only_with_known_disabled_stabilization():
    header = _field(1, b"dvtm_ac204.proto") + _field(10, b"DJI Action")
    clip = (
        _field(1, header)
        + _field(8, b"\x0d" + struct.pack("<f", 3000))
        + _field(3, _field(1, bytes(16)))
    )
    assert parse_dji_lens(_field(1, clip), 7680, 4320).lens is None
    result = parse_dji_lens(_field(1, clip + _field(9, b"")), 7680, 4320)
    assert result.lens.distortion == pytest.approx((0, 0, 0), abs=1e-10)
    assert result.calibration_fov == pytest.approx(math.degrees(7680 / 3000))


def should_import_raw_insta_v3_and_reject_flowstate():
    values = [1, 1, 4000, 4000, 1920, 1080] + [0] * 11 + [3840, 2160, 0, 0]
    data = (
        _field(2, b"Insta360")
        + _field(19, _scalar(1, 3840) + _scalar(2, 2160))
        + _field(54, "_".join(map(str, values)).encode())
    )
    result = parse_insta_lens(data)
    assert result.horizontal_fov > 0
    # The approximation threshold may reject this synthetic unified model, but
    # it still exposes independently measured physical FOV.
    assert parse_insta_lens(data + _scalar(42, 1)).lens is None


def should_resolve_independent_automation_and_freeze_experiments(tmp_path):
    paths = [tmp_path / f"{side}.mp4" for side in ("left", "right")]
    for path in paths:
        path.write_bytes(_movie(_gopro()))
    settings = read_stitching_settings({})
    resolved, notes = resolve_video_lens({}, settings, paths)
    assert resolved.input_lens.projection == "full-frame-fisheye"
    assert resolved.horizontal_fov > 125 and any("imported" in note for note in notes)
    manual_lens = replace(settings, camera_auto_lens=False)
    resolved, _ = resolve_video_lens({}, manual_lens, paths)
    assert resolved.input_lens == InputLens()
    assert resolved.horizontal_fov == pytest.approx(120, abs=0.001)
    frozen = {"stitching": {"auto_input_lens_frozen": True}}
    assert resolve_video_lens(frozen, settings, paths)[0] == settings
    paths[1].write_bytes(_movie(_gopro(prescription=False)))
    mismatched, notes = resolve_video_lens({}, settings, paths)
    assert mismatched == settings and any("differs" in note for note in notes)


def should_apply_profile_coefficients_and_fixed_refinable_hugin_variables():
    settings = read_stitching_settings({"stitching": {"camera_config": "gopro-mission-1-factory"}})
    pto = '#hugin_optimizerMasterSwitch 1\ni w7680 h4320 f0 v108 a0 b0 c0 d0 e0 n"left.png"\ni w7680 h4320 f0 v=0 n"right.png"\nv r1 p1 y1\nv\n'
    fixed = apply_input_lens(pto, settings.input_lens)
    assert "f3" in fixed and "a=0 b=0 c=0 d=0 e=0" in fixed
    assert "v r0 p0 r1 p1 y1\n" in fixed and " v0" not in fixed
    refined = apply_input_lens(
        pto, replace(settings.input_lens, optimization="refine", center_offset=(0.01, -0.02))
    )
    assert " d76.799" in refined and "v0 b0 d0 e0" in refined
    with pytest.raises(ValueError, match="requires mapping_backend=nona"):
        read_stitching_settings(
            {
                "stitching": {
                    "camera_config": "gopro-mission-1-factory",
                    "mapping_backend": "opencv-magsac",
                }
            }
        )
    with pytest.raises(ValueError, match="within"):
        read_input_lens({"distortion": [0, 2, 0]})


@pytest.mark.parametrize("epipolar", [False, True])
def should_reject_epipolar_line_false_match_with_independent_homography(epipolar):
    rng = np.random.default_rng(4)
    left = rng.uniform([0, 0], [1800, 1000], size=(100, 2)).astype(np.float32)
    right = left + [25, 3]
    right[-1, 0] += 200
    kept = geometric_inliers(left, right, ControlPointFiltering(epipolar, True))
    assert 99 not in kept and len(kept) >= 95
    assert len(geometric_inliers(left, right, ControlPointFiltering(False, False))) == 100


def should_decline_homography_filter_without_strong_planar_support():
    rng = np.random.default_rng(17)
    left = rng.uniform(0, 1000, (100, 2))
    right = rng.uniform(0, 1000, (100, 2))
    assert len(geometric_inliers(left, right, ControlPointFiltering(False, True))) == 100
