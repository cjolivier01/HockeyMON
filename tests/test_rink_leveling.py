from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from hmlib.stitching import configure_stitching as shared_stitching
from hmlib.stitching.artifacts import stitching_lock
from hmlib.stitching.calibration_leveling import (
    CalibrationLevelingResult,
    CalibrationLevelingSelector,
    CalibrationLevelingSession,
)
from hmlib.stitching.calibration_leveling_page import PAGE as CALIBRATION_LEVELING_PAGE
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


def should_automatically_estimate_without_an_estimate_button():
    assert 'id="estimate"' not in CALIBRATION_LEVELING_PAGE
    assert "setTimeout" in CALIBRATION_LEVELING_PAGE
    assert ",150)" in CALIBRATION_LEVELING_PAGE
    assert "if(drag){scheduleEstimate();return}" in CALIBRATION_LEVELING_PAGE
    assert "if(hit>=0){clearTimeout(estimateTimer);++estimateSerial" in CALIBRATION_LEVELING_PAGE
    assert "if(finishing||(busy&&action==='use'))return" in CALIBRATION_LEVELING_PAGE
    assert "URL.revokeObjectURL" in CALIBRATION_LEVELING_PAGE
    assert "addEventListener('beforeunload',releaseObjectUrls)" in CALIBRATION_LEVELING_PAGE
    assert "Skip leveling" in CALIBRATION_LEVELING_PAGE
    assert "Cancel calibration" in CALIBRATION_LEVELING_PAGE


def should_run_selector_tools_with_final_calibration_locale_and_context(monkeypatch, tmp_path):
    captured = {}

    def run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stdout="converted")

    monkeypatch.setattr(shared_stitching.subprocess, "run", run)
    token = shared_stitching._command_directory.set(tmp_path)
    try:
        assert (
            shared_stitching._run_stitching_command(
                ["pano_trafo", "sphere.pto"], input_text="0 1 2\n", timeout_seconds=60
            )
            == "converted"
        )
    finally:
        shared_stitching._command_directory.reset(token)
    assert captured["cwd"] == tmp_path
    assert captured["env"]["LC_ALL"] == "C"
    assert captured["input"] == "0 1 2\n"
    assert captured["timeout"] == 60


def should_terminate_a_cancelled_stitching_tool_promptly():
    cancelled = threading.Event()
    timer = threading.Timer(0.25, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.SubprocessError, match="cancelled"):
            shared_stitching._run_stitching_command(
                [
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
                ],
                timeout_seconds=10,
                cancel_event=cancelled,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2


class _SelectorSession:
    published_rotation = (17.0, -9.0, 2.0)
    prepared = SimpleNamespace(image_sizes=((160, 100), (160, 100)))
    source_images = [b"left", b"right"]


class _BlockingSelectorSession(_SelectorSession):
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def _block(self, cancel_event):
        self.started.set()
        if not cancel_event.wait(5):
            raise AssertionError("selector operation was not cancelled")
        self.cancelled.set()
        raise subprocess.SubprocessError("Stitching command cancelled")

    def estimate(self, posts, rotation, *, cancel_event=None):
        return self._block(cancel_event)

    def preview(self, rotation, *, cancel_event=None):
        return self._block(cancel_event)


class _SupersedingSelectorSession(_SelectorSession):
    def __init__(self):
        self.first_started = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def estimate(self, posts, rotation, *, cancel_event=None):
        with self.lock:
            self.calls += 1
            call = self.calls
        if call == 1:
            self.first_started.set()
            if not cancel_event.wait(5):
                raise AssertionError("obsolete estimate was not cancelled")
            raise subprocess.SubprocessError("Stitching command cancelled")
        return {
            "rotation_degrees": [17, 4, -3],
            "inlier_indices": [0, 1, 2],
            "residual_degrees": [0.1, 0.2, 0.3],
            "rms_residual_degrees": 0.2,
        }


def _selector_post(selector, path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{selector.port}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Editor-Token": selector._token,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read())


def _wait_for_selector(selector):
    deadline = time.monotonic() + 3
    while not selector.access_urls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert selector.access_urls


@pytest.mark.parametrize(
    "operation,action,expected_cancel",
    [("estimate", "skip", False), ("preview", "cancel", True)],
)
def should_finish_promptly_while_a_selector_tool_is_active(operation, action, expected_cancel):
    session = _BlockingSelectorSession()
    selector = CalibrationLevelingSelector(
        session, game_id="demo", bind_host="127.0.0.1", open_browser=False
    )
    result = []
    run_thread = threading.Thread(target=lambda: result.append(selector.run()))
    run_thread.start()
    _wait_for_selector(selector)
    operation_errors = []
    payload = (
        {
            "posts": [
                {"image_index": 0, "first": [1, 1], "second": [1, 2]},
                {"image_index": 0, "first": [2, 1], "second": [2, 2]},
                {"image_index": 1, "first": [3, 1], "second": [3, 2]},
            ],
            "rotation": [17, -9, 2],
        }
        if operation == "estimate"
        else {"rotation": [17, 4, -3]}
    )

    def operate():
        try:
            _selector_post(selector, f"/api/{operation}", payload)
        except urllib.error.HTTPError as error:
            operation_errors.append(error.code)

    operation_thread = threading.Thread(target=operate)
    operation_thread.start()
    assert session.started.wait(2)
    started = time.monotonic()
    assert _selector_post(
        selector, "/api/complete", {"action": action, "rotation": [17, 4, -3]}
    ) == {"complete": True}
    operation_thread.join(2)
    run_thread.join(2)
    assert time.monotonic() - started < 2
    assert not operation_thread.is_alive() and not run_thread.is_alive()
    assert session.cancelled.is_set()
    assert operation_errors == [500]
    assert len(result) == 1 and result[0].cancel_calibration is expected_cancel


def should_backend_close_cancel_an_active_selector_tool():
    session = _BlockingSelectorSession()
    selector = CalibrationLevelingSelector(
        session, game_id="demo", bind_host="127.0.0.1", open_browser=False
    )
    result = []
    run_thread = threading.Thread(target=lambda: result.append(selector.run()))
    run_thread.start()
    _wait_for_selector(selector)
    operation_thread = threading.Thread(
        target=lambda: pytest.raises(
            urllib.error.HTTPError,
            _selector_post,
            selector,
            "/api/preview",
            {"rotation": [17, 4, -3]},
        )
    )
    operation_thread.start()
    assert session.started.wait(2)
    started = time.monotonic()
    selector.close()
    operation_thread.join(2)
    run_thread.join(2)
    assert time.monotonic() - started < 2
    assert not operation_thread.is_alive() and not run_thread.is_alive()
    assert session.cancelled.is_set()
    assert len(result) == 1 and result[0].cancel_calibration


def should_cancel_an_obsolete_estimate_before_running_the_latest_request():
    session = _SupersedingSelectorSession()
    selector = CalibrationLevelingSelector(
        session, game_id="demo", bind_host="127.0.0.1", open_browser=False
    )
    selector._start_server()
    posts = [
        {"image_index": 0, "first": [1, 1], "second": [1, 2]},
        {"image_index": 0, "first": [2, 1], "second": [2, 2]},
        {"image_index": 1, "first": [3, 1], "second": [3, 2]},
    ]
    errors = []

    def first_estimate():
        try:
            _selector_post(
                selector,
                "/api/estimate",
                {"posts": posts, "rotation": [17, -9, 2]},
            )
        except urllib.error.HTTPError as error:
            errors.append(error.code)

    first = threading.Thread(target=first_estimate)
    first.start()
    assert session.first_started.wait(2)
    latest = _selector_post(
        selector,
        "/api/estimate",
        {"posts": posts, "rotation": [17, -9, 2]},
    )
    first.join(2)
    selector.close()
    assert not first.is_alive()
    assert errors == [500]
    assert latest["rotation_degrees"] == [17, 4, -3]
    assert session.calls == 2


def should_distinguish_use_skip_cancel_and_backend_close():
    selector = CalibrationLevelingSelector(_SelectorSession(), game_id="demo")
    selector._complete("skip", [99, 1, 2])
    assert selector.result == CalibrationLevelingResult(False, (17, -9, 2))

    selector = CalibrationLevelingSelector(_SelectorSession(), game_id="demo")
    selector._complete("cancel", [17, 1, 2])
    assert selector.result.cancel_calibration

    selector = CalibrationLevelingSelector(_SelectorSession(), game_id="demo")
    selector._latest_preview = (17, 4, -3)
    with pytest.raises(ValueError, match="Preview"):
        selector._complete("use", [17, 4, -2])
    selector._complete("use", [91, 4, -3])
    assert selector.result == CalibrationLevelingResult(True, (17, 4, -3))

    selector = CalibrationLevelingSelector(_SelectorSession(), game_id="demo")
    selector.close()
    assert selector.result.cancel_calibration


def should_authenticate_selector_images_and_reject_dns_rebinding():
    selector = CalibrationLevelingSelector(
        _SelectorSession(), game_id="demo", bind_host="127.0.0.1", open_browser=False
    )
    selector._start_server()
    base = f"http://127.0.0.1:{selector.port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            assert b"Level the rink" in response.read()
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(base + "/image/0", timeout=2)
        assert unauthorized.value.code == 403
        request = urllib.request.Request(
            base + "/image/0", headers={"X-Editor-Token": selector._token}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.read() == b"left"
        rebound = urllib.request.Request(base + "/", headers={"Host": "attacker.example"})
        with pytest.raises(urllib.error.HTTPError) as forbidden:
            urllib.request.urlopen(rebound, timeout=2)
        assert forbidden.value.code == 403
    finally:
        selector.close()


def _calibration_leveling_session(tmp_path):
    images = [tmp_path / "left.png", tmp_path / "right.png"]
    for index, path in enumerate(images):
        Image.new("RGB", (160, 100), (20 + index * 50, 80, 140)).save(path)
    project = (
        'p f19 w4000 h2000 v150 P"110 10 -10" n"PNG"\n'
        'i w160 h100 f0 v100 y0 p0 r0 n"left.png"\n'
        'i w160 h100 f0 v100 y0 p0 r0 n"right.png"\n'
    )
    aligned = tmp_path / ".autooptimiser_out.aligned.pto"
    framed = tmp_path / "autooptimiser_out.pto"
    aligned.write_text(project)
    framed.write_text(project)
    settings = read_stitching_settings(
        {
            "stitching": {
                "mapping_backend": "nona",
                "run_autooptimizer": True,
                "projection": "general-panini",
                "projection_parameters": {"general-panini": [110, 10, -10]},
                "projection_framing": {
                    "auto_fov": True,
                    "auto_canvas": True,
                    "auto_crop": True,
                    "rotation_degrees": [17, -9, 2],
                },
            }
        }
    )
    commands = []

    def run(command, *, input_text=None, timeout_seconds=None, cancel_event=None):
        commands.append((list(command), input_text, timeout_seconds))
        tool = Path(command[0]).name
        if tool == "pano_trafo":
            return "1000 1000\n1000 700\n1800 1000\n1800 700\n2600 1000\n2600 700\n"
        if tool == "pano_modify":
            output = Path(command[command.index("-o") + 1])
            source = Path(command[-1])
            text = source.read_text()
            canvas = next(
                (item.split("=", 1)[1] for item in command if item.startswith("--canvas=")), None
            )
            if canvas and canvas != "AUTO":
                width, height = canvas.split("x")
                text = text.replace("w4000 h2000", f"w{width} h{height}")
            output.write_text(text)
            return ""
        if tool == "nona":
            output = Path(command[command.index("-o") + 1])
            geometry = read_panorama_geometry(command[-1])
            Image.new("RGB", (geometry.width, geometry.height), (40, 100, 150)).save(output)
            return ""
        raise AssertionError(command)

    session = CalibrationLevelingSession(
        aligned,
        framed,
        images,
        settings,
        run,
        lambda executable: f"/tools/{executable}",
    )
    return session, commands


def should_require_posts_from_both_cameras_and_pin_yaw_server_side(tmp_path):
    session, commands = _calibration_leveling_session(tmp_path)
    posts = [
        {"image_index": 0, "first": [10, 10], "second": [10, 80]},
        {"image_index": 0, "first": [50, 10], "second": [50, 80]},
        {"image_index": 0, "first": [90, 10], "second": [90, 80]},
    ]
    try:
        with pytest.raises(ValueError, match="each camera"):
            session.estimate(posts, [17, -9, 2])
        posts[-1]["image_index"] = 1
        with pytest.raises(ValueError, match="Yaw"):
            session.estimate(posts, [18, -9, 2])
        session.estimate(posts, [17, -9, 2])
        assert commands[-1][0][0] == "/tools/pano_trafo"
        assert commands[-1][1].count("\n") == 6
        assert commands[-1][2] == 60
    finally:
        session.close()


def should_frame_exact_selected_settings_before_downscaling_preview(tmp_path):
    session, commands = _calibration_leveling_session(tmp_path)
    try:
        preview = session.preview([17, -12.5, 3.25])
        assert preview.startswith(b"\x89PNG")
        projection, cap, nona = [entry[0] for entry in commands]
        assert projection[0] == "/tools/pano_modify"
        assert "--projection=19" in projection
        assert "--projection-parameter=110 10 -10" in projection
        assert "--rotate=17,-12.5,3.25" in projection
        assert "--fov=AUTO" in projection
        assert "--canvas=AUTO" in projection
        assert "--crop=AUTO" in projection
        assert cap[0] == "/tools/pano_modify"
        assert "--canvas=1600x800" in cap
        assert nona[0] == "/tools/nona"
        assert commands[-1][2] == 60
        assert read_panorama_geometry(session._preview_project).width == 1600
    finally:
        session.close()


def should_cap_preview_width_without_over_downscaling_a_tall_panorama(tmp_path):
    session, commands = _calibration_leveling_session(tmp_path)
    tall_project = session.aligned_project.read_text().replace("w4000 h2000", "w1000 h4000")
    session.aligned_project.write_text(tall_project)
    session.framed_project.write_text(tall_project)
    try:
        preview = session.preview([17, -12.5, 3.25])
        with Image.open(io.BytesIO(preview)) as image:
            assert image.size == (1000, 4000)
        assert [Path(command[0]).name for command, _, _ in commands] == [
            "pano_modify",
            "nona",
        ]
    finally:
        session.close()


def should_reject_an_extremely_tall_preview_before_running_nona(tmp_path):
    session, commands = _calibration_leveling_session(tmp_path)
    tall_project = session.aligned_project.read_text().replace("w4000 h2000", "w1000 h4002")
    session.aligned_project.write_text(tall_project)
    try:
        with pytest.raises(ValueError, match="too tall"):
            session.preview([17, -12.5, 3.25])
        assert [Path(command[0]).name for command, _, _ in commands] == ["pano_modify"]
    finally:
        session.close()


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
@pytest.mark.parametrize("game_name", ["game", "Montréal"])
def should_render_and_transform_using_real_hugin_without_changing_published_project(
    tmp_path, game_name
):
    game_dir = tmp_path / game_name
    game_dir.mkdir()
    config = _game(game_dir, rotation=[0, 0, 0])
    original = (game_dir / "autooptimiser_out.pto").read_bytes()
    session = LevelingSession(game_dir, lambda: config)
    try:
        state = _state(session)
        state["rotation_degrees"] = [4, -8, 3]
        preview = session.preview(state)
        assert preview["width"] <= 1920 and preview["height"] <= 1920
        posts = [{"image_index": 0, "first": [x, 10], "second": [x, 90]} for x in (10, 80, 150)]
        estimate = session.estimate(posts, 7)
        assert estimate["rotation_degrees"] == pytest.approx([7, 0, 0], abs=1e-5)
        assert (game_dir / "autooptimiser_out.pto").read_bytes() == original
    finally:
        session.close()
