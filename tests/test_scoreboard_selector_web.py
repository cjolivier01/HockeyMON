from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib import request

import pytest

if ("TEST_SRCDIR" in os.environ or "RUNFILES_DIR" in os.environ) and "hmlib" not in sys.modules:
    hmlib_root = Path(__file__).resolve().parents[1] / "hmlib"
    hmlib_package = ModuleType("hmlib")
    hmlib_package.__file__ = str(hmlib_root / "__init__.py")
    hmlib_package.__path__ = [str(hmlib_root)]
    sys.modules["hmlib"] = hmlib_package


def _selector_module() -> Any:
    from hmlib.scoreboard import selector as selector_module

    return selector_module


def _make_selector() -> Any:
    selector_module = _selector_module()
    return selector_module.ScoreboardSelector(
        image=selector_module.Image.new("RGB", (64, 48), color=(17, 43, 71)),
        game_id="test-game",
        bind_host="127.0.0.1",
        open_browser=False,
    )


def _read_text(url: str) -> str:
    with request.urlopen(url, timeout=3) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, payload: dict) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=3) as response:
        return response.read().decode("utf-8")


def should_serve_scoreboard_selector_page_and_image():
    selector = _make_selector()
    selector._start_server()
    try:
        html = _read_text(selector.primary_url)
        assert "Pin the four scoreboard corners" in html
        assert "Game: test-game" in html

        with request.urlopen(f"{selector.primary_url}image", timeout=3) as response:
            assert response.headers.get_content_type() == "image/png"
            image_bytes = response.read()

        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        selector.close()


def should_list_hostname_localhost_and_interface_ips(monkeypatch):
    selector_module = _selector_module()
    monkeypatch.setattr(selector_module.socket, "gethostname", lambda: "scorebox")
    monkeypatch.setattr(
        selector_module,
        "_iter_local_ipv4_addresses",
        lambda: ["10.1.2.3", "192.168.4.10"],
    )

    urls = selector_module._build_access_urls("0.0.0.0", 8123)

    assert urls == [
        "http://scorebox:8123/",
        "http://localhost:8123/",
        "http://127.0.0.1:8123/",
        "http://10.1.2.3:8123/",
        "http://192.168.4.10:8123/",
    ]


def should_parse_selector_args_without_consuming_common_hm_args():
    selector_module = _selector_module()

    selector_args, remaining_args = selector_module._parse_selector_cli_args(
        [
            "--selector-bind-host",
            "127.0.0.1",
            "--selector-port",
            "8123",
            "--selector-no-browser",
            "--game-id",
            "test-game",
        ]
    )

    assert selector_args.bind_host == "127.0.0.1"
    assert selector_args.port == 8123
    assert selector_args.open_browser is False
    assert remaining_args == ["--game-id", "test-game"]


def should_save_clockwise_points_from_web_submission():
    selector = _make_selector()
    selector._start_server()
    try:
        response_html = _post_json(
            f"{selector.primary_url}api/complete",
            {
                "action": "save",
                "points": [[55, 41], [6, 8], [54, 7], [5, 40]],
            },
        )

        assert "Thank you for playing!" in response_html
        assert selector.points == [(6, 8), (54, 7), (55, 41), (5, 40)]
    finally:
        selector.close()


def should_mark_missing_scoreboard_from_web_submission():
    selector = _make_selector()
    selector._start_server()
    try:
        selector_module = _selector_module()
        response_html = _post_json(
            f"{selector.primary_url}api/complete",
            {
                "action": "none",
                "points": [],
            },
        )

        assert "Thank you for playing!" in response_html
        assert selector.points == selector_module.ScoreboardSelector.NULL_POINTS
    finally:
        selector.close()


def should_serve_bounded_proxy_and_save_points_in_original_coordinates():
    module = _selector_module()
    selector = module.ScoreboardSelector(
        image=module.Image.new("RGB", (640, 240), color=(17, 43, 71)),
        max_display_height=48,
        game_id="large-game",
        bind_host="127.0.0.1",
        open_browser=False,
    )
    assert selector.image.size == (128, 48)
    selector._start_server()
    try:
        html = _read_text(selector.primary_url)
        assert '"imageWidth": 640' in html
        assert '"imageHeight": 240' in html
        with request.urlopen(f"{selector.primary_url}image", timeout=3) as response:
            with module.Image.open(io.BytesIO(response.read())) as preview:
                assert preview.size == (128, 48)
        _post_json(
            f"{selector.primary_url}api/complete",
            {"action": "save", "points": [[20, 40], [600, 40], [600, 200], [20, 200]]},
        )
        assert selector.points == [(20, 40), (600, 40), (600, 200), (20, 200)]
    finally:
        selector.close()


@pytest.mark.parametrize("size", [(16000, 8000), (64000, 100), (1, 64000)])
def should_bound_proxy_dimensions_and_memory(size):
    module = _selector_module()
    width, height = module._bounded_preview_size(size)
    assert 0 < width <= 8192
    assert 0 < height <= 8192
    assert width * height * 4 <= 96 * 1024 * 1024
    assert width <= size[0] and height <= size[1]


@pytest.mark.parametrize("size", [(0, 10), (65536, 10), (65000, 65000)])
def should_reject_unsupported_source_geometry(size):
    with pytest.raises(ValueError, match="Scoreboard source image"):
        _selector_module()._bounded_preview_size(size)


@pytest.mark.parametrize("limit", [0, -1, 3.5, True])
def should_reject_invalid_display_height(limit):
    module = _selector_module()
    with pytest.raises(ValueError, match="max_display_height"):
        module._prepare_selector_image(module.Image.new("RGB", (64, 48)), limit)


def should_resize_tensor_before_host_conversion_and_keep_caller_image(monkeypatch):
    module = _selector_module()
    source = module.torch.zeros((1, 3, 240, 640), dtype=module.torch.uint8)
    original_conversion = module.make_visible_image
    converted_sizes = []

    def visible(image, **kwargs):
        converted_sizes.append((module.image_width(image), module.image_height(image)))
        return original_conversion(image, **kwargs)

    monkeypatch.setattr(module, "make_visible_image", visible)
    proxy, source_size = module._prepare_selector_image(source, 48)
    assert converted_sizes == [(128, 48)]
    assert proxy.size == (128, 48)
    assert source_size == (640, 240)
    assert tuple(source.shape) == (1, 3, 240, 640)


def should_validate_source_size_before_decoding(monkeypatch):
    module = _selector_module()
    source = module.Image.new("RGB", (64, 48))
    monkeypatch.setattr(module, "_MAXIMUM_SOURCE_PIXELS", 100)

    def fail_decode(*args, **kwargs):
        raise AssertionError("unsupported source was decoded")

    monkeypatch.setattr(source, "load", fail_decode)
    with pytest.raises(ValueError, match="Scoreboard source image"):
        module._prepare_selector_image(source, None)
