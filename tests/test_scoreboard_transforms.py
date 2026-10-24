from __future__ import annotations

from typing import Any

import pytest
import torch

from hmlib.transforms.scoreboard_transforms import HmCaptureScoreboard, HmConfigureScoreboard


def _scoreboard_game_config() -> dict[str, Any]:
    return {
        "rink": {
            "scoreboard": {
                "perspective_polygon": [[10, 10], [20, 10], [20, 20], [10, 20]],
                "projected_width": "%10",
                "projected_height": "%20",
            }
        }
    }


def should_reuse_saved_polygon_without_opening_interactive_selector(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _fake_configure_scoreboard(game_id: str, image=None, **kwargs):
        calls.append({"game_id": game_id, "image": image})
        return [[10, 20], [30, 20], [30, 40], [10, 40]]

    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.configure_scoreboard",
        _fake_configure_scoreboard,
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_config",
        lambda game_id: _scoreboard_game_config(),
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_clip_box",
        lambda game_id: None,
    )

    transform = HmConfigureScoreboard(game_id="test-game")
    results = {"img": torch.zeros((1, 4, 4, 3), dtype=torch.uint8)}
    configured = transform(results)

    assert calls == []
    assert (
        configured["scoreboard_cfg"]["scoreboard_points"]
        == _scoreboard_game_config()["rink"]["scoreboard"]["perspective_polygon"]
    )


def should_fallback_to_current_frame_when_stitched_reference_is_missing(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _fake_configure_scoreboard(game_id: str, image=None, **kwargs):
        calls.append({"game_id": game_id, "image": image})
        if image is None:
            raise FileNotFoundError("missing s.png")
        return [[1, 2], [3, 2], [3, 4], [1, 4]]

    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.configure_scoreboard",
        _fake_configure_scoreboard,
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_config",
        lambda game_id: {"rink": {"scoreboard": {"perspective_polygon": None}}},
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_clip_box",
        lambda game_id: None,
    )

    image = torch.zeros((1, 4, 4, 3), dtype=torch.uint8)
    transform = HmConfigureScoreboard(game_id="test-game")
    configured = transform({"img": image})

    assert len(calls) == 2
    assert calls[0] == {"game_id": "test-game", "image": None}
    assert calls[1]["game_id"] == "test-game"
    assert calls[1]["image"] is image
    assert configured["scoreboard_cfg"]["scoreboard_points"] == [
        [1, 2],
        [3, 2],
        [3, 4],
        [1, 4],
    ]


def should_apply_live_polygon_and_disabled_transitions(monkeypatch):
    monkeypatch.setattr("hmlib.transforms.scoreboard_transforms.get_clip_box", lambda game_id: None)
    transform = HmConfigureScoreboard(game_id="test-game", allow_interactive_setup=False)
    config = _scoreboard_game_config()
    results = {"game_config": config}
    first = transform(results)["scoreboard_cfg"]["scoreboard_points"]
    config["rink"]["scoreboard"]["perspective_polygon"] = [[30, 10], [40, 10], [40, 20], [30, 20]]
    assert transform(results)["scoreboard_cfg"]["scoreboard_points"] != first
    config["rink"]["scoreboard"]["enabled"] = False
    assert "scoreboard_cfg" not in transform(results)
    config["rink"]["scoreboard"]["enabled"] = True
    assert transform(results)["scoreboard_cfg"]["scoreboard_points"][0] == [30, 10]
    config["rink"]["scoreboard"]["perspective_polygon"] = [[0, 0]] * 4
    assert "scoreboard_cfg" not in transform(results)


@pytest.mark.parametrize("saved_size", [(60, 40), (120, 80)])
@pytest.mark.parametrize("explicit_polygon", [False, True])
def should_preserve_saved_native_canvas_and_polygon_for_a_single_source_override(
    tmp_path, monkeypatch, saved_size, explicit_polygon
):
    import copy
    import json

    import cv2
    import numpy as np
    import yaml

    import hmlib.config as configuration

    session = tmp_path / "session"
    session.mkdir()
    source = tmp_path / "override.mp4"
    source.touch()
    monkeypatch.setenv("HM_SHARED_UI_BACKEND", "1")
    monkeypatch.setenv("HM_UI_SESSION_DIR", str(session))
    monkeypatch.setenv("HM_SHARED_SOURCE_CHAPTERS", json.dumps([str(source)]))
    monkeypatch.delenv("HM_SHARED_NATIVE_CALIBRATION", raising=False)
    monkeypatch.setattr(configuration, "get_game_dir", lambda game_id: str(tmp_path))
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_clip_box", lambda **kw: [50, 50, 60, 60]
    )
    config = _scoreboard_game_config()
    config["game"] = {"videos": {"left": "left.mp4", "right": "right.mp4"}}
    config["rink"]["scoreboard"]["enabled"] = False
    monkeypatch.delenv("HM_SHARED_SOURCE_SCOREBOARD_POLYGON", raising=False)
    if explicit_polygon:
        monkeypatch.setenv(
            "HM_SHARED_SOURCE_SCOREBOARD_POLYGON",
            json.dumps(config["rink"]["scoreboard"]["perspective_polygon"]),
        )
    original_config = yaml.safe_dump(config).encode()
    (tmp_path / "config.yaml").write_bytes(original_config)
    cv2.imwrite(
        str(tmp_path / "s.png"), np.full((saved_size[1], saved_size[0], 3), 23, dtype=np.uint8)
    )
    original_image = (tmp_path / "s.png").read_bytes()
    runtime = copy.deepcopy(config)
    transform = HmConfigureScoreboard(game_id="test-game", allow_interactive_setup=False)
    transform({"game_config": runtime, "img": torch.full((1, 40, 60, 3), 71, dtype=torch.uint8)})
    private = session / "scoreboard-source"
    image = cv2.imread(str(private / "s.png"))
    assert image.shape == (40, 60, 3) and (image == 71).all()
    assert runtime["rink"]["scoreboard"]["perspective_polygon"] == (
        config["rink"]["scoreboard"]["perspective_polygon"] if explicit_polygon else None
    )
    assert (tmp_path / "s.png").read_bytes() == original_image
    assert (tmp_path / "config.yaml").read_bytes() == original_config
    transform({"game_config": runtime, "img": torch.zeros((1, 40, 60, 3), dtype=torch.uint8)})
    assert (cv2.imread(str(private / "s.png")) == 71).all()
    # A run-local native save survives seek reconstruction and its source
    # coordinates never inherit the saved pair's crop offset.
    private_config = yaml.safe_load((private / "config.yaml").read_text())
    corners = [[2, 2], [30, 2], [30, 10], [2, 10]]
    private_config["rink"]["scoreboard"]["perspective_polygon"] = corners
    (private / "config.yaml").write_text(yaml.safe_dump(private_config))
    runtime = copy.deepcopy(config)
    runtime["rink"]["scoreboard"]["enabled"] = True
    resumed = HmConfigureScoreboard(game_id="test-game", allow_interactive_setup=False)
    result = resumed(
        {"game_config": runtime, "img": torch.zeros((1, 40, 60, 3), dtype=torch.uint8)}
    )
    assert result["scoreboard_cfg"]["scoreboard_points"] == corners
    assert (tmp_path / "s.png").read_bytes() == original_image
    assert (tmp_path / "config.yaml").read_bytes() == original_config


def should_skip_interactive_scoreboard_setup_when_polygon_is_missing(monkeypatch, caplog):
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.configure_scoreboard",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scoreboard selector should not be launched")
        ),
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_config",
        lambda game_id: {
            "rink": {
                "scoreboard": {
                    "projected_width": "%10",
                    "projected_height": "%20",
                }
            }
        },
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms._is_rocm_runtime",
        lambda: True,
    )

    transform = HmConfigureScoreboard(game_id="test-game")
    results = {"img": torch.zeros((1, 4, 4, 3), dtype=torch.uint8)}

    with caplog.at_level("WARNING"):
        configured = transform(results)

    assert configured is results
    assert "scoreboard_cfg" not in configured
    assert "skipping scoreboard capture for this run" in caplog.text


def should_allow_interactive_scoreboard_setup_by_default_off_rocm(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _fake_configure_scoreboard(game_id: str, image=None, **kwargs):
        calls.append({"game_id": game_id, "image": image})
        return [[1, 2], [3, 2], [3, 4], [1, 4]]

    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.configure_scoreboard",
        _fake_configure_scoreboard,
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_config",
        lambda game_id: {
            "rink": {
                "scoreboard": {
                    "projected_width": "%10",
                    "projected_height": "%20",
                }
            }
        },
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_clip_box",
        lambda game_id: None,
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms._is_rocm_runtime",
        lambda: False,
    )

    transform = HmConfigureScoreboard(game_id="test-game")
    configured = transform({"img": torch.zeros((1, 4, 4, 3), dtype=torch.uint8)})

    assert calls == [{"game_id": "test-game", "image": None}]
    assert configured["scoreboard_cfg"]["scoreboard_points"] == [
        [1, 2],
        [3, 2],
        [3, 4],
        [1, 4],
    ]


def should_skip_scoreboard_capture_when_polygon_is_outside_frame(caplog):
    capture = HmCaptureScoreboard()
    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    results = {
        "img": image,
        "scoreboard_cfg": {
            "scoreboard_points": [
                [1000, 10],
                [1100, 10],
                [1100, 40],
                [1000, 40],
            ],
            "dest_width": 64,
            "dest_height": 32,
        },
    }

    with caplog.at_level("WARNING"):
        captured = capture(results)

    assert captured is results
    assert "scoreboard_img" not in captured
    assert "scoreboard_cfg" not in captured
    assert "is outside current frame size" in caplog.text

    caplog.clear()
    next_results = {
        "img": image,
        "scoreboard_cfg": {
            "scoreboard_points": [
                [1000, 10],
                [1100, 10],
                [1100, 40],
                [1000, 40],
            ],
            "dest_width": 64,
            "dest_height": 32,
        },
    }

    with caplog.at_level("WARNING"):
        captured = capture(next_results)

    assert "scoreboard_img" not in captured
    assert "scoreboard_cfg" not in captured
    assert caplog.text == ""


def should_skip_scoreboard_capture_when_polygon_is_partially_outside_frame(caplog):
    capture = HmCaptureScoreboard()
    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    results = {
        "img": image,
        "scoreboard_cfg": {
            "scoreboard_points": [
                [-10, 10],
                [20, 10],
                [20, 40],
                [-10, 40],
            ],
            "dest_width": 64,
            "dest_height": 32,
        },
    }

    with caplog.at_level("WARNING"):
        captured = capture(results)

    assert captured is results
    assert "scoreboard_img" not in captured
    assert "scoreboard_cfg" not in captured
    assert "is outside current frame size" in caplog.text
