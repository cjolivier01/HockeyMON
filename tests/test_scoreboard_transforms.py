from __future__ import annotations

from typing import Any

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


def should_publish_single_source_selector_canvas_even_when_disabled(tmp_path, monkeypatch):
    import cv2

    import hmlib.config as configuration

    monkeypatch.setenv("HM_SHARED_UI_BACKEND", "1")
    monkeypatch.delenv("HM_SHARED_NATIVE_CALIBRATION", raising=False)
    monkeypatch.setattr(configuration, "get_game_dir", lambda game_id: str(tmp_path))
    config = {"rink": {"scoreboard": {"enabled": False}}}
    transform = HmConfigureScoreboard(game_id="test-game", allow_interactive_setup=False)
    transform({"game_config": config, "img": torch.full((1, 40, 60, 3), 71, dtype=torch.uint8)})
    image = cv2.imread(str(tmp_path / "s.png"))
    assert image.shape == (40, 60, 3) and (image == 71).all()
    transform({"game_config": config, "img": torch.zeros((1, 40, 60, 3), dtype=torch.uint8)})
    assert (cv2.imread(str(tmp_path / "s.png")) == 71).all()


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
