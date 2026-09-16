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


def should_configure_scoreboard_from_stitched_reference_frame(monkeypatch):
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

    assert calls == [{"game_id": "test-game", "image": None}]
    assert configured["scoreboard_cfg"]["scoreboard_points"] == [
        [10, 20],
        [30, 20],
        [30, 40],
        [10, 40],
    ]


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
        lambda game_id: _scoreboard_game_config(),
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
