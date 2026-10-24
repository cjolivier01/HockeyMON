from pathlib import Path
from typing import Any

import pytest

from hmlib import orientation


def should_configure_cameras_with_different_chapter_counts(monkeypatch, tmp_path):
    game_dir = tmp_path / "game"
    left_files = [game_dir / "cam1" / f"left-{index}.mp4" for index in range(1, 6)]
    right_files = [game_dir / "cam2" / f"right-{index}.mp4" for index in range(1, 5)]
    analyzed_videos = {
        "left": {index: str(path) for index, path in enumerate(left_files, start=1)},
        "right": {index: str(path) for index, path in enumerate(right_files, start=1)},
    }
    saved: dict[str, Any] = {}

    monkeypatch.setattr(orientation, "get_game_dir", lambda game_id: str(game_dir))
    monkeypatch.setattr(
        orientation, "get_available_videos", lambda dir_name: {"cam1": {}, "cam2": {}}
    )
    monkeypatch.setattr(orientation, "get_game_config_private", lambda game_id: {})
    monkeypatch.setattr(
        orientation,
        "get_game_videos_analysis",
        lambda game_id, device, inference_scale: analyzed_videos,
    )
    monkeypatch.setattr(
        orientation,
        "save_private_config",
        lambda game_id, data: saved.update(game_id=game_id, data=data),
    )

    result = orientation.configure_game_videos(game_id="game")

    assert result == {
        "left": [str(path) for path in left_files],
        "right": [str(path) for path in right_files],
    }
    assert saved == {
        "game_id": "game",
        "data": {
            "game": {
                "videos": {
                    "left": [str(Path("cam1") / path.name) for path in left_files],
                    "right": [str(Path("cam2") / path.name) for path in right_files],
                }
            }
        },
    }


def should_use_selected_sources_before_discovered_aliases(monkeypatch, tmp_path):
    monkeypatch.setattr(orientation, "get_game_dir", lambda game_id: str(tmp_path))
    monkeypatch.setattr(
        orientation,
        "get_game_config_private",
        lambda game_id: {
            "game": {"videos": {"left": ["selected-left.mp4"], "right": ["selected-right.mp4"]}}
        },
    )

    def discovery_must_not_run(**kwargs):
        raise AssertionError("Removed recordings must not be rediscovered")

    monkeypatch.setattr(orientation, "get_available_videos", discovery_must_not_run)
    assert orientation.configure_game_videos("game") == {
        "left": [str(tmp_path / "selected-left.mp4")],
        "right": [str(tmp_path / "selected-right.mp4")],
    }


def should_require_reimport_after_removing_an_explicit_source(monkeypatch, tmp_path):
    monkeypatch.setattr(orientation, "get_game_dir", lambda game_id: str(tmp_path))
    monkeypatch.setattr(
        orientation,
        "get_game_config_private",
        lambda game_id: {"game": {"videos": {"left": [], "right": ["right.mp4"]}}},
    )
    (tmp_path / "left.mp4").write_bytes(b"preserved original")
    with pytest.raises(ValueError, match="Import both left and right"):
        orientation.configure_game_videos("game")
