"""Join role labels to boxes without inventing labels for incomplete frames."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "team_id_annotations", Path(__file__).resolve().parents[1] / "scripts/team_id_annotations.py"
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def make_game(root, name):
    directory = root / name
    (directory / "all_frames").mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(directory / "all_frames" / f"{name}_10.png")
    (directory / "gt.txt").write_text(
        f"{name}_10_1.png,1\n{name}_10_2.png,2\n{name}_10_3.png,3\n"
        f"{name}_20_1.png,1\n{name}_30_1.png,1\n{name}_30_2.png,2\n"
    )
    (directory / "bbox_data.txt").write_text(
        f"-1,2,10,15,{name}_10_1.png;12,4,20,18,{name}_10_2.png;22,5,50,35,{name}_10_3.png;\n"
        f"1,2,10,15,{name}_20_1.png;12,4,20,18,{name}_20_2.png;\n"
        f"1,2,10,15,{name}_30_1.png;\n"
        f"1,2,10,15,{name}_40_1.png;\n"
    )
    return directory


def should_merge_teams_keep_referees_and_exclude_incomplete_frames(tmp_path):
    make_game(tmp_path, "game1")
    make_game(tmp_path, "game7")
    video = tmp_path / "video1"
    video.mkdir()
    (video / "gt.txt").write_text("video1_10_r_1.png,2\n")
    (video / "players_bbox_data.txt").write_text("")
    splits, report = reader.load_team_id(tmp_path, ["game7"])
    for split, name in [("train", "game1"), ("val", "game7")]:
        data = splits[split]
        assert len(data["images"]) == 1
        assert data["images"][0]["file_name"] == f"{name}/all_frames/{name}_10.png"
        assert data["images"][0]["frame_index"] == 9
        assert [a["category_id"] for a in data["annotations"]] == [1, 2, 1]
        assert [a["bbox"] for a in data["annotations"]] == [
            [0, 2, 10, 13],
            [12, 4, 8, 14],
            [22, 5, 18, 25],
        ]
        assert data["categories"] == [{"id": 1, "name": "person"}, {"id": 2, "name": "ref"}]
        assert set(report["sequences"][name]["excluded_frames"]) == {"20", "30", "40"}
    assert "omits referees" in report["sequences"]["video1"]["excluded_sequence_reason"]


def should_reject_unrecognized_roles(tmp_path):
    game = make_game(tmp_path, "game1")
    make_game(tmp_path, "game7")
    path = game / "gt.txt"
    path.write_text(path.read_text().replace(",3", ",4"))
    with pytest.raises(ValueError, match="Invalid or duplicate role"):
        reader.load_team_id(tmp_path, ["game7"])


def should_reject_unknown_validation_games(tmp_path):
    make_game(tmp_path, "game1")
    with pytest.raises(ValueError, match="validation games"):
        reader.load_team_id(tmp_path, ["typo"])
