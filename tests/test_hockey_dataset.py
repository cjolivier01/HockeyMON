"""Guard holdout isolation and body-only crop construction for hockey training."""

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.hockey_data import FLIP, JOINTS, load_dataset, prepare


def dataset(root: Path):
    count = 0
    for split, sequences in {"train": (1, 2), "valid": (3,), "test": (4,)}.items():
        directory = root / f"images_{split}"
        directory.mkdir()
        names, points = [], []
        for sequence in sequences:
            for frame in range(3):
                name = f"ImageSequencesBackward_{sequence}_{frame}.jpg"
                count += 1
                Image.new("RGB", (80, 100), (count, count * 3, count * 7)).save(
                    directory / name, format="PNG"
                )
                names.append([ord(c) for c in name] + [0] * (33 - len(name)))
                p = np.array([[10 + i, 15 + i * 2] for i in range(18)], dtype=np.float32)
                p[17] = (75, 90)
                points.append(p)
        with h5py.File(root / f"annot_{split}.h5", "w") as f:
            f["imgname"] = np.array(names, dtype=np.float64)
            f["part"] = np.array(points)


def should_remove_entire_duplicate_components_before_training(tmp_path):
    dataset(tmp_path)
    train = tmp_path / "images_train"
    validation = tmp_path / "images_valid/ImageSequencesBackward_3_0.jpg"
    (train / "ImageSequencesBackward_1_0.jpg").write_bytes(validation.read_bytes())
    (train / "ImageSequencesBackward_2_0.jpg").write_bytes(
        (train / "ImageSequencesBackward_1_1.jpg").read_bytes()
    )
    splits, excluded = load_dataset(tmp_path)
    assert not splits["train"]
    assert len(splits["valid"]) == len(splits["test"]) == 1
    assert set(excluded["overlap"]["train"]) == {"Backward_1", "Backward_2"}


def should_keep_stick_coordinates_out_of_inference_crop_definition(tmp_path):
    dataset(tmp_path)
    output = tmp_path / "converted"
    prepare(tmp_path, output)
    annotation = json.loads((output / "train.json").read_text())["annotations"][0]
    assert annotation["bbox"] == [10, 15, 15, 30]
    assert annotation["keypoints"][-3:] == [75, 90, 2]
    assert JOINTS[0] == "right_ankle" and JOINTS[15] == "left_wrist"
    assert FLIP[16:] == (16, 17)


def should_remove_irregular_temporal_triplet_and_record_it(tmp_path):
    dataset(tmp_path)
    with h5py.File(tmp_path / "annot_train.h5", "r+") as f:
        name = "ImageSequencesBackward_1_9.jpg"
        f["imgname"][2] = [ord(c) for c in name] + [0] * (33 - len(name))
    source = tmp_path / "images_train/ImageSequencesBackward_1_2.jpg"
    source.rename(source.with_name(name))
    splits, excluded = load_dataset(tmp_path)
    assert "Backward_1" not in splits["train"]
    assert excluded["irregular_frames"]["train"] == ["Backward_1"]
