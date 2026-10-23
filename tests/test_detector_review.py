"""CPU regressions for preview coordinates and persistent review preservation."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import fiftyone_review, review_coco_dataset


@pytest.fixture
def annotation_file(tmp_path):
    assert cv2.imwrite(str(tmp_path / "frame.png"), np.zeros((400, 400, 3), dtype=np.uint8))
    data = {
        "images": [
            {
                "id": 1,
                "file_name": "frame.png",
                "width": 400,
                "height": 400,
                "sequence": "game",
                "frame_index": 1,
            }
        ],
        "categories": [{"id": 1, "name": "player"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100]}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("stills", [False, True])
def should_align_scaled_boxes_in_still_and_video_previews(
    tmp_path, annotation_file, monkeypatch, stills
):
    output = tmp_path / "review"
    argv = [
        "review_coco_dataset.py",
        "--ann",
        str(annotation_file),
        "--images",
        str(tmp_path),
        "--out",
        str(output),
        "--scale",
        "0.5",
    ]
    captured = []

    def encode(frames, out_path, size, fps, crf):
        captured.extend(frames)
        assert size == (200, 200)
        out_path.write_bytes(b"test video")
        return len(captured)

    monkeypatch.setattr(review_coco_dataset, "encode", encode)
    if stills:
        argv.extend(["--stills", "1"])
    monkeypatch.setattr(sys, "argv", argv)
    assert review_coco_dataset.main() == 0
    frame = cv2.imread(str(output / "game" / "000001.jpg")) if stills else captured[0]
    assert frame.shape == (200, 200, 3)
    # Original left edge x=100, y=150 becomes x=50, y=75. The old renderer
    # left it at x=100, y=150, entirely outside the scaled person's box.
    assert int(frame[75, 50].max()) > 40
    assert int(frame[150, 100].max()) < 10


@pytest.fixture
def fake_fiftyone(monkeypatch):
    saved = {"review": {"tags": ["human-reviewed"], "corrected_label": "goalie"}}
    deleted = []

    class Dataset:
        def __init__(self, name, persistent):
            assert persistent
            assert name not in saved
            self.samples = []
            self.indices = []
            saved[name] = self

        def add_samples(self, samples):
            self.samples.extend(samples)

        def create_index(self, field):
            self.indices.append(field)

    def delete_dataset(name):
        deleted.append(name)
        del saved[name]

    module = SimpleNamespace(
        dataset_exists=lambda name: name in saved,
        delete_dataset=delete_dataset,
        Dataset=Dataset,
        Detection=dict,
        Detections=dict,
        Sample=dict,
    )
    monkeypatch.setitem(sys.modules, "fiftyone", module)
    return saved, deleted


def should_refuse_to_erase_an_existing_review(annotation_file, fake_fiftyone):
    saved, deleted = fake_fiftyone
    existing = saved["review"]
    with pytest.raises(SystemExit, match="--name.*--replace"):
        fiftyone_review.build(annotation_file, annotation_file.parent, 1, None, "review")
    assert saved["review"] is existing
    assert not deleted


@pytest.mark.parametrize("invalid", ["unknown_sequence", "missing_image", "dimensions", "empty"])
def should_validate_replacement_before_deleting_review(annotation_file, fake_fiftyone, invalid):
    saved, deleted = fake_fiftyone
    existing = saved["review"]
    data = json.loads(annotation_file.read_text())
    sequences = None
    if invalid == "unknown_sequence":
        sequences = ["game", "unknown"]
    elif invalid == "missing_image":
        data["images"][0]["file_name"] = "missing.png"
    elif invalid == "dimensions":
        data["images"][0]["width"] = 0
    else:
        data["images"] = []
    annotation_file.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        fiftyone_review.build(
            annotation_file, annotation_file.parent, 1, sequences, "review", replace=True
        )
    assert saved["review"] is existing
    assert not deleted


@pytest.mark.parametrize("replace", [False, True])
def should_create_a_new_review_or_explicitly_replace_one(annotation_file, fake_fiftyone, replace):
    saved, deleted = fake_fiftyone
    existing = saved["review"]
    name = "review" if replace else "new-review"
    dataset, counts = fiftyone_review.build(
        annotation_file, annotation_file.parent, 1, None, name, replace=replace
    )
    assert saved[name] is dataset
    assert deleted == (["review"] if replace else [])
    if not replace:
        assert saved["review"] is existing
    assert counts == {"dataset player": 1}
    assert len(dataset.samples) == 1
    assert dataset.samples[0]["ground_truth"]["detections"][0]["bounding_box"] == [
        0.25,
        0.25,
        0.25,
        0.25,
    ]
