"""Pixel and annotation contracts for the masked dataset exporter."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "gray_hockey_background",
    Path(__file__).resolve().parents[1] / "scripts/gray_hockey_background.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def should_preserve_ice_and_clipped_boxes_and_gray_everything_else():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    mask = np.zeros((6, 8), dtype=bool)
    mask[4:, :] = True
    output = exporter.paint_background(image, mask, [{"bbox": [-1.2, 0.5, 3.4, 1.7]}])
    expected = np.full_like(image, 128)
    expected[4:, :] = image[4:, :]
    expected[:3, :3] = image[:3, :3]
    np.testing.assert_array_equal(output, expected)
    assert not mask[:4].any()
    assert image[0, 0, 0] == 0


def should_remove_spectators_without_changing_role_ids_or_attributes():
    data = {
        "categories": [{"id": 3, "name": "official"}, {"id": 4, "name": "spectator"}],
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "track_id": 23},
            {"id": 2, "image_id": 1, "category_id": 4},
            {"id": 3, "image_id": 2, "category_id": 3},
        ],
    }
    result = exporter.select_annotations(data, limit=1)
    assert result["categories"] == [{"id": 3, "name": "official"}]
    assert result["annotations"] == [data["annotations"][0]]
    assert len(data["annotations"]) == 3


def should_reject_mask_with_wrong_dimensions():
    with pytest.raises(ValueError, match="Mask shape"):
        exporter.paint_background(np.zeros((3, 4, 3)), np.zeros((4, 3)), [])
