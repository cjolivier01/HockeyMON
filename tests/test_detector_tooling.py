"""CPU regressions for detector dataset preparation and partial-label evaluation."""

import collections
import json
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from scripts import compare_detectors as compare
from scripts import crowdhuman_to_hockey_classes as remap
from scripts import hockey_tracking_to_coco as convert
from scripts import slice_detector_head as reshape


def should_keep_duplicate_player_predictions_out_of_spectator_labels():
    annotated = [{"xyxy": (0, 0, 100, 100)}]
    detections = [
        {"xyxy": (0, 0, 100, 100)},
        {"xyxy": (10, 0, 110, 100)},
        {"xyxy": (200, 0, 300, 100)},
    ]
    assert convert.unmatched_detections(detections, annotated, 0.5) == [2]
    assert convert.unmatched_detections(detections, [], 0.5) == [0, 1, 2]
    assert convert.unmatched_detections([], annotated, 0.5) == []


def should_preserve_cvat_roles_attributes_and_interpolated_boxes(tmp_path):
    xml = tmp_path / "clip.xml"
    xml.write_text(
        "<annotations><meta><task><size>2</size><original_size>"
        "<width>1280</width><height>720</height></original_size></task></meta>"
        '<track id="42" label="keeper">'
        '<box frame="0" outside="0" occluded="1" keyframe="0" '
        'xtl="10" ytl="20" xbr="30" ybr="60">'
        '<attribute name="team">home</attribute><attribute name="id">31</attribute></box>'
        '<box frame="1" outside="1" /></track>'
        '<track id="43" label="ref"><box frame="0" outside="0" keyframe="1" '
        'xtl="40" ytl="20" xbr="60" ybr="60" /></track></annotations>'
    )
    frames, size, count = convert.read_cvat(xml)
    assert size == (1280, 720) and count == 2
    assert set(frames) == {0}
    assert [box["category"] for box in frames[0]] == ["goalie", "official"]
    keeper = frames[0][0]
    assert keeper["team"] == "home" and keeper["jersey"] == "31"
    assert keeper["interpolated"] and keeper["occluded"] and keeper["track_id"] == 42


def should_preserve_category_order_and_ignore_flags_when_remapping(tmp_path, monkeypatch):
    source, hockey, output = (
        tmp_path / name for name in ("source.json", "hockey.json", "out.json")
    )
    categories = [{"id": i, "name": name} for name, i in convert.CATEGORY_IDS.items()]
    original = {
        "images": [{"id": 1, "file_name": "crowd.jpg"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 1, "bbox": [2, 3, 4, 5]},
        ],
    }
    source.write_text(json.dumps(original))
    hockey.write_text(json.dumps({"categories": categories}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["remap", "--src", str(source), "--categories-from", str(hockey), "--dst", str(output)],
    )
    assert remap.main() == 0
    actual = json.loads(output.read_text())
    assert actual["categories"] == categories
    assert actual["images"] == original["images"]
    assert [a["category_id"] for a in actual["annotations"]] == [4, 4]
    assert [a["iscrowd"] for a in actual["annotations"]] == [0, 1]
    assert [a["bbox"] for a in actual["annotations"]] == [
        a["bbox"] for a in original["annotations"]
    ]


@pytest.mark.parametrize("source_rows, target_rows", [(80, 2), (1, 4)])
def should_seed_every_role_from_person_and_preserve_regression(source_rows, target_rows):
    key = "bbox_head.head_module.cls_preds.0.2.weight"
    weights = torch.arange(source_rows * 8).reshape(source_rows, 8, 1, 1)
    regression = torch.arange(32).reshape(4, 8, 1, 1)
    state, count = reshape.slice_state_dict(
        {key: weights, "reg.weight": regression}, source_rows, target_rows
    )
    assert count == 1
    assert state[key].shape == (target_rows, 8, 1, 1)
    assert torch.equal(state[key], weights[:1].expand(target_rows, -1, -1, -1))
    state[key][0, 0] = -1
    assert weights[0, 0].item() == 0  # Independent of the input checkpoint.
    assert torch.equal(state["reg.weight"], regression)


def should_select_mmengine_deployment_ema_and_raw_resume_weights():
    # This is the on-disk layout *after* EMAHook.before_save_checkpoint swaps
    # the live model and its EMA shadow. The bookkeeping is not a model weight.
    saved = {
        "state_dict": {"weight": torch.tensor([2.0])},
        "ema_state_dict": {"module.weight": torch.tensor([3.0]), "steps": torch.tensor(10)},
    }
    ema, _ = compare.checkpoint_weights(saved, use_ema=True)
    raw, _ = compare.checkpoint_weights(saved, use_ema=False)
    assert ema["weight"].item() == 2
    assert raw["weight"].item() == 3 and set(raw) == {"weight"}
    with pytest.raises(SystemExit, match="requires an EMAHook checkpoint"):
        compare.checkpoint_weights({"state_dict": {}}, use_ema=False)


def should_measure_recall_without_penalizing_unannotated_crowd_or_duplicates():
    images = [{"id": 1}]
    gt = {1: [{"bbox": [0, 0, 10, 10]}, {"bbox": [20, 0, 10, 10]}]}
    exact = [{"image_id": 1, "bbox": a["bbox"], "score": 0.8} for a in gt[1]]
    crowd = [{"image_id": 1, "bbox": [50 + i, 50, 1, 1], "score": 0.99} for i in range(150)]
    assert compare.known_box_recall(images, gt, exact + crowd + exact[:1], 0.3) == (1, 1, 1)
    assert compare.known_box_recall(images, gt, exact[:1] * 2, 0.3) == (0.5, 0.5, 0.5)
    assert compare.known_box_recall(images, gt, [], 0.3) == (0, 0, 0)
    assert compare.known_box_recall(images, gt, exact, 0.9) == (0, 0, 0)
    assert compare.known_box_recall(images, {}, exact, 0.3) is None


def should_exclude_pseudo_labels_from_comparison_ground_truth(tmp_path):
    path = tmp_path / "val.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "sequence": "clip"}],
                "categories": [{"id": 1, "name": "player"}, {"id": 4, "name": "spectator"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1},
                    {"id": 2, "image_id": 1, "category_id": 1, "pseudo_label": True},
                    {"id": 3, "image_id": 1, "category_id": 4},
                ],
            }
        )
    )
    images, gt, _ = compare.sample_images(path, 10, 0)
    assert len(images) == 1 and [a["id"] for a in gt[1]] == [1]


def should_apply_same_person_postprocessing_to_both_models(tmp_path, monkeypatch):
    config = {"multi_label": True, "max_per_img": 100, "nms": {"class_agnostic": False}}
    model = SimpleNamespace(bbox_head=SimpleNamespace(test_cfg=config))
    args = SimpleNamespace(score_thr=0.01, report_thr=0.3, batch_size=1, images=tmp_path)

    def infer(model, transforms, paths, score_thr, with_labels):
        assert model.bbox_head.test_cfg["multi_label"] is False
        assert model.bbox_head.test_cfg["nms"]["class_agnostic"] is True
        assert model.bbox_head.test_cfg["score_thr"] == score_thr == 0.01
        assert model.bbox_head.test_cfg["max_per_img"] > 100
        return [[{"xyxy": [0, 0, 10, 10], "score": 0.9, "label": 0}]]

    monkeypatch.setattr(compare, "detect", infer)
    stats, count, classes = compare.evaluate_model(
        model,
        ["player"],
        [{"id": 1, "file_name": "frame.jpg"}],
        {1: [{"bbox": [0, 0, 10, 10]}]},
        [],
        args,
    )
    assert stats == (1, 1, 1) and count == 1 and classes == {"player": 1}


def should_release_baseline_before_loading_finetune(monkeypatch):
    class Model:
        pass

    baseline = None

    def build_baseline(*_):
        nonlocal baseline
        model = Model()
        baseline = weakref.ref(model)
        return model

    def build_finetuned(*_):
        assert baseline() is None, "baseline weights still live when loading the next model"
        return Model(), ["player"]

    monkeypatch.setattr(sys, "argv", ["compare", "--new", "unused.pth", "--device", "cpu"])
    monkeypatch.setattr(compare, "sample_images", lambda *_: ([{"id": 1}], {}, {}))
    monkeypatch.setattr(compare, "build_transforms", lambda *_: ([], (100, 100)))
    monkeypatch.setattr(compare, "build_baseline", build_baseline)
    monkeypatch.setattr(compare, "build_finetuned", build_finetuned)
    monkeypatch.setattr(compare, "evaluate_model", lambda *_: ((0, 0, 0), 0, collections.Counter()))
    assert compare.main() == 0
