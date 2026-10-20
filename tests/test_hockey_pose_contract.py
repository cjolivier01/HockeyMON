"""Prevent valid tensor exports from silently changing native crop semantics."""

from copy import deepcopy

import pytest

from scripts.hockey_pose_contract import portable_pose_config


@pytest.fixture
def config():
    return {
        "model": {
            "type": "TopdownPoseEstimator",
            "test_cfg": {"flip_test": False},
            "data_preprocessor": {
                "type": "PoseDataPreprocessor",
                "bgr_to_rgb": True,
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
            },
            "head": {
                "type": "RTMCCHead",
                "out_channels": 18,
                "input_size": (192, 256),
                "simcc_split_ratio": 2.0,
                "decoder": {
                    "type": "SimCCLabel",
                    "input_size": (192, 256),
                    "simcc_split_ratio": 2.0,
                    "use_dark": False,
                },
            },
        },
        "val_dataloader": {
            "dataset": {
                "ann_file": "/training-only/data/valid.json",
                "pipeline": [
                    {"type": "LoadImage"},
                    {"type": "GetBBoxCenterScale", "padding": 3.5},
                    {"type": "TopdownAffine", "input_size": [192, 256]},
                    {"type": "PackPoseInputs"},
                ],
            }
        },
    }


def should_export_portable_pipeline_without_mutating_training_config(config):
    original = deepcopy(config)
    result = portable_pose_config(config, {"dataset_name": "harpet"})
    dataset = result["test_dataloader"]["dataset"]
    assert dataset["pipeline"][1] == {"type": "GetBBoxCenterScale", "padding": 3.5}
    assert "ann_file" not in dataset
    assert config == original


@pytest.mark.parametrize(
    "path,value",
    [
        (("val_dataloader", "dataset", "pipeline", 1, "padding"), 1.25),
        (("val_dataloader", "dataset", "pipeline", 2, "use_udp"), True),
        (("model", "data_preprocessor", "bgr_to_rgb"), False),
        (("model", "data_preprocessor", "mean"), [0, 0, 0]),
        (("model", "head", "decoder", "use_dark"), True),
        (("model", "head", "decoder", "simcc_split_ratio"), 1.0),
        (("model", "test_cfg", "flip_test"), True),
    ],
)
def should_refuse_native_preprocessing_mismatches(config, path, value):
    entry = config
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value
    with pytest.raises(ValueError, match="Unsupported HARPET native export settings"):
        portable_pose_config(config, {})
