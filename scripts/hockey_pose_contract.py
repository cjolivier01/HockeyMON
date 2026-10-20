"""Validate the fixed HStream pose profile before producing deployable assets."""

from copy import deepcopy

PREPROCESSING = {
    "id": "rtmpose-harpet18-affine-v1",
    "color": "rgb",
    "padding": 3.5,
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
    "simcc_split_ratio": 2.0,
    "flip_test": False,
}


def portable_pose_config(config, metainfo):
    """Reject incompatible inference settings and remove training-local data paths."""
    model = config["model"]
    pipeline = deepcopy(config["val_dataloader"]["dataset"]["pipeline"])
    expected_pipeline = [
        {"type": "LoadImage"},
        {"type": "GetBBoxCenterScale", "padding": PREPROCESSING["padding"]},
        {"type": "TopdownAffine", "input_size": (192, 256)},
        {"type": "PackPoseInputs"},
    ]
    # Lists and tuples have identical meaning for this MMPose argument.
    for step in pipeline:
        if "input_size" in step:
            step["input_size"] = tuple(step["input_size"])
    expected_preprocessor = {
        "type": "PoseDataPreprocessor",
        "bgr_to_rgb": True,
        "mean": PREPROCESSING["mean"],
        "std": PREPROCESSING["std"],
    }
    head = model["head"]
    decoder = head["decoder"]
    checks = {
        "validation pipeline (including padding 3.5)": pipeline == expected_pipeline,
        "RGB normalization": model["data_preprocessor"] == expected_preprocessor,
        "test configuration": model["test_cfg"] == {"flip_test": False},
        "pose estimator": model["type"] == "TopdownPoseEstimator",
        "18-joint SimCC head": head["type"] == "RTMCCHead" and head["out_channels"] == 18,
        "head input size": tuple(head["input_size"]) == (192, 256),
        "head SimCC ratio": head["simcc_split_ratio"] == 2.0,
        "decoder": decoder["type"] == "SimCCLabel",
        "decoder input size": tuple(decoder["input_size"]) == (192, 256),
        "decoder SimCC ratio": decoder["simcc_split_ratio"] == 2.0,
        "decoder refinement": not decoder.get("use_dark", False),
        "decoder visibility": not decoder.get("decode_visibility", False),
    }
    failures = [name for name, valid in checks.items() if not valid]
    if failures:
        raise ValueError("Unsupported HARPET native export settings: " + ", ".join(failures))
    return {
        "model": deepcopy(model),
        "dataset_meta": deepcopy(metainfo),
        "default_scope": "mmpose",
        "test_dataloader": {
            "dataset": {"type": "CocoDataset", "metainfo": deepcopy(metainfo), "pipeline": pipeline}
        },
    }
