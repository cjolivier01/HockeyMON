"""Native RTMPose-M COCO17 inference contract, with no test-time flipping.

Architecture and preprocessing mirror the OpenMMLab configuration pinned by
HStream: rtmpose-m_8xb256-420e_coco-256x192.py, SHA256
532dc32cd9c6ba4ea3300fdc0e0a1da622dd9f53510a1998968f7958e08b2a6b.
The Apache-2.0 upstream checkpoint and COCO/AIC data provenance are retained.
"""

from hmlib.visualization import PytorchPoseLocalVisualizer

default_scope = "mmpose"
dataset_type = "CocoDataset"
data_mode = "topdown"
visualizer = {"type": PytorchPoseLocalVisualizer, "name": "visualizer", "line_width": 2}

codec = {
    "type": "SimCCLabel",
    "input_size": (192, 256),
    "sigma": (4.9, 5.66),
    "simcc_split_ratio": 2.0,
    "normalize": False,
    "use_dark": False,
}
model = {
    "type": "TopdownPoseEstimator",
    "data_preprocessor": {
        "type": "PoseDataPreprocessor",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "bgr_to_rgb": True,
    },
    "backbone": {
        "_scope_": "mmdet",
        "type": "CSPNeXt",
        "arch": "P5",
        "expand_ratio": 0.5,
        "deepen_factor": 0.67,
        "widen_factor": 0.75,
        "out_indices": (4,),
        "channel_attention": True,
        "norm_cfg": {"type": "SyncBN"},
        "act_cfg": {"type": "SiLU"},
        "init_cfg": None,
    },
    "head": {
        "type": "RTMCCHead",
        "in_channels": 768,
        "out_channels": 17,
        "input_size": (192, 256),
        "in_featuremap_size": (6, 8),
        "simcc_split_ratio": 2.0,
        "final_layer_kernel_size": 7,
        "gau_cfg": {
            "hidden_dims": 256,
            "s": 128,
            "expansion_factor": 2,
            "dropout_rate": 0.0,
            "drop_path": 0.0,
            "act_fn": "SiLU",
            "use_rel_bias": False,
            "pos_enc": False,
        },
        "loss": {
            "type": "KLDiscretLoss",
            "use_target_weight": True,
            "beta": 10.0,
            "label_softmax": True,
        },
        "decoder": codec,
    },
    "test_cfg": {"flip_test": False},
}
val_pipeline = [
    {"type": "LoadImage", "backend_args": {"backend": "local"}},
    {"type": "GetBBoxCenterScale", "padding": 1.25},
    {"type": "TopdownAffine", "input_size": (192, 256)},
    {"type": "PackPoseInputs"},
]
val_dataloader = {"dataset": {"type": "CocoDataset", "pipeline": val_pipeline}}
test_dataloader = val_dataloader
