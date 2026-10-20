"""STGCN++ with the actual HARPET graph and a portable pixel-space input."""

import torch
from hockey_data import EDGES
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from torch import nn


class HockeyAction(nn.Module):
    """N,1,3,18,3 (x,y,confidence); normalize inside both PyTorch and ONNX."""

    def __init__(self):
        super().__init__()
        register_all_modules()
        model = MODELS.build(
            {
                "type": "RecognizerGCN",
                "backbone": {
                    "type": "STGCN",
                    "gcn_adaptive": "init",
                    "gcn_with_res": True,
                    "tcn_type": "mstcn",
                    "num_person": 1,
                    "graph_cfg": {
                        "layout": {
                            "num_node": 18,
                            "inward": list(EDGES) + [(16, 10), (16, 15)],
                            "center": 6,
                        },
                        "mode": "spatial",
                    },
                },
                "cls_head": {
                    "type": "GCNHead",
                    "num_classes": 4,
                    "in_channels": 256,
                    "dropout": 0.5,
                },
            }
        )
        self.backbone = model.backbone
        self.cls_head = model.cls_head

    def forward(self, skeleton):
        xy, confidence = skeleton[..., :2], skeleton[..., 2:3]
        visible = confidence >= 0.3
        body_y = xy[..., :16, 1:2]
        body_valid = visible[..., :16, :]
        low = torch.where(body_valid, body_y, torch.full_like(body_y, 1e6)).amin(-2, keepdim=True)
        high = torch.where(body_valid, body_y, torch.full_like(body_y, -1e6)).amax(-2, keepdim=True)
        scale = (high - low).clamp(min=1)
        centered = (xy - xy[..., 6:7, :]) / scale
        normalized = torch.cat(
            (
                torch.where(visible, centered, torch.zeros_like(centered)),
                torch.where(visible, confidence, torch.zeros_like(confidence)),
            ),
            dim=-1,
        )
        return self.cls_head(self.backbone(normalized))
