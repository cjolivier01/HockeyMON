# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

"""SuperPoint detector/descriptor, vendored from cvg/LightGlue.

Kornia supplies LightGlue but no SuperPoint, because the Magic Leap terms in
the banner above are incompatible with kornia's Apache-2.0 licence.

The network and its tensor math are upstream's. What differs: the
``Extractor``/``ImagePreprocessor`` helpers of ``lightglue.utils`` are folded
into :meth:`SuperPoint.extract` so the rest of ``lightglue`` is not needed,
the checkpoint is loaded with ``weights_only``, and the file is formatted to
this repo's Black/ruff settings rather than upstream's.

Source: https://github.com/cvg/LightGlue/blob/main/lightglue/superpoint.py
See SUPERPOINT_NOTICE.md in this directory for licensing.
"""

from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from kornia.color import rgb_to_grayscale
from kornia.geometry.transform import resize
from torch import nn

# Longest edge, in pixels, that SuperPoint sees. Upstream's ``preprocess_conf``.
SUPERPOINT_RESIZE = 1024
WEIGHTS_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    )
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_conf: ClassVar[dict[str, Any]] = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }

    required_data_keys: ClassVar[list[str]] = ["image"]

    def __init__(self, **conf: Any) -> None:
        super().__init__()
        unknown = set(conf) - set(self.default_conf)
        if unknown:
            raise TypeError(f"Unknown SuperPoint options: {', '.join(sorted(unknown))}")
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0)

        self.load_state_dict(
            torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", weights_only=True)
        )

    @torch.no_grad()
    def extract(self, image: torch.Tensor, *, resolution: str = "1k") -> dict[str, torch.Tensor]:
        """Detect and describe one image, resizing it to SUPERPOINT_RESIZE first.

        @param image: RGB tensor in ``[0, 1]``, either ``CHW`` or ``1CHW``.
        @return: Dict with ``keypoints`` (in ``image``'s own pixel coordinates),
            ``keypoint_scores``, ``descriptors``, and ``image_size`` as ``(w, h)``.
        """
        if image.dim() == 3:
            image = image[None]  # add batch dim
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError("SuperPoint.extract expects a single CHW or 1CHW image")
        if not image.is_floating_point():
            raise TypeError(f"SuperPoint.extract expects a float image, got {image.dtype}")
        height, width = image.shape[-2:]
        if resolution not in ("native", "1k", "2k"):
            raise ValueError("SuperPoint resolution must be native, 1k or 2k")
        if resolution == "native":
            resized = image
        elif resolution == "1k":
            if max(height, width) * 9 < SUPERPOINT_RESIZE:
                raise ValueError(
                    f"SuperPoint.extract needs an image of at least {SUPERPOINT_RESIZE // 9} pixels "
                    f"on its long edge, got {width}x{height}"
                )
            resized = resize(image, SUPERPOINT_RESIZE, side="long", antialias=True)
        else:
            scale = min(2048 / width, 1152 / height)
            resized = resize(
                image, (max(1, round(height * scale)), max(1, round(width * scale))), antialias=True
            )
        resized_height, resized_width = resized.shape[-2:]
        if resolution == "2k":
            padded_height, padded_width = 1152, 2048
        else:
            padded_height = ((resized_height + 7) // 8) * 8
            padded_width = ((resized_width + 7) // 8) * 8
        padded = torch.nn.functional.pad(
            resized, (0, padded_width - resized_width, 0, padded_height - resized_height)
        )
        # Keep the coordinate maths in float32: a float16 image would quantise the
        # scale factors enough to shift restored keypoints by pixels at 8K.
        factors = torch.tensor(
            [resized.shape[-1] / width, resized.shape[-2] / height],
            device=image.device,
            dtype=torch.float32,
        )
        feats = self.forward({"image": padded})
        points = feats["keypoints"][0]
        keep = (points[:, 0] < resized_width) & (points[:, 1] < resized_height)
        for key in ("keypoints", "keypoint_scores", "descriptors"):
            feats[key] = feats[key][:, keep]
        feats["image_size"] = factors.new_tensor([[width, height]])
        feats["keypoints"] = (feats["keypoints"].float() + 0.5) / factors[None] - 0.5
        points = feats["keypoints"][0]
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        for key in ("keypoints", "keypoint_scores", "descriptors"):
            feats[key] = feats[key][:, keep]
        return feats

    def forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        best_kp = torch.where(scores > self.conf.detection_threshold)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Keep the k keypoints with highest score
        if self.conf.max_num_keypoints is not None:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.conf.max_num_keypoints)
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0] for k, d in zip(keypoints, descriptors)
        ]

        return {
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(scores, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }
