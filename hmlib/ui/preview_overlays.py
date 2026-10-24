"""Metadata-only geometry and drawing for the shared desktop's bounded preview.

The recording tensor is never changed. Drawing happens after the preview worker
has resized and copied its selected frame to CPU. Track data is compact; the
static rink mask is reduced to a bounded contour proxy once per mask identity.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def copy_metadata(metadata: list[dict] | None) -> list[dict] | None:
    if metadata is None:
        return None
    return [
        {**item, "matrix": np.array(item["matrix"], dtype=np.float64, copy=True)}
        for item in metadata
    ]


def compose_transform(metadata: list[dict] | None, index: int, transform: np.ndarray) -> None:
    if metadata is not None:
        metadata[index]["matrix"] = np.asarray(transform) @ metadata[index]["matrix"]


def _array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class PreviewOverlayRenderer:
    def __init__(self) -> None:
        self._mask_identity: int | None = None
        self._mask_owner: Any = None
        self._mask_polygons: list[np.ndarray] = []

    def _rink_polygons(self, mask: Any) -> list[np.ndarray]:
        if mask is None:
            return []
        if id(mask) != self._mask_identity:
            if hasattr(mask, "detach"):
                from torch.nn import functional

                tensor = mask.detach().squeeze()
                if tensor.ndim != 2:
                    raise ValueError("Preview rink mask must be two dimensional")
                height, width = tensor.shape
                scale = min(1.0, 512 / max(height, width))
                proxy = (
                    functional.interpolate(
                        tensor[None, None].float(),
                        size=(max(1, round(height * scale)), max(1, round(width * scale))),
                        mode="nearest",
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
            else:
                array = np.asarray(mask).squeeze()
                if array.ndim != 2:
                    raise ValueError("Preview rink mask must be two dimensional")
                height, width = array.shape
                scale = min(1.0, 512 / max(height, width))
                proxy = cv2.resize(
                    array.astype(np.uint8),
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_NEAREST,
                )
            contours, _ = cv2.findContours(
                (proxy > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            scale_xy = np.array([width / proxy.shape[1], height / proxy.shape[0]])
            self._mask_polygons = [
                contour.reshape(-1, 2) * scale_xy for contour in contours if len(contour) >= 3
            ]
            self._mask_identity = id(mask)
            self._mask_owner = mask
        return self._mask_polygons

    def draw(
        self,
        image: np.ndarray,
        metadata: dict | None,
        source_size: tuple[int, int],
        layers: tuple[bool, bool, bool],
    ) -> np.ndarray:
        if metadata is None or not any(layers):
            return image
        output = image.copy()
        scale = np.diag([image.shape[1] / source_size[0], image.shape[0] / source_size[1], 1])
        matrix = scale @ np.asarray(metadata["matrix"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Invalid preview geometry transform")

        def project(points: Any) -> np.ndarray:
            vertices = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            homogeneous = np.column_stack((vertices, np.ones(len(vertices)))) @ matrix.T
            projected = homogeneous[:, :2] / homogeneous[:, 2:]
            if not np.isfinite(projected).all():
                raise ValueError("Invalid projected preview coordinates")
            return np.rint(np.clip(projected, -(1 << 20), 1 << 20)).astype(np.int32)

        def rectangle(box: Any, color: tuple[int, int, int], label: str = "") -> None:
            x0, y0, x1, y1 = map(float, box)
            polygon = project([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
            if label:
                x, y = polygon[0]
                cv2.putText(
                    output,
                    label,
                    (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        players, play, rink = layers
        if rink:
            polygons = [
                project(polygon) for polygon in self._rink_polygons(metadata.get("rink_mask"))
            ]
            if polygons:
                tint = output.copy()
                cv2.fillPoly(tint, polygons, (30, 170, 210))
                output = cv2.addWeighted(output, 0.8, tint, 0.2, 0)
        if players:
            boxes = _array(metadata.get("boxes", [])).reshape(-1, 4)
            ids = _array(metadata.get("ids", [])).reshape(-1)
            if len(boxes) != len(ids) or len(ids) > 10000:
                raise ValueError("Preview player metadata is malformed or oversized")
            for box, identity in zip(boxes, ids):
                index = int(identity)
                color = (
                    (index * 37 + 85) % 210 + 45,
                    (index * 71 + 35) % 210 + 45,
                    (index * 113 + 15) % 210 + 45,
                )
                rectangle(box, color, str(index))
        if play:
            for key, color, label in (
                ("play_box", (0, 255, 0), "Program"),
                ("fast_box", (0, 160, 255), "Play"),
            ):
                if metadata.get(key) is not None:
                    rectangle(_array(metadata[key]).reshape(4), color, label)
        return output
