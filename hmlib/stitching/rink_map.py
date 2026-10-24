"""Rink evidence in calibrated rays, independent of panorama projection.

Port of HStream's point/line plane fit. Public rink coordinates retain the
Python camera's corner origin; feature coordinates below use centre ice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

POINT_FEATURES = (
    "center",
    "center-near",
    "center-far",
    "blue-left-near",
    "blue-left-far",
    "blue-right-near",
    "blue-right-far",
    "goal-left",
    "goal-right",
    "goal-left-near",
    "goal-left-far",
    "goal-right-near",
    "goal-right-far",
)
LINE_FEATURES = (
    "center-line",
    "blue-left-line",
    "blue-right-line",
    "goal-left-line",
    "goal-right-line",
    "near-boards",
    "far-boards",
)
DEFAULT_LAYOUT = {
    "length": 60.96,
    "width": 25.908,
    "blue_line_distance": 7.62,
    "goal_inset": 3.3528,
    "corner_radius": 8.5344,
}


def layout_settings(values: Mapping | None = None) -> dict:
    result = {**DEFAULT_LAYOUT, **(values or {})}
    if set(result) != set(DEFAULT_LAYOUT):
        raise ValueError("Unknown rink dimension")
    result = {key: float(value) for key, value in result.items()}
    length, width, blue, goal, radius = (result[key] for key in DEFAULT_LAYOUT)
    if not all(math.isfinite(v) for v in result.values()) or not (
        30 <= length <= 80
        and 12 <= width <= 40
        and width < length
        and 1 <= goal <= 6
        and 1 <= blue < length / 2 - goal
        and 0 <= radius <= width / 2
    ):
        raise ValueError(
            "Check rink length, width, blue-line spacing, goal inset and corner radius"
        )
    return result


def feature_point(feature: str, layout: Mapping) -> np.ndarray:
    if feature not in POINT_FEATURES:
        raise ValueError(f"Unknown rink point: {feature}")
    x, y = 0.0, 0.0
    for prefix, value in (
        ("blue-left", -layout["blue_line_distance"]),
        ("blue-right", layout["blue_line_distance"]),
        ("goal-left", -layout["length"] / 2 + layout["goal_inset"]),
        ("goal-right", layout["length"] / 2 - layout["goal_inset"]),
    ):
        if feature.startswith(prefix):
            x = value
    if feature.endswith("-near"):
        y = -layout["width"] / 2
    elif feature.endswith("-far"):
        y = layout["width"] / 2
    radius = layout["corner_radius"]
    if y and radius and abs(x) > layout["length"] / 2 - radius:
        offset = abs(x) - (layout["length"] / 2 - radius)
        y = math.copysign(
            layout["width"] / 2 - radius + math.sqrt(max(0, radius**2 - offset**2)), y
        )
    return np.array([x, y])


def feature_line(feature: str, layout: Mapping) -> np.ndarray:
    if feature not in LINE_FEATURES:
        raise ValueError(f"Unknown rink line: {feature}")
    if feature in ("near-boards", "far-boards"):
        return np.array([0, 1, layout["width"] / 2 * (1 if feature == "near-boards" else -1)])
    x = feature_point(
        feature.removesuffix("-line") + ("-near" if feature != "center-line" else ""), layout
    )[0]
    return np.array([1, 0, -x])


def unit(values):
    values = np.asarray(values, dtype=float)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (lengths < 1e-12).any():
        raise ValueError("Invalid calibrated viewing ray")
    return values / lengths


def _rows(a, b, transpose=False, weight=1.0):
    cross = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rows = np.einsum("ij,k->ijk", cross, b)
    if transpose:
        rows = rows.transpose(0, 2, 1)
    return rows.reshape(3, 9) * weight


def fit_rink_map(observations: Sequence, layout: Mapping) -> dict:
    layout = layout_settings(layout)
    if len(observations) > 128:
        raise ValueError("Keep at most 128 rink observations")
    result = {
        "state": "partial",
        "message": "Add independent marks across and along the rink; parallel lines alone cannot locate players. Partial marks can be saved.",
        "constraints": 0,
        "rms_degrees": 0.0,
    }
    rows, ideal_rows, used, normals = [], [], [], []
    scale = layout["length"] / 2
    for mark in observations:
        feature, line = mark.get("feature", ""), bool(mark.get("line", False))
        pixels = np.asarray(mark["pixels"], dtype=float)
        rays = np.asarray(mark["rays"], dtype=float)
        if (
            pixels.ndim != 2
            or pixels.shape[1:] != (2,)
            or not 1 <= len(pixels) <= 64
            or rays.shape != (len(pixels), 3)
            or not np.isfinite(pixels).all()
            or not np.isfinite(rays).all()
            or not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-5)
            or (line and len(pixels) < 2)
            or (not line and len(pixels) != 1)
            or (mark.get("arc", False) and (not line or len(pixels) != 3))
            or not 0 <= float(mark.get("confidence", 1)) <= 1
        ):
            raise ValueError("Invalid rink landmark observation")
        if feature and feature not in (LINE_FEATURES if line else (*POINT_FEATURES, "custom")):
            raise ValueError("Unknown or mismatched rink feature")
        if not mark.get("enabled", True) or not feature:
            continue
        weight = math.sqrt(max(0.05, mark.get("confidence", 1))) if mark.get("automatic") else 1
        if line:
            values, vectors = np.linalg.eigh(rays.T @ rays)
            if values[1] < values[2] * 1e-5:
                return {
                    **result,
                    "state": "inconsistent",
                    "message": "A rink line needs marks farther apart",
                }
            normal = vectors[:, 0]
            if np.max(np.abs(rays @ normal)) > math.sin(math.radians(1.5)):
                return {
                    **result,
                    "state": "inconsistent",
                    "message": "A selected marking does not follow a straight physical line on the ice",
                }
            physical = feature_line(feature, layout)
            physical[2] /= scale
            physical = unit(physical)
            rows.extend(_rows(physical, normal, True, weight))
            ideal_rows.extend(_rows(physical, physical, True))
            normals.append(normal)
        else:
            point = (
                np.asarray(mark["rink"], dtype=float) - [layout["length"] / 2, layout["width"] / 2]
                if feature == "custom"
                else feature_point(feature, layout)
            )
            physical = np.r_[point / scale, 1]
            rows.extend(_rows(rays[0], physical, weight=weight))
            ideal_rows.extend(_rows(unit(physical), physical))
            normals.append(None)
        used.append(mark)
    result["constraints"] = len(used)
    if len(rows) < 9:
        return result
    singular = np.linalg.svd(ideal_rows, compute_uv=False)
    if len(singular) < 9 or singular[7] < singular[0] * 1e-8:
        return result
    _, singular, vectors = np.linalg.svd(rows, full_matrices=False)
    if singular[7] < singular[0] * 1e-3:
        return result
    transform = vectors[-1].reshape(3, 3)
    transform[:, :2] /= scale
    if abs(np.linalg.det(transform)) < 1e-16:
        return result
    inverse = np.linalg.inv(transform)
    sign = sum(np.sign((inverse @ np.asarray(ray))[2]) for mark in used for ray in mark["rays"])
    if sign < 0:
        transform *= -1
        inverse *= -1
    x, y, origin = transform.T
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    normal = np.cross(x, y)
    height = abs(normal @ origin) / max(1e-20, np.linalg.norm(normal) * (nx + ny) / 2)
    invalid = {
        **result,
        "state": "inconsistent",
        "message": "Marks cannot fit a reliable metric ice plane. Check feature identities, dimensions and camera calibration; partial evidence can be saved.",
    }
    if (
        nx < 1e-10
        or ny < 1e-10
        or not 0.8 < nx / ny < 1.25
        or abs(x @ y) / (nx * ny) > 0.15
        or not 0.3 < height < 50
    ):
        return invalid
    errors = []
    for mark, normal in zip(used, normals):
        if mark["line"]:
            predicted = unit(inverse.T @ feature_line(mark["feature"], layout))
            cosine = abs(predicted @ normal)
        else:
            point = (
                np.asarray(mark["rink"]) - [layout["length"] / 2, layout["width"] / 2]
                if mark["feature"] == "custom"
                else feature_point(mark["feature"], layout)
            )
            predicted = unit(transform @ np.r_[point, 1])
            cosine = predicted @ mark["rays"][0]
        if (np.asarray(mark["rays"]) @ inverse.T)[:, 2].min() <= 0:
            return invalid
        errors.append(math.degrees(math.acos(np.clip(cosine, -1, 1))))
    rms = float(np.sqrt(np.mean(np.square(errors))))
    if rms > 1 or max(errors) > 2:
        return {**invalid, "rms_degrees": rms}
    return {
        **result,
        "state": "ready",
        "message": "Metric rink map fitted. Review goal centres and board extents in both cameras.",
        "plane_to_ray": transform.tolist(),
        "rms_degrees": rms,
        "camera_height": float(height),
    }


def rink_leveling(observations, fit, layout, published_rotation, yaw):
    from hmlib.stitching.rink_leveling import rotation_matrix

    if fit["state"] == "ready":
        transform = np.asarray(fit["plane_to_ray"])
        rays = np.array(
            [
                unit(transform @ [x, y, 1])
                for x in (-layout["blue_line_distance"], layout["blue_line_distance"])
                for y in (-layout["width"] / 2, layout["width"] / 2)
            ]
        )
    else:
        rays = None
        for pair in (
            ("goal-left", "goal-right"),
            ("blue-left", "blue-right"),
            ("center", "blue-left"),
            ("center", "blue-right"),
        ):
            corners = []
            for end in pair:
                for side in ("near", "far"):
                    matches = [
                        m["rays"][0]
                        for m in observations
                        if m.get("enabled", True)
                        and not m["line"]
                        and m["feature"] == f"{end}-{side}"
                    ]
                    if matches:
                        average = unit(np.sum(matches, axis=0))
                        if any(np.linalg.norm(average - ray) > 0.035 for ray in matches):
                            raise ValueError(
                                "Repeated board intersections disagree between cameras"
                            )
                        corners.append(average)
            if len(corners) == 4:
                rays = np.array(corners)
                break
        if rays is None:
            raise ValueError(
                "Mark near/far board intersections of two goal or blue lines, or complete the rink map. Arbitrary detected line endpoints cannot determine leveling."
            )
    rays = rays @ rotation_matrix(published_rotation)

    def direction(a, b, c, d):
        first, second = np.cross(a, b), np.cross(c, d)
        if min(np.linalg.norm(first), np.linalg.norm(second)) < math.sin(math.radians(0.5)):
            raise ValueError("Select distinct corners farther apart on the ice")
        cross = np.cross(unit(first), unit(second))
        if np.linalg.norm(cross) < 0.01:
            raise ValueError("Corners are too aligned to determine the ice plane")
        return unit(cross)

    across = direction(*rays)
    along = direction(rays[0], rays[2], rays[1], rays[3])
    up = unit(np.cross(across, along))
    validation = up * (-1 if up @ rays.sum(axis=0) > 0 else 1)
    if (rays @ validation > -math.sin(math.radians(0.25))).any():
        raise ValueError("Corner order crosses the rectangle or lies too close to the horizon")
    if up[2] < 0:
        up *= -1
    rotation_matrix([yaw, 0, 0])
    return {
        "rotation_degrees": [
            yaw,
            math.degrees(math.atan2(up[0], math.hypot(up[1], up[2]))),
            math.degrees(math.atan2(up[1], up[2])),
        ],
        "orthogonality_error_degrees": math.degrees(math.asin(np.clip(abs(across @ along), 0, 1))),
    }


def layout_drawing(layout):
    """Rounded boards, cross-rink markings and goal-mouth crosses in metres."""
    lines = []
    radius, lx, wy = layout["corner_radius"], layout["length"] / 2, layout["width"] / 2
    outline = []
    for corner in range(4):
        cx = (1 if corner in (0, 3) else -1) * (lx - radius)
        cy = (1 if corner < 2 else -1) * (wy - radius)
        for step in range(13):
            angle = math.radians(corner * 90 + step * 7.5)
            outline.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
    lines.append([*outline, outline[0]])
    for feature in ("goal-left", "blue-left", "center", "blue-right", "goal-right"):
        lines.append(
            [feature_point(feature + "-near", layout), feature_point(feature + "-far", layout)]
        )
    for side in ("goal-left", "goal-right"):
        point = feature_point(side, layout)
        lines.extend([[point + [-0.6, 0], point + [0.6, 0]], [point + [0, -0.6], point + [0, 0.6]]])
    return lines
