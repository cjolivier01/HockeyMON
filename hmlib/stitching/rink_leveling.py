"""Fit camera-space rink tilt from vertical posts, using Hugin's PT axes.

Adapted from hstream's RinkLeveling. All points are original camera pixel
centers. Conversion uses a private equirectangular PTO, never the displayed
panorama or a two-dimensional approximation to the lens transform.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


def rotation_matrix(angles: Sequence[float]) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (np.abs(values) > 180).any():
        raise ValueError("Rink rotation must contain three finite angles between -180 and 180")
    y, p, r = np.radians(values) * [-1, -1, 1]
    cy, sy, cp, sp, cr, sr = (
        math.cos(y),
        math.sin(y),
        math.cos(p),
        math.sin(p),
        math.cos(r),
        math.sin(r),
    )
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def rotation_delta(published: Sequence[float], desired: Sequence[float]) -> tuple[float, ...]:
    """Compose desired * published.T; Euler-angle subtraction is not equivalent."""
    delta = rotation_matrix(desired) @ rotation_matrix(published).T
    pitch = math.asin(float(np.clip(delta[2, 0], -1, 1)))
    if math.hypot(delta[0, 0], delta[1, 0]) < 1e-10:
        yaw, roll = math.atan2(delta[0, 1], delta[1, 1]), 0
    else:
        yaw = math.atan2(-delta[1, 0], delta[0, 0])
        roll = math.atan2(delta[2, 1], delta[2, 2])
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))


@dataclass(frozen=True)
class LevelingProject:
    pto: str
    image_sizes: tuple[tuple[int, int], ...]


def prepare_project(pto: str) -> LevelingProject:
    if not pto or len(pto.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("The stitching project is empty or too large")
    output, sizes, links = [], [], []
    panoramas = 0
    for line in pto.splitlines():
        record = line.lstrip()
        if record.startswith(("p ", "p\t")):
            panoramas += 1
            output.append('p f2 w3600 h1800 v360 n"TIFF_m c:LZW r:CROP"')
            continue
        if record.startswith(("i ", "i\t")):
            dimensions = {}
            for token in shlex.split(record)[1:]:
                if token[0] in "wh":
                    value = float(token[1:])
                    if token[0] in dimensions or not 1 <= value <= 262144 or not value.is_integer():
                        raise ValueError("The project has invalid source image dimensions")
                    dimensions[token[0]] = int(value)
                if token.startswith(("TrX", "TrY", "TrZ")):
                    linked = token[3:].startswith("=")
                    value = float(token[4:] if linked else token[3:])
                    if (
                        not math.isfinite(value)
                        or (not linked and value != 0)
                        or (linked and (not value.is_integer() or not 0 <= value < 64))
                    ):
                        raise ValueError(
                            "Rink leveling requires a project without camera translation"
                        )
                    if linked:
                        links.append(int(value))
            if set(dimensions) != {"w", "h"}:
                raise ValueError("The project has missing source image dimensions")
            sizes.append((dimensions["w"], dimensions["h"]))
        output.append(line)
    if panoramas != 1 or not 1 <= len(sizes) <= 64:
        raise ValueError("The project must contain one panorama and 1 to 64 source images")
    if any(link >= len(sizes) for link in links):
        raise ValueError("The stitching project links a missing source image")
    return LevelingProject("\n".join(output) + "\n", tuple(sizes))


def format_points(posts: list[dict], image_sizes: Sequence[tuple[int, int]]) -> str:
    if not isinstance(posts, list) or not 3 <= len(posts) <= 64:
        raise ValueError("Select between 3 and 64 vertical posts")
    output = []
    for post in posts:
        if not isinstance(post, dict) or set(post) != {"image_index", "first", "second"}:
            raise ValueError("Each post requires image_index, first and second")
        index = post["image_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(image_sizes)
        ):
            raise ValueError("A selected post refers to a missing camera image")
        size = np.asarray(image_sizes[index])
        points = np.asarray([post["first"], post["second"]], dtype=float)
        if (
            points.shape != (2, 2)
            or not np.isfinite(points).all()
            or (points < 0).any()
            or (points > size - 1).any()
        ):
            raise ValueError("A selected post extends outside its camera image")
        if np.linalg.norm(points[0] - points[1]) < 4:
            raise ValueError("Mark a taller section of each post")
        output.extend(f"{index} {x:.17g} {y:.17g}\n" for x, y in points)
    return "".join(output)


def parse_rays(output: str, count: int) -> np.ndarray:
    if not 3 <= count <= 64 or len(output) > 32768:
        raise ValueError("Unexpected number of transformed rink marks")
    try:
        points = np.array([float(value) for value in output.split()]).reshape(count, 2, 2)
    except ValueError as exc:
        raise ValueError("The stitching transform returned unexpected output") from exc
    if (
        not np.isfinite(points).all()
        or (points < -0.500001).any()
        or (points > [3599.500001, 1799.500001]).any()
    ):
        raise ValueError("A selected point could not be transformed by the stitching calibration")
    longitude = (points[..., 0] - 1799.5) * math.pi / 1800
    latitude = (899.5 - points[..., 1]) * math.pi / 1800
    return np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            -np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class LevelingEstimate:
    rotation_degrees: tuple[float, ...]
    inlier_indices: tuple[int, ...]
    residual_degrees: tuple[float, ...]
    rms_residual_degrees: float


def estimate_leveling(
    ray_lines: np.ndarray, published_rotation: Sequence[float], preserved_yaw: float
) -> LevelingEstimate:
    rays = np.asarray(ray_lines, dtype=float)
    if rays.ndim != 3 or rays.shape[1:] != (2, 3) or not 3 <= len(rays) <= 64:
        raise ValueError("Select between 3 and 64 vertical posts")
    undo = rotation_matrix(published_rotation).T
    rotation_matrix([preserved_yaw, 0, 0])
    lengths = np.linalg.norm(rays, axis=-1, keepdims=True)
    if not np.isfinite(rays).all() or not np.isfinite(lengths).all() or (lengths < 1e-12).any():
        raise ValueError("A selected point has an invalid viewing ray")
    rays = (rays / lengths) @ undo.T
    crosses = np.cross(rays[:, 0], rays[:, 1])
    lengths = np.linalg.norm(crosses, axis=-1)
    angles = np.arctan2(lengths, np.sum(rays[:, 0] * rays[:, 1], axis=-1))
    if (angles < math.radians(0.5)).any() or (angles > math.pi / 2).any():
        raise ValueError("Mark a taller, clearly visible section of each vertical post")
    normals = crosses / lengths[:, None]
    weights = np.minimum(lengths, math.sin(math.radians(20))) ** 2
    threshold = math.radians(3)

    def residuals(up):
        return np.arcsin(np.clip(np.abs(normals @ up), 0, 1))

    selected, best_cost = np.array([], dtype=int), math.inf
    for first, second in combinations(normals, 2):
        candidate = np.cross(first, second)
        length = np.linalg.norm(candidate)
        if length < 0.15:
            continue
        errors = residuals(candidate / length)
        indices = np.flatnonzero(errors <= threshold)
        cost = np.sum(np.minimum(errors, threshold) ** 2)
        if len(indices) > len(selected) or (len(indices) == len(selected) and cost < best_cost):
            selected, best_cost = indices, cost
    required = max(3, math.ceil(0.7 * len(rays)))
    for _ in range(8):
        if len(selected) < required:
            raise ValueError("The marks disagree; select at least three well-spaced vertical posts")
        fit = normals[selected]
        values, vectors = np.linalg.eigh((fit.T * weights[selected]) @ fit)
        if values[1] < 0.0225 * values[2]:
            raise ValueError(
                "Choose posts farther apart across the rink; these marks cannot determine tilt"
            )
        up = vectors[:, 0]
        indices = np.flatnonzero(residuals(up) <= threshold)
        if np.array_equal(indices, selected):
            break
        selected = indices
    else:
        raise ValueError("These marks do not give a stable tilt; adjust the post endpoints")
    if up @ (undo @ [0, 0, 1]) < 0:
        up = -up
    errors = np.degrees(residuals(up))
    return LevelingEstimate(
        (
            float(preserved_yaw),
            math.degrees(math.atan2(up[0], math.hypot(up[1], up[2]))),
            math.degrees(math.atan2(up[1], up[2])),
        ),
        tuple(int(index) for index in selected),
        tuple(float(error) for error in errors),
        float(np.sqrt(np.mean(errors[selected] ** 2))),
    )
