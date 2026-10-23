"""Calibrated source rays and bounded projection lookup, built once in the editor.

Hugin owns every projection and lens equation. A checked ray field records its
panorama transform, including its pixel centre and crop convention. Playback
only interpolates this field; it never fits a homography to panorama pixels.
"""

from __future__ import annotations

import base64
import io
import math
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from hmlib.stitching.leveling_editor import run_hugin
from hmlib.stitching.projections import read_panorama_geometry
from hmlib.stitching.rink_leveling import prepare_project
from hmlib.stitching.rink_map import unit


def sphere_rays(points):
    points = np.asarray(points, dtype=float)
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


def ray_sphere(rays):
    rays = unit(rays)
    return np.stack(
        [
            1799.5 - np.arctan2(rays[..., 1], rays[..., 0]) * 1800 / math.pi,
            899.5 - np.arcsin(np.clip(rays[..., 2], -1, 1)) * 1800 / math.pi,
        ],
        axis=-1,
    )


def transform_points(pto, points, *, reverse=False, run=run_hugin):
    if not len(points):
        return np.empty((0, 2))
    with tempfile.TemporaryDirectory(prefix="hm-rink-rays-") as directory:
        project = Path(directory) / "rays.pto"
        project.write_text(pto, encoding="utf-8")
        output = run(
            ["pano_trafo", *(["-r"] if reverse else []), str(project)],
            input="".join(f"{int(camera)} {x:.17g} {y:.17g}\n" for camera, x, y in points),
        )
    try:
        values = np.array([float(v) for v in output.split()]).reshape(len(points), 2)
    except ValueError as error:
        raise ValueError("Calibrated coordinate transform returned malformed points") from error
    return values


def source_rays(pto, observations, *, run=run_hugin):
    project = prepare_project(pto)
    points = []
    for mark in observations:
        camera = mark.get("camera", 0)
        if (
            isinstance(camera, bool)
            or not isinstance(camera, int)
            or not 0 <= camera < len(project.image_sizes)
        ):
            raise ValueError("Rink observation refers to a missing camera")
        pixels = np.asarray(mark["pixels"], dtype=float)
        if (
            pixels.ndim != 2
            or pixels.shape[1:] != (2,)
            or not 1 <= len(pixels) <= 64
            or not np.isfinite(pixels).all()
            or (pixels < 0).any()
            or (pixels > np.asarray(project.image_sizes[camera]) - 1).any()
        ):
            raise ValueError("Keep rink observations inside their source camera image")
        points.extend([[camera, *pixel] for pixel in pixels])
    transformed = transform_points(project.pto, points, run=run)
    if (
        not np.isfinite(transformed).all()
        or (transformed < -0.50001).any()
        or (transformed > [3599.50001, 1799.50001]).any()
    ):
        raise ValueError("A mark lies outside the calibrated camera geometry")
    rays = sphere_rays(transformed)
    offset = 0
    for mark in observations:
        count = len(mark["pixels"])
        mark["rays"] = rays[offset : offset + count].tolist()
        offset += count
    return observations


def rays_to_sources(pto, rays, camera_count, *, run=run_hugin):
    pixels = ray_sphere(rays)
    result = transform_points(
        prepare_project(pto).pto,
        [[camera, *point] for camera in range(camera_count) for point in pixels],
        reverse=True,
        run=run,
    )
    return result.reshape(camera_count, -1, 2)


def _interpolate(grid, pixels, size):
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError("Panorama points must be finite")
    grid_size = np.array([grid.shape[1] - 1, grid.shape[0] - 1])
    position = (pixels + 0.5) / np.asarray(size) * grid_size
    # A small border extrapolation supports positions at the pixel edge.
    indices = np.minimum(np.maximum(np.floor(position).astype(int), 0), grid_size - 1)
    fraction = position - indices
    x, y = indices.T
    fx, fy = fraction.T
    result = (
        grid[y, x] * ((1 - fx) * (1 - fy))[:, None]
        + grid[y, x + 1] * (fx * (1 - fy))[:, None]
        + grid[y + 1, x] * ((1 - fx) * fy)[:, None]
        + grid[y + 1, x + 1] * (fx * fy)[:, None]
    )
    return unit(result)


def build_panorama_projection(project_path, image_size, *, run=run_hugin):
    project_path = Path(project_path)
    geometry = read_panorama_geometry(project_path)
    panorama = next(line for line in project_path.read_text().splitlines() if line.startswith("p "))
    # A virtual, complete equirectangular source exposes libpano's exact inverse
    # output transform. It has no optics/pose of its own to alter camera rays.
    pto = panorama + '\ni w3600 h1800 f4 v360 y0 p0 r0 a0 b0 c0 d0 e0 n"sphere.png"\n'
    crop = np.array([geometry.crop[0], geometry.crop[2]])
    extent = np.asarray(geometry.effective_size)
    size = np.asarray(image_size, dtype=float)

    def sample(pixels):
        full = (np.asarray(pixels) + 0.5) / size * extent + crop - 0.5
        transformed = transform_points(pto, [[0, *p] for p in full], reverse=True, run=run)
        if not np.isfinite(transformed).all():
            raise ValueError(
                "Panorama includes undefined projection coordinates; crop to valid canvas before mapping the rink"
            )
        return sphere_rays(transformed)

    for columns in (65, 129, 257, 513):
        rows = (columns + 1) // 2
        xx, yy = np.meshgrid(
            np.linspace(-0.5, size[0] - 0.5, columns), np.linspace(-0.5, size[1] - 0.5, rows)
        )
        grid = sample(np.stack([xx.ravel(), yy.ravel()], axis=-1)).reshape(rows, columns, 3)
        probe = np.stack(
            [(xx[:-1, :-1] + xx[1:, 1:]).ravel() / 2, (yy[:-1, :-1] + yy[1:, 1:]).ravel() / 2],
            axis=-1,
        )
        exact = sample(probe)
        error = float(np.max(np.linalg.norm(exact - _interpolate(grid, probe, size), axis=1)))
        if error < 2e-5:
            stream = io.BytesIO()
            np.savez_compressed(stream, rays=grid.astype(np.float32))
            return {
                "encoding": "hugin-unit-rays-v1",
                "data": base64.b64encode(stream.getvalue()).decode("ascii"),
                "image_size": list(image_size),
                "maximum_probe_error_radians": error,
            }
    raise ValueError(
        "Panorama curvature exceeds the rink projection accuracy limit; use a tighter canvas"
    )


@lru_cache(maxsize=4)
def _decode(data):
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("Saved rink ray field exceeds its size limit")
    decoded = base64.b64decode(data, validate=True)
    with zipfile.ZipFile(io.BytesIO(decoded)) as archive:
        members = archive.infolist()
        if (
            len(members) != 1
            or members[0].filename != "rays.npy"
            or members[0].file_size > 513 * 257 * 3 * 8 + 1024
        ):
            raise ValueError("Saved rink ray field exceeds its decoded size limit")
    with np.load(io.BytesIO(decoded), allow_pickle=False) as archive:
        grid = archive["rays"]
    if (
        grid.ndim != 3
        or grid.shape[2] != 3
        or not 2 <= grid.shape[0] <= 257
        or not 2 <= grid.shape[1] <= 513
        or not np.isfinite(grid).all()
    ):
        raise ValueError("Invalid saved panorama ray field")
    return grid


def rotate_panorama(points, size, degrees):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    center = (np.asarray(size) - 1) / 2
    angle = math.radians(float(degrees))
    rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    return (points - center) @ rotation + center


def panorama_rays(points, projection):
    if projection.get("encoding") != "hugin-unit-rays-v1":
        raise ValueError("Unknown saved rink projection")
    points = rotate_panorama(
        points, projection["image_size"], projection.get("rotation_degrees", 0)
    )
    return _interpolate(_decode(projection["data"]), points, projection["image_size"])


@lru_cache(maxsize=4)
def _ray_tree(data):
    from scipy.spatial import cKDTree

    return cKDTree(_decode(data).reshape(-1, 3))


def rays_to_panorama(rays, projection):
    rays = unit(np.asarray(rays).reshape(-1, 3))
    if not len(rays):
        return np.empty((0, 2))
    rotation = projection.get("rotation_degrees", 0)
    projection = {**projection, "rotation_degrees": 0}
    grid = _decode(projection["data"])
    size = np.asarray(projection["image_size"])
    _, indices = _ray_tree(projection["data"]).query(rays)
    points = (
        np.stack([indices % grid.shape[1], indices // grid.shape[1]], axis=-1)
        / [grid.shape[1] - 1, grid.shape[0] - 1]
        * size
        - 0.5
    )
    for _ in range(16):
        current = panorama_rays(points, projection)
        residual = current - rays
        if np.max(np.linalg.norm(residual, axis=1)) < 1e-7:
            return rotate_panorama(points, size, -rotation)
        jacobian = np.stack(
            [
                (panorama_rays(points + delta, projection) - current) / 0.01
                for delta in ([0.01, 0], [0, 0.01])
            ],
            axis=-1,
        )
        updates = np.array(
            [np.linalg.lstsq(j, r, rcond=None)[0] for j, r in zip(jacobian, residual)]
        )
        points -= np.clip(updates, -size / 4, size / 4)
    if np.max(np.linalg.norm(panorama_rays(points, projection) - rays, axis=1)) > 1e-5:
        raise ValueError("Rink point lies outside the calibrated panorama projection")
    return rotate_panorama(points, size, -rotation)
