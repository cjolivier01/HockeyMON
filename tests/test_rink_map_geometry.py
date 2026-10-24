"""Projection regressions use actual Hugin, independent metric ground truth."""

import base64
import io
import math

import numpy as np
import pytest

from hmlib.stitching.rink_map import (
    DEFAULT_LAYOUT,
    feature_line,
    feature_point,
    fit_rink_map,
    rink_leveling,
    unit,
)
from hmlib.stitching.rink_projection import (
    build_panorama_projection,
    panorama_rays,
    source_rays,
    transform_points,
)
from hmlib.ui.calibration_workspace import project_rink_points, unproject_rink_points


def _ray(point):
    x, y = point
    return unit([y + DEFAULT_LAYOUT["width"] / 2 + 20, -x, -10])


def _observation(feature, *, line=False):
    if line:
        physical = feature_line(feature, DEFAULT_LAYOUT)
        if physical[0]:
            world = [[-physical[2], -8], [-physical[2], 8]]
        else:
            world = [[-15, -physical[2]], [15, -physical[2]]]
    else:
        world = [feature_point(feature, DEFAULT_LAYOUT)]
    return {
        "feature": feature,
        "line": line,
        "camera": 0,
        "pixels": [[10 + i * 30, 20] for i in range(len(world))],
        "rays": [_ray(p).tolist() for p in world],
    }


def _corner_marks():
    return [
        _observation(feature)
        for feature in ("goal-left-near", "goal-left-far", "goal-right-near", "goal-right-far")
    ]


def should_fit_metric_rays_and_reject_parallel_lines_even_with_noise():
    observations = _corner_marks()
    fit = fit_rink_map(observations, DEFAULT_LAYOUT)
    assert fit["state"] == "ready"
    assert fit["camera_height"] == pytest.approx(10, abs=1e-9)
    matrix = np.asarray(fit["plane_to_ray"])
    np.testing.assert_allclose(unit(matrix @ [0, 0, 1]), _ray([0, 0]), atol=1e-12)
    for feature in ("center-line", "blue-left-line", "goal-right-line"):
        observations.append(_observation(feature, line=True))
    assert fit_rink_map(observations, DEFAULT_LAYOUT)["state"] == "ready"
    parallel = [
        _observation(feature, line=True)
        for feature in (
            "center-line",
            "blue-left-line",
            "goal-right-line",
            "blue-right-line",
            "goal-left-line",
        )
    ]
    rng = np.random.default_rng(124)
    for mark in parallel:
        mark["rays"] = unit(np.asarray(mark["rays"]) + rng.normal(0, 0.0001, (2, 3))).tolist()
    assert fit_rink_map(parallel, DEFAULT_LAYOUT)["state"] == "partial"
    complete_lines = [
        _observation(feature, line=True)
        for feature in ("goal-left-line", "goal-right-line", "near-boards", "far-boards")
    ]
    assert fit_rink_map(complete_lines, DEFAULT_LAYOUT)["state"] == "ready"


def should_reject_nonmetric_warp_despite_exact_correspondences():
    observations = _corner_marks()
    for mark in observations:
        point = feature_point(mark["feature"], DEFAULT_LAYOUT)
        point[0] *= 2
        mark["rays"] = [_ray(point).tolist()]
    assert fit_rink_map(observations, DEFAULT_LAYOUT)["state"] == "inconsistent"


def should_level_from_saved_plane_or_identified_corners_and_reject_arbitrary_segments():
    marks = _corner_marks()
    fit = fit_rink_map(marks, DEFAULT_LAYOUT)
    for state in (fit, {"state": "partial"}):
        estimated = rink_leveling(marks, state, DEFAULT_LAYOUT, [0, 0, 0], 23)
        np.testing.assert_allclose(estimated["rotation_degrees"], [23, 0, 0], atol=1e-10)
    with pytest.raises(ValueError, match="Arbitrary"):
        rink_leveling(
            [_observation("goal-left-line", line=True), _observation("goal-right-line", line=True)],
            {"state": "partial"},
            DEFAULT_LAYOUT,
            [0, 0, 0],
            0,
        )


def should_map_cylindrical_centre_correctly_where_pixel_homography_misses_by_17_metres(tmp_path):
    import cv2

    project = tmp_path / "cylinder.pto"
    project.write_text(
        f'p f1 w2000 h1000 v{math.degrees(4):.12g} S0,2000,400,1000\ni f0 w2000 h1000 v100 y0 p0 r0 n"camera.png"\n'
    )
    projection = build_panorama_projection(project, [2000, 600])
    corners = np.array([[0, 0], [60.96, 0], [60.96, 25.908], [0, 25.908]])

    def cylinder(points):
        x, y = np.asarray(points).T
        return np.stack(
            [
                999.5 + 500 * np.arctan2(x - 30.48, y + 20),
                99.5 + 5000 / np.sqrt((x - 30.48) ** 2 + (y + 20) ** 2),
            ],
            axis=-1,
        )

    pixels = cylinder(corners)
    old, _ = cv2.findHomography(pixels, corners)
    center = cylinder([[30.48, 12.954]])
    wrong = cv2.perspectiveTransform(center[None], old)[0, 0]
    assert abs(wrong[1] - 12.954) > 17
    observations = [
        {
            "feature": "custom",
            "line": False,
            "pixels": [pixel.tolist()],
            "rink": point.tolist(),
            "rays": [ray.tolist()],
        }
        for point, pixel, ray in zip(corners, pixels, panorama_rays(pixels, projection))
    ]
    fit = fit_rink_map(observations, DEFAULT_LAYOUT)
    assert fit["state"] == "ready", fit
    mapping = {
        "version": 2,
        "fit": fit,
        "projection": projection,
        "image_size": [2000, 600],
        "rink_size": [60.96, 25.908],
    }
    np.testing.assert_allclose(
        project_rink_points(center.tolist(), mapping), [[30.48, 12.954]], atol=0.002
    )
    np.testing.assert_allclose(unproject_rink_points([[30.48, 12.954]], mapping), center, atol=0.02)
    np.testing.assert_allclose(
        project_rink_points(((center + 0.5) / 2 - 0.5).tolist(), mapping, image_size=[1000, 300]),
        [[30.48, 12.954]],
        atol=0.002,
    )


@pytest.mark.parametrize("projection", [1, 2, 19])
def should_roundtrip_actual_nona_camera_lens_and_panorama_projection(tmp_path, projection):
    pto = f'p f{projection} w1600 h800 v140\ni f0 w1200 h800 v105 y-25 p-5 r3 a0.002 b-0.008 c0 d0 e0 n"left.png"\ni f0 w1200 h800 v105 y25 p-5 r3 a0.002 b-0.008 c0 d0 e0 n"right.png"\n'
    project = tmp_path / "project.pto"
    project.write_text(pto)
    observations = [
        {"camera": camera, "pixels": [[400, 500], [600, 400], [800, 550]]} for camera in (0, 1)
    ]
    resolved = source_rays(pto, observations)
    source = [[mark["camera"], *pixel] for mark in observations for pixel in mark["pixels"]]
    panorama = transform_points(pto, source)
    lookup = build_panorama_projection(project, [1600, 800])
    expected = np.concatenate([mark["rays"] for mark in resolved])
    np.testing.assert_allclose(panorama_rays(panorama, lookup), expected, atol=4e-5)


def planar_test_mapping():
    """A real pinhole camera looking straight down at a rectangular test rink."""
    width, height = 1000, 500
    xx, yy = np.meshgrid(np.linspace(-0.5, width - 0.5, 129), np.linspace(-0.5, height - 0.5, 65))
    rays = unit(
        np.stack([xx * 0.06096 - 30.48, yy * 0.051816 - 12.954, np.full(xx.shape, -40)], axis=-1)
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, rays=rays)
    return {
        "version": 2,
        "image_size": [width, height],
        "rink_size": [60.96, 25.908],
        "goals": [],
        "fit": {"state": "ready", "plane_to_ray": [[1, 0, 0], [0, 1, 0], [0, 0, -40]]},
        "projection": {
            "encoding": "hugin-unit-rays-v1",
            "image_size": [width, height],
            "data": base64.b64encode(stream.getvalue()).decode(),
        },
    }


def should_preserve_rotation_and_pixel_centres_in_runtime_projection():
    from hmlib.stitching.rink_projection import rotate_panorama

    mapping = planar_test_mapping()
    point = [400.25, 200.75]
    expected = project_rink_points([point], mapping)
    mapping["projection"]["rotation_degrees"] = 12
    rotated = rotate_panorama([point], mapping["image_size"], -12)
    np.testing.assert_allclose(project_rink_points(rotated.tolist(), mapping), expected, atol=1e-9)
    np.testing.assert_allclose(unproject_rink_points(expected, mapping), rotated, atol=1e-4)


def should_suggest_unchecked_far_line_and_three_handle_near_arc(tmp_path):
    import cv2

    from hmlib.stitching.rink_assistance import mask_boards

    pto = 'p f1 w1600 h900 v120\ni f0 w1600 h900 v120 y0 p0 r0 n"camera.png"\n'
    project = tmp_path / "autooptimiser_out.pto"
    project.write_text(pto)
    (tmp_path / ".stitching_artifacts.json").write_text("{}")
    mask = np.zeros((900, 1600), np.uint8)
    xs = np.arange(100, 1500)
    for x in xs:
        far = int(250 + (x - 800) ** 2 / 20000)
        near = int(740 - (x - 800) ** 2 / 6000)
        mask[far:near, x] = 255
    cv2.imwrite(str(tmp_path / "rink_mask_0.png"), mask)
    projection = build_panorama_projection(project, [1600, 900])
    marks = mask_boards(
        tmp_path,
        {"rink": {"ice_contours_mask_count": 1}},
        pto,
        [{"width": 1600, "height": 900}],
        projection,
    )
    assert {m["feature"] for m in marks} == {"near-boards", "far-boards"}
    assert all(not m["enabled"] and m["mask_revision"] for m in marks)
    near = next(m for m in marks if m.get("arc"))
    assert len(near["pixels"]) == 3
    project.write_text(pto)
    with pytest.raises(ValueError, match="predates"):
        mask_boards(tmp_path, {}, pto, [{"width": 1600, "height": 900}], projection)


def should_decode_pinned_landmark_outputs_without_assigning_blue_line_end():
    from hmlib.stitching.rink_assistance import postprocess

    classes = np.full((1, 100, 12), -10.0, np.float32)
    classes[:, :, -1] = 10
    masks = np.full((1, 100, 256, 256), -10.0, np.float32)
    for query, label in enumerate((0, 2, 7)):
        classes[0, query, -1] = -10
        classes[0, query, label] = 10
        masks[0, query, 20:220, 60 + 20 * query : 65 + 20 * query] = 10
    marks = postprocess(classes, masks, (1008, 800), (1920, 1524))
    assert len(marks) == 3
    assert {m["feature"] for m in marks} == {"", "center-line"}
    assert all(len(m["pixels"]) >= 2 and m["line"] for m in marks)


def should_keep_full_panini_canvas_when_even_resize_adds_one_row(tmp_path):
    from dataclasses import replace

    from hmlib.stitching.leveling_editor import run_hugin
    from hmlib.stitching.projections import cap_projection_canvas
    from hmlib.stitching.settings import read_stitching_settings

    project = tmp_path / "panini.pto"
    project.write_text('p f12 w1812 h2868 v177\ni f0 w1920 h1080 v100 y0 p0 r0 n"camera.png"\n')
    settings = replace(
        read_stitching_settings({}), max_output_dimension=1280, max_output_width=1280
    )
    result = cap_projection_canvas(project, settings, run_hugin)
    assert (result.width, result.height) == (808, 1280)
    assert result.crop == (0, 808, 0, 1280)


def should_skip_unlocatable_players_above_horizon_without_hiding_invalid_map():
    mapping = planar_test_mapping()
    mapping["fit"]["plane_to_ray"][2][2] = 40
    with pytest.raises(ValueError, match="intersect"):
        project_rink_points([[500, 250]], mapping)
    assert np.isnan(project_rink_points([[500, 250]], mapping, allow_missing=True)).all()


def should_apply_native_grid_bounds_and_rounded_rink_edges():
    mapping = planar_test_mapping()
    mapping.update(native=True, layout=dict(DEFAULT_LAYOUT))
    mapping["projection"]["native_bounds"] = True
    points = [[30.48, 12.954], [10, 12.954], [50, 12.954]]
    for angle in (0, -5, 5):
        mapping["projection"]["rotation_degrees"] = angle
        pixels = unproject_rink_points(points, mapping)
        np.testing.assert_allclose(project_rink_points(pixels, mapping), points, atol=0.002)
    outside = project_rink_points([[-50, 250], [0, 0]], mapping, allow_missing=True)
    assert np.isnan(outside).all()
    invisible = unproject_rink_points([[1000, 1000]], mapping)
    assert np.isnan(invisible).all()


def should_reject_native_projection_cells_with_missing_viewing_rays():
    mapping = planar_test_mapping()
    mapping["projection"]["native_bounds"] = True
    encoded = io.BytesIO(base64.b64decode(mapping["projection"]["data"]))
    rays = np.load(encoded)["rays"]
    rays[:2, :2] = 0
    replacement = io.BytesIO()
    np.savez_compressed(replacement, rays=rays)
    mapping["projection"]["data"] = base64.b64encode(replacement.getvalue()).decode()
    result = project_rink_points([[0, 0], [500, 250]], mapping, allow_missing=True)
    assert np.isnan(result[0]).all()
    assert np.isfinite(result[1]).all()
