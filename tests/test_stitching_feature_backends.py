from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
import torch

from hmlib.stitching import configure_stitching
from hmlib.stitching import control_points as control_points_module
from hmlib.stitching import homography_maps


def should_normalize_control_point_matcher_aliases() -> None:
    assert control_points_module.normalize_control_point_matcher("superpoint") == (
        "superpoint-lightglue"
    )
    assert control_points_module.normalize_control_point_matcher("DeDoDe") == ("dedode-lightglue")
    assert control_points_module.normalize_control_point_matcher("loftr") == "loftr"
    with pytest.raises(ValueError, match="Unsupported control-point matcher"):
        control_points_module.normalize_control_point_matcher("unknown")


def should_resize_dedode_inputs_to_3840_and_restore_original_coordinates() -> None:
    image = torch.empty((3, 4320, 7680), device="meta")
    resized, scale_x, scale_y = control_points_module._resize_for_matching(
        image,
        max_dimension=control_points_module._DEDODE_MAX_IMAGE_DIMENSION,
    )

    assert resized.shape == (3, 2160, 3840)
    assert scale_x == pytest.approx(2.0)
    assert scale_y == pytest.approx(2.0)
    resized_point = torch.tensor([1234.5, 678.25])
    original_point = resized_point * resized_point.new_tensor([scale_x, scale_y])
    torch.testing.assert_close(original_point, torch.tensor([2469.0, 1356.5]))


@pytest.mark.parametrize(
    ("matcher_name", "implementation_name"),
    [
        ("superpoint-lightglue", "_match_superpoint_lightglue"),
        ("dedode-lightglue", "_match_dedode_lightglue"),
        ("loftr", "_match_loftr"),
    ],
)
def should_route_control_point_matchers_without_duplicate_sampling(
    monkeypatch: pytest.MonkeyPatch,
    matcher_name: str,
    implementation_name: str,
) -> None:
    calls: list[str] = []
    points0 = torch.tensor([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    points1 = points0 + torch.tensor([5.0, 2.0])

    def fake_matcher(*_args, **_kwargs):
        calls.append(implementation_name)
        return points0, points1

    monkeypatch.setattr(control_points_module, implementation_name, fake_matcher)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    result = control_points_module.calculate_control_points(
        image,
        image,
        max_control_points=20,
        matcher=matcher_name,
        device=torch.device("cpu"),
    )

    assert calls == [implementation_name]
    assert result["m_kpts0"].shape == (5, 2)
    assert torch.unique(result["m_kpts0"], dim=0).shape[0] == 5


def should_write_complete_opencv_mapping_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left = np.zeros((3, 4, 3), dtype=np.uint8)
    left[:, :, 2] = 255
    right = np.zeros((3, 4, 3), dtype=np.uint8)
    right[:, :, 1] = 255
    left_file = tmp_path / "left.png"
    right_file = tmp_path / "right.png"
    assert cv2.imwrite(str(left_file), left)
    assert cv2.imwrite(str(right_file), right)

    x_map = np.tile(np.arange(4, dtype=np.uint16), (3, 1))
    y_map = np.tile(np.arange(3, dtype=np.uint16)[:, None], (1, 4))

    def fake_native(*_args, **_kwargs):
        return {
            "canvas_width": 6,
            "canvas_height": 4,
            "image_maps": [
                {"x_position": 0, "y_position": 0, "x_map": x_map, "y_map": y_map},
                {"x_position": 2, "y_position": 1, "x_map": x_map, "y_map": y_map},
            ],
        }

    monkeypatch.setattr(homography_maps, "_native_create_homography_maps", fake_native)
    control_points = {
        "m_kpts0": torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32),
        "m_kpts1": torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32),
    }
    mapping_files = homography_maps.create_opencv_magsac_mapping_files(
        [str(left_file), str(right_file)], control_points, tmp_path
    )

    assert [Path(path).name for path in mapping_files] == [
        "mapping_0000.tif",
        "mapping_0001.tif",
    ]
    for index in range(2):
        assert (tmp_path / f"mapping_{index:04d}.tif").is_file()
        assert (tmp_path / f"mapping_{index:04d}_x.tif").is_file()
        assert (tmp_path / f"mapping_{index:04d}_y.tif").is_file()
    assert configure_stitching.get_image_geo_position(mapping_files[0]) == (0, 0)
    assert configure_stitching.get_image_geo_position(mapping_files[1]) == (2, 1)
    with tifffile.TiffFile(mapping_files[1]) as tif:
        assert tif.pages[0].tags[33300].value == 6
        assert tif.pages[0].tags[33301].value == 4
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "mapping_0001_x.tif"), x_map)
    assert tifffile.imread(mapping_files[0]).shape == (3, 4, 4)


def should_use_native_mapping_backend_in_project_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left_file = tmp_path / "left.png"
    right_file = tmp_path / "right.png"
    left_file.touch()
    right_file.touch()
    project_file = tmp_path / "hm_project.pto"
    commands: list[list[str]] = []
    captured: dict[str, str] = {}
    points = torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32)

    def fake_command(command: list[str]) -> None:
        commands.append(command)
        if command[0] == "pto_gen":
            project_file.write_text("# hugin project\n# control points\n", encoding="utf-8")
        elif command[0] == "enblend":
            (tmp_path / "seam_file.png").touch()

    def fake_control_points(*_args, matcher: str, **_kwargs):
        captured["matcher"] = matcher
        return {"m_kpts0": points, "m_kpts1": points}

    def fake_mapping_files(*_args, **_kwargs):
        captured["mapping_backend"] = "opencv-magsac"
        outputs = [tmp_path / "mapping_0000.tif", tmp_path / "mapping_0001.tif"]
        for output in outputs:
            output.touch()
            output.with_name(f"{output.stem}_x.tif").touch()
            output.with_name(f"{output.stem}_y.tif").touch()
        return [str(output) for output in outputs]

    monkeypatch.setattr(configure_stitching, "_run_stitching_command", fake_command)
    monkeypatch.setattr(configure_stitching, "configure_control_points", fake_control_points)
    monkeypatch.setattr(
        configure_stitching,
        "create_opencv_magsac_mapping_files",
        fake_mapping_files,
    )
    monkeypatch.setattr(
        configure_stitching,
        "get_pixel_value_percentages",
        lambda _path: {0: 50.0, 255: 50.0},
    )
    monkeypatch.setattr(configure_stitching, "get_enblend_bin", lambda: "enblend")

    assert configure_stitching.build_stitching_project(
        str(project_file),
        [str(left_file), str(right_file)],
        max_control_points=20,
        skip_if_exists=False,
        control_point_matcher="loftr",
        mapping_backend="opencv-magsac",
    )
    assert captured == {"matcher": "loftr", "mapping_backend": "opencv-magsac"}
    assert [command[0] for command in commands] == ["pto_gen", "enblend"]
    autooptimiser_file = tmp_path / "autooptimiser_out.pto"
    assert autooptimiser_file.is_file()
    assert configure_stitching._stitch_project_is_complete(
        project_file,
        autooptimiser_file,
        control_point_matcher="loftr",
        mapping_backend="opencv-magsac",
    )
    assert not configure_stitching._stitch_project_is_complete(
        project_file,
        autooptimiser_file,
        control_point_matcher="superpoint-lightglue",
        mapping_backend="nona",
    )


@pytest.mark.parametrize(
    ("requested_matcher", "expected_use_hugin"),
    [
        ("dedode-lightglue", True),
        ("loftr", False),
    ],
)
def should_reuse_points_only_when_matcher_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requested_matcher: str,
    expected_use_hugin: bool,
) -> None:
    left_file = tmp_path / "left.png"
    right_file = tmp_path / "right.png"
    left_file.touch()
    right_file.touch()
    project_file = tmp_path / "hm_project.pto"
    project_file.write_text("# hugin project\n# control points\n", encoding="utf-8")
    (tmp_path / ".stitching_artifacts.json").write_text(
        '{"control_point_matcher": "dedode-lightglue", "mapping_backend": "nona"}\n',
        encoding="utf-8",
    )
    points = torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32)
    captured: dict[str, object] = {}

    def fake_control_points(*_args, use_hugin: bool, matcher: str, **_kwargs):
        captured["use_hugin"] = use_hugin
        captured["matcher"] = matcher
        return {"m_kpts0": points, "m_kpts1": points}

    def fake_mapping_files(*_args, **_kwargs):
        outputs = [tmp_path / "mapping_0000.tif", tmp_path / "mapping_0001.tif"]
        for output in outputs:
            output.touch()
            output.with_name(f"{output.stem}_x.tif").touch()
            output.with_name(f"{output.stem}_y.tif").touch()
        return [str(output) for output in outputs]

    def fake_command(command: list[str]) -> None:
        if command[0] == "enblend":
            (tmp_path / "seam_file.png").touch()

    monkeypatch.setattr(configure_stitching, "configure_control_points", fake_control_points)
    monkeypatch.setattr(
        configure_stitching,
        "create_opencv_magsac_mapping_files",
        fake_mapping_files,
    )
    monkeypatch.setattr(configure_stitching, "_run_stitching_command", fake_command)
    monkeypatch.setattr(
        configure_stitching,
        "get_pixel_value_percentages",
        lambda _path: {0: 50.0, 255: 50.0},
    )
    monkeypatch.setattr(configure_stitching, "get_enblend_bin", lambda: "enblend")

    assert configure_stitching.build_stitching_project(
        str(project_file),
        [str(left_file), str(right_file)],
        max_control_points=20,
        skip_if_exists=False,
        control_point_matcher=requested_matcher,
        mapping_backend="opencv-magsac",
    )
    assert captured == {
        "use_hugin": expected_use_hugin,
        "matcher": requested_matcher,
    }
