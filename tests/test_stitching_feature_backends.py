from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import kornia.feature as kornia_feature
import numpy as np
import pytest
import tifffile
import torch
from stitching_fixtures import write_mapping_files, write_seam

from hmlib.stitching import configure_stitching, homography_maps
from hmlib.stitching import control_points as control_points_module
from hmlib.stitching import superpoint as superpoint_module


def should_normalize_control_point_matcher_aliases() -> None:
    assert control_points_module.normalize_control_point_matcher("superpoint") == (
        "superpoint-lightglue"
    )
    assert control_points_module.normalize_control_point_matcher("DeDoDe") == ("dedode-lightglue")
    assert control_points_module.normalize_control_point_matcher("loftr") == "loftr"
    with pytest.raises(ValueError, match="Unsupported control-point matcher"):
        control_points_module.normalize_control_point_matcher("unknown")


def should_normalize_mapping_backend_and_dimension() -> None:
    assert configure_stitching.normalize_mapping_backend("OpenCV_Affine_RANSAC") == (
        "opencv-affine-ransac"
    )
    assert configure_stitching.normalize_max_output_dimension("4096") == 4096
    with pytest.raises(ValueError, match="Unsupported mapping backend"):
        configure_stitching.normalize_mapping_backend("unknown")
    with pytest.raises(ValueError, match="max_output_dimension"):
        configure_stitching.normalize_max_output_dimension(65535)


def should_resize_dedode_inputs_to_1920_and_restore_original_coordinates() -> None:
    image = torch.empty((3, 4320, 7680), device="meta")
    resized, scale_x, scale_y = control_points_module._resize_for_matching(
        image,
        max_dimension=control_points_module._DEDODE_MAX_IMAGE_DIMENSION,
    )

    assert resized.shape == (3, 1080, 1920)
    assert scale_x == pytest.approx(4.0)
    assert scale_y == pytest.approx(4.0)
    resized_point = torch.tensor([1234.5, 678.25])
    original_point = resized_point * resized_point.new_tensor([scale_x, scale_y])
    torch.testing.assert_close(original_point, torch.tensor([4938.0, 2713.0]))


def _superpoint_without_weights(
    monkeypatch: pytest.MonkeyPatch, **conf: object
) -> superpoint_module.SuperPoint:
    """Build a SuperPoint whose layers are random, so no checkpoint is fetched."""
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(superpoint_module.SuperPoint, "load_state_dict", lambda *_a, **_k: None)
    return superpoint_module.SuperPoint(**conf)


def should_resize_superpoint_inputs_to_the_long_edge_and_report_source_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = _superpoint_without_weights(monkeypatch)
    seen: list[tuple[int, int]] = []

    def fake_forward(data: dict) -> dict:
        height, width = data["image"].shape[-2:]
        seen.append((height, width))
        corners = torch.tensor([[0.0, 0.0], [width - 1.0, height - 1.0]])
        return {
            "keypoints": corners[None],
            "keypoint_scores": torch.ones((1, 2)),
            "descriptors": torch.zeros((1, 2, 256)),
        }

    monkeypatch.setattr(extractor, "forward", fake_forward)
    # 1920x1080 keeps the same 1.875 downscale as a 4K frame without the 95MB
    # allocation and full-resolution antialias blur that one would cost in CI.
    feats = extractor.extract(torch.zeros((3, 1080, 1920)))

    # SuperPoint sees a 1024-long-edge image regardless of the source resolution.
    assert seen == [(576, 1024)]
    # image_size is (w, h) -- kornia's normalize_keypoints reads it in that order.
    assert feats["image_size"].tolist() == [[1920.0, 1080.0]]
    # Corners of the downsampled image map back onto corners of the source frame.
    torch.testing.assert_close(
        feats["keypoints"],
        torch.tensor([[[0.4375, 0.4375], [1918.5625, 1078.5625]]]),
    )


def should_reject_images_superpoint_cannot_safely_upscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = _superpoint_without_weights(monkeypatch)

    with pytest.raises(ValueError, match="long edge"):
        extractor.extract(torch.zeros((3, 60, 100)))
    with pytest.raises(TypeError, match="float image"):
        extractor.extract(torch.zeros((3, 1080, 1920), dtype=torch.uint8))


def should_hand_superpoint_features_to_kornia_lightglue_unmodified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the real extract() -> kornia contract that both other tests stub out.

    ``features=None`` skips the checkpoint download, so this needs no network;
    kornia still validates key names, the (w, h) image_size order, descriptor
    shape and the normalized-keypoint range the way the real matcher does.
    """
    extractor = _superpoint_without_weights(monkeypatch, max_num_keypoints=64)
    matcher = kornia_feature.LightGlue(features=None).eval()

    rng = torch.Generator().manual_seed(0)
    image = torch.rand((3, 540, 960), generator=rng)
    feats0 = extractor.extract(image)
    feats1 = extractor.extract(torch.rot90(image, k=2, dims=(1, 2)).contiguous())

    count0 = feats0["keypoints"].shape[1]
    assert feats0["descriptors"].shape == (1, count0, 256)
    assert feats0["keypoints"].shape == (1, count0, 2)
    matches = matcher({"image0": feats0, "image1": feats1})["matches"][0]
    assert matches.ndim == 2 and matches.shape[-1] == 2
    assert bool((matches[:, 0] < count0).all())
    assert bool((matches[:, 1] < feats1["keypoints"].shape[1]).all())


class _FakeModule:
    """Absorbs the .eval().to(device) chain that the matcher applies to both models."""

    def eval(self) -> _FakeModule:
        return self

    def to(self, _device: torch.device) -> _FakeModule:
        return self


class _FakeSuperPoint(_FakeModule):
    """Keys its reply off the image it is given, so a swapped call site shows up."""

    def __init__(self, features_by_marker: dict[float, dict]) -> None:
        self._features_by_marker = features_by_marker

    def extract(self, image: torch.Tensor, *, resolution="2k") -> dict:
        return self._features_by_marker[float(image.flatten()[0])]


class _FakeLightGlue(_FakeModule):
    """Returns LightGlue's output shape: one [Si x 2] index tensor per batch element."""

    def __init__(self, matches: torch.Tensor) -> None:
        self._matches = matches
        self.seen: dict = {}

    def __call__(self, data: dict) -> dict:
        self.seen = data
        return {"matches": [self._matches]}


def should_pair_superpoint_keypoints_using_lightglue_match_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypoints0 = torch.tensor([[0.0, 0.0], [10.0, 1.0], [20.0, 2.0]])
    keypoints1 = torch.tensor([[30.0, 3.0], [40.0, 4.0], [50.0, 5.0]])
    feats0 = {"keypoints": keypoints0[None]}
    feats1 = {"keypoints": keypoints1[None]}
    # Each image carries a distinct marker value, so extracting the same one twice
    # or filling the matcher's slots in the wrong order fails the assertions below.
    image0 = torch.full((3, 8, 8), 1.0)
    image1 = torch.full((3, 8, 8), 2.0)
    extractor = _FakeSuperPoint({1.0: feats0, 2.0: feats1})
    matcher = _FakeLightGlue(torch.tensor([[2, 0], [0, 1]]))

    monkeypatch.setattr(superpoint_module, "SuperPoint", lambda **_kwargs: extractor)
    monkeypatch.setattr(kornia_feature, "LightGlue", lambda **_kwargs: matcher)

    points0, points1 = control_points_module._match_superpoint_lightglue(
        image0, image1, torch.device("cpu"), 128
    )

    assert matcher.seen["image0"] is feats0
    assert matcher.seen["image1"] is feats1
    # Column 0 indexes image0's keypoints and column 1 image1's; neither the batch
    # dim nor the column order may be transposed.
    torch.testing.assert_close(points0, torch.tensor([[20.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(points1, torch.tensor([[30.0, 3.0], [40.0, 4.0]]))


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
        control_point_filtering={"epipolar_ransac": False, "homography_consistency": False},
        device=torch.device("cpu"),
    )

    assert calls == [implementation_name]
    assert result["m_kpts0"].shape == (5, 2)
    assert torch.unique(result["m_kpts0"], dim=0).shape[0] == 5


def should_reject_too_few_requested_control_points() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least four"):
        control_points_module.calculate_control_points(
            image,
            image,
            max_control_points=3,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("native_name", "builder_name"),
    [
        ("_native_create_homography_maps", "create_opencv_magsac_mapping_files"),
        (
            "_native_create_affine_ransac_maps",
            "create_opencv_affine_ransac_mapping_files",
        ),
    ],
)
def should_write_complete_opencv_mapping_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_name: str,
    builder_name: str,
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

    monkeypatch.setattr(homography_maps, native_name, fake_native)
    control_points = {
        "m_kpts0": torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32),
        "m_kpts1": torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32),
    }
    mapping_files = getattr(homography_maps, builder_name)(
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


@pytest.mark.parametrize("maximum_dimension", [-1, 0, 65535])
def should_reject_invalid_opencv_mapping_dimension(tmp_path: Path, maximum_dimension: int) -> None:
    with pytest.raises(ValueError, match="max_output_dimension"):
        homography_maps.create_opencv_magsac_mapping_files(
            ["left.png", "right.png"],
            {},
            tmp_path,
            max_output_dimension=maximum_dimension,
        )


@pytest.mark.parametrize(
    ("mapping_backend", "mapping_builder_name"),
    [
        ("opencv-magsac", "create_opencv_magsac_mapping_files"),
        ("opencv-affine-ransac", "create_opencv_affine_ransac_mapping_files"),
    ],
)
def should_use_native_mapping_backend_in_project_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mapping_backend: str,
    mapping_builder_name: str,
) -> None:
    left_file = tmp_path / "left.png"
    right_file = tmp_path / "right.png"
    assert cv2.imwrite(str(left_file), np.zeros((3, 4, 3), np.uint8))
    assert cv2.imwrite(str(right_file), np.zeros((3, 4, 3), np.uint8))
    project_file = tmp_path / "hm_project.pto"
    commands: list[list[str]] = []
    captured: dict[str, str] = {}
    points = torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32)

    def fake_command(command: list[str]) -> None:
        commands.append(command)
        if command[0] == "pto_gen":
            Path(command[command.index("-o") + 1]).write_text(
                "# hugin project\n# control points\n", encoding="utf-8"
            )
        elif command[0] == "enblend":
            write_seam(Path(command[command.index("-o") + 1]).parent)

    def fake_control_points(*_args, matcher: str, **_kwargs):
        captured["matcher"] = matcher
        return {"m_kpts0": points, "m_kpts1": points}

    def fake_mapping_files(*_args, **_kwargs):
        captured["mapping_backend"] = mapping_backend
        return write_mapping_files(_args[2])

    monkeypatch.setattr(configure_stitching, "_run_stitching_command", fake_command)
    monkeypatch.setattr(configure_stitching, "configure_control_points", fake_control_points)
    monkeypatch.setattr(
        configure_stitching,
        mapping_builder_name,
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
        mapping_backend=mapping_backend,
        max_output_dimension=None,
    )
    assert captured == {"matcher": "loftr", "mapping_backend": mapping_backend}
    assert [command[0] for command in commands] == ["pto_gen", "enblend"]
    autooptimiser_file = tmp_path / "autooptimiser_out.pto"
    assert autooptimiser_file.is_file()
    assert configure_stitching._stitch_project_is_complete(
        project_file,
        autooptimiser_file,
        control_point_matcher="loftr",
        mapping_backend=mapping_backend,
    )
    assert not configure_stitching._stitch_project_is_complete(
        project_file,
        autooptimiser_file,
        control_point_matcher="superpoint-lightglue",
        mapping_backend="nona",
        max_output_dimension=None,
    )
    assert not configure_stitching._stitch_project_is_complete(
        project_file,
        autooptimiser_file,
        control_point_matcher="loftr",
        mapping_backend=mapping_backend,
        max_output_dimension=2048,
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
    assert cv2.imwrite(str(left_file), np.zeros((3, 4, 3), np.uint8))
    assert cv2.imwrite(str(right_file), np.zeros((3, 4, 3), np.uint8))
    project_file = tmp_path / "hm_project.pto"
    project_file.write_text("# hugin project\n# control points\n", encoding="utf-8")
    previous_settings = replace(
        configure_stitching.read_stitching_settings(
            control_point_matcher="dedode-lightglue", mapping_backend="opencv-magsac"
        ),
        max_control_points=20,
    )
    (tmp_path / ".stitching_artifacts.json").write_text(
        json.dumps(
            {
                **previous_settings.manifest(),
                "input_images": configure_stitching._image_content_provenance(
                    [left_file, right_file]
                ),
                "output_scale": "1",
            }
        ),
        encoding="utf-8",
    )
    points = torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32)
    captured: dict[str, object] = {}

    def fake_control_points(*_args, use_hugin: bool, matcher: str, **_kwargs):
        captured["use_hugin"] = use_hugin
        captured["matcher"] = matcher
        return {"m_kpts0": points, "m_kpts1": points}

    def fake_mapping_files(*_args, **_kwargs):
        return write_mapping_files(_args[2])

    def fake_command(command: list[str]) -> None:
        if command[0] == "pto_gen":
            Path(command[command.index("-o") + 1]).write_text(
                "# hugin project\n# control points\n", encoding="utf-8"
            )
        if command[0] == "enblend":
            write_seam(Path(command[command.index("-o") + 1]).parent)

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
