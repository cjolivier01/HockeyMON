import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_import_stubs() -> None:
    hmlib = types.ModuleType("hmlib")
    hmlib.__path__ = []
    hmlib_config = types.ModuleType("hmlib.config")
    hmlib_config.get_game_dir = lambda game_id, assert_exists=True: None
    hmlib_stitching = types.ModuleType("hmlib.stitching")
    hmlib_stitching.__path__ = []
    hmlib_stitching_configure = types.ModuleType("hmlib.stitching.configure_stitching")
    hmlib_stitching_configure.MAPPING_BACKENDS = (
        "nona",
        "opencv-magsac",
        "opencv-affine-ransac",
    )
    hmlib_stitching_configure.OPENCV_MAPPING_BACKENDS = (
        "opencv-magsac",
        "opencv-affine-ransac",
    )
    hmlib_stitching_configure.get_enblend_bin = lambda: "enblend"
    hmlib_stitching_control_points = types.ModuleType("hmlib.stitching.control_points")
    hmlib_stitching_control_points.CONTROL_POINT_MATCHERS = (
        "superpoint-lightglue",
        "dedode-lightglue",
        "loftr",
    )
    hmlib_stitching_control_points.calculate_control_points = lambda *_args, **_kwargs: {}
    hmlib_stitching_homography_maps = types.ModuleType("hmlib.stitching.homography_maps")
    hmlib_stitching_homography_maps.create_opencv_affine_ransac_mapping_files = (
        lambda *_args, **_kwargs: []
    )
    hmlib_stitching_homography_maps.create_opencv_magsac_mapping_files = (
        lambda *_args, **_kwargs: []
    )

    lightglue = types.ModuleType("lightglue")
    lightglue.LightGlue = object
    lightglue.SuperPoint = object
    lightglue.viz2d = types.SimpleNamespace()
    lightglue_utils = types.ModuleType("lightglue.utils")
    lightglue_utils.rbd = lambda value: value

    scipy = types.ModuleType("scipy")
    scipy.__path__ = []
    scipy_signal = types.ModuleType("scipy.signal")
    scipy.signal = scipy_signal

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.device = object

    numpy = types.ModuleType("numpy")
    numpy.ndarray = object

    tifffile = types.ModuleType("tifffile")
    tifffile.TiffFile = object

    cv2 = types.ModuleType("cv2")
    cv2.imwrite = lambda *_args, **_kwargs: True

    for name, module in {
        "cv2": cv2,
        "ffmpegio": types.ModuleType("ffmpegio"),
        "hmlib": hmlib,
        "hmlib.config": hmlib_config,
        "hmlib.stitching": hmlib_stitching,
        "hmlib.stitching.configure_stitching": hmlib_stitching_configure,
        "hmlib.stitching.control_points": hmlib_stitching_control_points,
        "hmlib.stitching.homography_maps": hmlib_stitching_homography_maps,
        "kornia": types.ModuleType("kornia"),
        "lightglue": lightglue,
        "lightglue.utils": lightglue_utils,
        "numpy": numpy,
        "scipy": scipy,
        "scipy.signal": scipy_signal,
        "tifffile": tifffile,
        "torch": torch,
        "yaml": types.ModuleType("yaml"),
    }.items():
        sys.modules.setdefault(name, module)


def _load_create_control_points():
    _install_import_stubs()
    module_path = Path(__file__).resolve().parents[1] / "hmlib" / "cli" / "create_control_points.py"
    spec = importlib.util.spec_from_file_location("create_control_points_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CreateControlPointsScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.create_control_points = _load_create_control_points()

    def test_read_pto_canvas_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pto_file = Path(tmpdir) / "project.pto"
            pto_file.write_text("# hugin project\np f1 w12092 h9267 v360\n", encoding="utf-8")

            self.assertEqual(
                self.create_control_points._read_pto_canvas_size(str(pto_file)), (12092, 9267)
            )

    def test_opencv_mapping_backends_reject_hugin_scale(self) -> None:
        for backend in ("opencv-magsac", "opencv-affine-ransac"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, backend):
                    self.create_control_points.configure_stitching(
                        object(),
                        object(),
                        "/tmp/unused-stitch-test",
                        scale=0.5,
                        mapping_backend=backend,
                    )

    def test_configure_stitching_scales_pto_before_nona_and_retries_final_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            calls = []
            mapping_sizes = [(8195, 1000), (8180, 1000)]

            def fake_run_stitching_command(cmd):
                calls.append(list(cmd))
                if cmd[0] == "pto_gen":
                    Path(cmd[cmd.index("-o") + 1]).write_text(
                        "p f1 w12092 h9267\n", encoding="utf-8"
                    )
                elif cmd[0] == "autooptimiser":
                    Path(cmd[cmd.index("-o") + 1]).write_text(
                        "p f1 w12092 h9267\n", encoding="utf-8"
                    )
                elif cmd[0] == "nona":
                    (tmp_path / "mapping_0000.tif").write_text("mapping0", encoding="utf-8")
                    (tmp_path / "mapping_0001.tif").write_text("mapping1", encoding="utf-8")
                elif cmd[0] == "enblend":
                    return
                else:
                    raise AssertionError(f"unexpected command: {cmd}")

            with (
                mock.patch.object(
                    self.create_control_points.cv2, "imwrite", lambda *_args, **_kwargs: True
                ),
                mock.patch.object(
                    self.create_control_points,
                    "calculate_control_points",
                    lambda *_args, **_kwargs: {},
                ),
                mock.patch.object(
                    self.create_control_points, "update_pto_file", lambda *_args, **_kwargs: None
                ),
                mock.patch.object(self.create_control_points, "get_enblend_bin", lambda: "enblend"),
                mock.patch.object(
                    self.create_control_points, "_run_stitching_command", fake_run_stitching_command
                ),
                mock.patch.object(
                    self.create_control_points,
                    "_read_mapping_canvas_size",
                    lambda _files: mapping_sizes.pop(0),
                ),
            ):
                self.assertTrue(
                    self.create_control_points.configure_stitching(
                        object(),
                        object(),
                        str(tmp_path),
                        force=True,
                        max_output_dimension=8192,
                    )
                )

            self.assertEqual(
                [cmd[0] for cmd in calls],
                [
                    "pto_gen",
                    "autooptimiser",
                    "autooptimiser",
                    "nona",
                    "autooptimiser",
                    "nona",
                    "enblend",
                ],
            )

            autooptimiser_commands = [cmd for cmd in calls if cmd[0] == "autooptimiser"]
            self.assertNotIn("-x", autooptimiser_commands[0])

            first_scale = float(
                autooptimiser_commands[1][autooptimiser_commands[1].index("-x") + 1]
            )
            retry_scale = float(
                autooptimiser_commands[2][autooptimiser_commands[2].index("-x") + 1]
            )
            self.assertAlmostEqual(first_scale, 8192 / 12092)
            self.assertLess(retry_scale, first_scale)

    def test_enblend_failure_removes_partial_seam(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def fake_run_stitching_command(cmd):
                if cmd[0] == "pto_gen":
                    Path(cmd[cmd.index("-o") + 1]).write_text("p f1 w100 h50\n", encoding="utf-8")
                elif cmd[0] == "autooptimiser":
                    Path(cmd[cmd.index("-o") + 1]).write_text("p f1 w100 h50\n", encoding="utf-8")
                elif cmd[0] == "nona":
                    (tmp_path / "mapping_0000.tif").write_text("mapping0", encoding="utf-8")
                    (tmp_path / "mapping_0001.tif").write_text("mapping1", encoding="utf-8")
                elif cmd[0] == "enblend":
                    (tmp_path / "seam_file.png").write_text("partial", encoding="utf-8")
                    raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
                else:
                    raise AssertionError(f"unexpected command: {cmd}")

            with (
                mock.patch.object(
                    self.create_control_points.cv2, "imwrite", lambda *_args, **_kwargs: True
                ),
                mock.patch.object(
                    self.create_control_points,
                    "calculate_control_points",
                    lambda *_args, **_kwargs: {},
                ),
                mock.patch.object(
                    self.create_control_points, "update_pto_file", lambda *_args, **_kwargs: None
                ),
                mock.patch.object(self.create_control_points, "get_enblend_bin", lambda: "enblend"),
                mock.patch.object(
                    self.create_control_points, "_run_stitching_command", fake_run_stitching_command
                ),
            ):
                self.assertTrue(
                    self.create_control_points.configure_stitching(
                        object(), object(), str(tmp_path), force=True
                    )
                )

            self.assertFalse((tmp_path / "seam_file.png").exists())


if __name__ == "__main__":
    unittest.main()
