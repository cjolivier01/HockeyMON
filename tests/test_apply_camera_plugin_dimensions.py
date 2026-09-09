from __future__ import annotations

import sys
import types

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]


requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


def _install_mmcv_transforms_stub(monkeypatch) -> None:
    transforms_module = types.ModuleType("mmcv.transforms")

    class _Compose:
        def __init__(self, pipeline):
            self._pipeline = pipeline or []

        def __iter__(self):
            return iter(())

        def __call__(self, data):
            return data

    transforms_module.Compose = _Compose

    mmcv_module = sys.modules.get("mmcv")
    if mmcv_module is None:
        mmcv_module = types.ModuleType("mmcv")
    monkeypatch.setattr(mmcv_module, "transforms", transforms_module, raising=False)
    monkeypatch.setitem(sys.modules, "mmcv", mmcv_module)
    monkeypatch.setitem(sys.modules, "mmcv.transforms", transforms_module)


@requires_torch
def should_report_actual_frame_size_when_full_panorama_is_not_cropped(monkeypatch):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin
    from hmlib.utils.gpu import unwrap_tensor

    plugin = ApplyCameraPlugin(
        video_out_pipeline=None,
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)

    out = plugin({"img": img, "shared": {}})

    out_img = unwrap_tensor(out["img"])
    assert tuple(out_img.shape) == (1, 9, 15, 3)
    assert out["video_frame_cfg"]["output_frame_width"] == 15
    assert out["video_frame_cfg"]["output_frame_height"] == 9


@requires_torch
def should_clamp_apply_camera_output_width_from_game_config(monkeypatch):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin

    plugin = ApplyCameraPlugin(
        video_out_pipeline=None,
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)
    context = {"img": img, "shared": {"game_config": {"video_out": {"output_width": 10}}}}

    plugin._ensure_initialized(context)

    assert plugin._video_frame_cfg is not None
    assert plugin._video_frame_cfg["output_frame_width"] == 10
    assert plugin._video_frame_cfg["output_frame_height"] == 6


@requires_torch
def should_clamp_apply_camera_output_height_from_game_config(monkeypatch):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin

    plugin = ApplyCameraPlugin(
        video_out_pipeline=None,
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)
    context = {"img": img, "shared": {"game_config": {"video_out": {"output_height": 6}}}}

    plugin._ensure_initialized(context)

    assert plugin._video_frame_cfg is not None
    assert plugin._video_frame_cfg["output_frame_width"] == 10
    assert plugin._video_frame_cfg["output_frame_height"] == 6


@requires_torch
@pytest.mark.parametrize(
    "rotation,expected", [(None, True), ([0, 0, 0], False), ([12, 0, 0], False)]
)
def should_refresh_fixed_edge_rotation_angle_from_runtime_config(monkeypatch, rotation, expected):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera import apply_camera_plugin as apply_camera_module

    class HmPerspectiveRotation:
        def __init__(self) -> None:
            self.values = []

        def set_fixed_edge_rotation_angle(self, value) -> None:
            self.values.append(value)

        def set_camera_space_leveling(self, enabled) -> None:
            self.leveling = enabled

    perspective = HmPerspectiveRotation()

    class RuntimeCompose:
        def __init__(self, _pipeline) -> None:
            self.transforms = [perspective]

        def __iter__(self):
            return iter(self.transforms)

        def __call__(self, data):
            return data

    monkeypatch.setattr(apply_camera_module, "Compose", RuntimeCompose)
    game_config = {"rink": {"camera": {"fixed_edge_rotation_angle": 12.5}}}
    game_config["stitching"] = {
        "mapping_backend": "nona",
        "run_autooptimizer": True,
        "projection": "equirectangular",
        "rink_config": "rink",
        "rink_configs": {"rink": {"rotation_degrees": [0, -20, 5]}},
        "projection_framing": {"rotation_degrees": rotation},
    }
    plugin = apply_camera_module.ApplyCameraPlugin(
        video_out_pipeline=[{"type": "HmPerspectiveRotation"}],
        crop_output_image=False,
    )
    context = {
        "img": torch.zeros((1, 9, 15, 3), dtype=torch.uint8),
        "shared": {"game_config": game_config},
    }

    plugin(context)
    game_config["rink"]["camera"]["fixed_edge_rotation_angle"] = [15.0, 35.0]
    plugin(context)

    assert perspective.values == [12.5, [15.0, 35.0]]
    assert perspective.leveling is expected


@requires_torch
def should_suppress_and_restore_program_rotation_without_discarding_angles(monkeypatch):
    from hmlib.transforms import perspective_rotation as module

    calls = []

    def rotate_image(**kwargs):
        calls.append(kwargs["angle"])
        return kwargs["img"]

    monkeypatch.setattr(module, "rotate_image", rotate_image)
    transform = module.HmPerspectiveRotation(fixed_edge_rotation_angle=[12, 24])
    monkeypatch.setattr(
        transform,
        "_get_gaussian",
        lambda _: types.SimpleNamespace(get_gaussian_y_from_image_x_position=lambda *a, **kw: 1),
    )
    data = {
        "img": torch.zeros((1, 10, 20, 3)),
        "camera_box": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
    }
    transform.set_camera_space_leveling(True)
    assert transform(data) is data
    assert calls == []
    transform.set_fixed_edge_rotation_angle([15, 30])
    transform.set_camera_space_leveling(False)
    transform(data)
    assert calls == [-15]
    assert transform._fixed_edge_rotation_angle == (15, 30)
