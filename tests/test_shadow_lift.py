"""Reference tone-curve and pipeline checks for portable shadow grading."""

import numpy as np
import pytest
import torch

from hmlib.hm_transforms import HmImageColorAdjust


def _adjust(image, **kwargs):
    return HmImageColorAdjust(**kwargs)({"img": image})["img"]


@pytest.mark.parametrize("use_numpy", [False, True])
def should_match_reference_luma_curve_and_preserve_neutral_endpoints(use_numpy):
    samples = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.75, 0.9, 1], dtype=np.float32)
    expected = (
        np.array(
            [
                0,
                0.087813976,
                0.154170045,
                0.270667653,
                0.475195931,
                0.660478698,
                0.791680132,
                0.918004727,
                1,
            ],
            dtype=np.float32,
        )
        * 255
    )
    image = np.broadcast_to(samples * 255, (3, 2, 9)).copy()
    source = image if use_numpy else torch.from_numpy(image)
    output = _adjust(source, shadow_lift=100)
    np.testing.assert_allclose(np.asarray(output)[0, 0], expected, atol=3e-5)
    np.testing.assert_array_equal(np.asarray(source), image)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def should_preserve_identity_without_conversion_or_allocation(dtype):
    image = np.zeros((3, 5, 7), dtype=dtype)
    for source in (image, torch.from_numpy(image)):
        assert _adjust(source, shadow_lift=0, shadow_lift_black_point=True) is source


def should_preserve_hue_and_apply_correct_bgr_weights():
    rgb = np.broadcast_to(np.array([0.1, 0.25, 0.5])[:, None, None] * 255, (3, 5, 7)).copy()
    output = _adjust(rgb, shadow_lift=100)
    np.testing.assert_allclose(output[0] / output[1], rgb[0] / rgb[1])
    np.testing.assert_allclose(output[2] / output[1], rgb[2] / rgb[1])
    bgr_output = _adjust(rgb[::-1].copy(), shadow_lift=100, channel_order="bgr")
    np.testing.assert_allclose(bgr_output[::-1], output)


def should_compose_exposure_white_balance_and_shadow_with_cpu_tensor_parity():
    image = np.broadcast_to(
        np.array([100, 230, 30], dtype=np.float32)[:, None, None], (3, 5, 7)
    ).copy()
    settings = dict(
        white_balance=[0.5, 2, 1],
        exposure_ev=1,
        brightness=1.5,
        shadow_lift=100,
        shadow_lift_black_point=True,
    )
    numpy_output = _adjust(image, **settings)
    tensor_output = _adjust(torch.from_numpy(image), **settings)
    np.testing.assert_allclose(numpy_output, tensor_output.numpy(), atol=3e-5)


def should_preserve_uint16_alpha_during_grading():
    image = torch.full((4, 5, 7), 65535, dtype=torch.uint16)
    image[:3] = 0
    image[3, 0] = 0
    output = _adjust(image, shadow_lift=100, shadow_lift_black_point=True)
    torch.testing.assert_close(output[3], image[3])
    torch.testing.assert_close(output[:, 0], image[:, 0])
    assert abs(int(output[0, 1, 0]) - 65535 * 0.15) < 2


@pytest.mark.parametrize("use_numpy", [False, True])
def should_preserve_alpha_and_unmapped_pixels_with_black_point_lift(use_numpy):
    image = np.full((2, 5, 7, 4), 255, dtype=np.uint8)
    image[..., :3] = 0
    image[:, 0, :, 3] = 0
    source = image if use_numpy else torch.from_numpy(image)
    output = np.asarray(_adjust(source, shadow_lift=100, shadow_lift_black_point=True))
    np.testing.assert_array_equal(output[..., 3], image[..., 3])
    np.testing.assert_array_equal(output[:, 0], image[:, 0])
    np.testing.assert_array_equal(output[:, 1:, :, :3], 38)


@pytest.mark.parametrize("use_numpy", [False, True])
def should_grade_uint16_without_overflow_or_clipping_to_eight_bits(use_numpy):
    image = np.full((3, 5, 7), 32768, dtype=np.uint16)
    source = image if use_numpy else torch.from_numpy(image)
    output = np.asarray(_adjust(source, shadow_lift=100))
    assert output.dtype == np.uint16
    assert np.isfinite(output).all()
    assert 32768 < int(output[0, 0, 0]) < 65535
    np.testing.assert_allclose(output, (32768 / 65535) ** 0.812 * 65535, atol=2)


def should_accept_explicit_normalized_signal_range_and_retain_fractional_steps():
    image = torch.linspace(0.25, 0.27, 35).reshape(1, 5, 7).expand(3, -1, -1)
    output = _adjust(image, shadow_lift=100, input_max_value=1)
    torch.testing.assert_close(output, image.pow(0.812), rtol=1e-6, atol=1e-7)
    assert torch.unique(output).numel() == 35


def should_refresh_runtime_settings_and_restore_disabled_default_when_removed():
    settings = {"shadow_lift": 100, "shadow_lift_black_point": True}
    transform = HmImageColorAdjust(config_ref=settings)
    image = torch.zeros(3, 5, 7)
    assert transform({"img": image})["img"].min() > 0
    settings["shadow_lift_black_point"] = False
    torch.testing.assert_close(transform({"img": image})["img"], image)
    settings.clear()
    assert transform({"img": image})["img"] is image


@pytest.mark.parametrize("value", [-1, 101, float("inf"), float("nan"), "invalid"])
def should_reject_invalid_runtime_shadow_settings(value):
    transform = HmImageColorAdjust(config_ref={"shadow_lift": value})
    with pytest.raises(ValueError):
        transform({"img": torch.zeros(3, 5, 7)})


def should_reject_ambiguous_boolean_configuration():
    with pytest.raises(ValueError, match="must be a boolean"):
        HmImageColorAdjust(shadow_lift=50, shadow_lift_black_point="false")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def should_keep_gpu_pixels_on_device_with_cpu_parity():
    image = torch.linspace(0, 255, 3 * 5 * 7).reshape(3, 5, 7)
    cpu = _adjust(image, shadow_lift=75, shadow_lift_black_point=True)
    gpu = _adjust(image.cuda(), shadow_lift=75, shadow_lift_black_point=True)
    assert gpu.is_cuda
    torch.testing.assert_close(gpu.cpu(), cpu, atol=3e-5, rtol=1e-6)
