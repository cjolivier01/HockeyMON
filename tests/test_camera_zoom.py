import pytest
import torch

from hmlib.camera.moving_box import MovingBox
from hmlib.camera.zoom import zoom_in_shrink_thresholds


def _box() -> MovingBox:
    return MovingBox(
        label="follower",
        bbox=torch.tensor([400.0, 300.0, 600.0, 500.0]),
        arena_box=torch.tensor([0.0, 0.0, 1000.0, 800.0]),
        max_speed_x=torch.tensor(20.0),
        max_speed_y=torch.tensor(20.0),
        max_accel_x=torch.tensor(5.0),
        max_accel_y=torch.tensor(5.0),
        max_width=1000,
        max_height=800,
        stop_on_dir_change=False,
        sticky_sizing=True,
        time_to_dest_speed_limit_frames=0,
        device="cpu",
    )


@pytest.mark.parametrize(
    "aggression,expected", [(0, (0.16, 0.2)), (25, (0.08, 0.1)), (100, (0.008, 0.01))]
)
def should_match_hstream_zoom_endpoints(aggression, expected):
    assert zoom_in_shrink_thresholds(aggression) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, 101, 25.5, 25.0, "25", True, None])
def should_reject_invalid_zoom_config(value):
    with pytest.raises(ValueError, match="zoom_in_aggressiveness"):
        zoom_in_shrink_thresholds(value)


def should_zoom_into_small_changes_only_at_higher_aggression():
    default = _box()
    eager = _box()
    eager.set_resizing_shrink_thresholds(*zoom_in_shrink_thresholds(100))
    target = torch.tensor([403.0, 303.0, 597.0, 497.0])
    for _ in range(5):
        default.forward(target)
        eager.forward(target)
    assert torch.equal(default.bounding_box(), torch.tensor([400.0, 300.0, 600.0, 500.0]))
    assert eager.bounding_box()[0] > default.bounding_box()[0]
    assert eager.bounding_box()[2] < default.bounding_box()[2]


@pytest.mark.parametrize("axis", ["x", "y"])
def should_finish_braking_on_original_deadline_after_repeated_overshoots(axis):
    box = _box()
    setattr(box, f"_current_speed_{axis}", torch.tensor(12.0))
    target = torch.tensor([700.0, 600.0, 900.0, 800.0])
    for frame in range(4):
        box.begin_stop_delay(**{f"delay_{axis}": 4})
        box.forward(target)
        if frame < 3:
            assert int(getattr(box, f"_stop_delay_{axis}_counter")) == frame + 1
    assert float(getattr(box, f"_current_speed_{axis}")) == 0.0
    assert int(getattr(box, f"_stop_delay_{axis}")) == 0


def should_start_braking_independently_on_other_axis():
    box = _box()
    box._current_speed_x = torch.tensor(12.0)
    box._current_speed_y = torch.tensor(8.0)
    box.begin_stop_delay(delay_x=4)
    box.forward(torch.tensor([700.0, 600.0, 900.0, 800.0]))
    box.begin_stop_delay(delay_x=8, delay_y=2)
    assert int(box._stop_delay_x) == 4
    assert int(box._stop_delay_x_counter) == 1
    assert int(box._stop_delay_y) == 2
    assert int(box._stop_delay_y_counter) == 0
