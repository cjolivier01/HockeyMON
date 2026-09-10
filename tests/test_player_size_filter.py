import pytest
import torch

from hmlib.bbox.box_functions import player_size_exclusion_mask


@pytest.mark.parametrize(
    "areas,count,enabled,percent,removed",
    [
        ([100, 100, 100, 100, 300, 1000], 0, False, 100, []),
        ([100, 100, 100, 100, 300, 1000], 1, False, 100, [5]),
        ([100, 100, 100, 100, 300, 1000], 2, False, 100, [4, 5]),
        ([100, 100, 100, 100, 300, 1000], 100000, False, 100, [0, 4, 5]),
        ([100, 100, 100, 100, 300, 1000], 1, True, 100, [4, 5]),
        ([100, 100, 100, 201], 0, True, 100, [3]),
        ([100, 100, 100, 200], 0, True, 100, []),
        ([100, 100, 100, 100, 300, 1000], 0, True, 100, [5]),
        ([0, 0, 0, 100], 0, True, 100, [3]),
        ([100, 100, 100], 5, True, 0, []),
        ([], 5, True, 0, []),
    ],
)
def should_filter_player_areas(areas, count, enabled, percent, removed):
    boxes = torch.tensor([[0, 0, area, 1] for area in areas], dtype=torch.float32).reshape(-1, 4)
    keep = player_size_exclusion_mask(boxes, count, enabled, percent)
    assert (~keep).nonzero().flatten().tolist() == removed
    assert keep.device == boxes.device


@pytest.mark.parametrize(
    "count,percent",
    [(-1, 100), (1.5, 100), (True, 100), (1, -1), (1, float("nan")), (1, float("inf"))],
)
def should_reject_invalid_settings(count, percent):
    with pytest.raises(ValueError):
        player_size_exclusion_mask(torch.empty((0, 4)), count, False, percent)
