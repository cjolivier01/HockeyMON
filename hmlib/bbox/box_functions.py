import math
from typing import Union

import numpy as np
import torch


def tlwh_centers(tlwhs: torch.Tensor):
    """
    Calculate the centers of bounding boxes given as [x1, y1, width, height].

    Parameters:
    bboxes (Tensor): A tensor of size (N, 4) where N is the number of boxes,
                     and each box is defined as [x1, y1, width, height].

    Returns:
    Tensor: A tensor of size (N, 2) containing [cx, cy] for each bounding box.
    """
    # Unpack the bounding box tensor into its components
    if tlwhs.numel() == 0:
        return tlwhs.new_empty((0, 2))

    x1, y1, width, height = tlwhs.unbind(-1)
    cx = x1 + (width * 0.5)
    cy = y1 + (height * 0.5)
    return torch.stack((cx, cy), dim=-1)


def width(box: torch.Tensor) -> torch.Tensor:
    return box[2] - box[0]


def height(box: torch.Tensor) -> torch.Tensor:
    return box[3] - box[1]


def center(box: torch.Tensor) -> torch.Tensor:
    return (box[:2] + box[2:]) * 0.5


def center_batch(boxes: torch.Tensor):
    if boxes.numel() == 0:
        return boxes.new_empty((0, 2))
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def clamp_value(low, val, high):
    return torch.clamp(val, min=low, max=high)


def clamp_box(box, clamp_box):
    clamped_box = box.clone()
    clamped_box[0] = torch.clamp(box[0], min=clamp_box[0], max=clamp_box[2])
    clamped_box[1] = torch.clamp(box[1], min=clamp_box[1], max=clamp_box[3])
    clamped_box[2] = torch.clamp(box[2], min=clamp_box[0], max=clamp_box[2])
    clamped_box[3] = torch.clamp(box[3], min=clamp_box[1], max=clamp_box[3])
    return clamped_box


def make_box_at_center(center_point: torch.Tensor, w: torch.Tensor, h: torch.Tensor):
    assert torch.is_floating_point(center_point)
    half_w = w * 0.5
    half_h = h * 0.5
    box = center_point.new_empty(4)
    box[0] = center_point[0] - half_w
    box[1] = center_point[1] - half_h
    box[2] = center_point[0] + half_w
    box[3] = center_point[1] + half_h
    return box


def move_box_to_center(box: torch.Tensor, center_point: torch.Tensor):
    ww = width(box)
    hh = height(box)
    return make_box_at_center(center_point=center_point, w=ww, h=hh)


def scale_box(
    box: torch.Tensor,
    scale_width: Union[torch.Tensor, float],
    scale_height: Union[torch.Tensor, float],
) -> torch.Tensor:
    center_point = center(box)
    w = width(box) * scale_width
    h = height(box) * scale_height
    return make_box_at_center(center_point=center_point, w=w, h=h)


def center_distance(box1, box2):
    assert box1 is not None and box2 is not None
    return torch.norm(center(box1) - center(box2), p=2)


def center_x_distance(box1, box2) -> float:
    assert box1 is not None and box2 is not None
    return torch.abs(center(box1)[0] - center(box2)[0])


def aspect_ratio(box):
    return width(box) / height(box)


def tlwh_to_tlbr_multiple(tlwh: torch.Tensor):
    top_left = tlwh[:, :2]
    bottom_right = tlwh[:, :2] + tlwh[:, 2:]
    return torch.cat((top_left, bottom_right), dim=1)


def tlwh_to_tlbr_single(tlwh: torch.Tensor):
    top_left = tlwh[:2]
    bottom_right = tlwh[:2] + tlwh[2:]
    return torch.cat((top_left, bottom_right), dim=0)


def shift_box_to_edge(box, bounding_box):
    """
    If a box is off the edge of the image, translate
    the box to be flush with the edge instead.
    """
    xw = width(bounding_box)
    xh = height(bounding_box)
    was_shifted_x = False
    was_shifted_y = False
    if box[0] < 0:
        shift = -box[0]
        box[0] += shift
        box[2] += shift
        was_shifted_x = True
    elif box[2] >= xw:
        offset = box[2] - xw
        box[0] -= offset
        box[2] -= offset
        was_shifted_x = True

    if box[1] < 0:
        shift = -box[1]
        box[1] += shift
        box[3] += shift
        was_shifted_y = True
    elif box[3] >= xh:
        offset = box[3] - xh
        box[1] -= offset
        box[3] -= offset
        was_shifted_y = True
    return box, was_shifted_x, was_shifted_y


def is_box_edge_on_or_outside_other_box_edge(box, bounding_box):
    return torch.stack(
        [
            box[0] <= bounding_box[0],
            box[1] <= bounding_box[1],
            box[2] >= bounding_box[2],
            box[3] >= bounding_box[3],
        ]
    )


def check_for_box_overshoot(
    box: torch.Tensor,
    bounding_box: torch.Tensor,
    movement_directions: torch.Tensor,
    epsilon: float = 0.01,
):
    """
    Check is a proposed movement direction would push the given box off of the edge of the given boundary box.

    :param box:                     Box which is proposed to be moved [left, top, right, bottomo]
    :param bounding_box:            Box which it is to be inscribed [left, top, right, bottomo]
    :param movement_directions:     Proposed movement direction [dx_axis, dy_axis] (signs only)
    """
    any_on_edge = is_box_edge_on_or_outside_other_box_edge(box, bounding_box)
    x_on_edge = torch.logical_or(
        any_on_edge[0] & (movement_directions[0] < epsilon),
        any_on_edge[2] & (movement_directions[0] > -epsilon),
    )
    y_on_edge = torch.logical_or(
        any_on_edge[1] & (movement_directions[1] < epsilon),
        any_on_edge[3] & (movement_directions[1] > -epsilon),
    )
    return torch.stack([x_on_edge, y_on_edge])


def player_size_exclusion_mask(
    batch_bboxes: torch.Tensor,
    largest_count: int,
    ignore_oversized: bool,
    oversized_percent: float,
) -> torch.Tensor:
    """Keep TLWH player boxes using the same area filter as the native tracker.

    Apply the largest count first, then compare against each player's peers in
    the remaining snapshot. Keep at least three players. Ties use input order.
    All tensor operations stay on the input device.
    """
    if isinstance(largest_count, bool) or not isinstance(largest_count, int) or largest_count < 0:
        raise ValueError("cam_ignore_largest_count must be a nonnegative integer")
    if not math.isfinite(oversized_percent) or oversized_percent < 0:
        raise ValueError("cam_oversized_percent must be finite and nonnegative")
    size = batch_bboxes.shape[0]
    keep = torch.ones(size, dtype=torch.bool, device=batch_bboxes.device)
    if size <= 3 or (largest_count == 0 and not ignore_oversized):
        return keep
    # BBox::area() uses float32 products. Promote the result for accumulation,
    # preserving native ties and strict threshold boundaries on fractional boxes.
    areas = (batch_bboxes[:, 2].float() * batch_bboxes[:, 3].float()).to(torch.float64)
    order = torch.argsort(areas, descending=True, stable=True)
    count = min(largest_count, size - 3)
    keep[order[:count]] = False
    if ignore_oversized and count < size - 3:
        remaining = order[count:]
        remaining_areas = areas[remaining]
        other_average = (remaining_areas.sum() - remaining_areas) / (size - count - 1)
        oversized = remaining_areas > (1.0 + oversized_percent / 100.0) * other_average
        # Rank on device so the safeguard never needs scalar GPU readback.
        selected = oversized & (oversized.to(torch.int64).cumsum(0) <= size - 3 - count)
        keep[remaining] = ~selected
    return keep


def remove_largest_bbox(batch_bboxes: torch.Tensor, min_boxes: int):
    """
    Remove the bounding box with the largest area from a batch of TLWH bboxes.

    :param batch_bboxes: A tensor of shape (N, 4) where N is the batch size,
                         and each bbox is in TLWH format.
    :param secondary_tensor: optional secondary tensor to also mast
    :return: A tensor of bboxes with one less item, excluding the largest bbox,
             along with the mast and the larges box itself
    """
    # Calculate areas (width * height)

    num_boxes = batch_bboxes.shape[0]
    if num_boxes < min_boxes:
        mask = torch.ones(num_boxes, dtype=torch.bool, device=batch_bboxes.device)
        return batch_bboxes, mask, None

    areas = batch_bboxes[:, 2] * batch_bboxes[:, 3]
    largest_bbox_idx = torch.argmax(areas)
    mask = torch.ones(num_boxes, dtype=torch.bool, device=batch_bboxes.device)
    mask[largest_bbox_idx] = False
    return batch_bboxes[mask], mask, batch_bboxes[largest_bbox_idx]


def get_enclosing_box(batch_boxes: torch.Tensor):
    """
    Get a bounding box that encompasses all bounding boxes in the batch.

    Parameters:
    - batch_boxes (Tensor): A tensor of shape [N, 4] representing bounding boxes,
                            where each box is [top, left, bottom, right].

    Returns:
    - Tensor: A tensor representing the enclosing bounding box.
    """
    min_top = torch.min(batch_boxes[:, 0])
    min_left = torch.min(batch_boxes[:, 1])
    max_bottom = torch.max(batch_boxes[:, 2])
    max_right = torch.max(batch_boxes[:, 3])
    return batch_boxes.new_tensor([min_top, min_left, max_bottom, max_right])


def scale_box_at_same_center(box: torch.Tensor, scale_factor):
    """
    Scale a bounding box by a given factor while maintaining the same center.

    Parameters:
    - bbox (Tensor): A tensor representing a single bounding box [top, left, bottom, right].
    - scale_factor (float): The scaling factor.

    Returns:
    - Tensor: The scaled bounding box.
    """
    top, left, bottom, right = box
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    width = (right - left) * scale_factor
    height = (bottom - top) * scale_factor
    new_left = center_x - width * 0.5
    new_right = center_x + width * 0.5
    new_top = center_y - height * 0.5
    new_bottom = center_y + height * 0.5
    return box.new_tensor([new_top, new_left, new_bottom, new_right])


# def scale_box_to_fit(
#     box: torch.Tensor, max_width: torch.Tensor, max_height: torch.Tensor
# ):
#     box_width = width(box)
#     box_height = height(box)
#     scale_w = None
#     scale_h = None
#     error_count = 0
#     if box_width > max_width:
#         error_count += 1
#         scale_w = box_width / max_width
#     if box_height > max_height:
#         error_count += 1
#         scale_h = box_height / max_height
#     if error_count:
#         if error_count == 2:
#             scale = torch.max(scale_h, scale_w)
#             return box / scale
#         elif scale_w is not None:
#             return box / scale_w
#         else:
#             return box / scale_h
#     return box


def scale_bbox_with_constraints(bbox, ratio_x, ratio_y, min_x, max_x, min_y, max_y):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) * 0.5
    y_center = (y1 + y2) * 0.5
    new_width = (x2 - x1) * ratio_x
    new_height = (y2 - y1) * ratio_y
    new_x1 = max(min_x, x_center - new_width * 0.5)
    new_y1 = max(min_y, y_center - new_height * 0.5)
    new_x2 = min(max_x, x_center + new_width * 0.5)
    new_y2 = min(max_y, y_center + new_height * 0.5)
    if new_x1 >= new_x2:
        if new_x1 < max_x:
            new_x2 = new_x1 + 1
        else:
            new_x1 = max_x - 1
            new_x2 = max_x
    if new_y1 >= new_y2:
        if new_y1 < max_y:
            new_y2 = new_y1 + 1
        else:
            new_y1 = max_y - 1
            new_y2 = max_y
    return (new_x1, new_y1, new_x2, new_y2)


def convert_tlbr_to_tlwh(tlbr: Union[np.ndarray, torch.Tensor]):
    """
    Convert bounding boxes from TLBR format to TLWH format.

    Parameters:
    - tlbr (Tensor): A tensor containing bounding boxes in TLBR format (x1, y1, x2, y2).

    Returns:
    - Tensor: Bounding boxes in TLWH format (x, y, w, h).
    """
    if tlbr.ndim != 2 or tlbr.shape[1] != 4:
        raise ValueError("Input tensor must be of shape [N, 4]")
    x = tlbr[:, 0]
    y = tlbr[:, 1]
    w = tlbr[:, 2] - tlbr[:, 0]
    h = tlbr[:, 3] - tlbr[:, 1]
    if isinstance(tlbr, np.ndarray):
        return np.stack([x, y, w, h], axis=1)
    return torch.stack([x, y, w, h], dim=1)
