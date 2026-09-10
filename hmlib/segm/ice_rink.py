"""
Ice Rink segmentation stuff, basically find the actual ice sheet in the image
"""

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Polygon
from PIL import Image

from hmlib.config import (
    get_game_config_private,
    get_game_dir,
    get_nested_value,
    prepend_root_dir,
    save_private_config,
    set_nested_value,
)
from hmlib.log import logger, logging
from hmlib.models.loader import get_model_config
from hmlib.segm.utils import calculate_centroid
from hmlib.ui import show_image as do_show_image
from hmlib.utils.gpu import GpuAllocator, StreamTensorBase
from hmlib.utils.image import (
    image_height,
    image_width,
    is_channels_first,
    make_channels_first,
    make_channels_last,
)

DEFAULT_SCORE_THRESH = 0.3

if TYPE_CHECKING:
    from mmdet.models.detectors.base import BaseDetector
    from mmdet.structures import DetDataSample


def _game_dir_for_id(game_id: str, assert_exists: bool = True) -> str:
    game_dir = get_game_dir(game_id=game_id, assert_exists=assert_exists)
    if game_dir is not None:
        return game_dir
    base_dir = os.environ.get("HM_GAME_DIR") or os.path.join(os.environ["HOME"], "Videos")
    return str(Path(base_dir) / game_id)


def _get_first_frame(video_path: str) -> Optional[torch.Tensor]:
    video_reader = mmcv.VideoReader(str(video_path))
    frame = video_reader.read()
    if frame is None:
        return None
    return torch.from_numpy(frame).unsqueeze(0)


def scale_images_nn(images: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale a batch of images by a given factor using nearest-neighbor interpolation.

    Args:
        images (torch.Tensor):
            Input batch of images, either
            - (B, C, H, W) for color/batched images, or
            - (B, H, W)   for grayscale batches.
        scale (float):
            Positive scaling factor.

    Returns:
        torch.Tensor:
            Scaled batch. If input was (B, C, H, W), output is (B, C, H', W').
            If input was (B, H, W), output is (B, H', W').
    """
    if scale <= 0.0:
        raise ValueError(f"Scale must be > 0, got {scale}")

    was_channels_first = None

    # Handle grayscale batch by adding a channel dim
    is_gray: bool = images.ndim == 3  # (B, H, W)
    if is_gray:
        # becomes (B, 1, H, W)
        images = images.unsqueeze(1)
    elif images.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(images.shape)}")
    else:
        was_channels_first = is_channels_first(images)
        images = make_channels_first(images)

    # compute new spatial size
    _, _, H, W = images.shape
    new_H: int = int(H * scale)
    new_W: int = int(W * scale)

    # interpolate all at once
    scaled: torch.Tensor = F.interpolate(images, size=(new_H, new_W), mode="nearest")

    # if grayscale, remove the channel dim back to (B, H', W')
    if is_gray:
        scaled = scaled.squeeze(1)
    else:
        if not was_channels_first:
            scaled = make_channels_last(scaled)

    return scaled


def find_extreme_points(
    mask: torch.Tensor,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Find the extreme points in a binary mask where the bit is set.

    Args:
    - mask (torch.Tensor): A 2D tensor representing the binary mask.

    Returns:
    - Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
      Returns the coordinates (y, x) of the smallest x, largest x, smallest y, and largest y that have the bit set.
    """
    # Get the indices where the bit is set (value is 1)
    y_indices, x_indices = torch.where(mask == 1)

    # Find minimum and maximum x and y
    min_x = torch.min(x_indices)
    max_x = torch.max(x_indices)
    min_y = torch.min(y_indices)
    max_y = torch.max(y_indices)

    # Get the corresponding y and x positions
    min_x_pos = (y_indices[x_indices == min_x][0].item(), min_x.item())
    max_x_pos = (y_indices[x_indices == max_x][0].item(), max_x.item())
    min_y_pos = (min_y.item(), x_indices[y_indices == min_y][0].item())
    max_y_pos = (max_y.item(), x_indices[y_indices == max_y][0].item())

    return min_x_pos, max_x_pos, min_y_pos, max_y_pos


def enclosing_bbox(bboxes: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)

    bboxes = bboxes[:, :4]

    # Calculate widths and heights
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]

    # Filter out bounding boxes where width or height is zero
    valid_indices = (widths > 0) & (heights > 0)
    valid_bboxes = bboxes[valid_indices]

    # If no valid bboxes, return None or an indicative value
    if valid_bboxes.shape[0] == 0:
        return None

    # Calculate min and max coordinates for the remaining valid bounding boxes
    min_xy, _ = torch.min(valid_bboxes[:, :2], dim=0)
    max_xy, _ = torch.max(valid_bboxes[:, 2:], dim=0)

    # Concatenate min and max values to form the enclosing bounding box
    return torch.cat([min_xy, max_xy])


def result_to_polygons(
    inference_result: "DetDataSample",
    category_id: int = 1,
    score_thr: float = 0,
    show: bool = False,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    """
    Theoretically, could return more than one polygon, especially if there's an obstruction
    """
    bboxes = inference_result.pred_instances.bboxes
    labels = inference_result.pred_instances.labels
    segms = inference_result.pred_instances.masks
    scores = inference_result.pred_instances.scores

    category_mask = labels == category_id
    bboxes = bboxes[category_mask, :]
    labels = labels[category_mask]
    if segms is not None:
        segms = segms[category_mask, ...]

    if score_thr > 0:
        # assert bboxes is not None and bboxes.shape[1] == 5
        # scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    if not len(segms):
        print("No ice rink found")
        return None

    masks = segms

    contours_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    combined_mask = None

    combined_bbox = enclosing_bbox(bboxes)

    from mmdet.structures.mask import bitmap_to_polygon

    for _, mask in enumerate(masks):
        contours, _ = bitmap_to_polygon(mask.cpu().numpy())
        # split_points_by_x_trend_efficient(contours)
        contours_list += contours
        mask = mask.to(torch.bool)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask | mask
        mask_list.append(mask)

        if show:
            mask_image = mask.to(np.uint8) * 255
            # cv2.namedWindow("Ice-rink", 0)
            # mmcv.imshow(mask_image, "Ice-rink Mask", wait_time=90)
            do_show_image("Ice-rink", mask_image)

    results: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]] = {}
    results["contours"] = contours_list
    results["masks"] = [m.cpu() for m in mask_list]
    results["combined_mask"] = combined_mask.cpu()
    results["centroid"] = calculate_centroid(contours_list).cpu()
    results["bboxes"] = bboxes.cpu()
    results["combined_bbox"] = combined_bbox.cpu()

    return results


def contours_to_polygons(contours: List[np.ndarray]) -> List[Polygon]:
    return [Polygon(c) for c in contours]


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    mask_tensor = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(mask_tensor, size=(height, width), mode="nearest")
    return resized.squeeze(0).squeeze(0).to(torch.bool)


def _rescale_rink_results(
    rink_results: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]],
    scale_factor: float,
    original_size: Tuple[int, int],
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    if scale_factor == 1.0:
        return rink_results

    inv_scale = 1.0 / scale_factor
    orig_height, orig_width = original_size

    if "contours" in rink_results and rink_results["contours"]:
        scaled_contours: List[np.ndarray] = []
        for contour in rink_results["contours"]:
            contour_np = np.asarray(contour, dtype=np.float32)
            scaled_contours.append(contour_np * inv_scale)
        rink_results["contours"] = scaled_contours

    if "masks" in rink_results and rink_results["masks"]:
        rink_results["masks"] = [
            _resize_mask(mask, orig_height, orig_width) for mask in rink_results["masks"]
        ]

    if "combined_mask" in rink_results and rink_results["combined_mask"] is not None:
        rink_results["combined_mask"] = _resize_mask(
            rink_results["combined_mask"], orig_height, orig_width
        )

    if "centroid" in rink_results and rink_results["centroid"] is not None:
        rink_results["centroid"] = rink_results["centroid"] * inv_scale

    if "bboxes" in rink_results and rink_results["bboxes"] is not None:
        rink_results["bboxes"] = rink_results["bboxes"] * inv_scale

    if "combined_bbox" in rink_results and rink_results["combined_bbox"] is not None:
        rink_results["combined_bbox"] = rink_results["combined_bbox"] * inv_scale

    return rink_results


def detect_ice_rink_mask(
    image: Union[torch.Tensor, np.ndarray],
    model: "BaseDetector",
    show: bool = False,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    from mmdet.apis import inference_detector

    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            assert image.shape[0] == 1
            image = image.squeeze(0)
        image = image.cpu().numpy()
    image = make_channels_last(image)
    result: "DetDataSample" = inference_detector(model, image)

    if show:
        show_image = image.cpu().unsqueeze(0).numpy() if isinstance(image, torch.Tensor) else image
        # show_image = model.show_result(show_image, result, score_thr=DEFAULT_SCORE_THRESH, show=False)
        do_show_image("Ice-rink", show_image, wait=True)

    return result_to_polygons(inference_result=result, score_thr=DEFAULT_SCORE_THRESH, show=False)


def find_ice_rink_masks(
    image: Union[torch.Tensor, List[torch.Tensor]],
    config_file: str,
    checkpoint: str,
    device: Optional[torch.device] = None,
    show: bool = False,
    inference_scale: Optional[float] = None,
) -> List[Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]]:
    from mmdet.apis import init_detector

    if device is None:
        # device = torch.device("cuda:0")
        device = torch.device("cpu")

    was_list = isinstance(image, list)
    if not was_list:
        image = [image]

    # image = [make_channels_first(img) for img in image]
    for i, img in enumerate(image):
        if len(img.shape) == 4:
            assert img.shape[0] == 1
            image[i] = img.squeeze(0)

    if device.type == "cpu":
        logger.info("Looking for the ice at the rink, this may take awhile...")
    model = init_detector(config_file, checkpoint, device=device)
    results: List[
        Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]
    ] = []
    for img in image:
        orig_height = image_height(img)
        orig_width = image_width(img)
        infer_image: Union[np.ndarray, torch.Tensor]
        if isinstance(img, torch.Tensor):
            infer_image = img.cpu().numpy()
        else:
            infer_image = img
        infer_image = np.ascontiguousarray(infer_image)

        if inference_scale and inference_scale != 1.0:
            new_width = max(1, int(round(orig_width * inference_scale)))
            new_height = max(1, int(round(orig_height * inference_scale)))
            interpolation = cv2.INTER_AREA if inference_scale < 1.0 else cv2.INTER_LINEAR
            infer_image = cv2.resize(
                infer_image, (new_width, new_height), interpolation=interpolation
            )

        rink_result = detect_ice_rink_mask(image=infer_image, model=model, show=show)

        if rink_result is None:
            results.append(rink_result)
            continue

        if inference_scale and inference_scale != 1.0:
            rink_result = _rescale_rink_results(
                rink_result, inference_scale, (orig_height, orig_width)
            )

        results.append(rink_result)

    # Try to free up some CUDA memory
    del model
    gc.collect()

    if device.type == "cpu":
        logger.info("Found the ice at the rink, continuing...")

    if not was_list:
        assert len(results) == 1
        return results[0]
    return results


def load_png_as_boolean_tensor(filename: str) -> torch.Tensor:
    """
    Loads a PNG image and converts it to a 2D boolean tensor.

    Args:
        filename (str): The path to the PNG image file.

    Returns:
        torch.Tensor: A 2D boolean tensor where each value represents the pixel's thresholded result.
    """
    # Load the image
    image = Image.open(filename).convert("L")  # Convert to grayscale

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Define a threshold (this example uses 128 as a threshold for mid-gray)
    threshold = 128

    # Apply threshold to create a boolean array
    boolean_array = image_array > threshold

    # Convert to a PyTorch boolean tensor
    boolean_tensor = torch.from_numpy(boolean_array).to(torch.bool)

    return boolean_tensor


def save_boolean_tensor_as_png(tensor: Union[torch.Tensor, np.ndarray], filename: str):
    """
    Saves a PyTorch boolean tensor as a PNG image.

    Args:
        tensor (torch.Tensor): A 2D boolean tensor.
        filename (str): The filename where the image will be saved.
    """
    # Ensure the tensor is on CPU and convert to uint8
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    tensor = tensor.cpu().to(torch.uint8) * 255  # Convert boolean to 0 and 255

    # Convert to a PIL image
    image = Image.fromarray(tensor.numpy())

    # Save the image
    image.save(filename)


def save_rink_profile_config(
    game_id: str,
    rink_profile: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]],
) -> Dict[str, Any]:
    game_config = get_game_config_private(game_id=game_id)
    masks = rink_profile.get("masks")
    mask_count = len(masks) if masks is not None else 0
    set_nested_value(game_config, "rink.ice_contours_mask_count", mask_count)
    centroid = rink_profile["centroid"]
    centroid = [float(centroid[0]), float(centroid[1])]
    set_nested_value(game_config, "rink.ice_contours_mask_centroid", centroid)

    combined_bbox = None
    if rink_profile["combined_bbox"] is not None:
        combined_bbox = [float(i) for i in rink_profile["combined_bbox"]]
    set_nested_value(game_config, "rink.ice_contours_combined_bbox", combined_bbox)
    game_dir = _game_dir_for_id(game_id, assert_exists=False)
    Path(game_dir).mkdir(parents=True, exist_ok=True)
    mask_image_file_base = str(Path(game_dir) / "rink_mask_")
    for i in range(mask_count):
        mask = masks[i]
        image_file = mask_image_file_base + str(i) + ".png"
        save_boolean_tensor_as_png(mask, image_file)
    save_private_config(game_id=game_id, data=game_config, verbose=True)


def load_rink_combined_mask(
    game_id: str,
) -> Optional[Dict[str, Optional[torch.Tensor]]]:
    game_config = get_game_config_private(game_id=game_id)
    if not game_config:
        return None
    mask_count = get_nested_value(game_config, "rink.ice_contours_mask_count", None)
    if mask_count is None:
        return None
    combined_mask = None
    game_dir = _game_dir_for_id(game_id, assert_exists=False)
    mask_image_file_base = str(Path(game_dir) / "rink_mask_")
    for i in range(mask_count):
        image_file = mask_image_file_base + str(i) + ".png"
        if not os.path.exists(image_file):
            # Missing the actual mask file, so return as if nothing was found
            return None
        mask = load_png_as_boolean_tensor(image_file)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask | mask
    mask_count = get_nested_value(game_config, "rink.ice_contours_mask_count", None)
    centroid = get_nested_value(game_config, "rink.ice_contours_mask_centroid", None)
    if centroid is not None:
        centroid = torch.tensor(centroid, dtype=torch.float)
    combined_bbox = get_nested_value(game_config, "rink.ice_contours_combined_bbox", None)
    results: Dict[str, Optional[torch.Tensor]] = {
        "combined_mask": combined_mask,
        "centroid": centroid,
        "combined_bbox": combined_bbox,
    }
    return results


def get_device_to_use_for_rink(
    gpu_allocator: GpuAllocator, default_device: torch.device = torch.device("cpu")
) -> torch.device:
    if gpu_allocator is not None and not gpu_allocator.is_single_lowmem_gpu():
        device_index = gpu_allocator.get_largest_mem_gpu()
        if device_index >= 0:
            return torch.device("cuda", device_index)
    return default_device


def configure_ice_rink_mask(
    game_id: str,
    expected_shape: Optional[torch.Size],
    device: Optional[torch.device] = None,
    force: bool = False,
    show: bool = False,
    image: torch.Tensor = None,
    scale: Optional[float] = None,
    persist: bool = True,
) -> Optional[torch.Tensor]:
    if expected_shape is None and image is not None:
        expected_shape = torch.Size((image_height(image), image_width(image)))
    if not force:
        combined_mask_profile = load_rink_combined_mask(game_id=game_id)
        if combined_mask_profile:
            if expected_shape is None:
                return combined_mask_profile
            combined_mask = combined_mask_profile["combined_mask"]
            mask_w = image_width(combined_mask)
            mask_h = image_height(combined_mask)
            assert len(expected_shape) == 2  # (H, W)
            if mask_w == expected_shape[1] and mask_h == expected_shape[0]:
                return combined_mask_profile
            else:
                logging.warning(
                    f"Expected rink mask of size w={expected_shape[1]}, h={expected_shape[0]} does not match actual"
                    f"rink mask size of w={mask_w}, h={mask_h}, so mask must be reconstructed"
                )

    model_config_file, model_checkpoint = get_model_config(
        game_id=game_id, model_name="ice_rink_segm"
    )

    assert model_config_file
    assert model_checkpoint
    # TODO: Should prioritize passed-in image over s.png and then make
    # sure everything is the expected size within the config
    game_dir = get_game_dir(game_id=game_id)
    if not game_dir:
        raise AttributeError(f"Could not determine game dir for game_id={game_id}")
    image_file: Path = Path(game_dir) / "s.png"
    image_frame: Optional[torch.Tensor] = None
    if image is not None:
        image_frame = image
        if isinstance(image_frame, StreamTensorBase):
            image_frame = image_frame.get()
        if isinstance(image_frame, torch.Tensor) and image_frame.ndim == 4:
            image_frame = image_frame[0]
        if expected_shape is not None:
            assert image_width(image_frame) == expected_shape[-1]
            assert image_height(image_frame) == expected_shape[-2]
        if device is not None and image_frame.device != device:
            # Just synchronize everyone since this only happens once
            if image_frame.is_cuda and device.type == "cuda":
                torch.cuda.synchronize()
            image_frame = image_frame.to(device)
    elif not image_file.exists():
        if image is None:
            raise AttributeError(f"Could not find stitched frame image: {image_file}")
        assert image_width(image) == expected_shape[-1]
        assert image_height(image) == expected_shape[-2]
        image_frame = image
        if device is not None and image_frame.device != device:
            if isinstance(image_frame, StreamTensorBase):
                image_frame = image_frame.get()
                assert image_frame.ndim == 4
                image_frame = image_frame[0]
                # Just synchronize everyone since this only happens once
                torch.cuda.synchronize()
            image_frame = image_frame.to(device)
    else:
        image_frame = _get_first_frame(image_file)
    if expected_shape is None:
        expected_shape = torch.Size((image_height(image_frame), image_width(image_frame)))
    else:
        assert image_height(image_frame) == expected_shape[0]
        assert image_width(image_frame) == expected_shape[1]

    rink_results = find_ice_rink_masks(
        image=image_frame,
        config_file=prepend_root_dir(model_config_file),
        checkpoint=prepend_root_dir(model_checkpoint),
        show=show,
        device=device,
        inference_scale=scale,
    )
    if "combined_mask" in rink_results:
        rink_mask = rink_results["combined_mask"]
        assert image_width(rink_mask) == image_width(image_frame)
        assert image_height(rink_mask) == image_height(image_frame)
    if not persist:
        return rink_results
    if rink_results:
        save_rink_profile_config(game_id=game_id, rink_profile=rink_results)
    return load_rink_combined_mask(game_id=game_id)


@dataclass
class MaskEdgeDistances:
    """
    Precomputed edge distances for a binary mask.
    """

    top_edges: torch.Tensor  # Shape: (W,)
    bottom_edges: torch.Tensor  # Shape: (W,)
    left_edges: torch.Tensor  # Shape: (H,)
    right_edges: torch.Tensor  # Shape: (H,)
    mask: torch.Tensor  # Original mask for reference

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> "MaskEdgeDistances":
        """
        Precompute the edge positions for each row and column.

        Parameters:
        - mask (torch.Tensor): A 2D binary tensor of shape (H, W).

        Returns:
        - MaskEdgeDistances: An instance with precomputed edges.
        """
        # Ensure mask is binary
        assert mask.dim() == 2, "Mask must be a 2D tensor"
        H, W = mask.shape

        # Precompute top and bottom edges for each column (x)
        top_edges = torch.full((W,), -1, dtype=torch.long)  # Initialize with -1
        bottom_edges = torch.full((W,), -1, dtype=torch.long)

        for x in range(W):
            column_indices = torch.nonzero(mask[:, x]).squeeze()
            if column_indices.numel() > 0:
                top_edges[x] = column_indices.min().item()
                bottom_edges[x] = column_indices.max().item()

        # Precompute left and right edges for each row (y)
        left_edges = torch.full((H,), -1, dtype=torch.long)
        right_edges = torch.full((H,), -1, dtype=torch.long)

        for y in range(H):
            row_indices = torch.nonzero(mask[y, :]).squeeze()
            if row_indices.numel() > 0:
                left_edges[y] = row_indices.min().item()
                right_edges[y] = row_indices.max().item()

        return cls(
            top_edges=top_edges,
            bottom_edges=bottom_edges,
            left_edges=left_edges,
            right_edges=right_edges,
            mask=mask,
        )

    def distances_to_edges(
        self, x: int, y: int
    ) -> Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
        """
        Calculate distances from point (x, y) to the precomputed edges.

        Parameters:
        - x (int): The x-coordinate (column index) of the point.
        - y (int): The y-coordinate (row index) of the point.

        Returns:
        - Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
          (top_distance, bottom_distance, left_distance, right_distance)
          If the point lies outside the bitmask, returns None.
          Individual distances can also be None if there is no edge in that direction.
        """
        H, W = self.mask.shape

        # Check if x and y are within bounds
        if not (0 <= x < W and 0 <= y < H):
            return None

        # Check if the point lies within the bitmask
        if self.mask[y, x].item() == 0:
            return None

        # Get top and bottom edges for column x
        top_edge = self.top_edges[x].item()
        bottom_edge = self.bottom_edges[x].item()

        # Compute vertical distances
        top_distance: Optional[int] = y - top_edge if top_edge != -1 else None
        bottom_distance: Optional[int] = bottom_edge - y if bottom_edge != -1 else None

        # Get left and right edges for row y
        left_edge = self.left_edges[y].item()
        right_edge = self.right_edges[y].item()

        # Compute horizontal distances
        left_distance: Optional[int] = x - left_edge if left_edge != -1 else None
        right_distance: Optional[int] = right_edge - x if right_edge != -1 else None

        return top_distance, bottom_distance, left_distance, right_distance


def main(args: argparse.Namespace = None, device: torch.device = torch.device("cpu")) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Ice rink segmentation script")
        parser.add_argument("--game-id", "-g", type=str, required=True, help="Game ID to process")
        parser.add_argument(
            "--show-image", "--show", action="store_true", help="Show the image with the mask"
        )
        parser.add_argument(
            "--force", "-f", action="store_true", help="Force reconfiguration of the ice rink mask"
        )
        parser.add_argument(
            "--scale", "-s", type=float, default=None, help="Scale factor for the image"
        )
        parser.add_argument(
            "--device", "-d", type=str, default=None, help="Device used for inference"
        )
        args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif device is None:
        device = torch.device("cpu")

    game_dir = _game_dir_for_id(args.game_id)
    stitched_frame_file = str(Path(game_dir) / "s.png")
    if not os.path.exists(stitched_frame_file):
        print(f"Could not find stitched frame image: {stitched_frame_file}")
        exit(1)

    image = cv2.imread(stitched_frame_file)

    mask_scale = getattr(args, "ice_rink_inference_scale", None)
    if mask_scale is None:
        mask_scale = getattr(args, "scale", None)
    if mask_scale is not None:
        setattr(args, "ice_rink_inference_scale", mask_scale)

    results = configure_ice_rink_mask(
        game_id=args.game_id,
        device=device,
        show=args.show_image,
        force=args.force,
        image=image,
        expected_shape=(image_height(image), image_width(image)),
        scale=mask_scale,
    )
    mask = results["combined_mask"]
    centroid = [int(i) for i in results["centroid"]]
    checker = MaskEdgeDistances.from_mask(mask)
    cent_dist = checker.distances_to_edges(x=centroid[0], y=centroid[1])
    logger.info(
        f"centroid={centroid}, distances=(top={cent_dist[0]}, bottom={cent_dist[1]}, left={cent_dist[2]}, right={cent_dist[3]}"
    )


if __name__ == "__main__":
    main()
