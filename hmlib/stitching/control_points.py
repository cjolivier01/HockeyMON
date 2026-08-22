"""Control-point utilities for panorama stitching.

Provides selectable learned control-point extraction and simple coordinate
filters used when building two-camera stitching projects.
"""

import gc
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hmlib.utils.image import image_height, image_width

CONTROL_POINT_MATCHERS = (
    "superpoint-lightglue",
    "dedode-lightglue",
    "loftr",
)
_MATCHER_ALIASES = {
    "superpoint": "superpoint-lightglue",
    "lightglue": "superpoint-lightglue",
    "dedode": "dedode-lightglue",
}
_DEDODE_MAX_IMAGE_DIMENSION = 1920
_LOFTR_MAX_IMAGE_DIMENSION = 1600


def evenly_spaced_indices(n_points, n_samples):
    """Generate indices to pick ``n_samples`` evenly spaced from ``n_points``."""
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch, n_samples):
    """Select a subset of points that are evenly spaced in Y.

    @param batch: Tensor of shape ``(N, 2)`` with ``(x, y)`` coordinates.
    @param n_samples: Number of sample indices to return.
    @return: 1D tensor of indices into ``batch``.
    """
    if batch.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=batch.device)
    n_samples = min(int(n_samples), int(batch.size(0)))
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    # Sort the points based on Y values
    _, sorted_indices = torch.sort(batch[:, 1])

    # Calculate indices that would space the points evenly
    sample_indices = evenly_spaced_indices(batch.size(0), n_samples)

    # Select indices of the original batch
    selected_indices = sorted_indices[sample_indices]

    return selected_indices


def normalize_control_point_matcher(matcher: str) -> str:
    """Return the canonical control-point matcher name or raise."""
    normalized = str(matcher).strip().lower().replace("_", "-")
    normalized = _MATCHER_ALIASES.get(normalized, normalized)
    if normalized not in CONTROL_POINT_MATCHERS:
        choices = ", ".join(CONTROL_POINT_MATCHERS)
        raise ValueError(f"Unsupported control-point matcher {matcher!r}; choose one of: {choices}")
    return normalized


def _image_to_rgb_tensor(
    image: Union[str, Path, np.ndarray, torch.Tensor],
) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read control-point image: {image}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("NumPy control-point images must have shape HxWx3 or HxWx4")
        image_bgr = image[:, :, :3]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("Batched control-point image tensors must have batch size one")
            tensor = tensor[0]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3, 4):
            if tensor.shape[-1] not in (1, 3, 4):
                raise ValueError("Control-point image tensors must be CHW or HWC")
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3 or tensor.shape[0] not in (1, 3, 4):
            raise ValueError("Control-point image tensors must have one, three, or four channels")
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, -1, -1)
        elif tensor.shape[0] == 4:
            tensor = tensor[:3]
    else:
        raise TypeError(f"Unsupported control-point image type: {type(image).__name__}")

    tensor = tensor.contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    elif not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError("Control-point images must contain only finite values")
    min_value = float(tensor.min())
    max_value = float(tensor.max())
    if min_value < 0.0 or max_value > 255.0:
        raise ValueError("Control-point image values must be in the range [0, 255]")
    if max_value > 1.0:
        tensor = tensor.div(255.0)
    return tensor


def _match_superpoint_lightglue(
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: torch.device,
    max_num_keypoints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = (
        LightGlue(
            features="superpoint",
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=0.2,
        )
        .eval()
        .to(device)
    )
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(value) for value in (feats0, feats1, matches01)]
    matches = matches01["matches"]
    return (
        feats0["keypoints"][matches[..., 0]],
        feats1["keypoints"][matches[..., 1]],
    )


def _match_dedode_lightglue(
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: torch.device,
    max_num_keypoints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import kornia.feature as kornia_feature

    resized0, scale_x0, scale_y0 = _resize_for_matching(
        image0,
        max_dimension=_DEDODE_MAX_IMAGE_DIMENSION,
    )
    resized1, scale_x1, scale_y1 = _resize_for_matching(
        image1,
        max_dimension=_DEDODE_MAX_IMAGE_DIMENSION,
    )
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    extractor = kornia_feature.DeDoDe.from_pretrained(
        detector_weights="L-C4-v2",
        descriptor_weights="B-upright",
        amp_dtype=amp_dtype,
    ).to(device)
    matcher = (
        kornia_feature.LightGlueMatcher(
            "dedodeb",
            params={
                "depth_confidence": -1,
                "width_confidence": -1,
                "filter_threshold": 0.2,
            },
        )
        .eval()
        .to(device)
    )
    keypoints0, _, descriptors0 = extractor(resized0.unsqueeze(0), n=max_num_keypoints)
    keypoints1, _, descriptors1 = extractor(resized1.unsqueeze(0), n=max_num_keypoints)
    lafs0 = kornia_feature.laf_from_center_scale_ori(keypoints0)
    lafs1 = kornia_feature.laf_from_center_scale_ori(keypoints1)
    _, matches = matcher(
        descriptors0[0],
        descriptors1[0],
        lafs0,
        lafs1,
        hw1=tuple(resized0.shape[-2:]),
        hw2=tuple(resized1.shape[-2:]),
    )
    points0 = keypoints0[0, matches[:, 0]]
    points1 = keypoints1[0, matches[:, 1]]
    points0 = points0 * points0.new_tensor([scale_x0, scale_y0])
    points1 = points1 * points1.new_tensor([scale_x1, scale_y1])
    return points0, points1


def _resize_for_matching(
    image: torch.Tensor,
    max_dimension: int,
    dimension_multiple: int = 1,
) -> Tuple[torch.Tensor, float, float]:
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    if dimension_multiple <= 0:
        raise ValueError("dimension_multiple must be positive")
    height, width = image.shape[-2:]
    scale = min(1.0, max_dimension / float(max(height, width)))
    resized_height = max(
        dimension_multiple,
        int(round(height * scale / dimension_multiple)) * dimension_multiple,
    )
    resized_width = max(
        dimension_multiple,
        int(round(width * scale / dimension_multiple)) * dimension_multiple,
    )
    if resized_height == height and resized_width == width:
        return image, 1.0, 1.0
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized, width / float(resized_width), height / float(resized_height)


def _resize_for_loftr(image: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    return _resize_for_matching(
        image,
        max_dimension=_LOFTR_MAX_IMAGE_DIMENSION,
        dimension_multiple=8,
    )


def _match_loftr(
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import kornia.feature as kornia_feature

    resized0, scale_x0, scale_y0 = _resize_for_loftr(image0)
    resized1, scale_x1, scale_y1 = _resize_for_loftr(image1)
    grayscale0 = resized0[0:1] * 0.299 + resized0[1:2] * 0.587 + resized0[2:3] * 0.114
    grayscale1 = resized1[0:1] * 0.299 + resized1[1:2] * 0.587 + resized1[2:3] * 0.114
    matcher = kornia_feature.LoFTR(pretrained="outdoor").eval().to(device)
    matches = matcher(
        {
            "image0": grayscale0.unsqueeze(0),
            "image1": grayscale1.unsqueeze(0),
        }
    )
    points0 = matches["keypoints0"]
    points1 = matches["keypoints1"]
    points0 = points0 * points0.new_tensor([scale_x0, scale_y0])
    points1 = points1 * points1.new_tensor([scale_x1, scale_y1])
    confidence = matches.get("confidence")
    if confidence is not None and confidence.numel() == points0.shape[0]:
        order = torch.argsort(confidence, descending=True)
        points0 = points0[order]
        points1 = points1[order]
    return points0, points1


def _to_visualization_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return cv2.cvtColor((array * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _save_match_visualizations(
    image0: torch.Tensor,
    image1: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    output_directory: str,
) -> None:
    left = _to_visualization_image(image0)
    right = _to_visualization_image(image1)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    keypoint_canvas = canvas.copy()
    for point0, point1 in zip(
        points0.detach().cpu().numpy(), points1.detach().cpu().numpy(), strict=True
    ):
        xy0 = tuple(np.rint(point0).astype(int))
        xy1 = tuple(np.rint(point1).astype(int) + np.array([left.shape[1], 0]))
        cv2.line(canvas, xy0, xy1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(keypoint_canvas, xy0, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(keypoint_canvas, xy1, 3, (0, 255, 0), -1, cv2.LINE_AA)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path / "matches.png"), canvas):
        raise OSError(f"Failed to write {output_path / 'matches.png'}")
    if not cv2.imwrite(str(output_path / "keypoints.png"), keypoint_canvas):
        raise OSError(f"Failed to write {output_path / 'keypoints.png'}")


def indices_below_y_threshold(batch: torch.Tensor, y_threshold: float):
    """Return indices of points whose Y is greater than ``y_threshold``."""
    # Find the indices where Y value is below the threshold
    indices = (batch[:, 1] > y_threshold).nonzero().squeeze()

    return indices


def indices_below_x_threshold(batch: torch.Tensor, x_threshold: float):
    """Return indices of points whose X is less than ``x_threshold``."""
    # Find the indices where X value is below the threshold
    indices = (batch[:, 1] < x_threshold).nonzero().squeeze()

    return indices


def indices_above_x_threshold(batch: torch.Tensor, x_threshold: float):
    """Return indices of points whose X is greater than ``x_threshold``."""
    # Find the indices where X value is below the threshold
    indices = (batch[:, 1] > x_threshold).nonzero().squeeze()

    return indices


def indices_of_min_x(points: torch.Tensor, N: int):
    """Return indices of the ``N`` points with smallest X in a set."""
    # Concatenate the two batches

    # Sort based on the X values and get the indices
    _, indices = torch.sort(points[:, 0])

    # Select the indices of the N smallest X values
    return indices[:N]


def cvtcolor_bgr_to_rgb(image_bgr: torch.Tensor) -> torch.Tensor:
    """Convert BGR image tensor(s) to RGB."""
    if image_bgr.ndim == 3:
        return image_bgr[[2, 1, 0], :, :]
    return image_bgr[:, [2, 1, 0], :, :]


def compute_destination_size_wh(
    img: torch.Tensor, homography_matrix: torch.Tensor
) -> Tuple[int, int]:
    """Compute destination width/height after applying a homography.

    @param img: Source image tensor (H, W or B, C, H, W).
    @param homography_matrix: 3×3 homography matrix (torch or NumPy).
    @return: Tuple ``(new_width, new_height)``.
    """
    width = image_width(img)
    height = image_height(img)
    corners = np.array(
        [
            [0, 0],  # Top-left corner
            [width, 0],  # Top-right corner
            [width, height],  # Bottom-right corner
            [0, height],  # Bottom-left corner
        ],
        dtype="float32",
    )

    # Reshape for perspectiveTransform
    corners = corners.reshape(-1, 1, 2)

    # Apply homography
    if isinstance(homography_matrix, torch.Tensor):
        homography_matrix = homography_matrix.cpu().numpy()
    transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)

    # Calculate the bounding box of the transformed corners
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]

    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    # Compute the dimensions of the bounding box
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    return new_width, new_height


def calculate_control_points(
    image0: Union[str, Path, np.ndarray, torch.Tensor],
    image1: Union[str, Path, np.ndarray, torch.Tensor],
    max_control_points: int,
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    output_directory: Optional[str] = None,
    matcher: str = "superpoint-lightglue",
) -> Dict[str, torch.Tensor]:
    """Compute control points for a pair of images with a selected matcher.

    @param image0: First image (path or tensor).
    @param image1: Second image (path or tensor).
    @param max_control_points: Maximum number of matched points to keep.
    @param device: Optional PyTorch device for inference.
    @param max_num_keypoints: Maximum raw detector keypoints per image.
    @param output_directory: Optional directory for match visualizations.
    @param matcher: ``superpoint-lightglue``, ``dedode-lightglue``, or ``loftr``.
    @return: Dict containing tensors ``m_kpts0`` and ``m_kpts1`` (Nx2).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = normalize_control_point_matcher(matcher)
    max_control_points = int(max_control_points)
    if max_control_points < 4:
        raise ValueError("max_control_points must be at least four")
    image0_tensor = _image_to_rgb_tensor(image0).to(device)
    image1_tensor = _image_to_rgb_tensor(image1).to(device)

    with torch.inference_mode():
        if matcher == "superpoint-lightglue":
            m_kpts0, m_kpts1 = _match_superpoint_lightglue(
                image0_tensor, image1_tensor, device, max_num_keypoints
            )
        elif matcher == "dedode-lightglue":
            m_kpts0, m_kpts1 = _match_dedode_lightglue(
                image0_tensor, image1_tensor, device, max_num_keypoints
            )
        else:
            m_kpts0, m_kpts1 = _match_loftr(image0_tensor, image1_tensor, device)

    if m_kpts0.shape[0] < 4:
        raise RuntimeError(
            f"{matcher} found {m_kpts0.shape[0]} matches; at least four are required"
        )
    indices = select_evenly_spaced(m_kpts0, max_control_points)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    if output_directory:
        _save_match_visualizations(
            image0_tensor,
            image1_tensor,
            m_kpts0,
            m_kpts1,
            output_directory,
        )
    m_kpts0 = m_kpts0.detach().cpu()
    m_kpts1 = m_kpts1.detach().cpu()
    del image0_tensor
    del image1_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    control_points = dict(m_kpts0=m_kpts0, m_kpts1=m_kpts1)
    return control_points


if __name__ == "__main__":
    results = calculate_control_points(
        image0=f"{os.environ['HOME']}/Videos/ev-sabercats-2/left.png",
        image1=f"{os.environ['HOME']}/Videos/ev-sabercats-2/right.png",
        device=torch.device("cuda", 0),
        output_directory=".",
        max_control_points=240,
    )
    print("Done.")
