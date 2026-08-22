"""Helpers for reading/writing Hugin PTO files and homographies.

This module parses PTO control points and image transforms, optionally
replaces or augments them using :func:`calculate_control_points`, and
provides utilities for building homography matrices and applying them.
"""

import math
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.log import get_logger
from hmlib.stitching.control_points import calculate_control_points

_CONTROL_POINTS_LINE = "# control points"


def load_pto_file(file_path: str) -> List[str]:
    """Load the content of a PTO file into a list of lines."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    # trim trailing whitespace
    for i, line in enumerate(lines):
        lines[i] = line.rstrip()
    return lines


def parse_pto_content(lines: List[str]) -> Dict[str, str]:
    """Parse PTO content to a simple ``key=value`` dict."""
    parsed_data: Dict[str, str] = {}
    for line in lines:
        if line.startswith("#"):  # Skip comments
            continue
        key_value_pair = line.strip().split("=", 1)
        if len(key_value_pair) == 2:
            key, value = key_value_pair
            parsed_data[key] = value
    return parsed_data


def save_pto_file(file_path: str, data: List[str]):
    """Save modified PTO file content back to disk."""
    with open(file_path, "w") as file:
        for line in data:
            file.write(f"{line}\n")


def remove_control_points(lines: List[str]) -> Tuple[List[str], int]:
    """Remove existing control-point lines from a PTO file.

    @param lines: PTO file lines.
    @return: Tuple ``(new_lines, prev_control_point_count)``.
    """
    prev_control_point_count: int = 0
    new_lines: List[str] = []
    for line in lines:
        if line.startswith(_CONTROL_POINTS_LINE):
            continue
        if line.startswith("c "):
            prev_control_point_count += 1
            continue
        new_lines.append(line)

    return (new_lines, prev_control_point_count)


def strip(s: str) -> str:
    return re.sub(r"\s+", "", s)


def split_string_with_letter_prefix(s):
    # Initialize the index for where non-letter characters start
    index = 0
    # Loop through each character in the string
    for char in s:
        # Check if the character is a letter
        if char.isalpha():
            index += 1
        else:
            # Stop the loop once a non-letter is found
            break
    # Split the string into letters and the rest
    letters = s[:index]
    rest = s[index:]
    return letters, rest


def extract_prefix_map(tokens: Union[List[str], str]) -> OrderedDict[str, str]:
    if isinstance(tokens, str):
        tokens = tokens.split(" ")
    results: Dict[str, str] = OrderedDict()
    for token in tokens:
        if not token:
            continue
        key, value = split_string_with_letter_prefix(token)
        results[key] = value
    return results


def parse_hugin_control_points(lines: List[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    points0: List[Tuple[float, float]] = []
    points1: List[Tuple[float, float]] = []
    for line in lines:
        if line.startswith("c "):
            tokens = line.split(" ")
            tokens = [strip(t) for t in tokens]
            control_point = extract_prefix_map(tokens)
            pt0 = torch.tensor(
                [float(control_point["x"]), float(control_point["y"])], dtype=torch.float
            )
            pt1 = torch.tensor(
                [float(control_point["X"]), float(control_point["Y"])], dtype=torch.float
            )
            points0.append(pt0)
            points1.append(pt1)
    if points0:
        assert points1
        return torch.stack(points0), torch.stack(points1)
    return None


def configure_control_points(
    project_file_path: str,
    image0: str,
    image1: str,
    max_control_points: int,
    force: bool = False,
    output_directory: Optional[str] = None,
    use_hugin: bool = False,
    matcher: str = "superpoint-lightglue",
) -> Dict[str, torch.Tensor]:
    """Populate or update control points in a Hugin PTO project.

    If existing control points are present and ``use_hugin`` is true,
    they are reused unless ``force`` is set. Otherwise, control points are
    computed with the selected learned matcher and written back to the PTO file.

    @param project_file_path: Path to PTO project file.
    @param image0: Left image filename.
    @param image1: Right image filename.
    @param max_control_points: Cap on number of control points to keep.
    @param force: If True, always recompute control points.
    @param output_directory: Optional output dir for debug visualizations.
    @param use_hugin: If True, prefer Hugin control points when present.
    @param matcher: Learned matcher backend used when points must be computed.
    """
    #  c n0 N1 x5162 y1173 X1416.1875 Y1252.78125 t0
    torch.manual_seed(1)
    pto_file = load_pto_file(project_file_path)
    hugin_ctrl_points = parse_hugin_control_points(pto_file)
    if hugin_ctrl_points is None:
        use_hugin = False
    if hugin_ctrl_points is not None and not force:
        return {
            "m_kpts0": hugin_ctrl_points[0],
            "m_kpts1": hugin_ctrl_points[1],
        }

    control_points = None

    if use_hugin and hugin_ctrl_points is not None:
        m_kpts0 = hugin_ctrl_points[0]
        m_kpts1 = hugin_ctrl_points[1]
        control_points = dict(m_kpts0=m_kpts0, m_kpts1=m_kpts1)

    if not control_points:
        start = time.time()
        control_points = calculate_control_points(
            output_directory=output_directory,
            image0=image0,
            image1=image1,
            max_control_points=max_control_points,
            matcher=matcher,
        )
        print(f"Calculated control points in {time.time() - start} seconds")

    if use_hugin and hugin_ctrl_points is not None:
        # Don't rewrite if we got them from the Hugin project file.
        return control_points

    pts0 = control_points["m_kpts0"]
    pts1 = control_points["m_kpts1"]
    assert len(pts0) == len(pts1)
    print(f"Found {len(pts0)} control points")
    assert len(pts0) and len(pts1)
    pto_file, _ = remove_control_points(lines=pto_file)
    pto_file.append("")
    pto_file.append(_CONTROL_POINTS_LINE)

    def _to_hugin_decimal(val: str) -> str:
        val = float(val)
        if val == float(int(val)):
            return f"{int(val)}"
        return f"{val:.12f}"

    for i in range(len(pts0)):
        point0 = [float(c) for c in pts0[i]]
        point1 = [float(c) for c in pts1[i]]
        line = f"c n0 N1 x{_to_hugin_decimal(point0[0])} y{_to_hugin_decimal(point0[1])} X{_to_hugin_decimal(point1[0])} Y{_to_hugin_decimal(point1[1])} t0"
        pto_file.append(line)
    save_pto_file(file_path=project_file_path, data=pto_file)
    get_logger(__name__).info("Done with control points")
    return control_points


def parse_pto_transformations(lines: List[str]) -> List[Dict[str, Any]]:
    image_params: List[Dict[str, Any]] = []
    for line in lines:
        if line.startswith("i "):  # Image transformation line
            params: Dict[str, Any] = {}

            # Euler angles (fallback if quaternion missing)
            params["yaw"] = float(re.search(r"y(-?\d+\.?\d*)", line).group(1))  # type: ignore
            params["pitch"] = float(re.search(r"p(-?\d+\.?\d*)", line).group(1))  # type: ignore
            params["roll"] = float(re.search(r"r(-?\d+\.?\d*)", line).group(1))  # type: ignore

            # Field of view & image dimensions
            params["fov"] = float(re.search(r"v(-?\d+\.?\d*)", line).group(1))  # type: ignore
            params["width"] = int(re.search(r"w(\d+)", line).group(1))  # type: ignore
            params["height"] = int(re.search(r"h(\d+)", line).group(1))  # type: ignore

            # Optional translations
            params["TrX"] = float(re.search(r"TrX(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore
            params["TrY"] = float(re.search(r"TrY(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore
            params["TrZ"] = float(re.search(r"TrZ(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore

            # Lens distortion parameters (optional)
            params["b"] = float(re.search(r"b(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore
            params["c"] = float(re.search(r"c(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore
            params["d"] = float(re.search(r"d(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore
            params["e"] = float(re.search(r"e(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore
            params["g"] = float(re.search(r"g(-?\d+\.?\d*)", line).group(1) or 0.0)  # type: ignore

            # Quaternion rotation (if available)
            try:
                params["Ra"] = float(re.search(r"Ra(-?\d+\.?\d*)", line).group(1))  # type: ignore
                params["Rb"] = float(re.search(r"Rb(-?\d+\.?\d*)", line).group(1))  # type: ignore
                params["Rc"] = float(re.search(r"Rc(-?\d+\.?\d*)", line).group(1))  # type: ignore
                params["Rd"] = float(re.search(r"Rd(-?\d+\.?\d*)", line).group(1))  # type: ignore
                params["Re"] = float(re.search(r"Re(-?\d+\.?\d*)", line).group(1))  # type: ignore
                params["use_quaternion"] = True
            except AttributeError:
                params["use_quaternion"] = False

            image_params.append(params)

    return image_params


def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert yaw/pitch/roll Euler angles to a 3×3 rotation matrix."""
    yaw, pitch, roll = map(math.radians, [yaw, pitch, roll])

    Rz: np.ndarray = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )

    Ry: np.ndarray = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )

    Rx: np.ndarray = np.array(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    )

    return Rz @ Ry @ Rx  # Combined rotation matrix


def compute_homography(params: Dict[str, Any]) -> np.ndarray:
    """Compute an approximate homography matrix from PTO parameters."""
    R: np.ndarray = euler_to_rotation_matrix(params["yaw"], params["pitch"], params["roll"])

    # Compute focal length from FOV
    f: float = (0.5 * params["width"]) / math.tan(math.radians(params["fov"]) / 2)

    # Intrinsic camera matrix
    K: np.ndarray = np.array([[f, 0, params["width"] / 2], [0, f, params["height"] / 2], [0, 0, 1]])

    # Approximate homography (assuming Z=0)
    H: np.ndarray = K @ R @ np.linalg.inv(K)

    return H


def compute_output_size(image: np.ndarray, H: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Compute the ideal width and height after applying the homography."""
    height, width = image.shape[:2]

    # Define the four corners of the image
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)

    # Convert to homogeneous coordinates (add a 1 for matrix multiplication)
    corners = np.hstack([corners, np.ones((4, 1))])

    # Transform the corners using the homography matrix
    transformed_corners = H @ corners.T
    transformed_corners /= transformed_corners[2]  # Normalize by third coordinate

    # Get min/max x and y coordinates
    x_min, y_min = np.min(transformed_corners[:2], axis=1)
    x_max, y_max = np.max(transformed_corners[:2], axis=1)

    # Compute new width and height
    new_width = int(np.ceil(x_max - x_min))
    new_height = int(np.ceil(y_max - y_min))

    # Translation matrix to shift transformed image into positive coordinates
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

    return new_width, new_height, translation_matrix


def apply_homography(image_path: str, H: np.ndarray) -> None:
    """Apply homography transformation to an image using OpenCV."""
    image: np.ndarray = cv2.imread(image_path)

    if image is None:
        get_logger(__name__).error("Error: Unable to load image %s", image_path)
        return

    height, width = image.shape[:2]

    # Apply perspective transformation
    transformed_image = cv2.warpPerspective(image, H, (width, height))

    # Save & display the result
    image_path = Path(image_path)
    output_path = str(image_path.parent / f"transformed_{image_path.name}")
    cv2.imwrite(output_path, transformed_image)
    get_logger(__name__).info("Transformed image saved as %s", output_path)

    # Display the image
    cv2.imshow("Warped Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_homography_with_size_compute(image_path: str, H: np.ndarray) -> None:
    """Apply homography transformation to an image using OpenCV with an optimal output size."""
    image_path = Path(image_path)  # Convert to Path object
    image = cv2.imread(str(image_path))

    if image is None:
        get_logger(__name__).error("Error: Unable to load image %s", image_path)
        return

    # Compute the ideal output size
    new_width, new_height, translation_matrix = compute_output_size(image, H)

    # Adjust homography to include translation
    H_adjusted = translation_matrix @ H

    # Apply warping with computed dimensions
    transformed_image = cv2.warpPerspective(image, H_adjusted, (new_width, new_height))

    # Save the result
    output_path = image_path.parent / f"transformed_{image_path.name}"
    cv2.imwrite(str(output_path), transformed_image)
    get_logger(__name__).info("Transformed image saved as %s", output_path)

    # Display the image
    cv2.imshow("Warped Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_control_points(pto_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extracts control points from a Hugin .pto file and returns two sets of points."""
    with open(pto_path, "r") as file:
        lines = file.readlines()

    src_points = []
    dst_points = []

    for line in lines:
        if line.startswith("c "):  # Control point line
            match = re.search(r"x([\d.]+) y([\d.]+) X([\d.]+) Y([\d.]+)", line)
            if match:
                x, y, X, Y = map(float, match.groups())
                src_points.append([x, y])
                dst_points.append([X, Y])

    return np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32)


def compute_homography_from_control_points(pto_path: str) -> np.ndarray:
    """Compute the homography matrix from control points in a Hugin .pto file."""
    src_points, dst_points = extract_control_points(pto_path)

    if len(src_points) < 4:
        raise ValueError("At least 4 control points are required to compute homography.")

    H, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    return H


if __name__ == "__main__":
    # filename = f"{os.environ['HOME']}/Videos/pdp/autooptimiser_out.pto"
    filename = f"{os.environ['HOME']}/Videos/pdp/hm_project.pto"
    H = compute_homography_from_control_points(filename)

    # lines = load_pto_file(filename)
    # content = parse_pto_content(lines)
    # params = parse_pto_transformations(lines)
    # H = compute_homography(params[0])
    get_logger(__name__).info("Homography:\n%s", H)

    # # Compute the ideal output size
    apply_homography(f"{os.environ['HOME']}/Videos/pdp/GX010087.png", H)
    # apply_homography(f"{os.environ['HOME']}/Videos/pdp/GX010087.png", H)
    # apply_homography_with_size_compute(f"{os.environ['HOME']}/Videos/pdp/GX010087.png", H)
    pass
