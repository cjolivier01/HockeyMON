import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from hmlib.config import get_game_config, get_nested_value
from hmlib.hm_opts import hm_opts
from hmlib.utils.image import make_channels_first, make_channels_last


def point_distance(pt0: torch.Tensor, pt1: torch.Tensor) -> torch.Tensor:
    return torch.norm(pt0 - pt1)


def order_points_clockwise(pts: torch.Tensor):
    # Ensure pts is a NumPy array of shape (4, 2)
    pts = pts.to(torch.float32).cpu().numpy()

    # Compute the sum and difference of the points.
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Allocate an array for the ordered points: [top-left, top-right, bottom-right, bottom-left]
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]  # top-left: smallest sum
    ordered[2] = pts[np.argmax(s)]  # bottom-right: largest sum
    ordered[1] = pts[np.argmin(diff)]  # top-right: smallest difference
    ordered[3] = pts[np.argmax(diff)]  # bottom-left: largest difference

    return torch.from_numpy(ordered)


class Scoreboard(torch.nn.Module):
    def __init__(
        self,
        src_pts: torch.Tensor,
        dest_height: int,
        dest_width: int,
        dtype: torch.dtype,
        clip_box: torch.Tensor | None = None,
        device: torch.device | None = None,
        auto_aspect: bool = True,
        scoreboard_scale: float | None = None,
        resize_filter: str = "nearest",
        sharpen_amount: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if resize_filter not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError(f"Unknown scoreboard resize filter: {resize_filter}")
        if not np.isfinite(sharpen_amount) or not 0 <= sharpen_amount <= 4:
            raise ValueError("Scoreboard sharpening must be between 0 and 4")
        self._resize_filter = resize_filter
        self._sharpen_amount = sharpen_amount
        if not isinstance(src_pts, torch.Tensor):
            src_pts = torch.tensor(src_pts, dtype=torch.float)
        assert len(src_pts) == 4
        # Check for all zeros
        if torch.sum(src_pts.to(torch.int64)) == 0:
            assert False
        self._src_pts = src_pts.clone().to(device)

        # Doesn't work as intended
        # self._src_pts = order_points_clockwise(src_pts)

        if clip_box is not None:
            if not isinstance(clip_box, torch.Tensor):
                clip_box = torch.tensor(
                    clip_box, dtype=self._src_pts.dtype, device=self._src_pts.device
                )
            self._src_pts[:] -= clip_box[0:2]

        bbox_tensor = int_bbox(get_bbox(self._src_pts))
        bbox_tuple: tuple[int, int, int, int] = tuple(int(v) for v in bbox_tensor.tolist())
        self._bbox_src = bbox_tuple
        x0, y0, x1, y1 = self._bbox_src

        self._src_pts[:, 0] -= x0
        self._src_pts[:, 1] -= y0

        src_width = x1 - x0
        src_height = y1 - y0

        if auto_aspect:
            #
            # Points expected to be in clockwise order, starting from top-left:
            #
            #  0----------------1
            #  |                |
            #  |                |
            #  3----------------2
            #
            w_across_top = point_distance(self._src_pts[0], self._src_pts[1])
            w_across_bottom = point_distance(self._src_pts[2], self._src_pts[3])
            h_left = point_distance(self._src_pts[3], self._src_pts[0])
            h_right = point_distance(self._src_pts[1], self._src_pts[2])
            w_avg = (w_across_top + w_across_bottom) / 2
            h_avg = (h_left + h_right) / 2
            aspect_ratio = w_avg / h_avg
            dest_width_new = int(float(dest_height) * aspect_ratio)
            dest_height_new = int(float(dest_width) / aspect_ratio)
            # Try whichever one changes the least
            if (
                abs(dest_width - dest_width_new) / dest_width
                < abs(dest_height - dest_height_new) / dest_height
            ):
                # Always thisn one ATM
                dest_width = dest_width_new
            else:
                dest_height = dest_height_new

        if scoreboard_scale is not None:
            dest_width *= scoreboard_scale
            dest_height *= scoreboard_scale

        self._dest_width = int(dest_width)
        self._dest_height = int(dest_height)

        totw = max(self._dest_width, src_width)
        toth = max(self._dest_height, src_height)
        if totw > src_width or toth > src_height:
            ratio_w = totw / src_width
            ratio_h = toth / src_height
            self._dest_w = int(totw)
            self._dest_h = int(toth)
            self._src_pts[:, 0] *= ratio_w
            self._src_pts[:, 1] *= ratio_h
        else:
            self._dest_w = int(src_width)
            self._dest_h = int(src_height)

        dst_pts = torch.tensor(
            [
                # TL
                [0, 0],
                # TR
                [self._dest_width - 1, 0],
                # BR
                [self._dest_width - 1, self._dest_height - 1],
                # BL
                [0, self._dest_height - 1],
            ],
            dtype=torch.float32,
            device=self._src_pts.device,
        )
        # print(src_pts)
        # print(dst_pts)
        perspective_coeffs = _get_perspective_coeffs(startpoints=self._src_pts, endpoints=dst_pts)

        ow = self._dest_w
        oh = self._dest_h

        grid = _perspective_grid(
            perspective_coeffs,
            ow=ow,
            oh=oh,
            dtype=dtype,
            device=torch.device("cpu") if device is None else device,
        )
        # Register as buffer so it moves with the module and stays constant-shape.
        self.register_buffer("_grid", grid, persistent=False)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        # Expect a fixed scoreboard region for this instance; src pts are set at construction.
        original_image = make_channels_first(input_image)

        x0, y0, x1, y1 = self._bbox_src
        src_image = original_image[:, :, y0:y1, x0:x1]

        # Ensure floating point input for interpolation/grid sampling.
        if not torch.is_floating_point(src_image):
            src_image = src_image.to(dtype=self._grid.dtype)

        # Resize cropped region to the fixed working size.
        src_image = F.interpolate(
            src_image,
            size=(self._dest_h, self._dest_w),
            mode=self._resize_filter,
            **({"align_corners": False} if self._resize_filter != "nearest" else {}),
        )

        grid = self._grid
        if grid.device != src_image.device or grid.dtype != src_image.dtype:
            grid = grid.to(device=src_image.device, dtype=src_image.dtype)

        # Apply the precomputed perspective grid; batch dimension is preserved.
        if src_image.shape[0] == 1:
            warped_image = F.grid_sample(
                src_image,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            warped_image = F.grid_sample(
                src_image,
                grid.expand(src_image.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        result = warped_image[:, :, : self._dest_height, : self._dest_width]
        if self._sharpen_amount:
            blur = F.avg_pool2d(F.pad(result, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            result = (result + self._sharpen_amount * (result - blur)).clamp(0, 255)
        return result

    @property
    def width(self):
        return self._dest_width

    @property
    def height(self):
        return self._dest_height


def get_bbox(point_list: torch.Tensor | list[list[float]]):
    if isinstance(point_list, list):
        points = torch.tensor(point_list)
    else:
        points = point_list
    mins = torch.min(points, dim=0)[0]
    maxs = torch.max(points, dim=0)[0]
    return torch.cat((mins, maxs), dim=0)


def int_bbox(bbox: torch.Tensor):
    bbox[:2] = torch.floor(bbox[:2])
    bbox[2:] = torch.ceil(bbox[2:])
    return bbox.to(torch.int32)


def _get_perspective_coeffs(
    startpoints: torch.Tensor | list[list[int]],
    endpoints: torch.Tensor | list[list[int]],
) -> torch.Tensor:
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        Tensor containing (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    if not isinstance(startpoints, torch.Tensor):
        startpoints = torch.tensor(startpoints, dtype=torch.float32)
    else:
        startpoints = startpoints.detach().to(dtype=torch.float32)

    device = startpoints.device

    if not isinstance(endpoints, torch.Tensor):
        endpoints = torch.tensor(endpoints, dtype=torch.float32, device=device)
    else:
        endpoints = endpoints.detach().to(device=device, dtype=torch.float32)

    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float32, device=device)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, 0] = p1[0]
        a_matrix[2 * i, 1] = p1[1]
        a_matrix[2 * i, 2] = 1.0
        a_matrix[2 * i, 6] = -p2[0] * p1[0]
        a_matrix[2 * i, 7] = -p2[0] * p1[1]
        a_matrix[2 * i + 1, 3] = p1[0]
        a_matrix[2 * i + 1, 4] = p1[1]
        a_matrix[2 * i + 1, 5] = 1.0
        a_matrix[2 * i + 1, 6] = -p2[1] * p1[0]
        a_matrix[2 * i + 1, 7] = -p2[1] * p1[1]

    b_matrix = startpoints.view(8)
    if device.type == "cpu":
        raise RuntimeError("Scoreboard perspective coefficients must be computed on a GPU tensor")

    return torch.linalg.solve(a_matrix, b_matrix)


def _perspective_grid(
    coeffs: torch.Tensor | list[float],
    ow: int,
    oh: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    if not isinstance(coeffs, torch.Tensor):
        coeffs = torch.tensor(coeffs, dtype=dtype, device=device)
    else:
        coeffs = coeffs.to(dtype=dtype, device=device)
    theta1 = torch.stack((coeffs[:3], coeffs[3:6])).unsqueeze(0)
    row2 = torch.stack((coeffs[6], coeffs[7], coeffs.new_tensor(1.0)))
    theta2 = torch.stack((row2, row2)).unsqueeze(0)

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh, device=device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor(
        [0.5 * ow, 0.5 * oh], dtype=dtype, device=device
    )
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)


def _cast_squeeze_in(
    img: Tensor, req_dtypes: list[torch.dtype]
) -> tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype
) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def _apply_grid_transform(
    img: Tensor, grid: Tensor, mode: str, fill: float | list[float] | None
) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        mask = torch.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]),
            dtype=img.dtype,
            device=img.device,
        )
        img = torch.cat((img, mask), dim=1)

    img = F.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        fill_list, len_fill = (
            (fill, len(fill)) if isinstance(fill, (tuple, list)) else ([float(fill)], 1)
        )
        fill_img = (
            torch.tensor(fill_list, dtype=img.dtype, device=img.device)
            .view(1, len_fill, 1, 1)
            .expand_as(img)
        )
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def sb_main(game_id: str):

    image_path = f"{os.environ['HOME']}/Videos/{game_id}/s.png"  # Specify the path to your image
    image = cv2.imread(image_path)
    image_tensor = make_channels_first(torch.from_numpy(image).unsqueeze(0))

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / ".." / ".." / "..")
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    selected_points = get_nested_value(game_config, "rink.scoreboard.perspective_polygon")

    if selected_points:
        for pt in selected_points:
            print(pt)

        scoreboard = Scoreboard(
            src_pts=selected_points,
            dest_width=700,
            dest_height=300,
            dtype=(image_tensor.dtype if torch.is_floating_point(image_tensor) else torch.float),
        )

        warped_image = scoreboard.forward(image_tensor.to(torch.float) / 255)

        warped_image = torch.clamp(warped_image * 255, min=0, max=255).to(torch.uint8)

        cv2.imshow("warped_image", make_channels_last(warped_image)[0].numpy())
        cv2.waitKey(0)


if __name__ == "__main__":
    # main()
    opts = hm_opts()
    args = opts.parse()
    sb_main(args.game_id)
