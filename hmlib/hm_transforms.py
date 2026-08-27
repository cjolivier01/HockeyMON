"""Custom image and tensor transforms used in HockeyMON pipelines.

Extends MMDetection/MMPose-style transforms with GPU-aware resizing,
stream-compatible interpolation and mixed tensor/NumPy handling.
"""

import math
import numbers
import time
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmcv.image.geometric import _scale_size
from mmcv.transforms import LoadImageFromFile
from mmengine.registry import TRANSFORMS
from mmengine.utils import to_2tuple
from mmpose.structures.bbox.transforms import bbox_xywh2cs, get_warp_matrix

from hmlib.utils.gpu import StreamTensor, StreamTensorBase
from hmlib.utils.image import (
    image_height,
    image_width,
    is_channels_first,
    make_channels_first,
    make_channels_last,
)

try:
    from PIL import Image
except ImportError:
    Image = None


cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}


def _get_results_profiler(results: Any) -> Optional[Any]:
    """Extract a profiler instance from a results dict if present."""
    if not isinstance(results, dict):
        return None
    profiler = results.get("profiler") or results.get("_profiler")
    if profiler is None:
        data_samples = results.get("data_samples")
        profiler = getattr(data_samples, "profiler", None)
    return profiler


def _transform_profile_scope(results: Any, name: str):
    """Return a torch profiling context manager for a transform call."""
    label = f"transform.{name}"
    profiler = _get_results_profiler(results)
    if profiler is not None:
        try:
            if getattr(profiler, "enabled", True):
                return profiler.rf(label)
        except Exception:
            # Fall through to torch.autograd profiler path
            pass

    enabled_flag = getattr(torch.autograd.profiler, "_is_profiler_enabled", False)
    try:
        enabled = bool(enabled_flag()) if callable(enabled_flag) else bool(enabled_flag)
    except Exception:
        enabled = False
    if enabled:
        try:
            return torch.autograd.profiler.record_function(label)
        except Exception:
            return nullcontext()
    return nullcontext()


def get_affine_transform(center, scale, rot, output_size):
    """Lightweight wrapper to compute an affine transform matrix."""
    center_np = np.array(center, dtype=np.float32)
    scale_np = np.array(scale, dtype=np.float32)
    if scale_np.shape == ():
        scale_np = np.array([scale_np, scale_np], dtype=np.float32)
    return get_warp_matrix(center_np, scale_np, rot, output_size)


def affine_transform(point, mat):
    """Apply affine transform to a single point."""
    pt = np.array([point[0], point[1], 1.0], dtype=np.float32)
    return np.dot(mat, pt)[:2]


def warp_affine_joints(joints: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply affine transform to an array of joints."""
    ones = np.ones((joints.shape[0], 1), dtype=joints.dtype)
    joints_h = np.concatenate([joints, ones], axis=1)
    warped = (mat @ joints_h.T).T
    return warped[:, :2]


PIPELINE_SUBSTITUTIONS: Dict[str, str] = {
    "TopDownAffine": "HmTopDownAffine",
    "ToTensor": "HmToTensor",
}


def update_data_pipeline(
    pipeline: List[Dict[str, Any]],
    substitutions: Dict[str, str] = PIPELINE_SUBSTITUTIONS,
) -> List[Dict[str, Any]]:
    for i, item in enumerate(pipeline):
        # TODO: make a simple translatiuon function and array argument
        type_name = item["type"]
        if type_name in substitutions:
            pipeline[i]["type"] = substitutions[type_name]
    return pipeline


def extract_subimage(img: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """Extract a subimage from an image tensor based on a bounding box.

    @param img: Source image tensor of shape ``(C, H, W)`` or ``(H, W, C)``.
    @param bbox: Bounding box tensor with values ``(top, left, bottom, right)``.
    @return: Extracted subimage tensor.
    """
    left, top, right, bottom = bbox
    # Ensure coordinates are within image dimensions
    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, image_height(img) - 1)
    right = min(right, image_width(img) - 1)

    # Extract the region of the image corresponding to the bounding box
    if is_channels_first(img):
        subimage = img[:, top:bottom, left:right]
    else:
        subimage = img[top:bottom, left:right, :]

    return subimage


def hm_imrescale(
    img: np.ndarray,
    scale: Union[float, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray, Tuple[Union[np.ndarray, torch.Tensor], float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    if img.ndim == 3:
        h, w = img.shape[:2]
    else:
        assert img.ndim == 4 and img.shape[-1] in (3, 4)
        h, w = img.shape[1:3]
    new_size, scale_factor = mmcv.rescale_size((w, h), scale, return_scale=True)
    rescaled_img = hm_imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def hm_imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    # Use helpers to robustly get height/width for both numpy and torch tensors
    orig_h, orig_w = image_height(img), image_width(img)
    if backend is None:
        backend = mmcv.imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    target_w, target_h = size

    # Pure PyTorch path for tensors (no torchvision or Pillow codes)
    if isinstance(img, torch.Tensor) or isinstance(img, StreamTensorBase):
        tensor_stream = isinstance(img, StreamTensorBase)
        tensor = img.wait() if tensor_stream else img
        ndim = tensor.ndim

        # Normalize to NCHW
        if ndim == 4:
            # Heuristic: NHWC if last dim is channel-like
            is_nhwc = tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4)
            tensor_nchw = tensor.permute(0, 3, 1, 2) if is_nhwc else tensor
            restore = "nhwc" if is_nhwc else "nchw"
        elif ndim == 3:
            # Use helper for 3D images
            if is_channels_first(tensor):
                tensor_nchw = tensor.unsqueeze(0)
                restore = "chw"
            else:
                tensor_nchw = tensor.permute(2, 0, 1).unsqueeze(0)
                restore = "hwc"
        else:
            raise ValueError(f"Unsupported tensor ndim={ndim} for resize")

        # Interpolate requires floating point; keep original dtype to restore
        orig_dtype = tensor_nchw.dtype
        if not torch.is_floating_point(tensor_nchw):
            tensor_nchw = tensor_nchw.to(torch.float32)

        # Map interpolation names to torch modes; fall back where unsupported
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "area": "area",
            # Lanczos not supported by torch.interpolate; use bicubic as a high-quality fallback
            "lanczos": "bicubic",
        }
        if interpolation not in mode_map:
            raise ValueError(f"Unsupported interpolation: {interpolation}")
        mode = mode_map[interpolation]

        kwargs: Dict[str, Any] = {"size": (target_h, target_w), "mode": mode}
        if mode in ("bilinear", "bicubic"):
            # align_corners=False is standard for image resizing
            kwargs["align_corners"] = False

        tensor_resized = torch.nn.functional.interpolate(tensor_nchw, **kwargs)

        # Restore dtype if needed (clamp for integer ranges typical of images)
        if orig_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            tensor_resized = tensor_resized.clamp(0, 255).to(orig_dtype)

        # Restore original layout
        if restore == "nhwc":
            img = tensor_resized.permute(0, 2, 3, 1)
        elif restore == "nchw":
            img = tensor_resized
        elif restore == "hwc":
            img = tensor_resized.squeeze(0).permute(1, 2, 0)
        elif restore == "chw":
            img = tensor_resized.squeeze(0)
        else:
            # Should not reach here
            img = tensor_resized

        if tensor_stream:
            img = StreamTensor(img)

    else:
        # Numpy path: keep fast cv2 implementation; do not use Pillow
        if backend == "pillow":
            # Avoid Pillow; emulate via cv2 for numpy arrays
            img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
        else:
            img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])

    if not return_scale:
        return img
    else:
        w_scale = target_w / float(orig_w)
        h_scale = target_h / float(orig_h)
        return img, w_scale, h_scale


def hm_imresize_to_multiple(
    img: np.ndarray,
    divisor: Union[int, Tuple[int, int]],
    size: Union[int, Tuple[int, int], None] = None,
    scale_factor: Union[float, Tuple[float, float], None] = None,
    keep_ratio: bool = False,
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image according to a given size or scale factor and then rounds
    up the the resized or rescaled image size to the nearest value that can be
    divided by the divisor.

    Args:
        img (ndarray): The input image.
        divisor (int | tuple): Resized image size will be a multiple of
            divisor. If divisor is a tuple, divisor should be
            (w_divisor, h_divisor).
        size (None | int | tuple[int]): Target size (w, h). Default: None.
        scale_factor (None | float | tuple[float]): Multiplier for spatial
            size. Should match input size if it is a tuple and the 2D style is
            (w_scale_factor, h_scale_factor). Default: None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: False.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = image_height(img), image_width(img)
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is None and scale_factor is None:
        raise ValueError("one of size or scale_factor should be defined")
    elif size is not None:
        size = to_2tuple(size)
        if keep_ratio:
            size = mmcv.rescale_size((w, h), size, return_scale=False)
    else:
        size = _scale_size((w, h), scale_factor)

    divisor = to_2tuple(divisor)
    size = tuple(int(np.ceil(s / d)) * d for s, d in zip(size, divisor))
    resized_img, w_scale, h_scale = hm_imresize(
        img,
        size,
        return_scale=True,
        interpolation=interpolation,
        out=out,
        backend=backend,
    )
    if return_scale:
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def hm_impad(
    img: np.ndarray,
    *,
    shape: Optional[Tuple[int, int]] = None,
    padding: Union[int, tuple, None] = None,
    pad_val: Union[float, List] = 0,
    padding_mode: str = "constant",
) -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        iw = image_width(img)
        ih = image_height(img)
        # padding should be making it larger or not at all
        assert shape[1] >= iw
        assert shape[0] >= ih
        width = max(shape[1] - iw, 0)
        height = max(shape[0] - ih, 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError("pad_val must be a int or a tuple. " f"But received {type(pad_val)}")

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(
            "Padding must be a int or a 2, or 4 element tuple." f"But received {padding}"
        )

    # check padding mode
    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    if isinstance(img, torch.Tensor | StreamTensorBase):
        if img.ndim == 3:
            img = img.permute(2, 0, 1)
        else:
            assert img.ndim == 4
            if img.shape[-1] in (3, 4):
                img = img.permute(0, 3, 1, 2)
        was_stream_tensor = isinstance(img, StreamTensorBase)
        tensor_input = img.wait() if was_stream_tensor else img
        padded = torch.nn.functional.pad(
            tensor_input,
            (padding[0], padding[2], padding[1], padding[3]),
            padding_mode,
            value=pad_val,
        )
        img = StreamTensor(padded) if was_stream_tensor else padded
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        else:
            img = img.permute(0, 2, 3, 1)
    else:
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val,
        )

    return img


def hm_impad_to_multiple(
    img: np.ndarray, divisor: int, pad_val: Union[float, List] = 0
) -> np.ndarray:
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(image_height(img) / divisor)) * divisor
    pad_w = int(np.ceil(image_width(img) / divisor)) * divisor
    return hm_impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


@TRANSFORMS.register_module()
class HmImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys, scale_factor=None, dtype: torch.dtype = torch.float):
        self.keys = keys
        self.scale_factor = scale_factor
        self.dtype = dtype

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        permute the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and permuted to (C, H, W) order.
        """
        with _transform_profile_scope(results, self.__class__.__name__):
            for key in self.keys:
                img = results[key]
                if len(img.shape) < 3:
                    img = img.unsqueeze(0)
                if isinstance(img, torch.Tensor) and self.dtype is not None:
                    # If uint8, convert to float and optionally scale
                    if not torch.is_floating_point(img):
                        assert img.dtype == torch.uint8
                        img = img.to(self.dtype)
                        assert (
                            torch.is_floating_point(img)
                            and "When converting uint8 Tensor to float, please specify the dtype in ToTensor"
                        )
                        if self.scale_factor is not None:
                            img *= self.scale_factor
                    else:
                        # Float tensors: optionally scale (e.g., 0..1 -> 0..255)
                        if self.scale_factor is not None:
                            img * self.scale_factor
                results[key] = img
            return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"


@TRANSFORMS.register_module()
class HmPad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(
        self,
        size=None,
        size_divisor=None,
        pad_to_square=False,
        pad_val=dict(img=0, masks=0, seg=255),
    ):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                "pad_val of float type is deprecated now, "
                f"please use pad_val=dict(img={pad_val}, "
                f"masks={pad_val}, seg=255) instead.",
                DeprecationWarning,
            )
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, (
                "The size and size_divisor must be None " "when pad2square is True"
            )
        else:
            assert (
                size is not None or size_divisor is not None
            ), "only one of size and size_divisor should be valid"
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get("img", 0)
        for key in results.get("img_fields", ["img"]):
            if self.pad_to_square:
                max_size = max([image_width(results[key]), image_height(results[key])])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = hm_impad(results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = hm_impad_to_multiple(results[key], self.size_divisor, pad_val=pad_val)
            else:
                continue
            # Currently only support one for this padding stuff
            assert "pre_pad_shape" not in results
            results["pre_pad_shape"] = [image_height(results[key]), image_width(results[key])]
            results[key] = padded_img
        results["pad_shape"] = [image_height(padded_img), image_width(padded_img)]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results["pad_shape"][:2]
        pad_val = self.pad_val.get("masks", 0)
        for key in results.get("mask_fields", []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get("seg", 255)
        for key in results.get("seg_fields", []):
            results[key] = hm_impad(results[key], shape=results["pad_shape"][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        with _transform_profile_scope(results, self.__class__.__name__):
            self._pad_img(results)
            self._pad_masks(results)
            self._pad_seg(results)
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_to_square={self.pad_to_square}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@TRANSFORMS.register_module()
class HmResize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        interpolation="bilinear",
        override=False,
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            # Accept YAML-loaded lists and normalize to a list of tuples
            if isinstance(img_scale, list):
                # if provided as [w, h], treat as single tuple
                if len(img_scale) == 2 and all(not isinstance(x, (list, tuple)) for x in img_scale):
                    self.img_scale = [tuple(img_scale)]
                else:
                    self.img_scale = [tuple(s) if isinstance(s, list) else s for s in img_scale]
            elif isinstance(img_scale, tuple):
                self.img_scale = [img_scale]
            else:
                # last resort: wrap as single tuple if possible
                try:
                    self.img_scale = [tuple(img_scale)]
                except Exception:
                    self.img_scale = [img_scale]
            assert all(
                isinstance(s, tuple) and len(s) == 2 for s in self.img_scale
            ), "img_scale must be tuple(s)"

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmengine.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmengine.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get("img_fields", ["img"]):
            if self.keep_ratio:
                img, scale_factor = hm_imrescale(
                    results[key],
                    results["scale"],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                result_img = results[key]
                new_h, new_w = [image_height(img), image_width(img)]
                h, w = [image_height(result_img), image_width(result_img)]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = hm_imresize(
                    results[key],
                    results["scale"],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
            results[key] = img
            batch_size = img.shape[0]
            scale_factor = torch.tensor([w_scale, h_scale], dtype=torch.float32).to(
                img.device, non_blocking=False
            )
            iw, ih = image_width(img), image_height(img)
            results["img_shape"] = [(ih, iw) for _ in range(batch_size)]
            # in case that there is no padding
            results["pad_shape"] = results["img_shape"].copy()
            results["scale_factor"] = [scale_factor for _ in range(batch_size)]
            results["keep_ratio"] = [self.keep_ratio for _ in range(batch_size)]

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get("bbox_fields", []):
            bboxes = results[key] * results["scale_factor"]
            if self.bbox_clip_border:
                img_shape = results["img_shape"]
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get("mask_fields", []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results["scale"])
            else:
                results[key] = results[key].resize(results["img_shape"][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get("seg_fields", []):
            if self.keep_ratio:
                gt_seg = hm_imrescale(
                    results[key],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            else:
                gt_seg = hm_imresize(
                    results[key],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        with _transform_profile_scope(results, self.__class__.__name__):
            if "scale" not in results:
                if "scale_factor" in results:
                    img_shape = results["img"].shape[:2]
                    scale_factor = results["scale_factor"]
                    assert isinstance(scale_factor, float)
                    results["scale"] = tuple([int(x * scale_factor) for x in img_shape][::-1])
                else:
                    self._random_scale(results)
            else:
                if not self.override:
                    assert (
                        "scale_factor" not in results
                    ), "scale and scale_factor cannot be both set."
                else:
                    results.pop("scale")
                    if "scale_factor" in results:
                        results.pop("scale_factor")
                    self._random_scale(results)

            self._resize_img(results)
            self._resize_bboxes(results)
            self._resize_masks(results)
            self._resize_seg(results)
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


@TRANSFORMS.register_module()
class HmImageColorAdjust:
    """Apply optional color adjustments on images.

    No-op unless one or more adjustments are specified.

    Supported adjustments (all optional):
    - white_balance: list[float] of length 3, per-channel RGB gains
    - brightness: float, multiplicative gain (>1 brighter)
    - exposure_ev: float, exposure compensation in stops (e.g., +1.0 doubles brightness)
    - contrast: float, contrast factor (>1 more contrast)
    - gamma: float, gamma exponent (>1 darker)
    - config_ref: optional dict-like object containing runtime values; if provided,
      values are refreshed from ``config_paths`` before each call.

    Args:
        keys (list[str]): Result dict keys to process (default: ["img"]).
        white_balance (list[float] | None): Per-channel RGB gains.
        brightness (float | None): Multiplicative brightness.
        exposure_ev (float | None): Exposure compensation in stops (EV).
        contrast (float | None): Contrast factor.
        gamma (float | None): Gamma exponent applied to normalized [0, 1].
        config_ref (dict | None): Live config dict to read overrides from.
        config_paths (list[list[str]] | list[tuple[str]] | None): Ordered list of
            nested paths to search within ``config_ref`` for color settings. Defaults
            to ``[('rink','camera','color'), ('rink','camera')]`` when a
            ``config_ref`` is provided.
        refresh_from_config (bool): If True, pull values from ``config_ref`` on each
            call (noop when ``config_ref`` is ``None``).
    """

    _NOT_PROVIDED = object()

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        white_balance: Optional[List[float]] = None,
        white_balance_temp: Optional[Union[float, str]] = None,
        brightness: Optional[float] = None,
        exposure_ev: Optional[float] = None,
        contrast: Optional[float] = None,
        gamma: Optional[float] = None,
        config_ref: Optional[Dict[str, Any]] = None,
        config_paths: Optional[List[Union[List[str], Tuple[str, ...], str]]] = None,
        refresh_from_config: bool = True,
    ):
        self.keys = keys or ["img"]
        # If a Kelvin temperature is provided, convert to per-channel gains.
        if white_balance is None and white_balance_temp is not None:
            self.white_balance = self._gains_from_kelvin(white_balance_temp)
        else:
            self.white_balance = white_balance
        self.brightness = brightness
        self.exposure_ev = exposure_ev
        self.contrast = contrast
        self.gamma = gamma
        self._refresh_from_config_enabled = bool(refresh_from_config)
        # Default search paths if a config reference is provided.
        if config_ref is not None and config_paths is None:
            config_paths = [("rink", "camera", "color"), ("rink", "camera")]
        self.config_ref = config_ref
        self._config_paths: List[Tuple[str, ...]] = self._normalize_paths(config_paths)

        # Validate white balance gains if provided
        if self.white_balance is not None:
            assert isinstance(self.white_balance, (list, tuple)) and len(self.white_balance) == 3

    @staticmethod
    def _normalize_paths(
        config_paths: Optional[List[Union[List[str], Tuple[str, ...], str]]],
    ) -> List[Tuple[str, ...]]:
        if not config_paths:
            return []
        normalized: List[Tuple[str, ...]] = []
        for path in config_paths:
            if isinstance(path, str):
                parts = [p for p in path.replace("/", ".").split(".") if p]
                if parts:
                    normalized.append(tuple(parts))
            elif isinstance(path, (list, tuple)):
                tup = tuple(path)
                if tup:
                    normalized.append(tup)
        return normalized

    @staticmethod
    def _get_nested_dict(cfg: Any, path: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        cur = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur if isinstance(cur, dict) else None

    def _pick_first(self, sources: List[Dict[str, Any]], keys: Tuple[str, ...]):
        for src in sources:
            for key in keys:
                if key in src:
                    return src[key]
        return self._NOT_PROVIDED

    def _set_scalar_from_config(self, attr: str, value: Any):
        if value is self._NOT_PROVIDED:
            return
        try:
            setattr(self, attr, None if value is None else float(value))
        except Exception:
            # Leave existing value if conversion fails
            pass

    def _refresh_from_config(self) -> None:
        if not self._refresh_from_config_enabled or self.config_ref is None:
            return
        cfg_dict: Optional[Dict[str, Any]] = None
        for path in self._config_paths:
            cfg_dict = self._get_nested_dict(self.config_ref, path)
            if cfg_dict is not None:
                break
        if cfg_dict is None and isinstance(self.config_ref, dict):
            cfg_dict = self.config_ref
        if not isinstance(cfg_dict, dict):
            return
        sources: List[Dict[str, Any]] = []
        color_sub = cfg_dict.get("color") if isinstance(cfg_dict, dict) else None
        if isinstance(color_sub, dict):
            sources.append(color_sub)
        sources.append(cfg_dict)
        wb_temp = self._pick_first(sources, ("white_balance_temp", "white_balance_k"))
        wb = self._pick_first(sources, ("white_balance",))
        brightness = self._pick_first(sources, ("brightness", "color_brightness"))
        exposure_ev = self._pick_first(sources, ("exposure_ev",))
        contrast = self._pick_first(sources, ("contrast", "color_contrast"))
        gamma = self._pick_first(sources, ("gamma", "color_gamma"))

        if wb_temp is not self._NOT_PROVIDED:
            if wb_temp is None:
                self.white_balance = None
            else:
                try:
                    self.white_balance = self._gains_from_kelvin(wb_temp)
                except Exception:
                    pass
        elif wb is not self._NOT_PROVIDED:
            if wb is None:
                self.white_balance = None
            else:
                try:
                    if isinstance(wb, (list, tuple)) and len(wb) == 3:
                        self.white_balance = [float(x) for x in wb]
                except Exception:
                    pass

        self._set_scalar_from_config("brightness", brightness)
        self._set_scalar_from_config("exposure_ev", exposure_ev)
        self._set_scalar_from_config("contrast", contrast)
        self._set_scalar_from_config("gamma", gamma)

    @staticmethod
    def _isclose(a, b, atol=1e-6):
        # Fast isclose for float or tensor
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            return abs(a - b) <= atol
        if hasattr(a, "device"):
            assert False  # This will cause a sync
            return (torch.abs(torch.tensor(a) - torch.tensor(b)) <= atol).all().item()
        return np.isclose(a, b, atol=atol).all() if hasattr(a, "__iter__") else abs(a - b) <= atol

    def _has_any_adjustment(self) -> bool:
        # Only return True if any adjustment is non-identity
        if self.white_balance is not None and not self._isclose(
            self.white_balance, [1.0, 1.0, 1.0]
        ):
            return True
        if self.brightness is not None and not self._isclose(self.brightness, 1.0):
            return True
        if self.exposure_ev is not None and not self._isclose(self.exposure_ev, 0.0):
            return True
        if self.contrast is not None and not self._isclose(self.contrast, 1.0):
            return True
        if self.gamma is not None and not self._isclose(self.gamma, 1.0):
            return True
        return False

    @staticmethod
    def _apply_exposure_ev(img: torch.Tensor, ev: float) -> torch.Tensor:
        if ev is None or ev == 0.0:
            return img
        try:
            gain = float(2.0 ** float(ev))
        except Exception:
            return img
        return HmImageColorAdjust._apply_brightness(img, gain)

    @staticmethod
    def _apply_white_balance(img: torch.Tensor, gains: List[float]) -> torch.Tensor:
        # img expected: NCHW or CHW float or uint8
        dtype = img.dtype
        if not torch.is_floating_point(img):
            img = img.to(torch.float16)
        if img.ndim == 3:
            g = (
                torch.tensor(gains, dtype=img.dtype)
                .to(device=img.device, non_blocking=True)
                .view(3, 1, 1)
            )
        else:
            g = (
                torch.tensor(gains, dtype=img.dtype)
                .to(device=img.device, non_blocking=True)
                .view(1, 3, 1, 1)
            )
        img = img * g
        # Clamp to 0..255 if original type was integer-like or float in 0..255
        img = img.clamp(0.0, 255.0)
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            img = img.to(dtype)
        return img

    @staticmethod
    def _apply_brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
        if factor is None or factor == 1.0:
            return img
        dtype = img.dtype
        if not torch.is_floating_point(img):
            img = img.to(torch.float16)
        img = img * float(factor)
        img = img.clamp(0.0, 255.0)
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            img = img.to(dtype)
        return img

    @staticmethod
    def _apply_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
        if factor is None or factor == 1.0:
            return img
        dtype = img.dtype
        if not torch.is_floating_point(img):
            img = img.to(torch.float16)
        # Compute per-image mean intensity and scale around it
        if img.ndim == 3:
            # CHW
            mean_val = img.mean(dim=(1, 2), keepdim=True)
        else:
            # NCHW
            mean_val = img.mean(dim=(2, 3), keepdim=True)
        img = (img - mean_val) * float(factor) + mean_val
        img = img.clamp(0.0, 255.0)
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            img = img.to(dtype)
        return img

    @staticmethod
    def _apply_gamma(img: torch.Tensor, gamma: float) -> torch.Tensor:
        if gamma is None or gamma == 1.0:
            return img
        dtype = img.dtype
        if not torch.is_floating_point(img):
            img = img.to(torch.float16)
        # Normalize to [0,1] assuming 0..255 images, apply gamma, then scale back
        img01 = img / 255.0
        img01.clamp_(0.0, 1.0)
        img01.pow_(float(gamma))
        img01.mul_(255.0)
        img01.clamp_(0.0, 255.0)
        # img = (img01 * 255.0).clamp(0.0, 255.0)
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            img = img.to(dtype)
        return img

    def _adjust_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # Only convert dtype/layout if any adjustment is non-identity
        need_adjust = self._has_any_adjustment()
        if not need_adjust:
            return t
        icf = is_channels_first(t)
        orig_dtype = t.dtype
        # Only convert once if needed
        if not icf:
            t = make_channels_first(t)
        if not torch.is_floating_point(t):
            t = t.to(torch.float16)
        # Apply adjustments
        if self.white_balance is not None and not self._isclose(
            self.white_balance, [1.0, 1.0, 1.0]
        ):
            t = HmImageColorAdjust._apply_white_balance(t, self.white_balance)
        if self.exposure_ev is not None and not self._isclose(self.exposure_ev, 0.0):
            t = HmImageColorAdjust._apply_exposure_ev(t, self.exposure_ev)
        if self.brightness is not None and not self._isclose(self.brightness, 1.0):
            t = HmImageColorAdjust._apply_brightness(t, self.brightness)
        if self.contrast is not None and not self._isclose(self.contrast, 1.0):
            t = HmImageColorAdjust._apply_contrast(t, self.contrast)
        if self.gamma is not None and not self._isclose(self.gamma, 1.0):
            t = HmImageColorAdjust._apply_gamma(t, self.gamma)
        # Restore layout and dtype if needed
        if not icf:
            t = make_channels_last(t)
        if t.dtype != orig_dtype:
            t = t.to(orig_dtype)
        return t

    def _adjust_numpy(self, a: np.ndarray) -> np.ndarray:
        icf = is_channels_first(a)
        if not icf:
            a = make_channels_first(a)
        # Convert to float for ops
        orig_dtype = a.dtype
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32)
        if self.white_balance is not None:
            gains = np.array(self.white_balance, dtype=a.dtype).reshape(3, 1, 1)
            if a.ndim == 4:
                gains = gains.reshape(1, 3, 1, 1)
            a = a * gains
        if self.exposure_ev is not None and self.exposure_ev != 0.0:
            try:
                a = a * float(2.0 ** float(self.exposure_ev))
            except Exception:
                pass
        if self.brightness is not None and self.brightness != 1.0:
            a = a * float(self.brightness)
        if self.contrast is not None and self.contrast != 1.0:
            if a.ndim == 3:
                mean_val = a.mean(axis=(1, 2), keepdims=True)
            else:
                mean_val = a.mean(axis=(2, 3), keepdims=True)
            a = (a - mean_val) * float(self.contrast) + mean_val
        if self.gamma is not None and self.gamma != 1.0:
            a01 = np.clip(a / 255.0, 0.0, 1.0)
            a01 = np.power(a01, float(self.gamma))
            a = a01 * 255.0
        a = np.clip(a, 0.0, 255.0)
        if orig_dtype != a.dtype:
            a = a.astype(orig_dtype)
        if not icf:
            a = make_channels_last(a)
        return a

    def __call__(self, results):
        with _transform_profile_scope(results, self.__class__.__name__):
            self._refresh_from_config()
            if not self._has_any_adjustment():
                return results
            for key in self.keys:
                if key not in results:
                    continue
                img = results[key]
                if isinstance(img, (torch.Tensor, StreamTensorBase)):
                    was_stream_tensor = isinstance(img, StreamTensorBase)
                    tensor_input = img.wait() if was_stream_tensor else img
                    adjusted = self._adjust_tensor(tensor_input)
                    if was_stream_tensor:
                        adjusted = StreamTensor(adjusted)
                    results[key] = adjusted
                else:
                    # numpy array
                    results[key] = self._adjust_numpy(img)
            return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(keys={self.keys}, white_balance={self.white_balance}, "
            f"brightness={self.brightness}, contrast={self.contrast}, gamma={self.gamma})"
        )

    @staticmethod
    def _kelvin_to_rgb(kelvin: float) -> List[float]:
        """Approximate RGB for a given color temperature in Kelvin.

        Returns RGB components in 0..255 range (non-linear, sRGB-like approx).
        Based on Tanner Helland's approximation.
        """
        # Clamp to reasonable range
        k = float(kelvin)
        k = max(1000.0, min(40000.0, k))
        t = k / 100.0

        # Red
        if t <= 66.0:
            r = 255.0
        else:
            r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
            r = max(0.0, min(255.0, r))

        # Green
        if t <= 66.0:
            g = 99.4708025861 * math.log(max(1.0, t)) - 161.1195681661
        else:
            g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        g = max(0.0, min(255.0, g))

        # Blue
        if t >= 66.0:
            b = 255.0
        elif t <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(max(1.0, t - 10.0)) - 305.0447927307
            b = max(0.0, min(255.0, b))

        return [r, g, b]

    @staticmethod
    def _parse_kelvin_value(value: Union[float, str]) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower()
        if s.endswith("k"):
            s = s[:-1]
            # Allow formats like "3.5" meaning 3500K
            if "." in s:
                return float(float(s) * 1000.0)
        try:
            return float(s)
        except Exception:
            # Fallback: ignore invalid input
            return 6500.0

    @classmethod
    def _gains_from_kelvin(cls, value: Union[float, str]) -> List[float]:
        k = cls._parse_kelvin_value(value)
        r, g, b = cls._kelvin_to_rgb(k)
        # Normalize to 0..1
        r /= 255.0
        g /= 255.0
        b /= 255.0
        # Gains are inverse of illuminant RGB, normalized so mean gain = 1
        inv = np.array(
            [1.0 / max(1e-6, r), 1.0 / max(1e-6, g), 1.0 / max(1e-6, b)], dtype=np.float32
        )
        inv /= float(inv.mean())
        return inv.tolist()


@TRANSFORMS.register_module()
class HmCrop:
    def __init__(
        self,
        rectangle: List[int] = list(),
        keys: List[str] = list(),
    ):
        self.keys = keys
        self.rectangle = rectangle
        self.calculated_clip_boxes: Dict = dict()
        if self.rectangle is not None:
            if isinstance(self.rectangle, (list, tuple)):
                if not any(item is not None for item in self.rectangle):
                    self.rectangle = []

    @staticmethod
    def fix_clip_box(clip_box, hw: List[int]):
        if isinstance(clip_box, list):
            if clip_box[0] is None:
                clip_box[0] = 0
            if clip_box[1] is None:
                clip_box[1] = 0
            if clip_box[2] is None:
                clip_box[2] = hw[1]
            if clip_box[3] is None:
                clip_box[3] = hw[0]
            clip_box = np.array(clip_box, dtype=np.int64)
        return clip_box

    def __call__(self, results):
        with _transform_profile_scope(results, self.__class__.__name__):
            if self.rectangle and self.keys:
                for key in self.keys:
                    img = results[key]
                    if key not in self.calculated_clip_boxes:
                        self.calculated_clip_boxes[key] = HmCrop.fix_clip_box(
                            self.rectangle, [image_height(img), image_width(img)]
                        )
                    clip_box = self.calculated_clip_boxes[key]
                    icf = is_channels_first(img)
                    if not icf:
                        img = make_channels_first(img)
                    if len(img.shape) == 4:
                        img = img[
                            :,
                            :,
                            clip_box[1] : clip_box[3],
                            clip_box[0] : clip_box[2],
                        ]
                    else:
                        assert len(img.shape) == 3
                        img = img[
                            :,
                            clip_box[1] : clip_box[3],
                            clip_box[0] : clip_box[2],
                        ]
                    if not icf:
                        img = make_channels_last(img)
                    results[key] = img
                    if key == "img":
                        # TODO: shape probably needs to be 2 elements only
                        results["img_shape"] = [torch.tensor(img.shape, dtype=torch.int64)]
                        results["ori_shape"] = [results["img_shape"][0].clone()]
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(rectangle={self.rectangle})"
        return repr_str


@TRANSFORMS.register_module()
class CloneImage:
    def __init__(
        self,
        source_key: str,
        dest_key: str,
    ):
        self.source_key = source_key
        self.dest_key = dest_key
        assert (self.source_key and self.dest_key) or (not self.source_key and not self.dest_key)

    def __call__(self, results):
        with _transform_profile_scope(results, self.__class__.__name__):
            if self.source_key:
                img = results.get(self.source_key, None)
                if img is not None:
                    results[self.dest_key] = img.clone()
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(source_key={self.source_key}, dest_key={self.dest_key})"
        return repr_str


# @POSE_PIPELINES.register_module()
# class HmTopDownAffine:
#     """Affine transform the image to make input.

#     Required key:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
#     'rotation' and 'center'.

#     Modified key:'img', 'joints_3d', and 'joints_3d_visible'.

#     Args:
#         use_udp (bool): To use unbiased data processing.
#             Paper ref: Huang et al. The Devil is in the Details: Delving into
#             Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
#     """

#     def __init__(self, use_udp=False):
#         self.use_udp = use_udp

#     def __call__(self, results):
#         image_size = results["ann_info"]["image_size"]

#         img = results["img"]
#         joints_3d = results["joints_3d"]
#         joints_3d_visible = results["joints_3d_visible"]
#         c = results["center"]
#         s = results["scale"]
#         r = results["rotation"]

#         if self.use_udp:
#             trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
#             if not isinstance(img, list):
#                 img = cv2.warpAffine(
#                     img,
#                     trans,
#                     (int(image_size[0]), int(image_size[1])),
#                     flags=cv2.INTER_LINEAR,
#                 )
#             else:
#                 img = [
#                     cv2.warpAffine(
#                         i,
#                         trans,
#                         (int(image_size[0]), int(image_size[1])),
#                         flags=cv2.INTER_LINEAR,
#                     )
#                     for i in img
#                 ]

#             joints_3d[:, 0:2] = warp_affine_joints(joints_3d[:, 0:2].copy(), trans)
#         else:
#             trans = get_affine_transform(c, s, r, image_size)
#             if not isinstance(img, list):
#                 if isinstance(img, torch.Tensor):
#                     if False:
#                         device = img.device
#                         img = img.clamp(0, 255) / 255.0
#                         img = cv2.warpAffine(
#                             img.cpu().numpy(),
#                             trans,
#                             (int(image_size[0]), int(image_size[1])),
#                             flags=cv2.INTER_LINEAR,
#                         )
#                         show_image("warped", img, wait=False)
#                         img = torch.from_numpy(img).to(device, non_blocking=True)
#                     else:
#                         ih = image_height(img)
#                         iw = image_width(img)
#                         if r == 0:
#                             output_w = image_size[0]
#                             output_h = image_size[1]
#                             half_h = float(output_h) / 2
#                             half_w = float(output_w) / 2
#                             cx = c[0]
#                             cy = c[1]
#                             # scale_x = 4
#                             # scale_y = 4

#                             scale_x = s[1]
#                             scale_y = s[0]

#                             tlwh = torch.from_numpy(
#                                 bbox_cs2xywh(c, s, padding=1.25)
#                             ).to(torch.int64)

#                             # tlwh = torch.from_numpy(results["bbox"][:4]).to(torch.int64)
#                             tlbr = torch.tensor(
#                                 [
#                                     tlwh[0],
#                                     tlwh[1],
#                                     tlwh[0] + tlwh[2],
#                                     tlwh[1] + tlwh[3],
#                                 ],
#                                 dtype=torch.int64,
#                             )

#                             # tlbr = torch.tensor(
#                             #     [
#                             #         int(cx - half_w / scale_x),
#                             #         int(cy - half_h / scale_y),
#                             #         int(cx + half_w / scale_x),
#                             #         int(cy + half_h / scale_y),
#                             #     ],
#                             #     dtype=torch.int64,
#                             # )
#                             img = extract_subimage(img=img, bbox=tlbr)
#                             img = resize_image(
#                                 img, new_width=output_w, new_height=output_h
#                             )
#                         else:
#                             assert False
#                             # Does not seem to work
#                             # c = c.copy()
#                             # c[0] -= float(ih) / 2
#                             # c[0] /= ih
#                             # c[1] -= float(iw) / 2
#                             # c[1] /= iw
#                             # trans = get_affine_transform(c, s, r, image_size)
#                             trans = torch.from_numpy(trans).to(
#                                 img.device, non_blocking=True
#                             )
#                             img = make_channels_first(img)
#                             if not torch.is_floating_point(img):
#                                 img = img.to(torch.float, non_blocking=True)
#                             # show_image("warped", img, wait=False)
#                             img = warp_affine_pytorch(
#                                 img,
#                                 trans,
#                                 # (int(image_size[1]), int(image_size[0])),
#                                 (int(image_size[0]), int(image_size[1])),
#                             )
#                         # img = resize_image(img, new_width=288, new_height=384)
#                         if img.ndim == 4:
#                             assert img.shape[0] == 1
#                             img = img.squeeze(0)
#                         # show_image("warped", img, wait=False)
#                         # time.sleep(0.25)
#                 else:
#                     img = cv2.warpAffine(
#                         img,
#                         trans,
#                         (int(image_size[0]), int(image_size[1])),
#                         flags=cv2.INTER_LINEAR,
#                     )
#             else:
#                 img = [
#                     cv2.warpAffine(
#                         i,
#                         trans,
#                         (int(image_size[0]), int(image_size[1])),
#                         flags=cv2.INTER_LINEAR,
#                     )
#                     for i in img
#                 ]
#             for i in range(results["ann_info"]["num_joints"]):
#                 if joints_3d_visible[i, 0] > 0.0:
#                     joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

#         results["img"] = img
#         results["joints_3d"] = joints_3d
#         results["joints_3d_visible"] = joints_3d_visible

#         return results


# @POSE_PIPELINES.register_module()
# class HmToTensor:
#     """Transform image to Tensor.

#     Required key: 'img'. Modifies key: 'img'.

#     Args:
#         results (dict): contain all information about training.
#     """

#     def __init__(self, device="cpu"):
#         self.device = device

#     def _to_tensor(self, x):
#         if isinstance(x, torch.Tensor):
#             if not torch.is_floating_point(x):
#                 x = x.to(torch.float32, non_blocking=True)
#             return make_channels_first(x / 255.0)
#         else:
#             return (
#                 torch.from_numpy(x.astype("float32"))
#                 .permute(2, 0, 1)
#                 .to(self.device)
#                 .div_(255.0)
#             )

#     def __call__(self, results):
#         if isinstance(results["img"], (list, tuple)):
#             results["img"] = [self._to_tensor(img) for img in results["img"]]
#         else:
#             results["img"] = self._to_tensor(results["img"])

#         return results


@TRANSFORMS.register_module()
class HmTopDownAffine:
    """Affine transform the image to make input.

    Required key:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified key:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        with _transform_profile_scope(results, self.__class__.__name__):
            image_size = results["ann_info"]["image_size"]

            img = results["img"]
            joints_3d = results["joints_3d"]
            joints_3d_visible = results["joints_3d_visible"]
            c = results["center"]
            s = results["scale"]
            r = results["rotation"]

            if self.use_udp:
                trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
                if not isinstance(img, list):
                    img = cv2.warpAffine(
                        img,
                        trans,
                        (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR,
                    )
                else:
                    img = [
                        cv2.warpAffine(
                            i,
                            trans,
                            (int(image_size[0]), int(image_size[1])),
                            flags=cv2.INTER_LINEAR,
                        )
                        for i in img
                    ]

                joints_3d[:, 0:2] = warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

            else:
                if r == 0 and isinstance(img, torch.Tensor):
                    pass
                else:
                    trans = get_affine_transform(c, s, r, image_size)
                    if not isinstance(img, list):
                        img = cv2.warpAffine(
                            img,
                            trans,
                            (int(image_size[0]), int(image_size[1])),
                            flags=cv2.INTER_LINEAR,
                        )
                    else:
                        img = [
                            cv2.warpAffine(
                                i,
                                trans,
                                (int(image_size[0]), int(image_size[1])),
                                flags=cv2.INTER_LINEAR,
                            )
                            for i in img
                        ]
                    for i in range(results["ann_info"]["num_joints"]):
                        if joints_3d_visible[i, 0] > 0.0:
                            joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

            results["img"] = img
            results["joints_3d"] = joints_3d
            results["joints_3d_visible"] = joints_3d_visible

            return results


@TRANSFORMS.register_module()
class HmExtractBoundingBoxes:
    def __init__(self, source_name: str = "det_bboxes"):
        self.source_name = source_name

    def __call__(self, results):
        with _transform_profile_scope(results, self.__class__.__name__):
            results["bbox"] = results[self.source_name]
            return results


@TRANSFORMS.register_module()
class HmTopDownGetBboxCenterScale:
    """Convert bbox from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required key: 'bbox', 'ann_info'

    Modifies key: 'center', 'scale'

    Args:
        padding (float): bbox padding scale that will be multilied to scale.
            Default: 1.25
    """

    # Pixel std is 200.0, which serves as the normalization factor to
    # to calculate bbox scales.
    pixel_std: float = 200.0

    def __init__(self, padding: float = 1.25):
        self.padding = padding

    def __call__(self, results):
        with _transform_profile_scope(results, self.__class__.__name__):
            if "center" in results and "scale" in results:
                warnings.warn(
                    'Use the "center" and "scale" that already exist in the data '
                    "sample. The padding will still be applied."
                )
                results["scale"] *= self.padding
            else:
                bbox = results["bbox"]
                centers = []
                scales = []
                for video_data_sample in results["data_samples"].video_data_samples:
                    image_size = video_data_sample.metainfo["ori_shape"]
                    aspect_ratio = image_size[0] / image_size[1]

                    center, scale = bbox_xywh2cs(
                        bbox,
                        aspect_ratio=aspect_ratio,
                        padding=self.padding,
                        pixel_std=self.pixel_std,
                    )

                    centers.append(center)
                    scales.append(scale)

            results["centers"] = centers
            results["scales"] = scales
            return results


def _to_float(t: Union[np.ndarray, torch.Tensor]):
    if isinstance(t, torch.Tensor):
        if not torch.is_floating_point(t):
            return t.to(torch.float, non_blocking=True)
        return t
    return t.astype(np.float32)


@TRANSFORMS.register_module()
class HmLoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        with _transform_profile_scope(results, self.__class__.__name__):
            img = results["img"]
            if self.to_float32:
                img = _to_float(img)
            assert img.ndim == 4
            img = make_channels_last(img)
            batch_size = img.size(0)
            shape = img.shape[1:3]
            results["img"] = img
            results["filename"] = [None for _ in range(batch_size)]
            results["ori_filename"] = [None for _ in range(batch_size)]
            results["img_shape"] = [shape for _ in range(batch_size)]
            results["ori_shape"] = [shape for _ in range(batch_size)]
            results["img_fields"] = ["img"]
            return results


@TRANSFORMS.register_module()
class HmRealTime:
    """Throttle inference batches so they do not run faster than the source FPS.

    The transform inspects the batch FPS metadata (``hm_real_time_fps`` by default)
    and sleeps before returning if the caller is ahead of real time. A ``scale``
    parameter allows speeding up (>1.0) or slowing down (<1.0) relative to the
    FPS reported by the dataset.
    """

    def __init__(
        self, enabled: bool = False, scale: float = 1.0, fps_key: str = "hm_real_time_fps"
    ):
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.fps_key = fps_key
        self._next_available_time: Optional[float] = None

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        with _transform_profile_scope(results, self.__class__.__name__):
            if not self.enabled:
                return results

            fps = self._extract_fps(results)
            if fps is None:
                return results

            effective_fps = fps * self.scale if self.scale else fps
            if effective_fps <= 0:
                return results

            frame_count = max(1, self._count_frames(results))
            interval = frame_count / effective_fps

            now = time.perf_counter()
            next_time = self._next_available_time
            if next_time is None:
                self._next_available_time = now + interval
                return results

            wait_time = next_time - now
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.perf_counter()
                next_time = max(next_time, now)
            else:
                next_time = now

            self._next_available_time = next_time + interval
            return results

    def _count_frames(self, results: Dict[str, Any]) -> int:
        for key in ("inputs", "img"):
            count = self._batch_size_from_value(results.get(key))
            if count is not None:
                return count
        data_samples = results.get("data_samples")
        if data_samples is not None and hasattr(data_samples, "video_data_samples"):
            return len(data_samples.video_data_samples)
        return 1

    def _batch_size_from_value(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return len(value)
        if isinstance(value, (torch.Tensor, StreamTensorBase)):
            if value.ndim == 0:
                return 1
            return int(value.shape[0]) if value.shape else 1
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return 1
            return int(value.shape[0])
        return None

    def _extract_fps(self, results: Dict[str, Any]) -> Optional[float]:
        direct_value = results.get(self.fps_key)
        fps = self._coerce_fps(direct_value)
        if fps is not None:
            return fps

        img_info = results.get("img_info")
        if isinstance(img_info, dict):
            fps = self._coerce_fps(img_info.get("fps"))
            if fps is not None:
                return fps

        data_samples = results.get("data_samples")
        if data_samples is not None:
            fps = self._fps_from_data_samples(data_samples)
            if fps is not None:
                return fps

        return None

    def _fps_from_data_samples(self, data_samples: Any) -> Optional[float]:
        fps = self._coerce_fps(getattr(data_samples, self.fps_key, None))
        if fps is not None:
            return fps
        video_samples = getattr(data_samples, "video_data_samples", None)
        if isinstance(video_samples, list):
            for sample in video_samples:
                fps = self._coerce_fps(self._get_from_sample(sample))
                if fps is not None:
                    return fps
        return None

    def _get_from_sample(self, sample: Any) -> Any:
        if sample is None:
            return None
        if hasattr(sample, "get"):
            value = sample.get(self.fps_key, None)
            if value is not None:
                return value
        metainfo = getattr(sample, "metainfo", None)
        if isinstance(metainfo, dict):
            return metainfo.get(self.fps_key)
        return None

    def _coerce_fps(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                fps = self._coerce_fps(item)
                if fps is not None:
                    return fps
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.item() if value.numel() == 1 else value.flatten()[0].item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = float(value.reshape(-1)[0])
        if isinstance(value, (int, float)):
            fps = float(value)
            return fps if fps > 0 else None
        try:
            fps = float(value)
        except (TypeError, ValueError):
            return None
        return fps if fps > 0 else None
