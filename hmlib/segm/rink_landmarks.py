"""
Rink landmark segmentation: the painted features on the ice -- blue lines, the
center line, faceoff circles and dots, creases, goals, trapezoids -- as opposed
to :mod:`hmlib.segm.ice_rink`, which segments the playing surface itself.

The model is the 11-class Mask2Former trained by ``openmm/train_rink_landmarks.sh``
on the Roboflow "hockey-rink-landmarks" v2 dataset. Class names and colors come
from the checkpoint's ``dataset_meta``, so this module never hardcodes them.

Everything here is panorama-oriented: ``detect_rink_landmarks`` runs once on a
stitched frame, and ``build_overlay_layer`` bakes the result into a single
(color, alpha) pair so that per-frame compositing is one blend on the GPU.
"""

import argparse
import gc
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np
import torch

from hmlib.config import prepend_root_dir
from hmlib.log import logger
from hmlib.models.loader import get_model_config
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import (
    image_height,
    image_width,
    is_channels_first,
    make_channels_first,
    make_channels_last,
)
from hmlib.utils.model_registry import model_registry_scope

if TYPE_CHECKING:
    from mmdet.models.detectors.base import BaseDetector
    from mmdet.structures import DetDataSample

# The landmark model emits up to 50 instances per image and is far less
# confident than the single-class rink model, so it needs its own floor.
DEFAULT_SCORE_THRESH = 0.5

# "Field" is the whole playing surface -- the ice rink model already covers that
# far better, and as a full-frame mask it would bury every other class under a
# flat tint. Drop it from overlays unless it is asked for by name.
DEFAULT_EXCLUDED_CLASSES = ("Field",)

# Overlay appearance. Single source of truth: the plugin, the CLI flags and
# build_overlay_layer all default to these rather than restating the literals.
# The rink tint matches SegmBoundaries.draw so the stitch preview and the
# tracking overlay render the same mask the same way.
DEFAULT_RINK_MASK_COLOR = (0, 255, 0)
DEFAULT_RINK_MASK_ALPHA = 0.10
DEFAULT_FILL_ALPHA = 0.35
DEFAULT_OUTLINE_ALPHA = 0.95
DEFAULT_OUTLINE_THICKNESS = 3


class OverlayLayer(NamedTuple):
    """A baked overlay, cropped to the region it actually covers.

    ``box`` is ``(y0, y1, x0, x1)`` in full-frame coordinates; ``color`` and
    ``alpha`` are sized to that box, not to the frame.
    """

    color: torch.Tensor  # uint8 [3, h, w]
    alpha: torch.Tensor  # float32 [1, h, w]
    box: tuple[int, int, int, int]


def parse_class_list(classes: str | Sequence[str] | None) -> list[str] | None:
    """Normalize a class allow-list from a CLI string, YAML string, or sequence.

    A ``str`` is a ``Sequence[str]``, so ``list("Blue Line")`` would silently
    become an allow-list of single characters.
    """
    if classes is None:
        return None
    if isinstance(classes, str):
        classes = classes.split(",")
    names = [str(name).strip() for name in classes if str(name).strip()]
    return names or None


# Longest edge handed to the detector when no explicit scale is given.
#
# This is a memory cap, not a quality knob. The model's own test pipeline
# resizes the long edge to 1008 regardless, so the network sees the same
# pixels either way -- but Mask2Former's instance_postprocess first upsamples
# all 50 query masks back to the *input* resolution, which on a 12407x4710
# panorama is 50 * 58 Mpx * 4 bytes = 11.7 GiB and OOMs a 32 GB card. Masks are
# resampled back to full resolution afterwards, so capping here costs nothing
# the 1008-pixel network output had to begin with.
DEFAULT_MAX_INFERENCE_EDGE = 2048


def _auto_inference_scale(width: int, height: int) -> float | None:
    """Scale that brings the longest edge down to ``DEFAULT_MAX_INFERENCE_EDGE``."""
    longest = max(int(width), int(height))
    if longest <= DEFAULT_MAX_INFERENCE_EDGE:
        return None
    return DEFAULT_MAX_INFERENCE_EDGE / float(longest)


def landmark_metadata(model: "BaseDetector") -> tuple[tuple[str, ...], list[tuple[int, int, int]]]:
    """Return ``(classes, palette)`` from a detector's ``dataset_meta``."""
    meta = getattr(model, "dataset_meta", None) or {}
    classes = tuple(meta.get("classes") or ())
    # mmdet stores a palette *name* ("random") when neither the checkpoint nor
    # the config carries real colors, so this has to survive a str as well as
    # a list of triples or an ndarray of rows.
    try:
        palette = [tuple(int(channel) for channel in color) for color in meta.get("palette") or []]
    except (TypeError, ValueError):
        palette = []
    if any(len(color) != 3 for color in palette):
        palette = []
    if len(palette) < len(classes):
        # Deterministic fallback so an untagged checkpoint still renders. Say
        # so: otherwise the colors just silently stop matching the ones mmdet's
        # own visualizer uses for this checkpoint.
        logger.warning(
            "Rink landmark checkpoint has no usable palette (%r); falling back to "
            "generated colors, which will not match the training visualizations.",
            meta.get("palette"),
        )
        palette = [_hsv_color(i, len(classes)) for i in range(len(classes))]
    return classes, palette


def _hsv_color(index: int, total: int) -> tuple[int, int, int]:
    """Evenly spaced fallback color. RGB, because palettes here are RGB."""
    hue = int(179 * index / max(1, total))
    pixel = np.uint8([[[hue, 255, 255]]])
    rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _resolve_model_path(path: str) -> str:
    """Resolve a model path that may be a URL, repo-relative, or CWD-relative.

    Config-declared paths are repo-root-relative, but the same parameters also
    carry values a user typed, which are naturally relative to where they are
    standing. The graph threads config values through those parameters too, so
    the callee cannot tell the two apart -- accept either and let the one that
    exists win.
    """
    if "://" in path:
        return path
    from_root = prepend_root_dir(path)
    if os.path.exists(from_root) or os.path.isabs(path):
        return from_root
    from_cwd = os.path.abspath(path)
    return from_cwd if os.path.exists(from_cwd) else from_root


def _as_numpy_image(image: torch.Tensor | np.ndarray | StreamTensorBase) -> np.ndarray:
    if isinstance(image, StreamTensorBase):
        image = image.get()
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            # Calibration only ever needs one frame; a larger stitch batch is
            # still a single panorama repeated through time.
            image = image[0]
        image = make_channels_last(image)
        if torch.is_floating_point(image):
            image = image.clamp(0, 255)
        image = image.to(torch.uint8).cpu().numpy()
    else:
        image = make_channels_last(image)
    return np.ascontiguousarray(image)


def detect_rink_landmarks(
    image: torch.Tensor | np.ndarray,
    model: "BaseDetector",
    score_thr: float = DEFAULT_SCORE_THRESH,
) -> dict[str, Any] | None:
    """Run the landmark detector on one frame and return its instance masks."""
    from mmdet.apis import inference_detector

    frame = _as_numpy_image(image)
    with model_registry_scope("mmdet"):
        result: DetDataSample = inference_detector(model, frame)

    instances = result.pred_instances
    scores = instances.scores
    keep = scores >= float(score_thr)
    if not bool(keep.any()):
        return None

    # InstanceData raises AttributeError for a field that was never set, so a
    # bbox-only checkpoint has to be probed with getattr rather than `is None`.
    masks = getattr(instances, "masks", None)
    if masks is None:
        logger.warning("Rink landmark model returned no masks; is it a bbox-only checkpoint?")
        return None

    classes, palette = landmark_metadata(model)
    return {
        "labels": instances.labels[keep].cpu(),
        "scores": scores[keep].cpu(),
        "bboxes": instances.bboxes[keep].cpu(),
        "masks": masks[keep].to(torch.bool).cpu(),
        "classes": classes,
        "palette": palette,
    }


def _rescale_landmarks(
    landmarks: dict[str, Any], inference_scale: float, target_hw: tuple[int, int]
) -> dict[str, Any]:
    """Map masks and boxes produced at ``inference_scale`` back to full resolution.

    One mask at a time, through the sibling module's _resize_mask: upsampling
    the whole stack in a single float32 interpolate would recreate the very
    allocation DEFAULT_MAX_INFERENCE_EDGE exists to avoid, only on the host
    (N * 234 MB on a 12407x4710 panorama).
    """
    from hmlib.segm.ice_rink import _resize_mask

    height, width = target_hw
    masks = landmarks["masks"]
    if masks.numel():
        resized = torch.empty((masks.shape[0], height, width), dtype=torch.bool)
        for index in range(masks.shape[0]):
            resized[index] = _resize_mask(masks[index], height, width)
        landmarks["masks"] = resized
    inv_scale = 1.0 / inference_scale
    landmarks["bboxes"] = landmarks["bboxes"] * inv_scale
    return landmarks


def find_rink_landmarks(
    image: torch.Tensor | np.ndarray,
    config_file: str,
    checkpoint: str,
    device: torch.device | None = None,
    score_thr: float = DEFAULT_SCORE_THRESH,
    inference_scale: float | None = None,
    include: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Load the landmark detector, run it on one frame, and free it again.

    ``include`` is applied before the masks are rescaled, so a discarded class
    -- "Field" covers the whole sheet -- never pays for a full-resolution
    upsample it is about to be thrown away by.
    """
    from mmdet.apis import init_detector

    if device is None:
        device = torch.device("cpu")

    orig_height = image_height(image)
    orig_width = image_width(image)
    infer_image = _as_numpy_image(image)

    if inference_scale is not None and inference_scale <= 0:
        raise ValueError(f"Rink landmark inference scale must be > 0, got {inference_scale}")

    # A clamp, not a fallback: the cap exists to stop an 11.7 GiB allocation,
    # so an explicit --rink-landmarks-inference-scale must not be able to
    # disable it. Taking the minimum lets callers ask for less, never more.
    auto_scale = _auto_inference_scale(orig_width, orig_height)
    if auto_scale is not None and (inference_scale is None or inference_scale > auto_scale):
        if inference_scale is not None:
            logger.warning(
                "Clamping the requested rink landmark inference scale %.4f to %.4f so the "
                "longest edge stays within DEFAULT_MAX_INFERENCE_EDGE=%d.",
                inference_scale,
                auto_scale,
                DEFAULT_MAX_INFERENCE_EDGE,
            )
        inference_scale = auto_scale
        logger.info(
            "Running rink landmark inference at scale %.4f (%dx%d -> %dx%d).",
            inference_scale,
            orig_width,
            orig_height,
            round(orig_width * inference_scale),
            round(orig_height * inference_scale),
        )

    if inference_scale and inference_scale != 1.0:
        new_width = max(1, round(orig_width * inference_scale))
        new_height = max(1, round(orig_height * inference_scale))
        interpolation = cv2.INTER_AREA if inference_scale < 1.0 else cv2.INTER_LINEAR
        infer_image = cv2.resize(infer_image, (new_width, new_height), interpolation=interpolation)

    if device.type == "cpu":
        logger.info("Looking for the painted rink landmarks, this may take awhile...")
    with model_registry_scope("mmdet"):
        model = init_detector(config_file, checkpoint, device=device)
    try:
        landmarks = detect_rink_landmarks(infer_image, model=model, score_thr=score_thr)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if landmarks is None:
        return None
    landmarks = filter_landmarks(landmarks, include=include)
    if landmarks is None:
        return None
    if inference_scale and inference_scale != 1.0:
        landmarks = _rescale_landmarks(landmarks, inference_scale, (orig_height, orig_width))
    return landmarks


def configure_rink_landmarks(
    game_id: str,
    image: torch.Tensor | np.ndarray,
    device: torch.device | None = None,
    score_thr: float = DEFAULT_SCORE_THRESH,
    inference_scale: float | None = None,
    checkpoint: str | None = None,
    model_config: str | None = None,
    include: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve the landmark model from the game config and run it on ``image``."""
    model_config_file, model_checkpoint = model_config, checkpoint
    if not (model_config_file and model_checkpoint):
        # get_model_config reaches the game directory, which raises when the
        # game has none. Skip it entirely when the caller already supplied
        # both paths, so explicit overrides keep working without a game dir.
        config_default, checkpoint_default = get_model_config(
            game_id=game_id, model_name="rink_landmarks_segm"
        )
        model_config_file = model_config_file or config_default
        model_checkpoint = model_checkpoint or checkpoint_default
    if not model_config_file or not model_checkpoint:
        logger.warning(
            "No rink landmark model configured (model.rink_landmarks_segm); skipping landmarks."
        )
        return None

    config_path = _resolve_model_path(model_config_file)
    checkpoint_path = _resolve_model_path(model_checkpoint)
    if "://" not in checkpoint_path and not os.path.exists(checkpoint_path):
        logger.warning(
            "Rink landmark checkpoint not found: %s. Train it with "
            "openmm/train_rink_landmarks.sh or pass --rink-landmarks-checkpoint.",
            checkpoint_path,
        )
        return None

    return find_rink_landmarks(
        image=image,
        config_file=config_path,
        checkpoint=checkpoint_path,
        device=device,
        score_thr=score_thr,
        inference_scale=inference_scale,
        include=include,
    )


def filter_landmarks(
    landmarks: dict[str, Any],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDED_CLASSES,
) -> dict[str, Any] | None:
    """Keep only the named classes (case-insensitive). ``include`` wins over ``exclude``."""
    classes = landmarks["classes"]
    labels = landmarks["labels"]
    if not len(labels):
        return None

    if include:
        wanted = {str(name).strip().lower() for name in include}
        keep_ids = {i for i, name in enumerate(classes) if str(name).lower() in wanted}
        unknown = wanted - {str(classes[i]).lower() for i in keep_ids}
        if unknown:
            logger.warning(
                "Unknown rink landmark class(es) %s; known classes are %s",
                sorted(unknown),
                list(classes),
            )
    else:
        dropped = {str(name).strip().lower() for name in (exclude or ())}
        keep_ids = {i for i, name in enumerate(classes) if str(name).lower() not in dropped}

    keep = torch.tensor([int(label) in keep_ids for label in labels], dtype=torch.bool)
    if not bool(keep.any()):
        return None

    filtered = dict(landmarks)
    filtered["labels"] = labels[keep]
    filtered["scores"] = landmarks["scores"][keep]
    filtered["bboxes"] = landmarks["bboxes"][keep]
    filtered["masks"] = landmarks["masks"][keep]
    return filtered


def _mask_outline(mask_np: np.ndarray, thickness: int) -> np.ndarray:
    """Return a uint8 edge mask for one instance, drawn ``thickness`` pixels wide."""
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = np.zeros_like(mask_np)
    if contours:
        cv2.drawContours(edges, contours, -1, color=1, thickness=max(1, int(thickness)))
    return edges


def build_overlay_layer(
    height: int,
    width: int,
    landmarks: dict[str, Any] | None = None,
    rink_mask: torch.Tensor | None = None,
    rink_mask_color: tuple[int, int, int] = DEFAULT_RINK_MASK_COLOR,
    rink_mask_alpha: float = DEFAULT_RINK_MASK_ALPHA,
    fill_alpha: float = DEFAULT_FILL_ALPHA,
    outline_alpha: float = DEFAULT_OUTLINE_ALPHA,
    outline_thickness: int = DEFAULT_OUTLINE_THICKNESS,
    label_text: bool = False,
    bgr: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Bake masks into one ``(color[3,H,W] uint8, alpha[1,H,W] float32)`` pair.

    Later marks win: the rink tint goes down first, then landmark fills, then
    landmark outlines on top, so the thin classes stay legible over the fills.

    The checkpoint's palette is RGB while frames here are BGR end to end (the
    mmdet preprocessor does the ``bgr_to_rgb`` flip itself), so ``bgr`` swaps
    the palette rather than the much larger image.
    """
    color = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.float32)
    drew_anything = False

    def channel_order(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        return (rgb[2], rgb[1], rgb[0]) if bgr else rgb

    if rink_mask is not None and rink_mask_alpha > 0:
        mask_np = rink_mask.detach().to(torch.bool).cpu().numpy()
        if mask_np.shape != (height, width):
            logger.warning(
                "Rink mask is %s but the overlay plane is %s; skipping the rink tint.",
                mask_np.shape,
                (height, width),
            )
        else:
            color[mask_np] = channel_order(rink_mask_color)
            alpha[mask_np] = float(rink_mask_alpha)
            drew_anything = True

    if landmarks is not None and len(landmarks["labels"]):
        classes = landmarks["classes"]
        palette = landmarks["palette"]
        masks = landmarks["masks"]
        labels = landmarks["labels"]
        scores = landmarks["scores"]
        # Outlines are deferred so thin classes stay legible over the fills,
        # but they are accumulated into one shared plane rather than one
        # full-resolution plane per instance (58 MB each on a 12407x4710
        # panorama). 0 means "no outline here", otherwise 1 + palette index.
        outline_owner = np.zeros((height, width), dtype=np.uint16)
        outline_ids: set[int] = set()

        for mask, label in zip(masks, labels):
            selected = mask.detach().to(torch.bool).cpu().numpy()
            mask_np = selected.view(np.uint8)
            if mask_np.shape != (height, width):
                logger.warning(
                    "Landmark mask is %s but the overlay plane is %s; skipping it.",
                    mask_np.shape,
                    (height, width),
                )
                continue
            index = int(label)
            rgb = channel_order(palette[index] if index < len(palette) else (255, 255, 255))
            if fill_alpha > 0:
                color[selected] = rgb
                alpha[selected] = float(fill_alpha)
                drew_anything = True
            if outline_thickness > 0:
                edges = _mask_outline(mask_np, outline_thickness)
                # Later instances win, matching the previous two-pass order.
                np.copyto(outline_owner, index + 1, where=edges.astype(bool))
                outline_ids.add(index + 1)
                drew_anything = True

        # Iterate the ids actually written rather than np.unique()-ing a
        # panorama-sized plane, which would sort 58M elements to learn what
        # the fill loop already knew.
        for index in sorted(outline_ids):
            rgb = channel_order(palette[index - 1] if index - 1 < len(palette) else (255, 255, 255))
            selected = outline_owner == index
            color[selected] = rgb
            alpha[selected] = float(outline_alpha)

        if label_text:
            _draw_labels(color, alpha, landmarks, classes, palette, labels, scores, channel_order)

    if not drew_anything:
        return None

    # Crop to the covered region. Painted lines cover a few percent of a
    # panorama, so blending only this box turns the per-frame cost into a
    # fraction of what a full-frame blend would read and write.
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    if not rows.size or not cols.size:
        # Every mask was empty, or the alphas were all zero.
        logger.warning("Rink overlay covers no pixels; drawing nothing.")
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    color_t = torch.from_numpy(color[y0:y1, x0:x1]).permute(2, 0, 1).contiguous()
    alpha_t = torch.from_numpy(alpha[y0:y1, x0:x1]).unsqueeze(0).contiguous()
    return OverlayLayer(color=color_t, alpha=alpha_t, box=(y0, y1, x0, x1))


def _draw_labels(
    color: np.ndarray,
    alpha: np.ndarray,
    landmarks: dict[str, Any],
    classes: Sequence[str],
    palette: Sequence[tuple[int, int, int]],
    labels: torch.Tensor,
    scores: torch.Tensor,
    channel_order,
) -> None:
    # Scale text with the panorama so it stays readable on a 5k-wide frame.
    font_scale = max(0.5, color.shape[1] / 2400.0)
    thickness = max(1, round(font_scale * 2))
    for bbox, label, score in zip(landmarks["bboxes"], labels, scores):
        index = int(label)
        name = classes[index] if index < len(classes) else str(index)
        rgb = channel_order(palette[index] if index < len(palette) else (255, 255, 255))
        text = f"{name} {float(score):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        x = int(bbox[0])
        y = max(int(bbox[1]) - 4, text_height)

        # Rasterize into a box the size of the text, not the size of the
        # panorama: a full-frame scratch plane per label would be 58 MB each
        # on a 12407x4710 frame, to stamp a couple of dozen characters.
        height, width = color.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y - text_height - baseline)
        x1 = min(width, x + text_width)
        y1 = min(height, y + baseline)
        if x1 <= x0 or y1 <= y0:
            continue
        scratch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.putText(
            scratch,
            text,
            (x - x0, y - y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        selected = scratch > 127
        color[y0:y1, x0:x1][selected] = rgb
        alpha[y0:y1, x0:x1][selected] = 1.0


def overlay_layer_to(layer: OverlayLayer, device: torch.device) -> OverlayLayer:
    """Move a baked layer onto ``device`` once, as float32 ready for the blend.

    Callers cache the result. Doing this per frame instead would push the layer
    back over PCIe every frame, which costs more than the blend it feeds.
    """
    return layer._replace(
        color=layer.color.to(device=device, dtype=torch.float32),
        alpha=layer.alpha.to(device=device, dtype=torch.float32),
    )


def composite_overlay(img: torch.Tensor, layer: OverlayLayer) -> torch.Tensor:
    """Alpha-blend a prebuilt overlay onto ``img``, preserving its layout and dtype.

    Only ``layer.box`` is touched: the painted features cover a small part of a
    panorama, so blending the whole frame would read and write two orders of
    magnitude more than the marks need. Pass a layer already on ``img``'s device
    (see :func:`overlay_layer_to`); the conversions below are a correctness
    fallback, not the intended path.
    """
    was_channels_first = is_channels_first(img)
    work = make_channels_first(img)
    original_dtype = work.dtype

    color = layer.color.to(device=work.device, dtype=torch.float32)
    alpha = layer.alpha.to(device=work.device, dtype=torch.float32)
    if work.ndim == 4:
        color = color.unsqueeze(0)
        alpha = alpha.unsqueeze(0)

    y0, y1, x0, x1 = layer.box
    region = work[..., y0:y1, x0:x1]

    # lerp keeps this to one output allocation instead of the five a hand-rolled
    # `a * (1 - t) + b * t` would materialize.
    blended = torch.lerp(region.to(torch.float32), color, alpha)
    if not original_dtype.is_floating_point:
        blended.round_()
        if original_dtype == torch.uint8:
            blended.clamp_(0, 255)

    # Write the blended box back into a copy of the frame; the caller's tensor
    # is shared pipeline state and must not be mutated.
    out = work.clone()
    out[..., y0:y1, x0:x1] = blended.to(original_dtype)

    return out if was_channels_first else make_channels_last(out)


def summarize_landmarks(landmarks: dict[str, Any] | None) -> str:
    """One-line human-readable census of what was detected."""
    if landmarks is None or not len(landmarks["labels"]):
        return "no rink landmarks detected"
    classes = landmarks["classes"]
    counts: dict[str, int] = {}
    for label in landmarks["labels"]:
        index = int(label)
        name = classes[index] if index < len(classes) else str(index)
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{name} x{count}" for name, count in sorted(counts.items())]
    return f"{len(landmarks['labels'])} landmarks: " + ", ".join(parts)


def main(args: argparse.Namespace | None = None) -> int:
    """Render the rink mask and landmarks onto one stitched frame and save it."""
    from hmlib.config import get_game_dir
    from hmlib.segm.ice_rink import configure_ice_rink_mask

    if args is None:
        parser = argparse.ArgumentParser(
            description="Overlay the ice rink mask and painted rink landmarks on one frame"
        )
        parser.add_argument("--game-id", "-g", type=str, required=True, help="Game ID to process")
        parser.add_argument(
            "--image",
            type=str,
            default=None,
            help="Frame to annotate; defaults to <game_dir>/s.png (written by stitch --configure-only)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Where to write the annotated frame; defaults to <game_dir>/rink_landmarks.png",
        )
        parser.add_argument("--show", action="store_true", help="Display the annotated frame")
        parser.add_argument(
            "--device", "-d", type=str, default=None, help="Device used for inference"
        )
        parser.add_argument(
            "--score-thr",
            type=float,
            default=DEFAULT_SCORE_THRESH,
            help="Minimum landmark instance score",
        )
        parser.add_argument(
            "--scale",
            "-s",
            type=float,
            default=None,
            help="Downscale factor applied before landmark inference",
        )
        parser.add_argument(
            "--classes",
            type=str,
            default=None,
            help=f"Comma-separated class allow-list; default excludes {', '.join(DEFAULT_EXCLUDED_CLASSES)}",
        )
        parser.add_argument(
            "--checkpoint", "-c", type=str, default=None, help="Override the landmark checkpoint"
        )
        parser.add_argument(
            "--model-config",
            "-m",
            type=str,
            default=None,
            help="Override the landmark mmdet config",
        )
        parser.add_argument(
            "--no-rink-mask", action="store_true", help="Skip the ice rink mask tint"
        )
        parser.add_argument("--no-labels", action="store_true", help="Skip per-instance class text")
        parser.add_argument(
            "--fill-alpha", type=float, default=0.35, help="Opacity of the landmark fills"
        )
        parser.add_argument(
            "--outline-thickness", type=int, default=3, help="Landmark outline width in pixels"
        )
        args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cpu")
    game_dir = get_game_dir(game_id=args.game_id)
    if not game_dir:
        raise AttributeError(f"Could not determine game dir for game_id={args.game_id}")

    image_path = args.image or str(Path(game_dir) / "s.png")
    if not os.path.exists(image_path):
        print(f"Could not find stitched frame image: {image_path}")
        print("Generate one with: ./stitch.sh --configure-only")
        return 1

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read frame: {image_path}")
    height, width = frame.shape[:2]
    print(f"Annotating {image_path} ({width} x {height})")

    rink_mask = None
    if not args.no_rink_mask:
        rink_profile = configure_ice_rink_mask(
            game_id=args.game_id,
            device=device,
            expected_shape=torch.Size((height, width)),
            image=frame,
            # Annotating a frame must not rewrite the game's calibration; use
            # hmfind_ice_rink when you actually mean to regenerate it.
            persist=False,
        )
        rink_mask = (rink_profile or {}).get("combined_mask")
        print("rink mask: " + ("found" if rink_mask is not None else "unavailable"))

    include = [name for name in (args.classes or "").split(",") if name.strip()]
    landmarks = configure_rink_landmarks(
        game_id=args.game_id,
        image=frame,
        device=device,
        score_thr=args.score_thr,
        inference_scale=args.scale,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        include=include or None,
    )
    print(summarize_landmarks(landmarks))

    layer = build_overlay_layer(
        height=height,
        width=width,
        landmarks=landmarks,
        rink_mask=rink_mask,
        fill_alpha=args.fill_alpha,
        outline_thickness=args.outline_thickness,
        label_text=not args.no_labels,
    )
    if layer is None:
        print("Nothing to draw.")
        return 1

    annotated = composite_overlay(torch.from_numpy(frame), layer)

    output_path = args.output or str(Path(game_dir) / "rink_landmarks.png")
    cv2.imwrite(output_path, annotated.numpy())
    print(f"Wrote {output_path}")

    if args.show:
        from hmlib.ui import show_image as do_show_image

        do_show_image("Rink landmarks", annotated.numpy(), wait=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
