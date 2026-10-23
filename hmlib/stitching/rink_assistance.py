"""Reviewable rink landmarks and board suggestions from calibration stills."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from hmlib.stitching.rink_projection import panorama_rays, rays_to_sources

LOGGER = logging.getLogger(__name__)
MODEL_SHA256 = "76b96e34a9386267c08e0edc6cb09b2196a94a933b27e2f92fcae5d9f0426a08"
MODEL_NAME = f"rink-landmarks-mask2former-swin-s-{MODEL_SHA256[:16]}.onnx"
MODEL_URL = (
    "https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/"
    + MODEL_NAME
)
LABELS = (
    "Blue Line",
    "Center Ice Circle",
    "Center Line",
    "Crease",
    "Faceoff Dot",
    "Field",
    "Goal",
    "Goal Line",
    "Slot Box",
    "Trapezoid",
    "Zone Circle",
)


def model_path():
    override = os.environ.get("HM_RINK_LANDMARKS_ONNX_MODEL")
    path = (
        Path(override).expanduser()
        if override
        else Path.home() / ".cache/hstream/models" / MODEL_NAME
    )
    if path.is_file():
        if not override:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != MODEL_SHA256:
                raise ValueError(f"Rink landmark model checksum mismatch: {path}")
        return path
    if override:
        raise FileNotFoundError(f"Rink landmark model does not exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    LOGGER.info("Downloading pinned rink landmark model to %s", path)
    try:
        digest, total = hashlib.sha256(), 0
        with (
            urlopen(MODEL_URL, timeout=60) as response,
            tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output,
        ):
            temporary = Path(output.name)
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 1024 * 1024 * 1024:
                    raise ValueError("Rink landmark model exceeds 1 GiB")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded rink landmark model checksum mismatch")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def line_samples(mask, original):
    yy, xx = np.nonzero(mask)
    if len(xx) < 20:
        return []
    data = np.stack([xx, yy], axis=-1).astype(float)
    _, _, vectors = np.linalg.svd(data - data.mean(axis=0), full_matrices=False)
    along = data @ vectors[0]
    span = np.ptp(along)
    if span < 20:
        return []
    bins = np.clip(((along - along.min()) / span * 24).astype(int), 0, 23)
    samples = [
        (data[bins == index].mean(axis=0) + 0.5)
        * np.asarray(original)
        / [mask.shape[1], mask.shape[0]]
        - 0.5
        for index in range(1, 23)
        if np.count_nonzero(bins == index) >= 3
    ]
    return [point.tolist() for point in samples] if len(samples) >= 2 else []


def postprocess(classes, masks, resized, original):
    classes, masks = np.asarray(classes), np.asarray(masks)
    if (
        classes.shape != (1, 100, 12)
        or masks.shape != (1, 100, 256, 256)
        or not np.isfinite(classes).all()
        or not np.isfinite(masks).all()
    ):
        raise ValueError("Rink landmark output does not match the pinned model contract")
    logits = classes[0] - classes[0].max(axis=1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ranked = np.argsort(probabilities[:, :11].ravel())[-50:][::-1]
    result = []
    for candidate in ranked:
        query, label = divmod(int(candidate), 11)
        if probabilities[query, label] < 0.5 or label == 5:
            continue
        raw = cv2.resize(masks[0, query], (1024, 1024))[: resized[1], : resized[0]]
        selected = raw > 0
        if selected.sum() < 10:
            continue
        confidence = float(probabilities[query, label] * np.mean(1 / (1 + np.exp(-raw[selected]))))
        if confidence < 0.5:
            continue
        samples = line_samples(selected, original) if label in (0, 2, 7) else []
        if samples:
            result.append(
                {
                    "feature": "center-line" if label == 2 else "",
                    "label": LABELS[label],
                    "line": True,
                    "automatic": True,
                    "enabled": True,
                    "confidence": confidence,
                    "pixels": samples,
                }
            )
    return result


def detect_landmarks(sources):
    import onnxruntime as ort

    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    if providers[0] == "CPUExecutionProvider":
        LOGGER.warning(
            "Rink landmark inference is using CPU because ONNX Runtime CUDA is unavailable"
        )
    session = ort.InferenceSession(str(model_path()), providers=providers)
    if [item.name for item in session.get_inputs()] != ["images"] or {
        item.name for item in session.get_outputs()
    } != {"class_logits", "mask_logits"}:
        raise ValueError("Rink landmark model has an unexpected input/output contract")
    result = []
    for camera, source in enumerate(sources):
        bgr = cv2.imread(source["path"])
        if bgr is None:
            raise ValueError(f"Cannot decode camera reference: {source['path']}")
        scale = 1008 / max(bgr.shape[:2])
        resized = (max(1, int(bgr.shape[1] * scale + 0.5)), max(1, int(bgr.shape[0] * scale + 0.5)))
        rgb = cv2.resize(bgr, resized)[..., ::-1].astype(np.float32)
        tensor = np.zeros((1, 3, 1024, 1024), np.float32)
        tensor[0, :, : resized[1], : resized[0]] = (
            (rgb - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        ).transpose(2, 0, 1)
        classes, masks = session.run(["class_logits", "mask_logits"], {"images": tensor})
        result.extend(
            {**mark, "camera": camera}
            for mark in postprocess(classes, masks, resized, (source["width"], source["height"]))
        )
    return result


def mask_boards(game_dir, config, pto, sources, projection):
    rink = config.get("rink", {})
    revision = rink.get("ice_contours_geometry_revision")
    token = hashlib.sha256(str(revision).encode()).hexdigest()[:32] if revision else None
    prefix = f"rink_mask_{token}_" if token else "rink_mask_"
    count = int(rink.get("ice_contours_mask_count") or 1)
    if not 1 <= count <= 64:
        raise ValueError("Saved rink mask count is invalid")
    paths = [Path(game_dir) / f"{prefix}{index}.png" for index in range(count)]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(
            "Run the game once to create a current rink mask before requesting board suggestions"
        )
    calibration_time = max(
        (Path(game_dir) / name).stat().st_mtime_ns
        for name in ("autooptimiser_out.pto", ".stitching_artifacts.json")
    )
    if any(path.stat().st_mtime_ns < calibration_time for path in paths):
        raise ValueError("The saved mask predates this calibration; run the game to regenerate it")
    masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(mask is None for mask in masks) or len({mask.shape for mask in masks}) != 1:
        raise ValueError("Rink masks cannot be decoded or have inconsistent dimensions")
    mask = np.maximum.reduce(masks) > 0
    height, width = mask.shape
    projection_size = np.asarray(projection["image_size"])
    if abs(width / height - projection_size[0] / projection_size[1]) > 0.01:
        raise ValueError(
            "Saved rink mask dimensions differ from this calibrated panorama; regenerate it"
        )
    # Rotate the saved mask back into the calibration reference before sampling.
    rotation = float(config.get("stitching", {}).get("post_stitch_rotate_degrees") or 0)
    if rotation:
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), -rotation, 1)
        mask = (
            cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)
            > 0
        )
    columns = np.arange(max(1, int(width * 0.03)), min(width - 1, int(width * 0.97)))
    columns = columns[mask[:, columns].sum(axis=0) > height * 0.05]
    if len(columns) < width * 0.15:
        raise ValueError("The saved ice mask has no sufficiently wide board boundary")
    revision = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    result = []
    for near in (False, True):
        boundary = (
            height - 1 - np.argmax(mask[::-1, columns], axis=0)
            if near
            else np.argmax(mask[:, columns], axis=0)
        )
        valid = (boundary > height * 0.02) & (boundary < height * 0.98)
        x, y = columns[valid], boundary[valid]
        if len(x) < 20:
            continue
        # Robust quadratic smooths doors/occlusions while preserving the bowed
        # near-board silhouette. All three handles remain editable and unchecked.
        normalized = (x - width / 2) / width
        keep = np.ones(len(x), dtype=bool)
        for _ in range(4):
            if keep.sum() < 12:
                break
            coefficients = np.polyfit(normalized[keep], y[keep], 2)
            residual = np.abs(y - np.polyval(coefficients, normalized))
            keep = residual <= max(3, np.median(residual) * 3)
        if keep.sum() < 12:
            continue
        sample_x = np.linspace(float(x[keep].min()), float(x[keep].max()), 64)
        sample_y = np.polyval(coefficients, (sample_x - width / 2) / width)
        points = (np.stack([sample_x, sample_y], axis=-1) + 0.5) * projection_size / [
            width,
            height,
        ] - 0.5
        rays = panorama_rays(points, projection)
        projected = rays_to_sources(pto, rays, len(sources))
        best = None
        for camera, pixels in enumerate(projected):
            inside = (
                np.isfinite(pixels).all(axis=1)
                & (pixels >= 0).all(axis=1)
                & (pixels < [sources[camera]["width"], sources[camera]["height"]]).all(axis=1)
            )
            # Preserve a continuous visible segment, never bridge a camera seam.
            for segment in np.split(np.arange(len(pixels)), np.flatnonzero(~inside)):
                segment = segment[inside[segment]]
                if len(segment) < 8:
                    continue
                span = float(np.linalg.norm(rays[segment[0]] - rays[segment[-1]]))
                if best is None or span > best[0]:
                    best = span, camera, pixels[segment]
        if best is None or best[0] < 0.2:
            continue
        _, camera, pixels = best
        pixels = pixels[[0, (len(pixels) - 1) // 2, -1]] if near else pixels
        result.append(
            {
                "feature": "near-boards" if near else "far-boards",
                "label": "Ice mask near-board arc" if near else "Ice mask far boards",
                "camera": camera,
                "line": True,
                "arc": near,
                "automatic": True,
                "enabled": False,
                "confidence": 0.5,
                "mask_revision": revision,
                "pixels": pixels.tolist(),
            }
        )
    if not result:
        raise ValueError("No supported board boundary is visible in the calibrated source cameras")
    return result
