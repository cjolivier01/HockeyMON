"""One-shot scoreboard proposals. No training libraries or per-frame state."""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
MODEL_SHA256 = "2cd82b9e3f1d87d1a5a93e5f80491189750ab0e9ec201e9500cccc57334bd704"
MODEL_NAME = f"scoreboard-mask2former-r50-{MODEL_SHA256[:16]}.pt"
MODEL_URL = (
    "https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/"
    + MODEL_NAME
)
WIDTH, HEIGHT = 2048, 768


def model_path() -> Path:
    override = os.environ.get("HM_SCOREBOARD_TORCH_MODEL")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Scoreboard model does not exist: {path}")
        return path
    path = Path.home() / ".cache" / "hstream" / "models" / MODEL_NAME
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError(f"Scoreboard model checksum mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading scoreboard proposal model to %s", path)
    temporary = None
    try:
        digest = hashlib.sha256()
        with (
            urlopen(MODEL_URL, timeout=30) as response,
            tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output,
        ):
            temporary = Path(output.name)
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 256 * 1024 * 1024:
                    raise ValueError("Scoreboard model download exceeds 256 MiB")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded scoreboard model checksum mismatch")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _run_model(path: Path, tensor: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    # Scope every GPU reference to this call, including a partly constructed
    # model if loading fails. The caller retries only after the exception scope
    # has ended and its traceback has released these objects.
    with torch.inference_mode():
        model = torch.jit.load(str(path), map_location=device).eval()
        classes, masks = model(tensor.to(device))
        return classes.cpu().numpy(), masks.cpu().numpy()


def infer(path: Path, tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        logger.info("Scoreboard proposal inference uses CPU (CUDA unavailable)")
        return _run_model(path, tensor, "cpu")
    result = None
    try:
        result = _run_model(path, tensor, "cuda:0")
    except (RuntimeError, torch.jit.Error) as error:
        # TorchScript wraps device errors, including OOM, in a generic error.
        if not isinstance(error, torch.cuda.OutOfMemoryError) and not re.search(
            r"(?:CUDA|HIP) out of memory|(?:CUDA|HIP) error: out of memory|"
            r"cudaErrorMemoryAllocation|hipErrorOutOfMemory",
            str(error),
        ):
            raise
        logger.warning("Scoreboard GPU memory exhausted; retrying once on CPU")
    # Do not retry inside the handler: the traceback can retain GPU tensors.
    gc.collect()
    torch.cuda.empty_cache()
    if result is not None:
        return result
    return _run_model(path, tensor, "cpu")


def postprocess(
    classes: np.ndarray, masks: np.ndarray, resized: tuple[int, int], original: tuple[int, int]
) -> list[list[int]] | None:
    if classes.shape != (1, 100, 2) or masks.shape != (1, 100, 192, 512):
        raise ValueError("Scoreboard model output does not match the exported contract")
    if not np.isfinite(classes).all() or not np.isfinite(masks).all():
        raise ValueError("Scoreboard model returned nonfinite values")
    logits = classes[0] - classes[0].max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    scores = probabilities[:, 0] / probabilities.sum(axis=1)
    width, height = resized
    best_score, best_quad = 0.5, None
    # Match the training model's ten highest classification queries. Expand
    # only one mask at a time, never 100 masks at source panorama resolution.
    for index in np.argsort(-scores, kind="stable")[:10]:
        if scores[index] < best_score:
            continue
        logits = cv2.resize(masks[0, index], (WIDTH, HEIGHT))[:height, :width]
        binary = logits > 0
        if not binary.any():
            continue
        score = float(scores[index] * (1 / (1 + np.exp(-logits[binary]))).mean())
        if score < best_score:
            continue
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        if cv2.contourArea(hull) <= 0:
            continue
        perimeter = cv2.arcLength(hull, True)
        quad = None
        for step in range(28):
            approximation = cv2.approxPolyDP(hull, (0.01 + step * 0.005) * perimeter, True)
            if len(approximation) == 4:
                quad = approximation.reshape(4, 2).astype(float)
                break
        if quad is None:
            logger.info("Scoreboard mask has no four-corner fit; using its minimum-area rectangle")
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
        # Clockwise in image coordinates, then rotate to the upper-left corner.
        centre = quad.mean(axis=0)
        quad = quad[np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))]
        quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)
        quad *= np.array(original) / np.array(resized)
        quad = np.clip(np.rint(quad), [0, 0], np.array(original) - 1).astype(int)
        if len(np.unique(quad, axis=0)) != 4 or not cv2.isContourConvex(quad.astype(np.int32)):
            continue
        best_score, best_quad = score, quad.tolist()
    return best_quad


def propose(image: Image.Image, original_size: tuple[int, int]) -> list[list[int]] | None:
    scale = min(WIDTH / image.width, HEIGHT / image.height, 1.0)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    rgb = cv2.resize(np.asarray(image.convert("RGB")), resized, interpolation=cv2.INTER_LINEAR)
    normalized = (rgb.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    canvas = np.zeros((1, 3, HEIGHT, WIDTH), dtype=np.float32)
    canvas[0, :, : resized[1], : resized[0]] = normalized.transpose(2, 0, 1)
    classes, masks = infer(model_path(), torch.from_numpy(canvas))
    return postprocess(classes, masks, resized, original_size)
