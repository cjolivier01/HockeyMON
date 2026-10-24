"""Pinned native player models and validated ONNX execution.

Contracts and source/license provenance mirror HockeyMONStream's model catalog.
Hockey PARSeq weights are CC BY-NC 3.0; STGCN++ labels are generic NTU60 actions.
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from hmlib.utils.checksum import sha256_file

logger = logging.getLogger(__name__)
_ASSETS = {
    "jersey": "parseq-hockey-cvprw2024-df48ad7463aa89a5.onnx",
    "action": "stgcnpp-coco2d-ntu60-a69d882572452c6c.onnx",
}
_RELEASE = "https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/"


@lru_cache(maxsize=2)
def model_contract(feature: str) -> dict:
    if feature not in _ASSETS:
        raise ValueError(f"Unknown player analytics model: {feature}")
    path = Path(__file__).resolve().parents[1] / "config" / "player-models" / f"{feature}.yaml"
    return yaml.safe_load(path.read_text())


def _digest(path: Path) -> str:
    return sha256_file(path)


def prepare_player_model(feature: str, cache_dir: Path | None = None) -> Path:
    """Download once, verify pinned bytes, and publish without partial cache files."""
    contract = model_contract(feature)
    directory = (
        cache_dir
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "hockeymon"
        / "player-models"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _ASSETS[feature]
    expected = contract["onnx"]["sha256"]
    if path.exists():
        if _digest(path) != expected:
            raise ValueError(f"Cached {feature} model checksum mismatch: {path}")
        return path
    logger.info("Preparing %s (%s)", contract["model_id"], contract["source"]["license"])
    descriptor, temporary = tempfile.mkstemp(prefix=f".{feature}-", suffix=".onnx", dir=directory)
    temporary_path = Path(temporary)
    try:
        with (
            os.fdopen(descriptor, "wb") as target,
            urllib.request.urlopen(_RELEASE + _ASSETS[feature], timeout=60) as source,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
        if _digest(temporary_path) != expected:
            raise ValueError(f"Downloaded {feature} model checksum mismatch")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


class PlayerModel:
    def __init__(self, feature: str, model_path: str | None = None, provider: str = "auto"):
        if provider not in {"auto", "cpu", "cuda"}:
            raise ValueError("Player model provider must be auto, cpu or cuda")
        self.contract = model_contract(feature)
        self.feature = feature
        self.path = Path(model_path).expanduser() if model_path else None
        self.provider = provider
        self.session = None

    def ensure_session(self):
        if self.session is not None:
            return self.session
        import onnxruntime as ort

        path = self.path or prepare_player_model(self.feature)
        if not path.is_file():
            raise FileNotFoundError(f"Player {self.feature} ONNX model does not exist: {path}")
        available = ort.get_available_providers()
        if self.provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"Requested CUDA provider unavailable for {self.feature}")
        providers = ["CPUExecutionProvider"]
        if self.provider != "cpu" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        elif self.provider == "auto":
            logger.warning("%s CUDA provider unavailable; inference runs on CPU", self.feature)
        session = ort.InferenceSession(str(path), providers=providers)
        if (
            providers[0] == "CUDAExecutionProvider"
            and "CUDAExecutionProvider" not in session.get_providers()
        ):
            raise RuntimeError(
                f"CUDA initialization failed for {self.feature}; refusing provider substitution"
            )
        for nodes, expected, kind in (
            (session.get_inputs(), self.contract["inputs"], "input"),
            (session.get_outputs(), self.contract["outputs"], "output"),
        ):
            if len(nodes) != len(expected):
                raise ValueError(f"{self.feature} ONNX {kind} count does not match native contract")
            for node, spec in zip(nodes, expected, strict=True):
                if (
                    node.type != "tensor(float)"
                    or len(node.shape) != len(spec["shape"])
                    or list(node.shape[1:]) != spec["shape"][1:]
                ):
                    raise ValueError(f"{self.feature} ONNX {kind} must be float32 {spec['shape']}")
                if isinstance(node.shape[0], int) and node.shape[0] != 1:
                    raise ValueError(f"{self.feature} ONNX batch must be dynamic or one")
        self.session = session
        return session

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        session = self.ensure_session()
        batch_size = 1 if session.get_inputs()[0].shape[0] == 1 else 8
        outputs = []
        for start in range(0, len(inputs), batch_size):
            values = session.run(
                None,
                {
                    session.get_inputs()[0].name: np.ascontiguousarray(
                        inputs[start : start + batch_size], dtype=np.float32
                    )
                },
            )[0]
            expected = (
                len(inputs[start : start + batch_size]),
                *self.contract["outputs"][0]["shape"][1:],
            )
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid {self.feature} output: expected finite {expected}, got {values.shape}"
                )
            outputs.append(values)
        return np.concatenate(outputs)
