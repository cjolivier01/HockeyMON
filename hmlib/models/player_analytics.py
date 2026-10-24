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
    "pose": "rtmpose-m-coco17-5449a830cd121954.onnx",
    "jersey": "parseq-hockey-cvprw2024-df48ad7463aa89a5.onnx",
    "action": "stgcnpp-coco2d-ntu60-a69d882572452c6c.onnx",
    "team": "reidentificationnet-deployable-v1.2.onnx",
    "reid": "reidentificationnet-deployable-v1.2.onnx",
}
_RELEASE = "https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/"


@lru_cache(maxsize=5)
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
    native = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "hstream/player-models/models"
        / contract["onnx"]["sha256"]
        / "model.onnx"
    )
    if cache_dir is None and native.is_file():
        if _digest(native) != contract["onnx"]["sha256"]:
            raise ValueError(f"Native cached {feature} model checksum mismatch: {native}")
        return native
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
            urllib.request.urlopen(
                (
                    contract["source"]["url"]
                    if feature in {"team", "reid"}
                    else _RELEASE + _ASSETS[feature]
                ),
                timeout=60,
            ) as source,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
        if _digest(temporary_path) != expected:
            raise ValueError(f"Downloaded {feature} model checksum mismatch")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_model_bundle(feature: str, bundle: str | Path) -> tuple[Path, dict]:
    """Validate and consume the native bundle's actual ONNX and semantic contract."""
    directory = Path(bundle).expanduser().resolve()
    manifest = directory / "manifest.json"
    if not manifest.is_file() or manifest.stat().st_size > 1024 * 1024:
        raise ValueError(f"Missing or oversized {feature} model manifest: {manifest}")
    contract = yaml.safe_load(manifest.read_text())
    expected = model_contract(feature)
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != 1
        or contract.get("feature") != feature
    ):
        raise ValueError(f"Model bundle does not describe {feature} schema 1")
    for key in ("model_family", "topology_id"):
        if contract.get(key) != expected.get(key):
            raise ValueError(f"Unsupported {feature} model {key}: {contract.get(key)}")
    for kind in ("inputs", "outputs"):
        specs = contract.get(kind)
        if not isinstance(specs, list) or len(specs) != len(expected[kind]):
            raise ValueError(f"Invalid {feature} model {kind}")
        for spec, pinned in zip(specs, expected[kind], strict=True):
            if (
                spec.get("dtype") != "float32"
                or spec.get("shape") != pinned["shape"]
                or not isinstance(spec.get("name"), str)
                or not spec["name"]
            ):
                raise ValueError(f"Unsupported {feature} {kind} tensor contract")
    preprocessing = contract.get("preprocessing", {})
    for key, value in expected.get("preprocessing", {}).items():
        if feature == "jersey" and key in {"charset", "eos_index"}:
            continue
        if preprocessing.get(key) != value:
            raise ValueError(f"Unsupported {feature} preprocessing setting {key}")
    if feature == "jersey":
        charset, eos = preprocessing.get("charset"), preprocessing.get("eos_index")
        if (
            not isinstance(charset, list)
            or len(charset) != 95
            or len(set(charset)) != 95
            or not isinstance(eos, int)
            or not 0 <= eos < 95
            or charset[eos] != "[E]"
            or set(charset) != set(expected["preprocessing"]["charset"])
        ):
            raise ValueError("Invalid PARSeq vocabulary or EOS binding")
    if feature == "action":
        labels = contract.get("labels")
        if (
            not isinstance(labels, list)
            or len(labels) != 60
            or any(not isinstance(value, str) or not value or len(value) > 128 for value in labels)
        ):
            raise ValueError("Activity bundle requires its 60 output labels")
    onnx = contract.get("onnx", {})
    filename = onnx.get("file")
    if not isinstance(filename, str) or Path(filename).is_absolute():
        raise ValueError(f"Invalid {feature} ONNX bundle filename")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size > 1024**3:
        raise ValueError(f"Missing, unsafe or oversized {feature} ONNX bundle file")
    if _digest(path) != onnx.get("sha256"):
        raise ValueError(f"Prepared {feature} ONNX checksum mismatch: {path}")
    return path, contract


class PlayerModel:
    def __init__(
        self,
        feature: str,
        model_path: str | None = None,
        provider: str = "auto",
        model_bundle: str | None = None,
    ):
        if provider not in {"auto", "cpu", "cuda"}:
            raise ValueError("Player model provider must be auto, cpu or cuda")
        self.contract = model_contract(feature)
        self.feature = feature
        self.path = Path(model_path).expanduser() if model_path else None
        self._model_bundle = model_bundle
        if model_bundle:
            if model_path:
                raise ValueError("Choose a model bundle or ONNX path, not both")
            self.path, self.contract = load_model_bundle(feature, model_bundle)
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
                    or (self._model_bundle and node.name != spec["name"])
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
