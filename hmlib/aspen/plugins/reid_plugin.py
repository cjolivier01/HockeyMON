"""Player appearance embeddings for the static DCF tracker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hmlib.utils.gpu import unwrap_tensor, wrap_tensor

from .base import Plugin
from .tracker_plugin import TrackerPlugin

logger = logging.getLogger(__name__)


def validate_reid_session(session) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ReID ONNX requires one image input and one appearance output")
    shape, features = inputs[0].shape, outputs[0].shape
    if (
        inputs[0].type != "tensor(float)"
        or len(shape) != 4
        or shape[1:] != [3, 256, 128]
        or len(features) != 2
        or features[1] != 256
    ):
        raise ValueError("ReID ONNX contract is float32 [N,3,256,128] -> [N,256]")


def player_crops(
    image: torch.Tensor, boxes: torch.Tensor, *, settings: dict | None = None
) -> torch.Tensor:
    """Bilinear RGB crops with the pinned NVIDIA ReIdentificationNet normalization."""
    if image.ndim != 3:
        raise ValueError("ReID requires one RGB image per frame")
    if image.shape[0] == 3:
        chw = image
    elif image.shape[-1] == 3:
        chw = image.permute(2, 0, 1)
    else:
        raise ValueError("ReID requires three RGB channels")
    boxes = boxes.to(device=chw.device, dtype=torch.float32)
    if not torch.isfinite(boxes).all() or (boxes[:, 2:4] <= boxes[:, :2]).any():
        raise ValueError("ReID boxes must be finite with positive width and height")
    height, width = chw.shape[-2:]
    y = (torch.arange(256, device=chw.device, dtype=torch.float32) + 0.5) / 256
    x = (torch.arange(128, device=chw.device, dtype=torch.float32) + 0.5) / 128
    xs = boxes[:, 0, None] + x * (boxes[:, 2] - boxes[:, 0])[:, None]
    ys = boxes[:, 1, None] + y * (boxes[:, 3] - boxes[:, 1])[:, None]
    grid = torch.stack(
        (
            (xs[:, None, :].expand(-1, 256, -1) / width) * 2 - 1,
            (ys[:, :, None].expand(-1, -1, 128) / height) * 2 - 1,
        ),
        dim=-1,
    )
    settings = settings or {}
    padding_mask = None
    if settings.get("keepAspc", 0):
        # Fit each ROI proportionally inside the network raster. Sample only
        # its resized extent, with zero-valued letterboxing outside the ROI.
        extent = boxes[:, 2:4] - boxes[:, :2]
        scale = torch.minimum(128 / extent[:, 0], 256 / extent[:, 1])
        fitted_w, fitted_h = extent[:, 0] * scale, extent[:, 1] * scale
        local_x = (
            (torch.arange(128, device=chw.device) + 0.5)[None] - (128 - fitted_w[:, None]) / 2
        ) / fitted_w[:, None]
        local_y = (
            (torch.arange(256, device=chw.device) + 0.5)[None] - (256 - fitted_h[:, None]) / 2
        ) / fitted_h[:, None]
        xs = boxes[:, 0, None] + local_x * extent[:, 0, None]
        ys = boxes[:, 1, None] + local_y * extent[:, 1, None]
        grid = torch.stack(
            (
                (xs[:, None, :].expand(-1, 256, -1) / width) * 2 - 1,
                (ys[:, :, None].expand(-1, -1, 128) / height) * 2 - 1,
            ),
            dim=-1,
        )
        padding_mask = (
            (local_y[:, :, None] >= 0)
            & (local_y[:, :, None] <= 1)
            & (local_x[:, None, :] >= 0)
            & (local_x[:, None, :] <= 1)
        )
    crops = F.grid_sample(
        chw.float()[None].expand(len(boxes), -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    if padding_mask is not None:
        crops = crops * padding_mask[:, None]
    if settings.get("colorFormat", 0) == 1:
        crops = crops.flip(1)
    mean = crops.new_tensor(settings.get("offsets", [123.675, 116.28, 103.53])).reshape(1, 3, 1, 1)
    return (crops - mean) * settings.get("netScaleFactor", 0.01735207357279195)


class ReidPlugin(Plugin):
    def __init__(
        self,
        onnx_path: str | None = None,
        enabled: bool = True,
        batch_size: int = 32,
        tracker_config: str | None = None,
        gpu: int | None = None,
    ):
        super().__init__(enabled=enabled)
        if gpu is not None:
            from hmlib.models.appearance import validate_appearance_gpu

            gpu = validate_appearance_gpu(gpu)
        self.gpu = gpu
        self.settings = {}
        self.engine = False
        if tracker_config:
            if onnx_path:
                raise ValueError("Choose an appearance ONNX model or native tracker config")
            from hmlib.models.appearance import read_tracker_model

            self.path, self.settings = read_tracker_model(tracker_config)
            self.engine = True
            manifest = Path(tracker_config).resolve().parent / "manifest.json"
            if manifest.is_file():
                import yaml

                data = yaml.safe_load(manifest.read_text())
                if data.get("model_id") == "reidentificationnet-deployable-v1.2":
                    from hmlib.models.player_analytics import load_model_bundle

                    self.path, _ = load_model_bundle("reid", manifest.parent)
                    self.engine = False
        elif onnx_path:
            self.path = Path(onnx_path)
        else:
            raise ValueError("Appearance extraction requires its selected prepared model")
        if not 1 <= batch_size <= 256:
            raise ValueError("ReID batch size must be between 1 and 256")
        self.batch_size = batch_size
        self.session = None

    def _ensure_session(self):
        if self.session is None:
            if self.engine:
                from hmlib.models.appearance import TensorRtAppearanceSession

                options = {} if self.gpu is None else {"gpu": self.gpu}
                self.session = TensorRtAppearanceSession(self.path, **options)
                self.batch_size = min(self.batch_size, self.session.maximum_batch)
                return self.session
            import onnxruntime as ort

            if not self.path.is_file():
                raise FileNotFoundError(f"Prepare the selected ReID model first: {self.path}")
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers.insert(
                    0,
                    (
                        "CUDAExecutionProvider"
                        if self.gpu is None
                        else ("CUDAExecutionProvider", {"device_id": self.gpu})
                    ),
                )
            else:
                logger.warning("ReID CUDA provider unavailable; appearance extraction runs on CPU")
            session = ort.InferenceSession(str(self.path), providers=providers)
            if (
                "CUDAExecutionProvider" in available
                and "CUDAExecutionProvider" not in session.get_providers()
            ):
                raise RuntimeError(
                    "ReID CUDA initialization failed; refusing provider substitution"
                )
            validate_reid_session(session)
            self.session = session
        return self.session

    def finalize(self):
        if self.engine and self.session is not None:
            self.session.close()
            self.session = None

    def forward(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        images = unwrap_tensor(context["original_images"])
        samples = context["data_samples"]
        if isinstance(samples, list):
            if len(samples) != 1:
                raise ValueError("ReID expects one tracked video batch")
            samples = samples[0]
        frames = samples.video_data_samples
        if len(images) != len(frames):
            raise ValueError("ReID image and detection batches must align")
        session = self._ensure_session()
        for image, frame in zip(images, frames):
            # Decoders and stitchers publish BGR original_images. The pinned
            # NVIDIA preprocessing contract (also used by team embeddings) is RGB.
            rgb = image.flip(0 if image.shape[0] == 3 else -1)
            instances = getattr(frame, "pred_instances", None)
            if instances is None:
                continue
            boxes = unwrap_tensor(instances.bboxes)
            scores = unwrap_tensor(instances.scores)
            count = getattr(instances, "num_detections", None)
            if count is None:
                count = getattr(instances, "metainfo", {}).get("num_detections", len(boxes))
            count = int(unwrap_tensor(count))
            if count < 0 or count > len(boxes):
                raise ValueError("ReID detection count exceeds the supplied boxes")
            active = torch.nonzero(
                (scores[:count] > 0) & (boxes[:count, 2:4] > boxes[:count, :2]).all(dim=1)
            ).flatten()
            features = boxes.new_zeros((len(boxes), 256), dtype=torch.float32)
            for start in range(0, len(active), self.batch_size):
                indices = active[start : start + self.batch_size]
                crops = player_crops(rgb, boxes[indices], settings=self.settings)
                output = session.run(None, {session.get_inputs()[0].name: crops.cpu().numpy()})[0]
                if output.shape != (len(indices), 256) or not np.isfinite(output).all():
                    raise ValueError("ReID inference returned invalid appearance embeddings")
                embedding = torch.as_tensor(output, device=features.device)
                features[indices] = F.normalize(embedding, p=2, dim=1)
            instances.reid_features = wrap_tensor(features)
        return {"data_samples": context["data_samples"]}

    def input_keys(self):
        return {"original_images", "data_samples"}

    def output_keys(self):
        return {"data_samples"}


class AppearanceTrackerPlugin(TrackerPlugin):
    """Adapt static payload metadata to the DCF tracker's CPU count contract."""

    def _build_static_tracker_inputs(self, *args, **kwargs):
        payload = super()._build_static_tracker_inputs(*args, **kwargs)
        payload["num_detections"] = payload["num_detections"].to(device="cpu")
        return payload


def configure_reid(config: dict, game: Path, *, validate_weights: bool = True) -> None:
    """Insert extraction immediately before tracking, leaving disabled graphs intact."""
    from hmlib.ui.model_preparation import REID_MODEL, cache_directory, model_status

    settings = config.get("pipeline", {}).get("tracker", {})
    tracker_enabled = settings.get("reid-enable", False)
    team = config.get("team", {})
    team_enabled = team.get("enabled", False) and team.get("backend") == "reidentificationnet"
    if not tracker_enabled and not team_enabled:
        return
    shared = config.get("hm_ui", {}).get("shared_analytics", {})
    custom = config.get("hm_ui", {}).get("run", {}).get("reid_onnx", "")
    path = None
    if tracker_enabled and "reid" in shared:
        native = settings.get("reid-config-file")
        if not native:
            if validate_weights:
                raise ValueError(
                    "Shared appearance model preparation did not provide its tracker configuration"
                )
            native = str(game / ".hm-ui" / "shared-models" / "tracker.yaml")
        extraction = {"tracker_config": str(native)}
        # Graph inspection can precede native model preparation. Executable
        # shared graphs require its resolved device instead of choosing one.
        if validate_weights or settings.get("gpu-id") is not None:
            from hmlib.models.appearance import validate_appearance_gpu

            extraction["gpu"] = validate_appearance_gpu(settings.get("gpu-id"))
    else:
        extraction = {}
    needs_legacy = (tracker_enabled and "reid" not in shared) or (
        team_enabled and "team" not in shared
    )
    if needs_legacy:
        model = settings.get("reid-model", REID_MODEL)
        if tracker_enabled and not custom and model != REID_MODEL:
            raise ValueError(f"Unsupported ReID model: {model}")
        if custom:
            path = Path(custom).expanduser()
            if not path.is_absolute():
                path = game / path
            if validate_weights and not path.is_file():
                raise FileNotFoundError(f"Configured ReID ONNX file does not exist: {path}")
        else:
            status = model_status(game, "reid", verify_digest=validate_weights)
            if validate_weights and status["state"] != "ready":
                raise ValueError(f"Prepare the ReID model before enabling it: {status['message']}")
            path = (
                Path(status["path"])
                if status["state"] == "ready"
                else cache_directory(game, "reid") / "reid.onnx"
            )
        if tracker_enabled and "reid" not in shared:
            extraction = {"onnx_path": str(path)}
    plugins = config.get("aspen", {}).get("plugins", {})
    if team_enabled:
        team_path = path
        if "team" in shared:
            from hmlib.models.player_analytics import load_model_bundle

            bundle = config["pipeline"]["player-analytics"]["team"].get("bundle")
            if not bundle and validate_weights:
                raise ValueError("Shared team model preparation did not provide its bundle")
            team_path = (
                load_model_bundle("team", bundle)[0]
                if bundle
                else game / ".hm-ui" / "shared-models" / "team.onnx"
            )
        team_nodes = [
            node
            for node in plugins.values()
            if str(node.get("class", "")).endswith("TeamAffiliationPlugin")
        ]
        if not team_nodes:
            raise ValueError("NVIDIA team appearance requires a TeamAffiliationPlugin stage")
        for node in team_nodes:
            node.setdefault("params", {}).update(
                reid_onnx_path=str(team_path), backend="reidentificationnet"
            )
    if not tracker_enabled:
        return
    tracker = plugins.get("tracker")
    if not tracker or not tracker.get("enabled", True):
        raise ValueError("Appearance matching requires an enabled tracker stage")
    if "player_reid" in plugins:
        raise ValueError("The graph already declares a player_reid stage")
    plugins["player_reid"] = {
        "class": "hmlib.aspen.plugins.reid_plugin.ReidPlugin",
        "depends": list(tracker.get("depends", [])),
        "params": extraction,
    }
    tracker["depends"] = ["player_reid"]
    tracker["class"] = "hmlib.aspen.plugins.reid_plugin.AppearanceTrackerPlugin"
    if tracker.get("params") is None:
        tracker["params"] = {}
    params = tracker["params"]
    params["tracker_class"] = "hockeymon.core.HmDcfTrackerCudaStatic"
    if params.get("tracker_kwargs") is None:
        params["tracker_kwargs"] = {}
    params["tracker_kwargs"]["reid_feature_dim"] = 256
