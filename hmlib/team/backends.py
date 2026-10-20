"""Feature backends for team affiliation.

A backend turns torso crops into two parallel signals:

``features``
    What the clusterer splits on. Either a mean-Lab colour triple or a learned
    embedding.
``lightness``
    Mean CIELAB L per crop. Always derived from pixels, never from the backend's
    embedding, because the dark/light ranking that names the clusters needs a
    real photometric quantity and a learned embedding has no lightness axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .features import batched_lab_mean

#: Backend names accepted in ``team.backend``.
LAB_BACKEND = "lab"
PRTREID_BACKEND = "prtreid"
BACKEND_NAMES = (LAB_BACKEND, PRTREID_BACKEND)

#: (H, W) every ROI is cut to. This is PRTReid's input geometry, so one
#: extraction serves both backends -- the lab path just pools it down.
CROP_SIZE = (256, 128)


@dataclass(frozen=True)
class BackendOutput:
    """Per-crop signals, all first-dimension aligned with the input crops."""

    features: torch.Tensor  # (N, D) float32
    lightness: torch.Tensor  # (N,) float32

    def __post_init__(self) -> None:
        if self.lightness.shape[0] != self.features.shape[0]:
            raise ValueError("BackendOutput fields must agree on the crop count")


def _empty_output(dim: int) -> BackendOutput:
    return BackendOutput(
        features=torch.zeros((0, dim), dtype=torch.float32),
        lightness=torch.zeros((0,), dtype=torch.float32),
    )


class LabBackend:
    """Mean-Lab colour of the torso crop.

    Training-free and dependency-free, which makes it the backend the unit
    tests exercise and the fallback an operator can always select.
    """

    name = LAB_BACKEND
    dim = 3

    def extract(self, batch: torch.Tensor) -> BackendOutput:
        if batch.shape[0] == 0:
            return _empty_output(self.dim)
        # One pooled conversion over the whole batch, one transfer at the end.
        features = batched_lab_mean(batch).to(dtype=torch.float32, device="cpu")
        return BackendOutput(features=features, lightness=features[:, 0].clone())


class PrtreidBackend:
    """PRTReid embeddings for clustering.

    PRTReid is trained on SoccerNet, where multi-task training against team
    labels made its embedding team-discriminative. That is what transfers;
    measured on a real hockey clip a two-means split of the embedding produced
    a 1004/1289 partition of the two squads with 78% of crops above the
    assignment margin.
    """

    name = PRTREID_BACKEND

    def __init__(
        self,
        weights: str,
        backbone_weights: str,
        device: str = "cuda",
    ) -> None:
        self._weights = weights
        self._backbone_weights = backbone_weights
        self._device = device
        self._lab = LabBackend()
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from .prtreid_backend import PrtreidModel

            self._model = PrtreidModel(
                weights=self._weights,
                backbone_weights=self._backbone_weights,
                device=self._device,
            )
        return self._model

    @property
    def dim(self) -> int:
        return self._ensure_model().embedding_dim

    def extract(self, batch: torch.Tensor) -> BackendOutput:
        if batch.shape[0] == 0:
            return _empty_output(self.dim)

        # Lightness still comes from pixels: the embedding has no lightness
        # axis, and naming the clusters dark-vs-light needs a photometric
        # quantity. It reuses the same batch, so the crops are cut once.
        lab_out = self._lab.extract(batch)
        embeddings = self._ensure_model().infer(batch)
        return BackendOutput(
            features=embeddings.to(dtype=torch.float32, device="cpu"),
            lightness=lab_out.lightness,
        )


def stack_crops(crops: Sequence[torch.Tensor], size: Sequence[int] = CROP_SIZE) -> torch.Tensor:
    """Resize a list of differently-sized crops into one ``(N, 3, H, W)`` batch.

    The team trunk does not use this -- it cuts every ROI out of the frame in a
    single `roi_align`, which is both faster and avoids the Python loop. This
    is for callers that genuinely hold separate images, such as the SoccerNet
    evaluation script.
    """
    if not crops:
        return torch.zeros((0, 3, *size), dtype=torch.float32)
    resized = []
    for crop in crops:
        chw = crop if crop.shape[0] == 3 else crop.permute(2, 0, 1)
        resized.append(
            F.interpolate(
                chw.to(dtype=torch.float32).unsqueeze(0),
                size=tuple(size),
                mode="bilinear",
                align_corners=False,
            )
        )
    return torch.cat(resized, dim=0)


def build_backend(
    backend: str,
    *,
    prtreid_weights: str = "",
    prtreid_backbone_weights: str = "",
    prtreid_device: str = "cuda",
):
    """Construct a backend by name.

    Raises on an unknown name rather than defaulting, so a typo in
    ``team.backend`` surfaces instead of silently running a different model than
    the operator configured.
    """
    normalised = str(backend).strip().lower().replace("_", "-")
    if normalised == LAB_BACKEND:
        return LabBackend()
    if normalised == PRTREID_BACKEND:
        return PrtreidBackend(
            weights=prtreid_weights,
            backbone_weights=prtreid_backbone_weights,
            device=prtreid_device,
        )
    raise ValueError(
        f"unknown team.backend {backend!r}; expected one of {', '.join(BACKEND_NAMES)}"
    )
