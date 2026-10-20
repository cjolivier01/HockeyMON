"""PRTReid inference wrapper.

PRTReid (Somers et al., the SoccerNet Game State Reconstruction baseline) is a
multi-task BPBreID: one backbone feeding a re-identification head, a team head
and a role head. Its ``forward`` returns an 8-tuple::

    embeddings, visibility_scores, id_cls_scores, team_cls_scores,
    Role_cls_scores, pixels_cls_scores, spatial_features, masks

Only ``embeddings[GLOBAL]`` is used here. The team *head* is not: it was
trained against SoccerNet's clip-relative ``left``/``right`` labels, which
carry no meaning across games. What transfers is the embedding, which
multi-task training made team-discriminative -- so the embedding is clustered
per game, exactly as the SoccerNet baseline's ``kmeans_embeddings`` module
does. The model's role head is also left unread -- nothing downstream consumes
a role today.

The package is not on PyPI and its ``requirements.txt`` pulls a training stack
(monai, tb-nightly, a pinned ``torchmetrics==0.10.3``) that collides with the
mmengine/mmdet versions this repo depends on. Install it without dependencies
and add back only what inference actually imports::

    pip install --no-deps git+https://github.com/VlSomers/prtreid.git
    pip install wandb tensorboard yacs gdown torchmetrics monai deepdiff

None of those are optional despite being training concerns: ``prtreid``'s
package ``__init__`` imports its training stack eagerly, so inference fails
without them. The upstream ``torchmetrics==0.10.3`` pin is ignored on purpose
-- a current torchmetrics imports fine and nothing on the inference path uses
it. :func:`_apply_compat_shims` handles the two upstream APIs that have since
been removed from their libraries.

Note the upstream license is Hippocratic HL3-LAW-MEDIA-MIL-SOC-SV, which is not
an OSI-approved open-source license and carries use restrictions.
"""

from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn.functional as F

from hmlib.log import logger

#: Published checkpoints. Both are required: the second is the HRNet32 backbone
#: that BPBreID loads by filename out of a directory.
PRTREID_WEIGHTS_URL = "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar"
HRNET_WEIGHTS_URL = "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth"
HRNET_WEIGHTS_FILENAME = "hrnetv2_w32_imagenet_pretrained.pth"

#: Role head classes, in the order SoccerNet's label encoder produced them.
#: PRTReid's fixed input geometry.
INPUT_HEIGHT = 256
INPUT_WIDTH = 128

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_INSTALL_HINT = (
    "PRTReid is not importable. Install it without its conflicting pins:\n"
    "    pip install --no-deps git+https://github.com/VlSomers/prtreid.git\n"
    "    pip install wandb tensorboard yacs gdown torchmetrics monai deepdiff\n"
    "(prtreid's package __init__ imports its training stack eagerly, so all of\n"
    "those are needed even for inference.)\n"
    "Or select a different backend with team.backend=lab."
)


def _apply_compat_shims() -> None:
    """Restore two APIs prtreid uses that its dependencies have since removed.

    Both live in training-only modules that prtreid's package ``__init__``
    imports anyway, so inference cannot get past them. Shimming here beats
    pinning matplotlib or albumentations backwards for the whole environment.

    * ``matplotlib.cm.get_cmap`` was removed in matplotlib 3.9 in favour of
      ``matplotlib.colormaps``; prtreid's ranking visualiser still calls it.
    * ``albumentations.functional`` was reorganised away in albumentations 2.x;
      prtreid's random-occlusion augmentation imports the name. The stub raises
      on attribute access, so the only thing it enables is the import -- a
      training path that actually used it would still fail loudly.
    """
    import matplotlib
    import matplotlib.cm

    if not hasattr(matplotlib.cm, "get_cmap"):
        matplotlib.cm.get_cmap = lambda name=None, lut=None: matplotlib.colormaps[name]

    import albumentations

    if not hasattr(albumentations, "functional"):
        stub = types.ModuleType("albumentations.functional")

        def _unavailable(name: str):
            raise AttributeError(
                f"albumentations.functional.{name} is a training-only API that "
                "albumentations 2.x removed; it is stubbed for import only"
            )

        stub.__getattr__ = _unavailable  # type: ignore[attr-defined]
        albumentations.functional = stub
        sys.modules["albumentations.functional"] = stub


def _resolve_weights(path: str, url: str, filename: str) -> str:
    """Return a local path for ``path``, downloading ``url`` when it is absent."""
    if path and os.path.isfile(path):
        return path

    hub_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(hub_dir, exist_ok=True)
    target = path if path else os.path.join(hub_dir, filename)
    if os.path.isfile(target):
        return target

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    logger.info("Downloading PRTReid weights %s -> %s", url, target)
    torch.hub.download_url_to_file(url, target, progress=True)
    return target


def _load_checkpoint_into(model, state_dict: dict, weights_path: str) -> None:
    """Load PRTReid weights and fail loudly if they did not actually land.

    prtreid's own ``load_pretrained_weights`` wraps ``torch.load`` in a bare
    try/except and merely prints on failure. Under torch >= 2.6 the
    ``weights_only=True`` default rejects this checkpoint's numpy scalars, so
    that path leaves the model silently running on ImageNet initialisation --
    producing plausible-looking embeddings that carry no team information.
    This does the load explicitly and raises when the match rate says the
    checkpoint did not apply.
    """
    cleaned = {(key.removeprefix("module.")): value for key, value in state_dict.items()}
    target = model.state_dict()
    matched = {
        key: value
        for key, value in cleaned.items()
        if key in target and target[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)

    ratio = len(matched) / max(1, len(target))
    if ratio < 0.5:
        raise RuntimeError(
            f"PRTReid checkpoint {weights_path} matched only {len(matched)} of "
            f"{len(target)} model tensors ({ratio:.0%}). The architecture config "
            "and the checkpoint disagree; the model would run on ImageNet "
            "initialisation and its embeddings would carry no team information."
        )
    logger.info(
        "PRTReid loaded %d/%d tensors (%.0f%%) from %s",
        len(matched),
        len(target),
        ratio * 100,
        weights_path,
    )


def _infer_num_classes(state_dict: dict) -> int:
    """Read the identity head width out of a checkpoint.

    The identity head is unused at inference, but ``build_model`` still has to
    size it or loading the checkpoint reports a shape mismatch for every
    classifier tensor. Reading it back beats hardcoding a number that only
    matches one released checkpoint.
    """
    for key, value in state_dict.items():
        if key.endswith("global_identity_classifier.classifier.weight"):
            return int(value.shape[0])
    raise RuntimeError(
        "could not find 'global_identity_classifier.classifier.weight' in the "
        "PRTReid checkpoint; it is not the published SoccerNet baseline"
    )


class PrtreidModel:
    """Loads PRTReid once and runs batched inference over torso crops."""

    def __init__(
        self,
        weights: str = "",
        backbone_weights: str = "",
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        self._device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if str(device) == "cuda" and self._device.type != "cuda":
            logger.warning("PRTReid requested cuda but no GPU is visible; running on CPU")
        self._batch_size = int(batch_size)

        try:
            _apply_compat_shims()
            from prtreid.models import build_model
            from prtreid.scripts.default_config import get_default_config
            from prtreid.utils.constants import GLOBAL
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(f"{_INSTALL_HINT}\n\nUnderlying error: {exc}") from exc

        self._global_key = GLOBAL

        weights_path = _resolve_weights(
            weights, PRTREID_WEIGHTS_URL, "prtreid-soccernet-baseline.pth.tar"
        )
        backbone_path = _resolve_weights(
            backbone_weights, HRNET_WEIGHTS_URL, HRNET_WEIGHTS_FILENAME
        )

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        num_classes = _infer_num_classes(state_dict)

        cfg = get_default_config()
        # Mirror sn-gamestate's modules/reid/prtreid.yaml. These are the values
        # the published checkpoint was trained with; changing any of them
        # silently produces a model the weights do not fit.
        cfg.model.name = "bpbreid"
        cfg.model.load_weights = weights_path
        cfg.data.height = INPUT_HEIGHT
        cfg.data.width = INPUT_WIDTH
        bp = cfg.model.bpbreid
        bp.pooling = "gwap"
        bp.normalization = "identity"
        bp.mask_filtering_training = False
        bp.mask_filtering_testing = False
        bp.training_binary_visibility_score = True
        bp.testing_binary_visibility_score = True
        bp.last_stride = 1
        bp.learnable_attention_enabled = False
        bp.dim_reduce = "after_pooling"
        bp.dim_reduce_output = 256
        bp.backbone = "hrnet32"
        bp.test_embeddings = ["globl"]
        bp.test_use_target_segmentation = "none"
        bp.shared_parts_id_classifier = False
        # BPBreID appends the filename itself, so this is a directory.
        bp.hrnet_pretrained_path = os.path.dirname(backbone_path) + os.sep

        model = build_model(
            name="bpbreid",
            num_classes=num_classes,
            loss="part_based",
            pretrained=True,
            use_gpu=(self._device.type == "cuda"),
            config=cfg,
        )

        _load_checkpoint_into(model, state_dict, weights_path)
        model.eval().to(self._device)
        if self._device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self._model = model
        self._use_amp = self._device.type == "cuda"

        mean = torch.tensor(_IMAGENET_MEAN, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self._device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std
        self._embedding_dim = int(bp.dim_reduce_output)
        logger.info(
            "PRTReid ready on %s (identity head %d classes, embedding dim %d)",
            self._device,
            num_classes,
            self._embedding_dim,
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _preprocess(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalise an already-uniform ``(N, 3, 256, 128)`` batch.

        The caller cuts every ROI straight to this geometry in one
        `roi_align`, so there is nothing per-crop left to do here -- just a
        dtype cast, a 0-1 scale and the ImageNet standardisation, all as whole
        batch ops.
        """
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"expected an (N, 3, H, W) batch, got {tuple(batch.shape)}")

        x = batch.to(device=self._device, dtype=torch.float32, non_blocking=True)
        if x.shape[-2:] != (INPUT_HEIGHT, INPUT_WIDTH):
            x = F.interpolate(
                x, size=(INPUT_HEIGHT, INPUT_WIDTH), mode="bilinear", align_corners=False
            )
        # 0-255 float by convention; a data-dependent check here would cost
        # a GPU->CPU sync on every frame.
        x = ((x / 255.0) - self._mean) / self._std
        # HRNet is convolution-heavy, so channels_last lets cuDNN pick the
        # tensor-core kernels; with autocast below it is worth 1.29x.
        return x.contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Return the global re-identification embeddings, ``(N, D)``."""
        if batch.shape[0] == 0:
            return torch.zeros((0, self._embedding_dim))

        embeddings: list[torch.Tensor] = []
        for start in range(0, batch.shape[0], self._batch_size):
            chunk = self._preprocess(batch[start : start + self._batch_size])
            # fp16 autocast: this is an inference-only embedding feeding a
            # two-means split, so the reduced precision is well inside the
            # margin the clusterer already tolerates.
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._use_amp):
                out = self._model(chunk)
            if not isinstance(out, tuple) or not out:
                raise RuntimeError(
                    "unexpected PRTReid forward output: expected the 8-tuple "
                    "(embeddings, visibility_scores, id_cls_scores, ...), got "
                    f"{type(out).__name__} of length "
                    f"{len(out) if isinstance(out, tuple) else 'n/a'}"
                )
            embeddings.append(out[0][self._global_key].detach().float().cpu())

        return torch.cat(embeddings, dim=0)
