"""Per-game two-way clustering of player appearance.

Team identity is game-relative, so this fits fresh centroids per game rather
than loading a trained classifier. Crops that land between the two centroids
are reported as ``UNKNOWN`` rather than assigned on a coin flip.

Clustering features and the dark/light ranking are deliberately separate inputs.
The PRTReid backend clusters on a learned embedding that has no lightness axis,
but naming the groups still needs real pixel lightness, so callers supply both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hmlib.log import logger


@dataclass(frozen=True)
class ClusterAssignment:
    """Result of :meth:`OnlineTeamClusterer.observe` for one batch of crops."""

    clusters: torch.Tensor  # int64 (N,), -1 where unassigned
    margins: torch.Tensor  # float32 (N,), 0.0 where unassigned


def _kmeans2(points: torch.Tensor, iters: int = 25) -> torch.Tensor:
    """Two-means over ``(N, D)`` points, returning ``(2, D)`` centroids.

    Seeded deterministically -- the point farthest from the mean, then the point
    farthest from that -- so a given game yields the same split on every run.
    A random seed would make the overlay flicker between runs on identical input.
    """
    if points.shape[0] < 2:
        raise ValueError("two-means needs at least two points")

    first = points[torch.argmax(torch.cdist(points, points.mean(dim=0, keepdim=True)).squeeze(1))]
    second = points[torch.argmax(torch.cdist(points, first.unsqueeze(0)).squeeze(1))]
    centroids = torch.stack((first, second))

    for _ in range(iters):
        labels = torch.argmin(torch.cdist(points, centroids), dim=1)
        moved = False
        for k in range(2):
            members = points[labels == k]
            if members.numel() == 0:
                # An empty cluster means the seed pair collapsed; leaving the
                # centroid put is better than reseeding into a random point.
                continue
            new_centroid = members.mean(dim=0)
            if not torch.allclose(new_centroid, centroids[k]):
                moved = True
            centroids[k] = new_centroid
        if not moved:
            break
    return centroids


class OnlineTeamClusterer:
    """Accumulates torso features, fits two centroids, then assigns against them.

    Before ``warmup_samples`` crops have been seen the clusterer
    reports nothing: an early fit on a handful of crops from one end of the rink
    produces two centroids that both describe the same team.
    """

    def __init__(
        self,
        warmup_samples: int = 200,
        refit_interval: int = 600,
        max_buffer: int = 4000,
        min_margin: float = 0.12,
    ) -> None:
        if warmup_samples < 2:
            raise ValueError(f"warmup_samples must be >= 2, got {warmup_samples}")
        self._warmup_samples = int(warmup_samples)
        self._refit_interval = int(refit_interval)
        self._max_buffer = int(max_buffer)
        self._min_margin = float(min_margin)

        self._buffer: list[torch.Tensor] = []
        self._lightness: list[float] = []
        self._centroids: torch.Tensor | None = None
        self._scale_mean: torch.Tensor | None = None
        self._scale_std: torch.Tensor | None = None
        self._dark_cluster: int | None = None
        self._since_fit = 0

    @property
    def ready(self) -> bool:
        """True once centroids exist and assignments are meaningful."""
        return self._centroids is not None

    @property
    def dark_cluster(self) -> int | None:
        """Index of the lower-lightness cluster, or ``None`` before the fit."""
        return self._dark_cluster

    @property
    def buffered(self) -> int:
        """How many samples are held toward the next fit."""
        return len(self._buffer)

    def _standardise(self, feats: torch.Tensor) -> torch.Tensor:
        if self._scale_mean is None or self._scale_std is None:
            raise RuntimeError("standardise called before a fit")
        return (feats - self._scale_mean) / self._scale_std

    def _fit(self) -> None:
        points = torch.stack(self._buffer).to(dtype=torch.float32)
        # Lab axes span very different ranges (L is 0..100, a/b are +-128), and
        # a raw euclidean fit would be dominated by whichever axis is widest.
        self._scale_mean = points.mean(dim=0)
        self._scale_std = points.std(dim=0).clamp_min(1e-6)
        centroids = _kmeans2(self._standardise(points))

        labels = torch.argmin(torch.cdist(self._standardise(points), centroids), dim=1)
        lightness = torch.tensor(self._lightness, dtype=torch.float32)
        means = []
        for k in range(2):
            members = lightness[labels == k]
            means.append(float(members.mean()) if members.numel() else float("inf"))

        self._centroids = centroids
        self._dark_cluster = 0 if means[0] <= means[1] else 1
        self._since_fit = 0
        logger.info(
            "Team clusterer fit on %d samples; cluster %d is the darker side "
            "(mean L %.1f vs %.1f)",
            points.shape[0],
            self._dark_cluster,
            means[self._dark_cluster],
            means[1 - self._dark_cluster],
        )

    def observe(
        self,
        features: torch.Tensor,
        lightness: torch.Tensor,
    ) -> ClusterAssignment:
        """Record a batch of crops and assign whatever can be assigned.

        ``features`` is ``(N, D)`` and ``lightness`` is ``(N,)``.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be (N, D), got {tuple(features.shape)}")
        count = features.shape[0]
        if lightness.shape[0] != count:
            raise ValueError("features and lightness must agree on N")

        feats = features.detach().to(dtype=torch.float32, device="cpu")
        light = lightness.detach().to(dtype=torch.float32, device="cpu")

        for i in range(count):
            self._buffer.append(feats[i])
            self._lightness.append(float(light[i]))
            self._since_fit += 1
        if len(self._buffer) > self._max_buffer:
            # Keep the most recent window so lighting drift over a period is
            # tracked rather than averaged away against first-period crops.
            drop = len(self._buffer) - self._max_buffer
            del self._buffer[:drop]
            del self._lightness[:drop]

        should_fit = len(self._buffer) >= self._warmup_samples and (
            self._centroids is None or self._since_fit >= self._refit_interval
        )
        if should_fit:
            self._fit()

        clusters = torch.full((count,), -1, dtype=torch.int64)
        margins = torch.zeros((count,), dtype=torch.float32)
        if self._centroids is None:
            return ClusterAssignment(clusters=clusters, margins=margins)

        dists = torch.cdist(self._standardise(feats), self._centroids)
        nearest = torch.argmin(dists, dim=1)
        d0, d1 = dists[:, 0], dists[:, 1]
        margin = (d0 - d1).abs() / (d0 + d1).clamp_min(1e-6)

        assignable = margin >= self._min_margin
        clusters[assignable] = nearest[assignable]
        margins[assignable] = margin[assignable]
        return ClusterAssignment(clusters=clusters, margins=margins)
