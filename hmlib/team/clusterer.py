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


#: How far a refit's centroids may sit from the previous pair, as a fraction
#: of the squared distance between those two, before the pairing between them
#: is disbelieved. Not a tuning knob: it asks whether the two fits describe
#: the same division at all.
#:
#: Deliberately a distance ratio rather than an angle between the separating
#: axes. A cosine threshold is dimension-dependent -- for a random direction
#: P(|cos| >= 0.5) is 0.5 at D=3 and ~0 at D=256 -- so one constant cannot
#: serve both the 3-D `lab` feature and the 256-D embedding, and at D=3 it
#: would be a coin flip in exactly the case it exists to catch. This ratio
#: behaves the same at every width: two fits of the same squads leave each
#: centroid near where it was (measured median 0.01 under lighting drift),
#: while a fit of one squad split down the middle must match one centroid to
#: the far old group and cannot get below ~0.78.
_MAX_PAIRING_DRIFT = 0.25

#: Consecutive refits that may be rejected before the split is abandoned and
#: relearned. Without a bound, a real change of scene leaves every new fit
#: disagreeing with a stale one forever.
_MAX_CONSECUTIVE_REJECTIONS = 3


def _pairing(current: torch.Tensor, previous: torch.Tensor) -> tuple[float, bool]:
    """``(drift, swapped)`` relating a new centroid pair to the previous one.

    ``drift`` is the better of the two matchings, as a fraction of the old
    pair's squared separation. ``swapped`` says the new fit numbered the
    groups the other way round.
    """
    keep = float(((current[0] - previous[0]) ** 2).sum() + ((current[1] - previous[1]) ** 2).sum())
    swap = float(((current[0] - previous[1]) ** 2).sum() + ((current[1] - previous[0]) ** 2).sum())
    separation = float(((previous[0] - previous[1]) ** 2).sum())
    if separation <= 1e-12:
        return float("inf"), False
    return min(keep, swap) / separation, swap < keep


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
        min_lightness_gap: float = 6.0,
        min_cluster_share: float = 0.25,
    ) -> None:
        if warmup_samples < 2:
            raise ValueError(f"warmup_samples must be >= 2, got {warmup_samples}")
        self._warmup_samples = int(warmup_samples)
        self._refit_interval = int(refit_interval)
        self._max_buffer = int(max_buffer)
        self._min_margin = float(min_margin)
        self._min_lightness_gap = float(min_lightness_gap)
        self._min_cluster_share = float(min_cluster_share)

        self._buffer: list[torch.Tensor] = []
        self._lightness: list[float] = []
        self._centroids: torch.Tensor | None = None
        self._scale_mean: torch.Tensor | None = None
        self._scale_std: torch.Tensor | None = None
        self._dark_cluster: int | None = None
        self._pending_dark: int | None = None
        self._balanced = False
        self._rejections = 0
        self._since_fit = 0

    @property
    def ready(self) -> bool:
        """True once centroids exist and assignments are meaningful."""
        return self._centroids is not None

    @property
    def dark_cluster(self) -> int | None:
        """Index of the lower-lightness cluster, or ``None`` when unnameable.

        Absent means the overlay names nobody, which is the honest answer in
        two cases. The two groups may be too close in lightness to rank: the
        torso crop takes in ice, boards and other skaters, so the per-crop
        mean is noisy, and ranking on a gap smaller than that noise lands
        differently at each refit and swaps every badge on screen together.
        Or the split may be too lopsided to be two squads -- most often the
        officials against everyone else, since they are plain detections to
        this clusterer -- and badging every skater the same with the referees
        opposite is worse than saying nothing.
        """
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
        previous_centroids = self._raw_centroids()
        previous_scale = (self._scale_mean, self._scale_std)

        # Lab axes span very different ranges (L is 0..100, a/b are +-128), and
        # a raw euclidean fit would be dominated by whichever axis is widest.
        self._scale_mean = points.mean(dim=0)
        self._scale_std = points.std(dim=0).clamp_min(1e-6)
        scaled = self._standardise(points)
        centroids = _kmeans2(scaled)

        labels = torch.argmin(torch.cdist(scaled, centroids), dim=1)
        counts = [int((labels == k).sum()) for k in range(2)]
        balanced = min(counts) >= self._min_cluster_share * points.shape[0]

        if previous_centroids is not None:
            current = centroids * self._scale_std + self._scale_mean
            drift, swapped = _pairing(current, previous_centroids)
            # Once a split is good enough to publish, protect it. k-means
            # reseeds from whatever is buffered now, so nothing ties this
            # fit's index 0 to the last one's, and the tracker keys a
            # settled skater's evidence on that index. The window can hold
            # one squad for minutes -- sustained zone play, or settled
            # tracks no longer being sampled -- and k=2 then splits that one
            # squad down the middle, leaving the pairing a coin flip. Keep
            # the previous split rather than rename the game on a guess.
            # Protect a split only once it is *named*, and only for a bounded
            # number of refits.
            #
            # Protecting from `_balanced` deadlocked the feature. `_balanced`
            # latches on the first fit that clears `min_cluster_share` -- and
            # k-means on a single squad splits it near 50/50, so the warmup
            # fit qualifies even though the class docstring calls that window
            # untrustworthy. From then on every later, better fit disagreed
            # with it and was rejected before the naming block, so the sides
            # were never named at all. There is nothing to protect before a
            # name exists: no badge is on screen for an index to be wrong
            # about.
            #
            # The bound matters too. A genuine change of scene -- a period
            # break, a lighting change -- leaves every new fit disagreeing
            # with a stale one forever. After a few consecutive rejections
            # the stored split is the thing that is wrong, so take the new
            # fit and relearn the naming rather than keep a name that no
            # longer describes the ice.
            protecting = self._dark_cluster is not None
            disagrees = not balanced or drift > _MAX_PAIRING_DRIFT
            if protecting and disagrees and self._rejections < _MAX_CONSECUTIVE_REJECTIONS:
                self._scale_mean, self._scale_std = previous_scale
                self._rejections += 1
                self._since_fit = 0
                logger.info(
                    "Team clusterer kept its previous split (%s); %d consecutive",
                    "window is not two squads" if not balanced else f"centroid drift {drift:.2f}",
                    self._rejections,
                )
                return
            if protecting and disagrees:
                logger.warning(
                    "Team clusterer abandoning its split after %d rejected refits; "
                    "relearning which side is which",
                    self._rejections,
                )
                self._dark_cluster = None
                self._pending_dark = None
            self._rejections = 0
            if swapped:
                centroids = centroids.flip(0)
                labels = 1 - labels
                counts.reverse()

        lightness = torch.tensor(self._lightness, dtype=torch.float32)
        means = []
        for k in range(2):
            members = lightness[labels == k]
            means.append(float(members.mean()) if members.numel() else None)

        self._centroids = centroids
        self._balanced = balanced
        self._since_fit = 0

        # Rank once, from two consecutive fits that agree, and keep it.
        #
        # Keeping it is what stops every badge on screen swapping together at
        # a refit. Requiring two fits to agree is what stops the ranking being
        # latched from one noisy window: the gap between two cluster means is
        # itself an estimate, and on a small sample two squads of genuinely
        # equal lightness clear a fixed threshold often enough to matter.
        # Sweaters do not change mid-game, so a real gap survives to the next
        # fit while a sampling artifact does not.
        candidate = None
        if balanced and None not in means and abs(means[0] - means[1]) >= self._min_lightness_gap:
            candidate = 0 if means[0] <= means[1] else 1
        if self._dark_cluster is None and candidate is not None:
            if self._pending_dark == candidate:
                self._dark_cluster = candidate
            self._pending_dark = candidate
        elif candidate is None:
            self._pending_dark = None

        if self._dark_cluster is None:
            logger.info(
                "Team clusterer fit on %d samples (%d/%d) but cannot name the sides yet",
                points.shape[0],
                counts[0],
                counts[1],
            )
        else:
            logger.info(
                "Team clusterer fit on %d samples (%d/%d); cluster %d is the darker side",
                points.shape[0],
                counts[0],
                counts[1],
                self._dark_cluster,
            )

    def _raw_centroids(self) -> torch.Tensor | None:
        """Centroids back in feature space, the only frame two fits share."""
        if self._centroids is None or self._scale_mean is None or self._scale_std is None:
            return None
        return self._centroids * self._scale_std + self._scale_mean

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
