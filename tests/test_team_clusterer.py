"""Per-game two-way clustering of player appearance."""

from __future__ import annotations

import pytest
import torch

from hmlib.team.clusterer import OnlineTeamClusterer, _kmeans2


def _two_populations(n: int = 120, spread: float = 0.4):
    """Two well-separated Lab-like populations: a dark side and a light side."""
    generator = torch.Generator().manual_seed(1234)
    dark = torch.tensor([20.0, 2.0, -4.0]) + torch.randn((n, 3), generator=generator) * spread
    light = torch.tensor([88.0, -1.0, 3.0]) + torch.randn((n, 3), generator=generator) * spread
    features = torch.cat([dark, light])
    lightness = features[:, 0].clone()
    truth = torch.cat([torch.zeros(n), torch.ones(n)])
    return features, lightness, truth


def should_split_two_separated_populations():
    centroids = _kmeans2(_two_populations()[0])
    assert centroids.shape == (2, 3)
    # The two centroids should land near the two population means.
    lightnesses = sorted(float(c[0]) for c in centroids)
    assert abs(lightnesses[0] - 20.0) < 3.0
    assert abs(lightnesses[1] - 88.0) < 3.0


def should_refuse_to_cluster_a_single_point():
    with pytest.raises(ValueError, match="at least two points"):
        _kmeans2(torch.zeros((1, 3)))


def should_report_nothing_before_warmup():
    clusterer = OnlineTeamClusterer(warmup_samples=50)
    features, lightness, _ = _two_populations(n=10)
    out = clusterer.observe(features, lightness)
    assert not clusterer.ready
    assert int(out.clusters.max()) == -1
    assert clusterer.dark_cluster is None


def should_recover_the_true_split_after_warmup():
    clusterer = OnlineTeamClusterer(warmup_samples=100, min_margin=0.0)
    features, lightness, truth = _two_populations()
    clusterer.observe(features, lightness)
    assert clusterer.ready

    out = clusterer.observe(features, lightness)
    predicted = out.clusters.to(torch.float32)
    agreement = max(
        float((predicted == truth).float().mean()),
        float((predicted == (1 - truth)).float().mean()),
    )
    assert agreement > 0.99


def should_identify_the_darker_cluster():
    clusterer = OnlineTeamClusterer(warmup_samples=100, min_margin=0.0)
    features, lightness, _ = _two_populations()
    out = clusterer.observe(features, lightness)

    dark = clusterer.dark_cluster
    assert dark in (0, 1)
    # Every sample the clusterer put in `dark` should be from the dark half.
    in_dark = out.clusters == dark
    assert float(lightness[in_dark].mean()) < 50.0


def should_leave_ambiguous_crops_unassigned():
    """A crop equidistant from both centroids is UNKNOWN, not a coin flip."""
    clusterer = OnlineTeamClusterer(warmup_samples=100, min_margin=0.25)
    features, lightness, _ = _two_populations()
    clusterer.observe(features, lightness)

    midpoint = torch.tensor([[54.0, 0.5, -0.5]])
    out = clusterer.observe(midpoint, midpoint[:, 0])
    assert int(out.clusters[0]) == -1
    assert float(out.margins[0]) == 0.0


def should_bound_the_sample_buffer():
    clusterer = OnlineTeamClusterer(warmup_samples=10, max_buffer=50)
    features, lightness, _ = _two_populations(n=100)
    clusterer.observe(features, lightness)
    assert clusterer.buffered == 50


def should_reject_mismatched_batch_shapes():
    clusterer = OnlineTeamClusterer()
    with pytest.raises(ValueError, match="agree on N"):
        clusterer.observe(torch.zeros((4, 3)), torch.zeros(3))


def should_be_deterministic_across_runs():
    """A random seed would make the overlay flicker between identical runs."""
    features, lightness, _ = _two_populations()
    first = OnlineTeamClusterer(warmup_samples=100, min_margin=0.0)
    second = OnlineTeamClusterer(warmup_samples=100, min_margin=0.0)
    a = first.observe(features, lightness)
    b = second.observe(features, lightness)
    assert torch.equal(a.clusters, b.clusters)
    assert first.dark_cluster == second.dark_cluster
