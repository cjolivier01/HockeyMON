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
    # Two fits: the ranking is latched only once consecutive fits agree, so
    # that one noisy window cannot name the sides for the rest of the game.
    clusterer = OnlineTeamClusterer(warmup_samples=100, refit_interval=1, min_margin=0.0)
    features, lightness, _ = _two_populations()
    clusterer.observe(features, lightness)
    assert clusterer.dark_cluster is None, "a single fit must not name the sides"
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


def _squad(rng: torch.Generator, lightness: float, chroma: tuple[float, float]) -> torch.Tensor:
    """One Lab torso feature from a squad centred at `lightness`/`chroma`."""
    noise = torch.randn(3, generator=rng) * torch.tensor([3.0, 2.0, 2.0])
    return torch.tensor([lightness, chroma[0], chroma[1]]) + noise


def should_keep_cluster_identity_across_a_refit():
    """A refit must not rename the squads under already-settled tracks.

    k-means reseeds from whatever is buffered, so nothing ties this fit's
    index 0 to the last one's -- and `TeamTracker` keys a skater's accumulated
    evidence on that index. An index swap silently inverts every track that
    settled before the refit.
    """
    for seed in range(8):
        rng = torch.Generator().manual_seed(seed)
        clusterer = OnlineTeamClusterer(warmup_samples=40, refit_interval=40, min_margin=0.05)
        dark, light = (28.0, (5.0, -30.0)), (70.0, (-4.0, 28.0))
        for i in range(120):
            spec = dark if i % 2 else light
            clusterer.observe(_squad(rng, *spec).view(1, 3), torch.tensor([spec[0]]))
        probe = torch.tensor([dark[0], dark[1][0], dark[1][1]]).view(1, 3)
        before = int(clusterer.observe(probe, torch.tensor([dark[0]])).clusters[0])
        assert before >= 0

        # Lighting drifts and play stays balanced across several refits.
        for i in range(400):
            spec = dark if i % 2 else light
            feat = _squad(rng, *spec).view(1, 3) + torch.tensor([4.0, 1.0, 1.0])
            clusterer.observe(feat, torch.tensor([spec[0] + 4.0]))
        after = int(clusterer.observe(probe, torch.tensor([dark[0]])).clusters[0])
        assert after in (-1, before), f"seed {seed}: the dark squad changed cluster index"


def should_not_name_sides_it_cannot_tell_apart():
    """Two squads of equal lightness must stay unnamed, not flip every refit."""
    rng = torch.Generator().manual_seed(3)
    clusterer = OnlineTeamClusterer(warmup_samples=40, refit_interval=40, min_margin=0.05)
    # Distinct in chroma -- red against blue -- but the same lightness.
    red, blue = (50.0, (45.0, 25.0)), (50.0, (-20.0, -40.0))
    ranks = []
    for i in range(400):
        spec = red if i % 2 else blue
        clusterer.observe(_squad(rng, *spec).view(1, 3), torch.tensor([spec[0]]))
        if clusterer.ready:
            ranks.append(clusterer.dark_cluster)
    assert ranks, "never fitted"
    assert all(r == ranks[0] for r in ranks), "the dark/light ranking changed mid-game"
    assert ranks[0] is None, "equal-lightness squads must not be named dark and light"


def should_refuse_a_split_that_is_not_two_squads():
    """Officials against everyone is lopsided; naming it is worse than silence.

    The two squads here are indistinguishable, so the only real structure in
    the feature space is the officials -- and they are plain detections to
    this clusterer. Left unguarded the fit isolates them, which has a large
    and entirely genuine lightness gap, so the ranking guard does not catch
    it: every skater would be badged one side and the referees the other.
    """
    rng = torch.Generator().manual_seed(7)
    clusterer = OnlineTeamClusterer(warmup_samples=40, refit_interval=40, min_margin=0.05)
    for i in range(300):
        official = i % 6 == 0
        spec = (85.0, (40.0, 40.0)) if official else (45.0, (0.0, 0.0))
        clusterer.observe(_squad(rng, *spec).view(1, 3), torch.tensor([spec[0]]))
    assert clusterer.ready
    assert clusterer.dark_cluster is None, "an officials-against-everyone split was named"


def should_standardise_axes_of_very_different_scale():
    """Lab axes span different ranges, so a raw euclidean fit is dominated by L.

    The squads here differ only in chroma, which is the narrow axis; L is
    noisy and much wider. Without standardisation the fit follows the noise
    and the split carries no team signal, so dropping `_standardise` -- or
    running zero Lloyd iterations, which leaves the seeds unrefined -- both
    show up as assignments that disagree with the squad that produced them.
    """
    rng = torch.Generator().manual_seed(11)
    clusterer = OnlineTeamClusterer(
        warmup_samples=60, refit_interval=60, min_margin=0.02, min_lightness_gap=0.0
    )

    def sample(second: bool) -> torch.Tensor:
        # L: wide and identical for both squads. a/b: narrow and separating.
        lightness = 50.0 + float(torch.randn(1, generator=rng)) * 25.0
        chroma = 6.0 if second else -6.0
        ab = torch.randn(2, generator=rng) * 1.5 + chroma
        return torch.tensor([lightness, float(ab[0]), float(ab[1])])

    for i in range(400):
        clusterer.observe(sample(i % 2 == 1).view(1, 3), torch.tensor([50.0]))

    agree = total = 0
    for i in range(120):
        second = i % 2 == 1
        out = clusterer.observe(sample(second).view(1, 3), torch.tensor([50.0]))
        cluster = int(out.clusters[0])
        if cluster < 0:
            continue
        total += 1
        agree += int(
            cluster
            == int(
                clusterer.observe(
                    (
                        torch.tensor([50.0, 6.0, 6.0])
                        if second
                        else torch.tensor([50.0, -6.0, -6.0])
                    ).view(1, 3),
                    torch.tensor([50.0]),
                ).clusters[0]
            )
        )
    assert total > 60, f"only {total} of 120 crops were assignable"
    assert agree / total > 0.9, f"only {agree}/{total} agreed with their squad"


def should_keep_the_previous_split_when_the_window_holds_one_squad():
    """The rejection branch itself, on the 3-D feature where it is hardest.

    A settled track's evidence is keyed on the cluster index, so renaming the
    groups inverts it permanently. The window goes single-squad whenever play
    stays in one zone, and k=2 then splits that squad down the middle -- two
    centroids inside one old group, which no pairing can relate to the old
    pair. Deleting this branch left every other test green, and an angle-based
    guard is a coin flip at this width.
    """
    flips = 0
    for seed in range(40):
        rng = torch.Generator().manual_seed(seed)
        clusterer = OnlineTeamClusterer(
            warmup_samples=60, refit_interval=60, min_margin=0.02, max_buffer=120
        )
        dark, light = (25.0, (6.0, -24.0)), (72.0, (-6.0, 26.0))

        def sample(spec, rng=rng):
            noise = torch.randn(3, generator=rng) * torch.tensor([3.0, 2.0, 2.0])
            return (torch.tensor([spec[0], spec[1][0], spec[1][1]]) + noise).view(1, 3)

        for i in range(240):
            spec = dark if i % 2 else light
            clusterer.observe(sample(spec), torch.tensor([spec[0]]))
        probe = torch.tensor([dark[0], dark[1][0], dark[1][1]]).view(1, 3)
        before = int(clusterer.observe(probe, torch.tensor([dark[0]])).clusters[0])
        if before < 0 or clusterer.dark_cluster is None:
            continue

        # Sustained zone play: only the dark squad is sampled, long enough to
        # fill the buffer and cross several refit intervals.
        for _ in range(300):
            clusterer.observe(sample(dark), torch.tensor([dark[0]]))
        after = int(clusterer.observe(probe, torch.tensor([dark[0]])).clusters[0])
        # The invariant is about what the operator sees: a *named* side must
        # never silently invert. A long enough one-squad stretch eventually
        # abandons the split and relearns -- that is the escape from the
        # deadlock, and it is safe precisely because it unnames first, so no
        # badge is on screen while the index is in flux.
        if after >= 0 and after != before and clusterer.dark_cluster is not None:
            flips += 1
    assert flips == 0, f"{flips}/40 seeds renamed a named split from a one-squad window"


def should_not_deadlock_on_an_untrustworthy_warmup_fit():
    """A bad first fit must not lock the feature out of ever naming the sides.

    `min_cluster_share` passes on a single squad -- k-means splits it near
    50/50 -- so the warmup fit can latch a split that describes nothing. An
    earlier version protected any balanced split from then on, which rejected
    every later, better fit before the naming block and left `dark_cluster`
    None for the whole game, with only a repeating log line to show for it.
    """
    rng = torch.Generator().manual_seed(5)
    clusterer = OnlineTeamClusterer(warmup_samples=60, refit_interval=60, min_margin=0.02)
    dark, light = (25.0, (6.0, -24.0)), (72.0, (-6.0, 26.0))

    def sample(spec):
        noise = torch.randn(3, generator=rng) * torch.tensor([3.0, 2.0, 2.0])
        return (torch.tensor([spec[0], spec[1][0], spec[1][1]]) + noise).view(1, 3)

    # Warmup sees one squad only: the fit is balanced but meaningless.
    for _ in range(80):
        clusterer.observe(sample(dark), torch.tensor([dark[0]]))
    assert clusterer.ready and clusterer.dark_cluster is None

    # Both squads from here on. Recovery takes three refits -- the buffer
    # still holds the one-squad warmup crops for the first of them, and the
    # ranking latch needs two agreeing fits after that. What matters is that
    # it recovers at all: the version this guards against rejected every
    # later fit before the naming block and stayed unnamed for the whole
    # game.
    for i in range(300):
        spec = dark if i % 2 else light
        clusterer.observe(sample(spec), torch.tensor([spec[0]]))
    assert clusterer.dark_cluster is not None, "never recovered from the warmup fit"

    # And the naming is right: the dark squad lands in the dark cluster.
    probe = torch.tensor([dark[0], dark[1][0], dark[1][1]]).view(1, 3)
    assigned = int(clusterer.observe(probe, torch.tensor([dark[0]])).clusters[0])
    assert assigned == clusterer.dark_cluster, "the darker squad is not the dark cluster"


def should_relearn_when_the_ice_stops_matching_its_split():
    """A named split must not be protected forever against a changed scene.

    Protecting it is right for a temporary one-squad window. It is wrong when
    the scene genuinely changes -- a period break, a lighting change, a
    different pair of teams -- because then every new fit disagrees with a
    stale one and the feature would keep publishing a name that no longer
    describes anybody.
    """
    rng = torch.Generator().manual_seed(9)
    clusterer = OnlineTeamClusterer(
        warmup_samples=60, refit_interval=60, min_margin=0.02, max_buffer=120
    )

    def sample(spec):
        noise = torch.randn(3, generator=rng) * torch.tensor([3.0, 2.0, 2.0])
        return (torch.tensor([spec[0], spec[1][0], spec[1][1]]) + noise).view(1, 3)

    first = ((25.0, (6.0, -24.0)), (72.0, (-6.0, 26.0)))
    for i in range(300):
        spec = first[i % 2]
        clusterer.observe(sample(spec), torch.tensor([spec[0]]))
    assert clusterer.dark_cluster is not None
    stale_probe = torch.tensor([first[0][0], first[0][1][0], first[0][1][1]]).view(1, 3)
    assert int(clusterer.observe(stale_probe, torch.tensor([first[0][0]])).clusters[0]) >= 0

    # Entirely different appearance: both squads elsewhere in the space.
    second = ((48.0, (40.0, 38.0)), (52.0, (-38.0, -40.0)))
    for i in range(600):
        spec = second[i % 2]
        clusterer.observe(sample(spec), torch.tensor([spec[0]]))

    # It must have moved on: the new population is assignable under the
    # centroids in force, rather than everything falling outside them.
    assigned = sum(
        int(clusterer.observe(sample(second[i % 2]), torch.tensor([second[i % 2][0]])).clusters[0])
        >= 0
        for i in range(40)
    )
    assert assigned > 30, f"only {assigned}/40 of the new scene was assignable; the split is stale"
