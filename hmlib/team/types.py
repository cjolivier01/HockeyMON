"""Shared vocabulary for team affiliation.

A track's team is only meaningful relative to the game it was observed in: there
is no transferable "home" appearance, so nothing here predicts ``HOME`` directly.
The clusterer splits players into two appearance groups, ranks those groups
dark-vs-light, and :func:`name_clusters` applies the per-game ``game.home_is_dark``
bit to name them. With that bit unset the labels stay ``DARK``/``LIGHT`` rather
than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TeamLabel(IntEnum):
    """Per-track affiliation.

    ``DARK``/``LIGHT`` are the honest answer when the game config does not say
    which sweater the home side wears; ``HOME``/``AWAY`` are only produced once
    ``game.home_is_dark`` resolves the mapping.
    """

    UNKNOWN = -1
    HOME = 0
    AWAY = 1
    DARK = 2
    LIGHT = 3


#: Display text per label. Kept separate from the enum names so the overlay can
#: stay short without renaming the enum.
TEAM_LABEL_TEXT = {
    TeamLabel.UNKNOWN: "?",
    TeamLabel.HOME: "HOME",
    TeamLabel.AWAY: "AWAY",
    TeamLabel.DARK: "DARK",
    TeamLabel.LIGHT: "LIGHT",
}

#: BGR draw colours.
TEAM_LABEL_COLOR = {
    TeamLabel.UNKNOWN: (128, 128, 128),
    TeamLabel.HOME: (60, 60, 220),
    TeamLabel.AWAY: (220, 160, 60),
    TeamLabel.DARK: (60, 60, 220),
    TeamLabel.LIGHT: (220, 160, 60),
}


@dataclass
class TrackTeamInfo:
    """One frame's affiliation observation for one track.

    Mirrors :class:`hmlib.jersey.number_classifier.TrackJerseyInfo` so the two
    pipelines aggregate the same way.

    ``cluster`` is the raw 0/1 group index before naming, kept so the aggregator
    can vote on a stable quantity even when the dark/light ranking flips while
    centroids are still settling. ``margin`` is the normalised distance gap
    between the two centroids; low values mean the crop sat between the clusters
    and the observation should carry little weight.
    """

    tracking_id: int = -1
    label: TeamLabel = TeamLabel.UNKNOWN
    score: float = 0.0
    cluster: int = -1
    margin: float = 0.0


def name_clusters(
    dark_cluster: int,
    home_is_dark: bool | None,
) -> tuple[TeamLabel, TeamLabel]:
    """Map cluster indices 0/1 onto display labels.

    ``dark_cluster`` is whichever of the two clusters has the lower mean
    lightness. Returns ``(label_for_cluster_0, label_for_cluster_1)``.

    ``home_is_dark`` of ``None`` means the game config did not say, so the
    result is ``DARK``/``LIGHT`` -- correct but unnamed -- rather than an even
    bet on which side is home.
    """
    if dark_cluster not in (0, 1):
        raise ValueError(f"dark_cluster must be 0 or 1, got {dark_cluster!r}")

    if home_is_dark is None:
        dark_label, light_label = TeamLabel.DARK, TeamLabel.LIGHT
    elif home_is_dark:
        dark_label, light_label = TeamLabel.HOME, TeamLabel.AWAY
    else:
        dark_label, light_label = TeamLabel.AWAY, TeamLabel.HOME

    if dark_cluster == 0:
        return dark_label, light_label
    return light_label, dark_label
