"""Track-level aggregation and drawing for team affiliation.

Per-frame crops are noisy: a player turning, a partial occlusion, or a crop that
caught the boards will all produce a wrong single-frame answer. This accumulates
decayed evidence per track and switches the displayed label only when a
contender wins decisively, which is the same shape as
:class:`hmlib.jersey.jersey_tracker.JerseyTracker`.

Voting happens on the raw cluster index rather than the named label, so a
dark/light ranking that flips while centroids are still settling does not erase
the evidence collected so far.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from hmlib.log import logger
from hmlib.utils.image import is_channels_last
from hmlib.vis.labels import draw_label_box

from .types import TEAM_LABEL_COLOR, TEAM_LABEL_TEXT, TeamLabel, TrackTeamInfo


@dataclass
class TrackingIdTeamInfo:
    """Accumulated affiliation evidence for one track."""

    tracking_id: int = -1
    current_cluster: int = -1
    current_score: float = 0.0
    last_frame_id: int = -1
    cluster_scores: dict[int, float] = field(default_factory=dict)
    anchor_ema: tuple[float, float] | None = None


class TeamTracker:
    """Aggregates :class:`TrackTeamInfo` observations and renders labels."""

    def __init__(
        self,
        evidence_decay: float = 0.92,
        min_display_evidence: float = 1.0,
        switch_ratio: float = 1.35,
        anchor_ema_alpha: float = 0.55,
        home_is_dark: bool | None = None,
    ) -> None:
        self._evidence_decay = float(evidence_decay)
        self._min_display_evidence = float(min_display_evidence)
        self._switch_ratio = float(switch_ratio)
        self._anchor_ema_alpha = float(anchor_ema_alpha)
        self._home_is_dark = home_is_dark
        self._dark_cluster: int | None = None
        self._tracks: dict[int, TrackingIdTeamInfo] = {}

    def set_dark_cluster(self, dark_cluster: int | None) -> None:
        """Record which cluster index is the darker side, for naming."""
        self._dark_cluster = dark_cluster

    def _decay(self, info: TrackingIdTeamInfo, frame_id: int) -> None:
        if info.last_frame_id < 0:
            return
        dt = max(1, int(frame_id) - int(info.last_frame_id))
        decay = float(self._evidence_decay**dt)
        if decay >= 0.999:
            return
        for k in list(info.cluster_scores.keys()):
            v = float(info.cluster_scores.get(k, 0.0)) * decay
            if v < 1e-4:
                info.cluster_scores.pop(k, None)
            else:
                info.cluster_scores[k] = v

    def _select_cluster(self, info: TrackingIdTeamInfo) -> None:
        if not info.cluster_scores:
            info.current_cluster = -1
            info.current_score = 0.0
            return
        ranked = sorted(info.cluster_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_cluster, best_score = ranked[0]
        current_score = float(info.cluster_scores.get(info.current_cluster, 0.0))

        if info.current_cluster < 0:
            if best_score >= self._min_display_evidence:
                info.current_cluster = int(best_cluster)
                info.current_score = float(best_score)
            return

        if best_cluster == info.current_cluster:
            info.current_score = float(best_score)
            return

        if best_score >= self._min_display_evidence and best_score >= (
            current_score * self._switch_ratio
        ):
            logger.info(
                "Team switch: ID %s cluster %s -> %s (%.3f -> %.3f)",
                info.tracking_id,
                info.current_cluster,
                int(best_cluster),
                current_score,
                best_score,
            )
            info.current_cluster = int(best_cluster)
            info.current_score = float(best_score)

    def observe(self, frame_id: int, info: TrackTeamInfo) -> None:
        """Fold one frame's observation for one track into its running state."""
        tracking_id = int(info.tracking_id)
        if tracking_id < 0:
            return

        state = self._tracks.get(tracking_id)
        if state is None:
            state = TrackingIdTeamInfo(tracking_id=tracking_id)
            self._tracks[tracking_id] = state

        self._decay(state, frame_id=frame_id)
        state.last_frame_id = int(frame_id)

        cluster = int(info.cluster)
        if cluster >= 0:
            # Weight by margin so a crop that sat between the centroids counts
            # for less than one that landed squarely on a team colour.
            weight = max(float(info.score), 0.0) * max(float(info.margin), 0.0)
            state.cluster_scores[cluster] = state.cluster_scores.get(cluster, 0.0) + weight
            self._select_cluster(state)

    def label_for(self, tracking_id: int) -> TeamLabel:
        """Display label for a track, or ``UNKNOWN`` when undecided."""
        state = self._tracks.get(int(tracking_id))
        if state is None:
            return TeamLabel.UNKNOWN
        if state.current_cluster < 0 or self._dark_cluster is None:
            return TeamLabel.UNKNOWN

        # Imported here rather than at module scope to keep the naming policy
        # in one place; `types.name_clusters` owns the home_is_dark semantics.
        from .types import name_clusters

        label_0, label_1 = name_clusters(self._dark_cluster, self._home_is_dark)
        return label_0 if state.current_cluster == 0 else label_1


class TeamLabelRenderer:
    """Draws already-settled team labels.

    Kept separate from :class:`TeamTracker` because the aggregation happens in
    the team trunk while the drawing happens in the overlay trunk, which may be
    several nodes downstream. The renderer consumes the published
    :class:`TrackTeamInfo` list rather than re-deriving labels, so the overlay
    cannot disagree with the mask the trunk already applied.

    It keeps only smoothing state: an EMA over each track's anchor point, so the
    label does not jitter frame to frame with the detection box.
    """

    def __init__(self, anchor_ema_alpha: float = 0.55) -> None:
        self._anchor_ema_alpha = float(anchor_ema_alpha)
        self._labels: dict[int, TeamLabel] = {}
        self._anchors: dict[int, tuple[float, float]] = {}

    def update(self, results: Sequence[TrackTeamInfo]) -> None:
        """Record this frame's labels, keyed by track."""
        for info in results:
            tracking_id = int(info.tracking_id)
            if tracking_id >= 0:
                self._labels[tracking_id] = info.label

    def _anchor(self, tracking_id: int, anchor_xy: tuple[float, float]) -> tuple[float, float]:
        previous = self._anchors.get(tracking_id)
        if previous is None:
            self._anchors[tracking_id] = (float(anchor_xy[0]), float(anchor_xy[1]))
            return self._anchors[tracking_id]
        alpha = self._anchor_ema_alpha
        smoothed = (
            float(alpha * anchor_xy[0] + (1.0 - alpha) * previous[0]),
            float(alpha * anchor_xy[1] + (1.0 - alpha) * previous[1]),
        )
        self._anchors[tracking_id] = smoothed
        return smoothed

    def draw(
        self,
        image: torch.Tensor,
        tracking_ids: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Draw a team label under each track's box.

        ``bboxes`` are tlwh. The label sits below the box, because the jersey
        number anchors to the top edge and the two would otherwise overlap.
        """
        if len(tracking_ids) != len(bboxes):
            raise ValueError("tracking_ids and bboxes must be the same length")

        if isinstance(image, torch.Tensor):
            img_w = int(image.shape[-1] if not is_channels_last(image) else image.shape[1])
        else:
            img_w = int(image.shape[1])
        font_scale = max(0.6, float(img_w) / 1800.0)

        for (x1, y1, w, h), tracking_id in zip(bboxes, tracking_ids):
            tracking_id = int(tracking_id)
            label = self._labels.get(tracking_id, TeamLabel.UNKNOWN)
            if label == TeamLabel.UNKNOWN:
                continue

            ax, ay = self._anchor(tracking_id, (float(x1 + (w * 0.5)), float(y1 + h)))
            image = draw_label_box(
                image=image,
                text=TEAM_LABEL_TEXT[label],
                center_x=int(ax),
                bottom_y=int(ay) + int(font_scale * 14),
                font_scale=font_scale,
                fg=(255, 255, 255),
                bg=TEAM_LABEL_COLOR[label],
                bg_alpha=200,
            )
        return image
