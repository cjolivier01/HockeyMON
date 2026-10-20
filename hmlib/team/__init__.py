"""Team affiliation: which side of the game a tracked player belongs to."""

from .backends import BACKEND_NAMES, LAB_BACKEND, PRTREID_BACKEND, BackendOutput, build_backend
from .clusterer import ClusterAssignment, OnlineTeamClusterer
from .team_tracker import TeamLabelRenderer, TeamTracker
from .types import TEAM_LABEL_COLOR, TEAM_LABEL_TEXT, TeamLabel, TrackTeamInfo, name_clusters

__all__ = [
    "BACKEND_NAMES",
    "LAB_BACKEND",
    "PRTREID_BACKEND",
    "TEAM_LABEL_COLOR",
    "TEAM_LABEL_TEXT",
    "BackendOutput",
    "ClusterAssignment",
    "OnlineTeamClusterer",
    "TeamLabel",
    "TeamLabelRenderer",
    "TeamTracker",
    "TrackTeamInfo",
    "build_backend",
    "name_clusters",
]
