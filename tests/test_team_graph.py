"""Graph-level checks for the team affiliation trunk.

Two things can only go wrong at the graph level: a published key that never
reaches the overlay because `minimal_context` filtered it out, and a trunk
inserted in a position that breaks AspenNet's fan-in or no-diamond rules. The
first is silent (the overlay simply draws nothing), so it is asserted here
rather than discovered on a clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml

from hmlib.aspen import AspenNet, Plugin

_GRAPHS = Path(__file__).resolve().parents[1] / "hmlib" / "config" / "aspen"
_TEAM_TRUNK = "hmlib.aspen.plugins.team_plugin.TeamAffiliationPlugin"


class _FakeTeamTrunk(Plugin):
    """Stands in for the real trunk so the graph test needs no image data."""

    def forward(self, context: dict[str, Any]):
        return {"team_results": [["HOME"]]}

    def output_keys(self) -> set[str]:
        return {"team_results"}


class _FakeConsumer(Plugin):
    """Records what a downstream trunk actually receives."""

    seen: ClassVar[dict[str, Any]] = {}

    def forward(self, context: dict[str, Any]):
        type(self).seen = dict(context)
        return {}

    def input_keys(self) -> set[str]:
        return {"team_results"}


def should_deliver_team_results_downstream_under_minimal_context():
    graph = {
        "minimal_context": True,
        "plugins": {
            "team_affiliation": {
                "class": f"{__name__}._FakeTeamTrunk",
                "depends": [],
                "params": {},
            },
            "consumer": {
                "class": f"{__name__}._FakeConsumer",
                "depends": ["team_affiliation"],
                "params": {},
            },
        },
    }
    _FakeConsumer.seen = {}
    AspenNet("team_keys", graph)({})

    assert _FakeConsumer.seen.get("team_results") == [["HOME"]]


def _graphs_with_team_trunk():
    found = []
    for path in sorted(_GRAPHS.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text())
        plugins = ((document or {}).get("aspen") or {}).get("plugins") or {}
        if any(
            isinstance(spec, dict) and spec.get("class") == _TEAM_TRUNK for spec in plugins.values()
        ):
            found.append((path, plugins))
    return found


def should_declare_the_team_trunk_in_at_least_one_graph():
    """Guard the guard: an empty sweep would pass every assertion below."""
    assert _graphs_with_team_trunk(), "no graph declares the team trunk; the sweep is broken"


def should_order_the_team_trunk_before_every_track_consumer():
    """The trunk refines `track_mask`; readers that bypass it would race the write.

    Nothing may depend on `tracker` directly once the team trunk is present --
    the trunk itself excepted -- or under the threaded graph that consumer can
    run concurrently with the mask write and see officials.
    """
    for path, plugins in _graphs_with_team_trunk():
        team_names = {name for name, spec in plugins.items() if spec.get("class") == _TEAM_TRUNK}
        for name, spec in plugins.items():
            if name in team_names or not isinstance(spec, dict):
                continue
            depends = spec.get("depends") or []
            assert "tracker" not in depends, (
                f"{path.name}: trunk {name!r} depends on 'tracker' directly while "
                f"{sorted(team_names)} refines the track mask downstream of it; "
                "depend on the team trunk instead so the write is ordered"
            )


def should_build_every_graph_that_declares_the_team_trunk():
    """Inserting a trunk can break AspenNet's fan-in or no-diamond rules.

    Disabling the heavy trunks substitutes no-op stubs, which preserves the
    graph shape, so the edges are validated without constructing detectors or
    pose models. JoinPlugin nodes stay enabled because only a real join carries
    `allow_multi_path_inputs`, and a stub in its place fails the fan-in check
    for reasons that have nothing to do with the team trunk.
    """
    for path, _ in _graphs_with_team_trunk():
        document = yaml.safe_load(path.read_text())
        graph = document["aspen"]
        for spec in graph["plugins"].values():
            if isinstance(spec, dict) and "JoinPlugin" not in spec.get("class", ""):
                spec["enabled"] = False
        AspenNet(path.stem, graph)


def should_gate_the_team_trunk_on_the_baseline_switch():
    """`team.enabled` is the only way the trunk stays off by default.

    The individual params are guarded by `_BASELINE_BACKED` in
    tests/test_aspen_graph_baseline_params.py, which is the canonical place for
    that check; `enabled` is a sibling of `params` so it is not covered there.
    """
    for path, plugins in _graphs_with_team_trunk():
        for name, spec in plugins.items():
            if spec.get("class") != _TEAM_TRUNK:
                continue
            assert (
                spec.get("enabled") == "GLOBAL.team.enabled"
            ), f"{path.name}: trunk {name!r} must be gated on GLOBAL.team.enabled"
