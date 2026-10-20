"""Graph-level checks for the team affiliation trunk.

Two things can only go wrong at the graph level: a published key that never
reaches the overlay because `minimal_context` filtered it out, and a trunk
inserted in a position that breaks AspenNet's fan-in or no-diamond rules. The
first is silent (the overlay simply draws nothing), so it is asserted here
rather than discovered on a clip.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, ClassVar

import yaml

from hmlib.aspen import AspenNet, Plugin


def _repo_root() -> Path:
    """Read the staged copies under Bazel, the checkout under bare pytest.

    Runfiles are a symlink tree, so `Path(__file__).resolve()` escapes back
    into the source checkout and the target's `data` list stops being
    load-bearing: it would keep passing with every staged graph broken, while
    Bazel quietly stopped re-running it when one changed. Mirrors
    `tests/test_aspen_graph_baseline_params.py`, which solved this first.
    """
    srcdir, workspace = os.environ.get("TEST_SRCDIR"), os.environ.get("TEST_WORKSPACE")
    if srcdir and workspace:
        return Path(srcdir) / workspace
    return Path(__file__).resolve().parents[1]


_GRAPHS = _repo_root() / "hmlib" / "config" / "aspen"
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


def _graph_specs():
    """Every aspen graph as ``(path, plugins)``."""
    for path in sorted(_GRAPHS.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text())
        plugins = ((document or {}).get("aspen") or {}).get("plugins") or {}
        if plugins:
            yield path, plugins


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


def should_declare_the_team_trunk_wherever_team_labels_are_drawn():
    """A graph that wires `plot_teams` must have something to publish them.

    This replaces a test that enforced an ordering rule on the grounds that
    the trunk "refines `track_mask`" -- it does not, and
    `tests/test_team_plugin.py` asserts it does not. The real failure was the
    opposite one: eleven graphs passed `plot_teams` to the overlay with no
    trunk in them at all, so `--plot-teams` on the default pose graph set the
    flag, enabled the feature and drew nothing.
    """
    missing = []
    for path, plugins in _graph_specs():
        draws = any(
            isinstance(spec, dict) and "plot_teams" in (spec.get("params") or {})
            for spec in plugins.values()
        )
        declares = any(
            isinstance(spec, dict) and spec.get("class") == _TEAM_TRUNK for spec in plugins.values()
        )
        if draws and not declares:
            missing.append(path.name)
    assert not missing, f"graphs draw team labels with no trunk to publish them: {missing}"


def should_place_the_team_trunk_after_the_tracks_it_reads():
    """The trunk needs `pred_track_instances`, so the track source must precede it.

    Inserting it next to the drawer is not enough: in
    `tracking_pose_pose_det.yaml` the tracker is downstream of pose, so a
    trunk placed beside the overlay ran before any tracks existed and
    published an empty list every frame -- silently, because the trunk
    returns early when `pred_track_instances` is absent.
    """
    for path, plugins in _graphs_with_team_trunk():
        depends = {
            name: list((spec or {}).get("depends") or [])
            for name, spec in plugins.items()
            if isinstance(spec, dict)
        }
        sources = [n for n in ("tracker", "load_tracking") if n in plugins]
        assert sources, f"{path.name}: no track source to read from"
        reachable, stack = set(), list(depends["team_affiliation"])
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(depends.get(current, []))
        assert (
            set(sources) & reachable
        ), f"{path.name}: the team trunk runs before {sources}, so it sees no tracks"


def should_order_the_team_trunk_before_the_overlay():
    """The overlay reads `team_results`, so it cannot run beside the writer."""
    for path, plugins in _graphs_with_team_trunk():
        team_names = {name for name, spec in plugins.items() if spec.get("class") == _TEAM_TRUNK}
        for name, spec in plugins.items():
            if not isinstance(spec, dict):
                continue
            if "plot_teams" not in (spec.get("params") or {}):
                continue
            reachable, seen = set(spec.get("depends") or []), set()
            while reachable:
                current = reachable.pop()
                if current in seen:
                    continue
                seen.add(current)
                upstream = plugins.get(current)
                if isinstance(upstream, dict):
                    reachable.update(upstream.get("depends") or [])
            assert (
                team_names & seen
            ), f"{path.name}: {name} draws team labels but does not depend on the trunk"


def should_build_every_graph_that_declares_the_team_trunk():
    """Inserting a trunk can break AspenNet's fan-in or no-diamond rules.

    Disabling the heavy trunks substitutes no-op stubs, which preserves the
    graph shape, so the edges are validated without constructing detectors or
    pose models. JoinPlugin nodes stay enabled because only a real join carries
    `allow_multi_path_inputs`, and a stub in its place fails the fan-in check
    for reasons that have nothing to do with the team trunk.
    """

    def stubbed(graph):
        graph = copy.deepcopy(graph)
        for spec in graph["plugins"].values():
            if isinstance(spec, dict) and "JoinPlugin" not in spec.get("class", ""):
                spec["enabled"] = False
        return graph

    already_broken = []
    for path, _ in _graphs_with_team_trunk():
        graph = yaml.safe_load(path.read_text())["aspen"]
        without = copy.deepcopy(graph)
        without["plugins"] = {
            name: spec
            for name, spec in without["plugins"].items()
            if not (isinstance(spec, dict) and spec.get("class") == _TEAM_TRUNK)
        }
        for spec in without["plugins"].values():
            if isinstance(spec, dict) and "team_affiliation" in (spec.get("depends") or []):
                spec["depends"] = [d for d in spec["depends"] if d != "team_affiliation"] or list(
                    graph["plugins"]["team_affiliation"].get("depends") or []
                )
        try:
            AspenNet(path.stem, stubbed(without))
        except ValueError:
            # Already unbuildable without the trunk. `tracking_pose.yaml` and
            # `tracking_pose_actions.yaml` are in this state on master:
            # `camera_controller` and `ice_boundaries` take two dependencies
            # each and only JoinPlugin declares `allow_multi_path_inputs`.
            # Pre-existing and out of scope here, but the trunk must not be
            # blamed for it, nor hide behind it.
            already_broken.append(path.name)
            continue
        AspenNet(path.stem, stubbed(graph))

    assert len(already_broken) < len(
        _graphs_with_team_trunk()
    ), "every graph is already unbuildable, so this guard proved nothing"


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
