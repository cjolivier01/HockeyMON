"""Guard the aspen graphs against silently inverting a shared baseline value.

`AspenNet._instantiate` calls `cls(**params)`, so a plugin parameter a graph
omits falls back to the constructor's own literal. When that literal disagrees
with `baseline.yaml` the graph runs the opposite of the shipped default, and
nothing says so -- which is how `play_tracker.no_wide_start` came to be `true`
in the baseline, honored by two graphs, and `False` in the third.

`baseline.yaml` is byte-identical to HockeyMONStream's `configs/baseline.yaml`,
and HStream honors this key natively, so a graph that drops it also puts the
two applications out of step.

`_BASELINE_BACKED` is maintained by hand, so this guards the parameters listed
there rather than the whole class. Add an entry when a plugin parameter starts
taking its shipped value from the baseline.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from hmlib.aspen.plugins.play_tracker_plugin import PlayTrackerPlugin
from hmlib.hm_opts import hm_opts

_GRAPHS = Path(__file__).resolve().parents[1] / "hmlib" / "config" / "aspen"
_BASELINE = Path(__file__).resolve().parents[1] / "hmlib" / "config" / "baseline.yaml"
# Parameters whose shipped value lives in baseline.yaml. Every graph declaring
# the plugin has to reference it rather than inherit the constructor literal.
_BASELINE_BACKED = {
    "hmlib.aspen.plugins.play_tracker_plugin.PlayTrackerPlugin": {
        "no_wide_start": "GLOBAL.play_tracker.no_wide_start",
    },
}


def _trunks(document):
    """Graph nodes, which these configs spell `aspen.plugins`."""
    aspen = document.get("aspen") if isinstance(document, dict) else None
    if not isinstance(aspen, dict):
        return {}
    nodes = aspen.get("plugins")
    if not isinstance(nodes, dict):
        nodes = aspen.get("trunks")
    return nodes if isinstance(nodes, dict) else {}


def _graph_files():
    # Recursive, so a graph under examples/ is swept rather than exempt.
    return sorted(_GRAPHS.rglob("*.yaml"))


def _rel(path):
    """Graph identity. Basenames collide once the sweep recurses."""
    return path.relative_to(_GRAPHS).as_posix()


def _declarations():
    """Every (file, trunk, spec) declaring a plugin with baseline-backed params."""
    found = []
    for path in _graph_files():
        for name, spec in _trunks(yaml.safe_load(path.read_text())).items():
            if isinstance(spec, dict) and spec.get("class") in _BASELINE_BACKED:
                found.append((path, name, spec))
    return found


def should_find_declarations_to_check():
    """Guard the guard.

    An empty sweep passes every assertion below, so the structural
    assumptions -- the glob, the `aspen.plugins` key, the class paths -- have
    to be asserted rather than relied on. This caught a first draft that read
    `aspen.trunks` and therefore checked nothing at all.
    """
    found = _declarations()
    assert found, "no graph declares a baseline-backed plugin; the sweep is broken"
    # Keyed by path, not basename: the recursive sweep can see two files with
    # one name, and an examples/tracking.yaml would otherwise stand in for a
    # real tracking.yaml that stopped declaring the plugin.
    names = {_rel(path) for path, _, _ in found}
    # A superset, so adding a graph is not a failure: the point is that these
    # three are still being reached, not that no fourth may exist.
    known = {"tracking.yaml", "tracking_pose.yaml", "tracking_pose_jersey.yaml"}
    assert names >= known, (
        f"the sweep stopped reaching {sorted(known - names)}; the glob, the "
        "`aspen.plugins` key or the plugin class path has moved"
    )


@pytest.mark.parametrize(
    "path, name, spec",
    _declarations(),
    ids=[f"{_rel(path)}:{name}" for path, name, _ in _declarations()],
)
def should_reference_the_baseline_for_backed_plugin_params(path, name, spec):
    params = spec.get("params") or {}
    for param, reference in _BASELINE_BACKED[spec["class"]].items():
        assert params.get(param) == reference, (
            f"{_rel(path)}: trunk {name!r} must pass {param}: {reference}; "
            "omitting it silently applies the constructor default instead of "
            "the baseline value"
        )


def should_route_the_no_wide_start_flag_into_the_config():
    """`--no-wide-start` has to reach the key it names.

    Without a mapping entry the flag parses and is then read by nobody, so an
    operator who sets it gets silence. Membership also nulls the argparse
    default, so the namespace carries None rather than a False that could be
    mistaken for a choice; what actually keeps an unset flag out of the config
    is `collect_explicit_arg_names` scanning argv.
    """
    assert hm_opts.ALL_YAML_ARG_TO_CONFIG_MAP["no_wide_start"] == "play_tracker.no_wide_start"
    assert hm_opts._config_value_for_arg({"play_tracker": {"no_wide_start": True}}, "no_wide_start")
    assert (
        hm_opts._config_value_for_arg({"play_tracker": {"no_wide_start": False}}, "no_wide_start")
        is False
    )
    # The nulled default, asserted rather than described. The flag is declared
    # by hmtrack's parser, and finalize_parser is what nulls it.
    from hmlib.cli.hmtrack import make_parser

    parser = make_parser()
    assert parser.parse_known_args([])[0].no_wide_start is None
    assert parser.parse_known_args(["--no-wide-start"])[0].no_wide_start is True


def should_disagree_between_the_plugin_default_and_the_baseline():
    """Record why the reference is required rather than optional.

    If these ever agree the test above still holds, but this one documents that
    the fallback is a real inversion today, not a theoretical one.
    """
    baseline = yaml.safe_load(_BASELINE.read_text())["play_tracker"]["no_wide_start"]
    literal = inspect.signature(PlayTrackerPlugin.__init__).parameters["no_wide_start"].default
    assert baseline is True
    assert literal is False
    assert baseline != literal
