"""Execution engine for Aspen plugin graphs.

Constructs a directed acyclic graph of plugins, then runs them in
topological order while sharing a mutable context dictionary.
"""

import contextlib
import importlib
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import torch

from hmlib.aspen.plugins.base import Plugin
from hmlib.log import get_logger
from hmlib.utils.containers import SidebandQueue as Queue
from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import stream_tensor_tracking

logger = get_logger(__name__)


@dataclass
class _Node:
    name: str
    cls_path: str
    depends: List[str]
    params: Dict[str, Any]
    module: torch.nn.Module
    graph_degree: Optional[int] = None
    stream: torch.cuda.Stream = None  # type: ignore[assignment]


@dataclass
class _WorkItem:
    seq: int
    context: Dict[str, Any]


@dataclass
class _GraphNodeState:
    queue: Queue
    next_seq: int
    ready: Dict[int, _WorkItem]
    lock: threading.Lock


@dataclass
class _GraphContextState:
    item: _WorkItem
    remaining_deps: List[int]
    remaining_nodes: int


class _ExceptionWrapper:
    __slots__ = ("exc", "tb")

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.tb = exc.__traceback__

    def reraise(self) -> None:
        raise self.exc.with_traceback(self.tb)


class AspenNet(torch.nn.Module):
    """Configurable directed-acyclic graph runner for Aspen plugins.

    - Loads a YAML-like dict (already parsed) with node definitions.
    - Each node has: name, class (import path), depends (list), and params.
    - Executes nodes in topological order, passing and accumulating a
      shared context dict across nodes.

    @see @ref hmlib.aspen.plugins.base.Plugin "Plugin" for the plugin interface.
    """

    def __init__(
        self,
        name: str,
        graph_cfg: Dict[str, Any],
        shared: Optional[Dict[str, Any]] = None,
        minimal_context: bool = False,
        max_concurrent: int = 2,
        verbose: bool = False,
        validate_output_keys_each_time: bool = False,
    ):
        super().__init__()
        self.name: str = self._normalize_name(name)
        self._safe_name: str = self._sanitize_name(self.name)
        self.dot_path: str = os.path.abspath(f"aspennet_{self._safe_name}.dot")
        self._last_dot_path: Optional[str] = None
        self._verbose = verbose
        self.shared: Dict[str, Any] = shared or {}
        self.nodes: List[_Node] = []
        self.node_map: Dict[str, _Node] = {}
        self.max_concurrent: int = max_concurrent
        self.num_concurrent: int = 0
        self._thread_error: Optional[BaseException] = None
        self._stop_token: Optional[object] = None
        self._timing_lock = threading.Lock()
        self._last_timing: Optional[Dict[str, Any]] = None
        self._progress_state_enabled: bool = False
        self._progress_state_lock: Optional[threading.Lock] = None
        self._progress_active_counts: Dict[str, int] = {}
        self._progress_sampler: Optional[Any] = None
        self._progress_sampler_index: Dict[str, int] = {}
        self._progress_last_sample_active: Optional[List[int]] = None
        # Track which plugins have already had their output_keys() contract validated.
        self._output_keys_validated: Set[str] = set()
        self._cuda_graph_deferred_nodes: List[str] = []
        # NetworkX DiGraph storing the plugins graph and attributes
        self.graph: nx.DiGraph = nx.DiGraph()
        self.max_graph_degree: int = 0
        self.minimal_context = bool(
            minimal_context
            or (isinstance(graph_cfg, dict) and graph_cfg.get("minimal_context", False))
        )
        pipeline_cfg: Dict[str, Any] = {}
        if isinstance(graph_cfg, dict):
            pipeline_cfg = graph_cfg.get("pipeline", {}) or {}
        if not isinstance(pipeline_cfg, dict):
            raise ValueError("AspenNet 'pipeline' configuration must be a mapping if provided.")
        threaded_flag = pipeline_cfg.get("threaded")
        if threaded_flag is None and isinstance(graph_cfg, dict):
            threaded_flag = graph_cfg.get("threaded_trunks", False)
        self.threaded_trunks: bool = bool(threaded_flag)
        graph_mode_flag = pipeline_cfg.get("graph", None)
        if graph_mode_flag is None:
            graph_mode_flag = pipeline_cfg.get("graph_mode", None)
        if isinstance(graph_mode_flag, str):
            mode = graph_mode_flag.strip().lower()
            if mode in ("true", "1", "yes", "graph", "dag", "parallel"):
                graph_mode_flag = True
            elif mode in ("false", "0", "no", "linear", "pipeline", "serial"):
                graph_mode_flag = False
        self.thread_graph_mode: bool = bool(graph_mode_flag)
        output_check_flag = pipeline_cfg.get("check_output_keys_each_time", None)
        if output_check_flag is None and isinstance(graph_cfg, dict):
            output_check_flag = graph_cfg.get("check_output_keys_each_time", None)
        self.check_output_keys_each_time: bool = bool(
            validate_output_keys_each_time or output_check_flag
        )
        queue_size_cfg = pipeline_cfg.get("queue_size", 1)
        try:
            self.thread_queue_size: int = max(1, int(queue_size_cfg))
        except Exception as exc:
            raise ValueError(
                f"AspenNet pipeline queue_size must be an integer, got {queue_size_cfg!r}"
            ) from exc
        max_concurrent_cfg = pipeline_cfg.get("max_concurrent", None)
        if max_concurrent_cfg is not None:
            try:
                self.max_concurrent = max(1, int(max_concurrent_cfg))
            except Exception as exc:
                raise ValueError(
                    "AspenNet pipeline max_concurrent must be an integer, "
                    f"got {max_concurrent_cfg!r}"
                ) from exc
        cuda_streams_flag = pipeline_cfg.get("cuda_streams", True)
        self.thread_cuda_streams: bool = bool(cuda_streams_flag) or self.thread_graph_mode
        cuda_graph_flag = pipeline_cfg.get("cuda_graph", None)
        if cuda_graph_flag is None and isinstance(graph_cfg, dict):
            cuda_graph_flag = graph_cfg.get("cuda_graph", False)
        self.cuda_graph_enabled: bool = bool(cuda_graph_flag)

        # Accept a dict with a required {plugins: {...}} mapping.
        plugins = graph_cfg.get("plugins") if isinstance(graph_cfg, dict) else None
        if plugins is None:
            raise ValueError("AspenNet expects a dict with a 'plugins' mapping.")

        # Profiler wiring (optional and zero-overhead when absent)
        self._profiler = self.shared.get("profiler", None)
        # Optional per-plugin audit hook (see hmlib.aspen.audit.AspenAuditHook).
        self._audit_hook = self.shared.get("_aspen_audit")

        self._build_nodes(plugins)
        self._build_graph()
        self.exec_order = self._toposort()
        if self.cuda_graph_enabled:
            self.set_cuda_graph_enabled(True)
        self.training: bool = False
        self._iter_num: int = 0
        self._call_seq: int = 0
        self.save_graphviz(self.dot_path)
        self.initialized = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for node in self.nodes:
            node.module.to(*args, **kwargs)
        return self

    @staticmethod
    def _normalize_name(name: str) -> str:
        if name is None:
            raise ValueError("AspenNet requires a non-empty name.")
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("AspenNet requires a non-empty name.")
        return normalized

    @staticmethod
    def _sanitize_name(name: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_")
        safe = safe.lstrip(".")
        return safe or "aspen"

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.nodes:
            module.module.train(mode)
        return super().train(mode)

    def eval(self, mode: bool = True):
        self.training = not mode
        for module in self.nodes:
            module.module.eval(mode)
        return super().eval(mode)

    # region build
    def _build_nodes(self, plugins: Dict[str, Any]):
        for name, spec in plugins.items():
            if spec is None:
                raise ValueError(f"Empty spec for plugin '{name}'")
            cls_path = spec.get("class")
            if not cls_path:
                raise ValueError(f"Plugin '{name}' missing 'class'")
            depends = list(spec.get("depends", []) or [])
            params = spec.get("params", {}) or {}
            enabled = spec.get("enabled", True)
            if not enabled:
                # Create a no-op stub to keep graph shape predictable
                module = _NoOpPlugin(name=name)
            else:
                module = self._instantiate(cls_path, params)
            if isinstance(module, Plugin):
                module.set_profiler(self._profiler)
            node = _Node(
                name=name, cls_path=cls_path, depends=depends, params=params, module=module
            )
            setattr(self, f"trunk_{name}", module)
            self.nodes.append(node)
            assert node.name not in self.node_map, f"Duplicate plugin name: {node.name}"
            self.node_map[node.name] = node

    def _build_graph(self):
        # Add all nodes with attributes
        for node in self.nodes:
            self.graph.add_node(
                node.name,
                cls_path=node.cls_path,
                params=node.params,
                module=node.module,
            )

        # Add all edges (dep -> node)
        unknown_deps: Dict[str, List[str]] = {}
        all_names = {n.name for n in self.nodes}
        for node in self.nodes:
            for dep in node.depends:
                if dep not in all_names:
                    unknown_deps.setdefault(node.name, []).append(dep)
                    continue
                self.graph.add_edge(dep, node.name)

        if unknown_deps:
            details = ", ".join(f"{k}: {v}" for k, v in unknown_deps.items())
            raise ValueError(f"Unknown dependencies referenced in plugins: {details}")

        self._assert_dag(self.graph)
        self._validate_fanin_requires_join(self.graph)
        self._validate_unique_inheritance_paths(self.graph)
        self.set_graph_degree(self.graph)

    def _validate_fanin_requires_join(self, G: nx.DiGraph) -> None:
        """Require explicit JoinPlugin nodes for fan-in.

        Rule: any node with more than one direct dependency must be a join node
        (i.e., its module has ``allow_multi_path_inputs = True``). All other
        plugins must have at most one dependency.

        This forces merges to be explicit and makes it harder to accidentally
        create multi-path ancestry or ambiguous execution intent.
        """

        join_nodes = {
            node.name
            for node in self.nodes
            if bool(getattr(node.module, "allow_multi_path_inputs", False))
        }
        offenders: List[Tuple[str, List[str]]] = []
        for node in G.nodes:
            indeg = int(G.in_degree(node))
            if indeg <= 1:
                continue
            if node in join_nodes:
                continue
            deps = sorted(str(p) for p in G.predecessors(node))
            offenders.append((str(node), deps))

        if not offenders:
            return

        lines: List[str] = []
        lines.append(
            "AspenNet dependency graph is invalid: fan-in requires an explicit JoinPlugin."
        )
        lines.append("")
        lines.append(
            "Any plugin with more than one dependency must be a join node (JoinPlugin or another "
            "plugin declaring allow_multi_path_inputs=True). All other plugins must have <= 1 dependency."
        )
        lines.append("")
        lines.append("Nodes with illegal fan-in:")
        for name, deps in sorted(offenders):
            lines.append(f"- '{name}' depends on {deps}")
        lines.append("")
        lines.append("Fix pattern:")
        lines.append("  1) Add a JoinPlugin node J that depends on the current deps")
        lines.append("  2) Make the original node depend only on J")
        lines.append("")
        lines.append("Example:")
        lines.append("  J: depends: [a, b]")
        lines.append("  X: depends: [J]  # instead of [a, b]")
        raise ValueError("\n".join(lines).rstrip())

    @staticmethod
    def _assert_dag(G: nx.DiGraph) -> None:
        if nx.is_directed_acyclic_graph(G):
            return
        try:
            cycle_nodes = nx.find_cycle(G)  # type: ignore[arg-type]
        except Exception:
            cycle_nodes = []
        raise ValueError(f"Cycle detected in plugins graph: {cycle_nodes}")

    def set_graph_degree(self, G: nx.DiGraph) -> None:
        self._assert_dag(G)

        for n in G.nodes:
            node = self.node_map[n]
            node.graph_degree = 0 if G.in_degree(n) == 0 else None

        for n in nx.topological_sort(G):
            node = self.node_map[n]
            if node.graph_degree is None:
                node.graph_degree = (
                    max(self.node_map[p].graph_degree for p in G.predecessors(n)) + 1
                )
                self.max_graph_degree = max(self.max_graph_degree, node.graph_degree)

    def set_cuda_graph_enabled(self, enabled: bool = True) -> int:
        enabled = bool(enabled)
        self.cuda_graph_enabled = enabled
        supported = 0
        deferred_plugins: List[str] = []
        supported_plugins: List[str] = []
        for node in self.nodes:
            module = node.module
            setter = getattr(node.module, "set_cuda_graph_enabled", None)
            if not callable(setter):
                continue
            try:
                if setter(enabled):
                    supported += 1
                    supported_plugins.append(node.name)
                if getattr(module, "disable_in_cuda_graph_pipeline", False):
                    if enabled:
                        deferred_plugins.append(node.name)
            except Exception:
                logger.exception(
                    "Failed to configure CUDA graph mode for Aspen plugin %s", node.name
                )
        self._cuda_graph_deferred_nodes = deferred_plugins if enabled else []
        if enabled and self._cuda_graph_deferred_nodes:
            self._validate_cuda_graph_deferred_nodes()
        self.shared["aspen_cuda_graph_enabled"] = enabled
        self.shared["aspen_cuda_graph_supported_plugins"] = supported_plugins
        self.shared["aspen_cuda_graph_disabled_plugins"] = []
        self.shared["aspen_cuda_graph_deferred_plugins"] = (
            self._cuda_graph_deferred_nodes if enabled else []
        )
        return supported

    def _validate_cuda_graph_deferred_nodes(self) -> None:
        deferred = set(self._cuda_graph_deferred_nodes)
        if not deferred:
            return
        for node_name in deferred:
            successors = list(self.graph.successors(node_name))
            non_deferred_successors = [name for name in successors if name not in deferred]
            if non_deferred_successors:
                raise ValueError(
                    "Aspen CUDA graph deferred sink plugins must be terminal. "
                    f"Plugin '{node_name}' feeds non-deferred successors {non_deferred_successors}."
                )

    @staticmethod
    def _instantiate(cls_path: str, params: Dict[str, Any]) -> torch.nn.Module:
        mod_name, _, cls_name = cls_path.rpartition(".")
        if not mod_name:
            raise ValueError(f"Invalid class path '{cls_path}'")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        if not issubclass(cls, torch.nn.Module):
            raise TypeError(f"Class '{cls_path}' must derive from torch.nn.Module")
        return cls(**params)

    def _toposort(self) -> List[_Node]:
        self._assert_dag(self.graph)

        name2node: Dict[str, _Node] = {n.name: n for n in self.nodes}
        order_names: List[str] = list(nx.topological_sort(self.graph))
        return [name2node[n] for n in order_names]

    # endregion

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all plugins in topological order.

        - 'context' is a mutable dict; plugins can read and write.
        - Returns the final context for convenience.
        """
        # Ensure plugins can access shared resources
        context.setdefault("shared", self.shared)
        context.setdefault("plugins", {})
        if "_aspen_seq" not in context:
            self._call_seq += 1
            context["_aspen_seq"] = self._call_seq
        if self.cuda_graph_enabled and self._cuda_graph_deferred_nodes:
            return self._forward_with_deferred_sinks(context)
        if self.threaded_trunks:
            self._maybe_reraise_thread_error()
            return self._forward_threaded(context)
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
        with grad_ctx:
            # do_trace: bool = True and self._iter_num == 10
            # if do_trace:
            #     pass
            for node in self.exec_order:
                self._execute_node(node, context)
            # if do_trace:
            #     pass
        self._iter_num += 1
        self._finalize_timing(context)
        return context

    def _forward_with_deferred_sinks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        deferred = set(self._cuda_graph_deferred_nodes)
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
        with grad_ctx:
            for node in self.exec_order:
                if node.name in deferred:
                    continue
                self._execute_node(node, context)
            for node in self.exec_order:
                if node.name not in deferred:
                    continue
                self._execute_node(node, context)
        self._iter_num += 1
        self._finalize_timing(context)
        return context

    # region graph export/visualization
    def to_networkx(self) -> nx.DiGraph:
        """Return a shallow copy of the internal NetworkX DiGraph."""
        return self.graph.copy()

    def _dot_lines(self) -> Iterable[str]:
        # Simple DOT writer without extra deps
        return self._dot_lines_with_styles()

    def _dot_lines_with_styles(
        self,
        *,
        edge_attrs: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        label: Optional[str] = None,
    ) -> Iterable[str]:
        def _dot_escape(value: Any) -> str:
            text = str(value)
            return text.replace("\\", "\\\\").replace('"', '\\"')

        yield "digraph AspenNet {"
        yield "  rankdir=TB;"
        yield "  node [shape=box, style=rounded];"
        graph_label = _dot_escape(label or self.name)
        yield f'  label="{graph_label}";'
        yield "  labelloc=t;"
        # Nodes with labels
        for n, data in self.graph.nodes(data=True):
            node_label = _dot_escape(f"{n}\n{data.get('cls_path', '')}")
            yield f'  "{n}" [label="{node_label}"];'
        # Edges
        attrs_map = edge_attrs or {}
        for u, v in self.graph.edges():
            attrs = attrs_map.get((u, v))
            if attrs:
                rendered = ", ".join(f'{k}="{_dot_escape(val)}"' for k, val in attrs.items())
                yield f'  "{u}" -> "{v}" [{rendered}];'
            else:
                yield f'  "{u}" -> "{v}";'
        yield "}"

    def to_dot(self) -> str:
        """Return the Graphviz DOT string for the plugins graph."""
        return "\n".join(self._dot_lines())

    def _to_dot_with_styles(
        self,
        *,
        edge_attrs: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
        label: Optional[str] = None,
    ) -> str:
        return "\n".join(self._dot_lines_with_styles(edge_attrs=edge_attrs, label=label))

    def save_graphviz(self, path: str) -> None:
        """
        Save the plugins graph as a Graphviz DOT file.

        Args:
            path: Destination file path (e.g., "graph.dot").
        """
        dot = self.to_dot()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot)
        self._last_dot_path = os.path.abspath(path)

    def _write_dot_diagnostics(self, *, suffix: str, dot: str) -> str:
        path = os.path.abspath(f"aspennet_{self._safe_name}.{suffix}.dot")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot)
        return path

    def _validate_unique_inheritance_paths(self, G: nx.DiGraph) -> None:
        """Reject graphs where any upstream node reaches a downstream node via >1 directed path.

        This prevents "diamond" shapes like:

            tracker -> camera_controller -> play_tracker
            tracker ---------------------> play_tracker
        """

        topo = list(nx.topological_sort(G))
        topo_index = {name: idx for idx, name in enumerate(topo)}
        ancestor_path_counts: Dict[str, Dict[str, int]] = {}
        violations: List[Tuple[str, str]] = []
        join_nodes = {
            node.name
            for node in self.nodes
            if bool(getattr(node.module, "allow_multi_path_inputs", False))
        }

        def _get_paths_to(node: str, ancestor: str) -> int:
            if node == ancestor:
                return 1
            return ancestor_path_counts.get(node, {}).get(ancestor, 0)

        for node in topo:
            if node in join_nodes:
                # Join nodes are an explicit semantic barrier: they may accept multiple
                # incoming paths from shared ancestors, and downstream ancestry should
                # not inherit those upstream duplicates.
                ancestor_path_counts[node] = {}
                continue
            counts: Dict[str, int] = {}
            parents = list(G.predecessors(node))
            for parent in parents:
                counts[parent] = min(2, counts.get(parent, 0) + 1)
                for ancestor, path_count in ancestor_path_counts.get(parent, {}).items():
                    if counts.get(ancestor, 0) >= 2:
                        continue
                    counts[ancestor] = min(2, counts.get(ancestor, 0) + path_count)
            ancestor_path_counts[node] = counts

            # Only report root-cause violations: the first node where an ancestor gains
            # multiple paths. Downstream nodes are suppressed because they are side effects.
            introduced: Set[str] = set()
            for ancestor, path_count in counts.items():
                if path_count <= 1:
                    continue
                if any(_get_paths_to(parent, ancestor) > 1 for parent in parents):
                    continue
                introduced.add(ancestor)
            if not introduced:
                continue

            # Reduce to "lowest" duplicated ancestors (closest to 'node') to avoid
            # reporting ancestors-of-ancestors which are also duplicated as a side effect.
            minimal_introduced = set(introduced)
            for a in introduced:
                for b in introduced:
                    if a == b:
                        continue
                    # If a can reach b, then a is more upstream than b; keep b only.
                    if nx.has_path(G, a, b):
                        minimal_introduced.discard(a)
                        break
            for ancestor in sorted(minimal_introduced, key=lambda n: topo_index.get(n, 0)):
                violations.append((ancestor, node))

        if not violations:
            return

        # Gather two concrete paths for each violation and highlight their edges.
        paths_by_pair: Dict[Tuple[str, str], List[List[str]]] = {}
        highlight_edges: Set[Tuple[str, str]] = set()
        for ancestor, node in violations:
            key = (ancestor, node)
            if key in paths_by_pair:
                continue
            paths = self._find_two_paths(G, ancestor, node)
            paths_by_pair[key] = paths
            for path in paths:
                highlight_edges.update(zip(path, path[1:]))

        dot = self._to_dot_with_styles(
            edge_attrs={edge: {"color": "red", "penwidth": "3"} for edge in highlight_edges},
            label=f"{self.name} (INVALID: multiple inheritance paths)",
        )
        diag_path = self._write_dot_diagnostics(suffix="INVALID_MULTI_PATHS", dot=dot)

        ordered_pairs = sorted(
            paths_by_pair.keys(),
            key=lambda pair: (topo_index.get(pair[1], 0), topo_index.get(pair[0], 0), pair[0]),
        )
        message = self._format_multipath_error_message(
            paths_by_pair, ordered_pairs=ordered_pairs, diag_path=diag_path
        )
        self._maybe_print_rich_multipath_tree(
            paths_by_pair, ordered_pairs=ordered_pairs, diag_path=diag_path
        )
        raise ValueError(message)

    @staticmethod
    def _find_two_paths(G: nx.DiGraph, source: str, target: str) -> List[List[str]]:
        if source == target:
            return [[source]]
        # Only consider nodes that can reach target (reverse-DFS from target).
        candidates = nx.ancestors(G, target) | {target}
        if source not in candidates:
            return []
        paths: List[List[str]] = []
        stack: List[Tuple[str, List[str]]] = [(source, [source])]
        while stack and len(paths) < 2:
            node, path = stack.pop()
            if node == target:
                paths.append(path)
                continue
            successors = [s for s in G.successors(node) if s in candidates]
            successors.sort(reverse=True)
            for succ in successors:
                if succ in path:
                    continue
                stack.append((succ, [*path, succ]))
        return paths

    def _format_multipath_error_message(
        self,
        paths_by_pair: Dict[Tuple[str, str], List[List[str]]],
        *,
        ordered_pairs: List[Tuple[str, str]],
        diag_path: str,
    ) -> str:
        lines: List[str] = []
        lines.append(
            "AspenNet dependency graph is invalid: an upstream plugin is reachable by more than one "
            "directed path from a downstream plugin."
        )
        lines.append("")
        lines.append(
            "This is forbidden because it makes dependency structure ambiguous and hard to reason about."
        )
        lines.append(
            "Fix by removing redundant edges so that for any pair (A -> ... -> B) there is at most one path."
        )
        lines.append("")
        lines.append(f"Graphviz diagnostics written to: {diag_path}")
        lines.append("Red edges are part of at least one multi-path violation.")
        lines.append("")
        lines.append(
            "Only root-cause violations are shown (the first nodes where multiple paths are introduced); "
            "downstream duplicates are suppressed as side effects."
        )
        lines.append("")
        tree_text = self._render_multipath_tree_text(
            paths_by_pair, ordered_pairs=ordered_pairs, diag_path=diag_path
        )
        if tree_text:
            lines.append("Graph:")
            lines.append(tree_text.rstrip())
            lines.append("")
        lines.append("Violations:")
        for ancestor, node in ordered_pairs:
            paths = paths_by_pair[(ancestor, node)]
            if not paths:
                lines.append(
                    f"- '{ancestor}' reaches '{node}' via multiple paths (paths unavailable)."
                )
                continue
            rendered = [" -> ".join(p) for p in paths[:2]]
            lines.append(f"- '{ancestor}' reaches '{node}' via multiple paths (showing up to 2):")
            for idx, p in enumerate(rendered, start=1):
                lines.append(f"    {idx}) {p}")
        return "\n".join(lines).rstrip()

    @staticmethod
    def _render_multipath_tree_text(
        paths_by_pair: Dict[Tuple[str, str], List[List[str]]],
        *,
        ordered_pairs: List[Tuple[str, str]],
        diag_path: str,
    ) -> str:
        try:
            from rich.console import Console  # type: ignore
            from rich.text import Text  # type: ignore
            from rich.tree import Tree  # type: ignore

            console = Console(
                width=120,
                color_system=None,
                force_terminal=False,
                record=True,
                file=io.StringIO(),
            )
            root = Tree(Text("AspenNet INVALID graph: multiple inheritance paths"))
            root.add(Text(f"DOT with red edges: {diag_path}"))
            for ancestor, node in ordered_pairs:
                paths = paths_by_pair[(ancestor, node)]
                branch = root.add(Text(f"{ancestor} → {node}"))
                for path in paths[:2]:
                    branch.add(Text(" -> ".join(path)))
            console.print(root)
            return console.export_text(styles=False)
        except Exception:
            lines: List[str] = []
            lines.append("AspenNet INVALID graph: multiple inheritance paths")
            lines.append(f"DOT with red edges: {diag_path}")
            for ancestor, node in ordered_pairs:
                paths = paths_by_pair[(ancestor, node)]
                lines.append(f"- {ancestor} → {node}")
                for path in paths[:2]:
                    lines.append(f"  - {' -> '.join(path)}")
            return "\n".join(lines)

    def _maybe_print_rich_multipath_tree(
        self,
        paths_by_pair: Dict[Tuple[str, str], List[List[str]]],
        *,
        ordered_pairs: List[Tuple[str, str]],
        diag_path: str,
    ) -> None:
        try:
            if not sys.stderr.isatty():
                return
            from rich.console import Console  # type: ignore
            from rich.text import Text  # type: ignore
            from rich.tree import Tree  # type: ignore

            console = Console(stderr=True)
            root = Tree(
                Text("AspenNet INVALID graph: multiple inheritance paths", style="bold red")
            )
            root.add(Text(f"DOT with red edges: {diag_path}", style="dim"))
            for ancestor, node in ordered_pairs:
                paths = paths_by_pair[(ancestor, node)]
                branch = root.add(Text(f"{ancestor} → {node}", style="red"))
                for path in paths[:2]:
                    branch.add(Text(" -> ".join(path), style="red"))
            console.print(root)
        except Exception:
            # Diagnostics should never prevent raising the real configuration error.
            return

    def display_graphviz(self) -> None:
        """
        Display the plugins graph.

        Tries, in order:
        - xdot executable (if available) for an interactive popup
        - graphviz.Source (if `graphviz` python package is installed)
        - matplotlib via networkx (if matplotlib is available)
        - Prints DOT to stdout as a fallback
        """
        dot = self.to_dot()
        dot_path = self._last_dot_path or self.dot_path

        # Try xdot binary
        try:
            xdot_bin = shutil.which("xdot")
            if xdot_bin:
                path = dot_path or os.path.abspath("aspennet.dot")
                self.save_graphviz(path)
                subprocess.Popen([xdot_bin, path])
                return
        except Exception as ex:
            print(f"AspenNet: xdot display failed: {ex}")

        # Try graphviz Python package
        try:
            from graphviz import Source  # type: ignore

            src = Source(dot)
            src.view(cleanup=True)
            return
        except Exception as ex:
            print(f"AspenNet: graphviz display failed: {ex}")

        # Try matplotlib networkx draw
        try:
            import matplotlib.pyplot as plt  # type: ignore

            pos = (
                nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
                if self._has_pygraphviz()
                else nx.spring_layout(self.graph)
            )
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_size=1500,
                node_color="#DDEEFF",
                font_size=8,
                arrows=True,
            )
            plt.title("AspenNet Plugins Graph")
            plt.show()
            return
        except Exception as e:
            print(f"AspenNet: matplotlib display failed: {e}")

        # Fallback: print DOT to stdout
        print(dot)

    @staticmethod
    def _has_pygraphviz() -> bool:
        try:
            import pygraphviz  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False

    def stop_progress_graph(self) -> None:
        """Stop the progress graph sampler thread if active."""
        sampler = self._progress_sampler
        if sampler is None:
            return
        try:
            sampler.stop()
        except Exception:
            logger.exception("AspenNet progress sampler stop failed")
        self._progress_sampler = None
        self._progress_last_sample_active = None

    def finalize(self) -> None:
        """Stop any threaded workers and invoke ``finalize`` on plugins.

        In threaded mode, AspenNet executes plugins in background threads. Callers
        typically use ``finalize()`` as an end-of-run cleanup hook, so we also
        stop/join worker threads here to avoid leaving CUDA work running during
        interpreter shutdown.
        """
        if self.threaded_trunks and getattr(self, "threads", None):
            try:
                self.stop(wait=True)
            except Exception:
                logger.exception("AspenNet stop failed during finalize")
        for node in self.nodes:
            finalize_fn = getattr(node.module, "finalize", None)
            if callable(finalize_fn):
                try:
                    finalize_fn()
                except Exception:
                    logger.exception("Aspen plugin %s finalize failed", node.name)
        self.stop_progress_graph()

    # endregion

    def _maybe_reraise_thread_error(self) -> None:
        """Raise any exception captured from threaded plugin workers."""
        err = self._thread_error
        if err is None:
            return
        # Prefer wrapped exceptions that preserve original traceback.
        reraise = getattr(err, "reraise", None)
        if callable(reraise):
            reraise()
        raise err

    def _execute_node(self, node: _Node, context: Dict[str, Any]) -> None:
        plugin = node.module
        subctx = self._make_subcontext(plugin, context) if self.minimal_context else context
        audit_hook = self._audit_hook
        if audit_hook is not None:
            try:
                audit_hook.before_plugin(node.name, subctx, context)
            except Exception:
                logger.exception("Aspen audit hook before_plugin failed for %s", node.name)
        name = f"aspen.plugin.{node.name}"
        start_time = None
        if context.get("_aspen_timing_enabled"):
            start_time = time.perf_counter()
        if self._verbose:
            print(f"AspenNet: Executing plugin '{node.name}' with class '{node.cls_path}'")
        if isinstance(plugin, Plugin):
            prof_ctx = plugin.profile_scope(name)
        elif getattr(self._profiler, "enabled", False):
            prof_ctx = self._profiler.rf(name)
        else:
            prof_ctx = contextlib.nullcontext()
        if self._progress_state_enabled:
            sampler = self._progress_sampler
            if sampler is not None:
                idx = self._progress_sampler_index.get(node.name)
                try:
                    if idx is not None:
                        sampler.enter_index(idx)
                    with prof_ctx:
                        out = plugin(subctx) or {}
                finally:
                    if idx is not None:
                        sampler.exit_index(idx)
            else:
                entered = self._progress_enter(node.name)
                try:
                    with prof_ctx:
                        out = plugin(subctx) or {}
                finally:
                    if entered:
                        self._progress_exit(node.name)
        else:
            with prof_ctx:
                out = plugin(subctx) or {}
        if audit_hook is not None:
            try:
                audit_hook.after_plugin(node.name, subctx, out, context)
            except Exception:
                logger.exception("Aspen audit hook after_plugin failed for %s", node.name)
        if start_time is not None:
            end_time = time.perf_counter()
            self._record_timing(context, node.name, start_time, end_time)

        declared = set(getattr(plugin, "output_keys", lambda: set())())
        returned_keys = set(out.keys())

        if declared:
            should_check = self.check_output_keys_each_time or (
                node.name not in self._output_keys_validated
            )
            if should_check:
                extra_keys = returned_keys - declared
                if extra_keys:
                    raise ValueError(
                        f"AspenNet plugin '{node.name}' ({node.cls_path}) returned keys "
                        f"{sorted(extra_keys)} not declared in output_keys(). "
                        f"Declared keys: {sorted(declared)}"
                    )
                if not self.check_output_keys_each_time:
                    self._output_keys_validated.add(node.name)

        update_keys = declared if declared else returned_keys

        from .plugins.base import DeleteKey  # local import avoids cycle

        for key in update_keys:
            if key in out:
                value = out[key]
                if isinstance(value, DeleteKey):
                    if key in context:
                        del context[key]
                else:
                    context[key] = value

        context["plugins"][node.name] = {k: out[k] for k in out.keys()}

    def _record_timing(self, context: Dict[str, Any], name: str, start: float, end: float) -> None:
        if not context.get("_aspen_timing_enabled"):
            return
        with self._timing_lock:
            timing = context.setdefault(
                "_aspen_timing", {"plugins": {}, "start": None, "end": None}
            )
            plugin_entry = timing["plugins"].setdefault(name, {})
            plugin_entry["start"] = start
            plugin_entry["end"] = end
            plugin_entry["duration"] = float(end - start)
            timing_start = timing.get("start")
            timing_end = timing.get("end")
            timing["start"] = start if timing_start is None else min(timing_start, start)
            timing["end"] = end if timing_end is None else max(timing_end, end)

    def _finalize_timing(self, context: Dict[str, Any]) -> None:
        if not context.get("_aspen_timing_enabled"):
            return
        timing = context.get("_aspen_timing")
        if not isinstance(timing, dict):
            return
        start = timing.get("start")
        end = timing.get("end")
        if start is None or end is None:
            return
        total = max(1e-9, float(end - start))
        plugins = timing.get("plugins", {})
        summary = {
            "total": total,
            "plugins": {name: float(info.get("duration", 0.0)) for name, info in plugins.items()},
        }
        with self._timing_lock:
            self._last_timing = summary

    def get_last_timing(self) -> Optional[Dict[str, Any]]:
        with self._timing_lock:
            if self._last_timing is None:
                return None
            return dict(self._last_timing)

    def enable_progress_graph(self) -> None:
        """Enable lightweight graph state tracking for progress UI rendering."""
        if self._progress_state_enabled:
            return
        self._progress_state_enabled = True
        self._progress_last_sample_active = None
        try:
            from hockeymon._hockeymon import AspenGraphSampler  # type: ignore

            sampler = AspenGraphSampler(max_samples=24, min_interval_ms=12, max_interval_ms=40)
            names = [node.name for node in self.exec_order]
            degrees = [
                node.graph_degree if node.graph_degree is not None else 0
                for node in self.exec_order
            ]
            self._progress_sampler_index = {name: idx for idx, name in enumerate(names)}
            edges = [
                (self._progress_sampler_index[u], self._progress_sampler_index[v])
                for u, v in self.graph.edges()
                if u in self._progress_sampler_index and v in self._progress_sampler_index
            ]
            sampler.configure_graph(names, degrees, edges)
            sampler.start()
            self._progress_sampler = sampler
            return
        except Exception:
            logger.exception("AspenNet progress sampler init failed; falling back to Python state")
            self._progress_sampler = None
            self._progress_sampler_index = {}
        self._progress_state_lock = threading.Lock()
        self._progress_active_counts = {node.name: 0 for node in self.exec_order}

    def _progress_enter(self, name: str) -> bool:
        if not self._progress_state_enabled:
            return False
        lock = self._progress_state_lock
        if lock is None:
            return False
        with lock:
            self._progress_active_counts[name] = self._progress_active_counts.get(name, 0) + 1
        return True

    def _progress_exit(self, name: str) -> None:
        if not self._progress_state_enabled:
            return
        lock = self._progress_state_lock
        if lock is None:
            return
        with lock:
            current = self._progress_active_counts.get(name, 0)
            if current > 0:
                self._progress_active_counts[name] = current - 1

    def get_progress_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return a snapshot of graph activity and queue state for UI rendering."""
        if not self._progress_state_enabled:
            return None
        lock = self._progress_state_lock
        active_flags: Optional[List[int]] = None
        if self._progress_sampler is not None:
            try:
                samples = self._progress_sampler.pop_samples(1)
            except Exception:
                samples = []
            if samples:
                sample = samples[-1]
                active_flags = list(sample.get("active", []))
                self._progress_last_sample_active = active_flags
            else:
                active_flags = self._progress_last_sample_active
        active_counts: Dict[str, int] = {}
        if lock is not None:
            with lock:
                active_counts = dict(self._progress_active_counts)
        nodes = []
        for idx, node in enumerate(self.exec_order):
            degree = node.graph_degree if node.graph_degree is not None else 0
            if active_flags is not None and idx < len(active_flags):
                active = bool(active_flags[idx])
            else:
                active = bool(active_counts.get(node.name, 0) > 0)
            nodes.append(
                {
                    "name": node.name,
                    "degree": int(degree),
                    "active": active,
                }
            )
        edges = list(self.graph.edges())
        node_queues: Dict[str, Dict[str, Any]] = {}
        queue_info = None
        queues = None
        labels: List[str] = []
        if self.threaded_trunks:
            if self.thread_graph_mode and hasattr(self, "graph_queues"):
                queues = list(self.graph_queues)
                labels = [node.name for node in self.exec_order]
                for node, q in zip(self.exec_order, queues):
                    try:
                        current = q.qsize()
                    except Exception:
                        current = 0
                    max_size = getattr(q, "_max_size", None)
                    node_queues[node.name] = {"current": int(current), "max": max_size}
            elif hasattr(self, "queues"):
                queues = list(self.queues)
                labels = [node.name for node in self.exec_order]
                if len(queues) > len(labels):
                    labels.append("out")
                for idx, node in enumerate(self.exec_order):
                    if idx >= len(queues):
                        break
                    q = queues[idx]
                    try:
                        current = q.qsize()
                    except Exception:
                        current = 0
                    max_size = getattr(q, "_max_size", None)
                    node_queues[node.name] = {"current": int(current), "max": max_size}
        if queues:
            items = []
            total_current = 0
            total_capacity = 0
            capacity_known = True
            for label, q in zip(labels, queues):
                try:
                    current = q.qsize()
                except Exception:
                    current = 0
                max_size = getattr(q, "_max_size", None)
                items.append({"label": label, "current": int(current), "max": max_size})
                total_current += int(current)
                if isinstance(max_size, int) and max_size > 0:
                    total_capacity += max_size
                else:
                    capacity_known = False
            queue_info = {
                "items": items,
                "total_current": total_current,
                "total_capacity": total_capacity if capacity_known else None,
                "count": len(items),
            }
        concurrency = {
            "current": int(self.num_concurrent) if self.threaded_trunks else 1,
            "max": int(self.max_concurrent) if self.threaded_trunks else 1,
            "threaded": bool(self.threaded_trunks),
        }
        return {
            "nodes": nodes,
            "max_degree": int(self.max_graph_degree),
            "queues": queue_info,
            "concurrency": concurrency,
            "edges": edges,
            "node_queues": node_queues,
            "order": [node.name for node in self.exec_order],
        }

    def _make_subcontext(self, plugin: torch.nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        req_keys = set(getattr(plugin, "input_keys", lambda: set())())
        if not req_keys:
            subctx: Dict[str, Any] = {}
        else:
            subctx = {k: context[k] for k in req_keys if k in context}
            for key in req_keys:
                if key not in subctx and key in self.shared:
                    subctx[key] = self.shared[key]
        subctx.setdefault("shared", self.shared)
        return subctx

    def _forward_threaded(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.thread_graph_mode:
            return self._forward_threaded_graph(context)
        return self._forward_threaded_linear(context)

    def _graph_enqueue_ready(self, node_index: int, item: _WorkItem) -> None:
        if getattr(self, "_graph_stop_event", None) is not None and self._graph_stop_event.is_set():
            return
        state = self._graph_node_state[node_index]
        with state.lock:
            if (
                getattr(self, "_graph_stop_event", None) is not None
                and self._graph_stop_event.is_set()
            ):
                return
            state.ready[item.seq] = item
            self._graph_drain_ready_locked(state)

    def _graph_drain_ready_locked(self, state: _GraphNodeState) -> None:
        # Enqueue only contiguous sequences to preserve per-node ordering.
        while True:
            next_item = state.ready.get(state.next_seq)
            if next_item is None:
                break
            try:
                state.queue.put(next_item, block=False)
            except (queue.Full, ValueError):
                break
            del state.ready[state.next_seq]
            state.next_seq += 1

    def _graph_mark_complete(self, node_index: int, item: _WorkItem) -> None:
        if getattr(self, "_graph_stop_event", None) is not None and self._graph_stop_event.is_set():
            return
        ready_children: List[int] = []
        finalize_ctx = None
        with self._graph_lock:
            if self._graph_stop_event.is_set():
                return
            state = self._graph_contexts.get(item.seq)
            if state is None:
                return
            for child_index in self._graph_children[node_index]:
                state.remaining_deps[child_index] -= 1
                if state.remaining_deps[child_index] == 0:
                    ready_children.append(child_index)
            state.remaining_nodes -= 1
            if state.remaining_nodes == 0:
                del self._graph_contexts[item.seq]
                if self.num_concurrent > 0:
                    self.num_concurrent -= 1
                finalize_ctx = state.item.context
        for child_index in ready_children:
            self._graph_enqueue_ready(child_index, item)
        if finalize_ctx is not None:
            self._finalize_timing(finalize_ctx)

    def _graph_request_stop(self) -> None:
        stop_event = getattr(self, "_graph_stop_event", None)
        if stop_event is None:
            return
        stop_event.set()
        stop_token = self._stop_token
        if stop_token is None:
            stop_token = object()
            self._stop_token = stop_token
        queues = getattr(self, "graph_queues", None)
        if queues is None:
            return
        for q in queues:
            try:
                q.put(stop_token, block=False)
            except (queue.Full, ValueError):
                continue

    def _wait_threaded_graph_idle(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + float(timeout)
        while True:
            self._maybe_reraise_thread_error()
            with self._graph_lock:
                pending_contexts = len(getattr(self, "_graph_contexts", {}))
                pending_concurrent = int(self.num_concurrent)
            if pending_contexts == 0 and pending_concurrent == 0:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for Aspen threaded graph to drain before stop: "
                    f"contexts={pending_contexts}, concurrent={pending_concurrent}"
                )
            time.sleep(0.01)

    def _join_threads(self, timeout: Optional[float] = None) -> List[str]:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        alive: List[str] = []
        for thread in self.threads:
            if not thread.is_alive():
                continue
            if deadline is None:
                thread.join()
            else:
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)
            if thread.is_alive():
                alive.append(str(getattr(thread, "name", thread)))
        return alive

    def _graph_worker(self, index: int, node: _Node) -> None:
        in_queue = self.graph_queues[index]
        stop_token = self._stop_token
        state = self._graph_node_state[index]
        try:
            while True:
                try:
                    item = in_queue.get(timeout=0.1)
                except queue.Empty:
                    if (
                        getattr(self, "_graph_stop_event", None) is not None
                        and self._graph_stop_event.is_set()
                    ):
                        break
                    continue
                if item is stop_token:
                    break
                if isinstance(item, _ExceptionWrapper):
                    self._thread_error = item
                    self._graph_request_stop()
                    break
                if (
                    getattr(self, "_graph_stop_event", None) is not None
                    and self._graph_stop_event.is_set()
                ):
                    break
                with state.lock:
                    self._graph_drain_ready_locked(state)
                grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
                try:
                    with grad_ctx:
                        self._execute_with_stream(node, item.context)
                    self._graph_mark_complete(index, item)
                except BaseException as exc:
                    wrapper = _ExceptionWrapper(exc)
                    self._thread_error = wrapper
                    self._graph_request_stop()
                    break
        finally:
            print(f"AspenNet: Thread for plugin '{node.name}' exiting.")

    def _init_threaded_graph(self) -> None:
        self.initialized = True
        stop_token = self._stop_token or object()
        self._stop_token = stop_token
        self._graph_stop_event = threading.Event()
        self._graph_lock = threading.Lock()
        self._graph_seq = 0
        self._graph_contexts = {}
        self._graph_node_index = {node.name: idx for idx, node in enumerate(self.exec_order)}
        self._graph_children = [[] for _ in self.exec_order]
        self._graph_indegree = [0 for _ in self.exec_order]
        for parent, child in self.graph.edges():
            parent_idx = self._graph_node_index[parent]
            child_idx = self._graph_node_index[child]
            self._graph_children[parent_idx].append(child_idx)
            self._graph_indegree[child_idx] += 1
        self._graph_roots = [idx for idx, count in enumerate(self._graph_indegree) if count == 0]
        self.graph_queues = [
            create_queue(
                mp=False,
                name=f"Aspen-{node.name}",
                max_size=self.thread_queue_size,
                warn_after=5.0,
            )
            for node in self.exec_order
        ]
        self._graph_node_state = [
            _GraphNodeState(
                queue=self.graph_queues[idx],
                next_seq=0,
                ready={},
                lock=threading.Lock(),
            )
            for idx in range(len(self.exec_order))
        ]
        self.threads = []
        for idx, node in enumerate(self.exec_order):
            thread = threading.Thread(
                target=self._graph_worker, args=(idx, node), daemon=True, name=node.name
            )
            thread.start()
            self.threads.append(thread)

    def _forward_threaded_graph(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            self._init_threaded_graph()
        while self.num_concurrent >= self.max_concurrent:
            self._maybe_reraise_thread_error()
            time.sleep(0.01)
        self._maybe_reraise_thread_error()
        roots = []
        with self._graph_lock:
            seq = self._graph_seq
            self._graph_seq += 1
            context["_aspen_seq"] = seq
            if self.thread_cuda_streams:
                context["_aspen_cuda_events"] = {}
            item = _WorkItem(seq=seq, context=context)
            state = _GraphContextState(
                item=item,
                remaining_deps=list(self._graph_indegree),
                remaining_nodes=len(self.exec_order),
            )
            self._graph_contexts[seq] = state
            self.num_concurrent += 1
            roots = list(self._graph_roots)
        for idx in roots:
            self._graph_enqueue_ready(idx, item)
        return None

    def _forward_threaded_linear(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.initialized:
            self.initialized = True
            stop_token = self._stop_token or object()
            self._stop_token = stop_token

            def make_grad_ctx():
                return torch.enable_grad() if self.training else torch.no_grad()

            def worker(index: int, node: _Node) -> None:
                in_queue = self.queues[index]
                is_last = index == len(self.exec_order) - 1
                out_queue = self.queues[index + 1]
                try:
                    while True:
                        item = in_queue.get()
                        if item is stop_token:
                            out_queue.put(stop_token)
                            break
                        if isinstance(item, _ExceptionWrapper):
                            out_queue.put(item)
                            self.stop(wait=False)
                            break
                        grad_ctx = make_grad_ctx()
                        try:
                            with grad_ctx:
                                self._execute_with_stream(node, item)
                            if not is_last:
                                out_queue.put(item)
                            else:
                                self._finalize_timing(item)
                                assert self.num_concurrent > 0
                                self.num_concurrent -= 1
                        except BaseException as exc:
                            wrapper = _ExceptionWrapper(exc)
                            self._thread_error = wrapper
                            print(exc)
                            # Propagate the wrapped exception downstream when possible
                            if not is_last:
                                out_queue.put(wrapper)
                            else:
                                if self.num_concurrent > 0:
                                    self.num_concurrent -= 1
                            break
                finally:
                    print(f"AspenNet: Thread for plugin '{node.name}' exiting.")

            self.queues: List[Queue] = [
                create_queue(
                    mp=False,
                    name=f"Aspen-{self.exec_order[i-1].name}",
                    max_size=self.thread_queue_size,
                )
                for i in range(len(self.exec_order) + 1)
            ]
            self.threads = []
            for idx, node in enumerate(self.exec_order):
                thread = threading.Thread(
                    target=worker, args=(idx, node), daemon=True, name=node.name
                )
                thread.start()
                self.threads.append(thread)
        while self.num_concurrent >= self.max_concurrent:
            self._maybe_reraise_thread_error()
            time.sleep(0.01)
        while True:
            self._maybe_reraise_thread_error()
            try:
                self.queues[0].put(context, block=False)
                break
            except queue.Full:
                time.sleep(0.01)
        self.num_concurrent += 1
        return None

    def stop(self, wait: bool = True) -> None:
        """Stop all threaded plugins and join their threads."""
        if not self.threaded_trunks:
            self.stop_progress_graph()
            return
        stop_token = self._stop_token or object()
        self._stop_token = stop_token
        if self.thread_graph_mode:
            if not hasattr(self, "graph_queues"):
                self.stop_progress_graph()
                return
            wait_error = None
            if wait:
                try:
                    self._wait_threaded_graph_idle()
                except BaseException as exc:
                    wait_error = exc
            self._graph_request_stop()
            alive_threads: List[str] = []
            if wait:
                join_timeout = 5.0 if wait_error is not None else 60.0
                alive_threads = self._join_threads(timeout=join_timeout)
            for q in self.graph_queues:
                q.close()
            del self.graph_queues
            self.stop_progress_graph()
            if alive_threads:
                join_error = RuntimeError(
                    "Timed out joining Aspen threaded graph workers after stop: "
                    + ", ".join(alive_threads)
                )
                if wait_error is not None:
                    raise join_error from wait_error
                raise join_error
            if wait_error is not None:
                raise wait_error
            return
        if not hasattr(self, "queues"):
            self.stop_progress_graph()
            return
        if wait:
            for _ in self.exec_order:
                self.queues[0].put(stop_token)
        else:
            for _ in self.exec_order:
                try:
                    self.queues[0].put(stop_token, block=False)
                except queue.Full:
                    break
        for thread in self.threads:
            if wait and thread.is_alive():
                thread.join()
        for q in self.queues:
            q.close()
        del self.queues
        self.stop_progress_graph()

    def _execute_with_stream(self, node: _Node, context: Dict[str, Any]) -> None:
        use_cuda_stream = (
            self.thread_cuda_streams  # and torch.cuda.is_available() and device is not None
        )
        if use_cuda_stream:
            profiler = self._profiler
            init_ctx = (
                profiler.rf(f"aspen.stream_init.{node.name}")
                if getattr(profiler, "enabled", False)
                else contextlib.nullcontext()
            )
            if node.stream is None:
                with init_ctx:
                    device = self._infer_device(context)
                    # Arbitrarily make normal priority 10
                    # Smaller/neg numbers are higher priority
                    priority: int = self.max_graph_degree + 1
                    if node.graph_degree is not None:
                        priority -= node.graph_degree
                    node.stream = torch.cuda.Stream(
                        device=device,
                        priority=priority,
                    )
                    print(f"Created stream for plugin {node.name} with priority {priority}")
            # Ensure plugins that fetch context["cuda_stream"] see the stream actually running them.
            prev_stream = context.get("cuda_stream")
            if prev_stream is None:
                prev_stream = torch.cuda.current_stream(device=node.stream.device)
            if prev_stream is not None:
                wait_ctx = (
                    profiler.rf(f"aspen.stream_wait.{node.name}")
                    if getattr(profiler, "enabled", False)
                    else contextlib.nullcontext()
                )
                with wait_ctx:
                    node.stream.wait_stream(prev_stream)
            if self.thread_graph_mode:
                parent_events = context.get("_aspen_cuda_events")
                if isinstance(parent_events, dict):
                    for parent_name in node.depends:
                        event = parent_events.get(parent_name)
                        if event is not None:
                            node.stream.wait_event(event)
            context["cuda_stream"] = node.stream
            try:
                with torch.cuda.stream(node.stream), stream_tensor_tracking("stream"):
                    self._execute_node(node, context)
                if self.thread_graph_mode:
                    event = torch.cuda.Event(blocking=False)
                    node.stream.record_event(event)
                    context.setdefault("_aspen_cuda_events", {})[node.name] = event
            finally:
                if prev_stream is not None:
                    context["cuda_stream"] = prev_stream
                else:
                    context.pop("cuda_stream", None)
        else:
            self._execute_node(node, context)

    def _infer_device(self, context: Dict[str, Any]) -> Optional[torch.device]:
        device = context.get("device")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception:
                pass
        cuda_stream = context.get("cuda_stream")
        if hasattr(cuda_stream, "device"):
            try:
                return torch.device(cuda_stream.device)  # type: ignore[arg-type]
            except Exception:
                pass
        # Prefer flattened Aspen context keys.
        for key in ("inputs", "img", "original_images"):
            t = context.get(key)
            if isinstance(t, torch.Tensor):
                return t.device
            dev = getattr(t, "device", None)
            if isinstance(dev, torch.device):
                return dev

        # Backwards compatibility: older pipelines may still use a nested `data` dict.
        data = context.get("data")
        if isinstance(data, dict):
            img = data.get("img")
            if isinstance(img, torch.Tensor):
                return img.device
            dev = getattr(img, "device", None)
            if isinstance(dev, torch.device):
                return dev
        shared_device = self.shared.get("device")
        if isinstance(shared_device, torch.device):
            return shared_device
        if isinstance(shared_device, str):
            try:
                return torch.device(shared_device)
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                pass
        return None


class _NoOpPlugin(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Intentionally does nothing
        return {}
