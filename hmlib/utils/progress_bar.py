"""Rich terminal progress bar and table utilities for HockeyMON CLIs.

Provides a rich-based, stationary progress UI used by many command-line
tools in :mod:`hmlib`.

@see @ref ProgressBar "ProgressBar" for the main iteration helper.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import shutil
import sys
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress as RichProgress
from rich.progress import ProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

progress_out = sys.stderr
logging_out = sys.stdout

# Shared rich console used for both progress bars and log output.
RICH_CONSOLE = Console(file=progress_out, stderr=True)
_CURSOR_SHOW = "\033[?25h"


def _restore_cursor():
    try:
        progress_out.write(_CURSOR_SHOW)
        progress_out.flush()
    except Exception:
        pass


atexit.register(_restore_cursor)


class FramesColumn(ProgressColumn):
    """Right-aligned ``completed/total`` frames column with fixed width.

    @brief Shows iteration progress in terms of frames, reserving space for
    up to eight digits for both completed and total counts.
    """

    def render(self, task) -> Text:  # type: ignore[override]
        completed = int(task.completed)
        total = task.total
        if total is None or (isinstance(total, (int, float)) and total <= 0):
            total_str = " " * 8
        else:
            try:
                total_str = f"{int(total):>8}"
            except Exception:
                total_str = " " * 8
        text = f"{completed:>8}/{total_str}"
        return Text(text, style="white")


def _get_terminal_width():
    """Return the current terminal width in columns."""
    return shutil.get_terminal_size().columns


class CallbackStreamHandler(logging.StreamHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.callback(message)


class RichProgressFormatter(logging.Formatter):
    """Format log records with rich markup for the progress log area."""

    def __init__(self, datefmt: str | None = "%H:%M:%S"):
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        time_str = self.formatTime(record, self.datefmt)
        level = record.levelname
        if record.levelno >= logging.ERROR:
            style = "bold red"
        elif record.levelno >= logging.WARNING:
            style = "bold yellow"
        elif record.levelno >= logging.INFO:
            style = "bold cyan"
        else:
            style = "dim"
        return f"[dim]{time_str}[/dim] [{style}]{level:>8}[/] {message}"


class ScrollOutput:
    """Maintain a simple scrolling text buffer for log output.

    @brief Collects recent log lines and forwards them into a sink callback.
    This is used together with :class:`ProgressBar` to feed a scrolling log
    area inside the rich-based progress UI.

    @param lines Maximum number of lines to retain in the buffer.
    """

    def __init__(self, lines=10):
        self.capture = []
        self.lines = lines
        self._sink: Optional[Callable[[str], None]] = None

    def write(self, msg):
        if msg == "\n":
            return
        text = msg.rstrip("\n")
        if not text:
            return
        self.capture.append(text)
        if len(self.capture) > self.lines:
            self.capture.pop(0)
        if self._sink is not None:
            try:
                self._sink(text)
            except Exception:
                # Best-effort; avoid breaking logging on sink errors.
                RICH_CONSOLE.print(text)

    def flush(self):
        pass

    def reset(self):
        """Legacy no-op kept for API compatibility."""
        return

    def refresh(self):
        """Legacy no-op kept for API compatibility."""
        return

    def register_logger(self, logger) -> ScrollOutput:
        """
        Register a callback stream handler which
        redirects logging to this class' "write" function.
        """
        # Remove all existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create and add our custom handler with rich-style formatting
        callback_handler = CallbackStreamHandler(self.write)
        callback_handler.setFormatter(RichProgressFormatter())
        logger.addHandler(callback_handler)
        return self

    # Internal: set by ProgressBar when using curses
    def _attach_curses_ui(self, ui: object | None):
        """Legacy no-op kept for API compatibility."""
        return

    # Optional: used by fallback ANSI renderer to align right border
    def _set_column_width(self, column_width: int | None):
        """Legacy no-op kept for API compatibility."""
        return

    def set_sink(self, sink: Callable[[str], None]) -> ScrollOutput:
        """Register a callback that receives each log line."""
        self._sink = sink
        return self


class ProgressBar:
    """Rich-based progress bar with a status table and scrolling log area.

    @brief Wraps an iterator (or manual counter) and renders a stationary
    progress UI using :mod:`rich`. The UI contains:

      - A two-column status table built from ``table_map``.
      - A progress bar with percentage and `completed/total` units.
      - A scrolling log area fed by :class:`ScrollOutput`.

    @param table_map Initial key-value mapping displayed in the status table.
    @param total Total number of iterations expected.
    @param iterator Optional iterator to wrap; if provided, the bar is iterable.
    @param scroll_output Optional :class:`ScrollOutput` used for log lines.
    @param bar_length Explicit bar width (currently unused; reserved for future).
    @param update_rate Minimum refresh interval in iterations.
    @param table_callback Optional callback to mutate ``table_map`` on refresh.
    @param use_curses Deprecated flag; kept for API compatibility only.
    @param units_per_iter Scale factor for the UI display. ``total`` and the
        internal counter track iterations; the rendered `completed/total` values
        are multiplied by ``units_per_iter`` so the bar can display frames when
        iterating over batches.
    @param min_refresh_interval_seconds Minimum wall-clock interval between
        rendered refreshes. Fast iterators are throttled so the progress UI
        updates at most once per this interval.
    """

    def __init__(
        self,
        table_map: OrderedDict[Any, Any] = OrderedDict(),
        total: int = 0,
        iterator: Optional[Iterator[Any]] = None,
        scroll_output: Optional[ScrollOutput] = None,
        bar_length: Optional[int] = None,
        update_rate: int = 1,
        title: Optional[str] = None,
        table_callback: Optional[Callable] = None,
        use_curses: bool = True,
        units_per_iter: int = 1,
        min_refresh_interval_seconds: float = 1.0,
    ):
        self._total = total
        self._counter: int = 0
        # Many datasets yield batches (e.g., B frames at once). We keep the
        # progress iteration counter in "iters" but scale what the UI displays
        # so completed/total reflect frames when desired.
        try:
            self._units_per_iter = max(1, int(units_per_iter))
        except Exception:
            self._units_per_iter = 1
        self.table_map = table_map
        self.original_stdout = logging_out
        self.scroll_output = scroll_output
        self.iterator = iterator
        self.update_rate = update_rate
        self.table_callbacks: List[Callable] = []
        self.bar_length = bar_length
        self._use_curses_requested = use_curses  # Deprecated, no effect
        self._title = title
        try:
            self._min_refresh_interval_seconds = max(0.0, float(min_refresh_interval_seconds))
        except Exception:
            self._min_refresh_interval_seconds = 1.0
        self._last_render_time: Optional[float] = None
        if not self.bar_length:
            self.terminal_width = _get_terminal_width()
            # Re-evaluate the terminal width periodically to handle resizes.
            self.terminal_width_interval = 250
        else:
            self.terminal_width = None
        # Delay initializing/rendering the rich UI until after a warm-up
        # period so that early startup text does not interfere with it.
        self._start_threshold: int = 50
        if table_callback is not None:
            self.add_table_callback(table_callback)

        # Rich progress UI setup
        self._console = RICH_CONSOLE
        self._progress = RichProgress(
            TextColumn("Progress:", justify="left", style="white"),
            BarColumn(bar_width=None, complete_style="bright_green", finished_style="bright_green"),
            FramesColumn(),
            console=self._console,
            transient=False,
        )
        description = ""  # description column is a fixed label
        total_value = self._total if self._total > 0 else None
        self._rich_task = self._progress.add_task(description=description, total=total_value)
        self._rich_started: bool = False
        self._live: Optional[Live] = None
        self._line_count: int = 0
        self._log_lines: List[str] = []
        self._log_max_lines: int = scroll_output.lines if scroll_output is not None else 11
        self._extra_panel_callback: Optional[Callable[[], Any]] = None
        self._extra_panel_title: Optional[str] = None
        self._extra_panel_style: str = "white on grey19"

        if self.scroll_output is not None:
            # Route ScrollOutput lines into this ProgressBar's log buffer.
            self.scroll_output.set_sink(self._append_log_line)

    def add_table_callback(self, callback: Callable):
        self.table_callbacks.append(callback)

    def set_extra_panel_callback(
        self, callback: Optional[Callable[[], Any]], title: Optional[str] = None
    ) -> None:
        """Register an optional renderable callback for an extra UI panel."""
        self._extra_panel_callback = callback
        if title is not None:
            self._extra_panel_title = title

    def set_iterator(self, iterator: Iterator[Any]) -> Iterator[Any]:
        self.iterator = iterator
        return iter(self)

    def _run_callbacks(self, table_map: OrderedDict[Any, Any]):
        for cb in self.table_callbacks:
            cb(table_map)

    def _format_description(self) -> str:
        """Format the current table_map into a single-line description."""
        if not self.table_map:
            return ""
        parts = [f"{key}: {value}" for key, value in self.table_map.items()]
        return " | ".join(parts)

    def _append_log_line(self, line: str) -> None:
        """Append a log line to the in-memory buffer used for the log area."""
        self._log_lines.append(line)
        if len(self._log_lines) > self._log_max_lines:
            self._log_lines = self._log_lines[-self._log_max_lines :]

    def _build_table(self) -> Table:
        """Build a two-column-of-entries rich Table from the current table_map.

        Layout:
          [label_1] [value_1]   [label_2] [value_2]
        """
        table = Table.grid(expand=True, padding=(0, 1))
        # Black background for labels and values; text white for readability.
        table.style = "white on black"
        table.add_column(justify="left", style="bold white", ratio=1)
        table.add_column(justify="right", style="white", ratio=1)
        table.add_column(justify="left", style="bold white", ratio=1)
        table.add_column(justify="right", style="white", ratio=1)

        items = list(self.table_map.items())
        if not items:
            return table

        half = (len(items) + 1) // 2
        left_items = items[:half]
        right_items = items[half:]

        # Pad right side to match left length
        while len(right_items) < len(left_items):
            right_items.append(("", ""))

        for (k1, v1), (k2, v2) in zip(left_items, right_items):
            table.add_row(str(k1), str(v1), str(k2), str(v2))
        return table

    def _build_log_table(self) -> Table:
        """Build a table containing the scrolling log area."""
        log_table = Table.grid(expand=True, padding=(0, 1))
        # Do not set a background style here so that per-line rich markup
        # colors and the enclosing panel's background can show through.
        log_table.style = ""
        log_table.add_column(justify="left")
        # Take the last N lines for display
        for line in self._log_lines[-self._log_max_lines :]:
            try:
                rendered = Text.from_markup(line)
            except Exception:
                rendered = Text(str(line))
            log_table.add_row(rendered)
        return log_table

    def _build_layout(self) -> Group:
        """Compose the status table, progress bar, and log area inside a single bordered panel."""
        status_table = self._build_table()
        log_table = self._build_log_table()
        extra_panel = None
        if self._extra_panel_callback is not None:
            try:
                extra_renderable = self._extra_panel_callback()
            except Exception:
                extra_renderable = Text("Extra panel unavailable", style="dim")
            if extra_renderable is not None:
                extra_panel = Panel(
                    extra_renderable,
                    border_style="black",
                    style=self._extra_panel_style,
                    padding=(0, 1),
                    title=self._extra_panel_title,
                )

        title = None
        if self._title:
            title = Text(f" {self._title} ", style="yellow on dark_blue")
        status_panel = Panel(
            status_table,
            border_style="black",
            style="white on dark_blue",
            padding=(0, 1),
            title=title,
            title_align="center",
        )
        progress_panel = Panel(
            self._progress,
            border_style="black",
            style="white on grey35",
            padding=(0, 1),
        )
        log_panel = Panel(
            log_table,
            border_style="black",
            # Gray background for the scrolling log area; individual lines
            # can still use rich markup colors on top of this.
            style="on grey23",
            padding=(0, 1),
        )

        # Horizontal ASCII separators between sections
        sep = Rule(style="black")

        sections = [status_panel, sep, progress_panel]
        if extra_panel is not None:
            sections.extend([sep, extra_panel])
        sections.extend([sep, log_panel])
        body = Group(*sections)
        outer = Panel(
            body,
            border_style="black",
            padding=(0, 1),
        )
        return outer

    @property
    def total(self) -> int:
        return self._total

    @property
    def current(self) -> int:
        return self._counter

    def _ensure_rich_started(self) -> None:
        if not self._rich_started:
            # Render RichProgress inside a single outer Live context.
            # Calling self._progress.start() would create a second live display
            # on the same console and raise rich.errors.LiveError.
            self._live = Live(
                self._build_layout(),
                console=self._console,
                screen=True,
                auto_refresh=False,
            )
            self._live.start()
            self._rich_started = True

    def _render_rich(self, final: bool = False) -> None:
        """Render or update the rich progress bar and surrounding layout."""
        self._ensure_rich_started()
        # Allow callers to update table_map before we format the description/table.
        self._run_callbacks(self.table_map)
        # Progress bar label is a fixed prefix; tabular values are shown only
        # in the status table above, not in the bar description.
        description = ""
        if self._total > 0:
            completed_iters = min(self._counter, self._total)
            total_iters = self._total
        else:
            completed_iters = self._counter
            total_iters = None

        completed = (
            int(completed_iters) * self._units_per_iter
            if completed_iters is not None
            else int(self._counter) * self._units_per_iter
        )
        total_val = int(total_iters) * self._units_per_iter if total_iters is not None else None
        self._progress.update(
            self._rich_task, completed=completed, total=total_val, description=description
        )
        if final and total_val is not None:
            # Ensure we show a fully-complete bar if requested.
            self._progress.update(self._rich_task, completed=total_val, description=description)

        # Render the composed layout (status table + progress bar + log panel)
        # in-place using Live so the UI appears stationary in the terminal.
        renderable = self._build_layout()
        if self._live is not None:
            self._live.update(renderable, refresh=True)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = None
        # Drive the wrapped iterator when provided; otherwise behave as a manual counter.
        if self.iterator is not None:
            try:
                next_item = next(self.iterator)
            except StopIteration:
                # Final update on completion
                self.refresh(final=True)
                self.close()
                raise
        else:
            if self._counter >= self._total:
                self.refresh(final=True)
                self.close()
                raise StopIteration()

        if not self.bar_length and self._counter % self.terminal_width_interval == 0:
            self.terminal_width = _get_terminal_width()

        if self._counter % self.update_rate == 0:
            self.refresh()
            if next_item is None:
                time.sleep(0.001)  # Simulating work by sleeping

        self._counter += 1
        return next_item

    def close(self):
        # Tear down the rich progress / Live context if it was started.
        if getattr(self, "_rich_started", False):
            try:
                if self._live is not None:
                    self._live.stop()
            except Exception:
                pass
            self._rich_started = False
        _restore_cursor()

    def refresh(self, final: bool = False):
        # Avoid starting the rich UI until after the warm-up threshold so
        # that early startup output does not interfere with the layout.
        if not getattr(self, "_rich_started", False) and self._counter < self._start_threshold:
            # Still allow table callbacks to keep table_map up to date.
            self._run_callbacks(self.table_map)
            return
        if not final and self._min_refresh_interval_seconds > 0:
            now = time.monotonic()
            if self._last_render_time is not None:
                elapsed = now - self._last_render_time
                if elapsed < self._min_refresh_interval_seconds:
                    return
            self._last_render_time = now
        elif final:
            self._last_render_time = time.monotonic()
        # Single rich-backed rendering path once started
        self._render_rich(final=final)


def convert_seconds_to_hms(total_seconds: Any) -> str:
    hours = int(total_seconds // 3600)  # Calculate the number of hours
    minutes = int((total_seconds % 3600) // 60)  # Calculate the remaining minutes
    seconds = int(total_seconds % 60)  # Calculate the remaining seconds

    # Format the time in "HH:MM:SS" format
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def build_aspen_graph_renderable(snapshot: Dict[str, Any]) -> Table:
    """Build a rich renderable showing AspenNet graph activity and queue stats."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(justify="left")
    stats_parts = []
    concurrency = snapshot.get("concurrency") or {}
    if concurrency:
        if concurrency.get("threaded"):
            stats_parts.append(
                f"Concurrent: {concurrency.get('current', 0)}/{concurrency.get('max', 0)}"
            )
        else:
            stats_parts.append("Concurrent: serial")
    queues = snapshot.get("queues")
    if isinstance(queues, dict):
        items = queues.get("items") or []
        if items:
            if len(items) <= 6:
                parts = []
                for item in items:
                    label = item.get("label", "")
                    current = item.get("current", 0)
                    max_size = item.get("max")
                    if isinstance(max_size, int) and max_size > 0:
                        parts.append(f"{label}:{current}/{max_size}")
                    else:
                        parts.append(f"{label}:{current}")
                stats_parts.append("Queues: " + " ".join(parts))
            else:
                total_current = queues.get("total_current", 0)
                total_capacity = queues.get("total_capacity", None)
                count = queues.get("count", len(items))
                if isinstance(total_capacity, int):
                    stats_parts.append(f"Queues: {total_current}/{total_capacity} (n={count})")
                else:
                    stats_parts.append(f"Queues: {total_current} (n={count})")
    elif concurrency:
        if concurrency.get("threaded"):
            stats_parts.append("Queues: pending")
        else:
            stats_parts.append("Queues: off")
    if stats_parts:
        table.add_row(Text(" | ".join(stats_parts), style="white"))

    nodes = snapshot.get("nodes") or []
    if not nodes:
        table.add_row(Text("No AspenNet nodes", style="dim"))
        return table

    order = snapshot.get("order") or [node.get("name", "") for node in nodes]
    order_index = {name: idx for idx, name in enumerate(order)}
    node_queues = snapshot.get("node_queues") or {}

    max_degree = int(snapshot.get("max_degree", 0))
    levels: Dict[int, List[Dict[str, Any]]] = {}
    node_degree: Dict[str, int] = {}
    for node in nodes:
        degree = int(node.get("degree", 0))
        name = str(node.get("name", ""))
        node_degree[name] = degree
        levels.setdefault(degree, []).append(node)

    for degree, level_nodes in levels.items():
        level_nodes.sort(key=lambda n: order_index.get(str(n.get("name", "")), 0))
        levels[degree] = level_nodes

    max_nodes = max((len(level_nodes) for level_nodes in levels.values()), default=1)

    label_info: Dict[str, Dict[str, Any]] = {}
    max_label_len = 0
    for node in nodes:
        name = str(node.get("name", ""))
        active = bool(node.get("active", False))
        marker = "[#]" if active else "[ ]"
        q_label = ""
        q_info = node_queues.get(name)
        if isinstance(q_info, dict):
            current = q_info.get("current", 0)
            max_size = q_info.get("max")
            if isinstance(max_size, int) and max_size > 0:
                q_label = f" q:{current}/{max_size}"
            else:
                q_label = f" q:{current}"
        label = f"{marker} {name}{q_label}"
        max_label_len = max(max_label_len, len(label))
        label_info[name] = {"label": label, "active": active}

    slot_padding = 2
    slot_width = max_label_len + slot_padding
    gap = 4
    width = slot_width * max_nodes + gap * (max_nodes - 1 if max_nodes > 1 else 0)

    def _spread_positions(count: int, slots: int) -> List[int]:
        if count <= 0:
            return []
        if slots <= 1:
            return [0] * count
        if count == 1:
            return [slots // 2]
        positions: List[int] = []
        last = -1
        for idx in range(count):
            pos = int(round(idx * (slots - 1) / (count - 1)))
            if pos <= last:
                pos = last + 1
            positions.append(pos)
            last = pos
        if positions[-1] >= slots:
            start = max(0, (slots - count) // 2)
            positions = list(range(start, start + count))
        return positions

    level_layouts: Dict[int, List[Dict[str, Any]]] = {}
    node_centers: Dict[str, int] = {}
    for degree in range(max_degree + 1):
        level_nodes = levels.get(degree, [])
        if not level_nodes:
            continue
        slots = _spread_positions(len(level_nodes), max_nodes)
        layouts = []
        for node, slot in zip(level_nodes, slots):
            name = str(node.get("name", ""))
            info = label_info.get(name, {})
            label = info.get("label", "")
            active = bool(info.get("active", False))
            slot_start = slot * (slot_width + gap)
            offset = max(0, (slot_width - len(label)) // 2)
            pos = slot_start + offset
            center = pos + (len(label) // 2 if label else 0)
            node_centers[name] = center
            layouts.append(
                {
                    "name": name,
                    "label": label,
                    "active": active,
                    "pos": pos,
                }
            )
        level_layouts[degree] = layouts

    edges = snapshot.get("edges") or []
    edges_by_level: Dict[int, List[tuple]] = {}
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        src, dst = edge
        src_name = str(src)
        dst_name = str(dst)
        src_degree = node_degree.get(src_name)
        dst_degree = node_degree.get(dst_name)
        if src_degree is None or dst_degree is None:
            continue
        if dst_degree == src_degree + 1:
            edges_by_level.setdefault(src_degree, []).append((src_name, dst_name))

    def _merge_char(existing: str, new_char: str) -> str:
        if existing == " ":
            return new_char
        if existing == new_char:
            return existing
        if existing in "|-+" and new_char in "|-+":
            return "+"
        return existing

    for degree in range(max_degree + 1):
        layouts = level_layouts.get(degree, [])
        if not layouts:
            continue
        row_chars = [" "] * width
        label_spans = []
        marker_spans = []
        for entry in layouts:
            label = entry["label"]
            pos = entry["pos"]
            active = entry["active"]
            for idx, ch in enumerate(label):
                if 0 <= pos + idx < width:
                    row_chars[pos + idx] = ch
            label_spans.append((pos, pos + len(label), active))
            marker_spans.append((pos, pos + 3, active))
        row_text = Text("".join(row_chars))
        for start, end, active in label_spans:
            row_text.stylize("bold white" if active else "dim", start, end)
        for start, end, active in marker_spans:
            row_text.stylize("bold green" if active else "dim", start, end)
        table.add_row(row_text)

        connectors = edges_by_level.get(degree, [])
        if connectors:
            conn_chars = [" "] * width
            for src_name, dst_name in connectors:
                x1 = node_centers.get(src_name)
                x2 = node_centers.get(dst_name)
                if x1 is None or x2 is None:
                    continue
                if x1 == x2:
                    conn_chars[x1] = _merge_char(conn_chars[x1], "|")
                    continue
                start = min(x1, x2)
                end = max(x1, x2)
                conn_chars[start] = _merge_char(conn_chars[start], "+")
                conn_chars[end] = _merge_char(conn_chars[end], "+")
                for idx in range(start + 1, end):
                    conn_chars[idx] = _merge_char(conn_chars[idx], "-")
            conn_text = Text("".join(conn_chars))
            conn_text.stylize("dim", 0, len(conn_chars))
            table.add_row(conn_text)
    return table


def convert_hms_to_seconds(timestr: str) -> float:
    """
    Convert a time string in HH:MM:ss.ssss format to total seconds as a float.
    'HH:' may be omitted and 'MM' can be more than 59, as can 'ss' be more than 59.

    Parameters:
    - timestr (str): Time string in "HH:MM:ss.ssss" or "MM:ss.ssss" format.

    Returns:
    - float: Total seconds represented by the input string.
    """
    # Split the time string by colon
    parts = timestr.split(":")

    # Initialize seconds, minutes, and hours
    seconds = 0.0
    minutes = 0
    hours = 0

    # Depending on the number of components, assign values
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
    elif len(parts) == 1:
        seconds = float(parts[0])
    else:
        raise ValueError("Invalid time format")

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


# Example usage
if __name__ == "__main__":
    table_map = OrderedDict({"Key1": "Value1", "Key2": "Value2", "Key3": "Value3"})
    # Redirect stdout to scroll output handler
    scroll_output = ScrollOutput()
    progress_bar = ProgressBar(total=20, table_map=table_map, scroll_output=scroll_output)

    # with contextlib.redirect_stdout(scroll_output):
    with contextlib.redirect_stderr(scroll_output):
        counter = 0
        for _ in progress_bar:
            print(f"Simulated stdout message for iteration {counter}", file=sys.stderr)
            counter += 1
