"""A small directed graph for diagrams, written as DOT and Mermaid and
rendered with Graphviz when it is installed, or with matplotlib otherwise.

Graphviz (`dot`) lays out layered graphs well and draws nested clusters; it
is a system program, not a Python package, so it is used only when found on
PATH. The matplotlib fallback draws the same nodes and edges in layers,
without cluster boxes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

NODE_KINDS = (
    "input", "output", "module", "op", "component", "resource", "hook", "step",
)
EDGE_STYLES = ("solid", "dashed", "dotted")


@dataclass
class Node:
    key: str
    label: str
    kind: str
    cluster: tuple[str, ...] = ()
    """Keys of the clusters holding this node, outermost first."""
    order: int = 0
    """Where the node first appeared, for a stable layout."""


@dataclass
class Graph:
    title: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str], str] = field(default_factory=dict)
    """(source, target) -> label."""
    edge_styles: dict[tuple[str, str], str] = field(default_factory=dict)
    """(source, target) -> one of `EDGE_STYLES`; solid when absent."""
    cluster_labels: dict[str, str] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        if node.key not in self.nodes:
            node.order = len(self.nodes)
            self.nodes[node.key] = node

    def add_edge(
            self,
            source: str,
            target: str,
            label: str = "",
            style: str = "solid",
    ) -> None:
        if source == target or (source, target) in self.edges:
            return
        self.edges[(source, target)] = label
        if style != "solid":
            self.edge_styles[(source, target)] = style

    def predecessors(self, key: str) -> list[str]:
        return [s for (s, t) in self.edges if t == key]

    def successors(self, key: str) -> list[str]:
        return [t for (s, t) in self.edges if s == key]

    def remove_node(self, key: str) -> None:
        del self.nodes[key]
        self.edges = {
            pair: label for pair, label in self.edges.items() if key not in pair
        }
        self.edge_styles = {
            pair: style for pair, style in self.edge_styles.items()
            if key not in pair
        }


# -- text formats ----------------------------------------------------------------


_DOT_SHAPES = {
    "input": 'shape=box, style="rounded,filled", fillcolor="#e8f1fb"',
    "output": 'shape=box, style="rounded,filled", fillcolor="#e9f6ec"',
    "module": 'shape=box, style="rounded,filled", fillcolor="#fff6e0"',
    "component": 'shape=box, style="rounded,filled", fillcolor="#fff6e0"',
    "resource": 'shape=box, style="rounded,filled", fillcolor="#fff6e0"',
    "hook": 'shape=box, style="rounded,filled", fillcolor="#efe8fb"',
    "step": 'shape=box, style="rounded,filled", fillcolor="#e8f6f6"',
    "op": 'shape=ellipse, style=filled, fillcolor="#f2f2f2", fontsize=10',
}


def _dot_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _dot_id(key: str, ids: dict[str, str]) -> str:
    return ids.setdefault(key, f"n{len(ids)}")


def to_dot(graph: Graph) -> str:
    """The graph as Graphviz DOT, clusters nested."""
    ids: dict[str, str] = {}
    lines = [
        "digraph diagram {",
        "  rankdir=TB;",
        f'  label="{_dot_text(graph.title)}"; labelloc=t; fontsize=16;',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9, color="#555555"];',
    ]
    tree: dict[tuple[str, ...], list[Node]] = {}
    for node in sorted(graph.nodes.values(), key=lambda n: n.order):
        tree.setdefault(node.cluster, []).append(node)
    clusters = sorted(
        {node.cluster[:i] for node in graph.nodes.values()
         for i in range(1, len(node.cluster) + 1)},
        key=lambda path: (len(path), path),
    )

    def emit(path: tuple[str, ...], indent: str) -> None:
        for node in tree.get(path, []):
            lines.append(
                f'{indent}{_dot_id(node.key, ids)} '
                f'[label="{_dot_text(node.label)}", {_DOT_SHAPES[node.kind]}];'
            )
        for child in clusters:
            if len(child) == len(path) + 1 and child[:len(path)] == path:
                label = graph.cluster_labels.get(child[-1], child[-1])
                lines.append(f"{indent}subgraph cluster_{len(ids)}_{len(lines)} {{")
                lines.append(
                    f'{indent}  label="{_dot_text(label)}"; style="rounded,dashed"; '
                    'color="#999999"; fontsize=10;'
                )
                emit(child, indent + "  ")
                lines.append(f"{indent}}}")

    emit((), "  ")
    for (source, target), label in graph.edges.items():
        attributes = []
        if label:
            attributes.append(f'label="{_dot_text(label)}"')
        style = graph.edge_styles.get((source, target))
        if style:
            attributes.append(f"style={style}")
        suffix = f" [{', '.join(attributes)}]" if attributes else ""
        lines.append(
            f"  {_dot_id(source, ids)} -> {_dot_id(target, ids)}{suffix};"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _mermaid_text(text: str) -> str:
    return text.replace('"', "#quot;").replace("\n", "<br/>")


def to_mermaid(graph: Graph) -> str:
    """The graph as a Mermaid flowchart, clusters as subgraphs."""
    ids: dict[str, str] = {}
    lines = ["---", f"title: {graph.title}", "---", "flowchart TD"]
    shapes = {
        "op": ('(["', '"])'),
    }
    tree: dict[tuple[str, ...], list[Node]] = {}
    for node in sorted(graph.nodes.values(), key=lambda n: n.order):
        tree.setdefault(node.cluster, []).append(node)
    clusters = sorted(
        {node.cluster[:i] for node in graph.nodes.values()
         for i in range(1, len(node.cluster) + 1)},
        key=lambda path: (len(path), path),
    )

    def emit(path: tuple[str, ...], indent: str) -> None:
        for node in tree.get(path, []):
            open_, close = shapes.get(node.kind, ('["', '"]'))
            lines.append(
                f"{indent}{_dot_id(node.key, ids)}{open_}"
                f"{_mermaid_text(node.label)}{close}"
            )
        for child in clusters:
            if len(child) == len(path) + 1 and child[:len(path)] == path:
                label = graph.cluster_labels.get(child[-1], child[-1])
                lines.append(
                    f'{indent}subgraph c{len(lines)}["{_mermaid_text(label)}"]'
                )
                emit(child, indent + "  ")
                lines.append(f"{indent}end")

    emit((), "  ")
    for (source, target), label in graph.edges.items():
        line = "-.->" if (source, target) in graph.edge_styles else "-->"
        arrow = f'{line}|"{_mermaid_text(label)}"|' if label else line
        lines.append(
            f"  {_dot_id(source, ids)} {arrow} {_dot_id(target, ids)}"
        )
    return "\n".join(lines) + "\n"


# -- rendering -------------------------------------------------------------------


GRAPHVIZ_HINT = (
    "Install Graphviz for a better layout (`conda install -c conda-forge "
    "graphviz`, or `apt-get install graphviz`), or say where its `dot` is "
    "with `graphviz_dot` or the GRAPHVIZ_DOT environment variable."
)

GRAPHVIZ_DOT_VARIABLE = "GRAPHVIZ_DOT"


def find_dot(explicit: str | None = None) -> tuple[str | None, list[str]]:
    """Graphviz's `dot`, and every place looked.

    In order: `explicit` (the `graphviz_dot` setting), the GRAPHVIZ_DOT
    environment variable, PATH, then the directory of the running Python --
    where a conda or virtual environment installs it, found even when that
    environment was not activated (an IDE running its interpreter directly).
    An explicit or environment path that is not an executable file is not
    passed over: nothing else is tried, and it is reported.
    """
    executable = "dot.exe" if os.name == "nt" else "dot"
    searched: list[str] = []
    for given, where in (
            (explicit, "graphviz_dot"),
            (os.environ.get(GRAPHVIZ_DOT_VARIABLE), GRAPHVIZ_DOT_VARIABLE),
    ):
        if given:
            searched.append(f"{where}={given}")
            usable = os.path.isfile(given) and os.access(given, os.X_OK)
            return (given if usable else None), searched
    searched.append("PATH")
    on_path = shutil.which("dot")
    if on_path is not None:
        return on_path, searched
    beside = os.path.join(os.path.dirname(sys.executable), executable)
    searched.append(beside)
    if os.path.isfile(beside) and os.access(beside, os.X_OK):
        return beside, searched
    return None, searched


def write_diagram(
        graph: Graph,
        directory: str,
        stem: str,
        formats: tuple[str, ...],
        dpi: int,
        graphviz_dot: str | None = None,
) -> tuple[list[str], str | None]:
    """Write `<stem>.dot`, `<stem>.mmd` and one picture per format.

    Returns the files written and, when the pictures came from the
    matplotlib fallback, why Graphviz was not used.
    """
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, stem)
    written = []
    dot_text = to_dot(graph)
    for path, text in (
            (f"{base}.dot", dot_text),
            (f"{base}.mmd", to_mermaid(graph)),
    ):
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        written.append(path)

    fallback_reason = None
    dot, searched = find_dot(graphviz_dot)
    if dot is None:
        fallback_reason = (
            "Graphviz (`dot`) was not found; looked at " + ", ".join(searched)
        )
    else:
        try:
            for fmt in formats:
                command = [dot, f"-T{fmt}", "-o", f"{base}.{fmt}", f"{base}.dot"]
                if fmt == "png":
                    command.insert(1, f"-Gdpi={dpi}")
                subprocess.run(
                    command, check=True, capture_output=True, timeout=120,
                )
                written.append(f"{base}.{fmt}")
        except (OSError, subprocess.SubprocessError) as error:
            stderr = getattr(error, "stderr", b"") or b""
            fallback_reason = (
                f"Graphviz failed ({error}"
                + (f": {stderr.decode(errors='replace').strip()}" if stderr else "")
                + ")"
            )
            written = written[:2]
    if fallback_reason is not None:
        for fmt in formats:
            draw_with_matplotlib(graph, f"{base}.{fmt}", dpi=dpi)
            written.append(f"{base}.{fmt}")
    return written, fallback_reason


def _forward_edges(graph: Graph) -> list[tuple[str, str]]:
    """The edges that keep the graph acyclic: a depth-first search from each
    node in first-seen order drops every edge that closes a cycle (one back
    to a node still on the search path) -- and only those."""
    order = sorted(graph.nodes, key=lambda key: graph.nodes[key].order)
    successors = {key: [] for key in order}
    for source, target in graph.edges:
        successors[source].append(target)
    state: dict[str, int] = {}  # 1 on the path, 2 done
    forward = []
    for start in order:
        if start in state:
            continue
        stack = [(start, iter(successors[start]))]
        state[start] = 1
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                state[node] = 2
                stack.pop()
                continue
            if state.get(child) == 1:
                continue  # closes a cycle
            forward.append((node, child))
            if child not in state:
                state[child] = 1
                stack.append((child, iter(successors[child])))
    return forward


def _rows(graph: Graph) -> list[list[str]]:
    """Longest-path rows over the forward edges, then a few sweeps placing
    each node near the mean position of its neighbours in the rows above
    and below, nodes of one cluster kept side by side."""
    forward = _forward_edges(graph)
    predecessors = {key: [] for key in graph.nodes}
    successors = {key: [] for key in graph.nodes}
    for source, target in forward:
        predecessors[target].append(source)
        successors[source].append(target)
    row: dict[str, int] = {}

    def depth(key: str) -> int:
        if key not in row:
            row[key] = 0  # guards nothing: forward edges are acyclic
            row[key] = max((depth(p) + 1 for p in predecessors[key]), default=0)
        return row[key]

    for key in sorted(graph.nodes, key=lambda k: graph.nodes[k].order):
        depth(key)
    rows: list[list[str]] = [[] for _ in range(max(row.values(), default=-1) + 1)]
    for key in sorted(graph.nodes, key=lambda k: graph.nodes[k].order):
        rows[row[key]].append(key)

    def cluster(key: str) -> tuple[str, ...]:
        return graph.nodes[key].cluster

    for sweep in range(4):
        downward = sweep % 2 == 0
        indices = range(1, len(rows)) if downward else range(len(rows) - 2, -1, -1)
        for index in indices:
            neighbour_row = rows[index - 1] if downward else rows[index + 1]
            position = {key: i for i, key in enumerate(neighbour_row)}
            links = predecessors if downward else successors

            current = {key: i for i, key in enumerate(rows[index])}

            def barycentre(key: str, position=position, links=links, current=current) -> float:
                near = [position[k] for k in links[key] if k in position]
                return sum(near) / len(near) if near else float(current[key])

            rows[index].sort(key=lambda key: (cluster(key), barycentre(key)))
    return rows


_CHAR_WIDTH = 0.075   # inches per character at the label font size
_LINE_HEIGHT = 0.17   # inches per label line
_GAP_X = 0.35         # inches between boxes in a row
_GAP_Y = 0.75         # inches between rows (room for edge labels)


def fallback_layout(graph: Graph) -> dict[str, tuple[float, float, float, float]]:
    """Box of every node as (centre x, centre y, width, height) in inches,
    rows top-down, boxes sized from their labels so none overlap."""
    sizes = {
        key: (
            max(len(line) for line in node.label.split("\n")) * _CHAR_WIDTH + 0.3,
            len(node.label.split("\n")) * _LINE_HEIGHT + 0.2,
        )
        for key, node in graph.nodes.items()
    }
    rows = _rows(graph)
    widths = [
        sum(sizes[key][0] for key in row) + _GAP_X * (len(row) - 1)
        for row in rows
    ]
    total_width = max(widths, default=0.0)
    boxes = {}
    y = 0.0
    for row, width in zip(rows, widths):
        height = max((sizes[key][1] for key in row), default=0.0)
        x = (total_width - width) / 2
        for key in row:
            w, h = sizes[key]
            boxes[key] = (x + w / 2, -(y + height / 2), w, h)
            x += w + _GAP_X
        y += height + _GAP_Y
    return boxes


_FALLBACK_COLOURS = {
    "input": "#e8f1fb", "output": "#e9f6ec", "module": "#fff6e0",
    "component": "#fff6e0", "op": "#f2f2f2", "resource": "#fff6e0",
    "hook": "#efe8fb", "step": "#e8f6f6",
}


def draw_with_matplotlib(graph: Graph, path: str, *, dpi: int) -> None:
    """Draw `graph` with `fallback_layout`, using matplotlib's object API:
    no `pyplot`, so the process-wide backend is left alone."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch, Patch

    boxes = fallback_layout(graph)
    margin = 0.5
    title_lines = graph.title.count("\n") + 1
    left = min((x - w / 2 for x, _, w, _ in boxes.values()), default=0.0)
    right = max((x + w / 2 for x, _, w, _ in boxes.values()), default=1.0)
    bottom = min((y - h / 2 for _, y, _, h in boxes.values()), default=-1.0)
    width = right - left + 2 * margin
    height = -bottom + 2 * margin + 0.3 * title_lines
    figure = Figure(figsize=(max(width, 4.0), max(height, 2.0)))
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    axes.set_xlim(left - margin, left - margin + max(width, 4.0))
    axes.set_ylim(margin - max(height, 2.0), margin + 0.3 * title_lines)
    axes.text(
        (left + right) / 2, margin / 2 + 0.3 * title_lines, graph.title,
        ha="center", va="top", fontsize=11,
    )

    for (source, target), label in graph.edges.items():
        x0, y0, _, h0 = boxes[source]
        x1, y1, _, h1 = boxes[target]
        if abs(y0 - y1) < 1e-6:
            start, end, bend = (x0, y0 - h0 / 2), (x1, y1 - h1 / 2), 0.35
        elif y0 > y1:
            start, end, bend = (x0, y0 - h0 / 2), (x1, y1 + h1 / 2), 0.0
        else:
            start, end, bend = (x0, y0 + h0 / 2), (x1, y1 - h1 / 2), 0.25
        axes.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="->", color="#555555", shrinkA=0, shrinkB=0,
                connectionstyle=f"arc3,rad={bend}",
                linestyle=graph.edge_styles.get((source, target), "solid"),
            ),
        )
        if label:
            axes.text(
                (start[0] + end[0]) / 2, (start[1] + end[1]) / 2, label,
                fontsize=7, color="#555555", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
            )

    for key, (x, y, w, h) in boxes.items():
        node = graph.nodes[key]
        axes.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc=_FALLBACK_COLOURS[node.kind], ec="#666666", lw=0.8,
        ))
        axes.text(x, y, node.label, ha="center", va="center", fontsize=8)

    kinds = sorted({node.kind for node in graph.nodes.values()} & {"resource", "hook", "step"})
    if kinds:
        axes.legend(
            handles=[Patch(fc=_FALLBACK_COLOURS[kind], ec="#666666", label=kind) for kind in kinds],
            loc="lower left", fontsize=8, frameon=False,
        )
    figure.savefig(path, dpi=dpi)


_STEM = re.compile(r"\A[A-Za-z0-9_.-]+\Z")


def valid_stem(value: str) -> bool:
    return bool(_STEM.match(value)) and value not in (".", "..")
