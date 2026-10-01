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
    "Install Graphviz for a better layout: `conda install -c conda-forge "
    "graphviz`, or your system's package (`apt-get install graphviz`)."
)


def write_diagram(
        graph: Graph,
        directory: str,
        stem: str,
        formats: tuple[str, ...],
        dpi: int,
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
    dot = shutil.which("dot")
    if dot is None:
        fallback_reason = "Graphviz (`dot`) is not on PATH"
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


def _layers(graph: Graph) -> list[list[str]]:
    """Longest-path layers, ignoring edges that point back in first-seen
    order (a cycle formed by merging modules), then one barycentre pass."""
    order = {key: node.order for key, node in graph.nodes.items()}
    layer: dict[str, int] = {}
    for key in sorted(graph.nodes, key=order.__getitem__):
        preds = [
            p for p in graph.predecessors(key) if order[p] < order[key]
        ]
        layer[key] = max((layer[p] + 1 for p in preds), default=0)
    rows: list[list[str]] = [[] for _ in range(max(layer.values(), default=-1) + 1)]
    for key in sorted(graph.nodes, key=order.__getitem__):
        rows[layer[key]].append(key)
    for index in range(1, len(rows)):
        position = {key: i for i, key in enumerate(rows[index - 1])}

        def barycentre(key: str) -> float:
            above = [position[p] for p in graph.predecessors(key) if p in position]
            return sum(above) / len(above) if above else float(len(position))

        rows[index].sort(key=barycentre)
    return rows


_FALLBACK_COLOURS = {
    "input": "#e8f1fb", "output": "#e9f6ec", "module": "#fff6e0",
    "component": "#fff6e0", "op": "#f2f2f2", "resource": "#fff6e0",
    "hook": "#efe8fb", "step": "#e8f6f6",
}


def draw_with_matplotlib(graph: Graph, path: str, *, dpi: int) -> None:
    """Draw `graph` in layers with matplotlib's object API: no `pyplot`,
    so the process-wide backend is left alone."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = _layers(graph)
    width = max((len(row) for row in rows), default=1)
    figure = Figure(figsize=(max(6.0, 3.2 * width), max(4.0, 1.4 * len(rows) + 1)))
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.text(0.5, 0.99, graph.title, ha="center", va="top", fontsize=12)
    where: dict[str, tuple[float, float]] = {}
    for depth, row in enumerate(rows):
        y = 1 - (depth + 1) / (len(rows) + 1)
        for index, key in enumerate(row):
            where[key] = ((index + 1) / (len(row) + 1), y)
    for (source, target), label in graph.edges.items():
        (x0, y0), (x1, y1) = where[source], where[target]
        axes.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color="#555555", shrinkA=18, shrinkB=18,
                linestyle=graph.edge_styles.get((source, target), "solid"),
            ),
        )
        if label:
            axes.text(
                (x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=7,
                color="#555555", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
            )
    for key, (x, y) in where.items():
        node = graph.nodes[key]
        axes.text(
            x, y, node.label, ha="center", va="center", fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.4" if node.kind != "op" else "circle,pad=0.3",
                fc=_FALLBACK_COLOURS[node.kind], ec="#666666",
            ),
        )
    figure.savefig(path, dpi=dpi)


_STEM = re.compile(r"\A[A-Za-z0-9_.-]+\Z")


def valid_stem(value: str) -> bool:
    return bool(_STEM.match(value)) and value not in (".", "..")
