"""Component wiring: which instance fills each role a component asked for.

Two scopes: the model and everything it is wired to (`build_wiring_graph`),
and every component of the session with what each requires, wraps, brings
along and reads (`build_session_graph`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import nn

from training_framework.components.base import Component, Hook, Resource, Step
from training_framework.components.builtin.diagram.graph import Graph, Node
from training_framework.components.edges import Edge, EdgeKind


def _parameter_counts(module: nn.Module, others: set[int]) -> tuple[int, int]:
    """(trainable, frozen) parameters `module` holds itself -- not those of
    the components it is wired to, which they count."""
    trainable = frozen = 0
    for parameter in module.parameters():
        if id(parameter) in others:
            continue
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return trainable, frozen


def _count(number: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if number >= size:
            return f"{number / size:.1f}{unit}"
    return str(number)


def component_label(component: Component, imported: Mapping[str, str]) -> str:
    """Instance name, class, the parameters it holds itself (not those of
    the components it was handed, which they count) and where it was
    imported from."""
    lines = [component.name, type(component).__name__]
    if isinstance(component, nn.Module):
        others = {
            id(parameter)
            for dependency in component._dependencies.values()
            if isinstance(dependency, nn.Module)
            for parameter in dependency.parameters()
        }
        trainable, frozen = _parameter_counts(component, others)
        if trainable or frozen:
            lines.append(
                f"params {_count(trainable)}"
                + (f" (+{_count(frozen)} frozen)" if frozen else "")
            )
    if component.name in imported:
        lines.append(f"imported by {imported[component.name]}")
    return "\n".join(lines)


def build_wiring_graph(
        root: Component,
        *,
        title: str,
        imported: Mapping[str, str],
) -> Graph:
    """`root` and every component it is wired to, transitively; an edge per
    role asked for, labelled with the role. `imported` maps an imported
    instance to the import that brought it in."""
    graph = Graph(title=title)
    pending = [root]
    seen: set[int] = set()
    while pending:
        component = pending.pop(0)
        if id(component) in seen:
            continue
        seen.add(id(component))
        dependencies = dict(component._dependencies)
        graph.add_node(Node(
            component.name, component_label(component, imported), "component",
        ))
        for asked, dependency in dependencies.items():
            name = getattr(dependency, "name", type(dependency).__name__)
            graph.add_edge(
                component.name, name, "" if asked == name else asked,
            )
            if isinstance(dependency, Component):
                pending.append(dependency)
    return graph


_CATEGORIES = (
    (Resource, "resource", "Resources"),
    (Hook, "hook", "Hooks"),
    (Step, "step", "Steps"),
)

SESSION_LEGEND = (
    "solid: requires (role) - dashed: data (iteration_context key) - "
    "dotted: wraps / activates"
)


def build_session_graph(
        components: Iterable[Component],
        edges: Mapping[str, list[Edge]],
        *,
        title: str,
        imported: Mapping[str, str],
) -> Graph:
    """Every component, grouped by kind, with its resolved `edges`:
    what it requires (solid, labelled with the role when it differs from
    the instance), the writer of each key it reads (dashed, writer -> reader,
    labelled with the key), what it wraps or brings along (dotted)."""
    graph = Graph(title=f"{title}\n{SESSION_LEGEND}")
    components = list(components)
    for category, kind, cluster in _CATEGORIES:
        graph.cluster_labels[cluster] = cluster
        for component in components:
            if isinstance(component, category):
                graph.add_node(Node(
                    component.name, component_label(component, imported),
                    kind, (cluster,),
                ))
    for name, component_edges in edges.items():
        if name not in graph.nodes:
            continue
        for edge in component_edges:
            target = edge.target
            if target is None or target not in graph.nodes:
                continue
            if edge.kind is EdgeKind.READS:
                graph.add_edge(target, name, edge.asked, "dashed")
            elif edge.kind is EdgeKind.REQUIRES:
                graph.add_edge(
                    name, target, "" if edge.asked == target else edge.asked,
                )
            elif edge.kind is EdgeKind.WRAPS:
                graph.add_edge(name, target, "wraps", "dotted")
            else:
                graph.add_edge(name, target, "activates", "dotted")
    return graph


def component_module_paths(root: nn.Module) -> dict[str, str]:
    """Module path -> instance name, for every submodule of `root` that is
    itself a framework component (a composite's role-named blocks)."""
    return {
        path: module.name
        for path, module in root.named_modules()
        if path and isinstance(module, Component)
        and isinstance(getattr(module, "name", None), str)
    }
