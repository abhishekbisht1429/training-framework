"""The relations between components, listed and resolved in one place.

Every question the framework asks about a session's components -- what must
be active, what is built first, what runs first, what a rank must keep --
is answered from the same edges. Each kind of edge says which of those it
takes part in, so a new kind cannot be followed by one of them and missed by
another.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from training_framework.components.base import (
    Component,
    ComponentDependencyError,
    Hook,
    IterationHook,
    Resource,
    Step,
)
from training_framework.components.naming import (
    implementation_of,
    is_instance_name,
)


class EdgeKind(Enum):
    REQUIRES = "requires"
    """`@requires_resource` / `@requires_hook` / `@requires_step`."""

    WRAPS = "wraps"
    """`@wraps`: the wrapper runs its pre callbacks before the target's."""

    COMPANION = "companion"
    """`@activates`: brought along, nothing more."""

    READS = "reads"
    """A declared `iteration_context` read; the target is the key's writer."""


@dataclass(frozen=True)
class Edge:
    """One relation from a component to another, and what it implies.

    `asked` is the name as declared -- a component name, or for `READS` the
    context key. `target` is filled in once the edge is resolved for a
    particular session: an instance name, or the key's writer (None when
    nothing writes it).
    """

    kind: EdgeKind
    asked: str
    expected_type: type[Component] | None = None
    target: str | None = None

    @property
    def activates(self) -> bool:
        """Activating the source activates the target."""
        return self.kind is not EdgeKind.READS

    @property
    def builds_first(self) -> bool:
        """The target is constructed before the source."""
        return self.kind in (EdgeKind.REQUIRES, EdgeKind.WRAPS)

    @property
    def injects(self) -> bool:
        """The target is handed to the source by `get_dependency`."""
        return self.kind is EdgeKind.REQUIRES and self.expected_type is Resource

    @property
    def kept_with_source(self) -> bool:
        """A rank that keeps the source must keep the target.

        Every kind: a component missing whatever it requires, wraps, brings
        along or reads from cannot run.
        """
        return True

    @property
    def execution(self) -> str | None:
        """How the source runs relative to the target: "after", "before",
        or None when the edge does not order them."""
        if self.kind in (EdgeKind.REQUIRES, EdgeKind.READS):
            return "after"
        if self.kind is EdgeKind.WRAPS:
            return "before"
        return None

    @property
    def reason(self) -> str:
        """How the edge reads in a cycle report."""
        if self.kind is EdgeKind.READS:
            return f"reads '{self.asked}'"
        if self.kind is EdgeKind.WRAPS:
            return "wraps"
        return self.kind.value

    def resolved(self, target: str | None) -> "Edge":
        return replace(self, target=target)


def declared_edges(
        component: Component | type[Component],
        context_reads: Iterable[str] = (),
) -> list[Edge]:
    """Every edge `component` declares, unresolved, in a fixed order.

    The order -- resources, hooks, steps, wrap targets, companions, reads --
    is the order prerequisites have always been constructed in.
    `context_reads` supplies the keys an instance reads; they come from the
    instance, or from a checkpoint's record of it, never from the class.
    """
    edges = [
        Edge(EdgeKind.REQUIRES, name, category)
        for attribute, category in (
            ("required_resources", Resource),
            ("required_hooks", Hook),
            ("required_steps", Step),
        )
        for name in getattr(component, attribute, ())
    ]
    if _is_type(component, Hook):
        edges.extend(
            Edge(EdgeKind.WRAPS, name, Hook)
            for name in getattr(component, "wrapped_hooks", ())
        )
    edges.extend(
        Edge(EdgeKind.COMPANION, name)
        for name in getattr(component, "activated_components", ())
    )
    edges.extend(Edge(EdgeKind.READS, key) for key in context_reads)
    return edges


def _is_type(component, category) -> bool:
    if isinstance(component, type):
        return issubclass(component, category)
    return isinstance(component, category)


# -- resolving a name to an instance -------------------------------------------


def instances_of(name: str, active: Iterable[str]) -> list[str]:
    """Return the active instances `name` could refer to.

    An exact match is the answer on its own: a component named `model` stays
    the answer to `model` however many `model#...` instances join it, so
    adding an instance never silently rewires anything. A suffixed name is a
    precise reference: when that instance is not active the answer is "not
    configured", never a sibling that happens to share its implementation.
    """
    active = set(active)
    if name in active:
        return [name]
    if is_instance_name(name):
        return []
    implementation = implementation_of(name)
    return sorted(
        instance for instance in active
        if implementation_of(instance) == implementation
    )


def resolve_component_name(
        bindings,
        name: str,
        *,
        consumer: str | None,
        active: Iterable[str],
) -> str:
    """Return the instance name that satisfies `name` for `consumer`.

    Three rules, in order: the consumer's own wiring decides if it has any;
    otherwise the sole active instance of the component; otherwise it is an
    error naming the candidates. Picking one of several would mean wiring a
    component to something its session never chose, which produces a run
    that works and is quietly wrong. A name with nothing active behind it is
    returned as bound, for the caller -- which knows what it was looking for
    -- to report.
    """
    resolved = bindings.resolve(name, consumer=consumer)
    candidates = instances_of(resolved, active)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return resolved
    # Imported lazily: diagnostics imports the registry, which imports this.
    from training_framework.components.diagnostics import with_explanation

    raise ComponentDependencyError(with_explanation(
        f"Component '{name}' resolves to '{resolved}', which {len(candidates)} "
        f"active components implement: {candidates}.",
        "Fix: name the one that is meant with per-component wiring, "
        "component_bindings: {'"
        f"{consumer if consumer is not None else '<component>'}"
        "': {'"
        f"{name}': '{candidates[0]}'"
        "}}.",
    ))


def resolve_edges(
        component: Component | type[Component],
        *,
        consumer: str,
        bindings,
        active: Iterable[str],
        context_reads: Iterable[str] = (),
        writers: Mapping[str, str] | None = None,
) -> list[Edge]:
    """`declared_edges`, each resolved for `consumer` in this session.

    Names resolve through the bindings; a read resolves to its key's writer
    in `writers` (None when there is none, for the caller to report).
    """
    active = set(active)
    writers = writers or {}
    return [
        edge.resolved(
            writers.get(edge.asked)
            if edge.kind is EdgeKind.READS
            else resolve_component_name(
                bindings, edge.asked, consumer=consumer, active=active,
            )
        )
        for edge in declared_edges(component, context_reads)
    ]


# -- iteration_context keys ------------------------------------------------------


ContextKeys = Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]
"""Component name -> (keys it reads, keys it writes)."""


def takes_part_in_iterations(component: Any) -> bool:
    """Whether a component runs inside each iteration and can pass values:
    a step (`run`) or an iteration hook (pre / post callbacks). A session
    hook has no iteration callback, so it can neither read nor write."""
    return _is_type(component, (Step, IterationHook))


_MISSING = object()


def is_valid_cadence(value: Any) -> bool:
    """Whether `value` can be a `call_every`: a positive, non-bool integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def iteration_cadence(component: Any) -> int:
    """How often a component runs its part of an iteration, validated.

    Every participant runs on the first and final iteration and on the
    multiples of its cadence: 1 for a step, `call_every` for an iteration
    hook. So one runs on every iteration another does exactly when its
    cadence is a positive multiple of the other's. The runtime takes the
    same value modulo each iteration, so an invalid one is reported here,
    when the session is built, rather than as an arithmetic error mid-run.
    """
    if _is_type(component, Step):
        return 1
    value = getattr(component, "call_every", _MISSING)
    if not is_valid_cadence(value):
        shown = "<missing>" if value is _MISSING else repr(value)
        raise RuntimeError(
            f"{getattr(component, 'id', component)} call_every must be a "
            f"positive integer; got {shown}. It sets on which iterations the "
            "hook runs: the first, the final, and every multiple of it."
        )
    return value


def context_keys_of(components: Iterable[Component]) -> dict:
    """The keys each component reads and writes, validated.

    Only steps and iteration hooks take part in an iteration, so any other
    component declaring keys -- through `@reads`/`@writes` on a base class,
    or by overriding `context_reads()` -- is rejected here, where every
    use of the keys starts.
    """
    keys = {}
    for component in components:
        if isinstance(component, type):
            # A class sorted on its own answers with its declarations; only
            # an instance has configuration that can change them.
            reads = tuple(getattr(component, "declared_reads", ()))
            writes = tuple(getattr(component, "declared_writes", ()))
        else:
            reads = tuple(component.context_reads())
            writes = tuple(component.context_writes())
        if not reads and not writes:
            continue
        if not takes_part_in_iterations(component):
            raise RuntimeError(
                f"{component.id} declares iteration_context keys (reads "
                f"{list(reads)}, writes {list(writes)}), but only steps and "
                "iteration hooks take part in an iteration; a "
                f"{type(component).__name__} never reads or writes them."
            )
        keys[component.name] = (reads, writes)
    return keys


def recorded_context_keys(components_state: Mapping[str, Mapping]) -> dict:
    """The keys a checkpoint recorded for each component, for deciding what
    a rank keeps before anything is rebuilt. A state written before keys were
    recorded holds none, which is what it declared."""
    return {
        name: (
            tuple(info.get("context_reads", ())),
            tuple(info.get("context_writes", ())),
        )
        for name, info in components_state.items()
        if info.get("context_reads") or info.get("context_writes")
    }


def writers_of(keys: ContextKeys) -> dict[str, str]:
    """Key -> the component that writes it. Uniqueness is the sort's to
    check; this only answers who writes what."""
    return {
        key: name
        for name, (_, writes) in keys.items()
        for key in writes
    }


__all__ = [
    "ContextKeys",
    "Edge",
    "EdgeKind",
    "context_keys_of",
    "declared_edges",
    "instances_of",
    "is_valid_cadence",
    "iteration_cadence",
    "recorded_context_keys",
    "resolve_component_name",
    "resolve_edges",
    "takes_part_in_iterations",
    "writers_of",
]
