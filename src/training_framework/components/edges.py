"""The relations between components, listed and resolved in one place.

Every question the framework asks about a session's components -- what must
be active, what is built first, what runs first, what a rank must keep --
is answered from the same edges. Each kind of edge says which of those it
takes part in, so a new kind cannot be followed by one of them and missed by
another.
"""

from __future__ import annotations

import inspect
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
        given: Mapping[str, str] | None = None,
) -> list[Edge]:
    """`declared_edges`, each resolved for `consumer` in this session.

    An injected edge the consumer was already handed an instance for
    (`given`: asked name -> instance) resolves to that instance. Other names
    resolve through the bindings; a read resolves to its key's writer in
    `writers` (None when there is none, for the caller to report).
    """
    active = set(active)
    writers = writers or {}
    given = given or {}

    def target(edge: Edge) -> str | None:
        if edge.kind is EdgeKind.READS:
            return writers.get(edge.asked)
        if edge.injects and edge.asked in given:
            return given[edge.asked]
        return resolve_component_name(
            bindings, edge.asked, consumer=consumer, active=active,
        )

    return [
        edge.resolved(target(edge))
        for edge in declared_edges(component, context_reads)
    ]


# -- the wiring components were given --------------------------------------------


Wiring = Mapping[str, Mapping[str, str]]
"""Component name -> (asked name -> the instance it was given)."""


def given_instance(consumer: Any, name: str) -> str | None:
    """The instance a live `consumer` was handed for `name`, if any.

    A class, or an instance given nothing under that name, answers None.
    """
    if isinstance(consumer, type):
        return None
    injected = getattr(consumer, "__dict__", {}).get(Component.DEPENDENCIES_ATTR)
    if not injected or name not in injected:
        return None
    dependency = injected[name]
    return getattr(dependency, "name", type(dependency).__name__)


def wiring_of(components: Iterable[Component]) -> dict:
    """What each live component was given, by asked name."""
    return {
        component.name: {
            asked: getattr(dependency, "name", type(dependency).__name__)
            for asked, dependency in component._dependencies.items()
        }
        for component in components
        if component._dependencies
    }


def recorded_wiring(components_state: Mapping[str, Mapping]) -> dict:
    """The wiring a checkpoint recorded for each component, for deciding
    what a rank keeps before anything is rebuilt. A state written before
    wiring was recorded holds none, and is resolved from its bindings."""
    return {
        name: dict(info["dependencies"])
        for name, info in components_state.items()
        if info.get("dependencies")
    }


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


def _describe(component: Any) -> str:
    if isinstance(component, type):
        return component.__name__
    return getattr(component, "id", type(component).__name__)


def context_mapping(component: Any, side: str) -> dict[str, str]:
    """What `component` reads (`side="reads"`) or writes, as name -> key,
    validated.

    A class answers with its declarations, where the name is the key; an
    instance with `context_reads()` / `context_writes()`, which
    configuration can change.
    """
    if isinstance(component, type):
        declared = getattr(component, f"declared_{side}", ())
        return {key: key for key in declared}
    mapping = getattr(component, f"context_{side}")()
    method = f"context_{side}()"
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"{_describe(component)}.{method} must return a mapping of "
            f"name -> iteration_context key; got {mapping!r}"
        )
    for name, key in mapping.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{_describe(component)}.{method} names must be non-empty "
                f"strings; got {name!r}"
            )
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"{_describe(component)}.{method} maps {name!r} to {key!r}; "
                "a key must be a non-empty string"
            )
    repeated = sorted({
        key for key in mapping.values()
        if list(mapping.values()).count(key) > 1
    })
    if repeated:
        raise ValueError(
            f"{_describe(component)}.{method} names iteration_context "
            f"{repeated} more than once"
        )
    return dict(mapping)


def context_keys(component: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The keys `component` reads and writes, validated."""
    return (
        tuple(context_mapping(component, "reads").values()),
        tuple(context_mapping(component, "writes").values()),
    )


def reading_callback(component: Any) -> str:
    """The callback a participant is given its reads in: a step's `run`, an
    iteration hook's `post_iteration_callback`."""
    return "run" if _is_type(component, Step) else "post_iteration_callback"


def check_callback_signature(component: Any, reads: Iterable[str]) -> None:
    """Hold a step's `run`, or an iteration hook's post callback, to the
    reads it declares.

    The runtime calls it with the session and then each read as a keyword
    argument, and nothing else. So every read must be a parameter (or
    absorbed by `**kwargs`), and every other parameter must have a default;
    a mismatch is reported now, when the session is built, rather than as a
    TypeError mid-run -- or, worse, as a parameter quietly left at its
    default.
    """
    callback_name = reading_callback(component)
    # The function on the class, so `self` is visible for an instance too:
    # a read may not land on it any more than on the session.
    cls = component if isinstance(component, type) else type(component)
    callback = getattr(cls, callback_name, None)
    if callback is None:
        return
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return
    parameters = list(signature.parameters.values())
    shown = f"{_describe(component)}.{callback_name}"
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    # `self` and the session are passed positionally.
    filled = []
    while parameters and len(filled) < 2 and parameters[0].kind in positional:
        filled.append(parameters.pop(0))
    if len(filled) < 2 and not any(
            p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters
    ):
        raise TypeError(
            f"{shown} must take the session as its first parameter"
        )
    reads = list(reads)
    taken = {
        p.name: p for p in filled
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    }
    colliding = [name for name in reads if name in taken]
    if colliding:
        raise TypeError(
            f"{shown} declares it reads {colliding}, which "
            f"{'is' if len(colliding) == 1 else 'are'} also the name of "
            f"a parameter the session fills positionally: {shown}{signature}. "
            "Passing the read by keyword would give that parameter two "
            "values. Take the read under another name (context_reads() "
            "returning e.g. {'value': 'session'}), or make those parameters "
            "positional-only: run(self, session, /, **values)."
        )
    by_keyword = {
        p.name for p in parameters
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    takes_any = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters
    )
    unaccepted = [name for name in reads if name not in by_keyword]
    if unaccepted and not takes_any:
        raise TypeError(
            f"{shown} declares it reads {unaccepted}, but does not take "
            f"{'it' if len(unaccepted) == 1 else 'them'} as keyword "
            f"parameters: {shown}{inspect.signature(callback)}. Each read is "
            "passed as the keyword argument of its name; add the parameter "
            "(or **kwargs, for names that are not identifiers)."
        )
    undeclared = [
        p.name for p in parameters
        if p.default is inspect.Parameter.empty
        and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        and p.name not in reads
    ]
    if undeclared:
        raise TypeError(
            f"{shown} has parameters {undeclared} that nothing fills: only "
            "declared reads are passed. Declare them with @reads(...) (or in "
            f"context_reads()), or give them defaults. Declared reads: {reads}."
        )


def context_keys_of(components: Iterable[Component]) -> dict:
    """The keys each component reads and writes, validated.

    Only steps and iteration hooks take part in an iteration, so any other
    component declaring keys -- through `@reads`/`@writes` on a base class,
    or by overriding `context_reads()` -- is rejected here, where every
    use of the keys starts. A participant's callback is checked against the
    reads it declares here too, so a mismatch surfaces when the session is
    built.
    """
    keys = {}
    for component in components:
        read_names = list(context_mapping(component, "reads"))
        reads, writes = context_keys(component)
        if takes_part_in_iterations(component):
            check_callback_signature(component, read_names)
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
    "Wiring",
    "check_callback_signature",
    "context_keys",
    "context_keys_of",
    "context_mapping",
    "declared_edges",
    "given_instance",
    "instances_of",
    "is_valid_cadence",
    "iteration_cadence",
    "recorded_context_keys",
    "reading_callback",
    "recorded_wiring",
    "resolve_component_name",
    "resolve_edges",
    "takes_part_in_iterations",
    "wiring_of",
    "writers_of",
]
