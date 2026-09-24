from collections import deque
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from training_framework.components.base import (
    Component,
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Step,
)
from training_framework.components.edges import (
    EdgeKind,
    context_keys_of,
    declared_edges,
    given_instance,
    is_valid_cadence,
    iteration_cadence,
    resolve_component_name,
)
from training_framework.components.naming import implementation_of

if TYPE_CHECKING:
    from training_framework.components.registry import RoleDeclaration


_ROLE_DECORATOR_NAMES = {Resource: "resource", Hook: "hook", Step: "step"}


def _missing_role_message(
        *,
        category: type[Component],
        name: str,
        resolved_name: str,
        declared_role: "RoleDeclaration",
        consumer: Component | type[Component],
) -> str:
    consumer_class = consumer if isinstance(consumer, type) else type(consumer)
    description = (
        f": {declared_role.description}" if declared_role.description else ""
    )
    decorator_name = _ROLE_DECORATOR_NAMES[category]
    return (
        f"Role '{resolved_name}' ({category.__name__}{description}) is "
        f"required by {consumer_class.__name__} but has no implementation "
        f"registered. Implement a {category.__name__} subclass and register "
        f"it via @{decorator_name}('{resolved_name}', ...), or bind an "
        "existing implementation via component_bindings: "
        f"{{'{name}': '<implementation_name>'}}."
    )


def _resolve_to_node(binding_resolver, nodes_by_name, consumer, name):
    """Return the node satisfying `name` for `consumer`, and the name tried.

    A live consumer that was already handed an instance for `name` is
    answered with that instance: what it holds is what it uses, whatever the
    bindings would say now -- a hand-registered sibling, or a checkpoint
    restored with the wiring it recorded, must not make it ambiguous.
    Otherwise resolution is `resolve_component_name`, the one the session
    uses: the consumer's own wiring first, then the sole node the name can
    mean, and an error naming the candidates when several could. The node is
    None when nothing active answers to the name, for the caller to report.
    """
    given = given_instance(consumer, name)
    if given is not None:
        return given, nodes_by_name.get(given)
    resolved_name = resolve_component_name(
        binding_resolver,
        name,
        consumer=getattr(consumer, "name", None),
        active=nodes_by_name,
    )
    return resolved_name, nodes_by_name.get(resolved_name)


def _is_component_type(
        component: Component | type[Component],
        component_type: type[Component],
) -> bool:
    if isinstance(component, type):
        return issubclass(component, component_type)
    return isinstance(component, component_type)


def _validate_wrapping_lifecycle(
        wrapper: Component | type[Component],
        wrapped: Component | type[Component],
        *,
        session_scoped: bool,
) -> None:
    shares_session_phase = (
        _is_component_type(wrapper, SessionHook)
        and _is_component_type(wrapped, SessionHook)
    )
    shares_iteration_phase = (
        _is_component_type(wrapper, IterationHook)
        and _is_component_type(wrapped, IterationHook)
    )
    if not shares_session_phase and not shares_iteration_phase:
        raise RuntimeError(
            f"Hook '{wrapper.name}' cannot wrap Hook '{wrapped.name}' because "
            "they do not share a lifecycle phase"
        )

    if not session_scoped or not shares_iteration_phase:
        return

    missing = object()
    wrapper_cadence = getattr(wrapper, "call_every", missing)
    wrapped_cadence = getattr(wrapped, "call_every", missing)
    valid_cadences = (
        is_valid_cadence(wrapper_cadence) and is_valid_cadence(wrapped_cadence)
    )
    if (
            wrapper_cadence is missing
            or wrapped_cadence is missing
            or not valid_cadences
            or wrapper_cadence % wrapped_cadence != 0
    ):
        wrapper_value = (
            "<missing>" if wrapper_cadence is missing
            else repr(wrapper_cadence)
        )
        wrapped_value = (
            "<missing>" if wrapped_cadence is missing
            else repr(wrapped_cadence)
        )
        raise RuntimeError(
            f"Wrapping iteration hooks '{wrapper.name}' and '{wrapped.name}' "
            "require the wrapper call_every to be a positive multiple of the "
            "wrapped Hook call_every; "
            f"got {wrapper_value} and {wrapped_value}"
        )


def topological_sort_components(
        *,
        binding_resolver,
        registry: Mapping[str, type[Component]],
        components: Iterable[Component | type[Component]] | None,
        roles: "Mapping[str, RoleDeclaration] | None" = None,
        session_type: str | None = None,
) -> dict[str, int]:
    # Imported lazily: diagnostics imports the registry, which imports graph.
    from training_framework.components.diagnostics import (
        explain_missing_component,
        with_explanation,
    )

    roles = roles or {}
    session_scoped = components is not None
    selected_components = (
        list(registry.values())
        if components is None
        else list(components)
    )

    prerequisites_graph: dict[str, list[str]] = {
        component.id: [] for component in selected_components
    }
    # Which node answers to a name. Sorting a session these are instances, so
    # two instances of one component are two nodes; sorting the registry
    # itself they are the classes. Either way a node is found by its own
    # `name`, which is why an edge cannot be resolved to the registered class
    # -- that would collapse every instance of a component onto one node.
    nodes_by_name = {
        component.name: component
        for component in selected_components
    }
    active_names = set(nodes_by_name)
    # Why each edge exists, so a cycle can be reported as a chain of reasons.
    # Keyed (dependent id, prerequisite id).
    edge_reasons: dict[tuple[str, str], str] = {}

    # Keys are what the session's instances declare; the registry-wide sort
    # has no instances, and unrelated classes may well share key names.
    context_keys = context_keys_of(selected_components) if session_scoped else {}
    writers = _checked_writers(nodes_by_name, context_keys)

    for component in selected_components:
        reads, writes = context_keys.get(component.name, ((), ()))
        wrapped_targets: set[str] = set()
        for edge in declared_edges(component, reads):
            if edge.kind is EdgeKind.READS:
                _add_read_edge(
                    component, edge, writes, writers, nodes_by_name,
                    prerequisites_graph, edge_reasons,
                )
                continue

            resolved_name, node = _resolve_to_node(
                binding_resolver, nodes_by_name, component, edge.asked,
            )
            # A name may identify an instance; the class it is an instance
            # of is what the registry holds.
            registered_class = registry.get(implementation_of(resolved_name))

            if edge.kind is EdgeKind.COMPANION:
                # Companions add no ordering edge; they are checked here
                # because this is where every way a session comes to hold
                # its components -- activation, hand registration, restore --
                # converges.
                _check_companion(
                    component, edge.asked, resolved_name, node,
                    registered_class, session_scoped, session_type,
                    active_names, explain_missing_component, with_explanation,
                )
                continue

            is_wrap = edge.kind is EdgeKind.WRAPS
            required_type = edge.expected_type
            if (
                    registered_class is None
                    or not issubclass(registered_class, required_type)
            ):
                declared_role = (
                    roles.get(resolved_name) if registered_class is None else None
                )
                if (
                        declared_role is not None
                        and declared_role.category is required_type
                ):
                    raise RuntimeError(
                        _missing_role_message(
                            category=required_type,
                            name=edge.asked,
                            resolved_name=resolved_name,
                            declared_role=declared_role,
                            consumer=component,
                        )
                    )
                raise RuntimeError(with_explanation(
                    (
                        f"invalid wraps target! Hook '{edge.asked}' resolves "
                        f"to '{resolved_name}', which is not registered as a "
                        "Hook."
                        if is_wrap else
                        f"unmet prerequisite! {required_type.__name__} "
                        f"'{edge.asked}' resolves to '{resolved_name}', which "
                        f"is not registered as a {required_type.__name__}."
                    ),
                    explain_missing_component(
                        edge.asked,
                        resolved_name,
                        expected_type=required_type,
                        session_type=session_type,
                        consumer=component,
                    ),
                ))

            target_id = node.id if node is not None else registered_class.id
            if is_wrap:
                if target_id == component.id:
                    raise RuntimeError(
                        f"Hook '{component.name}' cannot wrap itself"
                    )
                if target_id in wrapped_targets:
                    raise RuntimeError(
                        f"Hook '{component.name}' wraps Hook '{resolved_name}' "
                        "more than once after component binding resolution"
                    )
                wrapped_targets.add(target_id)

            if session_scoped and node is None:
                raise RuntimeError(with_explanation(
                    (
                        f"invalid wraps target! Hook '{edge.asked}' resolves "
                        f"to '{resolved_name}', which is not configured in "
                        "this session."
                        if is_wrap else
                        f"unmet prerequisite! {required_type.__name__} "
                        f"'{edge.asked}' resolves to '{resolved_name}', which "
                        "is not configured in this session."
                    ),
                    explain_missing_component(
                        edge.asked,
                        resolved_name,
                        expected_type=required_type,
                        session_type=session_type,
                        consumer=component,
                        active_names=active_names,
                    ),
                ))

            if is_wrap:
                _validate_wrapping_lifecycle(
                    component,
                    node if node is not None else registered_class,
                    session_scoped=session_scoped,
                )
            _add_execution_edge(
                component.id, target_id, edge,
                prerequisites_graph, edge_reasons,
            )

    if session_scoped:
        # The runtime takes every iteration hook's cadence modulo the
        # iteration, wrapping or reading or neither; validate them all.
        for component in selected_components:
            if isinstance(component, IterationHook):
                iteration_cadence(component)

    dependents_graph: dict[str, list[str]] = {
        component_id: [] for component_id in prerequisites_graph
    }
    for component_id, prerequisites in prerequisites_graph.items():
        for prerequisite in prerequisites:
            dependents_graph[prerequisite].append(component_id)

    queue = deque()
    prerequisite_count = {}
    for component_id, prerequisites in prerequisites_graph.items():
        prerequisite_count[component_id] = len(prerequisites)
        if prerequisite_count[component_id] == 0:
            queue.append(component_id)

    sorted_components = []
    while queue:
        front_node = queue.popleft()
        sorted_components.append(front_node)

        for dependent_id in dependents_graph[front_node]:
            prerequisite_count[dependent_id] -= 1
            if prerequisite_count[dependent_id] == 0:
                queue.append(dependent_id)

    if len(sorted_components) != len(prerequisites_graph):
        remaining = {
            component_id for component_id, count in prerequisite_count.items()
            if count > 0
        }
        raise RuntimeError(
            "Cyclic dependency detected in the component graph! "
            + _describe_cycle(remaining, prerequisites_graph, edge_reasons)
        )

    return {
        component_id: index
        for index, component_id in enumerate(sorted_components)
    }


def _add_execution_edge(
        source_id: str,
        target_id: str,
        edge,
        prerequisites_graph: dict[str, list[str]],
        edge_reasons: dict[tuple[str, str], str],
) -> None:
    """Record that `source` runs after (or, for a wrap, before) `target`."""
    if edge.execution == "after":
        prerequisites_graph[source_id].append(target_id)
        edge_reasons.setdefault((source_id, target_id), edge.reason)
    elif edge.execution == "before":
        prerequisites_graph[target_id].append(source_id)
        edge_reasons.setdefault((target_id, source_id), "wrapped by")


def _checked_writers(nodes_by_name, context_keys) -> dict[str, str]:
    """Key -> the one component that writes it.

    Two writers of one key are rejected: their order would be a guess, and a
    guess is what the resolution rules refuse to make.
    """
    writers: dict[str, str] = {}
    for name, (_, writes) in context_keys.items():
        for key in writes:
            other = writers.get(key)
            if other is not None and other != name:
                raise RuntimeError(
                    f"iteration_context key '{key}' is written by both "
                    f"{nodes_by_name[other].id} and {nodes_by_name[name].id}. "
                    "Each key has one writer, so the order of the two is "
                    "never guessed; have one of them write another key."
                )
            writers[key] = name
    return writers


def _add_read_edge(
        reader,
        edge,
        reader_writes,
        writers,
        nodes_by_name,
        prerequisites_graph,
        edge_reasons,
) -> None:
    """Order a step after the writer of a key it reads, after checking the
    key is there on every iteration the reader runs.

    A hook writes in its pre-iteration callback, before every step, and
    reads in its post-iteration callback, after every step, so hooks are
    checked but add no ordering edge.
    """
    key = edge.asked
    reader_is_step = isinstance(reader, Step)
    if reader_is_step and key in reader_writes:
        raise RuntimeError(
            f"{reader.id} reads and writes iteration_context key '{key}'. A "
            f"step cannot update a key in place: write a new key (e.g. "
            f"'{key}_updated') and read that instead."
        )
    writer_name = writers.get(key)
    if writer_name is None:
        available = ", ".join(sorted(writers)) or "none"
        raise RuntimeError(
            f"{reader.id} reads iteration_context key '{key}', which no step "
            f"or hook writes. Declare @writes('{key}') on the step that "
            "produces it, or configure a built-in step that does. Keys "
            f"written in this session: {available}."
        )
    writer = nodes_by_name[writer_name]
    # The context is cleared after every iteration, so a reader may only run
    # on iterations its writer runs on too -- whichever of the two is a step
    # or a hook.
    writer_cadence = iteration_cadence(writer)
    reader_cadence = iteration_cadence(reader)
    if reader_cadence % writer_cadence != 0:
        raise RuntimeError(
            f"{reader.id} reads iteration_context key '{key}', which "
            f"{writer.id} writes only every {writer_cadence} iterations, but "
            f"it runs every {reader_cadence}; on the others the key would be "
            "missing. A reader's cadence must be a multiple of its writer's: "
            f"give it a call_every that is a multiple of {writer_cadence}."
        )
    if isinstance(writer, Hook):
        return
    if reader_is_step:
        _add_execution_edge(
            reader.id, writer.id, edge.resolved(writer_name),
            prerequisites_graph, edge_reasons,
        )


def _check_companion(
        component,
        asked,
        resolved_name,
        node,
        registered_class,
        session_scoped,
        session_type,
        active_names,
        explain_missing_component,
        with_explanation,
) -> None:
    if registered_class is None:
        raise RuntimeError(with_explanation(
            f"'{component.name}' activates '{asked}', which resolves to "
            f"'{resolved_name}' and is not registered.",
            explain_missing_component(
                asked,
                resolved_name,
                session_type=session_type,
                consumer=component,
            ),
        ))
    if session_scoped and node is None:
        raise RuntimeError(with_explanation(
            f"'{component.name}' activates '{asked}', which resolves to "
            f"'{resolved_name}' and is not configured in this session. "
            "Activate it too, or build the session from configuration, "
            "which brings it along.",
            explain_missing_component(
                asked,
                resolved_name,
                session_type=session_type,
                consumer=component,
                active_names=active_names,
            ),
        ))


def _describe_cycle(
        remaining: set[str],
        prerequisites_graph: Mapping[str, list[str]],
        edge_reasons: Mapping[tuple[str, str], str],
) -> str:
    """Name one cycle among the components the sort could not place.

    Every one of them waits on at least one other that could not be placed,
    so following those waits from any of them must come back round.
    """
    current = min(remaining)
    path: list[str] = []
    seen: dict[str, int] = {}
    while current not in seen:
        seen[current] = len(path)
        path.append(current)
        current = next(
            prerequisite for prerequisite in prerequisites_graph[current]
            if prerequisite in remaining
        )
    cycle = path[seen[current]:] + [current]
    parts = [cycle[0]]
    for dependent, prerequisite in zip(cycle, cycle[1:]):
        reason = edge_reasons.get((dependent, prerequisite), "requires")
        # Read "A -> B (reads 'z')" as: A waits on B because A reads 'z'.
        parts.append(f"{prerequisite} ({reason})")
    return " -> ".join(parts)


def render_execution_graph(
        *,
        resources: Iterable[Resource],
        hooks: Iterable[Hook],
        steps: Iterable[Step],
        max_iterations: int,
        binding_resolver,
        session_type,
        order: Mapping[str, int],
) -> str:
    ordered_resources = sorted(
        resources,
        key=lambda component: order[component.id],
    )
    ordered_hooks = sorted(
        hooks,
        key=lambda component: order[component.id],
    )
    ordered_steps = sorted(
        steps,
        key=lambda component: order[component.id],
    )

    # Annotations resolve each dependency for the component that declared it,
    # to the instance it is actually given -- the same way the sort does --
    # so per-consumer wiring is shown as wired.
    nodes_by_name = {
        component.name: component
        for component in [*ordered_resources, *ordered_hooks, *ordered_steps]
    }

    session_hooks = [
        component
        for component in ordered_hooks
        if isinstance(component, SessionHook)
    ]
    iteration_hooks = [
        component
        for component in ordered_hooks
        if isinstance(component, IterationHook)
    ]

    title = f"{session_type.upper()} SESSION EXECUTION GRAPH"
    lines = [
        title,
        "================================",
        f"Max iterations: {max_iterations}",
    ]
    if binding_resolver:
        lines.extend(["", "COMPONENT BINDINGS"])
        lines.extend(
            f"  {role_name} -> {implementation_name}"
            for role_name, implementation_name
            in binding_resolver.bindings.items()
        )
        lines.extend(
            f"  {consumer}: {role_name} -> {target}"
            for consumer, wiring
            in binding_resolver.instance_bindings.items()
            for role_name, target in wiring.items()
        )
    dataflow = _dataflow_lines([*ordered_hooks, *ordered_steps])
    if dataflow:
        lines.extend(["", "DATAFLOW", *dataflow])
    lines.extend([
        "",
        "START",
        "  |",
        "  +-- SETUP",
    ])
    _append_execution_calls(
        lines,
        "  |   ",
        [(component, "setup") for component in ordered_resources]
        + [(component, "pre_session") for component in session_hooks],
        binding_resolver,
        nodes_by_name,
    )

    lines.extend([
        "  |",
        f"  +-- ITERATION (repeats 1..{max_iterations})",
        "  |   |",
        "  |   +-- PRE-ITERATION",
    ])
    _append_execution_calls(
        lines,
        "  |   |   ",
        [
            (component, "pre_iteration_callback")
            for component in iteration_hooks
        ],
        binding_resolver,
        nodes_by_name,
    )

    lines.extend([
        "  |   |",
        "  |   +-- STEPS",
    ])
    _append_execution_calls(
        lines,
        "  |   |   ",
        [(component, "run") for component in ordered_steps],
        binding_resolver,
        nodes_by_name,
    )

    lines.extend([
        "  |   |",
        "  |   +-- POST-ITERATION",
    ])
    _append_execution_calls(
        lines,
        "  |       ",
        [
            (component, "post_iteration_callback")
            for component in reversed(iteration_hooks)
        ],
        binding_resolver,
        nodes_by_name,
    )

    lines.extend([
        "  |",
        "  +-- TEARDOWN",
    ])
    _append_execution_calls(
        lines,
        "      ",
        [(component, "post_session") for component in reversed(session_hooks)]
        + [(component, "teardown") for component in reversed(ordered_resources)],
        binding_resolver,
        nodes_by_name,
    )
    lines.extend([
        "  |",
        "END",
    ])
    return "\n".join(lines)


def _append_execution_calls(
        lines,
        prefix,
        calls,
        binding_resolver,
        nodes_by_name,
) -> None:
    if not calls:
        lines.append(f"{prefix}(none)")
        return

    for index, (component, method_name) in enumerate(calls, start=1):
        annotations = []
        requirements = _component_requirements(
            component, binding_resolver, nodes_by_name,
        )
        if requirements:
            annotations.append(f"requires: {', '.join(requirements)}")
        wrapped_hooks = _component_wrapped_hooks(
            component, binding_resolver, nodes_by_name,
        )
        if wrapped_hooks:
            annotations.append(f"wraps: {', '.join(wrapped_hooks)}")
        companions = _component_companions(
            component, binding_resolver, nodes_by_name,
        )
        if companions:
            annotations.append(f"activates: {', '.join(companions)}")
        if method_name in {"run", "pre_iteration_callback"}:
            written = component.context_writes()
            if written:
                annotations.append(f"writes: {', '.join(written)}")
        if method_name in {"run", "post_iteration_callback"}:
            read = component.context_reads()
            if read:
                annotations.append(f"reads: {', '.join(read)}")
        if method_name in {
            "pre_iteration_callback",
            "post_iteration_callback",
        }:
            annotations.append(
                f"cadence: {_hook_cadence(component.call_every)}"
            )

        annotation = f" [{'; '.join(annotations)}]" if annotations else ""
        lines.append(
            f"{prefix}{index:02d}. {component.id}.{method_name}(){annotation}"
        )


def _dependency_display_name(binding_resolver, nodes_by_name, consumer, name):
    """Name the instance `consumer` is given for `name`, as the sort decides."""
    resolved_name, node = _resolve_to_node(
        binding_resolver,
        nodes_by_name,
        consumer,
        name,
    )
    return resolved_name if node is None else node.name


def _component_requirements(
        component,
        binding_resolver,
        nodes_by_name,
) -> list[str]:
    return [
        f"{edge.expected_type.__name__}."
        + _dependency_display_name(
            binding_resolver, nodes_by_name, component, edge.asked,
        )
        for edge in declared_edges(component)
        if edge.kind is EdgeKind.REQUIRES
    ]


def _component_wrapped_hooks(
        component,
        binding_resolver,
        nodes_by_name,
) -> list[str]:
    return [
        "Hook."
        + _dependency_display_name(
            binding_resolver, nodes_by_name, component, edge.asked,
        )
        for edge in declared_edges(component)
        if edge.kind is EdgeKind.WRAPS
    ]


def _dataflow_lines(components) -> list[str]:
    """`key: writer -> readers` for every declared iteration_context key."""
    writers: dict[str, str] = {}
    readers: dict[str, list[str]] = {}
    for component in components:
        for key in component.context_writes():
            writers[key] = component.id
        for key in component.context_reads():
            readers.setdefault(key, []).append(component.id)
    return [
        f"  {key}: {writers.get(key, '(no writer)')} -> "
        f"{', '.join(readers.get(key, [])) or '(not read)'}"
        for key in sorted(set(writers) | set(readers))
    ]


def _component_companions(
        component,
        binding_resolver,
        nodes_by_name,
) -> list[str]:
    names = []
    for edge in declared_edges(component):
        if edge.kind is not EdgeKind.COMPANION:
            continue
        resolved_name, node = _resolve_to_node(
            binding_resolver, nodes_by_name, component, edge.asked,
        )
        names.append(resolved_name if node is None else node.id)
    return names


def _hook_cadence(call_every: int) -> str:
    if call_every == 1:
        return "every iteration"
    return f"first, every {call_every}, final"
