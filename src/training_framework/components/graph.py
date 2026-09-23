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
from training_framework.components.naming import (
    implementation_of,
    is_instance_name,
)

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

    Mirrors the session's resolution: the consumer's own wiring first, then
    the sole node the name can mean. An exact match wins outright, so a
    second instance of a component never takes an edge away from the first.
    Ambiguity is left to the session, which can explain it properly; here it
    simply resolves to nothing and is reported as unconfigured.
    """
    resolved_name = binding_resolver.resolve(
        name,
        consumer=getattr(consumer, "name", None),
    )
    if resolved_name in nodes_by_name:
        return resolved_name, nodes_by_name[resolved_name]
    if is_instance_name(resolved_name):
        # A precise reference to an instance that is not here; never a sibling.
        return resolved_name, None

    implementation = implementation_of(resolved_name)
    candidates = [
        node for node_name, node in nodes_by_name.items()
        if implementation_of(node_name) == implementation
    ]
    if len(candidates) == 1:
        return resolved_name, candidates[0]
    return resolved_name, None


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
        isinstance(wrapper_cadence, int)
        and not isinstance(wrapper_cadence, bool)
        and wrapper_cadence > 0
        and isinstance(wrapped_cadence, int)
        and not isinstance(wrapped_cadence, bool)
        and wrapped_cadence > 0
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

    for component in selected_components:
        requirements = (
            ("required_hooks", Hook),
            ("required_steps", Step),
            ("required_resources", Resource),
        )
        for attribute, required_type in requirements:
            for required_name in getattr(component, attribute, []):
                resolved_name, prerequisite = _resolve_to_node(
                    binding_resolver,
                    nodes_by_name,
                    component,
                    required_name,
                )
                # A name may identify an instance; the class it is an
                # instance of is what the registry holds.
                registered_class = registry.get(
                    implementation_of(resolved_name),
                )
                if (
                        registered_class is None
                        or not issubclass(registered_class, required_type)
                ):
                    declared_role = (
                        roles.get(resolved_name)
                        if registered_class is None
                        else None
                    )
                    if (
                            declared_role is not None
                            and declared_role.category is required_type
                    ):
                        raise RuntimeError(
                            _missing_role_message(
                                category=required_type,
                                name=required_name,
                                resolved_name=resolved_name,
                                declared_role=declared_role,
                                consumer=component,
                            )
                        )
                    raise RuntimeError(with_explanation(
                        f"unmet prerequisite! {required_type.__name__} "
                        f"'{required_name}' resolves to '{resolved_name}', which "
                        f"is not registered as a {required_type.__name__}.",
                        explain_missing_component(
                            required_name,
                            resolved_name,
                            expected_type=required_type,
                            session_type=session_type,
                            consumer=component,
                        ),
                    ))

                assert registered_class is not None
                if session_scoped and prerequisite is None:
                    raise RuntimeError(with_explanation(
                        f"unmet prerequisite! {required_type.__name__} "
                        f"'{required_name}' resolves to '{resolved_name}', which "
                        "is not configured in this session.",
                        explain_missing_component(
                            required_name,
                            resolved_name,
                            expected_type=required_type,
                            session_type=session_type,
                            consumer=component,
                            active_names=active_names,
                        ),
                    ))
                prerequisite_id = (
                    prerequisite.id
                    if prerequisite is not None
                    else registered_class.id
                )
                prerequisites_graph[component.id].append(prerequisite_id)

    for wrapper in selected_components:
        if not _is_component_type(wrapper, Hook):
            continue

        resolved_targets = set()
        for wrapped_name in getattr(wrapper, "wrapped_hooks", ()):
            resolved_name, wrapped_node = _resolve_to_node(
                binding_resolver,
                nodes_by_name,
                wrapper,
                wrapped_name,
            )
            registered_class = registry.get(
                implementation_of(resolved_name),
            )
            if registered_class is None or not issubclass(registered_class, Hook):
                declared_role = (
                    roles.get(resolved_name) if registered_class is None else None
                )
                if declared_role is not None and declared_role.category is Hook:
                    raise RuntimeError(
                        _missing_role_message(
                            category=Hook,
                            name=wrapped_name,
                            resolved_name=resolved_name,
                            declared_role=declared_role,
                            consumer=wrapper,
                        )
                    )
                raise RuntimeError(with_explanation(
                    f"invalid wraps target! Hook '{wrapped_name}' resolves to "
                    f"'{resolved_name}', which is not registered as a Hook.",
                    explain_missing_component(
                        wrapped_name,
                        resolved_name,
                        expected_type=Hook,
                        session_type=session_type,
                        consumer=wrapper,
                    ),
                ))

            wrapped_id = (
                wrapped_node.id
                if wrapped_node is not None
                else registered_class.id
            )
            if wrapped_id == wrapper.id:
                raise RuntimeError(
                    f"Hook '{wrapper.name}' cannot wrap itself"
                )
            if wrapped_id in resolved_targets:
                raise RuntimeError(
                    f"Hook '{wrapper.name}' wraps Hook '{resolved_name}' "
                        "more than once after component binding resolution"
                )
            resolved_targets.add(wrapped_id)

            if session_scoped and wrapped_node is None:
                raise RuntimeError(with_explanation(
                    f"invalid wraps target! Hook '{wrapped_name}' resolves to "
                    f"'{resolved_name}', which is not configured in this session.",
                    explain_missing_component(
                        wrapped_name,
                        resolved_name,
                        expected_type=Hook,
                        session_type=session_type,
                        consumer=wrapper,
                        active_names=active_names,
                    ),
                ))

            wrapped = (
                wrapped_node if wrapped_node is not None else registered_class
            )
            _validate_wrapping_lifecycle(
                wrapper,
                wrapped,
                session_scoped=session_scoped,
            )
            prerequisites_graph[wrapped_id].append(wrapper.id)

    # Companions (`@activates`) add no edge: they are not ordered relative to
    # the component that brings them along. They are checked here because
    # this is where every way a session comes to hold its components --
    # activation, hand registration, restore -- converges.
    for component in selected_components:
        for companion_name in getattr(component, "activated_components", ()):
            resolved_name, companion_node = _resolve_to_node(
                binding_resolver,
                nodes_by_name,
                component,
                companion_name,
            )
            if registry.get(implementation_of(resolved_name)) is None:
                raise RuntimeError(with_explanation(
                    f"'{component.name}' activates '{companion_name}', which "
                    f"resolves to '{resolved_name}' and is not registered.",
                    explain_missing_component(
                        companion_name,
                        resolved_name,
                        session_type=session_type,
                        consumer=component,
                    ),
                ))
            if session_scoped and companion_node is None:
                raise RuntimeError(with_explanation(
                    f"'{component.name}' activates '{companion_name}', which "
                    f"resolves to '{resolved_name}' and is not configured in "
                    "this session. Activate it too, or build the session from "
                    "configuration, which brings it along.",
                    explain_missing_component(
                        companion_name,
                        resolved_name,
                        session_type=session_type,
                        consumer=component,
                        active_names=active_names,
                    ),
                ))

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
        raise RuntimeError("Cyclic dependency detected in the component graph!")

    return {
        component_id: index
        for index, component_id in enumerate(sorted_components)
    }


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
        f"{category}."
        + _dependency_display_name(
            binding_resolver,
            nodes_by_name,
            component,
            name,
        )
        for attribute, category in (
            ("required_resources", "Resource"),
            ("required_hooks", "Hook"),
            ("required_steps", "Step"),
        )
        for name in getattr(component, attribute, ())
    ]


def _component_wrapped_hooks(
        component,
        binding_resolver,
        nodes_by_name,
) -> list[str]:
    return [
        "Hook."
        + _dependency_display_name(
            binding_resolver,
            nodes_by_name,
            component,
            name,
        )
        for name in getattr(component, "wrapped_hooks", ())
    ]


def _component_companions(
        component,
        binding_resolver,
        nodes_by_name,
) -> list[str]:
    names = []
    for name in getattr(component, "activated_components", ()):
        _, node = _resolve_to_node(
            binding_resolver, nodes_by_name, component, name,
        )
        display = _dependency_display_name(
            binding_resolver, nodes_by_name, component, name,
        )
        names.append(display if node is None else node.id)
    return names


def _hook_cadence(call_every: int) -> str:
    if call_every == 1:
        return "every iteration"
    return f"first, every {call_every}, final"
