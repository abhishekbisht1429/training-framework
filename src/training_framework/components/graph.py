from collections import deque
from collections.abc import Iterable, Mapping

from training_framework.components.base import (
    Component,
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Step,
)


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
) -> dict[str, int]:
    session_scoped = components is not None
    selected_components = (
        list(registry.values())
        if components is None
        else list(components)
    )

    components_by_id = {
        component.id: component
        for component in selected_components
    }
    prerequisites_graph: dict[str, list[str]] = {
        component.id: [] for component in selected_components
    }

    for component in selected_components:
        requirements = (
            ("required_hooks", Hook),
            ("required_steps", Step),
            ("required_resources", Resource),
        )
        for attribute, required_type in requirements:
            for required_name in getattr(component, attribute, []):
                resolved_name = binding_resolver.resolve(required_name)
                registered_class = registry.get(resolved_name)
                if (
                        registered_class is None
                        or not issubclass(registered_class, required_type)
                ):
                    raise RuntimeError(
                        f"unmet prerequisite! {required_type.__name__} "
                        f"'{required_name}' resolves to '{resolved_name}', which "
                        f"is not registered as a {required_type.__name__}."
                    )

                assert registered_class is not None
                prerequisite_id = registered_class.id
                prerequisites_graph[component.id].append(prerequisite_id)
                if session_scoped and prerequisite_id not in prerequisites_graph:
                    raise RuntimeError(
                        f"unmet prerequisite! {required_type.__name__} "
                        f"'{required_name}' resolves to '{resolved_name}', which "
                        "is not configured in this session."
                    )

    for wrapper in selected_components:
        if not _is_component_type(wrapper, Hook):
            continue

        resolved_targets = set()
        for wrapped_name in getattr(wrapper, "wrapped_hooks", ()):
            resolved_name = binding_resolver.resolve(wrapped_name)
            registered_class = registry.get(resolved_name)
            if registered_class is None or not issubclass(registered_class, Hook):
                raise RuntimeError(
                    f"invalid wraps target! Hook '{wrapped_name}' resolves to "
                    f"'{resolved_name}', which is not registered as a Hook."
                )

            wrapped_id = registered_class.id
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

            if session_scoped and wrapped_id not in prerequisites_graph:
                raise RuntimeError(
                    f"invalid wraps target! Hook '{wrapped_name}' resolves to "
                    f"'{resolved_name}', which is not configured in this session."
                )

            wrapped = components_by_id.get(wrapped_id, registered_class)
            _validate_wrapping_lifecycle(
                wrapper,
                wrapped,
                session_scoped=session_scoped,
            )
            prerequisites_graph[wrapped_id].append(wrapper.id)

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
    )
    lines.extend([
        "  |",
        "END",
    ])
    return "\n".join(lines)


def _append_execution_calls(lines, prefix, calls, binding_resolver) -> None:
    if not calls:
        lines.append(f"{prefix}(none)")
        return

    for index, (component, method_name) in enumerate(calls, start=1):
        annotations = []
        requirements = _component_requirements(component, binding_resolver)
        if requirements:
            annotations.append(f"requires: {', '.join(requirements)}")
        wrapped_hooks = _component_wrapped_hooks(component, binding_resolver)
        if wrapped_hooks:
            annotations.append(f"wraps: {', '.join(wrapped_hooks)}")
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


def _component_requirements(component, binding_resolver) -> list[str]:
    return (
        [
            f"Resource.{binding_resolver.resolve(name)}"
            for name in getattr(component, "required_resources", ())
        ]
        + [
            f"Hook.{binding_resolver.resolve(name)}"
            for name in getattr(component, "required_hooks", ())
        ]
        + [
            f"Step.{binding_resolver.resolve(name)}"
            for name in getattr(component, "required_steps", ())
        ]
    )


def _component_wrapped_hooks(component, binding_resolver) -> list[str]:
    return [
        f"Hook.{binding_resolver.resolve(name)}"
        for name in getattr(component, "wrapped_hooks", ())
    ]


def _hook_cadence(call_every: int) -> str:
    if call_every == 1:
        return "every iteration"
    return f"first, every {call_every}, final"
