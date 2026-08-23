from collections import deque
from collections.abc import Iterable, Mapping

from training_framework.components import (
    Component,
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Step,
)


_COMPONENT_TYPES = (Resource, Hook, Step)
_COMPONENT_REGISTRY: dict[str, type[Component]] = {}


def _component_type(component: Component | type[Component]) -> type[Component]:
    component_class = component if isinstance(component, type) else type(component)
    matching_types = [
        component_type
        for component_type in _COMPONENT_TYPES
        if issubclass(component_class, component_type)
    ]
    if len(matching_types) != 1:
        categories = ", ".join(
            component_type.__name__ for component_type in matching_types
        ) or "none"
        raise TypeError(
            f"{component_class.__name__} must inherit exactly one component "
            f"category (Resource, Hook, or Step); found {categories}"
        )
    return matching_types[0]


def _component(name: str, *, expected_type=None, overwrite=False):
    def wrapper(cls):
        if not isinstance(cls, type) or not issubclass(cls, Component):
            expected_name = (
                expected_type.__name__ if expected_type is not None else "Component"
            )
            raise TypeError(
                f"{getattr(cls, '__name__', type(cls).__name__)} must be "
                f"subclass of {expected_name}"
            )
        if expected_type is not None and not issubclass(cls, expected_type):
            raise TypeError(
                f"{cls.__name__} must be subclass of {expected_type.__name__}"
            )

        registered_type = _component_type(cls)
        if name in _COMPONENT_REGISTRY:
            existing_type = _component_type(_COMPONENT_REGISTRY[name])
            if not overwrite:
                raise ValueError(f"Component with name '{name}' already registered")
            if existing_type is not registered_type:
                raise ValueError(
                    f"Cannot overwrite {existing_type.__name__} '{name}' with "
                    f"{registered_type.__name__} '{cls.__name__}'"
                )

        _COMPONENT_REGISTRY[name] = cls
        cls.name = name
        cls.id = f"{registered_type.__name__}.{name}"
        return cls

    return wrapper


def hook(name: str, overwrite=False):
    return _component(name, expected_type=Hook, overwrite=overwrite)


def resource(name: str, overwrite=False):
    return _component(name, expected_type=Resource, overwrite=overwrite)


def step(name: str, overwrite=False):
    return _component(name, expected_type=Step, overwrite=overwrite)

RESERVED_CONFIG_NAMES = frozenset({"aliases", "base_config", "components"})


class ComponentAliases:
    """Resolve session-scoped component roles to registered implementations."""

    def __init__(self, aliases: Mapping[str, str] | None = None):
        if aliases is None:
            aliases = {}
        if not isinstance(aliases, Mapping):
            raise TypeError("'aliases' must be a mapping of strings to strings")

        self._aliases = dict(aliases)
        self._validate()

    def _validate(self) -> None:
        targets = {}
        for expected_name, actual_name in self._aliases.items():
            if not isinstance(expected_name, str) or not isinstance(actual_name, str):
                raise TypeError("'aliases' must be a mapping of strings to strings")
            if not expected_name or not actual_name:
                raise ValueError("Alias names must not be empty")
            if (
                    expected_name in RESERVED_CONFIG_NAMES
                    or actual_name in RESERVED_CONFIG_NAMES
            ):
                raise ValueError(
                    "'aliases', 'base_config', and 'components' are reserved component names"
                )
            if expected_name == actual_name:
                raise ValueError(
                    f"Alias '{expected_name}' must refer to a different component"
                )
            if actual_name in self._aliases:
                raise ValueError(
                    f"Alias chains and cycles are not supported: "
                    f"'{expected_name}' resolves to alias '{actual_name}'"
                )
            if actual_name in targets:
                raise ValueError(
                    f"Aliases '{targets[actual_name]}' and '{expected_name}' "
                    f"cannot both target '{actual_name}'"
                )

            if actual_name not in _COMPONENT_REGISTRY:
                raise ValueError(
                    f"Alias target '{actual_name}' is not a registered component"
                )
            actual_type = _component_type(_COMPONENT_REGISTRY[actual_name])
            if (
                    expected_name in _COMPONENT_REGISTRY
                    and _component_type(_COMPONENT_REGISTRY[expected_name])
                    is not actual_type
            ):
                raise ValueError(
                    f"Alias '{expected_name}' -> '{actual_name}' changes the "
                    "component category"
                )

            targets[actual_name] = expected_name

    def validate_config(self, config: Mapping) -> None:
        selected_components = config.get("components", ())
        for expected_name, actual_name in self._aliases.items():
            if actual_name in config or actual_name in selected_components:
                raise ValueError(
                    f"Configure alias '{expected_name}', not both "
                    f"'{expected_name}' and '{actual_name}'"
                )

    def resolve(self, name: str) -> str:
        return self._aliases.get(name, name)

    def is_alias(self, name: str) -> bool:
        return name in self._aliases

    @property
    def bindings(self) -> dict[str, str]:
        return dict(self._aliases)

    def __bool__(self) -> bool:
        return bool(self._aliases)


def _alias_resolver(
        aliases: ComponentAliases | Mapping[str, str] | None,
) -> ComponentAliases:
    if isinstance(aliases, ComponentAliases):
        return aliases
    return ComponentAliases(aliases)


def requires_step(step_name: str):
    def wrapper(cls):
        if not issubclass(cls, Step):
            raise TypeError(
                f"@requires_step can only be applied to Step subclasses. "
                f"'{cls.__name__}' is not a Step."
            )

        if "required_steps" not in cls.__dict__:
            cls.required_steps = []

        requirements: list[str] = cls.required_steps
        requirements.append(step_name)
        return cls

    return wrapper


def requires_hook(hook_name: str):
    def wrapper(cls):
        if not issubclass(cls, Step):
            raise TypeError(
                f"@requires_hook can only be applied to Step subclasses. "
                f"'{cls.__name__}' is not a Step."
            )

        if "required_hooks" not in cls.__dict__:
            cls.required_hooks = list(getattr(cls, "required_hooks", ()))

        requirements: list[str] = cls.required_hooks
        requirements.append(hook_name)
        return cls

    return wrapper


def wraps(hook_name: str):
    def wrapper(cls):
        if not issubclass(cls, Hook):
            raise TypeError(
                f"@wraps can only be applied to Hook subclasses. "
                f"'{cls.__name__}' is not a Hook."
            )

        if "wrapped_hooks" not in cls.__dict__:
            cls.wrapped_hooks = list(getattr(cls, "wrapped_hooks", ()))

        wrapped_hooks: list[str] = cls.wrapped_hooks
        if hook_name in wrapped_hooks:
            raise ValueError(
                f"Hook '{cls.__name__}' already wraps '{hook_name}'"
            )
        wrapped_hooks.append(hook_name)
        return cls

    return wrapper


def requires_resource(resource_name: str):
    def wrapper(cls):
        if not issubclass(cls, (Step, Hook, Resource)):
            raise TypeError(
                "@requires_resource can only be applied to Step, Hook, "
                f"or Resource subclasses. '{cls.__name__}' is neither."
            )

        if "required_resources" not in cls.__dict__:
            cls.required_resources = list(
                getattr(cls, "required_resources", ())
            )

        cls.required_resources.append(resource_name)
        return cls

    return wrapper


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
    if (
            wrapper_cadence is missing
            or wrapped_cadence is missing
            or wrapper_cadence != wrapped_cadence
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
            "must have matching call_every values; "
            f"got {wrapper_value} and {wrapped_value}"
        )


def topological_sort_of_components(
        aliases: ComponentAliases | Mapping[str, str] | None = None,
        *,
        components: Iterable | None = None,
) -> dict[str, int]:
    alias_resolver = _alias_resolver(aliases)
    session_scoped = components is not None
    if components is None:
        components = list(_COMPONENT_REGISTRY.values())
    else:
        components = list(components)

    components_by_id = {
        component.id: component
        for component in components
    }
    prerequisites_graph: dict[str, list[str]] = {
        component.id: [] for component in components
    }

    for component in components:
        requirements = (
            ("required_hooks", Hook),
            ("required_steps", Step),
            ("required_resources", Resource),
        )
        for attribute, required_type in requirements:
            for required_name in getattr(component, attribute, []):
                resolved_name = alias_resolver.resolve(required_name)
                registered_class = _COMPONENT_REGISTRY.get(resolved_name)
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

    for wrapper in components:
        if not _is_component_type(wrapper, Hook):
            continue

        resolved_targets = set()
        for wrapped_name in getattr(wrapper, "wrapped_hooks", ()):
            resolved_name = alias_resolver.resolve(wrapped_name)
            registered_class = _COMPONENT_REGISTRY.get(resolved_name)
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
                    "more than once after alias resolution"
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
            # The wrapper must enter before the wrapped Hook. Reversing the
            # ordered hooks for post callbacks and teardown closes it afterward.
            prerequisites_graph[wrapped_id].append(wrapper.id)

    dependents_graph: dict[str, list[str]] = {
        component_id: [] for component_id in prerequisites_graph
    }
    for component_id, prerequisites in prerequisites_graph.items():
        for prerequisite in prerequisites:
            dependents_graph[prerequisite].append(component_id)

    queue = deque()
    prereq_count = {}
    for component_id, prerequisites in prerequisites_graph.items():
        prereq_count[component_id] = len(prerequisites)
        if prereq_count[component_id] == 0:
            queue.append(component_id)

    sorted_components = []
    while queue:
        front_node = queue.popleft()
        sorted_components.append(front_node)

        for dependent_id in dependents_graph[front_node]:
            prereq_count[dependent_id] -= 1
            if prereq_count[dependent_id] == 0:
                queue.append(dependent_id)

    if len(sorted_components) != len(prerequisites_graph):
        raise RuntimeError("Cyclic dependency detected in the component graph!")

    return {
        component_id: index
        for index, component_id in enumerate(sorted_components)
    }

def format_execution_graph(
        *,
        resources: Iterable[Resource],
        hooks: Iterable[Hook],
        steps: Iterable[Step],
        max_iterations: int,
        aliases: ComponentAliases | Mapping[str, str] | None = None,
) -> str:
    """Return the session's component lifecycle as a readable execution graph."""
    alias_resolver = _alias_resolver(aliases)
    resources = list(resources)
    hooks = list(hooks)
    steps = list(steps)
    order = topological_sort_of_components(
        alias_resolver,
        components=resources + hooks + steps,
    )
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

    lines = [
        "TRAINING SESSION EXECUTION GRAPH",
        "================================",
        f"Max iterations: {max_iterations}",
    ]
    if alias_resolver:
        lines.extend(["", "ALIASES"])
        lines.extend(
            f"  {expected_name} -> {actual_name}"
            for expected_name, actual_name in alias_resolver.bindings.items()
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
        + [(component, "setup") for component in session_hooks],
        alias_resolver,
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
        alias_resolver,
    )

    lines.extend([
        "  |   |",
        "  |   +-- STEPS",
    ])
    _append_execution_calls(
        lines,
        "  |   |   ",
        [(component, "run") for component in ordered_steps],
        alias_resolver,
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
        alias_resolver,
    )

    lines.extend([
        "  |",
        "  +-- TEARDOWN",
    ])
    _append_execution_calls(
        lines,
        "      ",
        [(component, "teardown") for component in reversed(session_hooks)]
        + [(component, "teardown") for component in reversed(ordered_resources)],
        alias_resolver,
    )
    lines.extend([
        "  |",
        "END",
    ])
    return "\n".join(lines)


def _append_execution_calls(lines, prefix, calls, alias_resolver) -> None:
    if not calls:
        lines.append(f"{prefix}(none)")
        return

    for index, (component, method_name) in enumerate(calls, start=1):
        annotations = []
        requirements = _component_requirements(component, alias_resolver)
        if requirements:
            annotations.append(f"requires: {', '.join(requirements)}")
        wrapped_hooks = _component_wrapped_hooks(component, alias_resolver)
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


def _component_requirements(
        component,
        aliases: ComponentAliases | Mapping[str, str] | None = None,
) -> list[str]:
    alias_resolver = _alias_resolver(aliases)
    return (
        [
            f"Resource.{alias_resolver.resolve(name)}"
            for name in getattr(component, "required_resources", ())
        ]
        + [
            f"Hook.{alias_resolver.resolve(name)}"
            for name in getattr(component, "required_hooks", ())
        ]
        + [
            f"Step.{alias_resolver.resolve(name)}"
            for name in getattr(component, "required_steps", ())
        ]
    )


def _component_wrapped_hooks(
        component,
        aliases: ComponentAliases | Mapping[str, str] | None = None,
) -> list[str]:
    alias_resolver = _alias_resolver(aliases)
    return [
        f"Hook.{alias_resolver.resolve(name)}"
        for name in getattr(component, "wrapped_hooks", ())
    ]


def _hook_cadence(call_every: int) -> str:
    if call_every == 1:
        return "every iteration"
    return f"first, every {call_every}, final"
