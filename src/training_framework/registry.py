from collections import deque
from collections.abc import Iterable, Mapping

from training_framework.components import (
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Step,
)


def make_registry(component_type):
    registry = {}

    def register(name: str, overwrite=False):
        def wrapper(cls):
            if not issubclass(cls, component_type):
                raise TypeError(
                    f"{cls.__name__} must be subclass of {component_type.__name__}"
                )
            if overwrite == False and name in registry:
                raise ValueError(
                    f"{component_type} with name '{name}' already registered"
                )
            registry[name] = cls
            cls.name = name
            cls.id = f"{component_type.__name__}.{cls.name}"
            return cls

        return wrapper

    return registry, register


HOOK_REGISTRY, hook = make_registry(Hook)
RESOURCE_REGISTRY, resource = make_registry(Resource)
STEP_REGISTRY, step = make_registry(Step)

RESERVED_CONFIG_NAMES = frozenset({"aliases", "base_config"})


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
        registries = (STEP_REGISTRY, HOOK_REGISTRY, RESOURCE_REGISTRY)

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
                    "'aliases' and 'base_config' are reserved component names"
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

            matching_registries = [
                registry for registry in registries if actual_name in registry
            ]
            if not matching_registries:
                raise ValueError(
                    f"Alias target '{actual_name}' is not a registered component"
                )
            if len(matching_registries) > 1:
                raise ValueError(
                    f"Alias target '{actual_name}' is registered in multiple "
                    "component categories"
                )

            actual_registry = matching_registries[0]
            expected_registries = [
                registry for registry in registries if expected_name in registry
            ]
            if expected_registries and actual_registry not in expected_registries:
                raise ValueError(
                    f"Alias '{expected_name}' -> '{actual_name}' changes the "
                    "component category"
                )

            targets[actual_name] = expected_name

    def validate_config(self, config: Mapping) -> None:
        for expected_name, actual_name in self._aliases.items():
            if expected_name not in config:
                raise ValueError(
                    f"Alias '{expected_name}' requires a top-level "
                    f"'{expected_name}' component configuration"
                )
            if actual_name in config:
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
        if not issubclass(cls, (Step, Hook)):
            raise TypeError(
                f"@requires_hook can only be applied to Step or Hook subclasses. "
                f"'{cls.__name__}' is neither."
            )

        if "required_hooks" not in cls.__dict__:
            cls.required_hooks = []

        requirements: list[str] = cls.required_hooks
        requirements.append(hook_name)
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


def topological_sort_of_components(
        aliases: ComponentAliases | Mapping[str, str] | None = None,
        *,
        components: Iterable | None = None,
) -> dict[str, int]:
    alias_resolver = _alias_resolver(aliases)
    session_scoped = components is not None
    if components is None:
        components = (
                list(STEP_REGISTRY.values())
                + list(HOOK_REGISTRY.values())
                + list(RESOURCE_REGISTRY.values())
        )
    else:
        components = list(components)

    prerequisites_graph: dict[str, list[str]] = {
        component.id: [] for component in components
    }
    for component in components:
        for required_hook_name in getattr(component, "required_hooks", []):
            resolved_name = alias_resolver.resolve(required_hook_name)
            if resolved_name not in HOOK_REGISTRY:
                raise RuntimeError(
                    f"unmet prerequisite! Hook '{required_hook_name}' resolves "
                    f"to '{resolved_name}', which is not registered as a Hook."
                )
            prerequisites_graph[component.id].append(
                HOOK_REGISTRY[resolved_name].id
            )
            if (
                    session_scoped
                    and HOOK_REGISTRY[resolved_name].id
                    not in prerequisites_graph
            ):
                raise RuntimeError(
                    f"unmet prerequisite! Hook '{required_hook_name}' resolves "
                    f"to '{resolved_name}', which is not configured in this "
                    "session."
                )

        for required_step_name in getattr(component, "required_steps", []):
            resolved_name = alias_resolver.resolve(required_step_name)
            if resolved_name not in STEP_REGISTRY:
                raise RuntimeError(
                    f"unmet prerequisite! Step '{required_step_name}' resolves "
                    f"to '{resolved_name}', which is not registered as a Step."
                )
            prerequisites_graph[component.id].append(
                STEP_REGISTRY[resolved_name].id
            )
            if (
                    session_scoped
                    and STEP_REGISTRY[resolved_name].id
                    not in prerequisites_graph
            ):
                raise RuntimeError(
                    f"unmet prerequisite! Step '{required_step_name}' resolves "
                    f"to '{resolved_name}', which is not configured in this "
                    "session."
                )

        for required_resource_name in getattr(
                component, "required_resources", []
        ):
            resolved_name = alias_resolver.resolve(required_resource_name)
            if resolved_name not in RESOURCE_REGISTRY:
                raise RuntimeError(
                    f"unmet prerequisite! Resource '{required_resource_name}' "
                    f"resolves to '{resolved_name}', which is not registered as "
                    "a Resource."
                )
            prerequisites_graph[component.id].append(
                RESOURCE_REGISTRY[resolved_name].id
            )
            if (
                    session_scoped
                    and RESOURCE_REGISTRY[resolved_name].id
                    not in prerequisites_graph
            ):
                raise RuntimeError(
                    f"unmet prerequisite! Resource '{required_resource_name}' "
                    f"resolves to '{resolved_name}', which is not configured in "
                    "this session."
                )

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


def _hook_cadence(call_every: int) -> str:
    if call_every == 1:
        return "every iteration"
    return f"first, every {call_every}, final"
