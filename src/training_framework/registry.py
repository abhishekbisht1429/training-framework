from collections import deque
from collections.abc import Iterable

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


def topological_sort_of_components() -> dict[str, int]:
    components = (
            list(STEP_REGISTRY.values())
            + list(HOOK_REGISTRY.values())
            + list(RESOURCE_REGISTRY.values())
    )

    prerequisites_graph: dict[str, list[str]] = {
        component.id: [] for component in components
    }
    for component in components:
        for required_hook_name in getattr(component, "required_hooks", []):
            if required_hook_name not in HOOK_REGISTRY:
                raise RuntimeError(
                    f"unmet prerequisite! '{required_hook_name}' not registered."
                )
            prerequisites_graph[component.id].append(
                HOOK_REGISTRY[required_hook_name].id
            )

        for required_step_name in getattr(component, "required_steps", []):
            if required_step_name not in STEP_REGISTRY:
                raise RuntimeError(
                    f"unmet prerequisite! '{required_step_name}' not registered."
                )
            prerequisites_graph[component.id].append(
                STEP_REGISTRY[required_step_name].id
            )

        for required_resource_name in getattr(
                component, "required_resources", []
        ):
            if required_resource_name not in RESOURCE_REGISTRY:
                raise RuntimeError(
                    f"unmet prerequisite! '{required_resource_name}' not registered."
                )
            prerequisites_graph[component.id].append(
                RESOURCE_REGISTRY[required_resource_name].id
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
) -> str:
    """Return the session's component lifecycle as a readable execution graph."""
    order = topological_sort_of_components()
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
        "",
        "START",
        "  |",
        "  +-- SETUP",
    ]
    _append_execution_calls(
        lines,
        "  |   ",
        [(component, "setup") for component in ordered_resources]
        + [(component, "setup") for component in session_hooks],
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
    )

    lines.extend([
        "  |   |",
        "  |   +-- STEPS",
    ])
    _append_execution_calls(
        lines,
        "  |   |   ",
        [(component, "run") for component in ordered_steps],
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
    )
    lines.extend([
        "  |",
        "END",
    ])
    return "\n".join(lines)


def _append_execution_calls(lines, prefix, calls) -> None:
    if not calls:
        lines.append(f"{prefix}(none)")
        return

    for index, (component, method_name) in enumerate(calls, start=1):
        annotations = []
        requirements = _component_requirements(component)
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


def _component_requirements(component) -> list[str]:
    return (
        [
            f"Resource.{name}"
            for name in getattr(component, "required_resources", ())
        ]
        + [
            f"Hook.{name}"
            for name in getattr(component, "required_hooks", ())
        ]
        + [
            f"Step.{name}"
            for name in getattr(component, "required_steps", ())
        ]
    )


def _hook_cadence(call_every: int) -> str:
    if call_every == 1:
        return "every iteration"
    return f"first, every {call_every}, final"
