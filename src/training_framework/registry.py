from collections import deque

from training_framework.components import Hook, Resource, Step


def make_registry(component_type):
    registry = {}

    def register(name: str):
        def wrapper(cls):
            if not issubclass(cls, component_type):
                raise TypeError(
                    f"{cls.__name__} must be subclass of {component_type.__name__}"
                )
            if name in registry:
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
