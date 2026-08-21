from training_framework.components import (
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Step,
)
from training_framework.registry import (
    HOOK_REGISTRY,
    RESOURCE_REGISTRY,
    STEP_REGISTRY,
    topological_sort_of_components,
)


class SessionComponents:
    def __init__(
            self,
            *,
            resources: dict[str, Resource] | None = None,
            steps: dict[str, Step] | None = None,
            hooks: dict[str, Hook] | None = None,
    ):
        self.resources = resources if resources is not None else {}
        self.steps = steps if steps is not None else {}
        self.hooks = hooks if hooks is not None else {}

    def register_from_config(self, config: dict) -> None:
        for name, component_config in config.items():
            if name == "base_config":
                continue
            if name in STEP_REGISTRY:
                self.add_step(STEP_REGISTRY[name](component_config), overwrite=True)
            elif name in HOOK_REGISTRY:
                self.register_hook(HOOK_REGISTRY[name](component_config), overwrite=True)
            elif name in RESOURCE_REGISTRY:
                self.register_resource(RESOURCE_REGISTRY[name](component_config), overwrite=True)
            else:
                raise ValueError(
                    f"No step, hook or resource registered with name '{name}'!"
                )

    def register_resource(self, component: Resource, overwrite=False) -> str:
        self._validate_component(
            component,
            Resource,
            RESOURCE_REGISTRY,
            self.resources,
            overwrite=overwrite
        )
        self.resources[component.name] = component
        return component.name

    def register_hook(self, component: Hook, overwrite=False) -> str:
        self._validate_component(
            component,
            Hook,
            HOOK_REGISTRY,
            self.hooks,
            overwrite=overwrite,
        )
        self.hooks[component.name] = component
        return component.name

    def add_step(self, component: Step, overwrite=False) -> str:
        self._validate_component(
            component,
            Step,
            STEP_REGISTRY,
            self.steps,
            overwrite=overwrite,
        )
        self.steps[component.name] = component
        return component.name

    @staticmethod
    def _validate_component(component, base_type, registry, collection, overwrite=False) -> None:
        if not isinstance(component, base_type):
            raise TypeError(
                f"The provided object '{type(component).__name__}' "
                f"is not an instance of {base_type.__name__}!"
            )

        if not hasattr(component, "name") or component.name not in registry:
            raise ValueError(
                f"{base_type.__name__} '{type(component).__name__}' "
                f"not registered in {base_type.__name__.upper()}_REGISTRY!"
            )

        if overwrite == False and component.name in collection:
            raise ValueError(
                f"{base_type.__name__} '{component.name}' already registered!"
            )

    def remove_step(self, name: str) -> None:
        self._remove_component(name, "Step", STEP_REGISTRY, self.steps)

    def unregister_hook(self, name: str) -> None:
        self._remove_component(name, "Hook", HOOK_REGISTRY, self.hooks)

    def unregister_resource(self, name: str) -> None:
        self._remove_component(
            name,
            "Resource",
            RESOURCE_REGISTRY,
            self.resources,
        )

    @staticmethod
    def _remove_component(name, kind, registry, collection) -> None:
        if name not in registry:
            raise ValueError(f"{kind} '{name}' not in {kind.upper()}_REGISTRY!")
        if name not in collection:
            if kind == "Step":
                raise ValueError(f"Step '{name}' is not added to this session!")
            raise ValueError(
                f"{kind} '{name}' not registered with current session!"
            )
        del collection[name]

    def get_resource(self, name: str) -> Resource:
        if name not in self.resources:
            raise KeyError(f"{name} not found in resources!")
        return self.resources[name]

    @property
    def ordered_hooks(self) -> list[Hook]:
        order = topological_sort_of_components()
        return sorted(self.hooks.values(), key=lambda component: order[component.id])

    @property
    def ordered_resources(self) -> list[Resource]:
        order = topological_sort_of_components()
        return sorted(
            self.resources.values(),
            key=lambda component: order[component.id],
        )

    @property
    def ordered_steps(self) -> list[Step]:
        order = topological_sort_of_components()
        return sorted(self.steps.values(), key=lambda component: order[component.id])

    @property
    def iteration_hooks(self) -> list[IterationHook]:
        return [
            component
            for component in self.ordered_hooks
            if isinstance(component, IterationHook)
        ]

    @property
    def session_hooks(self) -> list[SessionHook]:
        return [
            component
            for component in self.ordered_hooks
            if isinstance(component, SessionHook)
        ]
