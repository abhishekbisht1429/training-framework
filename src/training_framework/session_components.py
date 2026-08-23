from training_framework.components import (
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Step,
)
from training_framework.registry import (
    ComponentAliases,
    HOOK_REGISTRY,
    RESERVED_CONFIG_NAMES,
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
            aliases: dict[str, str] | None = None,
    ):
        self.resources = resources if resources is not None else {}
        self.steps = steps if steps is not None else {}
        self.hooks = hooks if hooks is not None else {}
        self.aliases = ComponentAliases(aliases)

    def register_from_config(self, config: dict) -> None:
        self.aliases.validate_config(config)
        for name, component_config in config.items():
            if name in RESERVED_CONFIG_NAMES:
                continue
            registered_name = self.resolve_name(name)
            if registered_name in STEP_REGISTRY:
                self.add_step(
                    STEP_REGISTRY[registered_name](component_config),
                    overwrite=True,
                )
            elif registered_name in HOOK_REGISTRY:
                self.register_hook(
                    HOOK_REGISTRY[registered_name](component_config),
                    overwrite=True,
                )
            elif registered_name in RESOURCE_REGISTRY:
                self.register_resource(
                    RESOURCE_REGISTRY[registered_name](component_config),
                    overwrite=True,
                )
            else:
                raise ValueError(
                    f"No step, hook or resource registered with name "
                    f"'{registered_name}'!"
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

    def _validate_component(
            self,
            component,
            base_type,
            registry,
            collection,
            overwrite=False,
    ) -> None:
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

        if self.aliases.is_alias(component.name):
            raise ValueError(
                f"Component role '{component.name}' is aliased to "
                f"'{self.resolve_name(component.name)}' in this session"
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

    def _remove_component(self, name, kind, registry, collection) -> None:
        registered_name = self.resolve_name(name)
        if registered_name not in registry:
            raise ValueError(
                f"{kind} '{name}' resolves to '{registered_name}', which is not "
                f"in {kind.upper()}_REGISTRY!"
            )
        if registered_name not in collection:
            if kind == "Step":
                raise ValueError(f"Step '{name}' is not added to this session!")
            raise ValueError(
                f"{kind} '{name}' not registered with current session!"
            )
        del collection[registered_name]

    def get_resource(self, name: str) -> Resource:
        registered_name = self.resolve_name(name)
        if registered_name not in self.resources:
            raise KeyError(f"{name} not found in resources!")
        return self.resources[registered_name]

    def has_resource(self, name: str) -> bool:
        return self.resolve_name(name) in self.resources

    def resolve_name(self, name: str) -> str:
        return self.aliases.resolve(name)

    @property
    def alias_bindings(self) -> dict[str, str]:
        return self.aliases.bindings

    @property
    def ordered_hooks(self) -> list[Hook]:
        order = topological_sort_of_components(self.aliases)
        return sorted(self.hooks.values(), key=lambda component: order[component.id])

    @property
    def ordered_resources(self) -> list[Resource]:
        order = topological_sort_of_components(self.aliases)
        return sorted(
            self.resources.values(),
            key=lambda component: order[component.id],
        )

    @property
    def ordered_steps(self) -> list[Step]:
        order = topological_sort_of_components(self.aliases)
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
