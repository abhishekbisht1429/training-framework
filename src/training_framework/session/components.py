from collections.abc import Iterable, Mapping
from typing import Any

from training_framework.components import (
    Component,
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Stateful,
    Step,
)
from training_framework.components.config import (
    RESERVED_CONFIG_NAMES,
    selected_component_names,
)
from training_framework.components.registry import (
    ComponentAliases,
    _component_type,
    component_registry,
    topological_sort_of_components,
)
from training_framework.session.config import TRAINING_SESSION_TYPE, normalize_session_type


class SessionComponents:
    def __init__(
            self,
            *,
            resources: dict[str, Resource] | None = None,
            steps: dict[str, Step] | None = None,
            hooks: dict[str, Hook] | None = None,
            aliases: dict[str, str] | None = None,
            session_type: str = TRAINING_SESSION_TYPE,
    ):
        self.session_type = normalize_session_type(session_type)
        self.registry = component_registry(self.session_type)
        self.components: dict[str, Component] = {}
        self._merge_components(resources, Resource)
        self._merge_components(hooks, Hook)
        self._merge_components(steps, Step)
        self.aliases = ComponentAliases(aliases, session_type=self.session_type)

    def get_state(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "component_type": _component_type(component).__name__,
                "state": (
                    component.get_state()
                    if isinstance(component, Stateful)
                    else None
                ),
                "init_args": getattr(component, "_init_args"),
            }
            for name, component in self.components.items()
        }

    def set_state(self, component_states: dict[str, dict[str, Any]]) -> None:
        restored_components: dict[str, Component] = {}

        for name, component_info in component_states.items():
            component_class = self.registry.get(name)
            if component_class is None:
                raise ValueError(
                    f"Checkpoint component '{name}' is not registered"
                )

            component_type = _component_type(component_class)
            stored_type = component_info["component_type"]
            if component_type.__name__ != stored_type:
                raise ValueError(
                    f"Checkpoint component '{name}' is stored as a "
                    f"{stored_type}, but is now registered as a "
                    f"{component_type.__name__}"
                )

            init_args = component_info["init_args"]
            component: Component = component_class(
                *init_args["args"],
                **init_args["kwargs"],
            )
            if isinstance(component, Stateful):
                component.set_state(component_info["state"])
            restored_components[name] = component

        self.components = restored_components

    def _merge_components(self, components, expected_type) -> None:
        for name, component in (components or {}).items():
            if not isinstance(component, expected_type):
                raise TypeError(
                    f"Restored component '{name}' is not a "
                    f"{expected_type.__name__}"
                )
            if name in self.components:
                raise ValueError(
                    f"Component name '{name}' appears in multiple checkpoint "
                    "categories and cannot be restored with the unified registry"
                )
            self.components[name] = component

    @property
    def resources(self) -> dict[str, Resource]:
        return {
            name: component
            for name, component in self.components.items()
            if isinstance(component, Resource)
        }

    @property
    def hooks(self) -> dict[str, Hook]:
        return {
            name: component
            for name, component in self.components.items()
            if isinstance(component, Hook)
        }

    @property
    def steps(self) -> dict[str, Step]:
        return {
            name: component
            for name, component in self.components.items()
            if isinstance(component, Step)
        }

    @staticmethod
    def _selected_component_names(config: Mapping) -> list[str]:
        return selected_component_names(config)

    @staticmethod
    def _dependency_specs(
            component_class: type[Component],
    ) -> Iterable[tuple[str, type[Component]]]:
        for name in getattr(component_class, "required_resources", ()):
            yield name, Resource
        for name in getattr(component_class, "required_hooks", ()):
            yield name, Hook
        for name in getattr(component_class, "required_steps", ()):
            yield name, Step
        if issubclass(component_class, Hook):
            for name in getattr(component_class, "wrapped_hooks", ()):
                yield name, Hook

    def _registered_component_class(
            self,
            name: str,
            expected_type: type[Component] | None = None,
    ) -> tuple[str, type[Component]]:
        resolved_name = self.resolve_name(name)
        component_class = self.registry.get(resolved_name)
        if component_class is None:
            if expected_type is None:
                raise ValueError(
                    f"No step, hook or resource registered with name "
                    f"'{resolved_name}'!"
                )
            raise RuntimeError(
                f"unmet prerequisite! {expected_type.__name__} '{name}' "
                f"resolves to '{resolved_name}', which is not registered as a "
                f"{expected_type.__name__}."
            )
        if expected_type is not None and not issubclass(
                component_class,
                expected_type,
        ):
            raise RuntimeError(
                f"unmet prerequisite! {expected_type.__name__} '{name}' "
                f"resolves to '{resolved_name}', which is not registered as a "
                f"{expected_type.__name__}."
            )
        return resolved_name, component_class

    def _register_component_instance(self, component: Component) -> None:
        component_type = _component_type(component)
        if component_type is Step:
            self.add_step(component, overwrite=True)
        elif component_type is Hook:
            self.register_hook(component, overwrite=True)
        else:
            self.register_resource(component, overwrite=True)

    def register_from_config(
            self,
            config: Mapping,
            *,
            default_configs: Mapping[str, Mapping] | None = None,
    ) -> None:
        selected_names = self._selected_component_names(config)
        self.aliases.validate_config(config)

        component_configs: dict[str, dict] = {}
        explicitly_configured: set[str] = set()
        configured_roots: list[str] = []
        for name, component_config in config.items():
            if name in RESERVED_CONFIG_NAMES:
                continue
            if not isinstance(component_config, Mapping):
                raise ValueError(
                    f"The value corresponding to the key '{name}' is not a mapping"
                )
            resolved_name = self.resolve_name(name)
            component_configs[resolved_name] = dict(component_config)
            explicitly_configured.add(resolved_name)
            configured_roots.append(name)

        roots: list[str] = []
        for name, default_config in (default_configs or {}).items():
            roots.append(name)
            resolved_name = self.resolve_name(name)
            if resolved_name not in component_configs:
                component_configs[resolved_name] = (
                    dict(default_config)
                    if resolved_name == name
                    else {}
                )

        for name in selected_names:
            roots.append(name)
            component_configs.setdefault(self.resolve_name(name), {})

        roots.extend(configured_roots)
        visiting: set[str] = set()

        def activate(name: str) -> None:
            resolved_name, component_class = self._registered_component_class(name)
            if resolved_name in self.components or resolved_name in visiting:
                return

            visiting.add(resolved_name)
            try:
                for dependency_name, dependency_type in self._dependency_specs(
                        component_class,
                ):
                    self._registered_component_class(
                        dependency_name,
                        dependency_type,
                    )
                    activate(dependency_name)

                component_config = component_configs.get(resolved_name, {})
                try:
                    component = component_class(component_config)
                except Exception as error:
                    if (
                            resolved_name not in explicitly_configured
                            and not component_config
                    ):
                        raise RuntimeError(
                            f"Failed to initialize auto-configured component "
                            f"'{resolved_name}' with an empty config. Add a "
                            f"top-level component mapping for its configuration."
                        ) from error
                    raise
                self._register_component_instance(component)
            finally:
                visiting.discard(resolved_name)

        for root in roots:
            activate(root)

    def dependency_closure(self, names: Iterable[str]) -> set[str]:
        closure: set[str] = set()

        def visit(name: str) -> None:
            resolved_name, component_class = self._registered_component_class(name)
            component = self.components.get(resolved_name)
            if component is None:
                raise RuntimeError(
                    f"Component '{name}' resolves to '{resolved_name}', which "
                    "is not configured in this session."
                )
            if resolved_name in closure:
                return

            closure.add(resolved_name)
            for dependency_name, dependency_type in self._dependency_specs(
                    component_class,
            ):
                self._registered_component_class(
                    dependency_name,
                    dependency_type,
                )
                visit(dependency_name)

        for name in names:
            visit(name)
        return closure

    def register_resource(self, component: Resource, overwrite=False) -> str:
        self._validate_component(
            component,
            Resource,
            overwrite=overwrite
        )
        self.components[component.name] = component
        return component.name

    def register_hook(self, component: Hook, overwrite=False) -> str:
        self._validate_component(
            component,
            Hook,
            overwrite=overwrite,
        )
        self.components[component.name] = component
        return component.name

    def add_step(self, component: Step, overwrite=False) -> str:
        self._validate_component(
            component,
            Step,
            overwrite=overwrite,
        )
        self.components[component.name] = component
        return component.name

    def _validate_component(
            self,
            component,
            base_type,
            overwrite=False,
    ) -> None:
        if not isinstance(component, base_type):
            raise TypeError(
                f"The provided object '{type(component).__name__}' "
                f"is not an instance of {base_type.__name__}!"
            )

        if (
                not hasattr(component, "name")
                or component.name not in self.registry
        ):
            raise ValueError(
                f"{base_type.__name__} '{type(component).__name__}' "
                "is not registered as a component!"
            )
        if _component_type(self.registry[component.name]) is not base_type:
            raise ValueError(
                f"Component '{component.name}' is not registered as a "
                f"{base_type.__name__}!"
            )

        if self.aliases.is_alias(component.name):
            raise ValueError(
                f"Component role '{component.name}' is aliased to "
                f"'{self.resolve_name(component.name)}' in this session"
            )

        existing = self.components.get(component.name)
        if existing is not None and not isinstance(existing, base_type):
            raise ValueError(
                f"Cannot replace {type(existing).__name__} '{component.name}' "
                f"with {base_type.__name__} '{type(component).__name__}'"
            )
        if overwrite == False and existing is not None:
            raise ValueError(
                f"{base_type.__name__} '{component.name}' already registered!"
            )

    def remove_step(self, name: str) -> None:
        self._remove_component(name, Step)

    def unregister_hook(self, name: str) -> None:
        self._remove_component(name, Hook)

    def unregister_resource(self, name: str) -> None:
        self._remove_component(name, Resource)

    def _remove_component(self, name, component_type) -> None:
        kind = component_type.__name__
        registered_name = self.resolve_name(name)
        registered_class = self.registry.get(registered_name)
        if (
                registered_class is None
                or not issubclass(registered_class, component_type)
        ):
            raise ValueError(
                f"{kind} '{name}' resolves to '{registered_name}', which is not "
                f"registered as a {kind}!"
            )
        component = self.components.get(registered_name)
        if component is None or not isinstance(component, component_type):
            if kind == "Step":
                raise ValueError(f"Step '{name}' is not added to this session!")
            raise ValueError(
                f"{kind} '{name}' not registered with current session!"
            )
        del self.components[registered_name]

    def get_resource(self, name: str) -> Resource:
        registered_name = self.resolve_name(name)
        component = self.components.get(registered_name)
        if not isinstance(component, Resource):
            raise KeyError(f"{name} not found in resources!")
        return component

    def has_resource(self, name: str) -> bool:
        component = self.components.get(self.resolve_name(name))
        return isinstance(component, Resource)

    def resolve_name(self, name: str) -> str:
        return self.aliases.resolve(name)

    @property
    def alias_bindings(self) -> dict[str, str]:
        return self.aliases.bindings

    def _component_order(self) -> dict[str, int]:
        return topological_sort_of_components(
            self.aliases,
            components=self.components.values(),
            session_type=self.session_type,
        )

    @property
    def ordered_hooks(self) -> list[Hook]:
        order = self._component_order()
        return sorted(self.hooks.values(), key=lambda component: order[component.id])

    @property
    def ordered_resources(self) -> list[Resource]:
        order = self._component_order()
        return sorted(
            self.resources.values(),
            key=lambda component: order[component.id],
        )

    @property
    def ordered_steps(self) -> list[Step]:
        order = self._component_order()
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
