from collections import ChainMap
from collections.abc import Iterable, Mapping

from training_framework.components.base import Component, Hook, Resource, Step
from training_framework.components.config import reserved_config_names
from training_framework.components.graph import (
    render_execution_graph,
    topological_sort_components,
)


_COMPONENT_TYPES = (Resource, Hook, Step)
TRAINING_SESSION_TYPE = "training"
ANALYSIS_SESSION_TYPE = "analysis"
_SHARED_COMPONENT_REGISTRY: dict[str, type[Component]] = {}
_SESSION_COMPONENT_REGISTRIES: dict[str, dict[str, type[Component]]] = {}
_COMPONENT_REGISTRY = _SHARED_COMPONENT_REGISTRY
_ANALYSIS_COMPONENT_REGISTRY = _SESSION_COMPONENT_REGISTRIES.setdefault(
    ANALYSIS_SESSION_TYPE,
    {},
)


def _normalize_component_session_type(session_type: str | None) -> str | None:
    if session_type is None:
        return None
    if not isinstance(session_type, str):
        raise TypeError("session_type must be a string or None")
    normalized = session_type.strip()
    if not normalized:
        raise ValueError("session_type must not be empty")
    return normalized


def _registration_registry(
        session_type: str | None,
) -> dict[str, type[Component]]:
    normalized = _normalize_component_session_type(session_type)
    if normalized is None:
        return _SHARED_COMPONENT_REGISTRY
    return _SESSION_COMPONENT_REGISTRIES.setdefault(normalized, {})


def component_registry(
        session_type: str | None = None,
) -> Mapping[str, type[Component]]:
    normalized = _normalize_component_session_type(session_type)
    if normalized is None:
        return _SHARED_COMPONENT_REGISTRY
    scoped = _SESSION_COMPONENT_REGISTRIES.setdefault(normalized, {})
    return ChainMap(scoped, _SHARED_COMPONENT_REGISTRY)


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


def _component(
        name: str,
        *,
        expected_type=None,
        overwrite=False,
        session_type: str | None = None,
):
    registry = _registration_registry(session_type)

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
        if name in registry:
            existing_type = _component_type(registry[name])
            if not overwrite:
                scope = session_type or "shared"
                raise ValueError(
                    f"Component with name '{name}' already registered in "
                    f"'{scope}' scope"
                )
            if existing_type is not registered_type:
                raise ValueError(
                    f"Cannot overwrite {existing_type.__name__} '{name}' with "
                    f"{registered_type.__name__} '{cls.__name__}'"
                )

        registry[name] = cls
        cls.name = name
        cls.id = f"{registered_type.__name__}.{name}"
        return cls

    return wrapper


def hook(
        name: str,
        overwrite=False,
        *,
        session_type: str | None = None,
):
    return _component(
        name,
        expected_type=Hook,
        overwrite=overwrite,
        session_type=session_type,
    )


def resource(
        name: str,
        overwrite=False,
        *,
        session_type: str | None = None,
):
    return _component(
        name,
        expected_type=Resource,
        overwrite=overwrite,
        session_type=session_type,
    )


def step(
        name: str,
        overwrite=False,
        *,
        session_type: str | None = None,
):
    return _component(
        name,
        expected_type=Step,
        overwrite=overwrite,
        session_type=session_type,
    )


class ComponentAliases:
    """Resolve session-scoped component roles to registered implementations."""

    def __init__(
            self,
            aliases: Mapping[str, str] | None = None,
            *,
            session_type: str | None = None,
    ):
        if aliases is None:
            aliases = {}
        if not isinstance(aliases, Mapping):
            raise TypeError("'aliases' must be a mapping of strings to strings")

        normalized = _normalize_component_session_type(session_type)
        self._session_type = normalized
        self._registry = component_registry(normalized)
        self._aliases = dict(aliases)
        self._validate()

    def _validate(self) -> None:
        targets = {}
        reserved_names = reserved_config_names(self._session_type)
        reserved = ", ".join(sorted(reserved_names))
        for expected_name, actual_name in self._aliases.items():
            if not isinstance(expected_name, str) or not isinstance(actual_name, str):
                raise TypeError("'aliases' must be a mapping of strings to strings")
            if not expected_name or not actual_name:
                raise ValueError("Alias names must not be empty")
            if (
                    expected_name in reserved_names
                    or actual_name in reserved_names
            ):
                raise ValueError(f"{reserved} are reserved component names")
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

            if actual_name not in self._registry:
                raise ValueError(
                    f"Alias target '{actual_name}' is not a registered component"
                )
            actual_type = _component_type(self._registry[actual_name])
            if (
                    expected_name in self._registry
                    and _component_type(self._registry[expected_name])
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

    @property
    def session_type(self) -> str | None:
        return self._session_type

    def __bool__(self) -> bool:
        return bool(self._aliases)


def _alias_resolver(
        aliases: ComponentAliases | Mapping[str, str] | None,
        *,
        session_type: str | None = None,
) -> ComponentAliases:
    if isinstance(aliases, ComponentAliases):
        if aliases.session_type != session_type:
            raise ValueError(
                f"ComponentAliases uses session_type '{aliases.session_type}', "
                f"not '{session_type}'"
            )
        return aliases
    return ComponentAliases(aliases, session_type=session_type)


def requires_step(step_name: str):
    def wrapper(cls):
        if not issubclass(cls, Step):
            raise TypeError(
                f"@requires_step can only be applied to Step subclasses. "
                f"'{cls.__name__}' is not a Step."
            )
        if "required_steps" not in cls.__dict__:
            cls.required_steps = []
        cls.required_steps.append(step_name)
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
        cls.required_hooks.append(hook_name)
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
        if hook_name in cls.wrapped_hooks:
            raise ValueError(
                f"Hook '{cls.__name__}' already wraps '{hook_name}'"
            )
        cls.wrapped_hooks.append(hook_name)
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
        session_type: str | None = None,
) -> dict[str, int]:
    normalized = _normalize_component_session_type(session_type)
    alias_resolver = _alias_resolver(
        aliases,
        session_type=normalized,
    )
    registry = component_registry(normalized)
    return topological_sort_components(
        alias_resolver=alias_resolver,
        registry=registry,
        components=components,
    )


def format_execution_graph(
        *,
        resources: Iterable[Resource],
        hooks: Iterable[Hook],
        steps: Iterable[Step],
        max_iterations: int,
        aliases: ComponentAliases | Mapping[str, str] | None = None,
        session_type: str = TRAINING_SESSION_TYPE,
) -> str:
    """Return the session's component lifecycle as a readable execution graph."""
    normalized = _normalize_component_session_type(session_type)
    assert normalized is not None
    alias_resolver = _alias_resolver(
        aliases,
        session_type=normalized,
    )
    resources = list(resources)
    hooks = list(hooks)
    steps = list(steps)
    order = topological_sort_of_components(
        alias_resolver,
        components=resources + hooks + steps,
        session_type=normalized,
    )
    return render_execution_graph(
        resources=resources,
        hooks=hooks,
        steps=steps,
        max_iterations=max_iterations,
        alias_resolver=alias_resolver,
        session_type=normalized,
        order=order,
    )
