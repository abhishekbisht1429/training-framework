import warnings
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


class ComponentBindings:
    """Bind session-scoped component roles to registered implementations."""

    def __init__(
            self,
            bindings: Mapping[str, str] | None = None,
            *,
            session_type: str | None = None,
    ):
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            raise TypeError(
                "'component_bindings' must be a mapping of strings to strings"
            )

        normalized = _normalize_component_session_type(session_type)
        self._session_type = normalized
        self._registry = component_registry(normalized)
        self._bindings = dict(bindings)
        self._validate()

    def _validate(self) -> None:
        targets = {}
        reserved_names = reserved_config_names(self._session_type)
        reserved = ", ".join(sorted(reserved_names))
        for role_name, implementation_name in self._bindings.items():
            if (
                    not isinstance(role_name, str)
                    or not isinstance(implementation_name, str)
            ):
                raise TypeError(
                    "'component_bindings' must be a mapping of strings to strings"
                )
            if not role_name or not implementation_name:
                raise ValueError("Component binding names must not be empty")
            if (
                    role_name in reserved_names
                    or implementation_name in reserved_names
            ):
                raise ValueError(f"{reserved} are reserved component names")
            if role_name == implementation_name:
                raise ValueError(
                    f"Component binding '{role_name}' must refer to a "
                    "different component"
                )
            if implementation_name in self._bindings:
                raise ValueError(
                    "Component binding chains and cycles are not supported: "
                    f"'{role_name}' resolves to bound role "
                    f"'{implementation_name}'"
                )
            if implementation_name in targets:
                raise ValueError(
                    f"Component roles '{targets[implementation_name]}' and "
                    f"'{role_name}' cannot both bind to "
                    f"'{implementation_name}'"
                )

            if implementation_name not in self._registry:
                raise ValueError(
                    f"Component binding target '{implementation_name}' is not "
                    "a registered component"
                )
            implementation_type = _component_type(
                self._registry[implementation_name]
            )
            if (
                    role_name in self._registry
                    and _component_type(self._registry[role_name])
                    is not implementation_type
            ):
                raise ValueError(
                    f"Component binding '{role_name}' -> "
                    f"'{implementation_name}' changes the component category"
                )

            targets[implementation_name] = role_name

    def validate_config(self, config: Mapping) -> None:
        for role_name, implementation_name in self._bindings.items():
            if role_name in config:
                raise ValueError(
                    f"Component role '{role_name}' is bound to "
                    f"'{implementation_name}'. Configure the implementation "
                    f"name '{implementation_name}' at the top level, not the "
                    f"role name '{role_name}'."
                )

    def resolve(self, name: str) -> str:
        return self._bindings.get(name, name)

    def is_bound(self, name: str) -> bool:
        return name in self._bindings

    def is_alias(self, name: str) -> bool:
        warnings.warn(
            "ComponentBindings.is_alias() is deprecated; use is_bound()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_bound(name)

    @property
    def bindings(self) -> dict[str, str]:
        return dict(self._bindings)

    @property
    def session_type(self) -> str | None:
        return self._session_type

    def __bool__(self) -> bool:
        return bool(self._bindings)

    def __setstate__(self, state) -> None:
        legacy_bindings = state.pop("_aliases", None)
        if "_bindings" not in state and legacy_bindings is not None:
            state["_bindings"] = legacy_bindings
        self.__dict__.update(state)


class ComponentAliases(ComponentBindings):
    """Deprecated compatibility name for ComponentBindings."""

    def __init__(
            self,
            aliases: Mapping[str, str] | None = None,
            *,
            session_type: str | None = None,
    ):
        warnings.warn(
            "ComponentAliases is deprecated; use ComponentBindings",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(aliases, session_type=session_type)


def _coalesce_component_bindings(
        component_bindings,
        aliases,
        *,
        stacklevel: int = 3,
):
    if component_bindings is not None and aliases is not None:
        raise ValueError(
            "Provide either 'component_bindings' or deprecated 'aliases', "
            "not both"
        )
    if aliases is not None:
        warnings.warn(
            "'aliases' is deprecated; use 'component_bindings'",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return aliases
    return component_bindings


def _binding_resolver(
        component_bindings: ComponentBindings | Mapping[str, str] | None,
        *,
        session_type: str | None = None,
) -> ComponentBindings:
    if isinstance(component_bindings, ComponentBindings):
        if component_bindings.session_type != session_type:
            raise ValueError(
                "ComponentBindings uses session_type "
                f"'{component_bindings.session_type}', "
                f"not '{session_type}'"
            )
        return component_bindings
    return ComponentBindings(
        component_bindings,
        session_type=session_type,
    )


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
        component_bindings: ComponentBindings | Mapping[str, str] | None = None,
        *,
        components: Iterable | None = None,
        session_type: str | None = None,
        aliases: ComponentBindings | Mapping[str, str] | None = None,
) -> dict[str, int]:
    component_bindings = _coalesce_component_bindings(
        component_bindings,
        aliases,
    )
    normalized = _normalize_component_session_type(session_type)
    binding_resolver = _binding_resolver(
        component_bindings,
        session_type=normalized,
    )
    registry = component_registry(normalized)
    return topological_sort_components(
        binding_resolver=binding_resolver,
        registry=registry,
        components=components,
    )


def format_execution_graph(
        *,
        resources: Iterable[Resource],
        hooks: Iterable[Hook],
        steps: Iterable[Step],
        max_iterations: int,
        component_bindings: (
            ComponentBindings | Mapping[str, str] | None
        ) = None,
        session_type: str = TRAINING_SESSION_TYPE,
        aliases: ComponentBindings | Mapping[str, str] | None = None,
) -> str:
    """Return the session's component lifecycle as a readable execution graph."""
    component_bindings = _coalesce_component_bindings(
        component_bindings,
        aliases,
    )
    normalized = _normalize_component_session_type(session_type)
    assert normalized is not None
    binding_resolver = _binding_resolver(
        component_bindings,
        session_type=normalized,
    )
    resources = list(resources)
    hooks = list(hooks)
    steps = list(steps)
    order = topological_sort_of_components(
        binding_resolver,
        components=resources + hooks + steps,
        session_type=normalized,
    )
    return render_execution_graph(
        resources=resources,
        hooks=hooks,
        steps=steps,
        max_iterations=max_iterations,
        binding_resolver=binding_resolver,
        session_type=normalized,
        order=order,
    )
