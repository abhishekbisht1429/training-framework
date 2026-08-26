from collections.abc import Iterable, Mapping

from training_framework.components.base import (
    Component,
    Hook,
    Resource,
    Step,
)
from training_framework.components.graph import (
    render_execution_graph,
    topological_sort_components,
)
from training_framework.components.config import RESERVED_CONFIG_NAMES


_COMPONENT_TYPES = (Resource, Hook, Step)
TRAINING_MODE = "training"
ANALYSIS_MODE = "analysis"
_COMPONENT_MODES = (TRAINING_MODE, ANALYSIS_MODE)
_COMPONENT_REGISTRY: dict[str, type[Component]] = {}
_ANALYSIS_COMPONENT_REGISTRY: dict[str, type[Component]] = {}
_COMPONENT_REGISTRIES: dict[str, dict[str, type[Component]]] = {
    TRAINING_MODE: _COMPONENT_REGISTRY,
    ANALYSIS_MODE: _ANALYSIS_COMPONENT_REGISTRY,
}


def normalize_component_mode(mode=TRAINING_MODE) -> str:
    normalized = getattr(mode, "value", mode)
    if normalized not in _COMPONENT_MODES:
        allowed = ", ".join(_COMPONENT_MODES)
        raise ValueError(
            f"Invalid component mode {mode!r}; expected one of: {allowed}"
        )
    return normalized


def component_registry(mode=TRAINING_MODE) -> dict[str, type[Component]]:
    return _COMPONENT_REGISTRIES[normalize_component_mode(mode)]


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
        mode=TRAINING_MODE,
):
    registry = component_registry(mode)

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
                raise ValueError(f"Component with name '{name}' already registered")
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
        mode=TRAINING_MODE,
):
    return _component(
        name,
        expected_type=Hook,
        overwrite=overwrite,
        mode=mode,
    )


def resource(
        name: str,
        overwrite=False,
        *,
        mode=TRAINING_MODE,
):
    return _component(
        name,
        expected_type=Resource,
        overwrite=overwrite,
        mode=mode,
    )


def step(
        name: str,
        overwrite=False,
        *,
        mode=TRAINING_MODE,
):
    return _component(
        name,
        expected_type=Step,
        overwrite=overwrite,
        mode=mode,
    )

class ComponentAliases:
    """Resolve session-scoped component roles to registered implementations."""

    def __init__(
            self,
            aliases: Mapping[str, str] | None = None,
            *,
            mode=TRAINING_MODE,
    ):
        if aliases is None:
            aliases = {}
        if not isinstance(aliases, Mapping):
            raise TypeError("'aliases' must be a mapping of strings to strings")

        self._mode = normalize_component_mode(mode)
        self._registry = component_registry(self._mode)
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
    def mode(self) -> str:
        return self._mode

    def __bool__(self) -> bool:
        return bool(self._aliases)


def _alias_resolver(
        aliases: ComponentAliases | Mapping[str, str] | None,
        *,
        mode=TRAINING_MODE,
) -> ComponentAliases:
    if isinstance(aliases, ComponentAliases):
        return aliases
    return ComponentAliases(aliases, mode=mode)


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


def topological_sort_of_components(
        aliases: ComponentAliases | Mapping[str, str] | None = None,
        *,
        components: Iterable | None = None,
        mode=TRAINING_MODE,
) -> dict[str, int]:
    session_mode = normalize_component_mode(mode)
    alias_resolver = _alias_resolver(aliases, mode=session_mode)
    registry = component_registry(alias_resolver.mode)
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
        mode=TRAINING_MODE,
) -> str:
    """Return the session's component lifecycle as a readable execution graph."""
    session_mode = normalize_component_mode(mode)
    alias_resolver = _alias_resolver(aliases, mode=session_mode)
    resources = list(resources)
    hooks = list(hooks)
    steps = list(steps)
    order = topological_sort_of_components(
        alias_resolver,
        components=resources + hooks + steps,
        mode=session_mode,
    )
    return render_execution_graph(
        resources=resources,
        hooks=hooks,
        steps=steps,
        max_iterations=max_iterations,
        alias_resolver=alias_resolver,
        session_mode=session_mode,
        order=order,
    )
