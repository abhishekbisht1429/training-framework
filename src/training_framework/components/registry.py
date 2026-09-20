import warnings
from collections import ChainMap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from training_framework.components.base import Component, Hook, Resource, Step
from training_framework.components.config import reserved_config_names
from training_framework.components.graph import (
    render_execution_graph,
    topological_sort_components,
)
from training_framework.components.naming import (
    parse_instance_name,
    validate_component_name,
)


_COMPONENT_TYPES = (Resource, Hook, Step)
_ROLE_DECORATOR_NAMES = {Resource: "resource", Hook: "hook", Step: "step"}
TRAINING_SESSION_TYPE = "training"
ANALYSIS_SESSION_TYPE = "analysis"
_SHARED_COMPONENT_REGISTRY: dict[str, type[Component]] = {}
_SESSION_COMPONENT_REGISTRIES: dict[str, dict[str, type[Component]]] = {}
_COMPONENT_REGISTRY = _SHARED_COMPONENT_REGISTRY
_ANALYSIS_COMPONENT_REGISTRY = _SESSION_COMPONENT_REGISTRIES.setdefault(
    ANALYSIS_SESSION_TYPE,
    {},
)


@dataclass(frozen=True)
class RoleDeclaration:
    """Describe an abstract component role with no required implementation."""

    name: str
    category: type[Component]
    description: str | None = None


_SHARED_ROLE_REGISTRY: dict[str, RoleDeclaration] = {}
_SESSION_ROLE_REGISTRIES: dict[str, dict[str, RoleDeclaration]] = {}


def _missing_role_message(
        *,
        category: type[Component],
        name: str,
        resolved_name: str,
        declared_role: RoleDeclaration,
        consumer: type[Component] | None = None,
) -> str:
    consumer_clause = (
        f" by {consumer.__name__}" if consumer is not None else ""
    )
    description = (
        f": {declared_role.description}" if declared_role.description else ""
    )
    decorator_name = _ROLE_DECORATOR_NAMES[category]
    return (
        f"Role '{resolved_name}' ({category.__name__}{description}) is "
        f"required{consumer_clause} but has no implementation registered. "
        f"Implement a {category.__name__} subclass and register it via "
        f"@{decorator_name}('{resolved_name}', ...), or bind an existing "
        "implementation via component_bindings: "
        f"{{'{name}': '<implementation_name>'}}."
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


def _registration_role_registry(
        session_type: str | None,
) -> dict[str, RoleDeclaration]:
    normalized = _normalize_component_session_type(session_type)
    if normalized is None:
        return _SHARED_ROLE_REGISTRY
    return _SESSION_ROLE_REGISTRIES.setdefault(normalized, {})


def role_registry(
        session_type: str | None = None,
) -> Mapping[str, RoleDeclaration]:
    normalized = _normalize_component_session_type(session_type)
    if normalized is None:
        return _SHARED_ROLE_REGISTRY
    scoped = _SESSION_ROLE_REGISTRIES.setdefault(normalized, {})
    return ChainMap(scoped, _SHARED_ROLE_REGISTRY)


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
    validate_component_name(name)
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
        declared_role = role_registry(session_type).get(name)
        if declared_role is not None and declared_role.category is not registered_type:
            raise ValueError(
                f"Cannot register {registered_type.__name__} '{name}'; "
                f"'{name}' is declared as a {declared_role.category.__name__} "
                "role"
            )
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


def role(
        name: str,
        category: type[Component],
        *,
        description: str | None = None,
        session_type: str | None = None,
        overwrite: bool = False,
) -> RoleDeclaration:
    """Declare `name` as an abstract role expecting a `category` implementation.

    Records that some component depends on `name` as a Resource, Hook, or
    Step without registering a concrete implementation. Application code
    satisfies the role with @resource/@hook/@step under the same name, or
    under a different name bound via `component_bindings`. Declaring a role
    is optional: @requires_resource/@requires_hook/@requires_step accept any
    name whether or not it has been declared as a role.
    """
    validate_component_name(name, kind="Role name")
    if category not in _COMPONENT_TYPES:
        raise TypeError(
            "role() category must be Resource, Hook, or Step; got "
            f"{getattr(category, '__name__', category)!r}"
        )

    registry = _registration_role_registry(session_type)
    if name in registry and not overwrite:
        scope = session_type or "shared"
        raise ValueError(f"Role '{name}' already declared in '{scope}' scope")

    registered_class = component_registry(session_type).get(name)
    if registered_class is not None:
        registered_type = _component_type(registered_class)
        if registered_type is not category:
            raise ValueError(
                f"Cannot declare role '{name}' as {category.__name__}; "
                f"'{name}' is already registered as a {registered_type.__name__}"
            )

    declaration = RoleDeclaration(name, category, description)
    registry[name] = declaration
    return declaration


class ComponentBindings:
    """Bind session-scoped component roles to registered implementations.

    Two forms share the mapping, told apart by the value:

    * ``role: implementation`` binds a role for the whole session, which is
      the original form and still the common one.
    * ``consumer: {role: target}`` binds a role for one consumer only. It is
      how a session says which instance a component was wired to when more
      than one instance of a component exists, and it is kept here rather
      than inside the consumer's own configuration because a component's
      configuration is passed verbatim to its constructor -- and because the
      wiring has to be readable before anything is constructed.

    A target may name an instance (``model#b``); a role name may not, since a
    role is what a component class declares and a class cannot know which
    instance it will be given.
    """

    def __init__(
            self,
            bindings: "Mapping[str, str | Mapping[str, str]] | None" = None,
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
        self._roles = role_registry(normalized)
        self._bindings: dict[str, str] = {}
        self._instance_bindings: dict[str, dict[str, str]] = {}
        for key, value in dict(bindings).items():
            if isinstance(value, Mapping):
                self._instance_bindings[key] = dict(value)
            else:
                self._bindings[key] = value
        self._validate()
        self._validate_instance_bindings()

    def _target_implementation(self, target: str, *, role_name: str) -> str:
        """Return the registered component name a binding target names.

        A target may carry an instance suffix, in which case the component it
        is an instance of is what has to be registered.
        """
        try:
            implementation, _ = parse_instance_name(target)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Component binding '{role_name}' -> '{target}': {error}"
            ) from error

        if implementation not in self._registry:
            # Imported lazily: diagnostics imports this module.
            from training_framework.components.diagnostics import (
                explain_missing_component,
                with_explanation,
            )

            raise ValueError(with_explanation(
                f"Component binding target '{target}' is not "
                "a registered component",
                explain_missing_component(
                    implementation,
                    implementation,
                    session_type=self._session_type,
                ),
            ))
        return implementation

    def _validate_instance_bindings(self) -> None:
        reserved_names = reserved_config_names(self._session_type)
        reserved = ", ".join(sorted(reserved_names))
        for consumer, wiring in self._instance_bindings.items():
            if not isinstance(consumer, str) or not consumer:
                raise ValueError("Component binding names must not be empty")
            if consumer in reserved_names:
                raise ValueError(f"{reserved} are reserved component names")
            # The consumer names an instance, so its component must exist even
            # though which instances are active is not known here.
            self._target_implementation(consumer, role_name=consumer)

            for role_name, target in wiring.items():
                if not isinstance(role_name, str) or not isinstance(target, str):
                    raise TypeError(
                        "'component_bindings' must be a mapping of strings "
                        "to strings"
                    )
                if not role_name or not target:
                    raise ValueError(
                        "Component binding names must not be empty"
                    )
                validate_component_name(
                    role_name,
                    kind=f"Component binding role name for '{consumer}'",
                )
                if role_name in reserved_names or target in reserved_names:
                    raise ValueError(
                        f"{reserved} are reserved component names"
                    )
                self._target_implementation(
                    target,
                    role_name=f"{consumer}.{role_name}",
                )

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
            validate_component_name(
                role_name,
                kind="Component binding role name",
            )
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

            implementation = self._target_implementation(
                implementation_name,
                role_name=role_name,
            )
            implementation_type = _component_type(
                self._registry[implementation]
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
            declared_role = self._roles.get(role_name)
            if (
                    declared_role is not None
                    and declared_role.category is not implementation_type
            ):
                raise ValueError(
                    f"Component binding '{role_name}' -> "
                    f"'{implementation_name}' binds {implementation_type.__name__} "
                    f"'{implementation_name}' to role '{role_name}' declared as "
                    f"{declared_role.category.__name__}"
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

    def resolve(self, name: str, *, consumer: str | None = None) -> str:
        """Return the name `name` is bound to, for `consumer` if it has wiring.

        A consumer's own wiring wins over the session-wide binding, so one
        component can be pointed at a particular instance without changing
        what every other component sees.
        """
        if consumer is not None:
            wiring = self._instance_bindings.get(consumer)
            if wiring is not None and name in wiring:
                return wiring[name]
        return self._bindings.get(name, name)

    def is_bound(self, name: str, *, consumer: str | None = None) -> bool:
        if consumer is not None:
            wiring = self._instance_bindings.get(consumer)
            if wiring is not None and name in wiring:
                return True
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
    def instance_bindings(self) -> dict[str, dict[str, str]]:
        """Return the per-consumer wiring, by consumer instance name."""
        return {
            consumer: dict(wiring)
            for consumer, wiring in self._instance_bindings.items()
        }

    @property
    def session_type(self) -> str | None:
        return self._session_type

    def __bool__(self) -> bool:
        return bool(self._bindings)

    def __setstate__(self, state) -> None:
        legacy_bindings = state.pop("_aliases", None)
        if "_bindings" not in state and legacy_bindings is not None:
            state["_bindings"] = legacy_bindings
        # Pickled before per-consumer wiring existed.
        state.setdefault("_instance_bindings", {})
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
    validate_component_name(step_name, kind="Required Step name")

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
    validate_component_name(hook_name, kind="Required Hook name")

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
    validate_component_name(hook_name, kind="Wrapped Hook name")

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


def singleton(cls):
    """Mark a component a session may hold only one instance of.

    Applied directly to the class, without arguments::

        @singleton
        @resource("my_resource")
        class MyResource(Resource):
            ...

    Components may be configured more than once by default. This is for the
    ones where a second instance could not work -- typically because the
    component owns something there is only one of in the process.
    """
    if not isinstance(cls, type) or not issubclass(cls, (Step, Hook, Resource)):
        name = getattr(cls, "__name__", repr(cls))
        raise TypeError(
            "@singleton can only be applied to Step, Hook, or Resource "
            f"subclasses. '{name}' is neither."
        )
    cls.singleton = True
    return cls


def rank_zero_only(cls):
    """Mark a component that a distributed session builds on rank 0 only.

    Applied directly to the class, without arguments::

        @rank_zero_only
        @hook("my_reporter")
        class MyReporter(LifecycleHook):
            ...

    Secondary ranks build every configured component except those marked
    this way, so this is the declaration for work that must happen once per
    run -- logging, checkpointing, reporting -- rather than once per rank.
    The mark is inherited, and a session can add to it with
    ``ddp.rank_zero_components``.
    """
    if not isinstance(cls, type) or not issubclass(cls, (Step, Hook, Resource)):
        name = getattr(cls, "__name__", repr(cls))
        raise TypeError(
            "@rank_zero_only can only be applied to Step, Hook, or Resource "
            f"subclasses. '{name}' is neither."
        )
    cls.rank_zero_only = True
    return cls


def requires_resource(resource_name: str):
    validate_component_name(resource_name, kind="Required Resource name")

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
    roles = role_registry(normalized)
    return topological_sort_components(
        binding_resolver=binding_resolver,
        registry=registry,
        components=components,
        roles=roles,
        session_type=normalized,
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
