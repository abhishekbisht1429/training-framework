from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from training_framework.components.naming import parse_instance_name
from training_framework.util import CaptureInitMeta, context_entry, context_exit

if TYPE_CHECKING:
    from training_framework.session.base import Session


class ComponentMeta(CaptureInitMeta):
    """Apply component lifecycle behavior to class-local overrides."""

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        if getattr(cls, "_context_managed_lifecycle", False):
            lifecycle_wrappers = {
                "setup": context_entry,
                "pre_session": context_entry,
                "teardown": context_exit,
                "post_session": context_exit,
            }
            for method_name, wrapper in lifecycle_wrappers.items():
                if method_name in namespace:
                    setattr(cls, method_name, wrapper(namespace[method_name]))

        return cls


class ComponentDependencyError(RuntimeError):
    """A component could not be wired to its prerequisite components."""


class Component(ABC, metaclass=ComponentMeta):
    """Common base for every executable training-framework component."""

    name: str
    id: str
    _context_managed_lifecycle = False

    config_schema: ClassVar[type | None] = None
    """Optional dataclass describing this component's configuration."""

    singleton: ClassVar[bool] = False
    """Whether a session may hold only one instance of this component.

    Set by :func:`singleton`. Most components may be configured more than
    once; this marks the ones where a second instance would be meaningless or
    harmful because they own something process-wide.
    """

    rank_zero_only: ClassVar[bool] = False
    """Whether a distributed session builds this component on rank 0 only.

    Set by :func:`rank_zero_only`. Components run on every rank by default:
    leaving one out of a rank is what deadlocks a collective, while running
    a rank-zero-only component everywhere merely duplicates its work.
    """

    def __init__(self, config: Mapping | None = None) -> None:
        """Initialize a component that does not require configuration."""
        self._parse_config_schema(config)

    @classmethod
    def _component_name(cls) -> str:
        return getattr(cls, "name", cls.__name__)

    @property
    def instance_suffix(self) -> str | None:
        """Return the suffix telling this instance from its siblings.

        None when the component is the only instance of itself, which is the
        usual case. A component that writes somewhere named after itself --
        a directory, a file, a run name -- uses this to keep two instances
        from landing on top of each other, while leaving the single-instance
        name exactly as it was.
        """
        name = getattr(self, "name", None)
        if not isinstance(name, str):
            return None
        try:
            _, suffix = parse_instance_name(name)
        except (TypeError, ValueError):
            return None
        return suffix

    def _stamp_identity(self, instance_name: str) -> None:
        """Give this instance its own name and id.

        Registration writes `name` and `id` onto the *class*, so every
        instance of a component would otherwise report the same pair. The
        session names the instance instead, which is what lets a name identify
        one component rather than one component class.

        Assigned through `__dict__` so that an `nn.Module` subclass needs no
        `nn.Module.__init__` to have run first.
        """
        self.__dict__["name"] = instance_name
        self.__dict__["id"] = (
            f"{self._component_category_name()}.{instance_name}"
        )

    @property
    def implementation_name(self) -> str:
        """Return the registered name of the class implementing this component.

        Distinct from ``name``, which the session overwrites per instance:
        several instances of one component share an implementation name and
        have different names.
        """
        return type(self)._component_name()

    def _parse_config_schema(self, config: Mapping | None) -> None:
        """Populate ``self._cfg`` when the class declares a ``config_schema``."""
        if type(self).config_schema is None:
            return
        # Imported lazily: config_schema imports base for the error type.
        from training_framework.components.config_schema import (
            parse_component_config,
        )
        self._cfg = parse_component_config(type(self), config)

    @property
    def _linked_components(self) -> dict[str, "Resource"]:
        """Return the prerequisites handed to this component, by asked name.

        Created on first use: a component may ask for a dependency before -- or
        without ever -- calling ``Component.__init__``.
        """
        linked = self.__dict__.get("_linked_components_map")
        if linked is None:
            linked = {}
            # Assigned through __dict__ so that an nn.Module subclass needs no
            # nn.Module.__init__ to have run first.
            self.__dict__["_linked_components_map"] = linked
        return linked

    DEPENDENCIES_ATTR = "_injected_dependencies"
    """Instance ``__dict__`` key holding the injected prerequisites."""

    @property
    def _dependencies(self) -> dict[str, "Resource"]:
        """Return the prerequisites the session injected, by declared name.

        Written into the instance ``__dict__`` by
        ``SessionComponents._construct`` *before* ``__init__`` runs, so a
        constructor may use them and an ``nn.Module`` subclass needs no
        ``nn.Module.__init__`` to have run first. Writing through
        ``__dict__`` also bypasses ``nn.Module.__setattr__``, so a
        prerequisite module is not registered as a submodule of its consumer.
        """
        injected = self.__dict__.get(self.DEPENDENCIES_ATTR)
        return {} if injected is None else injected

    def __getstate__(self) -> Any:
        """Leave injected prerequisites out of a component's own pickle.

        They belong to the session that injected them. A component pickled on
        its own -- rather than as part of a session, which rebuilds its
        components through construction -- would otherwise carry copies of
        other components, and registering it into a session replaces them
        anyway.

        Cooperative, so `Stateful`'s reconstruction envelope, which follows
        `Component` in the MRO of every stateful component, still decides
        what a stateful component pickles.
        """
        state = super().__getstate__()
        if isinstance(state, dict) and self.DEPENDENCIES_ATTR in state:
            state = dict(state)
            del state[self.DEPENDENCIES_ATTR]
        return state

    def get_dependency(self, name: str) -> "Resource":
        """Return a prerequisite resource declared with ``@requires_resource``.

        Valid at any point in a component's life. Prerequisites are resolved
        for *this* consumer -- honouring its own ``component_bindings`` wiring
        -- and injected before ``__init__`` runs, so construction, ``setup``
        and a running step all see the same instance.

        What is handed out is recorded, so the framework knows this component
        was wired to another one wherever the caller puts the reference --
        an attribute, a container module, or nowhere at all.
        """
        injected = self.__dict__.get(self.DEPENDENCIES_ATTR)
        if injected is None:
            raise ComponentDependencyError(
                f"{self._component_name()} requested resource '{name}' but "
                "was given no prerequisites. A component that declares "
                "dependencies must be activated by the session -- through "
                "configuration or Session.activate_component() -- rather than "
                "constructed directly."
            )
        if name not in injected:
            declared = ", ".join(sorted(injected)) or "nothing"
            raise ComponentDependencyError(
                f"{self._component_name()} requested resource '{name}' but "
                f"does not declare it. Add @requires_resource('{name}') so it "
                f"is constructed first. Declared: {declared}."
            )
        component = injected[name]
        self._linked_components[name] = component
        return component

    @property
    def linked_components(self) -> dict[str, str]:
        """Return the asked name -> instance name map of prerequisites.

        The *instance* is recorded, not merely the class implementing it, so
        that rewiring a consumer between two instances of one component is
        visible to the checkpoint guard rather than silently restoring one
        instance's state into another.
        """
        return {
            name: getattr(component, "name", type(component).__name__)
            for name, component in self._linked_components.items()
        }

    def has_dependency(self, name: str) -> bool:
        """Return whether a declared prerequisite was injected."""
        return name in self._dependencies

    @classmethod
    @abstractmethod
    def _component_category_name(cls) -> str:
        """Return the top-level lifecycle category implemented by the class."""
        raise NotImplementedError


class ExtendableComponent(ABC):
    """Opt a component into safe configuration changes during extension."""

    @abstractmethod
    def apply_extension_config(
            self,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        """Validate and apply an effective config to a restored component."""
        raise NotImplementedError


class Stateful(ABC):
    _PICKLE_VERSION_KEY = "__training_framework_pickle_version__"
    _PICKLE_VERSION = 1

    @abstractmethod
    def get_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: Any) -> None:
        raise NotImplementedError

    def __getstate__(self) -> Any:
        if not isinstance(self, Component):
            return self.get_state()

        envelope = {
            self._PICKLE_VERSION_KEY: self._PICKLE_VERSION,
            "init_args": self._init_args,
            "state": self.get_state(),
        }
        # Reconstruction runs __init__, which does not name the instance, so
        # a suffixed instance would come back under its class's name and, for
        # a component that names its output after itself, write on top of its
        # sibling. Optional: an envelope without it is simply unnamed, so the
        # pickle version does not change.
        instance_name = self.__dict__.get("name")
        if instance_name is not None:
            envelope["instance_name"] = instance_name
        return envelope

    def __setstate__(self, state: Any) -> None:
        if (
                isinstance(state, Mapping)
                and state.get(self._PICKLE_VERSION_KEY)
                == self._PICKLE_VERSION
        ):
            init_args = state["init_args"]
            self.__init__(
                *init_args["args"],
                **init_args["kwargs"],
            )
            instance_name = state.get("instance_name")
            if instance_name is not None:
                self._stamp_identity(instance_name)
            self.set_state(state["state"])
            return

        # Pickles created before the reconstruction envelope contained only
        # the component state. Preserve that best-effort restoration path.
        self.set_state(state)


class Hook(Component, ABC):
    """Base category for session and iteration hooks."""

    _context_managed_lifecycle = True

    @classmethod
    def _component_category_name(cls) -> str:
        return "Hook"


class SessionHook(Hook, ABC):
    @abstractmethod
    def pre_session(self, session: "Session") -> None:
        pass

    @abstractmethod
    def post_session(self, session: "Session") -> None:
        pass

    def rollback_pre_session(self, session: "Session") -> None:
        """Undo partial effects after :meth:`pre_session` fails.

        The default is intentionally a no-op so existing hooks remain
        backward compatible. Hooks that can create external effects before
        ``pre_session`` completes should override this method.
        """
        pass


class IterationHook(Hook, ABC):
    call_every: int

    @abstractmethod
    def pre_iteration_callback(self, session: "Session") -> None:
        pass

    @abstractmethod
    def post_iteration_callback(self, session: "Session") -> None:
        pass


class LifecycleHook(SessionHook, IterationHook, ABC):
    """Wrap callbacks around a training iteration."""


class Resource(Component, ABC):

    _context_managed_lifecycle = True

    @classmethod
    def _component_category_name(cls) -> str:
        return "Resource"

    @abstractmethod
    def setup(self, session: "Session") -> None:
        pass

    @abstractmethod
    def teardown(self, session: "Session") -> None:
        pass

    def rollback_setup(self, session: "Session") -> None:
        """Undo partial effects after :meth:`setup` fails.

        The default is intentionally a no-op so existing resources remain
        backward compatible. Resources that can create external effects
        before ``setup`` completes should override this method.
        """
        pass


class Step(Component, ABC):

    @classmethod
    def _component_category_name(cls) -> str:
        return "Step"

    @abstractmethod
    def run(self, session: "Session") -> None:
        pass


class StatefulIterationHook(IterationHook, Stateful, ABC):
    pass


class StatefulSessionHook(SessionHook, Stateful, ABC):
    pass


class StatefulLifeCycleHook(LifecycleHook, Stateful, ABC):
    pass


StatefulLifecycleHook = StatefulLifeCycleHook


class StatefulStep(Step, Stateful, ABC):
    pass


class StatefulResource(Resource, Stateful, ABC):
    pass
