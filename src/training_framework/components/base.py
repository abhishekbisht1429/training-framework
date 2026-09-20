from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
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


class ComponentView(ABC):
    """Narrow, session-free view of the components constructed so far.

    Bound while a component is being constructed. It deliberately exposes
    only component lookup: there is no session, no device, and no iteration
    context during construction.
    """

    @property
    @abstractmethod
    def session_type(self) -> str:
        """Return the session type the components belong to."""
        raise NotImplementedError

    @abstractmethod
    def resolve_name(self, name: str) -> str:
        """Return the implementation name a role name is bound to."""
        raise NotImplementedError

    @abstractmethod
    def has_resource(self, name: str) -> bool:
        """Return whether a resource is active under ``name``."""
        raise NotImplementedError

    @abstractmethod
    def get_resource(self, name: str) -> "Resource":
        """Return the active resource registered or bound to ``name``."""
        raise NotImplementedError


_COMPONENT_VIEW: ContextVar["ComponentView | None"] = ContextVar(
    "training_framework_component_view",
    default=None,
)


@contextmanager
def constructing_component(view: "ComponentView | None"):
    """Bind the dependency view visible to a component being constructed.

    Components are constructed prerequisite-first, so by the time a
    constructor runs every component it declared already exists. The view is
    bound rather than passed so that ``_init_args`` stays plain configuration
    and the config-free activation policy keeps working.
    """
    token = _COMPONENT_VIEW.set(view)
    try:
        yield
    finally:
        _COMPONENT_VIEW.reset(token)


def active_component_view() -> "ComponentView | None":
    """Return the view bound for the component currently being constructed."""
    return _COMPONENT_VIEW.get()


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

    def get_dependency(self, name: str) -> "Resource":
        """Return a prerequisite resource. Valid only during construction.

        Components are constructed prerequisite-first, so a component may ask
        for anything it declared via ``@requires_resource``. There is no
        session yet: no device, no iteration context, and no
        ``@requires_context`` access. Work needing those belongs in
        :meth:`Resource.setup`.

        What is handed out is recorded, so the framework knows this component
        was wired to another one wherever the caller puts the reference --
        an attribute, a container module, or nowhere at all.
        """
        view = active_component_view()
        if view is None:
            raise ComponentDependencyError(
                f"{self._component_name()} requested resource '{name}' while "
                "no component view is bound. A component that declares "
                "dependencies must be activated by the session -- through "
                "configuration or Session.activate_component() -- rather than "
                "constructed directly."
            )
        component = view.get_resource(name)
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
        """Return whether a declared prerequisite is active."""
        view = active_component_view()
        if view is None:
            return False
        return view.has_resource(name)

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

        return {
            self._PICKLE_VERSION_KEY: self._PICKLE_VERSION,
            "init_args": self._init_args,
            "state": self.get_state(),
        }

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
