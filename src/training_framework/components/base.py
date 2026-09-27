import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from training_framework.components.naming import parse_instance_name
from training_framework.util import CaptureInitMeta, context_entry, context_exit

if TYPE_CHECKING:
    from training_framework.session.base import Session


_DEPENDENCIES_KEYWORD = "__training_framework_dependencies__"
"""Private keyword carrying prerequisites into a component's construction."""

_SESSION_BOUND_ATTR = "_session_bound"
"""Instance ``__dict__`` key holding what a session gave the component."""

_CONSTRUCTOR_PREREQUISITES_ATTR = "_constructor_prerequisites"
"""Instance ``__dict__`` key naming the prerequisites its constructor
consulted."""


class _SessionBound:
    """What one session gives one component: the prerequisites it was handed
    (None: given none), the record of those it has asked for, and -- while
    its constructor runs -- every prerequisite the constructor consulted.

    It lives in the component's `__dict__`, so it lives and dies with the
    component like any attribute. It belongs to that one object and that
    session, though:

    - it pickles and deep-copies as an empty, ownerless holder (its own
      `__reduce__`), so whatever protocol pickles or copies the component,
      no prerequisite travels, and nothing has to strip it;
    - it keeps a weak reference to its owner and is only used by that
      object, so a shallow copy -- which shares the `__dict__` entry -- has
      no session either. A reference rather than an id: a copy can later sit
      at the address of an original that has been collected.

    Registering the component elsewhere replaces the whole holder, so no
    record outlives the prerequisites it describes.
    """

    __slots__ = ("owner", "dependencies", "linked", "constructing", "consulted")

    def __init__(
            self,
            owner: Any = None,
            dependencies: dict[str, Any] | None = None,
    ):
        self.owner = None if owner is None else weakref.ref(owner)
        self.dependencies = dependencies
        self.linked: dict[str, Any] = {}
        self.constructing = False
        self.consulted: dict[str, None] = {}

    def belongs_to(self, component: Any) -> bool:
        return self.owner is not None and self.owner() is component

    def consult(self, name: str | None = None) -> dict[str, Any] | None:
        """The prerequisites, as the component itself looks at them: for
        `name`, or all of them. A look taken while its constructor runs is
        recorded whatever it finds -- what the constructor did depends on
        it."""
        if self.constructing:
            names = (name,) if name is not None else tuple(
                self.dependencies or (),
            )
            self.consulted.update(dict.fromkeys(names))
        return self.dependencies

    def __reduce__(self):
        return (_SessionBound, ())


def _owned_holder(component: Any) -> _SessionBound | None:
    """`component`'s own holder: None when it has none, or when the one in
    its `__dict__` belongs to another object (it is a shallow copy)."""
    bound = getattr(component, "__dict__", {}).get(_SESSION_BOUND_ATTR)
    return bound if bound is not None and bound.belongs_to(component) else None


def _bind(component: Any, bound: _SessionBound) -> _SessionBound:
    # Through __dict__, which bypasses nn.Module.__setattr__ (so a
    # prerequisite module is not registered as a submodule of its consumer)
    # and needs no nn.Module.__init__ to have run first.
    component.__dict__[_SESSION_BOUND_ATTR] = bound
    return bound


def _session_bound(component: Any) -> _SessionBound:
    """The holder of what `component`'s session gave it, created empty on
    first use."""
    bound = _owned_holder(component)
    return _bind(component, _SessionBound(component)) if bound is None else bound


def _give_prerequisites(
        component: Any, dependencies: Mapping[str, Any],
) -> _SessionBound:
    """Hand `component` its prerequisites, replacing whatever an earlier
    session gave it -- the prerequisites and the record alike."""
    return _bind(component, _SessionBound(component, dict(dependencies)))


def _given_prerequisites(component: Any) -> dict[str, Any] | None:
    """The live prerequisites `component` was handed, or None if it was
    handed none (or is a class). The session's view: records nothing."""
    bound = _owned_holder(component)
    return None if bound is None else bound.dependencies


class ComponentMeta(CaptureInitMeta):
    """Apply component lifecycle behavior to class-local overrides."""

    def __call__(cls, *args, **kwargs):
        """Construct a component, handing it its prerequisites first.

        The session passes them under a private keyword. They are written into
        the instance after ``__new__`` and before ``__init__`` -- so a
        constructor can use them -- and otherwise construction is what
        ``type.__call__`` does: ``__new__`` receives the arguments, and
        ``__init__`` runs only when ``__new__`` returned an instance of the
        class. A metaclass that overrides ``__call__`` and defers to ``super()``
        passes the keyword through untouched. ``__init__`` never sees it, so
        the captured constructor arguments stay plain configuration.
        """
        if _DEPENDENCIES_KEYWORD not in kwargs:
            return super().__call__(*args, **kwargs)
        dependencies = kwargs.pop(_DEPENDENCIES_KEYWORD)
        instance = cls.__new__(cls, *args, **kwargs)
        if isinstance(instance, cls):
            bound = _give_prerequisites(instance, dependencies)
            bound.constructing = True
            try:
                type(instance).__init__(instance, *args, **kwargs)
            finally:
                bound.constructing = False
            # What the constructor looked at, it depends on: a fact about
            # how the instance was built, so it outlives any session.
            if bound.consulted:
                instance.__dict__[_CONSTRUCTOR_PREREQUISITES_ATTR] = tuple(
                    bound.consulted,
                )
        return instance

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

    declared_reads: ClassVar[tuple[str, ...]] = ()
    """`iteration_context` keys this component reads; set by `@reads`."""

    declared_writes: ClassVar[tuple[str, ...]] = ()
    """`iteration_context` keys this component writes; set by `@writes`."""

    state_version: ClassVar[int] = 1
    """The version of what this component checkpoints.

    Recorded with every checkpoint. Raise it when `get_state()` or the
    constructor arguments change shape, and implement `migrate_state` (and
    `migrate_init_args`, if the constructor changed) so checkpoints written
    by an earlier version still restore.
    """

    @classmethod
    def migrate_state(cls, from_version: int, state: Any) -> Any:
        """Return `state`, written at `from_version`, in the current shape.

        Called before `set_state` when a checkpoint was written by an older
        `state_version`. The default has no migration to offer.
        """
        raise ValueError(
            f"{cls._component_name()} was checkpointed at state_version "
            f"{from_version}, and is now at {cls.state_version} with no "
            "migrate_state to bring the old state forward"
        )

    @classmethod
    def migrate_init_args(cls, from_version: int, init_args: dict) -> dict:
        """Return constructor arguments recorded at `from_version`, updated.

        `init_args` is `{"args": tuple, "kwargs": dict}`. Called before the
        component is rebuilt from an older checkpoint; the default keeps
        them, for a version change that touched only the state.
        """
        return init_args

    def __init__(self, config: Mapping | None = None) -> None:
        """Initialize a component that does not require configuration."""
        self._parse_config_schema(config)

    def context_reads(self) -> dict[str, str]:
        """Return the `iteration_context` values this instance reads, as
        parameter name -> context key.

        Each is passed to `run` (a step) or `post_iteration_callback` (a
        hook) as the keyword argument of that name. A step runs after the
        step that writes each key. Override when a key comes from the
        configuration rather than the class: the parameter name stays fixed
        while the key it is filled from changes.
        """
        return {key: key for key in type(self).declared_reads}

    def context_writes(self) -> dict[str, str]:
        """Return the `iteration_context` values this instance writes, as
        output name -> context key, in the order a tuple result lists them.

        A step returns them from `run`; a hook from its pre-iteration
        callback, so they are there before any step runs. One output is the
        return value itself; several are a tuple in this order or a mapping
        by output name. Override when a key comes from the configuration.
        """
        return {key: key for key in type(self).declared_writes}

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
        """Return the prerequisites handed to this component, by asked name."""
        return _session_bound(self).linked

    @property
    def _dependencies(self) -> dict[str, "Resource"]:
        """Return the prerequisites the session injected, by declared name.

        Given by ``SessionComponents._construct`` *before* ``__init__`` runs,
        so a constructor may use them, or on registration.
        """
        injected = _session_bound(self).consult()
        return {} if injected is None else injected

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
        injected = _session_bound(self).consult(name)
        if injected is None:
            raise ComponentDependencyError(
                f"{self._component_name()} requested resource '{name}' but "
                "was given no prerequisites. A component that declares "
                "dependencies must be activated by the session -- through "
                "configuration or Session.activate_component() -- rather than "
                "constructed directly."
            )
        if name not in injected and name in getattr(
                type(self), "required_resources", (),
        ):
            raise ComponentDependencyError(
                f"{self._component_name()} requested resource '{name}', which "
                "it declares but which is no longer in its session: it was "
                "removed and nothing has been registered in its place."
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
        injected = _session_bound(self).consult(name)
        return injected is not None and name in injected

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
        self._refuse_rebuild_without_prerequisites()

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

    def _refuse_rebuild_without_prerequisites(self) -> None:
        """Refuse a pickle or copy this component could not be rebuilt from.

        The envelope is rebuilt by running the constructor outside any
        session, where no prerequisites exist. A constructor that consulted
        one -- took it, or only checked for it -- would fail there at
        `loads`, or silently take another branch. Refused here, where the
        mistake is.
        """
        asked = self.__dict__.get(_CONSTRUCTOR_PREREQUISITES_ATTR)
        if asked:
            name = self.__dict__.get("name", type(self).__name__)
            raise TypeError(
                f"{name} cannot be pickled or copied on its own: its "
                f"constructor consults the prerequisites {list(asked)} "
                "(get_dependency or has_dependency in __init__), and a pickle "
                "or copy is rebuilt by running the constructor outside any "
                "session, where there are none. Consult them in setup instead, or save it with "
                "Checkpointer.save_checkpoint and read it back "
                "with Checkpointer.load_component, which rebuild it with its "
                "prerequisites."
            )

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
    def pre_iteration_callback(self, session: "Session") -> Any:
        """Run before the iteration's steps; return what `@writes` declares
        (None when it declares nothing)."""
        pass

    @abstractmethod
    def post_iteration_callback(
        self, session: "Session", /, *args: Any, **reads: Any
    ) -> None:
        """Run after the iteration's steps, given what `@reads` declares as
        keyword arguments.

        Nothing is ever passed positionally; ``*args`` is here only so that
        an override taking its reads by name type-checks. What an override
        must take is set by `@reads` and checked when the session is built.
        """
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
    def run(self, session: "Session", /, *args: Any, **reads: Any) -> Any:
        """Run once per iteration.

        What `@reads` declares arrives as keyword arguments, and what
        `@writes` declares is returned: one value as is, several as a tuple
        in declaration order or a mapping by name. A step declaring no
        writes returns None.

        Nothing is ever passed positionally; ``*args`` is here only so that
        an override taking its reads by name type-checks. What an override
        must take is set by `@reads` and checked when the session is built.
        """
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
