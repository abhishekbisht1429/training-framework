import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from training_framework.components import (
    Component,
    ComponentDependencyError,
    ExtendableComponent,
    Hook,
    IterationHook,
    Resource,
    SessionHook,
    Stateful,
    Step,
)
from training_framework.components.base import (
    _DEPENDENCIES_KEYWORD,
    _give_prerequisites,
    _given_prerequisites,
)
from training_framework.components.config_schema import has_all_defaults
from training_framework.components.edges import (
    Edge,
    EdgeKind,
    context_keys,
    context_keys_of,
    declared_edges,
    instances_of,
    wiring_of,
    resolve_component_name,
    resolve_edges,
    writers_of,
)
from training_framework.components.config import (
    SessionConfigView,
    parse_session_config,
)
from training_framework.components.diagnostics import (
    explain_missing_component,
    with_explanation,
)
from training_framework.components.naming import (
    check_not_imported_suffix,
    has_imported_suffix,
    implementation_of,
    is_instance_name,
    parse_instance_name,
)
from training_framework.components.registry import (
    RoleBindings,
    _coalesce_role_bindings,
    _component_type,
    _missing_role_message,
    component_registry,
    role_registry,
    topological_sort_of_components,
)
from training_framework.session.checkpoint_format import IMPORTED_INTO_NAMESPACE_KEY
from training_framework.session.config import TRAINING_SESSION_TYPE, normalize_session_type


_GROUP_KINDS = {"resources": Resource, "hooks": Hook, "steps": Step}


def read_session_config(
        config: Mapping,
        *,
        session_type: str | None,
        warn_legacy: bool = False,
) -> SessionConfigView:
    """Read a session configuration and check what needs the registry.

    The one way session code reads a configuration: `parse_session_config`
    reads the layout, and this adds the checks that need to know which
    components exist -- a component listed under a group of another kind is
    refused. A build and an extension read alike, so neither can accept a
    configuration the other would refuse.
    """
    view = parse_session_config(
        config, session_type=session_type, warn_legacy=warn_legacy,
    )
    registry = component_registry(normalize_session_type(session_type))
    for name, entry in view.components.items():
        if entry.group is not None:
            _check_component_group(registry, name, entry.group)
    return view


def _check_component_group(
        registry: Mapping[str, type[Component]],
        name: str,
        group: str,
) -> None:
    """Refuse a component listed under a group of another kind.

    A configured name is an implementation or one of its instances -- a
    bound role cannot be configured -- so the registry answers directly. An
    unregistered name is left to the diagnostics activation gives.
    """
    component_class = registry.get(implementation_of(name))
    if component_class is None:
        return
    expected = _GROUP_KINDS[group]
    if issubclass(component_class, expected):
        return
    actual = _component_type(component_class)
    right_group = next(
        key for key, kind in _GROUP_KINDS.items() if kind is actual
    )
    raise ValueError(
        f"'{name}' is listed under '{group}', but it is a "
        f"{actual.__name__}, not a {expected.__name__}. List it under "
        f"'{right_group}'."
    )


@dataclass(frozen=True)
class _PlannedComponent:
    component_class: type[Component]
    constructor_args: tuple
    edges: list[Edge]


@dataclass(frozen=True)
class _RestorePlan:
    component_class: type[Component]
    dependencies: dict[str, str]
    init_args: dict
    state: Any
    #: Why the saved state is not restored, when it is not.
    reinitialized: str | None = None


def _edge_description(edge: Edge) -> str:
    """How an edge reads in the rank-zero dependency error."""
    if edge.kind is EdgeKind.READS:
        return f"reads '{edge.asked}', written by '{edge.target}'"
    if edge.kind is EdgeKind.COMPANION:
        return f"activates '{edge.target}'"
    return f"{edge.kind.value} '{edge.target}'"


class ComponentNotFoundError(KeyError):
    """A KeyError whose message keeps its line breaks when printed."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class CheckpointMismatchError(ValueError):
    """Several checkpointed components cannot be restored; lists them all."""


class SessionComponents:
    def __init__(
            self,
            *,
            resources: dict[str, Resource] | None = None,
            steps: dict[str, Step] | None = None,
            hooks: dict[str, Hook] | None = None,
            role_bindings: Mapping[str, Any] | None = None,
            session_type: str = TRAINING_SESSION_TYPE,
            dependency_bindings: Mapping[str, Mapping[str, Any]] | None = None,
            component_bindings: Mapping[str, Any] | None = None,
            aliases: Mapping[str, Any] | None = None,
            imports: Mapping[str, Any] | None = None,
    ):
        """`imports` is a stored session state's record of its imports; its
        role bindings must already be in `role_bindings`.

        `component_bindings` and `aliases` are deprecated names for
        `role_bindings`."""
        self.session_type = normalize_session_type(session_type)
        imports = imports or {}
        # The role bindings this session's imports added, saved with the
        # session state: the configuration does not say them.
        self.import_bindings: dict[str, str] = dict(imports.get("bindings") or {})
        # The instances imports brought in (instance -> import key). Each
        # is saved on its own component's entry (`imported_by`), and read
        # back when that component is restored.
        self.imported: dict[str, str] = {}
        # The imported instances that are ordinary members of the namespace
        # (brought in by a keyed import). The rest, from a deprecated
        # labelled import, are reached only through a binding naming them.
        # Saved as `imported_into_namespace` on the component's entry.
        self.namespace_imports: set[str] = set()
        # Set once a session is built from configuration: its per-consumer
        # bindings still have to name components it holds, which is known
        # only once nothing more is activated by hand.
        self._binding_check_pending = False
        self.registry = component_registry(self.session_type)
        self.roles = role_registry(self.session_type)
        self.components: dict[str, Component] = {}
        self._merge_components(resources, Resource)
        self._merge_components(hooks, Hook)
        self._merge_components(steps, Step)
        role_bindings = _coalesce_role_bindings(
            role_bindings,
            component_bindings,
            aliases,
        )
        self.role_bindings = RoleBindings(
            role_bindings,
            dependency_bindings=dependency_bindings,
            session_type=self.session_type,
        )

    def __setstate__(self, state) -> None:
        # Pickled under the attribute's former names.
        legacy_bindings = state.pop("component_bindings", None)
        legacy_aliases = state.pop("aliases", None)
        if "role_bindings" not in state:
            state["role_bindings"] = (
                legacy_bindings if legacy_bindings is not None
                else legacy_aliases
            )
        state.pop("_links_dirty", None)
        state.setdefault("import_bindings", {})
        state.setdefault("imported", {})
        state.setdefault("namespace_imports", set())
        state.setdefault("_binding_check_pending", False)
        self.__dict__.update(state)

    def imports_state(self) -> dict[str, dict[str, str]]:
        """What the session state records about this session's imports, apart
        from each component's own entry: the role bindings they added."""
        return {"bindings": dict(self.import_bindings)}

    def _check_tensor_ownership(self) -> None:
        """Check no tensor is about to be checkpointed by two components.

        This is the ground truth the per-component check cannot see: a
        component only knows its own tree, so it cannot tell a module it owns
        privately from one another component also captures. Checking the
        captured tensors themselves catches double ownership whatever shape
        the module trees take, and needs no bookkeeping to stay correct.
        """
        owners: dict[int, tuple[str, str]] = {}
        for name, component in self.components.items():
            captured_tensors = getattr(component, "captured_tensors", None)
            if captured_tensors is None:
                continue
            for key, tensor in captured_tensors().items():
                previous = owners.get(id(tensor))
                if previous is not None:
                    other_name, other_key = previous
                    raise ComponentDependencyError(
                        f"'{name}.{key}' and '{other_name}.{other_key}' are "
                        "the same tensor, so it would be checkpointed twice. "
                        "A component that holds another component's weights "
                        "must take it from get_dependency(), so the component "
                        "that created them is the one that saves them."
                    )
                owners[id(tensor)] = (name, key)

    def get_state(self) -> dict[str, dict[str, Any]]:
        self._check_tensor_ownership()
        return {
            name: {
                "component_type": _component_type(component).__name__,
                # The class behind the instance. A checkpoint key names an
                # instance, which need not be the name it was registered
                # under, so the two are recorded separately.
                "implementation": component.implementation_name,
                "state": (
                    component.get_state()
                    if isinstance(component, Stateful)
                    else None
                ),
                "init_args": getattr(component, "_init_args"),
                # What a worker needs to decide which components its rank
                # keeps before anything is rebuilt from this state.
                "context_reads": list(context_keys(component)[0]),
                "context_writes": list(context_keys(component)[1]),
                # The wiring this instance was actually given, so a restore
                # rebuilds it as it was rather than re-deriving it, and one
                # component can be restored with only what it was given.
                "dependencies": {
                    asked: getattr(dependency, "name", type(dependency).__name__)
                    for asked, dependency in component._dependencies.items()
                },
                "state_version": type(component).state_version,
                # Which import brought it in, if one did.
                **(
                    {"imported_by": self.imported[name]}
                    if name in self.imported else {}
                ),
                **(
                    {IMPORTED_INTO_NAMESPACE_KEY: True}
                    if name in self.namespace_imports else {}
                ),
            }
            for name, component in self.components.items()
        }

    def set_state(
            self,
            component_states: dict[str, dict[str, Any]],
            *,
            on_mismatch: str | Iterable[str] = "raise",
            partial: bool = False,
            wired_to: Iterable[str] | None = None,
            names: Mapping[str, str] | None = None,
    ) -> None:
        """Rebuild the components a state holds and restore their state.

        `on_mismatch` decides what happens to a component whose saved state
        this version cannot take -- written by a newer `state_version`, or an
        older one with no `migrate_state`: "raise" (the default) refuses the
        whole restore; "reinit" keeps every such component as freshly built
        (from its constructor arguments, migrated if they changed), with a
        warning naming them; a collection of instance names does that for
        those only. Whatever cannot be built at all -- including constructor
        arguments `migrate_init_args` cannot bring forward -- always raises.

        `partial` says the state holds a chosen subset of a session's
        components (each with what it depends on), not a whole session.

        `wired_to`, for an import: the state is another run's components,
        restored *into* this session next to the components it already
        holds, and may be wired to the ones named here as well as to each
        other. Each is given exactly the wiring the state recorded: a
        prerequisite its class declares that the recorded wiring lacks is
        refused, as is a binding of this session aimed at it. `names` (source
        instance -> this session's name) is handed to each component's
        `rename_instances`, after its state is migrated.
        """
        # Components are rebuilt into a fresh mapping -- or, for an import,
        # one that starts with this session's components -- and a component
        # being constructed must see the ones already rebuilt, so the view
        # reads through to it. The previous mapping is put back if the
        # restore fails.
        previous_components = self.components
        if wired_to is None:
            restored_components: dict[str, Component] = {}
        else:
            clashes = sorted(set(component_states) & set(previous_components))
            if clashes:
                raise ValueError(
                    f"Cannot restore {clashes} into this session: it already "
                    "holds components with those names"
                )
            restored_components = dict(previous_components)
        self.components = restored_components
        try:
            self._restore_components(
                component_states,
                restored_components,
                on_mismatch=on_mismatch,
                partial=partial,
                wired_to=None if wired_to is None else frozenset(wired_to),
                names=names or {},
            )
        except BaseException:
            self.components = previous_components
            raise
        self.imported.update(
            (name, info["imported_by"])
            for name, info in component_states.items()
            if info.get("imported_by")
        )
        self.namespace_imports.update(
            name for name, info in component_states.items()
            if info.get("imported_by") and info.get(IMPORTED_INTO_NAMESPACE_KEY)
        )

    def _restore_components(
            self,
            component_states: dict[str, dict[str, Any]],
            restored_components: dict[str, Component],
            *,
            on_mismatch: str | Iterable[str] = "raise",
            partial: bool = False,
            wired_to: frozenset[str] | None = None,
            names: Mapping[str, str] | None = None,
    ) -> None:
        # A checkpoint is not a trusted plan: it may predate a component being
        # marked @singleton, or have been edited. Checked before anything is
        # built; set_state restores the previous components if it raises.
        # Restored into a session, against everything it will then hold.
        self._check_instance_limits(
            set(component_states) | set(restored_components),
        )

        # Every component is checked before any is built, and every problem
        # is reported at once rather than the first one only.
        plans = self._plan_restore(
            component_states, on_mismatch,
            wired_to=wired_to, names=names,
        )

        # Pass 1: rebuild every component from its constructor arguments,
        # prerequisites first. The stored order is usually already
        # prerequisite-first, since it is the activation order, but a
        # component registered by hand -- say a prerequisite replaced after
        # its consumer was built -- is stored after the components that use
        # it. Each component is handed its prerequisites as it is built, so
        # they are built first here rather than trusted to come first.
        building: list[str] = []
        build_order: list[str] = []

        def build(name: str) -> None:
            if name in restored_components:
                return
            if name in building:
                chain = " -> ".join([*building, name])
                raise ValueError(
                    f"Checkpoint components depend on each other in a "
                    f"cycle: {chain}"
                )
            plan = plans[name]
            building.append(name)
            try:
                for target in plan.dependencies.values():
                    build(target)
            finally:
                building.pop()

            restored_components[name] = self._construct(
                plan.component_class,
                name,
                *plan.init_args["args"],
                dependencies=plan.dependencies,
                **plan.init_args["kwargs"],
            )
            build_order.append(name)

        for name in component_states:
            build(name)

        # Pass 2: restore state in prerequisite-first order, so a component
        # that inspects a dependency sees it already restored. A partial
        # state is not a session, so it is not sorted as one; the build
        # order is prerequisite-first by construction.
        order = (
            build_order
            if partial
            else self._state_restore_order(component_states)
        )
        failures = []
        for name in order:
            component = self.components[name]
            plan = plans[name]
            if not isinstance(component, Stateful) or plan.reinitialized:
                continue
            try:
                component.set_state(plan.state)
            except Exception as error:
                failures.append((name, error))
        if len(failures) == 1:
            raise failures[0][1]
        if failures:
            raise CheckpointMismatchError(
                "Checkpoint state could not be restored into "
                f"{len(failures)} components:\n"
                + "\n".join(
                    f"  - '{name}': {type(error).__name__}: {error}"
                    for name, error in failures
                )
            ) from failures[0][1]

        reinitialized = sorted(
            name for name, plan in plans.items() if plan.reinitialized
        )
        if reinitialized:
            warnings.warn(
                "Checkpoint restored without the saved state of "
                f"{reinitialized}, which is kept as freshly built: "
                + "; ".join(
                    plans[name].reinitialized for name in reinitialized
                ),
                RuntimeWarning,
                stacklevel=4,
            )

    def _plan_restore(
            self,
            component_states: Mapping[str, Mapping[str, Any]],
            on_mismatch: str | Iterable[str],
            *,
            wired_to: frozenset[str] | None = None,
            names: Mapping[str, str] | None = None,
    ) -> dict[str, "_RestorePlan"]:
        """Check every checkpointed component and decide how to rebuild it.

        Nothing is constructed here. What cannot be built at all -- a
        component no longer registered, registered as another kind, or
        missing a prerequisite -- is always an error; a saved state this
        version cannot take is one unless `on_mismatch` lets that component
        start fresh. All problems are reported together.
        """
        reinit_all = on_mismatch == "reinit"
        if isinstance(on_mismatch, str):
            if on_mismatch not in ("raise", "reinit"):
                raise ValueError(
                    "on_mismatch must be 'raise', 'reinit' or a collection "
                    f"of component names; got {on_mismatch!r}"
                )
            reinit_names: set[str] = set()
        else:
            reinit_names = set(on_mismatch)
            unknown = sorted(reinit_names - set(component_states))
            if unknown:
                raise ValueError(
                    f"on_mismatch names {unknown}, which the checkpoint does "
                    f"not hold; it holds {sorted(component_states)}"
                )

        plans: dict[str, _RestorePlan] = {}
        problems: list[str] = []
        active = set(component_states) | set(wired_to or ())
        for name, component_info in component_states.items():
            try:
                component_class = self._restorable_class(name, component_info)
                if wired_to is not None:
                    self._check_imported_wiring(
                        name,
                        component_class,
                        component_info.get("dependencies") or {},
                    )
                # Resolved against every name in the checkpoint, not just the
                # ones rebuilt so far, so a sibling instance not yet rebuilt
                # cannot make this one look like the only one.
                dependencies = self._resource_dependencies(
                    component_class,
                    name,
                    active=active,
                    recorded=component_info.get("dependencies") or {},
                )
                for asked, target in dependencies.items():
                    if target not in active:
                        raise ValueError(
                            f"Checkpoint component '{name}' requires "
                            f"'{asked}', which resolves to '{target}', but the "
                            "checkpoint does not contain it"
                        )
            except (ValueError, ComponentDependencyError) as error:
                problems.append(str(error))
                continue

            init_args = component_info["init_args"]
            state = component_info.get("state")
            reinitialized = None
            recorded_version = component_info.get("state_version")
            current_version = component_class.state_version
            if recorded_version is not None and recorded_version != current_version:
                def described(error: Exception) -> str:
                    message = str(error)
                    if f"'{name}'" not in message:
                        message = f"Checkpoint component '{name}': {message}"
                    return message

                # The constructor first: without arguments it takes, the
                # component cannot be built at all, so a failure here is never
                # something `on_mismatch` can start fresh from.
                if recorded_version < current_version:
                    try:
                        init_args = component_class.migrate_init_args(
                            recorded_version, init_args,
                        )
                    except Exception as error:
                        problems.append(described(error))
                        continue

                # Then the state, which a fresh component can do without. A
                # newer version has nothing to migrate back; its recorded
                # constructor arguments are the best there is.
                try:
                    if recorded_version > current_version:
                        raise ValueError(
                            f"Checkpoint component '{name}' was written at "
                            f"state_version {recorded_version}, newer than "
                            f"this version of it ({current_version})"
                        )
                    if state is not None:
                        state = component_class.migrate_state(
                            recorded_version, state,
                        )
                except Exception as error:
                    if reinit_all or name in reinit_names:
                        reinitialized = described(error)
                    else:
                        problems.append(described(error))
                        continue

            if names and state is not None and reinitialized is None and (
                    issubclass(component_class, Stateful)
            ):
                # After migration: the hook only knows the current format.
                try:
                    state = component_class.rename_instances(state, names)
                except Exception as error:
                    problems.append(
                        f"Checkpoint component '{name}': renaming the "
                        f"instances its state names failed: {error}"
                    )
                    continue

            plans[name] = _RestorePlan(
                component_class=component_class,
                dependencies=dependencies,
                init_args=init_args,
                state=state,
                reinitialized=reinitialized,
            )

        if len(problems) == 1:
            raise ValueError(problems[0])
        if problems:
            raise CheckpointMismatchError(
                f"Checkpoint cannot be restored ({len(problems)} problems):\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
        return plans

    def _check_imported_wiring(
            self,
            name: str,
            component_class: type[Component],
            stored_wiring: Mapping[str, str],
    ) -> None:
        """Refuse wiring an imported component other than as it was recorded.

        An import restores what its checkpoint recorded, and the framework
        does not try to absorb code changes made since: a prerequisite the
        class declares now but the source run never wired is refused, not
        resolved here, and so is a binding of this session aimed at an
        imported component. `overwritten_dependencies` is the one change an
        import allows.
        """
        if name in self.role_bindings.instance_bindings:
            raise ComponentDependencyError(
                f"'{name}' is imported, and an imported component keeps the "
                "wiring its checkpoint recorded, so this session's bindings "
                f"cannot wire it. To give it one of this session's components "
                "instead of one the source run gave it, name that dependency "
                "in the import's `overwritten_dependencies`."
            )
        unwired = [
            edge.asked for edge in declared_edges(component_class)
            if edge.injects and edge.asked not in stored_wiring
        ]
        if unwired:
            raise ComponentDependencyError(
                f"Imported component '{name}' declares {unwired}, which the "
                "run it comes from never wired it to: "
                f"{component_class.__name__} has changed since that "
                "checkpoint was saved, and an import restores only what was "
                "recorded. Import from a checkpoint saved with this class."
            )

    def _restorable_class(
            self,
            name: str,
            component_info: Mapping[str, Any],
    ) -> type[Component]:
        implementation, _ = parse_instance_name(name)
        component_class = self.registry.get(implementation)
        if component_class is None:
            raise ValueError(f"Checkpoint component '{name}' is not registered")
        self._check_recorded_implementation(
            name,
            implementation,
            component_info,
        )
        component_type = _component_type(component_class)
        stored_type = component_info["component_type"]
        if component_type.__name__ != stored_type:
            raise ValueError(
                f"Checkpoint component '{name}' is stored as a "
                f"{stored_type}, but is now registered as a "
                f"{component_type.__name__}"
            )
        return component_class

    @staticmethod
    def _check_recorded_implementation(
            name: str,
            implementation: str,
            component_info: Mapping[str, Any],
    ) -> None:
        """Check a checkpointed instance agrees with its own name.

        An instance name states its own implementation, so parsing it is
        authoritative and a state written before instances were named needs
        no special case: its keys are names with no instance suffix.

        The recorded `implementation` is carried so that a future name which
        does *not* encode its implementation can be restored without
        migrating the state format again. Until then it is cross-checked
        rather than trusted: a recorded value disagreeing with the key means
        the state was written by something that did not share this encoding.
        """
        recorded = component_info.get("implementation")
        if recorded is not None and recorded != implementation:
            raise ValueError(
                f"Checkpoint component '{name}' records implementation "
                f"'{recorded}', which does not match its name"
            )

    def _resource_dependencies(
            self,
            component_class: type[Component],
            consumer: str,
            active: Iterable[str] | None = None,
            recorded: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Resolve a consumer's declared resources to instance names.

        Only resources are injected: `required_hooks` / `required_steps` and
        `@wraps` targets are ordering declarations, and a hook or step is not
        servable through `get_dependency`.

        `recorded` is the wiring a checkpoint says the consumer was given:
        a declared name it records is restored to the same instance rather
        than resolved again. What the class declares decides what is
        injected, so a name it no longer declares is dropped, and one it has
        declared since is resolved as usual.
        """
        recorded = recorded or {}
        return {
            edge.asked: (
                recorded[edge.asked]
                if edge.asked in recorded
                else self.resolve_dependency(
                    edge.asked,
                    consumer=consumer,
                    active=active,
                )
            )
            for edge in declared_edges(component_class)
            if edge.injects
        }

    def _construct(
            self,
            component_class: type[Component],
            instance_name: str,
            *args,
            dependencies: Mapping[str, str] | None = None,
            **kwargs,
    ) -> Component:
        """Construct a component with its declared prerequisites injected.

        `dependencies` maps each declared name to the instance name that
        satisfies it *for this consumer*. The caller resolves them, because
        only it knows the full set of components the session will end up
        holding: resolving against the components built so far would make a
        "sole instance" sole only because its sibling is not built yet.

        Construction goes through the class call, so a custom `__new__` and
        the metaclass `__call__` behave as they would anywhere else;
        `ComponentMeta.__call__` writes the prerequisites into the instance
        `__dict__` before `__init__` runs, so a constructor can use them, an
        `nn.Module` subclass needs no `nn.Module.__init__` to have run first,
        and a prerequisite module is not registered as a submodule of its
        consumer.
        """
        component = component_class(
            *args,
            **{
                _DEPENDENCIES_KEYWORD: {
                    asked: self.components[target]
                    for asked, target in (dependencies or {}).items()
                },
            },
            **kwargs,
        )
        self._stamp_identity(component, instance_name)
        return component

    @staticmethod
    def _stamp_identity(component: Component, instance_name: str) -> None:
        """Give a constructed component its own name and id."""
        component._stamp_identity(instance_name)

    def _state_restore_order(
            self,
            component_states: Mapping[str, dict[str, Any]],
    ) -> list[str]:
        if not self._any_component_attaches_components():
            # Without any component holding another, the stored order is the
            # activation order, which is already dependency-first. Keep it so
            # restoring a checkpoint whose graph no longer sorts fails where
            # it used to.
            return list(component_states)

        order = self._component_order()
        return sorted(
            component_states,
            key=lambda name: order[self.components[name].id],
        )

    def config_for_extension(self, name: str) -> dict[str, Any]:
        resolved_name = self.resolve_name(name)
        component = self.components.get(resolved_name)
        if component is None:
            raise ValueError(
                f"Component '{name}' is not active and cannot be extended"
            )
        init_args = getattr(component, "_init_args")
        args = init_args["args"]
        kwargs = init_args["kwargs"]
        if args and isinstance(args[0], Mapping):
            return dict(args[0])
        if isinstance(kwargs.get("config"), Mapping):
            return dict(kwargs["config"])
        raise ValueError(
            f"Component '{resolved_name}' was not constructed from a "
            "configuration mapping and cannot be extended"
        )

    def apply_extension_config(
            self,
            name: str,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        resolved_name = self.resolve_name(name)
        component = self.components.get(resolved_name)
        if component is None:
            raise ValueError(
                f"Component '{name}' is not active and cannot be extended"
            )
        if resolved_name in self.imported:
            # Its key would enter the stored configuration, and a fresh run
            # from that configuration would then clash with the import.
            raise ValueError(
                f"Component '{resolved_name}' is imported "
                f"({self.imported[resolved_name]}) and cannot be extended"
            )
        if not isinstance(component, ExtendableComponent):
            raise ValueError(
                f"Component '{resolved_name}' does not allow configuration "
                "changes during session extension"
            )

        effective_config = dict(config)
        component.apply_extension_config(effective_config, changed_paths)

        init_args = getattr(component, "_init_args")
        args = list(init_args["args"])
        kwargs = dict(init_args["kwargs"])
        if args and isinstance(args[0], Mapping):
            args[0] = effective_config
        elif isinstance(kwargs.get("config"), Mapping):
            kwargs["config"] = effective_config
        else:
            raise ValueError(
                f"Component '{resolved_name}' was not constructed from a "
                "configuration mapping and cannot be extended"
            )
        component._init_args = {
            "args": tuple(args),
            "kwargs": kwargs,
        }

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

    def _registered_component_class(
            self,
            name: str,
            expected_type: type[Component] | None = None,
            *,
            consumer: type[Component] | None = None,
            resolved_name: str | None = None,
    ) -> tuple[str, type[Component]]:
        # `name` is kept as asked so the diagnostics can say what was looked
        # for and which binding redirected it; only the lookup uses the
        # resolved name, which a caller may have worked out per consumer.
        if resolved_name is None:
            resolved_name = self.resolve_name(name)
        # The name identifies an instance; the class is registered under the
        # component name the instance name is built from. The two differ only
        # once a name carries an instance suffix.
        component_class = self.registry.get(
            implementation_of(resolved_name),
        )
        explanation = lambda: explain_missing_component(  # noqa: E731
            name,
            resolved_name,
            expected_type=expected_type,
            session_type=self.session_type,
            consumer=consumer,
        )
        if component_class is None:
            if expected_type is None:
                raise ValueError(with_explanation(
                    f"No step, hook or resource registered with name "
                    f"'{resolved_name}'!",
                    explanation(),
                ))
            declared_role = self.roles.get(resolved_name)
            if declared_role is not None and declared_role.category is expected_type:
                # The role message already carries its own fix; add the reason.
                reason = "\n".join(
                    line for line in explanation().splitlines()
                    if not line.lstrip().startswith(("Fix:", "Required by:"))
                )
                raise RuntimeError(with_explanation(
                    _missing_role_message(
                        category=expected_type,
                        name=name,
                        resolved_name=resolved_name,
                        declared_role=declared_role,
                        consumer=consumer,
                    ),
                    reason,
                ))
            raise RuntimeError(with_explanation(
                f"unmet prerequisite! {expected_type.__name__} '{name}' "
                f"resolves to '{resolved_name}', which is not registered as a "
                f"{expected_type.__name__}.",
                explanation(),
            ))
        if expected_type is not None and not issubclass(
                component_class,
                expected_type,
        ):
            raise RuntimeError(with_explanation(
                f"unmet prerequisite! {expected_type.__name__} '{name}' "
                f"resolves to '{resolved_name}', which is not registered as a "
                f"{expected_type.__name__}.",
                explanation(),
            ))
        return resolved_name, component_class

    def _register_component_instance(self, component: Component) -> None:
        """Register a component `_construct` built, keeping its injection.

        `_construct` already injected prerequisites resolved against every
        component the activation will hold; injecting again here would
        resolve against only those built so far.
        """
        self._add_component(
            component,
            _component_type(component),
            overwrite=True,
            inject=False,
        )

    def register_from_config(
            self,
            config: Mapping,
            *,
            default_configs: Mapping[str, Mapping] | None = None,
            view: SessionConfigView | None = None,
    ) -> None:
        """Activate what `config` configures.

        `view` is `config` already read by `read_session_config`; its
        bindings must be the ones this instance was created with.
        """
        if view is None:
            view = read_session_config(config, session_type=self.session_type)
        self.role_bindings.validate_config(view.components)

        component_configs: dict[str, dict] = {}
        configured_roots: list[str] = []
        for name, entry in view.components.items():
            check_not_imported_suffix(name, "Configured component")
            resolved_name = self.resolve_name(name)
            component_configs[resolved_name] = dict(entry.config)
            configured_roots.append(name)

        roots: list[str] = []
        for name, default_config in (default_configs or {}).items():
            roots.append(name)
            resolved_name = self.resolve_name(name)
            check_not_imported_suffix(resolved_name, "Default component")
            if resolved_name not in component_configs:
                component_configs[resolved_name] = (
                    dict(default_config)
                    if resolved_name == name
                    else {}
                )

        roots.extend(configured_roots)

        # Imported lazily: the imports module uses this one's error types.
        from training_framework.session.imports import (
            IMPORT_COMPONENTS_KEY,
            parse_imports,
            plan_import,
        )
        # An `overwritten_dependencies` target is resolved like a dependency,
        # against this session's own instances and what an earlier keyed
        # import brought in (never an `#imported` one by itself); what an
        # earlier labelled import brought in only when named by a suffixed
        # name.
        own = set(component_configs) | set(self.components)
        earlier: dict[str, str] = {}
        earlier_in_namespace: set[str] = set()
        imports = []
        specs = parse_imports(config.get(IMPORT_COMPONENTS_KEY))
        for index, spec in enumerate(specs):
            # Roles are bound in listing order, each as soon as its import
            # is planned, so a later import's `overwritten_dependencies` can
            # name an earlier import's.
            unbound = {
                other.role: other.key for other in specs[index:] if other.role
            }

            def resolve(target, spec=spec, unbound=unbound):
                if target in unbound:
                    owner = unbound[target]
                    raise ComponentDependencyError(
                        f"'{target}' is the role {owner} binds to its own "
                        "resource, so the import would be wired to itself"
                        if owner == spec.key else
                        f"'{target}' is the role {owner} binds, and imports "
                        f"are planned in the order they are listed. List "
                        f"{owner} before {spec.key}."
                    )
                return self._resolve_import_target(
                    target, own | earlier_in_namespace, earlier,
                )

            planned = plan_import(spec, self.registry, resolve)
            imports.append(planned)
            # A clash is reported before the role is bound: it is the cause
            # of what binding would otherwise refuse.
            earlier = self._check_imported_names(imports, component_configs)
            if not planned.spec.legacy:
                earlier_in_namespace.update(planned.instances)
            self._bind_import_role(planned, view.components)
        self.imported.update(earlier)
        self._check_overwritten_targets(imports)
        self._check_binding_consumers(
            set(component_configs) | set(self.imported), suffixed=True,
        )
        if imports:
            self._restore_imports(imports, component_configs)
        self._activate_all(roots, component_configs)
        self._binding_check_pending = True
        # Validate the whole graph -- requirements, wrapping, companions and
        # dataflow -- while the session is being built, which with the engine
        # is in the parent process, instead of when a worker first enters it.
        self._component_order()

    def activate_component(
            self,
            name: str,
            config: Mapping | None = None,
    ) -> str:
        """Activate a registered component and its prerequisites.

        The supported way to add a component that declares dependencies after
        the session was built: it resolves bindings, activates the dependency
        closure, and constructs each component with its prerequisites visible,
        none of which constructing an instance by hand can do.
        """
        resolved_name = self.resolve_name(name)
        if resolved_name not in self.components:
            check_not_imported_suffix(resolved_name, "Activated component")
        component_configs = (
            {resolved_name: dict(config)} if config is not None else {}
        )
        self._activate_all([name], component_configs)
        return resolved_name

    def check_role_bindings(self) -> None:
        """Refuse a per-consumer binding for a component this session does
        not hold, once, for a session built from configuration.

        Checked when the session is first entered, and by the engine before it
        starts the ranks -- after anything activated by hand -- and never on a
        restored session: a rank holds only part of the components.
        """
        if not self._binding_check_pending:
            return
        self._check_binding_consumers(set(self.components), suffixed=False)
        self._binding_check_pending = False

    def _resolved_edges(
            self,
            component: Component | type[Component],
            consumer: str,
            active: Iterable[str],
            *,
            context_keys=None,
            wiring=None,
    ) -> list[Edge]:
        """The shared edges of `component`, resolved for `consumer` here.

        `context_keys` (name -> (reads, writes)) adds the edges from the keys
        the component reads to their writers; without it only named edges
        are listed, which is all activation can know before anything exists.
        `wiring` (name -> asked -> instance) is what each component was
        given; an injected edge follows it rather than the bindings.
        """
        reads = ()
        writers = None
        if context_keys:
            reads = context_keys.get(consumer, ((), ()))[0]
            writers = writers_of(context_keys)
        return resolve_edges(
            component,
            consumer=consumer,
            bindings=self.role_bindings,
            active=active,
            context_reads=reads,
            writers=writers,
            given=(wiring or {}).get(consumer),
        )

    def _activate_all(
            self,
            roots: Iterable[str],
            component_configs: Mapping[str, Mapping],
    ) -> None:
        """Activate `roots` and everything they need, in three phases.

        Which components must be active, the order they are built in, and the
        order they run in are three different relations, and only the last
        two can have a cycle. Planning (reachability over every edge, with
        all validation), ordering (prerequisite edges only) and construction
        are therefore separate, and no constructor runs until the first two
        have succeeded.
        """
        roots = list(roots)
        plan = self._plan_activation(roots, component_configs)
        self._check_imports_reached_by_binding(plan)
        for name in self._construction_order(plan, roots):
            planned = plan[name]
            component = self._construct(
                planned.component_class,
                name,
                *planned.constructor_args,
                dependencies={
                    edge.asked: edge.target
                    for edge in planned.edges
                    if edge.injects
                },
            )
            self._register_component_instance(component)

    # -- imports ---------------------------------------------------------------

    def _check_imported_names(
            self,
            imports,
            component_configs: Mapping[str, Mapping],
    ) -> dict[str, str]:
        """Return imported instance -> its import's key, refusing a clash.

        An import keeps its source's instance names unless it has an
        `instance_name` (or, deprecated, a `suffix`), so they may meet this
        session's own instances or another import's.
        """
        imported: dict[str, str] = {}
        for planned in imports:
            for name in planned.instances:
                if name in component_configs or name in self.components:
                    other = "this session configures one"
                elif name in imported:
                    other = f"{imported[name]} imports one too"
                else:
                    imported[name] = planned.spec.key
                    continue
                rename = (
                    "a `suffix:`" if planned.spec.legacy
                    else "an `instance_name:`"
                )
                raise ValueError(
                    f"{planned.spec.key} imports '{name}', but {other}. Give "
                    f"the import {rename} so its instances get names of "
                    "their own, or, if this session's component should serve "
                    "instead, name it in the import's "
                    "`overwritten_dependencies`."
                )
        return imported

    def _bind_import_role(self, planned, configured: Mapping) -> None:
        """Bind an import's `role` to its resource, in this session's
        bindings and the session state's record of them -- not in the
        configuration. `configured` holds the configured component names."""
        role = planned.spec.role
        if role is None:
            return
        bindings = self.role_bindings.bindings
        if role in bindings:
            raise ValueError(
                f"{planned.spec.key}.role binds '{role}', which is already "
                f"bound to '{bindings[role]}'"
            )
        if role in configured:
            raise ValueError(
                f"{planned.spec.key}.role binds '{role}', which is also "
                "configured as a component"
            )
        bindings[role] = planned.root
        self.import_bindings[role] = planned.root
        self.role_bindings = RoleBindings(
            bindings,
            dependency_bindings=self.role_bindings.instance_bindings,
            session_type=self.session_type,
        )

    def _restore_imports(
            self,
            imports,
            component_configs: Mapping[str, Mapping],
    ) -> None:
        """Build what each import is wired to, then restore the import into
        this session next to it, in the order the imports are listed."""
        for planned in imports:
            wired_to = list(dict.fromkeys(
                planned.overwritten_dependencies.values()
            ))
            # Planned first, so what an import is wired to cannot need an
            # import not restored yet -- this one included: that would have
            # no construction order.
            pending = {
                instance: key for instance, key in self.imported.items()
                if instance not in self.components
            }
            self._plan_activation(
                wired_to, component_configs, pending,
                importing=planned.spec.key,
            )
            self._activate_all(wired_to, component_configs)
            self.set_state(
                planned.components_state,
                partial=True,
                wired_to=set(wired_to),
                names=planned.names,
            )

    def _resolve_import_target(
            self,
            name: str,
            own: set[str],
            imported: Mapping[str, str],
    ) -> str:
        """Resolve an `overwritten_dependencies` target of an import.

        Resolved like a dependency against `own`: this session's instances
        and what earlier keyed imports brought in. An instance any earlier
        import brought in -- `imported` -- is taken when named exactly by a
        suffixed name, which every keyed import's are. One an earlier
        labelled (deprecated) import brought in is never handed over on its
        own, as the sole instance of an implementation, and an unsuffixed
        one is refused: this session would build one of that name itself.
        """
        target = self.role_bindings.resolve(name)
        if target in imported:
            if is_instance_name(target):
                return target
            raise ComponentDependencyError(
                f"'{target}' is also an instance {imported[target]} imports "
                "under its source name, so it cannot be told whether that one "
                f"or one of this session's is meant. Give {imported[target]} "
                "a `suffix` and name the suffixed instance."
            )
        candidates = instances_of(target, own)
        if len(candidates) > 1:
            raise ComponentDependencyError(
                f"'{target}' could be any of {candidates}; name the instance "
                "that is meant."
            )
        return candidates[0] if candidates else target

    def _check_overwritten_targets(self, imports) -> None:
        """Refuse an `overwritten_dependencies` target that is not a
        resource, naming the key."""
        for planned in imports:
            for key, target in planned.overwritten_dependencies.items():
                try:
                    self._registered_component_class(target, Resource)
                except (ValueError, RuntimeError) as error:
                    raise ValueError(
                        f"{planned.spec.key}.overwritten_dependencies.{key} "
                        f"names '{target}', which cannot serve as a "
                        f"prerequisite: {error}"
                    ) from None

    @staticmethod
    def _raise_needs_import(
            chain: list[str],
            owner: str,
            importing: str | None,
    ) -> None:
        if importing is None or owner == importing:
            raise ComponentDependencyError(
                f"{owner}: '{chain[0]}' is wired to the import (by "
                "`overwritten_dependencies`), so it is built before the import, "
                f"but it needs the imported '{chain[-1]}': "
                f"{' -> '.join(chain)}. What an import is wired to cannot "
                "depend on the import."
            )
        raise ComponentDependencyError(
            f"{importing}: '{chain[0]}' is wired to the import (by "
            f"`overwritten_dependencies`), but it needs '{chain[-1]}', which {owner} "
            f"imports, and imports are restored in the order they are listed: "
            f"{' -> '.join(chain)}. List {owner} before {importing}."
        )

    def _check_binding_consumers(
            self,
            known: set[str],
            *,
            suffixed: bool,
    ) -> None:
        """Refuse a per-consumer binding for a component the session does not
        hold: it would be ignored, which a misspelt name makes likely.

        A suffixed consumer names an instance, and only a configured key or
        an import creates one, so it is checked before anything is built. An
        unsuffixed one may be activated as a prerequisite, so it is checked
        once activation is done.
        """
        for consumer in self.role_bindings.instance_bindings:
            if is_instance_name(consumer) != suffixed or consumer in known:
                continue
            implementation = implementation_of(consumer)
            siblings = sorted(
                name for name in known
                if implementation_of(name) == implementation
            )
            raise ValueError(
                f"A binding wires '{consumer}', which this session "
                "does not hold, so the binding would do nothing. Instances of "
                f"'{implementation}' here: {siblings or 'none'}."
            )

    def _check_imports_reached_by_binding(
            self,
            plan: Mapping[str, "_PlannedComponent"],
    ) -> None:
        """Refuse planned wiring that gives this session's own component an
        instance a labelled (deprecated) import brought in by anything but a
        binding that names it. A keyed import's instances are not checked:
        they resolve like the session's own.

        Checked on the plan, before anything is constructed, so a refusal
        leaves nothing behind. Resolution would otherwise hand one over on
        its own -- by exact name, or as the sole instance of an
        implementation -- and an import would quietly rewire the session.

        The import's `role` names its instance. Any other binding names one
        only by a suffixed name, which nothing but the import creates: an
        unsuffixed one is also what this session would build on its own, so
        which of the two is meant cannot be told.
        """
        # A keyed import's instances are ordinary members of the namespace;
        # only a labelled import's are held apart.
        imported = {
            instance: key for instance, key in self.imported.items()
            if instance not in self.namespace_imports
        }
        if not imported:
            return
        bindings = self.role_bindings.bindings
        instance_bindings = self.role_bindings.instance_bindings
        for name, planned in plan.items():
            for edge in planned.edges:
                instance = edge.target
                if not edge.injects or instance not in imported:
                    continue
                asked = edge.asked
                per_consumer = instance_bindings.get(name, {}).get(asked)
                if per_consumer is None and self.import_bindings.get(asked) == instance:
                    continue
                bound = per_consumer if per_consumer is not None else bindings.get(asked)
                if bound == instance and is_instance_name(instance):
                    continue
                if bound == instance:
                    raise ComponentDependencyError(
                        f"'{name}' is bound to '{instance}' for '{asked}', and "
                        f"{imported[instance]} imports an instance of that name, "
                        "so it cannot be told whether a new one or the "
                        "imported one is meant. Reach the imported one through "
                        "the import's `role`, or give the import a `suffix` "
                        "and bind the suffixed name."
                    )
                how = (
                    f"{name}: {{dependencies_role_bindings: {{{asked}: {instance}}}}}"
                    if is_instance_name(instance)
                    else "or give the import a `suffix` and bind the suffixed name"
                )
                raise ComponentDependencyError(
                    f"'{name}' asks for '{asked}' and would be given "
                    f"'{instance}', which {imported[instance]} imports, "
                    "though no binding names it. An import fills this "
                    "session's own dependencies only when a binding says so: "
                    f"the import's `role`, {how}."
                )

    def _plan_activation(
            self,
            roots: list[str],
            component_configs: Mapping[str, Mapping],
            forbidden: Mapping[str, str] | None = None,
            *,
            importing: str | None = None,
    ) -> dict[str, "_PlannedComponent"]:
        """Return every component to build, in discovery order, validated.

        Reachability over prerequisites and companions alike: a component
        reached twice is simply already planned, so no relation here can
        form a cycle. Each edge is resolved once, against every instance this
        call will hold that is known when it is resolved: the configured and
        root names from the start (the only way an instance name enters the
        session), plus unsuffixed names as they are discovered. The ordering
        and construction phases reuse these resolutions rather than resolving
        again.
        """
        known = (
            set(component_configs)
            | {self.resolve_name(root) for root in roots}
            | set(self.components)
        )
        plan: dict[str, _PlannedComponent] = {}

        def visit(
                name: str,
                component_class: type[Component],
                path: tuple[str, ...] = (),
        ) -> None:
            if name in self.components or name in plan:
                return
            edges = [
                edge for edge in self._resolved_edges(component_class, name, known)
                if edge.activates
            ]
            plan[name] = _PlannedComponent(
                component_class=component_class,
                constructor_args=self._constructor_args(
                    name, component_class, component_configs,
                ),
                edges=edges,
            )
            for edge in edges:
                _, target_class = self._registered_component_class(
                    edge.asked,
                    edge.expected_type,
                    consumer=component_class,
                    resolved_name=edge.target,
                )
                if edge.target in forbidden:
                    self._raise_needs_import(
                        [*path, name, edge.target], forbidden[edge.target],
                        importing,
                    )
                known.add(edge.target)
                visit(edge.target, target_class, (*path, name))

        forbidden = forbidden or {}
        for root in roots:
            # A configured name says which instance to create, so it is taken
            # literally. Only a *dependency* is resolved to an instance.
            resolved, root_class = self._registered_component_class(root)
            if resolved in forbidden:
                self._raise_needs_import(
                    [resolved], forbidden[resolved], importing,
                )
            visit(resolved, root_class)

        # Checked against everything the session will hold -- what this call
        # plans, companions included, and what an earlier call added -- or a
        # second activate_component() would slip a second instance past it.
        self._check_instance_limits(set(plan) | set(self.components))
        return plan

    def _constructor_args(
            self,
            name: str,
            component_class: type[Component],
            component_configs: Mapping[str, Mapping],
    ) -> tuple:
        """Return the arguments `name` is constructed with, or say why none.

        A configured component gets its mapping. An unconfigured one is built
        only when that needs no decision: its constructor is the inherited
        one, or its `config_schema` gives every field a default, in which
        case it gets `{}` so it holds a configuration mapping like any
        configured component (and so can be extended later).
        """
        if name in component_configs:
            return (component_configs[name],)
        if is_instance_name(name):
            # Only a configured key declares an instance. Creating one because
            # something is wired to it would turn a mistyped suffix into a
            # fresh, unconfigured instance -- a run that works and is quietly
            # wrong.
            configured = sorted(
                configured_name
                for configured_name in {*component_configs, *self.components}
                if implementation_of(configured_name) == implementation_of(name)
            )
            raise ComponentDependencyError(
                f"Component instance '{name}' is not configured in this "
                "session, so nothing can be wired to it. An instance is "
                f"created only by a top-level '{name}' key. Configured "
                f"instances of '{implementation_of(name)}': "
                f"{configured or 'none'}."
            )
        if has_all_defaults(component_class.config_schema):
            return ({},)
        if component_class.__init__ is Component.__init__:
            return ()
        imported = sorted(
            instance for instance in self.components
            if implementation_of(instance) == name
            and has_imported_suffix(instance)
        )
        raise RuntimeError(
            f"Component '{name}' is required but defines a custom "
            "constructor. Add a top-level component mapping for "
            f"'{name}'."
            + (
                f" Imported without `instance_name`, {imported} fill a "
                "dependency only when named: bind one to use it."
                if imported else ""
            )
        )

    def _construction_order(
            self,
            plan: Mapping[str, "_PlannedComponent"],
            roots: list[str],
    ) -> list[str]:
        """Return the planned names prerequisite-first, or report a cycle.

        Depth-first over prerequisite edges only; a companion is never a
        prerequisite, so a companion that requires the component activating
        it is not a cycle. Walked from the roots in configured order and then
        from every planned name in discovery order, which is how a companion
        is reached. Without companions this is exactly the order activation
        has always constructed in, so seeded constructors draw the same
        numbers.
        """
        order: list[str] = []
        done: set[str] = set(self.components)
        visiting: list[str] = []

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                # Components are wired as they are constructed, so a cycle has
                # no valid construction order. Report it here, where the chain
                # that closed it is still known.
                chain = " -> ".join([*visiting, name])
                raise RuntimeError(
                    f"Cyclic dependency detected in the component graph! {chain}"
                )
            visiting.append(name)
            try:
                for edge in plan[name].edges:
                    if edge.builds_first:
                        visit(edge.target)
            finally:
                visiting.pop()
            done.add(name)
            order.append(name)

        starts = [self.resolve_name(root) for root in roots]
        for name in [*starts, *plan]:
            visit(name)
        return order

    def _check_instance_limits(self, planned: Iterable[str]) -> None:
        """Reject a second instance of a component that must stay unique.

        Checked over everything the session will hold rather than as each is
        constructed, so the error does not depend on activation order and can
        name every instance involved.
        """
        by_implementation: dict[str, set[str]] = {}
        for name in planned:
            implementation = implementation_of(self.resolve_name(name))
            by_implementation.setdefault(implementation, set()).add(name)

        for implementation, names in sorted(by_implementation.items()):
            if len(names) < 2:
                continue
            component_class = self.registry.get(implementation)
            # An unregistered name is not this check's problem; activation
            # reports it with the diagnostics that explain why.
            if component_class is None:
                continue
            if not getattr(component_class, "singleton", False):
                continue
            raise ValueError(
                f"Component '{implementation}' allows only one instance per "
                f"session, but {sorted(names)} are configured. It is marked "
                "@singleton because a second instance could not work "
                "alongside the first."
            )

    def dependency_closure(
            self,
            names: Iterable[str],
            *,
            active_names: Iterable[str] | None = None,
            context_keys=None,
            wiring=None,
    ) -> set[str]:
        """Return `names` plus everything they need active, transitively.

        Follows every edge the shared model lists -- prerequisites,
        `@activates` companions, and the writers of the keys a component
        reads -- so a rank keeps everything its components need to run.
        `active_names` lets a caller resolve the closure
        before any component is constructed -- the graph is class-level, so
        the worker can decide what a rank needs without building the session
        first. `context_keys` (name -> (reads, writes)) supplies the keys
        when the components are not live -- a worker deciding from a
        checkpoint's record; with live components they are read from them.
        `wiring` likewise gives what each component was handed, so an
        injected edge leads to the instance it holds; with live components
        it is read from them.
        """
        active = (
            set(self.components)
            if active_names is None
            else set(active_names)
        )
        if context_keys is None:
            context_keys = context_keys_of(
                component for name, component in self.components.items()
                if name in active
            )
        if wiring is None:
            wiring = wiring_of(
                component for name, component in self.components.items()
                if name in active
            )
        closure: set[str] = set()

        def visit(name: str) -> None:
            resolved_name, component_class = self._registered_component_class(
                self.resolve_dependency(name, active=active),
            )
            if resolved_name not in active:
                raise RuntimeError(with_explanation(
                    f"Component '{name}' resolves to '{resolved_name}', which "
                    "is not configured in this session.",
                    explain_missing_component(
                        name,
                        resolved_name,
                        session_type=self.session_type,
                        active_names=active,
                    ),
                ))
            if resolved_name in closure:
                return

            closure.add(resolved_name)
            for edge in self._resolved_edges(
                    component_class, resolved_name, active,
                    context_keys=context_keys,
                    wiring=wiring,
            ):
                if not edge.kept_with_source or edge.target is None:
                    # A read nobody writes is the sort's error to report.
                    continue
                self._registered_component_class(
                    edge.asked,
                    edge.expected_type,
                    consumer=component_class,
                    resolved_name=edge.target,
                )
                visit(edge.target)

        for name in names:
            visit(name)
        return closure

    def _names_depending_on(
            self,
            target_name: str,
            active: set[str],
            context_keys=None,
            wiring=None,
    ) -> set[str]:
        """Return the active components that reach `target_name` transitively."""
        dependents = set()
        for name in active:
            try:
                closure = self.dependency_closure(
                    [name], active_names=active, context_keys=context_keys,
                    wiring=wiring,
                )
            except (RuntimeError, KeyError):
                # A component whose graph cannot be resolved is not this
                # diagnostic's problem; the activation path reports it.
                continue
            if name != target_name and target_name in closure:
                dependents.add(name)
        return dependents

    def validate_component_names(
            self,
            names: Iterable[str] | None,
            *,
            source: str,
            active_names: Iterable[str] | None = None,
    ) -> set[str]:
        """Resolve configured component names, or say which one is wrong.

        `source` names the configuration key being checked, so a typo or a
        component belonging to another session is reported against the line
        that wrote it instead of silently doing nothing.
        """
        active = (
            set(self.components)
            if active_names is None
            else set(active_names)
        )
        resolved_names = set()
        for name in names or ():
            resolved_name, _ = self._registered_component_class(name)
            if resolved_name not in active:
                raise RuntimeError(with_explanation(
                    f"{source} names '{name}', which resolves to "
                    f"'{resolved_name}' and is not configured in this "
                    "session.",
                    explain_missing_component(
                        name,
                        resolved_name,
                        session_type=self.session_type,
                        active_names=active,
                    ),
                ))
            resolved_names.add(resolved_name)
        return resolved_names

    def rank_zero_component_names(
            self,
            *,
            active_names: Iterable[str] | None = None,
            declared: Iterable[str] | None = None,
            context_keys=None,
            wiring=None,
    ) -> set[str]:
        """Return the active components a secondary rank does not build.

        A component is rank-zero-only when its class is marked with
        `@rank_zero_only` or when this session names it in
        `ddp.rank_zero_components` (`declared`). The DDP resource itself is
        never rank-zero-only.
        """
        active = (
            set(self.components)
            if active_names is None
            else set(active_names)
        )
        ddp_name = self.resolve_name("ddp")
        rank_zero, declared_names = self._rank_zero_names(active, declared)

        # Only the names this session wrote down are questioned. A class-level
        # @rank_zero_only is its author's settled decision -- a reporter may
        # require something that requires `ddp` and still be rank-zero-only
        # on purpose -- while a config entry is a per-run override worth a
        # second look, because excluding a participant in the collectives is
        # what leaves the other ranks waiting.
        using_ddp = sorted(
            name for name in declared_names - {ddp_name}
            if ddp_name in self.dependency_closure(
                [name], active_names=active, context_keys=context_keys,
                wiring=wiring,
            )
        )
        if using_ddp:
            warnings.warn(
                "ddp.rank_zero_components excludes components that require "
                f"the DDP resource: {using_ddp}. They are built on rank 0 "
                "only, so a collective they take part in can hang. Mark them "
                "@rank_zero_only if they really are rank-zero-only work.",
                RuntimeWarning,
                stacklevel=2,
            )
        return rank_zero

    def _rank_zero_names(
            self,
            active: set[str],
            declared: Iterable[str] | None,
    ) -> tuple[set[str], set[str]]:
        """The rank-zero-only names among `active`, and those of them the
        session declared in `ddp.rank_zero_components`."""
        rank_zero = {
            name for name in active
            if getattr(self._registered_component_class(name)[1],
                       "rank_zero_only", False)
        }
        declared_names = self.validate_component_names(
            declared,
            source="ddp.rank_zero_components",
            active_names=active,
        )
        rank_zero |= declared_names
        rank_zero.discard(self.resolve_name("ddp"))
        return rank_zero, declared_names

    def check_rank_zero_dependants(
            self,
            *,
            rank_zero_components: Iterable[str] | None = None,
    ) -> None:
        """Refuse a component that runs on every rank but depends on a
        rank-zero-only one.

        `rank_parallel_names` does this while planning the secondary ranks;
        this is the same check on its own, for a single-rank launch, which
        has no ranks to plan for but would otherwise carry the mistake until
        the same configuration is run on more than one.
        """
        active = set(self.components)
        context_keys, wiring = self._live_edge_inputs(active)
        rank_zero, declared_names = self._rank_zero_names(
            active, rank_zero_components,
        )
        self._raise_for_rank_zero_dependants(
            active, rank_zero, declared_names, context_keys, wiring,
        )

    def _live_edge_inputs(self, active: set[str]):
        """The context keys and wiring of the live components in `active`."""
        live = [
            component for name, component in self.components.items()
            if name in active
        ]
        return context_keys_of(live), wiring_of(live)

    def _raise_for_rank_zero_dependants(
            self,
            active: set[str],
            rank_zero: set[str],
            declared_names: set[str],
            context_keys,
            wiring,
    ) -> None:
        """Raise when a component outside `rank_zero` has an edge into it.

        A secondary rank does not build a rank-zero-only component, so
        whatever needs it -- requires, wraps, brings along with
        `@activates`, or reads a key it writes -- cannot run there. Building
        it on every rank anyway would override its declaration, so the
        dependant has to be declared rank-zero-only too, or the edge removed.
        Only direct edges are listed: fixing those fixes the rest.
        """
        found = []
        for name in sorted(active - rank_zero):
            _, component_class = self._registered_component_class(name)
            for edge in self._resolved_edges(
                    component_class, name, active,
                    context_keys=context_keys,
                    wiring=wiring,
            ):
                if not edge.kept_with_source or edge.target not in rank_zero:
                    continue
                declared_by = (
                    "ddp.rank_zero_components"
                    if edge.target in declared_names
                    else "@rank_zero_only"
                )
                found.append(
                    f"'{name}' {_edge_description(edge)} ({declared_by})"
                )
        if found:
            raise RuntimeError(
                "Components that run on every rank depend on rank-zero-only "
                "components, which the other ranks do not build: "
                + "; ".join(found)
                + ". Declare each dependant rank-zero-only too "
                "(@rank_zero_only or ddp.rank_zero_components), or remove "
                "the dependency -- for instance, write the value to "
                "iteration_context and let a rank-zero-only hook read it."
            )

    def rank_parallel_names(
            self,
            *,
            active_names: Iterable[str] | None = None,
            parallel_components: Iterable[str] | None = None,
            rank_zero_components: Iterable[str] | None = None,
            context_keys=None,
            wiring=None,
    ) -> set[str]:
        """Return the component names a secondary rank builds.

        Every configured component is kept except those declared
        rank-zero-only. Nothing else is inferred: a component is dropped
        because it was declared rank-zero-only, never because the framework
        decided it was only needed by one. Leaving a component out of a rank
        is what deadlocks a collective, so the mistake worth avoiding is
        dropping too much, and a resource kept for nothing costs one
        constructor call.

        A rank-zero-only component that a kept one depends on is an error:
        a prerequisite has to exist wherever its consumer does, and building
        it there anyway would override its declaration.

        `parallel_components` is the deprecated opt-in list. When a session
        provides it (even empty) it decides the answer on its own: only those
        roots, their closure, and the DDP resource are kept.

        What a component needs is every edge of the shared model, the writers
        of the keys it reads included. `context_keys` gives those keys when
        the components are not live -- a worker deciding from a checkpoint's
        record, and `wiring` what each component was given (see
        `dependency_closure`). With live components the reduced set is also
        sorted before it
        is returned, so a rank that could not run is rejected here, in the
        parent, rather than in a worker the others are waiting for.
        """
        active = (
            set(self.components)
            if active_names is None
            else set(active_names)
        )
        if context_keys is None or wiring is None:
            live_keys, live_wiring = self._live_edge_inputs(active)
            context_keys = live_keys if context_keys is None else context_keys
            wiring = live_wiring if wiring is None else wiring
        keep = self._rank_names(
            active, parallel_components, rank_zero_components, context_keys,
            wiring,
        )
        self._check_rank_graph(keep)
        return keep

    def _rank_names(
            self,
            active: set[str],
            parallel_components,
            rank_zero_components,
            context_keys,
            wiring,
    ) -> set[str]:
        ddp_name = self.resolve_name("ddp")

        if parallel_components is not None:
            keep = self.dependency_closure(
                list(parallel_components) + ["ddp"],
                active_names=active,
                context_keys=context_keys,
                wiring=wiring,
            )
            pruned = sorted(
                self._names_depending_on(
                    ddp_name, active, context_keys, wiring,
                ) - keep
            )
            if pruned:
                # The classic mistake the opt-in list invites: a component
                # that takes part in the collectives is simply forgotten, and
                # the run hangs instead of failing.
                warnings.warn(
                    "ddp.parallel_components omits components that require "
                    f"the DDP resource: {pruned}. They are built on rank 0 "
                    "only, so a collective they take part in can hang. Add "
                    "them to the list, or remove parallel_components and let "
                    "the framework decide.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return keep

        rank_zero = self.rank_zero_component_names(
            active_names=active,
            declared=rank_zero_components,
            context_keys=context_keys,
            wiring=wiring,
        )
        _, declared_names = self._rank_zero_names(active, rank_zero_components)
        self._raise_for_rank_zero_dependants(
            active, rank_zero, declared_names, context_keys, wiring,
        )
        keep = self.dependency_closure(
            active - rank_zero,
            active_names=active,
            context_keys=context_keys,
            wiring=wiring,
        )
        keep.add(ddp_name)
        return keep

    def _check_rank_graph(self, keep: set[str]) -> None:
        """Sort a rank's reduced component set when it can be: every name
        live, which is the parent planning a launch. A worker holds no live
        components when it prunes, and applies a plan checked here."""
        if not keep <= set(self.components):
            return
        try:
            topological_sort_of_components(
                self.role_bindings,
                components=[self.components[name] for name in sorted(keep)],
                session_type=self.session_type,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"A secondary rank would build {sorted(keep)}, which cannot "
                f"run on its own: {error}"
            ) from error

    def register_resource(self, component: Resource, overwrite=False) -> str:
        return self._add_component(
            component, Resource, overwrite=overwrite, by_hand=True,
        )

    def register_hook(self, component: Hook, overwrite=False) -> str:
        return self._add_component(
            component, Hook, overwrite=overwrite, by_hand=True,
        )

    def add_step(self, component: Step, overwrite=False) -> str:
        return self._add_component(
            component, Step, overwrite=overwrite, by_hand=True,
        )

    def _add_component(
            self,
            component: Component,
            base_type: type[Component],
            *,
            overwrite: bool,
            inject: bool = True,
            by_hand: bool = False,
    ) -> str:
        self._validate_component(component, base_type, overwrite=overwrite)
        if by_hand:
            check_not_imported_suffix(component.name, "Registered component")
        existing = self.components.get(component.name)
        if existing is not None and existing is not component:
            self._refuse_if_held(existing, "replace")
        if inject:
            self._check_instance_limits(
                set(self.components) | {component.name},
            )
            self._inject_dependencies(component)
            self._repoint_consumers(component)
        self.components[component.name] = component
        return component.name

    def _holders_of(self, instance: Component) -> list[tuple[str, str]]:
        """Return `(consumer, asked name)` for every consumer that has been
        handed `instance` -- in `__init__`, `setup` or later."""
        return sorted(
            (consumer.name, asked)
            for consumer in self.components.values()
            if consumer is not instance
            for asked, held in consumer._linked_components.items()
            if held is instance
        )

    def _refuse_if_held(self, instance: Component, action: str) -> None:
        """Refuse to take out an instance a consumer already holds.

        A consumer may have kept the reference wherever it liked -- an
        attribute, a container module -- and nothing can reach it there. Taking
        the instance out would leave that consumer using a component the
        session no longer sets up, tears down or checkpoints, and, after a
        replacement, holding one instance while `get_dependency` returns
        another.
        """
        holders = self._holders_of(instance)
        if holders:
            held_by = ", ".join(
                f"'{consumer}' (as '{asked}')" for consumer, asked in holders
            )
            raise ValueError(
                f"Cannot {action} '{instance.name}': it has already been "
                f"handed to {held_by}. Replace or remove a component before "
                "anything takes it, or rebuild the session."
            )

    def _repoint_consumers(self, replacement: Component) -> None:
        """Give consumers wired to `replacement`'s name the new instance.

        Only reached when none of them has been handed the instance being
        replaced (see `_refuse_if_held`), so there is no saved reference that
        could disagree with what `get_dependency` returns from here on. This
        also restores a prerequisite dropped by an earlier removal.
        """
        active = set(self.components) | {replacement.name}
        for consumer in self.components.values():
            if consumer is replacement:
                continue
            injected = _given_prerequisites(consumer)
            if injected is None:
                continue
            if consumer.name in self.imported:
                # An imported component keeps the wiring its checkpoint
                # recorded: only the instance it holds may be replaced, by
                # one of the same name. Bindings do not wire it.
                for asked, held in list(injected.items()):
                    if getattr(held, "name", None) == replacement.name:
                        injected[asked] = replacement
                continue
            for edge in declared_edges(type(consumer)):
                if not edge.injects:
                    continue
                asked = edge.asked
                try:
                    target = self.resolve_dependency(
                        asked,
                        consumer=consumer.name,
                        active=active,
                    )
                except ComponentDependencyError:
                    continue
                if target == replacement.name:
                    injected[asked] = replacement

    def _inject_dependencies(self, component: Component) -> None:
        """Give a component built outside the session its prerequisites.

        A component reaches the session this way when it was constructed by
        hand, or rebuilt by unpickling it on its own -- a `Stateful`
        component replays `__init__` from its constructor arguments, which
        leaves nothing injected. Whatever it carries is replaced rather than
        kept: references from another session would point at components this
        one does not hold.

        Available from `setup` onwards. A component that needs a prerequisite
        in `__init__` must still be activated by the session. The record of
        what it asked for starts over with them.
        """
        _give_prerequisites(component, {
            edge.asked: self.get_resource(edge.asked, consumer=component.name)
            for edge in declared_edges(type(component))
            if edge.injects
        })

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

        registered_name = (
            implementation_of(component.name)
            if hasattr(component, "name")
            else None
        )
        if registered_name is None or registered_name not in self.registry:
            raise ValueError(
                f"{base_type.__name__} '{type(component).__name__}' "
                "is not registered as a component!"
            )
        if _component_type(self.registry[registered_name]) is not base_type:
            raise ValueError(
                f"Component '{component.name}' is not registered as a "
                f"{base_type.__name__}!"
            )

        if self.role_bindings.is_bound(component.name):
            raise ValueError(
                f"Component role '{component.name}' is bound to "
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
        # The name may identify an instance; the registry holds its class.
        registered_class = self.registry.get(implementation_of(registered_name))
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
        self._refuse_if_held(component, "remove")
        del self.components[registered_name]
        # Nobody holds it, so dropping it from the consumers' prerequisites is
        # all it takes for get_dependency to stop handing out an object the
        # session no longer owns. A later registration under the name puts
        # it back.
        for consumer in self.components.values():
            injected = _given_prerequisites(consumer)
            if not injected:
                continue
            for asked in [
                asked for asked, held in injected.items() if held is component
            ]:
                del injected[asked]

    def get_resource(self, name: str, *, consumer: str | None = None) -> Resource:
        registered_name = self.resolve_dependency(name, consumer=consumer)
        component = self.components.get(registered_name)
        if not isinstance(component, Resource):
            active_names = {
                active_name
                for active_name, active in self.components.items()
                if isinstance(active, Resource)
            }
            raise ComponentNotFoundError(with_explanation(
                f"{name} not found in resources!",
                explain_missing_component(
                    name,
                    registered_name,
                    expected_type=Resource,
                    session_type=self.session_type,
                    active_names=active_names,
                ),
            ))
        return component

    def has_resource(self, name: str, *, consumer: str | None = None) -> bool:
        try:
            registered_name = self.resolve_dependency(name, consumer=consumer)
        except ComponentDependencyError:
            # Several instances answer to the name. Which one is meant is
            # undecided, but the question this answers -- is there such a
            # resource -- is still yes; asking for it is what reports it.
            return True
        component = self.components.get(registered_name)
        return isinstance(component, Resource)

    def resolve_name(self, name: str) -> str:
        """Return the name `name` is bound to, applying bindings only."""
        return self.role_bindings.resolve(name)

    def resolve_dependency(
            self,
            name: str,
            *,
            consumer: str | None = None,
            active: Iterable[str] | None = None,
    ) -> str:
        """Return the instance name that satisfies `name` for `consumer`.

        The rule is `resolve_component_name`, shared with the sort, against
        the components this session holds unless `active` says otherwise.
        """
        return resolve_component_name(
            self.role_bindings,
            name,
            consumer=consumer,
            active=self.components if active is None else active,
        )

    @property
    def bindings(self) -> dict[str, str]:
        return self.role_bindings.bindings

    @property
    def aliases(self) -> RoleBindings:
        warnings.warn(
            "SessionComponents.aliases is deprecated; use role_bindings",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.role_bindings

    @property
    def component_bindings(self) -> RoleBindings:
        warnings.warn(
            "SessionComponents.component_bindings is deprecated; use "
            "role_bindings",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.role_bindings

    @property
    def alias_bindings(self) -> dict[str, str]:
        warnings.warn(
            "SessionComponents.alias_bindings is deprecated; use bindings",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bindings

    def _component_order(self) -> dict[str, int]:
        return topological_sort_of_components(
            self.role_bindings,
            components=self.components.values(),
            session_type=self.session_type,
        )

    def _any_component_attaches_components(self) -> bool:
        # What a component was given, not what it has asked for yet: this
        # decides restore order before `setup` has run, and a component may
        # take its prerequisite in `setup` -- or in `set_state` itself.
        return any(
            component._dependencies
            for component in self.components.values()
        )

    @property
    def ordered_components(self) -> list[Component]:
        order = self._component_order()
        return sorted(
            self.components.values(),
            key=lambda component: order[component.id],
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
