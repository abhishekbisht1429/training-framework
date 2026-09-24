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
from training_framework.components.base import _DEPENDENCIES_KEYWORD
from training_framework.components.config_schema import has_all_defaults
from training_framework.components.edges import (
    Edge,
    context_keys_of,
    declared_edges,
    resolve_component_name,
    resolve_edges,
    writers_of,
)
from training_framework.components.config import (
    reject_legacy_components_entry,
    reserved_config_names,
)
from training_framework.components.diagnostics import (
    explain_missing_component,
    with_explanation,
)
from training_framework.components.naming import (
    implementation_of,
    is_instance_name,
    parse_instance_name,
)
from training_framework.components.registry import (
    ComponentBindings,
    _coalesce_component_bindings,
    _component_type,
    _missing_role_message,
    component_registry,
    role_registry,
    topological_sort_of_components,
)
from training_framework.session.config import TRAINING_SESSION_TYPE, normalize_session_type


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
            component_bindings: Mapping[str, str] | None = None,
            session_type: str = TRAINING_SESSION_TYPE,
            aliases: Mapping[str, str] | None = None,
    ):
        self.session_type = normalize_session_type(session_type)
        self.registry = component_registry(self.session_type)
        self.roles = role_registry(self.session_type)
        self.components: dict[str, Component] = {}
        self._merge_components(resources, Resource)
        self._merge_components(hooks, Hook)
        self._merge_components(steps, Step)
        component_bindings = _coalesce_component_bindings(
            component_bindings,
            aliases,
        )
        self.component_bindings = ComponentBindings(
            component_bindings,
            session_type=self.session_type,
        )

    def __setstate__(self, state) -> None:
        legacy_bindings = state.pop("aliases", None)
        if "component_bindings" not in state and legacy_bindings is not None:
            state["component_bindings"] = legacy_bindings
        state.pop("_links_dirty", None)
        self.__dict__.update(state)

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
                "context_reads": list(component.context_reads()),
                "context_writes": list(component.context_writes()),
                # The wiring this instance was actually given, so a restore
                # rebuilds it as it was rather than re-deriving it, and one
                # component can be restored with only what it was given.
                "dependencies": {
                    asked: getattr(dependency, "name", type(dependency).__name__)
                    for asked, dependency in component._dependencies.items()
                },
                "state_version": type(component).state_version,
            }
            for name, component in self.components.items()
        }

    def set_state(
            self,
            component_states: dict[str, dict[str, Any]],
            *,
            on_mismatch: str | Iterable[str] = "raise",
            partial: bool = False,
    ) -> None:
        """Rebuild the components a state holds and restore their state.

        `on_mismatch` decides what happens to a component whose saved state
        this version cannot take -- written by a newer `state_version`, or an
        older one with no `migrate_state`: "raise" (the default) refuses the
        whole restore; "reinit" keeps every such component as freshly built,
        with a warning naming them; a collection of instance names does that
        for those only. Whatever cannot be built at all always raises.

        `partial` says the state holds a chosen subset of a session's
        components (each with what it depends on), not a whole session.
        """
        restored_components: dict[str, Component] = {}
        # Components are rebuilt into a fresh mapping, but a component being
        # constructed must see the ones already rebuilt, so the view reads
        # through to it. The previous mapping is put back if the restore fails.
        previous_components = self.components
        self.components = restored_components
        try:
            self._restore_components(
                component_states,
                restored_components,
                on_mismatch=on_mismatch,
                partial=partial,
            )
        except BaseException:
            self.components = previous_components
            raise

    def _restore_components(
            self,
            component_states: dict[str, dict[str, Any]],
            restored_components: dict[str, Component],
            *,
            on_mismatch: str | Iterable[str] = "raise",
            partial: bool = False,
    ) -> None:
        # A checkpoint is not a trusted plan: it may predate a component being
        # marked @singleton, or have been edited. Checked before anything is
        # built; set_state restores the previous components if it raises.
        self._check_instance_limits(component_states)

        # Every component is checked before any is built, and every problem
        # is reported at once rather than the first one only.
        plans = self._plan_restore(component_states, on_mismatch)

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
        active = set(component_states)
        for name, component_info in component_states.items():
            try:
                component_class = self._restorable_class(name, component_info)
                # Resolved against every name in the checkpoint, not just the
                # ones rebuilt so far, so a sibling instance not yet rebuilt
                # cannot make this one look like the only one.
                dependencies = self._resource_dependencies(
                    component_class,
                    name,
                    active=active,
                    recorded=component_info.get("dependencies"),
                )
                for asked, target in dependencies.items():
                    if target not in component_states:
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
                try:
                    if recorded_version > current_version:
                        raise ValueError(
                            f"Checkpoint component '{name}' was written at "
                            f"state_version {recorded_version}, newer than "
                            f"this version of it ({current_version})"
                        )
                    init_args = component_class.migrate_init_args(
                        recorded_version, init_args,
                    )
                    if state is not None:
                        state = component_class.migrate_state(
                            recorded_version, state,
                        )
                except Exception as error:
                    message = str(error)
                    if f"'{name}'" not in message:
                        message = f"Checkpoint component '{name}': {message}"
                    if reinit_all or name in reinit_names:
                        reinitialized = message
                        # Rebuilt from what the checkpoint recorded, as far as
                        # the constructor still takes it.
                        init_args = component_info["init_args"]
                    else:
                        problems.append(message)
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
    ) -> None:
        reject_legacy_components_entry(config)
        self.component_bindings.validate_config(config)
        reserved_names = reserved_config_names(self.session_type)

        component_configs: dict[str, dict] = {}
        configured_roots: list[str] = []
        for name, component_config in config.items():
            if name in reserved_names:
                continue
            if not isinstance(component_config, Mapping):
                raise ValueError(
                    f"The value corresponding to the key '{name}' is not a mapping"
                )
            resolved_name = self.resolve_name(name)
            component_configs[resolved_name] = dict(component_config)
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

        roots.extend(configured_roots)
        self._activate_all(roots, component_configs)
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
        component_configs = (
            {resolved_name: dict(config)} if config is not None else {}
        )
        self._activate_all([name], component_configs)
        return resolved_name

    def _resolved_edges(
            self,
            component: Component | type[Component],
            consumer: str,
            active: Iterable[str],
            *,
            context_keys=None,
    ) -> list[Edge]:
        """The shared edges of `component`, resolved for `consumer` here.

        `context_keys` (name -> (reads, writes)) adds the edges from the keys
        the component reads to their writers; without it only named edges
        are listed, which is all activation can know before anything exists.
        """
        reads = ()
        writers = None
        if context_keys:
            reads = context_keys.get(consumer, ((), ()))[0]
            writers = writers_of(context_keys)
        return resolve_edges(
            component,
            consumer=consumer,
            bindings=self.component_bindings,
            active=active,
            context_reads=reads,
            writers=writers,
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

    def _plan_activation(
            self,
            roots: list[str],
            component_configs: Mapping[str, Mapping],
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

        def visit(name: str, component_class: type[Component]) -> None:
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
                known.add(edge.target)
                visit(edge.target, target_class)

        for root in roots:
            # A configured name says which instance to create, so it is taken
            # literally. Only a *dependency* is resolved to an instance.
            visit(*self._registered_component_class(root))

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
        raise RuntimeError(
            f"Component '{name}' is required but defines a custom "
            "constructor. Add a top-level component mapping for "
            f"'{name}'."
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
    ) -> set[str]:
        """Return the active components that reach `target_name` transitively."""
        dependents = set()
        for name in active:
            try:
                closure = self.dependency_closure(
                    [name], active_names=active, context_keys=context_keys,
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
        rank_zero = {
            name for name in active
            if getattr(self._registered_component_class(name)[1],
                       "rank_zero_only", False)
        }
        ddp_name = self.resolve_name("ddp")
        declared_names = self.validate_component_names(
            declared,
            source="ddp.rank_zero_components",
            active_names=active,
        )
        rank_zero |= declared_names
        rank_zero.discard(ddp_name)

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

    def rank_parallel_names(
            self,
            *,
            active_names: Iterable[str] | None = None,
            parallel_components: Iterable[str] | None = None,
            rank_zero_components: Iterable[str] | None = None,
            context_keys=None,
    ) -> set[str]:
        """Return the component names a secondary rank builds.

        Every configured component is kept except those declared
        rank-zero-only. Nothing else is inferred: a component is dropped
        because it was declared rank-zero-only, never because the framework
        decided it was only needed by one. Leaving a component out of a rank
        is what deadlocks a collective, so the mistake worth avoiding is
        dropping too much, and a resource kept for nothing costs one
        constructor call.

        A rank-zero-only component that a kept one depends on is kept anyway,
        with a warning: a prerequisite has to exist wherever its consumer does.

        `parallel_components` is the deprecated opt-in list. When a session
        provides it (even empty) it decides the answer on its own: only those
        roots, their closure, and the DDP resource are kept.

        What a component needs is every edge of the shared model, the writers
        of the keys it reads included. `context_keys` gives those keys when
        the components are not live -- a worker deciding from a checkpoint's
        record. With live components the reduced set is also sorted before it
        is returned, so a rank that could not run is rejected here, in the
        parent, rather than in a worker the others are waiting for.
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
        keep = self._rank_names(
            active, parallel_components, rank_zero_components, context_keys,
        )
        self._check_rank_graph(keep)
        return keep

    def _rank_names(
            self,
            active: set[str],
            parallel_components,
            rank_zero_components,
            context_keys,
    ) -> set[str]:
        ddp_name = self.resolve_name("ddp")

        if parallel_components is not None:
            keep = self.dependency_closure(
                list(parallel_components) + ["ddp"],
                active_names=active,
                context_keys=context_keys,
            )
            pruned = sorted(
                self._names_depending_on(ddp_name, active, context_keys) - keep
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
        )
        keep = self.dependency_closure(
            active - rank_zero,
            active_names=active,
            context_keys=context_keys,
        )
        keep.add(ddp_name)

        still_needed = sorted(rank_zero & keep)
        if still_needed:
            # Correctness wins over pruning: a prerequisite of a component
            # this rank runs has to exist, whatever it is marked.
            warnings.warn(
                "Rank-zero-only components are built on every rank because "
                "components this rank runs depend on them: "
                f"{still_needed}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return keep

    def _check_rank_graph(self, keep: set[str]) -> None:
        """Sort a rank's reduced component set when it can be: every name
        live, which is the parent planning a launch. A worker holds no live
        components when it prunes, and applies a plan checked here."""
        if not keep <= set(self.components):
            return
        try:
            topological_sort_of_components(
                self.component_bindings,
                components=[self.components[name] for name in sorted(keep)],
                session_type=self.session_type,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"A secondary rank would build {sorted(keep)}, which cannot "
                f"run on its own: {error}"
            ) from error

    def register_resource(self, component: Resource, overwrite=False) -> str:
        return self._add_component(component, Resource, overwrite=overwrite)

    def register_hook(self, component: Hook, overwrite=False) -> str:
        return self._add_component(component, Hook, overwrite=overwrite)

    def add_step(self, component: Step, overwrite=False) -> str:
        return self._add_component(component, Step, overwrite=overwrite)

    def _add_component(
            self,
            component: Component,
            base_type: type[Component],
            *,
            overwrite: bool,
            inject: bool = True,
    ) -> str:
        self._validate_component(component, base_type, overwrite=overwrite)
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
            injected = consumer.__dict__.get(Component.DEPENDENCIES_ATTR)
            if injected is None:
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
        in `__init__` must still be activated by the session.
        """
        component.__dict__[Component.DEPENDENCIES_ATTR] = {
            edge.asked: self.get_resource(edge.asked, consumer=component.name)
            for edge in declared_edges(type(component))
            if edge.injects
        }

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

        if self.component_bindings.is_bound(component.name):
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
            injected = consumer.__dict__.get(Component.DEPENDENCIES_ATTR)
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
        return self.component_bindings.resolve(name)

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
            self.component_bindings,
            name,
            consumer=consumer,
            active=self.components if active is None else active,
        )

    @property
    def bindings(self) -> dict[str, str]:
        return self.component_bindings.bindings

    @property
    def aliases(self) -> ComponentBindings:
        warnings.warn(
            "SessionComponents.aliases is deprecated; use component_bindings",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.component_bindings

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
            self.component_bindings,
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
