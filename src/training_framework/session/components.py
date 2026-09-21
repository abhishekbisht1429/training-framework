import warnings
from collections.abc import Iterable, Mapping
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


class ComponentNotFoundError(KeyError):
    """A KeyError whose message keeps its line breaks when printed."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


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
            }
            for name, component in self.components.items()
        }

    def set_state(self, component_states: dict[str, dict[str, Any]]) -> None:
        restored_components: dict[str, Component] = {}
        # Components are rebuilt into a fresh mapping, but a component being
        # constructed must see the ones already rebuilt, so the view reads
        # through to it. The previous mapping is put back if the restore fails.
        previous_components = self.components
        self.components = restored_components
        try:
            self._restore_components(component_states, restored_components)
        except BaseException:
            self.components = previous_components
            raise

    def _restore_components(
            self,
            component_states: dict[str, dict[str, Any]],
            restored_components: dict[str, Component],
    ) -> None:
        # A checkpoint is not a trusted plan: it may predate a component being
        # marked @singleton, or have been edited. Checked before anything is
        # built; set_state restores the previous components if it raises.
        self._check_instance_limits(component_states)

        # Pass 1: rebuild every component from its constructor arguments,
        # prerequisites first. The stored order is usually already
        # prerequisite-first, since it is the activation order, but a
        # component registered by hand -- say a prerequisite replaced after
        # its consumer was built -- is stored after the components that use
        # it. Each component is handed its prerequisites as it is built, so
        # they are built first here rather than trusted to come first.
        building: list[str] = []

        def build(name: str) -> None:
            if name in restored_components:
                return
            if name in building:
                chain = " -> ".join([*building, name])
                raise ValueError(
                    f"Checkpoint components depend on each other in a "
                    f"cycle: {chain}"
                )
            component_info = component_states[name]
            implementation, _ = parse_instance_name(name)
            component_class = self.registry.get(implementation)
            if component_class is None:
                raise ValueError(
                    f"Checkpoint component '{name}' is not registered"
                )
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

            # Resolved against every name in the checkpoint, not just the ones
            # rebuilt so far, so a sibling instance not yet rebuilt cannot make
            # this one look like the only one.
            dependencies = self._resource_dependencies(
                component_class,
                name,
                active=set(component_states),
            )
            building.append(name)
            try:
                for asked, target in dependencies.items():
                    if target not in component_states:
                        raise ValueError(
                            f"Checkpoint component '{name}' requires "
                            f"'{asked}', which resolves to '{target}', but the "
                            "checkpoint does not contain it"
                        )
                    build(target)
            finally:
                building.pop()

            init_args = component_info["init_args"]
            restored_components[name] = self._construct(
                component_class,
                name,
                *init_args["args"],
                dependencies=dependencies,
                **init_args["kwargs"],
            )

        for name in component_states:
            build(name)

        # Pass 2: restore state in prerequisite-first order, so a component
        # that inspects a dependency sees it already restored.
        for name in self._state_restore_order(component_states):
            component = self.components[name]
            if isinstance(component, Stateful):
                component.set_state(component_states[name]["state"])

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
    ) -> dict[str, str]:
        """Resolve a consumer's declared resources to instance names.

        Only resources are injected: `required_hooks` / `required_steps` and
        `@wraps` targets are ordering declarations, and a hook or step is not
        servable through `get_dependency`.
        """
        return {
            name: self.resolve_dependency(
                name,
                consumer=consumer,
                active=active,
            )
            for name, dependency_type in self._dependency_specs(component_class)
            if dependency_type is Resource
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
        holding: `_instances_of` reads the components built so far, so "the
        sole instance" could be sole only because its sibling is not built
        yet.

        The prerequisites are written into the instance `__dict__` before
        `__init__` runs, so a constructor can use them, an `nn.Module`
        subclass needs no `nn.Module.__init__` to have run first, and a
        prerequisite module is not registered as a submodule of its consumer.
        """
        component = component_class.__new__(component_class)
        component.__dict__[Component.DEPENDENCIES_ATTR] = {
            asked: self.components[target]
            for asked, target in (dependencies or {}).items()
        }
        component.__init__(*args, **kwargs)
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

    def _activate_all(
            self,
            roots: Iterable[str],
            component_configs: Mapping[str, Mapping],
    ) -> None:
        visiting: list[str] = []
        # Every instance this call will end up holding. A dependency has to be
        # resolved against all of them, not against the ones built so far, or
        # a component activated early would see a second instance as the only
        # one simply because the first had not been constructed yet.
        planned = set(component_configs) | {
            self.resolve_name(root) for root in roots
        }

        # Checked against everything the session will hold, not only what
        # this call adds, or a second activate_component() would slip a
        # second instance past it.
        self._check_instance_limits(planned | set(self.components))

        def activate(target: str) -> None:
            resolved_name, component_class = self._registered_component_class(
                target,
            )
            if resolved_name in self.components:
                return
            if resolved_name in visiting:
                # Components are wired as they are constructed, so a cycle has
                # no valid construction order. Report it here, where the chain
                # that closed it is still known.
                chain = " -> ".join([*visiting, resolved_name])
                raise RuntimeError(
                    f"Cyclic dependency detected in the component graph! {chain}"
                )

            visiting.append(resolved_name)
            try:
                # Kept as each dependency is resolved and handed to
                # _construct, so the instance injected is the very one this
                # call decided to activate. Resolving again inside _construct
                # would read the components built so far, where a sibling
                # that is merely not built yet looks like it does not exist.
                resource_dependencies: dict[str, str] = {}
                for dependency_name, dependency_type in self._dependency_specs(
                        component_class,
                ):
                    # Resolved against this consumer, so a component wired to
                    # a particular instance activates that one.
                    dependency_target = self.resolve_dependency(
                        dependency_name,
                        consumer=resolved_name,
                        active=planned | set(self.components),
                    )
                    self._registered_component_class(
                        dependency_name,
                        dependency_type,
                        consumer=component_class,
                        resolved_name=dependency_target,
                    )
                    activate(dependency_target)
                    if dependency_type is Resource:
                        resource_dependencies[dependency_name] = (
                            dependency_target
                        )

                if resolved_name in component_configs:
                    component = self._construct(
                        component_class,
                        resolved_name,
                        component_configs[resolved_name],
                        dependencies=resource_dependencies,
                    )
                elif is_instance_name(resolved_name):
                    # Only a configured key declares an instance. Creating one
                    # because something is wired to it would turn a mistyped
                    # suffix into a fresh, unconfigured instance -- a run that
                    # works and is quietly wrong.
                    configured = sorted(
                        name for name in planned | set(self.components)
                        if implementation_of(name)
                        == implementation_of(resolved_name)
                    )
                    raise ComponentDependencyError(
                        f"Component instance '{resolved_name}' is not "
                        "configured in this session, so nothing can be wired "
                        "to it. An instance is created only by a top-level "
                        f"'{resolved_name}' key. Configured instances of "
                        f"'{implementation_of(resolved_name)}': "
                        f"{configured or 'none'}."
                    )
                elif component_class.__init__ is Component.__init__:
                    component = self._construct(
                        component_class,
                        resolved_name,
                        dependencies=resource_dependencies,
                    )
                else:
                    raise RuntimeError(
                        f"Component '{resolved_name}' is required but defines "
                        "a custom constructor. Add a top-level component "
                        f"mapping for '{resolved_name}'."
                    )
                self._register_component_instance(component)
            finally:
                visiting.pop()

        for root in roots:
            # A configured name says which instance to create, so it is taken
            # literally. Only a *dependency* is resolved to an instance.
            activate(root)

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
    ) -> set[str]:
        """Return `names` plus everything they depend on, transitively.

        `active_names` lets a caller resolve the closure before any component
        is constructed -- the dependency graph is class-level, so the worker
        can decide what a rank needs without building the session first.
        """
        active = (
            set(self.components)
            if active_names is None
            else set(active_names)
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
            for dependency_name, dependency_type in self._dependency_specs(
                    component_class,
            ):
                dependency_target = self.resolve_dependency(
                    dependency_name,
                    consumer=resolved_name,
                    active=active,
                )
                self._registered_component_class(
                    dependency_name,
                    dependency_type,
                    consumer=component_class,
                    resolved_name=dependency_target,
                )
                visit(dependency_target)

        for name in names:
            visit(name)
        return closure

    def _names_depending_on(
            self,
            target_name: str,
            active: set[str],
    ) -> set[str]:
        """Return the active components that reach `target_name` transitively."""
        dependents = set()
        for name in active:
            try:
                closure = self.dependency_closure([name], active_names=active)
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
        # @rank_zero_only is its author's settled decision -- `timer` requires
        # `optimizer`, which requires `ddp`, and it is still rank-zero-only on
        # purpose -- while a config entry is a per-run override worth a second
        # look, because excluding a participant in the collectives is what
        # leaves the other ranks waiting.
        using_ddp = sorted(
            name for name in declared_names - {ddp_name}
            if ddp_name in self.dependency_closure([name], active_names=active)
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
        """
        active = (
            set(self.components)
            if active_names is None
            else set(active_names)
        )
        ddp_name = self.resolve_name("ddp")

        if parallel_components is not None:
            keep = self.dependency_closure(
                list(parallel_components) + ["ddp"],
                active_names=active,
            )
            pruned = sorted(self._names_depending_on(ddp_name, active) - keep)
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
        )
        keep = self.dependency_closure(
            active - rank_zero,
            active_names=active,
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
        if inject:
            self._check_instance_limits(
                set(self.components) | {component.name},
            )
            self._inject_dependencies(component)
            self._repoint_consumers(component)
        self.components[component.name] = component
        return component.name

    def _repoint_consumers(self, replacement: Component) -> None:
        """Hand consumers of a replaced instance the one now registered.

        A consumer is wired to an instance *name*, which is unique within a
        session, so whichever component is registered under that name is the
        one it gets. Without this, replacing a component -- unregistering it
        and registering a copy, or overwriting it -- would leave consumers
        built earlier holding the instance that was taken out.

        Only what `get_dependency` hands out from here on is affected. A
        reference a consumer already stored for itself, typically in
        `__init__`, stays where the consumer put it.
        """
        for consumer in self.components.values():
            injected = consumer.__dict__.get(Component.DEPENDENCIES_ATTR)
            if not injected:
                continue
            for asked, held in injected.items():
                if (
                        held is not replacement
                        and getattr(held, "name", None) == replacement.name
                ):
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
            name: self.get_resource(name, consumer=component.name)
            for name, dependency_type in self._dependency_specs(type(component))
            if dependency_type is Resource
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
        del self.components[registered_name]

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

    def _instances_of(
            self,
            name: str,
            active: Iterable[str] | None = None,
    ) -> list[str]:
        """Return the active instances `name` could refer to.

        An exact match is the answer on its own: a component named `model`
        stays the answer to `model` however many `model#...` instances join
        it, so adding an instance never silently rewires anything.
        """
        active = self.components if active is None else set(active)
        if name in active:
            return [name]
        if is_instance_name(name):
            # A suffixed name is a precise reference. When that instance is
            # not active the answer is "not configured", never a sibling that
            # happens to share its implementation.
            return []
        implementation = implementation_of(name)
        return sorted(
            instance_name for instance_name in active
            if implementation_of(instance_name) == implementation
        )

    def resolve_dependency(
            self,
            name: str,
            *,
            consumer: str | None = None,
            active: Iterable[str] | None = None,
    ) -> str:
        """Return the instance name that satisfies `name` for `consumer`.

        Three rules, in order: the consumer's own wiring decides if it has
        any; otherwise the sole active instance of the component; otherwise
        it is an error naming the candidates. Picking one of several would
        mean wiring a component to something its session never chose, which
        produces a run that works and is quietly wrong.

        A name with nothing active behind it is returned as bound, so a
        component that is simply not configured is reported by the caller
        that knows what it was looking for.
        """
        resolved = self.component_bindings.resolve(name, consumer=consumer)
        candidates = self._instances_of(resolved, active)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return resolved
        raise ComponentDependencyError(with_explanation(
            f"Component '{name}' resolves to '{resolved}', which {len(candidates)} "
            f"active components implement: {candidates}.",
            "Fix: name the one that is meant with per-component wiring, "
            "component_bindings: {'"
            f"{consumer if consumer is not None else '<component>'}"
            "': {'"
            f"{name}': '{candidates[0]}'"
            "}}.",
        ))

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
