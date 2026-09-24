from __future__ import annotations

import os
import warnings
from collections.abc import Collection, Mapping
from contextlib import ExitStack
from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch

from training_framework.components import (
    ExtendableComponent,
    LifecycleHook,
    Stateful,
)
from training_framework.components import hook, rank_zero_only
from training_framework.components.base import ComponentDependencyError
from training_framework.components.config import component_bindings_from_config
from training_framework.components.edges import instances_of
from training_framework.session.checkpoint_format import (
    is_checkpoint_directory,
    read_checkpoint,
    read_component_state,
    read_manifest,
    read_session_record,
    write_checkpoint,
)
from training_framework.session.components import (
    ComponentNotFoundError,
    SessionComponents,
)
from training_framework.session.state import (
    configuration_from_state,
    rng_preserved,
    rng_restore_suppressed,
)
from training_framework.util import import_all_modules, timestamp_str

if TYPE_CHECKING:
    from training_framework.session import Session


@rank_zero_only
@hook("checkpointer")
class Checkpointer(LifecycleHook, Stateful, ExtendableComponent):

    def __init__(self, config: dict):
        self._config = config
        self._checkpoints_dir = None
        self.call_every = config["checkpoint_every"]

    @override
    def pre_session(self, session: Session) -> Any:
        if "checkpoints_dir" in self._config:
            self._checkpoints_dir = self._config["checkpoints_dir"]
        else:
            # Two checkpointers writing timestamped files into one directory
            # would interleave their runs, so an instance that has siblings
            # gets its own. The sole checkpointer keeps the plain name.
            directory = "checkpoints"
            if self.instance_suffix is not None:
                directory = f"checkpoints_{self.instance_suffix}"
            self._checkpoints_dir = os.path.join(
                session.session_config.session_dir,
                directory,
            )
        os.makedirs(self._checkpoints_dir, exist_ok=True)

    @override
    def post_session(self, session):
        pass

    @override
    def pre_iteration_callback(self, session: Session) -> None:
        pass

    @override
    def post_iteration_callback(self, session: Session) -> None:
        if (
                session.iteration == 1
                and session.session_config.max_iterations > 1
                and not self._config.get("checkpoint_first", False)
        ):
            return

        print("Creating checkpoint...")
        self.save_checkpoint(
            session,
            os.path.join(self._checkpoints_dir, timestamp_str()),
        )

    @override
    def get_state(self) -> Any:
        return {"config": self._config}

    @override
    def set_state(self, state: Any) -> None:
        self._config = state["config"]
        self.call_every = self._config["checkpoint_every"]

    @override
    def apply_extension_config(
            self,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        allowed = {("checkpoint_every",), ("checkpoint_first",)}
        unsupported = changed_paths - allowed
        if unsupported:
            names = ", ".join(".".join(path) for path in sorted(unsupported))
            raise ValueError(
                "Checkpointer session extension does not allow changes to: "
                + names
            )
        self._config = deepcopy(dict(config))
        self.call_every = self._config["checkpoint_every"]

    @staticmethod
    def save_checkpoint(session: Session, path) -> str:
        """Write `session` as a checkpoint directory at `path`.

        The directory holds a `manifest.json` describing the session and its
        components, the session's own state, and one file per component, all
        plain data (see `training_framework.session.checkpoint_format`). It
        is written under a temporary name and renamed into place when
        complete. Raises, naming the component and the value, when a state
        holds something other than plain data. Returns the path.
        """
        return write_checkpoint(session.get_state(), path)

    @classmethod
    def load_checkpoint(
            cls,
            path,
            map_location="cpu",
            restore_rng: bool = True,
            *,
            on_mismatch: str | Collection[str] = "raise",
    ) -> Session:
        """Load a checkpointed session.

        `path` is a checkpoint directory, as the checkpointer writes. A single
        file is a checkpoint from before 0.5.0: it is still read, for this
        release only, by unpickling the whole session.

        `restore_rng=False` loads it without adopting its RNG, for a caller
        that wants what the session holds rather than the run it came from.
        The caller's RNG is then left exactly as it was: whatever rebuilding
        the components draws is undone, so the caller's seed still decides
        what comes next.

        `on_mismatch` decides what happens to a component whose saved state
        this version of it cannot take (see `Component.state_version`):
        "raise" refuses the checkpoint, listing every such component;
        "reinit" keeps them as freshly built and warns; a collection of
        instance names does that for those only.
        """
        from training_framework.session import Session as FrameworkSession

        state = (
            read_checkpoint(path, map_location=map_location)
            if is_checkpoint_directory(path)
            else None
        )
        with ExitStack() as stack:
            if not restore_rng:
                stack.enter_context(rng_restore_suppressed())
                stack.enter_context(rng_preserved())
            if state is not None:
                return FrameworkSession.from_state(
                    state,
                    on_mismatch=on_mismatch,
                )
            if on_mismatch != "raise":
                raise ValueError(
                    "on_mismatch applies to checkpoint directories only; "
                    f"{os.fspath(path)} is a single-file checkpoint"
                )
            return cls._load_legacy_checkpoint(path, map_location)

    @staticmethod
    def _load_legacy_checkpoint(path, map_location):
        warnings.warn(
            f"{os.fspath(path)} is a single-file checkpoint, written before "
            "0.5.0. It is read by unpickling the whole session, which the "
            "next release no longer does. Convert it once with "
            "Checkpointer.save_checkpoint(Checkpointer.load_checkpoint(path), "
            "new_path).",
            FutureWarning,
            stacklevel=4,
        )
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    @staticmethod
    def read_manifest(path) -> dict[str, Any]:
        """Return what a checkpoint directory holds, without loading it.

        The session type, iteration, configuration, and for every component
        its implementation, kind, `state_version`, the instances it was
        wired to and the file holding its state. Reads `manifest.json` only,
        so it needs none of the checkpointed classes.
        """
        return read_manifest(path)

    @classmethod
    def load_component_state(cls, path, name: str, *, map_location="cpu"):
        """Return one component's saved state -- its weights, for a model --
        without building anything.

        Reads that component's file only and imports none of the
        checkpointed classes, so it works when the component, or anything
        else in the checkpoint, no longer loads. `name` is an instance name,
        a name the checkpoint's `component_bindings` bind, or a name one of
        its components asked for as a dependency; a role only a component
        package declares needs `load_component`.
        """
        if is_checkpoint_directory(path):
            instance = _instance_named_in_manifest(read_manifest(path), name)
            return read_component_state(
                path, instance, map_location=map_location,
            )
        source = cls.load_checkpoint(
            path, map_location=map_location, restore_rng=False,
        )
        components = source._components
        component = components.components.get(
            components.resolve_dependency(name),
        )
        if component is None:
            raise ComponentNotFoundError(
                f"Checkpoint {os.fspath(path)} has no component '{name}'"
            )
        return component.get_state() if isinstance(component, Stateful) else None

    @classmethod
    def load_component(
            cls,
            path,
            name: str,
            *,
            with_dependencies: bool = True,
            map_location="cpu",
            session_type: str | None = None,
    ):
        """Return one resource out of a checkpoint.

        For a component that needs something a *different* run produced --
        the trained model an analysis session inspects, say. `name` is resolved
        through the checkpoint's own bindings, so a role such as `model` finds
        whatever that run bound it to; the loading session's wiring says
        nothing about another run. The checkpoint's RNG is not adopted, and
        the caller's is left exactly as it was.

        Only that resource and what it was wired to are rebuilt, as that run
        wired them; nothing else in the checkpoint is read, so another
        component that no longer loads does not stand in the way.
        `with_dependencies=False` refuses a resource that was wired to
        others -- it could not be built without them -- and points at
        `load_component_state`, which returns its saved state alone.

        `session_type`, when given, is the kind of session the checkpoint must
        hold. Raises `KeyError` when the checkpoint has no such resource, and
        `ComponentDependencyError` when several instances answer and nothing
        in the checkpoint decides between them.
        """
        if not is_checkpoint_directory(path):
            return cls._load_legacy_component(
                path, name, with_dependencies, map_location, session_type,
            )

        record = read_session_record(path, map_location=map_location)
        _check_session_type(record.get("session_type"), session_type)
        config, session_settings, _ = configuration_from_state(record)
        import_all_modules(session_settings["components_package"])
        components = SessionComponents(
            component_bindings=component_bindings_from_config(config),
            session_type=record["session_type"],
        )
        component_records = record["components"]
        instance = components.resolve_dependency(
            name, active=component_records,
        )
        if component_records.get(instance, {}).get("component_type") != "Resource":
            resources = sorted(
                other for other, info in component_records.items()
                if info.get("component_type") == "Resource"
            )
            raise ComponentNotFoundError(
                f"{name} not found in resources! The checkpoint holds the "
                f"resources {resources}."
            )

        needed = _wired_closure(component_records, instance)
        if not with_dependencies and needed != {instance}:
            raise ValueError(
                f"Checkpoint component '{instance}' was wired to "
                f"{sorted(needed - {instance})}, and cannot be built without "
                "them. Load it with with_dependencies=True, or read its saved "
                "state alone with Checkpointer.load_component_state()."
            )
        state = read_checkpoint(
            path, map_location=map_location, components=needed,
        )
        with rng_preserved():
            components.set_state(state["components_state"], partial=True)
        return components.components[instance]

    @classmethod
    def _load_legacy_component(
            cls, path, name, with_dependencies, map_location, session_type,
    ):
        from training_framework.session import Session as FrameworkSession

        source = cls.load_checkpoint(
            path,
            map_location=map_location,
            restore_rng=False,
        )
        if not isinstance(source, FrameworkSession):
            raise TypeError("Checkpoint must contain a framework Session")
        _check_session_type(source.session_type, session_type)
        resource = source._components.get_resource(name)
        if not with_dependencies and resource._dependencies:
            raise ValueError(
                f"Checkpoint component '{resource.name}' was wired to "
                f"{sorted(dependency.name for dependency in resource._dependencies.values())}, "
                "and cannot be built without them. Load it with "
                "with_dependencies=True, or read its saved state alone with "
                "Checkpointer.load_component_state()."
            )
        return resource


def _check_session_type(stored: str | None, expected: str | None) -> None:
    if expected is not None and stored != expected:
        article = "an" if expected[:1] in "aeiou" else "a"
        raise ValueError(
            f"Checkpoint must contain {article} {expected} session, "
            f"but holds a '{stored}' one"
        )


def _wired_closure(records: Mapping[str, Mapping], root: str) -> set[str]:
    """`root` and every instance it was wired to, transitively."""
    needed: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        dependencies = records.get(name, {}).get("dependencies") or {}
        pending.extend(
            target for target in dependencies.values() if target in records
        )
    return needed


def _instance_named_in_manifest(manifest: Mapping, name: str) -> str:
    """Resolve `name` from what a checkpoint recorded, importing nothing.

    In order: an instance name; a top-level `component_bindings` entry; the
    instance the checkpoint's components were given when they asked for
    `name`; the sole instance of that implementation.
    """
    records = manifest["components"]
    if name in records:
        return name
    config = manifest.get("config") or {}
    bindings = config.get("component_bindings") or config.get("aliases") or {}
    bound = bindings.get(name) if isinstance(bindings, Mapping) else None
    if isinstance(bound, str):
        candidates = instances_of(bound, records)
    else:
        given = sorted({
            (record.get("dependencies") or {})[name]
            for record in records.values()
            if name in (record.get("dependencies") or {})
        })
        candidates = given or instances_of(name, records)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ComponentNotFoundError(
            f"The checkpoint has no component '{name}'; it holds "
            f"{sorted(records)}"
        )
    raise ComponentDependencyError(
        f"'{name}' could be any of {candidates} in this checkpoint; name "
        "the instance"
    )
