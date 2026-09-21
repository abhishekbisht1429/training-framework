from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch

from training_framework.components import (
    ExtendableComponent,
    LifecycleHook,
    Stateful,
)
from training_framework.components import hook, rank_zero_only
from training_framework.session.state import rng_restore_suppressed
from training_framework.util import timestamp_str

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
        filepath = os.path.join(self._checkpoints_dir, timestamp_str())
        torch.save(session, filepath)

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

    @classmethod
    def load_checkpoint(
            cls,
            path,
            map_location="cpu",
            restore_rng: bool = True,
    ) -> Session:
        """Load a checkpointed session.

        `restore_rng=False` loads it without adopting its RNG, for a caller
        that wants what the session holds rather than the run it came from.
        """
        rng = nullcontext() if restore_rng else rng_restore_suppressed()
        with rng:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )

    @classmethod
    def load_component(
            cls,
            path,
            name: str,
            *,
            map_location="cpu",
            session_type: str | None = None,
    ):
        """Return one resource out of a checkpoint.

        For a component that needs something a *different* run produced --
        the trained model an analysis session inspects, say. `name` is resolved
        through the checkpoint's own bindings, so a role such as `model` finds
        whatever that run bound it to; the loading session's wiring says
        nothing about another run. The checkpoint's RNG is not adopted.

        `session_type`, when given, is the kind of session the checkpoint must
        hold. Raises `KeyError` when the checkpoint has no such resource, and
        `ComponentDependencyError` when several instances answer and nothing
        in the checkpoint decides between them.
        """
        from training_framework.session import Session as FrameworkSession

        source = cls.load_checkpoint(
            path,
            map_location=map_location,
            restore_rng=False,
        )
        if not isinstance(source, FrameworkSession):
            raise TypeError("Checkpoint must contain a framework Session")
        if session_type is not None and source.session_type != session_type:
            article = "an" if session_type[:1] in "aeiou" else "a"
            raise ValueError(
                f"Checkpoint must contain {article} {session_type} session, "
                f"but holds a '{source.session_type}' one"
            )
        return source._components.get_resource(name)
