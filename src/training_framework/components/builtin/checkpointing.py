from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, override

import torch

from training_framework.components import LifecycleHook, Stateful
from training_framework.components import hook
from training_framework.util import timestamp_str

if TYPE_CHECKING:
    from training_framework.session import Session


@hook("checkpointer")
class Checkpointer(LifecycleHook, Stateful):

    def __init__(self, config: dict):
        self._config = config
        self._checkpoints_dir = None
        self.call_every = config["checkpoint_every"]

    @override
    def setup(self, session: Session) -> Any:
        if "checkpoints_dir" in self._config:
            self._checkpoints_dir = self._config["checkpoints_dir"]
        else:
            self._checkpoints_dir = os.path.join(
                session.session_config.session_dir,
                "checkpoints",
            )
        os.makedirs(self._checkpoints_dir, exist_ok=True)

    @override
    def teardown(self, session):
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

    @classmethod
    def load_checkpoint(
            cls,
            path,
            map_location="cpu",
    ) -> Session:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
