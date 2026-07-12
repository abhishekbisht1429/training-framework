import os
from typing import Any, override

import torch

from training_framework.training_session import IterationComponent, TrainingSession, Wrapper, SessionConfig
from training_framework.util import timestamp_str

class Checkpointer(Wrapper):

    def __init__(self, config: dict):
        self._config = config
        self._checkpoints_dir = None
        self.call_wrapper_every = config['checkpoint_every']

    @override
    def setup(self, session: TrainingSession) -> Any:
        if "checkpoints_dir" in self._config:
            self._checkpoints_dir = self._config["checkpoints_dir"]
        else:
            self._checkpoints_dir = os.path.join(session.session_config.session_dir, 'checkpoints')

        # Create directory if it doesn't exist
        os.makedirs(self._checkpoints_dir, exist_ok=True)

    @override
    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    @override
    def post_iteration_callback(self, session: "TrainingSession") -> None:
        print("Creating checkpoint...")
        # File path
        filepath = os.path.join(self._checkpoints_dir, timestamp_str())

        # Save
        torch.save(session, filepath)


    @override
    def __getstate__(self) -> Any:
        return {
            'config': self._config,
        }

    @override
    def __setstate__(self, state: Any) -> None:
        self._config = state['config']
        self.call_wrapper_every = self._config['checkpoint_every']


    @classmethod
    def load_checkpoint(cls, path, map_location="cpu") -> TrainingSession:
        return torch.load(path, map_location=map_location, weights_only=False)