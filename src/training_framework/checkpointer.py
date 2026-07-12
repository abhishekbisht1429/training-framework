import os
import time
from datetime import datetime
from typing import Any, override

import numpy as np
import torch

from training_framework.training_session import IterationComponent, TrainingSession, Wrapper, SessionConfig
from training_framework.util import timestamp_str


# def save_checkpoint(checkpoint, save_dir="checkpoints", prefix="model"):
#     # Create directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
#
#     # File path
#     filepath = os.path.join(save_dir, f"{prefix}_{checkpoint['timestamp']}.pt")
#
#     # Save
#     torch.save(checkpoint, filepath)
#
#     return filepath
#
# def load_checkpoint(filepath, map_location="cpu"):
#     checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
#
#     return checkpoint


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