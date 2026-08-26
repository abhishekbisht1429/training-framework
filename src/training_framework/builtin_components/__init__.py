"""Built-in training framework components.

The package facade preserves the historical
``training_framework.builtin_components`` import path while the
implementations live in focused modules.
"""

from training_framework.builtin_components.checkpointing import Checkpointer
from training_framework.builtin_components.data import DataManager
from training_framework.builtin_components.distributed import DDPResource
from training_framework.builtin_components.observability import (
    Logger,
    Tensorboard,
    Timer,
)
from training_framework.builtin_components.optimization import OptimizerHook

__all__ = [
    "Checkpointer",
    "DataManager",
    "DDPResource",
    "Logger",
    "OptimizerHook",
    "Tensorboard",
    "Timer",
]
