"""Building and running the tiny DINO and BYOL sessions in one process."""

from __future__ import annotations

import torch

from training_framework.session import TrainingSession


def quiet_session(config) -> TrainingSession:
    session = TrainingSession(config)
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    return session


def run(session, iterations=None):
    with session:
        if iterations is None:
            return list(session)
        return [next(session) for _ in range(iterations)]


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}
