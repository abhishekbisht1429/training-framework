from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from training_framework.training_engine import TrainingEngine
from training_framework.training_session import TrainingSession
from tests.test_utils import (
    iteration_events,
    register_test_components,
    session_config,
)


@dataclass
class _ExtendConfig:
    checkpoint_path: str
    new_max_iters: int
    mode: str = "extend"
    process_timeout_on_join: float = 5.0
    session_configs: tuple[dict[str, Any], ...] = ()
    heartbeat_timeout: float = 10.0


def _training_step(session: TrainingSession):
    return next(step for step in session.get_all_steps() if step.name == "it_3d45_train")


def test_extend_mode_matches_uninterrupted_training_through_checkpoint_and_spawn(tmp_path):
    """Cover checkpoint load, max-iteration extension, spawn, and exact continuation."""

    register_test_components()
    baseline_path = tmp_path / "baseline.jsonl"
    extended_path = tmp_path / "extended.jsonl"

    baseline = TrainingSession(
        session_config(tmp_path / "baseline", max_iterations=6, event_path=baseline_path)
    )
    with baseline:
        list(baseline)
    baseline_step = _training_step(baseline)

    partial = TrainingSession(
        session_config(tmp_path / "partial", max_iterations=4, event_path=extended_path)
    )
    with partial:
        assert next(partial) == 1
        assert next(partial) == 2

    checkpoint_path = tmp_path / "partial_session.pt"
    torch.save(partial, checkpoint_path)

    engine_config = _ExtendConfig(
        checkpoint_path=str(checkpoint_path),
        new_max_iters=6,
    )
    with TrainingEngine(engine_config) as engine:
        engine.start_session()

    events = iteration_events(extended_path)
    assert [event["iteration"] for event in events] == [1, 2, 3, 4, 5, 6]
    assert [event["weight"] for event in events] == pytest.approx(
        baseline_step.weight_history,
        rel=1e-12,
        abs=1e-12,
    )
    assert [event["noise"] for event in events] == pytest.approx(
        baseline_step.noise_history,
        rel=1e-12,
        abs=1e-12,
    )

    assert [event["pid"] for event in events[:2]] == [os.getpid(), os.getpid()]
    child_pids = {event["pid"] for event in events[2:]}
    assert len(child_pids) == 1
    assert os.getpid() not in child_pids

    # The checkpointed parent object remains at the original pause point.
    assert partial.iteration == 2
    assert len(_training_step(partial).weight_history) == 2
