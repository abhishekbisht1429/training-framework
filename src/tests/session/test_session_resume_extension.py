from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import yaml

from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import TrainingSession
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


def test_extension_updates_effective_config_and_writes_config_history(tmp_path):
    register_test_components()
    session = TrainingSession(
        session_config(tmp_path / "extended", max_iterations=2)
    )

    session.apply_extension_overrides((
        "session_config.max_iterations=5",
        "logger.log_every=2",
        "checkpointer.checkpoint_every=4",
        "checkpointer.checkpoint_first=true",
    ))
    restored = TrainingSession.from_state(session.get_state())

    assert restored.session_config.max_iterations == 5
    assert restored.full_config["session_config"]["max_iterations"] == 5
    assert restored.full_config["logger"]["log_every"] == 2
    assert restored.full_config["checkpointer"] == {
        "checkpoint_every": 4,
        "checkpoint_first": True,
    }

    with restored:
        pass

    session_dir = restored.session_config.session_dir
    with open(os.path.join(session_dir, "config.yaml")) as config_file:
        dumped = yaml.safe_load(config_file)
    assert dumped == restored.full_config

    history_files = [
        name for name in os.listdir(session_dir)
        if name.startswith("config_extension_") and name.endswith(".yaml")
    ]
    assert len(history_files) == 1
    with open(os.path.join(session_dir, history_files[0])) as config_file:
        assert yaml.safe_load(config_file) == restored.full_config


@pytest.mark.parametrize(
    "override",
    [
        "session_config.rng_seed=7",
        "it_3d45_model.initial_weight=0.5",
        "ddp.world_size=2",
        "data_manager.batch_size=8",
    ],
)
def test_extension_rejects_non_opted_in_configuration(tmp_path, override):
    register_test_components()
    session = TrainingSession(
        session_config(tmp_path / "rejected", max_iterations=2)
    )

    with pytest.raises(ValueError, match="does not allow|not active"):
        session.apply_extension_overrides((override,))


def test_configurator_accepts_extension_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "pytest",
        "--extend-session",
        "checkpoint.pt",
        "--override",
        "session_config.max_iterations=20",
        "optimizer.optimizer.kwargs.lr=0.01",
    ])

    configurator = Configurator()

    assert configurator.mode == "extend"
    assert configurator.checkpoint_path == "checkpoint.pt"
    assert configurator.extension_overrides == (
        "session_config.max_iterations=20",
        "optimizer.optimizer.kwargs.lr=0.01",
    )


def test_configurator_keeps_deprecated_positional_extension(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pytest", "--extend-session", "checkpoint.pt", "20"],
    )

    with pytest.warns(DeprecationWarning, match="positional"):
        configurator = Configurator()

    assert configurator.new_max_iters == 20
    assert configurator.extension_overrides == (
        "session_config.max_iterations=20",
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "--extend-session", "checkpoint.pt"],
        [
            "pytest",
            "--extend-session",
            "checkpoint.pt",
            "20",
            "--override",
            "session_config.max_iterations=30",
        ],
    ],
)
def test_configurator_rejects_missing_or_duplicate_extension_values(
        monkeypatch,
        argv,
):
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.warns(DeprecationWarning) if len(argv) > 3 else nullcontext():
        with pytest.raises(SystemExit):
            Configurator()
