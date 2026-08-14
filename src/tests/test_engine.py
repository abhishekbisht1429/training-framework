from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from training_framework.training_engine import (
    SessionProcessWrapper,
    TrainingEngine,
    load_session_for_worker,
)
from training_framework.training_session import TrainingSession
from tests.test_utils import (
    COMPONENTS_PACKAGE,
    iteration_events,
    register_test_components,
    session_config,
)


@dataclass
class _EngineConfig:
    mode: str
    process_timeout_on_join: float = 5.0
    session_configs: tuple[dict[str, Any], ...] = ()
    checkpoint_path: str | None = None
    new_max_iters: int | None = None


def _training_step(session: TrainingSession):
    return next(step for step in session.get_all_steps() if step.name == "it_3d45_train")


def _run_to_completion(config: dict[str, Any]) -> TrainingSession:
    session = TrainingSession(config)
    with session:
        list(session)
    return session


def _join_spawned_wrapper(wrapper: SessionProcessWrapper, timeout: float = 30.0) -> int | None:
    wrapper.start()
    wrapper.process.join(timeout=timeout)
    if wrapper.process.is_alive():
        wrapper.request_stop()
        wrapper.process.terminate()
        wrapper.process.join(timeout=5.0)
        pytest.fail("spawned training worker did not finish")
    exitcode = wrapper.process.exitcode
    wrapper.process.close()
    return exitcode


def test_spawned_worker_reconstructs_paused_state_and_continues_exactly(tmp_path):
    """The child must resume model, optimizer, step, iteration, and RNG state."""

    register_test_components()
    baseline_path = tmp_path / "baseline.jsonl"
    resumed_path = tmp_path / "resumed.jsonl"

    baseline = _run_to_completion(
        session_config(tmp_path / "baseline", max_iterations=5, event_path=baseline_path)
    )
    baseline_step = _training_step(baseline)

    paused = TrainingSession(
        session_config(tmp_path / "paused", max_iterations=5, event_path=resumed_path)
    )
    with paused:
        assert next(paused) == 1
        assert next(paused) == 2

    parent_history = list(_training_step(paused).weight_history)
    wrapper = SessionProcessWrapper(paused, session_id=0, rank=0)
    assert _join_spawned_wrapper(wrapper) == 0

    resumed_iterations = iteration_events(resumed_path)
    assert [event["iteration"] for event in resumed_iterations] == [1, 2, 3, 4, 5]
    assert [event["weight"] for event in resumed_iterations] == pytest.approx(
        baseline_step.weight_history,
        rel=1e-12,
        abs=1e-12,
    )
    assert [event["noise"] for event in resumed_iterations] == pytest.approx(
        baseline_step.noise_history,
        rel=1e-12,
        abs=1e-12,
    )

    assert [event["pid"] for event in resumed_iterations[:2]] == [os.getpid(), os.getpid()]
    child_pids = {event["pid"] for event in resumed_iterations[2:]}
    assert len(child_pids) == 1
    assert os.getpid() not in child_pids

    # The spawned worker received state, not the live parent object.
    assert paused.iteration == 2
    assert _training_step(paused).weight_history == parent_history


def test_engine_surfaces_a_real_spawned_worker_failure(tmp_path):
    """Use a real child process and verify that its non-zero exit reaches the caller."""

    register_test_components()
    config = {
        "base_config": {
            "rng_seed": 11,
            "sessions_dir": str(tmp_path),
            "max_iterations": 3,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "it_3d45_fail": {
            "fail_at": 2,
            "message": "failure raised inside spawned worker",
        },
    }
    engine_config = _EngineConfig(mode="new", session_configs=(config,))

    with pytest.raises(
        RuntimeError,
        match=r"training workers failed: session=0, rank=0, exitcode=",
    ):
        with TrainingEngine(engine_config) as engine:
            engine.start_all()


def test_worker_loading_builds_rank_specific_ddp_sessions_without_patching(tmp_path):
    """Verify the commit's rank filtering with genuine registered components."""

    register_test_components(include_builtins=True)
    config = {
        "base_config": {
            "rng_seed": 31,
            "sessions_dir": str(tmp_path),
            "max_iterations": 3,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "ddp": {
            "world_size": 2,
            "backend": "gloo",
            "parallel_components": ["it_3d45_model", "it_3d45_train"],
            "master_addr": "localhost",
            "master_port": "12355"
        },
        "it_3d45_model": {},
        "it_3d45_train": {},
        "it_3d45_rank0_resource": {},
        "it_3d45_rank0_step": {},
        "it_3d45_rank0_hook": {"call_every": 1},
    }
    source = TrainingSession(config)
    state = source.get_state()

    rank_zero = load_session_for_worker(
        state,
        rank=0,
        session_update_params={"max_iterations": 8},
    )
    rank_one = load_session_for_worker(
        state,
        rank=1,
        session_update_params={"max_iterations": 8},
    )

    assert source.get_resource("ddp").rank == -1
    assert source.session_config.max_iterations == 3
    assert rank_zero.get_resource("ddp").rank == 0
    assert rank_one.get_resource("ddp").rank == 1
    assert rank_zero.session_config.max_iterations == 8
    assert rank_one.session_config.max_iterations == 8

    assert {resource.name for resource in rank_zero.get_all_resources()} == {
        "ddp",
        "it_3d45_model",
        "it_3d45_rank0_resource",
    }
    assert {step.name for step in rank_zero.get_all_steps()} == {
        "it_3d45_train",
        "it_3d45_rank0_step",
    }
    assert {hook.name for hook in rank_zero.get_all_hooks()} == {"it_3d45_rank0_hook"}

    assert {resource.name for resource in rank_one.get_all_resources()} == {
        "ddp",
        "it_3d45_model",
    }
    assert {step.name for step in rank_one.get_all_steps()} == {"it_3d45_train"}
    assert rank_one.get_all_hooks() == []
