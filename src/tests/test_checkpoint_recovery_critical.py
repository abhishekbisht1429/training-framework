"""Critical checkpoint and recovery tests for training-framework.
The two strict xfail tests describe important checkpoint contracts that the
current implementation does not yet satisfy.  Once the underlying behavior is
fixed, pytest will report XPASS as a failure so the xfail marker can be removed.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from training_framework.resources import Checkpointer
from training_framework.training_session import (
    LifecycleHook,
    Resource,
    SessionHook,
    SessionPhase,
    Stateful,
    Step,
    TrainingSession,
    hook,
    resource,
    step,
)


@step("critical_checkpoint_accumulator_step")
class CriticalCheckpointAccumulatorStep(Step, Stateful):
    """A deterministic stateful step suitable for exact resume comparisons."""

    def __init__(self, start: int = 0, increment: int = 1):
        self.start = start
        self.increment = increment
        self.value = start
        self.history: list[int] = []

    def run(self, session: TrainingSession) -> None:
        self.value += self.increment
        self.history.append(self.value)

        session.iteration_context["accumulator"] = self.value
        session.session_context["last_accumulator"] = self.value

    def get_state(self) -> Any:
        return {
            "value": self.value,
            "history": list(self.history),
        }

    def set_state(self, state: Any) -> None:
        self.value = state["value"]
        self.history = list(state["history"])


@step("critical_checkpoint_rng_step")
class CriticalCheckpointRngStep(Step, Stateful):
    """Records values from every RNG whose state TrainingSession saves."""

    def __init__(self):
        # Deliberately avoid random work in __init__.  A separate regression
        # test below covers constructors that do consume random numbers.
        self.samples: list[tuple[int, int, int]] = []

    def run(self, session: TrainingSession) -> None:
        sample = (
            random.randint(0, 10**9),
            int(np.random.randint(0, 10**9)),
            int(torch.randint(0, 10**9, (1,)).item()),
        )
        self.samples.append(sample)
        session.iteration_context["rng_sample"] = sample

    def get_state(self) -> Any:
        return {"samples": list(self.samples)}

    def set_state(self, state: Any) -> None:
        self.samples = [tuple(sample) for sample in state["samples"]]




@step("critical_checkpoint_optimizer_step")
class CriticalCheckpointOptimizerStep(Step, Stateful):
    """A tiny deterministic optimization step with momentum state."""

    def __init__(
        self,
        initial_value: float = 0.25,
        learning_rate: float = 0.05,
        momentum: float = 0.9,
    ):
        self.initial_value = initial_value
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight = torch.nn.Parameter(
            torch.tensor(initial_value, dtype=torch.float64)
        )
        self.optimizer = torch.optim.SGD(
            [self.weight],
            lr=learning_rate,
            momentum=momentum,
        )
        self.weight_history: list[float] = []

    def run(self, session: TrainingSession) -> None:
        target = torch.tensor(float(session.iteration), dtype=torch.float64)
        loss = (self.weight - target).square()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        current_weight = float(self.weight.detach().item())
        self.weight_history.append(current_weight)
        session.iteration_context["optimized_weight"] = current_weight

    def get_state(self) -> Any:
        return {
            "weight": self.weight.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "weight_history": list(self.weight_history),
        }

    def set_state(self, state: Any) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.weight_history = list(state["weight_history"])


@step("critical_checkpoint_random_init_step")
class CriticalCheckpointRandomInitStep(Step, Stateful):
    """Consumes global RNG state in its constructor to exercise restore order."""

    def __init__(self):
        self.constructor_sample = (
            random.randint(0, 10**9),
            int(np.random.randint(0, 10**9)),
            int(torch.randint(0, 10**9, (1,)).item()),
        )
        self.samples: list[tuple[int, int, int]] = []

    def run(self, session: TrainingSession) -> None:
        sample = (
            random.randint(0, 10**9),
            int(np.random.randint(0, 10**9)),
            int(torch.randint(0, 10**9, (1,)).item()),
        )
        self.samples.append(sample)
        session.iteration_context["rng_sample"] = sample

    def get_state(self) -> Any:
        return {
            "constructor_sample": self.constructor_sample,
            "samples": list(self.samples),
        }

    def set_state(self, state: Any) -> None:
        self.constructor_sample = tuple(state["constructor_sample"])
        self.samples = [tuple(sample) for sample in state["samples"]]


@resource("critical_checkpoint_stateful_resource")
class CriticalCheckpointStatefulResource(Resource, Stateful):
    def __init__(self, token: str, multiplier: int = 1):
        self.token = token
        self.multiplier = multiplier
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self, session: TrainingSession):
        self.setup_calls += 1
        session.session_context["resource_token"] = self.token

    def teardown(self, session: TrainingSession):
        self.teardown_calls += 1

    def get_state(self) -> Any:
        return {
            "setup_calls": self.setup_calls,
            "teardown_calls": self.teardown_calls,
        }

    def set_state(self, state: Any) -> None:
        self.setup_calls = state["setup_calls"]
        self.teardown_calls = state["teardown_calls"]


@hook("critical_checkpoint_stateful_hook")
class CriticalCheckpointStatefulHook(LifecycleHook, Stateful):
    def __init__(self, label: str, call_every: int = 1):
        self.label = label
        self.call_every = call_every
        self.setup_calls = 0
        self.teardown_calls = 0
        self.observed_values: list[int] = []

    def setup(self, session: TrainingSession):
        self.setup_calls += 1

    def teardown(self, session: TrainingSession):
        self.teardown_calls += 1

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        pass

    def post_iteration_callback(self, session: TrainingSession) -> None:
        self.observed_values.append(session.iteration_context["accumulator"])

    def get_state(self) -> Any:
        return {
            "setup_calls": self.setup_calls,
            "teardown_calls": self.teardown_calls,
            "observed_values": list(self.observed_values),
        }

    def set_state(self, state: Any) -> None:
        self.setup_calls = state["setup_calls"]
        self.teardown_calls = state["teardown_calls"]
        self.observed_values = list(state["observed_values"])


@hook("critical_checkpoint_stateless_hook")
class CriticalCheckpointStatelessHook(SessionHook):
    """Its constructor configuration is restored, but runtime events are not."""

    def __init__(self, label: str):
        self.label = label
        self.runtime_events: list[str] = []

    def setup(self, session: TrainingSession):
        self.runtime_events.append("setup")

    def teardown(self, session: TrainingSession):
        self.runtime_events.append("teardown")


def make_config(
    directory: Path,
    *,
    max_iterations: int = 6,
    seed: int = 2026,
) -> dict[str, Any]:
    return {
        "rng_seed": seed,
        "sessions_dir": str(directory),
        "max_iterations": max_iterations,
        "device": "cpu",
    }


def run_to_completion(session: TrainingSession) -> list[int]:
    with session:
        completed_iterations = list(session)

    assert session._phase is SessionPhase.FINISHED
    return completed_iterations


def test_pickle_resume_matches_uninterrupted_run_and_preserves_rng_streams(tmp_path):
    """A resumed run must produce exactly the same RNG samples as a full run."""

    baseline = TrainingSession(
        make_config(tmp_path / "baseline", max_iterations=6, seed=314159)
    )
    baseline_step = CriticalCheckpointRngStep()
    baseline.add_step(baseline_step)

    assert run_to_completion(baseline) == [1, 2, 3, 4, 5, 6]
    expected_samples = list(baseline_step.samples)

    partial = TrainingSession(
        make_config(tmp_path / "partial", max_iterations=6, seed=314159)
    )
    partial_step = CriticalCheckpointRngStep()
    partial.add_step(partial_step)

    with partial:
        assert [next(partial) for _ in range(3)] == [1, 2, 3]
        checkpoint_payload = pickle.dumps(partial)

    assert partial._phase is SessionPhase.PAUSED

    restored = pickle.loads(checkpoint_payload)
    restored_step = restored._steps["critical_checkpoint_rng_step"]

    assert restored is not partial
    assert restored._phase is SessionPhase.NEW
    assert restored.iteration == 3
    assert restored_step.samples == expected_samples[:3]
    with pytest.raises(RuntimeError, match="Use within"):
        _ = restored.iteration_context

    assert run_to_completion(restored) == [4, 5, 6]
    assert restored_step.samples == expected_samples


def test_checkpoint_resume_preserves_model_and_optimizer_state(tmp_path):
    """Momentum and tensor state must match an uninterrupted optimization run."""

    baseline = TrainingSession(
        make_config(tmp_path / "optimizer-baseline", max_iterations=6, seed=44)
    )
    baseline_step = CriticalCheckpointOptimizerStep(
        initial_value=0.25,
        learning_rate=0.05,
        momentum=0.8,
    )
    baseline.add_step(baseline_step)
    run_to_completion(baseline)

    partial = TrainingSession(
        make_config(tmp_path / "optimizer-partial", max_iterations=6, seed=44)
    )
    partial_step = CriticalCheckpointOptimizerStep(
        initial_value=0.25,
        learning_rate=0.05,
        momentum=0.8,
    )
    partial.add_step(partial_step)

    with partial:
        assert [next(partial) for _ in range(3)] == [1, 2, 3]
        checkpoint_payload = pickle.dumps(partial)

    restored = pickle.loads(checkpoint_payload)
    restored_step = restored._steps["critical_checkpoint_optimizer_step"]

    assert restored_step.initial_value == 0.25
    assert restored_step.learning_rate == 0.05
    assert restored_step.momentum == 0.8
    assert restored_step.weight_history == pytest.approx(
        baseline_step.weight_history[:3]
    )

    run_to_completion(restored)

    assert restored_step.weight_history == pytest.approx(
        baseline_step.weight_history
    )
    torch.testing.assert_close(restored_step.weight, baseline_step.weight)

    baseline_optimizer_state = baseline_step.optimizer.state_dict()
    restored_optimizer_state = restored_step.optimizer.state_dict()

    assert restored_optimizer_state["param_groups"] == (
        baseline_optimizer_state["param_groups"]
    )
    assert restored_optimizer_state["state"].keys() == (
        baseline_optimizer_state["state"].keys()
    )

    for parameter_id in baseline_optimizer_state["state"]:
        torch.testing.assert_close(
            restored_optimizer_state["state"][parameter_id]["momentum_buffer"],
            baseline_optimizer_state["state"][parameter_id]["momentum_buffer"],
        )


def test_builtin_checkpointer_loads_mid_run_checkpoint_and_resumes_exactly(tmp_path):
    """Exercise the real Checkpointer save/load path, not only pickle directly."""

    checkpoints_dir = tmp_path / "checkpoints"
    session = TrainingSession(
        make_config(tmp_path / "session", max_iterations=5, seed=77)
    )
    step_obj = CriticalCheckpointAccumulatorStep(start=10, increment=3)
    checkpointer = Checkpointer(
        {
            "checkpoint_every": 2,
            "checkpoints_dir": str(checkpoints_dir),
        }
    )

    session.add_step(step_obj)
    session.register_hook(checkpointer)

    assert run_to_completion(session) == [1, 2, 3, 4, 5]
    expected_final_state = step_obj.get_state()

    checkpoint_paths = sorted(path for path in checkpoints_dir.iterdir() if path.is_file())
    assert len(checkpoint_paths) == 4

    loaded_by_iteration = {
        loaded.iteration: loaded
        for loaded in (
            Checkpointer.load_checkpoint(path, map_location="cpu")
            for path in checkpoint_paths
        )
    }

    # Checkpointer runs on the first iteration, the last iteration, and every
    # iteration divisible by checkpoint_every.
    assert set(loaded_by_iteration) == {1, 2, 4, 5}

    restored = loaded_by_iteration[2]
    restored_step = restored._steps["critical_checkpoint_accumulator_step"]
    restored_checkpointer = restored._hooks["checkpointer"]

    assert restored._phase is SessionPhase.NEW
    assert restored_step.start == 10
    assert restored_step.increment == 3
    assert restored_step.history == [13, 16]
    assert restored_checkpointer.call_every == 2

    assert run_to_completion(restored) == [3, 4, 5]
    assert restored_step.get_state() == expected_final_state


def test_checkpoint_restores_constructor_args_stateful_state_and_stateless_config(
    tmp_path,
):
    """Verify mixed component reconstruction from one actual session payload."""

    session = TrainingSession(
        make_config(tmp_path / "mixed", max_iterations=3, seed=101)
    )
    resource_obj = CriticalCheckpointStatefulResource("resource-A", multiplier=9)
    stateful_hook = CriticalCheckpointStatefulHook("observer", call_every=1)
    stateless_hook = CriticalCheckpointStatelessHook("stateless-config")
    step_obj = CriticalCheckpointAccumulatorStep(start=1, increment=4)

    resource_id = session.register_resource(resource_obj)
    session.register_hook(stateful_hook)
    session.register_hook(stateless_hook)
    session.add_step(step_obj)

    with session:
        assert next(session) == 1
        assert session.session_context == {
            "resource_token": "resource-A",
            "last_accumulator": 5,
        }
        checkpoint_payload = pickle.dumps(session)

    # These mutations happen after the checkpoint bytes were captured and must
    # therefore not appear in the restored objects.
    assert resource_obj.teardown_calls == 1
    assert stateful_hook.teardown_calls == 1
    assert stateless_hook.runtime_events == ["setup", "teardown"]
    assert session.session_context == {}

    restored = pickle.loads(checkpoint_payload)
    restored_resource = restored.get_resource(resource_id)
    restored_stateful_hook = restored._hooks["critical_checkpoint_stateful_hook"]
    restored_stateless_hook = restored._hooks["critical_checkpoint_stateless_hook"]
    restored_step = restored._steps["critical_checkpoint_accumulator_step"]

    assert type(restored_resource) is CriticalCheckpointStatefulResource
    assert type(restored_stateful_hook) is CriticalCheckpointStatefulHook
    assert type(restored_stateless_hook) is CriticalCheckpointStatelessHook
    assert type(restored_step) is CriticalCheckpointAccumulatorStep

    assert restored_resource.token == "resource-A"
    assert restored_resource.multiplier == 9
    assert restored_resource.setup_calls == 1
    assert restored_resource.teardown_calls == 0

    assert restored_stateful_hook.label == "observer"
    assert restored_stateful_hook.call_every == 1
    assert restored_stateful_hook.setup_calls == 1
    assert restored_stateful_hook.teardown_calls == 0
    assert restored_stateful_hook.observed_values == [5]

    # Non-Stateful components are reconstructed from constructor arguments.
    # Runtime mutations from the original instance intentionally do not persist.
    assert restored_stateless_hook.label == "stateless-config"
    assert restored_stateless_hook.runtime_events == []

    assert restored_step.start == 1
    assert restored_step.increment == 4
    assert restored_step.value == 5
    assert restored_step.history == [5]
    assert restored.session_context == {
        "resource_token": "resource-A",
        "last_accumulator": 5,
    }

    assert run_to_completion(restored) == [2, 3]
    assert restored_step.history == [5, 9, 13]
    assert restored_stateful_hook.observed_values == [5, 9, 13]
    assert restored_resource.setup_calls == 2
    assert restored_resource.teardown_calls == 1
    assert restored_stateful_hook.setup_calls == 2
    assert restored_stateful_hook.teardown_calls == 1
    assert restored_stateless_hook.runtime_events == ["setup", "teardown"]
    assert restored.session_context == {}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TrainingSession.get_state() currently returns session_context by reference; "
        "the snapshot changes when the live context is mutated or cleared."
    ),
)
def test_get_state_returns_a_detached_session_context_snapshot(tmp_path):
    """A state snapshot should represent values at get_state() call time."""

    session = TrainingSession(
        make_config(tmp_path / "snapshot", max_iterations=1, seed=3)
    )

    with session:
        session.session_context["nested"] = {"values": [1]}
        state = session.get_state()

        session.session_context["nested"]["values"].append(2)
        assert state["session_context"] == {"nested": {"values": [1]}}

    assert session.session_context == {}
    assert state["session_context"] == {"nested": {"values": [1]}}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TrainingSession.set_state() restores global RNG state before rebuilding "
        "components, so a component constructor that consumes randomness advances "
        "the restored stream."
    ),
)
def test_component_constructors_do_not_advance_restored_rng_streams(tmp_path):
    """Component reconstruction must not perturb checkpointed global RNG state."""

    baseline = TrainingSession(
        make_config(tmp_path / "rng-baseline", max_iterations=5, seed=999)
    )
    baseline_step = CriticalCheckpointRandomInitStep()
    baseline.add_step(baseline_step)
    run_to_completion(baseline)

    partial = TrainingSession(
        make_config(tmp_path / "rng-partial", max_iterations=5, seed=999)
    )
    partial_step = CriticalCheckpointRandomInitStep()
    partial.add_step(partial_step)

    with partial:
        assert [next(partial) for _ in range(2)] == [1, 2]
        checkpoint_payload = pickle.dumps(partial)

    restored = pickle.loads(checkpoint_payload)
    restored_step = restored._steps["critical_checkpoint_random_init_step"]
    run_to_completion(restored)

    assert restored_step.constructor_sample == baseline_step.constructor_sample
    assert restored_step.samples == baseline_step.samples
