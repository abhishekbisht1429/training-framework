from __future__ import annotations

import itertools
import os
import re
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

import training_framework
from training_framework.configurator import Configurator
from training_framework.dataloader import InfiniteSampler
from training_framework.training_engine import TrainingEngine
from training_framework.training_session import (
    HOOK_REGISTRY,
    RESOURCE_REGISTRY,
    STEP_REGISTRY,
    SessionPhase,
    TrainingSession,
    hook,
    resource,
    step, Stateful, Hook, LifecycleHook, Resource, Step,
)
from training_framework.util import timestamp_str


class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 12, num_features: int = 5, num_classes: int = 2):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, num_features)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        xs, ys = zip(*batch)
        return torch.stack(list(xs)), torch.stack(list(ys))


class AdditionalStepBase(Step, Stateful):
    def __init__(self):
        self.calls = 0
        self.last_seen_loss = None

    def run(self, session: TrainingSession) -> None:
        self.calls += 1
        session.iteration_context["step_called"] = True
        session.iteration_context["step_index"] = self.calls
        self.last_seen_loss = self.calls * 1.0

    def get_state(self) -> Any:
        return {"calls": self.calls, "last_seen_loss": self.last_seen_loss}

    def set_state(self, state: Any) -> None:
        self.calls = state["calls"]
        self.last_seen_loss = state["last_seen_loss"]



class AdditionalResourceBase(Resource, Stateful):
    def __init__(self):
        self.setup_calls = 0
        self.teardown_calls = 0
        self.events: list[str] = []
        self.session_dirs: list[str] = []

    def setup(self, session: TrainingSession):
        self.setup_calls += 1
        self.events.append("setup")
        self.session_dirs.append(session.session_config.session_dir)

    def teardown(self, session):
        self.teardown_calls += 1
        self.events.append("teardown")

    def get_state(self) -> Any:
        return {
            "setup_calls": self.setup_calls,
            "teardown_calls": self.teardown_calls,
            "events": list(self.events),
            "session_dirs": list(self.session_dirs),
        }

    def set_state(self, state: Any) -> None:
        self.setup_calls = state["setup_calls"]
        self.teardown_calls = state["teardown_calls"]
        self.events = list(state["events"])
        self.session_dirs = list(state["session_dirs"])


class AdditionalHookBase(LifecycleHook, Stateful):
    def __init__(self, call_every: int = 1):
        self.call_every = call_every
        self.events: list[str] = []
        self.pre_iterations: list[int] = []
        self.post_iterations: list[int] = []
        self.shared_snapshots: list[dict[str, Any]] = []

    def setup(self, session: TrainingSession):
        self.events.append("setup")

    def teardown(self, session):
        self.events.append("teardown")

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        self.events.append(f"pre:{session.iteration}")
        self.pre_iterations.append(session.iteration)

    def post_iteration_callback(self, session: TrainingSession) -> None:
        self.events.append(f"post:{session.iteration}")
        self.post_iterations.append(session.iteration)
        self.shared_snapshots.append(dict(session._shared_state))

    def get_state(self) -> Any:
        return {
            "call_every": self.call_every,
            "events": list(self.events),
            "pre_iterations": list(self.pre_iterations),
            "post_iterations": list(self.post_iterations),
            "shared_snapshots": [dict(item) for item in self.shared_snapshots],
        }

    def set_state(self, state: Any) -> None:
        self.call_every = state["call_every"]
        self.events = list(state["events"])
        self.pre_iterations = list(state["pre_iterations"])
        self.post_iterations = list(state["post_iterations"])
        self.shared_snapshots = [dict(item) for item in state["shared_snapshots"]]


@pytest.fixture
def minimal_session_config_1(tmp_path):
    return {
        "base_config": {
            "max_iterations": 3,
            "sessions_dir": str(tmp_path / "sessions"),
            "device": "cpu",
            "rng_seed": 7,
            "components_package": "training_framework.builtin_components",
        }
    }


@pytest.fixture
def minimal_session_config_2(tmp_path):
    return {
        "base_config": {
            "max_iterations": 2,
            "batch_size": 4,
            "sessions_dir": str(tmp_path / "sessions"),
            "device": "cpu",
            "rng_seed": 11,
            "components_package": "training_framework.builtin_components",
        }
    }

def _write_yaml(tmp_path: Path, data: dict, name: str = "config.yaml") -> str:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return str(path)


def test_timestamp_str_has_expected_shape():
    assert re.fullmatch(r"\d{8}_\d{6}_\d{9}", timestamp_str())


def test_registry_decorators_register_classes_and_reject_duplicates():
    @step("test_additional_step")
    class AdditionalStep(AdditionalStepBase):
        pass

    @resource("test_additional_resource")
    class AdditionalResource(AdditionalResourceBase):
        pass

    @hook("test_additional_hook")
    class AdditionalHook(AdditionalHookBase):
        pass

    assert STEP_REGISTRY["test_additional_step"] is AdditionalStep
    assert RESOURCE_REGISTRY["test_additional_resource"] is AdditionalResource
    assert HOOK_REGISTRY["test_additional_hook"] is AdditionalHook

    with pytest.raises(ValueError):
        @step("test_additional_step")
        class _DuplicateStep(Step):
            def run(self, session: TrainingSession) -> None:
                pass

            def get_state(self):
                return {}

            def set_state(self, state):
                pass


def test_training_session_initialization_and_device_validation(minimal_session_config_2, monkeypatch):
    session = TrainingSession(minimal_session_config_2)

    assert session.session_config.rng_seed == minimal_session_config_2["base_config"]["rng_seed"]
    assert session.session_config.max_iterations == minimal_session_config_2["base_config"]["max_iterations"]
    assert session.session_config.session_dir.startswith(minimal_session_config_2["base_config"]["sessions_dir"])
    assert session.device.type == "cpu"
    assert session._phase is SessionPhase.NEW

    bad_device = deepcopy(minimal_session_config_2)
    bad_device["base_config"]["device"] = "tpu"

    assert TrainingSession(bad_device).device == torch.device("cpu")

    unavailable_cuda = deepcopy(minimal_session_config_2)
    unavailable_cuda["base_config"]["device"] = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert TrainingSession(unavailable_cuda).device == torch.device("cpu")


def test_requires_context_for_shared_state_and_iteration(minimal_session_config_2):
    session = TrainingSession(minimal_session_config_2)

    with pytest.raises(RuntimeError, match="Use within"):
        session.iteration_context["x"] = 1

    with pytest.raises(RuntimeError, match="Use within"):
        session.iteration_context["x"]

    with pytest.raises(RuntimeError, match="Use within"):
        next(session)


def test_registration_validation_and_lookup(minimal_session_config_2):
    @step("test_additional_step")
    class AdditionalStep(AdditionalStepBase):
        pass

    @resource("test_additional_resource")
    class AdditionalResource(AdditionalResourceBase):
        pass

    @hook("test_additional_hook")
    class AdditionalHook(AdditionalHookBase):
        pass

    session = TrainingSession(minimal_session_config_2)

    with pytest.raises(TypeError):
        session.add_step(object())

    with pytest.raises(TypeError):
        session.register_hook(object())

    with pytest.raises(TypeError):
        session.register_resource(object())

    class UnregisteredStep(Step):
        def run(self, session: TrainingSession) -> None:
            pass

        def get_state(self):
            return {}

        def set_state(self, state):
            pass

    class UnregisteredHook(LifecycleHook):
        def __init__(self):
            self.call_every = 1

        def setup(self, session: TrainingSession):
            pass

        def teardown(self, session):
            pass

        def pre_iteration_callback(self, session: TrainingSession) -> None:
            pass

        def post_iteration_callback(self, session: TrainingSession) -> None:
            pass

    class UnregisteredResource(Resource):
        def setup(self, session: TrainingSession):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {}

        def set_state(self, state):
            pass

    with pytest.raises(ValueError, match="STEP_REGISTRY"):
        session.add_step(UnregisteredStep())

    with pytest.raises(ValueError, match="HOOK_REGISTRY"):
        session.register_hook(UnregisteredHook())

    with pytest.raises(ValueError, match="RESOURCE_REGISTRY"):
        session.register_resource(UnregisteredResource())

    step_obj = AdditionalStep()
    hook_obj = AdditionalHook(call_every=1)
    resource_obj = AdditionalResource()

    session.add_step(step_obj)
    session.register_hook(hook_obj)
    resource_id = session.register_resource(resource_obj)

    assert resource_id == "test_additional_resource"
    assert session.get_resource(resource_id) is resource_obj
    with pytest.raises(KeyError):
        session.get_resource("missing-resource")

    assert len(session._steps) == 1
    assert len(session._hooks) == 1
    assert len(session._resources) == 1

def test_context_lifecycle_and_iteration_order(minimal_session_config_1):
    @step("test_additional_step")
    class AdditionalStep(AdditionalStepBase):
        pass

    @resource("additional_resource_a")
    class AdditionalResourceA(AdditionalResourceBase):
        pass

    @resource("additional_resource_b")
    class AdditionalResourceB(AdditionalResourceBase):
        pass

    @hook("additional_hook_a")
    class AdditionalHookA(AdditionalHookBase):
        pass

    @hook("additional_hook_b")
    class AdditionalHookB(AdditionalHookBase):
        pass

    @step("toy_model_step")
    class ToyModelStep(Step, Stateful):
        def __init__(self):
            dataset = DummyDataset()
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                sampler=InfiniteSampler(len(dataset)),
                collate_fn=dataset.collate_fn,
            )
            self._iterator = iter(dataloader)
            self._model = nn.Sequential(nn.Linear(5, 2))
            self.seen_losses: list[float] = []

        def run(self, session: TrainingSession) -> None:
            x, y = next(self._iterator)
            x = x.to(session.device)
            y = y.to(session.device)
            output = self._model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            self.seen_losses.append(float(loss.item()))
            session.iteration_context["loss"] = float(loss.item())

        def get_state(self) -> Any:
            return {"seen_losses": list(self.seen_losses)}

        def set_state(self, state: Any) -> None:
            self.seen_losses = list(state["seen_losses"])

    session = TrainingSession(minimal_session_config_1)

    # Each resource and hook has a distinct registered name.
    resource_a = AdditionalResourceA()
    resource_b = AdditionalResourceB()
    hook_a = AdditionalHookA(call_every=1)
    hook_b = AdditionalHookB(call_every=2)

    step_a = AdditionalStep()
    step_b = ToyModelStep()

    resource_a_id = session.register_resource(resource_a)
    resource_b_id = session.register_resource(resource_b)

    session.register_hook(hook_a)
    session.register_hook(hook_b)

    session.add_step(step_a)
    session.add_step(step_b)

    # Verify the session contains two distinct logical resources/hooks,
    # rather than duplicate instances of one registered component.
    assert resource_a.name == "additional_resource_a"
    assert resource_b.name == "additional_resource_b"
    assert resource_a_id != resource_b_id

    assert [test_hook.name for test_hook in session._hooks.values()] == [
        "additional_hook_a",
        "additional_hook_b",
    ]

    with session:
        assert session._phase is SessionPhase.READY

        assert resource_a.setup_calls == 1
        assert resource_b.setup_calls == 1
        assert resource_a.events == ["setup"]
        assert resource_b.events == ["setup"]

        assert hook_a.events == ["setup"]
        assert hook_b.events == ["setup"]

        first = next(session)

        assert first == 1
        assert session.iteration == 1

        # Iteration-scoped shared state must be cleared after next() returns.
        assert session._shared_state == {}

        second = next(session)

        assert second == 2
        assert session.iteration == 2
        assert session._shared_state == {}

    assert session._phase is SessionPhase.PAUSED

    assert resource_a.events == ["setup", "teardown"]
    assert resource_b.events == ["setup", "teardown"]
    assert resource_a.setup_calls == 1
    assert resource_b.setup_calls == 1
    assert resource_a.teardown_calls == 1
    assert resource_b.teardown_calls == 1

    with session:
        assert session.iteration == 2
        assert session._phase is SessionPhase.READY

        assert resource_a.setup_calls == 2
        assert resource_b.setup_calls == 2

        assert hook_a.events == [
            "setup",
            "pre:1",
            "post:1",
            "pre:2",
            "post:2",
            "teardown",
            "setup",
        ]
        assert hook_b.events == [
            "setup",
            "pre:1",
            "post:1",
            "pre:2",
            "post:2",
            "teardown",
            "setup",
        ]

        third = next(session)

        assert third == 3
        assert session.iteration == 3
        assert session._shared_state == {}

    assert resource_a.events == [
        "setup",
        "teardown",
        "setup",
        "teardown",
    ]
    assert resource_b.events == [
        "setup",
        "teardown",
        "setup",
        "teardown",
    ]
    assert resource_a.setup_calls == 2
    assert resource_b.setup_calls == 2
    assert resource_a.teardown_calls == 2
    assert resource_b.teardown_calls == 2

    # The first and final iterations call every hook regardless of
    # call_every. Hook B is also called on iteration 2 because call_every=2.
    expected_hook_events = [
        "setup",
        "pre:1",
        "post:1",
        "pre:2",
        "post:2",
        "teardown",
        "setup",
        "pre:3",
        "post:3",
        "teardown",
    ]

    assert hook_a.events == expected_hook_events
    assert hook_b.events == expected_hook_events

    assert hook_a.pre_iterations == [1, 2, 3]
    assert hook_a.post_iterations == [1, 2, 3]
    assert hook_b.pre_iterations == [1, 2, 3]
    assert hook_b.post_iterations == [1, 2, 3]

    # Values shared by the steps were visible to the hooks before the
    # iteration-scoped state was cleared.
    for test_hook in (hook_a, hook_b):
        assert [snapshot["step_index"] for snapshot in test_hook.shared_snapshots] == [
            1,
            2,
            3,
        ]
        assert all(
            snapshot["step_called"] is True
            for snapshot in test_hook.shared_snapshots
        )

def test_state_round_trip_restores_nested_resources_steps_and_hooks(minimal_session_config_1):
    @step("test_additional_step")
    class AdditionalStep(AdditionalStepBase):
        pass

    @resource("test_additional_resource")
    class AdditionalResource(AdditionalResourceBase):
        pass

    @hook("test_additional_hook")
    class AdditionalHook(AdditionalHookBase):
        pass

    session = TrainingSession(minimal_session_config_1)
    additional_resource = AdditionalResource()
    additional_hook = AdditionalHook(call_every=1)
    additional_step = AdditionalStep()

    session.register_resource(additional_resource)
    session.register_hook(additional_hook)
    session.add_step(additional_step)

    with session:
        next(session)
        next(session)

    state = session.get_state()
    restored = TrainingSession(minimal_session_config_1)
    restored.set_state(state)

    assert restored.iteration == session.iteration
    assert restored.session_config == session.session_config
    assert len(restored._resources) == 1
    assert len(restored._steps) == 1
    assert len(restored._hooks) == 1

    restored_resource = next(iter(restored._resources.values()))
    restored_step = list(restored._steps.values())[0]
    restored_hook = list(restored._hooks.values())[0]

    assert isinstance(restored_resource, AdditionalResource)
    assert isinstance(restored_step, AdditionalStep)
    assert isinstance(restored_hook, AdditionalHook)
    assert restored_resource.events == additional_resource.events
    assert restored_step.calls == additional_step.calls
    assert restored_hook.events == additional_hook.events
    assert restored_hook.call_every == additional_hook.call_every


def test_infinite_sampler_yields_permutations_forever():
    sampler = InfiniteSampler(5)
    iterator = iter(sampler)

    first_round = list(itertools.islice(iterator, 5))
    second_round = list(itertools.islice(iterator, 5))

    assert sorted(first_round) == [0, 1, 2, 3, 4]
    assert sorted(second_round) == [0, 1, 2, 3, 4]
    assert len(set(first_round)) == 5
    assert len(set(second_round)) == 5


def test_configurator_reads_overrides_and_returns_deep_copies(tmp_path, monkeypatch):
    sample_config = {
        "sessions": [
            {
                "max_iterations": 5,
                "batch_size": 4,
                "sessions_dir": str(tmp_path / "sessions_1"),
                "device": "cpu",
                "rng_seed": 123,
                "logger": {"log_every": 1, "log_file": str(tmp_path / "log_1.txt"), "nested": {"enabled": True}},
                "checkpointer": {"checkpoint_every": 2, "checkpoints_dir": str(tmp_path / "ckpts_1")},
                "tensorboard": {"host": "0.0.0.0", "port": 16040},
            },
            {
                "max_iterations": 7,
                "batch_size": 8,
                "sessions_dir": str(tmp_path / "sessions_2"),
                "device": "cpu",
                "rng_seed": 456,
                "logger": {"log_every": 3, "log_file": str(tmp_path / "log_2.txt")},
                "checkpointer": {"checkpoint_every": 4, "checkpoints_dir": str(tmp_path / "ckpts_2")},
                "tensorboard": {"host": "0.0.0.0", "port": 16041},
            },
        ]
    }

    config_path = _write_yaml(tmp_path, sample_config)
    monkeypatch.setattr(sys, "argv", ["pytest", "--config", config_path, "--override", "sessions[0].checkpointer.checkpoint_every=11", "sessions.1.logger.log_every=9"])

    configurator = Configurator()
    session_config = configurator.get_base_config(0)
    resource_config = configurator.get_component_config(0, "logger")

    assert session_config["checkpointer"]["checkpoint_every"] == 11
    assert configurator.get_component_config(1, "logger")["log_every"] == 9
    assert resource_config == sample_config["sessions"][0]["logger"]

    resource_config["nested"]["enabled"] = False
    assert sample_config["sessions"][0]["logger"]["nested"]["enabled"] is True

    with pytest.raises(KeyError):
        configurator.get_component_config(0, "missing")

    sample_config["sessions"][0]["logger"]["log_every"] = 99
    assert configurator.get_base_config(0)["logger"]["log_every"] == 1


def test_configurator_create_sessions_attaches_expected_components(tmp_path, monkeypatch):
    # The registry is cleared before every test, so to re-register the builtin components a
    # reload is required
    import importlib
    importlib.reload(training_framework.builtin_components)
    from training_framework.builtin_components import Checkpointer, Logger, Tensorboard

    sample_config = {
        "sessions": [
            {
                "base_config": {
                    "max_iterations": 2,
                    "batch_size": 4,
                    "sessions_dir": str(tmp_path / "s1"),
                    "device": "cpu",
                    "rng_seed": 1,
                    "components_package": "training_framework.builtin_components",
                },
                "logger": {"log_every": 1, "log_file": str(tmp_path / "log1.txt")},
                "checkpointer": {"checkpoint_every": 1, "checkpoints_dir": str(tmp_path / "ckpts1")},
                "tensorboard": {"host": "0.0.0.0", "port": 16050},
            },
            {
                "base_config": {
                    "max_iterations": 2,
                    "batch_size": 4,
                    "sessions_dir": str(tmp_path / "s1"),
                    "device": "cpu",
                    "rng_seed": 1,
                    "components_package": "training_framework.builtin_components",
                },
                "checkpointer": {"checkpoint_every": 2, "checkpoints_dir": str(tmp_path / "ckpts2")},
            },
        ]
    }

    config_path = _write_yaml(tmp_path, sample_config, "config_create_sessions.yaml")
    monkeypatch.setattr(sys, "argv", ["pytest", "--config", config_path])

    configurator = Configurator()
    sessions = [TrainingSession(config) for config in configurator.session_configs]

    assert len(sessions) == 2
    assert len(sessions[0]._hooks) == 2
    assert len(sessions[0]._resources) == 1
    assert any(isinstance(h, Logger) for h in sessions[0]._hooks.values())
    assert any(isinstance(h, Checkpointer) for h in sessions[0]._hooks.values())
    assert any(isinstance(r, Tensorboard) for r in sessions[0]._resources.values())

    assert len(sessions[1]._hooks) == 1
    assert len(sessions[1]._resources) == 0
    assert any(isinstance(h, Checkpointer) for h in sessions[1]._hooks.values())
