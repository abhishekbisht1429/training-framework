import pickle
import random

import numpy as np
import pytest
import torch

from training_framework.training_session import (
    Resource,
    SessionHook,
    Stateful,
    Step,
    TrainingSession,
    hook,
    resource,
    step, StatefulResource,
)


def make_config(tmp_path, max_iterations=2, seed=123):
    return {
        "rng_seed": seed,
        "sessions_dir": str(tmp_path),
        "max_iterations": max_iterations,
        "device": "cpu",
    }


@step("checkpoint_rng_step")
class CheckpointRngStep(Step, Stateful):
    def __init__(self, label, scale=1):
        self.label = label
        self.scale = scale
        self.samples = []

    def run(self, session):
        sample = (
            random.randint(0, 10**6),
            int(np.random.randint(0, 10**6)),
            int(torch.randint(0, 10**6, (1,)).item()),
        )
        self.samples.append(sample)
        session.iteration_context[f"{self.label}_sample"] = sample

    def get_state(self):
        return {"samples": list(self.samples)}

    def set_state(self, state):
        self.samples = list(state["samples"])


@resource("checkpoint_resource")
class CheckpointResource(StatefulResource):
    def __init__(self, prefix, multiplier=2):
        self.prefix = prefix
        self.multiplier = multiplier
        self.setup_calls = 0
        self.teardown_calls = 0
        self.last_seen_iteration = None

    def setup(self, session):
        self.setup_calls += 1
        self.last_seen_iteration = session.iteration

    def teardown(self, session):
        self.teardown_calls += 1

    def get_state(self):
        return {
            "setup_calls": self.setup_calls,
            "teardown_calls": self.teardown_calls,
            "last_seen_iteration": self.last_seen_iteration,
        }

    def set_state(self, state):
        self.setup_calls = state["setup_calls"]
        self.teardown_calls = state["teardown_calls"]
        self.last_seen_iteration = state["last_seen_iteration"]


@hook("checkpoint_hook")
class CheckpointHook(SessionHook, Stateful):
    def __init__(self, token, level=1):
        self.token = token
        self.level = level
        self.setup_calls = 0
        self.teardown_calls = 0
        self.seen_session_dirs = []

    def setup(self, session):
        self.setup_calls += 1
        self.seen_session_dirs.append(session.session_config.session_dir)

    def teardown(self, session):
        self.teardown_calls += 1

    def get_state(self):
        return {
            "setup_calls": self.setup_calls,
            "teardown_calls": self.teardown_calls,
            "seen_session_dirs": list(self.seen_session_dirs),
        }

    def set_state(self, state):
        self.setup_calls = state["setup_calls"]
        self.teardown_calls = state["teardown_calls"]
        self.seen_session_dirs = list(state["seen_session_dirs"])


class BaseInheritedResource(StatefulResource):
    def __init__(self, label, factor=11):
        self.label = label
        self.factor = factor
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self, session):
        self.setup_calls += 1

    def teardown(self, session):
        self.teardown_calls += 1

    def get_state(self):
        return {
            "setup_calls": self.setup_calls,
            "teardown_calls": self.teardown_calls,
        }

    def set_state(self, state):
        self.setup_calls = state["setup_calls"]
        self.teardown_calls = state["teardown_calls"]


@resource("inherited_checkpoint_resource")
class InheritedCheckpointResource(BaseInheritedResource):
    pass


def test_checkpoint_pickle_round_trip_restores_resources_hooks_and_state(tmp_path):
    session = TrainingSession(make_config(tmp_path / "full", max_iterations=3, seed=42))

    resource_obj = CheckpointResource("alpha", multiplier=9)
    hook_obj = CheckpointHook("beta", level=5)
    step_obj = CheckpointRngStep("gamma", scale=7)

    resource_id = session.register_resource(resource_obj)
    session.register_hook(hook_obj)
    session.add_step(step_obj)

    with session:
        assert resource_obj.setup_calls == 1
        assert hook_obj.setup_calls == 1
        assert session.iteration == 0

        assert next(session) == 1
        assert step_obj.samples
        assert resource_obj.last_seen_iteration == 0

    assert resource_obj.teardown_calls == 1
    assert hook_obj.teardown_calls == 1

    payload = pickle.dumps(session)
    restored = pickle.loads(payload)

    assert restored.iteration == 1
    assert restored.session_config.max_iterations == 3

    restored_resource = restored.get_resource(resource_id)
    restored_hook = list(restored._hooks.values())[0]
    restored_step = list(restored._steps.values())[0]

    assert restored_resource.prefix == "alpha"
    assert restored_resource.multiplier == 9
    assert restored_resource.setup_calls == 1
    assert restored_resource.teardown_calls == 1
    assert restored_resource.last_seen_iteration == 0

    assert restored_hook.token == "beta"
    assert restored_hook.level == 5
    assert restored_hook.setup_calls == 1
    assert restored_hook.teardown_calls == 1
    assert restored_hook.seen_session_dirs == [session.session_config.session_dir]

    assert restored_step.label == "gamma"
    assert restored_step.scale == 7
    assert restored_step.samples == step_obj.samples


def test_checkpoint_restores_inherited_constructor_args(tmp_path):
    session = TrainingSession(make_config(tmp_path / "inherited", max_iterations=1, seed=99))
    resource_obj = InheritedCheckpointResource("delta", factor=13)
    resource_id = session.register_resource(resource_obj)

    with session:
        assert resource_obj.setup_calls == 1

    assert resource_obj.teardown_calls == 1

    restored = pickle.loads(pickle.dumps(session))
    restored_resource = restored.get_resource(resource_id)

    assert restored_resource.label == "delta"
    assert restored_resource.factor == 13
    assert restored_resource.setup_calls == 1
    assert restored_resource.teardown_calls == 1