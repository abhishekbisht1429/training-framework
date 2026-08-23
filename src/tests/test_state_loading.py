import pickle
import random

import numpy as np
import pytest
import torch

from tests.test_utils import make_config
from training_framework.training_session import (
    TrainingSession,
    hook,
    resource,
    step, Stateful, SessionHook, Step, StatefulResource, )


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


def test_checkpoint_uses_generic_component_state_schema(tmp_path):
    session = TrainingSession(make_config(tmp_path / "generic-schema"))
    state = session.get_state()

    assert "components_state" in state
    assert "resources_state" not in state
    assert "hooks_state" not in state
    assert "steps_state" not in state
    assert state["components_state"]["logger"]["component_type"] == "Hook"


def test_checkpoint_rejects_legacy_component_state_schema(tmp_path):
    session = TrainingSession(make_config(tmp_path / "legacy-schema"))
    state = session.get_state()
    del state["components_state"]
    state["resources_state"] = {}
    state["hooks_state"] = {}
    state["steps_state"] = {}

    with pytest.raises(
            ValueError,
            match="unsupported component state schema.*components_state",
    ):
        TrainingSession.from_state(state)


def test_checkpoint_rejects_component_category_changes(tmp_path):
    session = TrainingSession(make_config(tmp_path / "category-change"))
    state = session.get_state()
    state["components_state"]["logger"]["component_type"] = "Step"

    with pytest.raises(
            ValueError,
            match="logger.*stored as a Step.*registered as a Hook",
    ):
        TrainingSession.from_state(state)


def test_checkpoint_rejects_unregistered_components(tmp_path):
    session = TrainingSession(make_config(tmp_path / "missing-component"))
    state = session.get_state()
    state["components_state"]["missing_component"] = (
        state["components_state"].pop("logger")
    )

    with pytest.raises(
            ValueError,
            match="missing_component.*not registered",
    ):
        TrainingSession.from_state(state)


def test_failed_checkpoint_restore_preserves_existing_components(tmp_path):
    session = TrainingSession(make_config(tmp_path / "atomic-restore"))
    existing_components = {
        component.name: component
        for component in (
            session.get_all_resources()
            + session.get_all_hooks()
            + session.get_all_steps()
        )
    }
    state = session.get_state()
    state["components_state"]["logger"]["component_type"] = "Step"

    with pytest.raises(
            ValueError,
            match="logger.*stored as a Step.*registered as a Hook",
    ):
        session.set_state(state)

    restored_components = {
        component.name: component
        for component in (
            session.get_all_resources()
            + session.get_all_hooks()
            + session.get_all_steps()
        )
    }
    assert restored_components == existing_components
    assert all(
        restored_components[name] is component
        for name, component in existing_components.items()
    )


def test_checkpoint_pickle_round_trip_restores_resources_hooks_and_state(tmp_path):
    @step("checkpoint_rng_step")
    class CheckpointRngStep(Step, Stateful):
        def __init__(self, label, scale=1):
            self.label = label
            self.scale = scale
            self.samples = []

        def run(self, session):
            sample = (
                random.randint(0, 10 ** 6),
                int(np.random.randint(0, 10 ** 6)),
                int(torch.randint(0, 10 ** 6, (1,)).item()),
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
    restored_hook = next(
        component
        for component in restored.get_all_hooks()
        if isinstance(component, CheckpointHook)
    )
    restored_step = next(
        component
        for component in restored.get_all_steps()
        if isinstance(component, CheckpointRngStep)
    )

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
    @resource("inherited_checkpoint_resource")
    class InheritedCheckpointResource(BaseInheritedResource):
        pass


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