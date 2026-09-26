import pickle
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_utils import (
    AdditionalHookBase,
    AdditionalResourceBase,
    AdditionalStepBase,
    make_config,
    resource_named,
)
from training_framework.components import (
    hook,
    resource,
    step, Stateful, SessionHook, Step, StatefulResource, )
from training_framework.session import TrainingSession


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

        def pre_session(self, session):
            self.setup_calls += 1
            self.seen_session_dirs.append(session.session_config.session_dir)

        def post_session(self, session):
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

    restored_resource = resource_named(restored, resource_id)
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
    restored_resource = resource_named(restored, resource_id)

    assert restored_resource.label == "delta"
    assert restored_resource.factor == 13
    assert restored_resource.setup_calls == 1
    assert restored_resource.teardown_calls == 1


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
    restored_resources = [
        component
        for component in restored.get_all_resources()
        if isinstance(component, AdditionalResource)
    ]
    restored_steps = [
        component
        for component in restored.get_all_steps()
        if isinstance(component, AdditionalStep)
    ]
    restored_hooks = [
        component
        for component in restored.get_all_hooks()
        if isinstance(component, AdditionalHook)
    ]

    assert len(restored_resources) == 1
    assert len(restored_steps) == 1
    assert len(restored_hooks) == 1

    restored_resource = restored_resources[0]
    restored_step = restored_steps[0]
    restored_hook = restored_hooks[0]

    assert isinstance(restored_resource, AdditionalResource)
    assert isinstance(restored_step, AdditionalStep)
    assert isinstance(restored_hook, AdditionalHook)
    assert restored_resource.events == additional_resource.events
    assert restored_step.calls == additional_step.calls
    assert restored_hook.events == additional_hook.events
    assert restored_hook.call_every == additional_hook.call_every


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


def test_session_state_uses_clean_session_type_and_config_keys(tmp_path):
    session = TrainingSession(
        make_config(tmp_path / "checkpoint-schema", max_iterations=1)
    )

    state = session.get_state()

    assert state["session_type"] == "training"
    assert state["config"]["session_config"] == (
        session.full_config["session_config"]
    )
    assert state["session_config"] == asdict(session.session_config)
    assert "base_config" not in state
    assert "mode" not in state

    invalid_state = dict(state)
    del invalid_state["config"]
    with pytest.raises(ValueError, match="configuration state schema"):
        TrainingSession.from_state(invalid_state)


def test_from_state_does_not_repeat_normal_session_initialization(tmp_path):
    session = TrainingSession(
        make_config(tmp_path / "side-effect-free-restore", max_iterations=1)
    )
    state = session.get_state()
    written_config = Path(session.session_config.session_dir) / "config.yaml"

    restored = TrainingSession.from_state(state)

    # The config file is written when a session is entered, never on restore.
    assert not written_config.exists()
    assert restored.session_config == session.session_config
    assert restored.full_config == session.full_config
