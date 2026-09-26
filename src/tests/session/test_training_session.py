from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from training_framework.components import (
    LifecycleHook,
    Resource,
    Step,
    hook,
    resource,
    step,
)
from training_framework.session import TrainingSession
from tests.test_utils import (
    AdditionalHookBase,
    AdditionalResourceBase,
    AdditionalStepBase,
    has_resource_named,
    read_events,
    register_test_components,
    resource_named,
    session_config,
)


def _component(session: TrainingSession, name: str):
    components = session.get_all_resources() + session.get_all_steps() + session.get_all_hooks()
    return next(component for component in components if component.name == name)


def test_config_driven_session_runs_training_and_lifecycle_end_to_end(tmp_path):
    """Exercise construction, hooks, autograd, optimizer state, and teardown together."""

    register_test_components()
    event_path = tmp_path / "lifecycle.jsonl"
    session = TrainingSession(
        session_config(
            tmp_path,
            max_iterations=3,
            event_path=event_path,
            include_metrics=True,
        )
    )

    with session:
        completed_iterations = list(session)

    model = _component(session, "it_3d45_model")
    training_step = _component(session, "it_3d45_train")
    metrics = _component(session, "it_3d45_metrics")

    assert completed_iterations == [1, 2, 3]
    assert session.iteration == 3
    assert session.session_context == {}

    assert model.setup_count == 1
    assert model.teardown_count == 1
    assert len(training_step.weight_history) == 3
    assert len(training_step.noise_history) == 3
    assert metrics.setup_count == 1
    assert metrics.teardown_count == 1
    assert [item["iteration"] for item in metrics.observations] == [1, 2, 3]
    assert [item["weight"] for item in metrics.observations] == pytest.approx(
        training_step.weight_history
    )

    events = read_events(event_path)
    assert [event["event"] for event in events] == [
        "model_setup",
        "metrics_setup",
        "iteration",
        "metrics",
        "iteration",
        "metrics",
        "iteration",
        "metrics",
        "metrics_teardown",
        "model_teardown",
    ]



def test_default_components_are_available_through_public_api(tmp_path):
    register_test_components()
    session = TrainingSession(session_config(tmp_path, max_iterations=4))

    hook_names = {
        component.name for component in session.get_all_hooks()
    }
    assert {"logger", "checkpointer"} <= hook_names
    assert not has_resource_named(session, "tensorboard")


def test_explicit_builtin_configs_override_defaults_and_enable_tensorboard(tmp_path):
    register_test_components()
    config = session_config(tmp_path, max_iterations=4)
    config.update({
        "logger": {"log_every": 3},
        "checkpointer": {"checkpoint_every": 2},
        "tensorboard": {"host": "127.0.0.1", "port": 6006},
    })

    session = TrainingSession(config)

    hooks = {component.name: component for component in session.get_all_hooks()}
    assert hooks["logger"].call_every == 3
    assert hooks["checkpointer"].call_every == 2
    assert has_resource_named(session, "tensorboard")


@pytest.mark.parametrize(
    ("rank", "should_print_graph"),
    [(0, True), (1, False)],
)
def test_execution_graph_is_printed_only_for_ddp_rank_zero(
        tmp_path,
        capsys,
        rank,
        should_print_graph,
):
    @resource("ddp", overwrite=True, session_type="training")
    class DDPTestResource(Resource):
        def __init__(self, config):
            self.rank = config["rank"]

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    config = {
        "session_config": {
            "rng_seed": 1,
            "sessions_dir": str(tmp_path / f"rank-{rank}"),
            "max_iterations": 1,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
            "show_execution_graph": True,
        },
        "ddp": {"rank": rank},
    }

    with TrainingSession(config):
        pass

    graph_was_printed = (
        "TRAINING SESSION EXECUTION GRAPH" in capsys.readouterr().out
    )
    assert graph_was_printed is should_print_graph


def test_training_session_initialization_and_device_validation(minimal_session_config_2, monkeypatch):
    session = TrainingSession(minimal_session_config_2)

    assert session.session_config.rng_seed == minimal_session_config_2["session_config"]["rng_seed"]
    assert session.session_config.max_iterations == minimal_session_config_2["session_config"]["max_iterations"]
    assert session.session_config.session_dir.startswith(minimal_session_config_2["session_config"]["sessions_dir"])
    assert session.device.type == "cpu"
    assert session.iteration == 0

    bad_device = deepcopy(minimal_session_config_2)
    bad_device["session_config"]["device"] = "tpu"

    assert TrainingSession(bad_device).device == torch.device("cpu")

    unavailable_cuda = deepcopy(minimal_session_config_2)
    unavailable_cuda["session_config"]["device"] = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert TrainingSession(unavailable_cuda).device == torch.device("cpu")


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

        def pre_session(self, session: TrainingSession):
            pass

        def post_session(self, session):
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

    with pytest.raises(ValueError, match="not registered as a component"):
        session.add_step(UnregisteredStep())

    with pytest.raises(ValueError, match="not registered as a component"):
        session.register_hook(UnregisteredHook())

    with pytest.raises(ValueError, match="not registered as a component"):
        session.register_resource(UnregisteredResource())

    step_obj = AdditionalStep()
    hook_obj = AdditionalHook(call_every=1)
    resource_obj = AdditionalResource()

    step_id = session.add_step(step_obj)
    hook_id = session.register_hook(hook_obj)
    resource_id = session.register_resource(resource_obj)

    assert step_id == "test_additional_step"
    assert hook_id == "test_additional_hook"
    assert resource_id == "test_additional_resource"
    assert step_obj in session.get_all_steps()
    assert hook_obj in session.get_all_hooks()
    assert resource_named(session, resource_id) is resource_obj
    with pytest.raises(KeyError):
        resource_named(session, "missing-resource")
