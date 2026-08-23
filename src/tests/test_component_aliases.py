from __future__ import annotations

import pickle

import pytest

from tests.test_utils import COMPONENTS_PACKAGE, register_test_components
from training_framework.configurator import Configurator
from training_framework.training_engine import load_session_for_worker
from training_framework.training_session import (
    Hook,
    Resource,
    SessionHook,
    Step,
    TrainingSession,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
)


def _base_config(tmp_path, *, components_package="training_framework.builtin_components"):
    return {
        "rng_seed": 17,
        "sessions_dir": str(tmp_path),
        "max_iterations": 2,
        "device": "cpu",
        "components_package": components_package,
        "show-execution-graph": False,
    }


def test_aliases_substitute_components_dependencies_and_public_names(tmp_path):
    @resource("custom_model")
    class CustomModel(Resource):
        def __init__(self, config):
            self.label = config["label"]

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @hook("custom_metrics")
    @requires_resource("model")
    class CustomMetrics(SessionHook):
        def __init__(self, config):
            self.prefix = config["prefix"]

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("custom_optimizer")
    @requires_hook("metrics")
    class CustomOptimizer(Step):
        def __init__(self, config):
            self.learning_rate = config["learning_rate"]

        def run(self, session):
            pass

    @step("consumer")
    @requires_step("optimizer_step")
    class Consumer(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    config = {
        "base_config": _base_config(tmp_path),
        "aliases": {
            "model": "custom_model",
            "metrics": "custom_metrics",
            "optimizer_step": "custom_optimizer",
        },
        "model": {"label": "primary"},
        "metrics": {"prefix": "train"},
        "optimizer_step": {"learning_rate": 0.001},
        "consumer": {},
    }

    session = TrainingSession(config)

    model_component = session.get_resource("model")
    assert model_component is session.get_resource("custom_model")
    assert model_component.name == "custom_model"
    assert model_component.label == "primary"
    assert session.has_resource("model")
    assert session.has_resource("custom_model")
    assert session.component_aliases == config["aliases"]
    assert session.resolve_component_name("optimizer_step") == "custom_optimizer"

    hook_names = {component.name for component in session.get_all_hooks()}
    step_names = {component.name for component in session.get_all_steps()}
    assert "custom_metrics" in hook_names
    assert "metrics" not in hook_names
    assert {"custom_optimizer", "consumer"} <= step_names
    assert "optimizer_step" not in step_names

    graph = session.execution_graph()
    assert "ALIASES" in graph
    assert "optimizer_step -> custom_optimizer" in graph
    assert "requires: Resource.custom_model" in graph
    assert "requires: Hook.custom_metrics" in graph
    assert "requires: Step.custom_optimizer" in graph
    assert graph.index("Resource.custom_model.setup()") < graph.index(
        "Hook.custom_metrics.setup()"
    )
    assert graph.index("Step.custom_optimizer.run()") < graph.index(
        "Step.consumer.run()"
    )

    session.remove_step("optimizer_step")
    session.unregister_hook("metrics")
    session.unregister_resource("model")
    assert "custom_optimizer" not in {
        component.name for component in session.get_all_steps()
    }
    assert not session.has_resource("model")


def test_alias_replaces_a_default_component_without_duplicate(tmp_path):
    @hook("custom_logger")
    class CustomLogger(Hook):
        def __init__(self, config):
            self.destination = config["destination"]

    session = TrainingSession({
        "base_config": _base_config(tmp_path),
        "aliases": {"logger": "custom_logger"},
        "logger": {"destination": "memory"},
    })

    hooks = {component.name: component for component in session.get_all_hooks()}
    assert "logger" not in hooks
    assert hooks["custom_logger"].destination == "memory"
    assert "checkpointer" in hooks


def test_aliases_survive_pickle_and_state_round_trips(tmp_path):
    @step("checkpoint_actual")
    class CheckpointStep(Step):
        def __init__(self, config):
            self.label = config["label"]

        def run(self, session):
            pass

    config = {
        "base_config": _base_config(tmp_path),
        "aliases": {"checkpoint_role": "checkpoint_actual"},
        "checkpoint_role": {"label": "restored"},
    }
    session = TrainingSession(config)

    from_state = TrainingSession.from_state(session.get_state())
    from_pickle = pickle.loads(pickle.dumps(session))

    for restored in (from_state, from_pickle):
        assert restored.component_aliases == config["aliases"]
        assert restored.resolve_component_name("checkpoint_role") == (
            "checkpoint_actual"
        )
        step_names = {
            component.name for component in restored.get_all_steps()
        }
        assert step_names == {"checkpoint_actual"}
        assert "checkpoint_role -> checkpoint_actual" in restored.execution_graph()


def test_ddp_parallel_components_accept_expected_alias_names(tmp_path):
    register_test_components()
    config = {
        "base_config": _base_config(
            tmp_path,
            components_package=COMPONENTS_PACKAGE,
        ),
        "aliases": {
            "model": "it_3d45_model",
            "train_role": "it_3d45_train",
        },
        "ddp": {
            "world_size": 2,
            "backend": "gloo",
            "parallel_components": ["model", "train_role"],
            "master_addr": "localhost",
            "master_port": "12355",
        },
        "model": {},
        "train_role": {},
        "it_3d45_rank0_resource": {},
        "it_3d45_rank0_step": {},
        "it_3d45_rank0_hook": {"call_every": 1},
    }

    rank_one = load_session_for_worker(
        TrainingSession(config).get_state(),
        rank=1,
    )

    assert rank_one.has_resource("model")
    assert rank_one.has_resource("it_3d45_model")
    assert "it_3d45_train" in {
        component.name for component in rank_one.get_all_steps()
    }
    assert "it_3d45_rank0_resource" not in {
        component.name for component in rank_one.get_all_resources()
    }
    assert "it_3d45_rank0_step" not in {
        component.name for component in rank_one.get_all_steps()
    }


@pytest.mark.parametrize(
    ("aliases", "component_configs", "error_type", "match"),
    (
        pytest.param([], {}, TypeError, "must be a mapping", id="not-a-mapping"),
        pytest.param(
            {"role": 1},
            {"role": {}},
            TypeError,
            "strings to strings",
            id="non-string",
        ),
        pytest.param(
            {"role": "missing"},
            {"role": {}},
            ValueError,
            "not a registered component",
            id="unknown-target",
        ),
        pytest.param(
            {"target": "target"},
            {"target": {}},
            ValueError,
            "different component",
            id="no-op",
        ),
        pytest.param(
            {"base_config": "target"},
            {},
            ValueError,
            "reserved",
            id="reserved-name",
        ),
        pytest.param(
            {"first": "target", "second": "target"},
            {"first": {}, "second": {}},
            ValueError,
            "cannot both target",
            id="duplicate-target",
        ),
        pytest.param(
            {"first": "second", "second": "target"},
            {"first": {}, "second": {}},
            ValueError,
            "chains and cycles",
            id="alias-chain",
        ),
        pytest.param(
            {"role": "target"},
            {"role": {}, "target": {}},
            ValueError,
            "not both",
            id="expected-and-actual-configured",
        ),
    ),
)
def test_invalid_alias_mappings_are_rejected(
        tmp_path,
        aliases,
        component_configs,
        error_type,
        match,
):
    @step("target")
    class Target(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    config = {
        "base_config": _base_config(tmp_path),
        "aliases": aliases,
        **component_configs,
    }

    with pytest.raises(error_type, match=match):
        TrainingSession(config)


def test_alias_target_names_are_globally_unique_across_categories():
    @step("ambiguous")
    class AmbiguousStep(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    with pytest.raises(ValueError, match="already registered"):
        @resource("ambiguous")
        class AmbiguousResource(Resource):
            def __init__(self, config):
                pass

            def setup(self, session):
                pass

            def teardown(self, session):
                pass


def test_alias_dependency_must_resolve_to_the_required_category(tmp_path):
    @hook("actual_hook")
    class ActualHook(Hook):
        def __init__(self, config):
            pass

    @step("consumer")
    @requires_step("virtual_step")
    class Consumer(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    with pytest.raises(RuntimeError, match="not registered as a Step"):
        TrainingSession({
            "base_config": _base_config(tmp_path),
            "aliases": {"virtual_step": "actual_hook"},
            "virtual_step": {},
            "consumer": {},
        })


def test_configurator_excludes_special_entries_from_component_configs():
    configurator = Configurator.__new__(Configurator)
    configurator._session_configs = [{
        "base_config": {"max_iterations": 1},
        "aliases": {"role": "target"},
        "components": ["no_config"],
        "role": {"value": 3},
    }]

    assert configurator.get_all_component_configs(0) == {
        "no_config": {},
        "role": {"value": 3},
    }
