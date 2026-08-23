from __future__ import annotations

import pytest

from training_framework.configurator import Configurator
from training_framework.training_engine import load_session_for_worker
from training_framework.training_session import (
    Hook,
    LifecycleHook,
    Resource,
    Step,
    TrainingSession,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
    wraps,
)


def _base_config(tmp_path):
    return {
        "rng_seed": 23,
        "sessions_dir": str(tmp_path),
        "max_iterations": 1,
        "device": "cpu",
        "components_package": "training_framework.builtin_components",
        "show-execution-graph": False,
    }


def test_components_list_activates_empty_config_and_mapping_wins(tmp_path):
    @step("optional_config_step")
    class OptionalConfigStep(Step):
        def __init__(self, config):
            self.config = dict(config)

        def run(self, session):
            session.iteration_context["configured_value"] = self.config["value"]

    @step("unselected_step")
    class UnselectedStep(Step):
        def __init__(self, config):
            raise AssertionError("an unselected component must not be constructed")

        def run(self, session):
            pass

    session = TrainingSession({
        "base_config": _base_config(tmp_path),
        "components": ["optional_config_step"],
        "optional_config_step": {"value": 7},
    })

    assert {component.name for component in session.get_all_steps()} == {
        "optional_config_step"
    }
    with session:
        assert next(session) == 1

    component = session.get_all_steps()[0]
    assert component.config == {"value": 7}


def test_dependencies_and_wrapped_hooks_are_activated_recursively(tmp_path):
    @resource("closure_leaf")
    class ClosureLeaf(Resource):
        def __init__(self, config):
            self.config = dict(config)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @resource("closure_parent")
    @requires_resource("closure_leaf")
    class ClosureParent(Resource):
        def __init__(self, config):
            self.config = dict(config)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @hook("closure_inner")
    class ClosureInner(LifecycleHook):
        def __init__(self, config):
            self.config = dict(config)
            self.call_every = 1

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @hook("closure_outer")
    @wraps("closure_inner")
    @requires_resource("closure_parent")
    class ClosureOuter(LifecycleHook):
        def __init__(self, config):
            self.config = dict(config)
            self.call_every = 1

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @step("closure_prerequisite")
    @requires_resource("closure_parent")
    class ClosurePrerequisite(Step):
        def __init__(self, config):
            self.config = dict(config)

        def run(self, session):
            pass

    @step("closure_root")
    @requires_hook("closure_outer")
    @requires_step("closure_prerequisite")
    class ClosureRoot(Step):
        def __init__(self, config):
            self.config = dict(config)

        def run(self, session):
            pass

    session = TrainingSession({
        "base_config": _base_config(tmp_path),
        "components": ["closure_root"],
    })

    assert {resource.name for resource in session.get_all_resources()} == {
        "closure_leaf",
        "closure_parent",
    }
    assert {"closure_inner", "closure_outer"} <= {
        component.name for component in session.get_all_hooks()
    }
    assert {component.name for component in session.get_all_steps()} == {
        "closure_prerequisite",
        "closure_root",
    }
    graph = session.execution_graph()
    assert graph.index("Hook.closure_outer.pre_iteration_callback()") < graph.index(
        "Hook.closure_inner.pre_iteration_callback()"
    )
    assert graph.index("Hook.closure_inner.post_iteration_callback()") < graph.index(
        "Hook.closure_outer.post_iteration_callback()"
    )


def test_alias_can_be_activated_only_through_a_dependency(tmp_path):
    @resource("alias_actual_resource")
    class AliasActualResource(Resource):
        def __init__(self, config):
            self.config = dict(config)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @resource("unused_alias_actual")
    class UnusedAliasActual(Resource):
        def __init__(self, config):
            raise AssertionError("an unused alias target must not be constructed")

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("alias_consumer")
    @requires_resource("alias_resource_role")
    class AliasConsumer(Step):
        def __init__(self, config):
            self.config = dict(config)

        def run(self, session):
            pass

    session = TrainingSession({
        "base_config": _base_config(tmp_path),
        "aliases": {
            "alias_resource_role": "alias_actual_resource",
            "unused_resource_role": "unused_alias_actual",
        },
        "components": ["alias_consumer"],
    })

    assert session.get_resource("alias_resource_role").config == {}
    assert not session.has_resource("unused_resource_role")
    restored = TrainingSession.from_state(session.get_state())
    assert restored.get_resource("alias_resource_role").config == {}


def test_aliased_default_is_activated_with_empty_config(tmp_path):
    @hook("empty_config_logger")
    class EmptyConfigLogger(Hook):
        def __init__(self, config):
            self.config = dict(config)

    session = TrainingSession({
        "base_config": _base_config(tmp_path),
        "aliases": {"logger": "empty_config_logger"},
    })

    hooks = {component.name: component for component in session.get_all_hooks()}
    assert "logger" not in hooks
    assert hooks["empty_config_logger"].config == {}
    assert "checkpointer" in hooks


@pytest.mark.parametrize(
    ("components", "error_type", "match"),
    (
        pytest.param("some_step", TypeError, "must be a list", id="not-a-list"),
        pytest.param([1], TypeError, "entries must be strings", id="non-string"),
        pytest.param([""], ValueError, "must not be empty", id="empty-name"),
        pytest.param(
            ["base_config"],
            ValueError,
            "reserved configuration name",
            id="reserved-name",
        ),
        pytest.param(
            ["duplicate_step", "duplicate_step"],
            ValueError,
            "more than once",
            id="duplicate",
        ),
        pytest.param(
            ["unknown_optional_component"],
            ValueError,
            "No step, hook or resource registered",
            id="unknown",
        ),
    ),
)
def test_invalid_components_lists_are_rejected(
        tmp_path,
        components,
        error_type,
        match,
):
    @step("duplicate_step")
    class DuplicateStep(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    with pytest.raises(error_type, match=match):
        TrainingSession({
            "base_config": _base_config(tmp_path),
            "components": components,
        })


def test_alias_target_cannot_be_selected_directly(tmp_path):
    @step("direct_alias_target")
    class DirectAliasTarget(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    with pytest.raises(ValueError, match="not both"):
        TrainingSession({
            "base_config": _base_config(tmp_path),
            "aliases": {"virtual_step": "direct_alias_target"},
            "components": ["direct_alias_target"],
        })


def test_auto_configured_constructor_failure_requests_mapping(tmp_path):
    @step("configuration_required_step")
    class ConfigurationRequiredStep(Step):
        def __init__(self, config):
            self.value = config["required_value"]

        def run(self, session):
            pass

    with pytest.raises(
        RuntimeError,
        match="auto-configured component 'configuration_required_step'.*top-level",
    ):
        TrainingSession({
            "base_config": _base_config(tmp_path),
            "components": ["configuration_required_step"],
        })


def test_configurator_unifies_list_and_mapping_components():
    configurator = Configurator.__new__(Configurator)
    configurator._session_configs = [{
        "base_config": {"max_iterations": 1},
        "components": ["empty_component", "configured_component"],
        "configured_component": {"value": 11},
    }]

    assert configurator.get_component_config(0, "empty_component") == {}
    assert configurator.get_component_config(0, "configured_component") == {
        "value": 11
    }
    assert configurator.get_all_component_configs(0) == {
        "empty_component": {},
        "configured_component": {"value": 11},
    }
    with pytest.raises(KeyError):
        configurator.get_component_config(0, "inactive_component")


def test_secondary_rank_keeps_parallel_component_dependency_closure(tmp_path):
    @resource("closure_ddp")
    class ClosureDDP(Resource):
        def __init__(self, config, rank=-1):
            self.config = dict(config)
            self.rank = rank
            self.world_size = int(self.config["world_size"])
            self.parallel_components = list(
                self.config.get("parallel_components", ())
            )

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @resource("parallel_dependency")
    class ParallelDependency(Resource):
        def __init__(self, config):
            self.config = dict(config)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("parallel_root")
    @requires_resource("parallel_dependency")
    class ParallelRoot(Step):
        def __init__(self, config):
            self.config = dict(config)

        def run(self, session):
            pass

    @resource("rank_zero_dependency")
    class RankZeroDependency(Resource):
        def __init__(self, config):
            self.config = dict(config)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("rank_zero_only")
    @requires_resource("rank_zero_dependency")
    class RankZeroOnly(Step):
        def __init__(self, config):
            self.config = dict(config)

        def run(self, session):
            pass

    config = {
        "base_config": _base_config(tmp_path),
        "aliases": {"ddp": "closure_ddp"},
        "components": ["parallel_root", "rank_zero_only"],
        "ddp": {
            "world_size": 2,
            "parallel_components": ["parallel_root"],
        },
    }

    rank_one = load_session_for_worker(
        TrainingSession(config).get_state(),
        rank=1,
    )

    assert rank_one.get_resource("ddp").rank == 1
    assert rank_one.has_resource("parallel_dependency")
    assert not rank_one.has_resource("rank_zero_dependency")
    assert {component.name for component in rank_one.get_all_steps()} == {
        "parallel_root"
    }
