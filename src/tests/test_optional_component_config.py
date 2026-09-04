from __future__ import annotations

import pytest

from training_framework.engine import Configurator
from training_framework.engine import load_session_for_worker
from training_framework.components import (
    Hook,
    LifecycleHook,
    Resource,
    StatefulStep,
    Step,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
    wraps,
)
from training_framework.session import TrainingSession


def _session_config(tmp_path):
    return {
        "rng_seed": 23,
        "sessions_dir": str(tmp_path),
        "max_iterations": 1,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def test_empty_mapping_activates_component_and_supplies_config(tmp_path):
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
        "session_config": _session_config(tmp_path),
        "optional_config_step": {"value": 7},
    })

    assert {component.name for component in session.get_all_steps()} == {
        "optional_config_step"
    }
    with session:
        assert next(session) == 1

    component = session.get_all_steps()[0]
    assert component.config == {"value": 7}


def test_component_can_omit_constructor_and_restore_state(tmp_path):
    @step("constructor_free_step")
    class ConstructorFreeStep(StatefulStep):
        executions = 0

        def run(self, session):
            self.executions += 1

        def get_state(self):
            return {"executions": self.executions}

        def set_state(self, state):
            self.executions = state["executions"]

    session = TrainingSession({
        "session_config": _session_config(tmp_path),
        "constructor_free_step": {},
    })

    with session:
        assert next(session) == 1

    restored = TrainingSession.from_state(session.get_state())

    assert restored.get_all_steps()[0].executions == 1


def test_dependencies_and_wrapped_hooks_are_activated_recursively(tmp_path):
    @resource("closure_leaf")
    class ClosureLeaf(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @resource("closure_parent")
    @requires_resource("closure_leaf")
    class ClosureParent(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @hook("closure_inner")
    class ClosureInner(LifecycleHook):
        call_every = 1

        def pre_session(self, session):
            pass

        def post_session(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @hook("closure_outer")
    @wraps("closure_inner")
    @requires_resource("closure_parent")
    class ClosureOuter(LifecycleHook):
        call_every = 1

        def pre_session(self, session):
            pass

        def post_session(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @step("closure_prerequisite")
    @requires_resource("closure_parent")
    class ClosurePrerequisite(Step):
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
        "session_config": _session_config(tmp_path),
        "closure_root": {},
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
        "session_config": _session_config(tmp_path),
        "component_bindings": {
            "alias_resource_role": "alias_actual_resource",
            "unused_resource_role": "unused_alias_actual",
        },
        "alias_consumer": {},
    })

    assert session.has_resource("alias_resource_role")
    assert not session.has_resource("unused_resource_role")
    state = session.get_state()
    restored = TrainingSession.from_state(state)
    assert restored.has_resource("alias_resource_role")


def test_aliased_default_is_activated_with_empty_config(tmp_path):
    @hook("empty_config_logger")
    class EmptyConfigLogger(Hook):
        def __init__(self, config):
            self.config = dict(config)

    session = TrainingSession({
        "session_config": _session_config(tmp_path),
        "component_bindings": {"logger": "empty_config_logger"},
    })

    hooks = {component.name: component for component in session.get_all_hooks()}
    assert "logger" not in hooks
    assert hooks["empty_config_logger"].config == {}
    assert "checkpointer" in hooks


def test_legacy_components_entry_is_rejected_with_migration_guidance(tmp_path):
    with pytest.raises(
            ValueError,
            match="no longer supported.*top-level mappings",
    ):
        TrainingSession({
            "session_config": _session_config(tmp_path),
            "components": ["legacy_step"],
        })


def test_bound_role_cannot_be_configured_as_a_component(tmp_path):
    @step("direct_alias_target")
    class DirectAliasTarget(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    with pytest.raises(ValueError, match="Configure the implementation name"):
        TrainingSession({
            "session_config": _session_config(tmp_path),
            "component_bindings": {
                "virtual_step": "direct_alias_target",
            },
            "virtual_step": {},
        })


def test_required_custom_constructor_requires_mapping(tmp_path):
    @resource("configuration_required_resource")
    class ConfigurationRequiredResource(Resource):
        def __init__(self, config):
            self.value = config["required_value"]

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("configuration_consumer")
    @requires_resource("configuration_required_resource")
    class ConfigurationConsumer(Step):
        def run(self, session):
            pass

    with pytest.raises(
        RuntimeError,
        match="'configuration_required_resource' is required.*custom constructor.*top-level",
    ):
        TrainingSession({
            "session_config": _session_config(tmp_path),
            "configuration_consumer": {},
        })


def test_inherited_custom_constructor_requires_mapping(tmp_path):
    class ConfiguredResourceBase(Resource):
        def __init__(self, config):
            self.value = config["required_value"]

    @resource("inherited_configuration_resource")
    class InheritedConfigurationResource(ConfiguredResourceBase):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("inherited_configuration_consumer")
    @requires_resource("inherited_configuration_resource")
    class InheritedConfigurationConsumer(Step):
        def run(self, session):
            pass

    with pytest.raises(
            RuntimeError,
            match="'inherited_configuration_resource'.*custom",
    ):
        TrainingSession({
            "session_config": _session_config(tmp_path),
            "inherited_configuration_consumer": {},
        })


def test_required_custom_constructor_uses_supplied_mapping(tmp_path):
    @resource("configured_dependency")
    class ConfiguredDependency(Resource):
        def __init__(self, config):
            self.value = config["value"]

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @step("configured_consumer")
    @requires_resource("configured_dependency")
    class ConfiguredConsumer(Step):
        def run(self, session):
            pass

    session = TrainingSession({
        "session_config": _session_config(tmp_path),
        "configured_consumer": {},
        "configured_dependency": {"value": 11},
    })

    assert session.get_resource("configured_dependency").value == 11


def test_configurator_returns_mapping_components_only():
    configurator = Configurator.__new__(Configurator)
    configurator._session_configs = [{
        "session_config": {"max_iterations": 1},
        "empty_component": {},
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


def test_configurator_rejects_legacy_components_entry():
    configurator = Configurator.__new__(Configurator)
    configurator._session_configs = [{
        "session_config": {"max_iterations": 1},
        "components": ["legacy_component"],
    }]

    with pytest.raises(ValueError, match="no longer supported"):
        configurator.get_session_definition(0)


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
        "session_config": _session_config(tmp_path),
        "component_bindings": {"ddp": "closure_ddp"},
        "parallel_root": {},
        "rank_zero_only": {},
        "closure_ddp": {
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
