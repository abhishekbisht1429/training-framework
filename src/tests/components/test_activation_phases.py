"""Activation: what must be active, the order it is built in, and companions.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

from dataclasses import dataclass

import pytest

from tests.test_utils import build_session, component_named, component_names
from training_framework.components import (
    ComponentDependencyError,
    ExtendableComponent,
    Resource,
    Step,
    activates,
    rank_zero_only,
    requires_resource,
    resource,
    singleton,
    step,
)


class _InertResource(Resource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class _InertStep(Step):
    def run(self, session):
        pass


# -- F1: components whose configuration is entirely defaulted -------------


def test_a_defaulted_schema_component_with_its_own_constructor_is_auto_activated(
        tmp_path,
):
    @dataclass
    class DefaultedConfig:
        value: int = 1

    @resource("act_defaulted")
    class Defaulted(_InertResource, ExtendableComponent):
        config_schema = DefaultedConfig

        def __init__(self, config=None):
            super().__init__(config)
            self.value = self._cfg.value

        def apply_extension_config(self, config, changed_paths):
            self.value = config["value"]

    @requires_resource("act_defaulted")
    @resource("act_consumer")
    class Consumer(_InertResource):
        pass

    session = build_session(tmp_path, {"act_consumer": {}})

    defaulted = component_named(session, "act_defaulted")
    assert defaulted.value == 1
    # Built from a mapping like any configured component, so an extension
    # can change it although it was never written down.
    session.apply_extension_overrides(["act_defaulted.value=3"])
    assert defaulted.value == 3


def test_a_required_field_still_needs_a_top_level_mapping(tmp_path):
    @dataclass
    class RequiredConfig:
        value: int

    @resource("act_required")
    class Required(_InertResource):
        config_schema = RequiredConfig

        def __init__(self, config=None):
            super().__init__(config)

    @requires_resource("act_required")
    @resource("act_needs_required")
    class Consumer(_InertResource):
        pass

    with pytest.raises(RuntimeError, match="Add a top-level component mapping"):
        build_session(tmp_path, {"act_needs_required": {}})


# -- F2: companions --------------------------------------------------------


def _declare_driver_and_driven(*, driven_decorators=()):
    """A resource that activates the step it is driven by."""

    @activates("act_driven")
    @resource("act_driver")
    class Driver(_InertResource):
        pass

    driven = requires_resource("act_driver")(type("Driven", (_InertStep,), {}))
    for decorate in driven_decorators:
        driven = decorate(driven)
    step("act_driven")(driven)
    return Driver, driven


def test_a_companion_that_requires_its_activator_is_not_a_cycle(tmp_path):
    _declare_driver_and_driven()

    session = build_session(tmp_path, {"act_driver": {}})

    assert {"act_driver", "act_driven"} <= component_names(session)
    driven = component_named(session, "act_driven")
    assert driven.get_dependency("act_driver") is component_named(
        session, "act_driver",
    )


def test_components_that_activate_each_other_are_both_activated(tmp_path):
    @activates("act_mutual_b")
    @resource("act_mutual_a")
    class A(_InertResource):
        pass

    @activates("act_mutual_a")
    @resource("act_mutual_b")
    class B(_InertResource):
        pass

    session = build_session(tmp_path, {"act_mutual_a": {}})

    assert {"act_mutual_a", "act_mutual_b"} <= component_names(session)


def test_a_consumer_binding_redirects_a_companion(tmp_path):
    @activates("act_role")
    @resource("act_activator")
    class Activator(_InertResource):
        pass

    @resource("act_implementation")
    class Implementation(_InertResource):
        pass

    session = build_session(
        tmp_path,
        {"act_activator": {}},
        bindings={"act_activator": {"act_role": "act_implementation"}},
    )

    assert "act_implementation" in component_names(session)


def test_two_activators_sharing_a_companion_that_needs_one_is_ambiguous(
        tmp_path,
):
    _declare_driver_and_driven()

    with pytest.raises(ComponentDependencyError, match="2 active components"):
        build_session(tmp_path, {"act_driver#a": {}, "act_driver#b": {}})


def test_a_companion_resolves_to_the_configured_instance(tmp_path):
    _declare_driver_and_driven(driven_decorators=(singleton,))

    session = build_session(
        tmp_path, {"act_driver": {}, "act_driven#only": {}},
    )

    # The sole-instance rule applies to companions as to any dependency, so
    # no second, unconfigured instance is created beside the configured one.
    driven = {
        name for name in component_names(session)
        if name.startswith("act_driven")
    }
    assert driven == {"act_driven#only"}


def test_nothing_is_constructed_before_every_problem_is_known(tmp_path):
    constructed = []

    @resource("act_recorder")
    class Recorder(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            constructed.append("act_recorder")

    @activates("act_unregistered")
    @resource("act_broken")
    class Broken(_InertResource):
        pass

    with pytest.raises(ValueError, match="act_unregistered"):
        build_session(tmp_path, {"act_recorder": {}, "act_broken": {}})
    assert constructed == []


def test_a_prerequisite_cycle_is_reported_with_its_chain_before_construction(
        tmp_path,
):
    constructed = []

    @resource("act_first")
    class First(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            constructed.append("act_first")

    @requires_resource("act_cycle_b")
    @resource("act_cycle_a")
    class CycleA(_InertResource):
        pass

    @requires_resource("act_cycle_a")
    @resource("act_cycle_b")
    class CycleB(_InertResource):
        pass

    with pytest.raises(
            RuntimeError,
            match="act_cycle_a -> act_cycle_b -> act_cycle_a",
    ):
        build_session(tmp_path, {"act_first": {}, "act_cycle_a": {}})
    assert constructed == []


def test_construction_order_is_prerequisite_first_in_configured_order(tmp_path):
    constructed = []

    @dataclass
    class NoSettings:
        pass

    def recording(name, *requires):
        cls = type(name, (_InertResource,), {
            # Defaulted configuration, so an unconfigured one is auto-built
            # although it records itself in its own constructor.
            "config_schema": NoSettings,
            "__init__": lambda self, config=None: (
                _InertResource.__init__(self, config),
                constructed.append(name),
            )[0],
        })
        for required in requires:
            cls = requires_resource(required)(cls)
        return resource(name)(cls)

    recording("ord_leaf")
    recording("ord_left", "ord_leaf")
    recording("ord_right", "ord_leaf")
    recording("ord_top", "ord_left", "ord_right")
    recording("ord_other")

    build_session(tmp_path, {"ord_other": {}, "ord_top": {}})

    assert constructed == [
        "ord_other", "ord_leaf", "ord_left", "ord_right", "ord_top",
    ]


def test_a_rank_keeps_what_its_components_activate(tmp_path):
    @activates("act_reporter")
    @resource("act_owner")
    class Owner(_InertResource):
        pass

    @rank_zero_only
    @resource("act_reporter")
    class Reporter(_InertResource):
        pass

    session = build_session(tmp_path, {"act_owner": {}})

    with pytest.warns(RuntimeWarning, match="act_reporter"):
        keep = session.rank_parallel_names()
    assert "act_reporter" in keep


def test_a_session_missing_a_companion_is_rejected_when_ordered(tmp_path):
    Driver, _ = _declare_driver_and_driven()

    @resource("act_bystander")
    class Bystander(_InertResource):
        pass

    session = build_session(tmp_path, {"act_bystander": {}})
    session.register_resource(Driver({}))

    with pytest.raises(RuntimeError, match="activates 'act_driven'"):
        session.execution_graph()


def test_the_execution_graph_shows_companions(tmp_path):
    _declare_driver_and_driven()

    session = build_session(tmp_path, {"act_driver": {}})

    assert "activates: Step.act_driven" in session.execution_graph()
