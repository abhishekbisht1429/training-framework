"""Tests for construction-time component wiring.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pytest

from tests.test_utils import make_config
from training_framework.components import (
    ComponentDependencyError,
    Resource,
    requires_resource,
    resource,
)
from training_framework.session import TrainingSession
from training_framework.session.components import SessionComponents


def _activate(config, *, session_type="training"):
    components = SessionComponents(session_type=session_type)
    components.register_from_config(config)
    return components


class _InertResource(Resource):
    """A resource with no lifecycle of its own."""

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def test_components_are_constructed_prerequisite_first():
    constructed = []

    @resource("wire_child")
    class Child(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            constructed.append("wire_child")

    @requires_resource("wire_child")
    @resource("wire_parent")
    class Parent(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            constructed.append("wire_parent")
            self.child = self.get_dependency("wire_child")

    components = _activate({"wire_parent": {}, "wire_child": {}})

    assert constructed == ["wire_child", "wire_parent"]
    parent = components.get_resource("wire_parent")
    assert parent.child is components.get_resource("wire_child")


def test_dependencies_resolve_through_bindings():
    @resource("wire_real_encoder")
    class Encoder(_InertResource):
        pass

    @requires_resource("wire_encoder")
    @resource("wire_model")
    class Model(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.encoder = self.get_dependency("wire_encoder")

    components = SessionComponents(
        component_bindings={"wire_encoder": "wire_real_encoder"},
    )
    components.register_from_config({"wire_model": {}, "wire_real_encoder": {}})

    model = components.get_resource("wire_model")
    assert model.encoder is components.get_resource("wire_real_encoder")


def test_requesting_an_undeclared_resource_is_rejected():
    @resource("wire_secret")
    class Secret(_InertResource):
        pass

    @resource("wire_snooper")
    class Snooper(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.get_dependency("wire_secret")

    with pytest.raises(ComponentDependencyError) as raised:
        _activate({"wire_snooper": {}, "wire_secret": {}})

    message = str(raised.value)
    assert "wire_snooper" in message
    assert "@requires_resource('wire_secret')" in message


def test_requesting_a_dependency_outside_construction_is_rejected():
    @resource("wire_lonely_child")
    class Child(_InertResource):
        pass

    @requires_resource("wire_lonely_child")
    @resource("wire_lonely")
    class Lonely(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.child = self.get_dependency("wire_lonely_child")

    with pytest.raises(ComponentDependencyError, match="activate_component"):
        Lonely()


def test_a_dependency_cycle_is_reported_before_anything_is_constructed():
    constructed = []

    @requires_resource("wire_cycle_b")
    @resource("wire_cycle_a")
    class A(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            constructed.append("a")

    @requires_resource("wire_cycle_a")
    @resource("wire_cycle_b")
    class B(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            constructed.append("b")

    with pytest.raises(RuntimeError, match="Cyclic dependency"):
        _activate({"wire_cycle_a": {}, "wire_cycle_b": {}})

    assert constructed == []


def test_activate_component_wires_a_late_component(tmp_path):
    @resource("wire_late_child")
    class Child(_InertResource):
        pass

    @requires_resource("wire_late_child")
    @resource("wire_late_parent")
    class Parent(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.child = self.get_dependency("wire_late_child")

    config = make_config(tmp_path / "late-wire")
    session = TrainingSession(config)

    session.activate_component("wire_late_child", {})
    session.activate_component("wire_late_parent", {})

    parent = session.get_resource("wire_late_parent")
    assert parent.child is session.get_resource("wire_late_child")


def test_activating_a_component_after_setup_is_rejected(tmp_path):
    @resource("wire_frozen")
    class Frozen(_InertResource):
        pass

    config = make_config(tmp_path / "frozen-wire")
    session = TrainingSession(config)

    with session:
        with pytest.raises(RuntimeError, match="NEW phase"):
            session.activate_component("wire_frozen", {})


def test_dependency_closure_can_be_resolved_before_construction():
    @resource("wire_closure_child")
    class Child(_InertResource):
        pass

    @requires_resource("wire_closure_child")
    @resource("wire_closure_parent")
    class Parent(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.child = self.get_dependency("wire_closure_child")

    @resource("wire_closure_other")
    class Other(_InertResource):
        pass

    components = SessionComponents()

    closure = components.dependency_closure(
        ["wire_closure_parent"],
        active_names={
            "wire_closure_parent",
            "wire_closure_child",
            "wire_closure_other",
        },
    )

    assert closure == {"wire_closure_parent", "wire_closure_child"}
    assert components.components == {}
