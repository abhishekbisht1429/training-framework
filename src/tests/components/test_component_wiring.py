"""Tests for construction-time component wiring.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pytest

from tests.test_utils import build_session, make_config, resource_named
from training_framework.components import (
    ComponentDependencyError,
    Resource,
    requires_resource,
    resource,
)
from training_framework.session import TrainingSession


class _InertResource(Resource):
    """A resource with no lifecycle of its own."""

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def test_components_are_constructed_prerequisite_first(tmp_path):
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

    session = build_session(tmp_path, {"wire_parent": {}, "wire_child": {}})

    assert constructed == ["wire_child", "wire_parent"]
    parent = resource_named(session, "wire_parent")
    assert parent.child is resource_named(session, "wire_child")


def test_dependencies_resolve_through_bindings(tmp_path):
    @resource("wire_real_encoder")
    class Encoder(_InertResource):
        pass

    @requires_resource("wire_encoder")
    @resource("wire_model")
    class Model(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.encoder = self.get_dependency("wire_encoder")

    session = build_session(
        tmp_path,
        {"wire_model": {}, "wire_real_encoder": {}},
        bindings={"wire_encoder": "wire_real_encoder"},
    )

    model = resource_named(session, "wire_model")
    assert model.encoder is resource_named(session, "wire_real_encoder")


def test_requesting_an_undeclared_resource_is_rejected(tmp_path):
    @resource("wire_secret")
    class Secret(_InertResource):
        pass

    @resource("wire_snooper")
    class Snooper(_InertResource):
        def __init__(self, config=None):
            super().__init__(config)
            self.get_dependency("wire_secret")

    with pytest.raises(ComponentDependencyError) as raised:
        build_session(tmp_path, {"wire_snooper": {}, "wire_secret": {}})

    message = str(raised.value)
    assert "wire_snooper" in message
    assert "@requires_resource('wire_secret')" in message


def test_a_dependency_cycle_is_reported_before_anything_is_constructed(
        tmp_path,
):
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
        build_session(tmp_path, {"wire_cycle_a": {}, "wire_cycle_b": {}})

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

    parent = resource_named(session, "wire_late_parent")
    assert parent.child is resource_named(session, "wire_late_child")


def test_activating_a_component_after_setup_is_rejected(tmp_path):
    @resource("wire_frozen")
    class Frozen(_InertResource):
        pass

    config = make_config(tmp_path / "frozen-wire")
    session = TrainingSession(config)

    with session:
        with pytest.raises(RuntimeError, match="NEW phase"):
            session.activate_component("wire_frozen", {})
