"""Tests for the component link phase.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pytest

from tests.test_utils import make_config
from training_framework.components import (
    ComponentLinkError,
    Resource,
    requires_resource,
    resource,
)
from training_framework.session import TrainingSession
from training_framework.session.components import SessionComponents


def _activate(config, *, session_type="training", link=True):
    components = SessionComponents(session_type=session_type)
    components.register_from_config(config)
    if link:
        components.link_components()
    return components


class _InertResource(Resource):
    """A resource that takes part in no linking."""

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def test_link_runs_prerequisite_first_and_can_reach_declared_dependencies():
    linked = []

    @resource("link_child")
    class Child(_InertResource):
        def link(self, components):
            linked.append("link_child")

    @requires_resource("link_child")
    @resource("link_parent")
    class Parent(_InertResource):
        def link(self, components):
            linked.append("link_parent")
            self.child = components.get_resource("link_child")

    components = _activate({"link_parent": {}})

    assert linked == ["link_child", "link_parent"]
    parent = components.get_resource("link_parent")
    assert parent.child is components.get_resource("link_child")


def test_link_resolves_dependencies_through_bindings():
    @resource("link_real_encoder")
    class Encoder(_InertResource):
        pass

    @requires_resource("link_encoder")
    @resource("link_model")
    class Model(_InertResource):
        def link(self, components):
            self.encoder = components.get_resource("link_encoder")

    components = SessionComponents(
        component_bindings={"link_encoder": "link_real_encoder"},
    )
    components.register_from_config({"link_model": {}})
    components.link_components()

    model = components.get_resource("link_model")
    assert model.encoder is components.get_resource("link_real_encoder")


def test_components_without_a_link_override_take_the_unlinked_path():
    @resource("link_plain")
    class Plain(_InertResource):
        pass

    components = _activate({"link_plain": {}})

    assert components._any_component_links() is False
    assert components.get_resource("link_plain") is not None


def test_requesting_an_undeclared_resource_while_linking_is_rejected():
    @resource("link_secret")
    class Secret(_InertResource):
        pass

    @resource("link_snooper")
    class Snooper(_InertResource):
        def link(self, components):
            components.get_resource("link_secret")

    with pytest.raises(ComponentLinkError) as raised:
        _activate({"link_snooper": {}, "link_secret": {}})

    message = str(raised.value)
    assert "link_snooper" in message
    assert "@requires_resource('link_secret')" in message


def test_a_dependency_cycle_is_reported_before_any_component_links():
    linked = []

    @requires_resource("link_cycle_b")
    @resource("link_cycle_a")
    class A(_InertResource):
        def link(self, components):
            linked.append("a")

    @requires_resource("link_cycle_a")
    @resource("link_cycle_b")
    class B(_InertResource):
        def link(self, components):
            linked.append("b")

    with pytest.raises(RuntimeError, match="Cyclic dependency"):
        _activate({"link_cycle_a": {}})

    assert linked == []


def test_linking_again_reuses_the_same_dependency_instances():
    @resource("link_relink_child")
    class Child(_InertResource):
        pass

    @requires_resource("link_relink_child")
    @resource("link_relink_parent")
    class Parent(_InertResource):
        def __init__(self, config=None):
            self.link_calls = 0

        def link(self, components):
            self.link_calls += 1
            self.child = components.get_resource("link_relink_child")

    components = _activate({"link_relink_parent": {}})
    parent = components.get_resource("link_relink_parent")
    child = components.get_resource("link_relink_child")

    components.link_components()

    assert parent.link_calls == 2
    assert parent.child is child


def test_a_failed_link_leaves_the_graph_stale_so_it_is_retried():
    attempts = []

    @resource("link_flaky")
    class Flaky(_InertResource):
        def link(self, components):
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise RuntimeError("first link fails")

    components = _activate({"link_flaky": {}}, link=False)

    with pytest.raises(RuntimeError, match="first link fails"):
        components.link_components()
    assert components._links_dirty is True

    components.ensure_linked()

    assert len(attempts) == 2
    assert components._links_dirty is False


def test_registering_a_component_marks_links_stale_and_entering_relinks(tmp_path):
    @resource("link_late_child")
    class Child(_InertResource):
        pass

    @requires_resource("link_late_child")
    @resource("link_late_parent")
    class Parent(_InertResource):
        def link(self, components):
            self.child = components.get_resource("link_late_child")

    config = make_config(tmp_path / "late-link")
    config["link_late_parent"] = {}
    session = TrainingSession(config)
    parent = session.get_resource("link_late_parent")

    replacement = Child()
    session.register_resource(replacement, overwrite=True)
    assert session._components._links_dirty is True

    with session:
        assert parent.child is replacement


def test_relinking_after_setup_is_rejected(tmp_path):
    @resource("link_frozen")
    class Frozen(_InertResource):
        def link(self, components):
            pass

    config = make_config(tmp_path / "frozen-link")
    config["link_frozen"] = {}
    session = TrainingSession(config)

    with session:
        with pytest.raises(RuntimeError, match="NEW phase"):
            session.relink_components()
