"""Tests for deciding which instance satisfies a declared dependency.

Three rules, in order: the consumer's own wiring; otherwise the sole active
instance of the component; otherwise an error. Configuration cannot ask for a
second instance yet, so these tests add one through the session's own
machinery, which is the state a later phase reaches from config.
"""

import pytest

from training_framework.components import (
    ComponentBindings,
    ComponentDependencyError,
    Resource,
    requires_resource,
    resource,
)
from training_framework.session.components import SessionComponents


def make_components(bindings=None):
    """Return session components holding `wire_dep` twice and a consumer class."""

    @resource("wire_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("wire_dep")
    @resource("wire_consumer")
    class Consumer(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.dependency = self.get_dependency("wire_dep")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    components = SessionComponents(
        component_bindings=bindings,
        session_type="training",
    )
    return components, Dependency, Consumer


def add_instance(components, component_class, name):
    components.components[name] = components._construct(component_class, name)
    return components.components[name]


# -- the resolution rules -------------------------------------------------


def test_wiring_applies_only_to_the_consumer_that_declared_it():
    components, Dependency, _ = make_components(
        {"wire_consumer": {"wire_dep": "wire_dep#b"}},
    )
    add_instance(components, Dependency, "wire_dep#a")
    add_instance(components, Dependency, "wire_dep#b")

    with pytest.raises(ComponentDependencyError):
        components.resolve_dependency("wire_dep", consumer="someone_else")


def test_a_name_with_nothing_behind_it_is_left_to_the_caller():
    components, _, _ = make_components()

    # Not an ambiguity error: reporting an unconfigured component belongs to
    # whoever knows what it was being looked up for.
    assert components.resolve_dependency("wire_dep") == "wire_dep"


# -- lookups that apply the rules -----------------------------------------


def test_an_ambiguous_resource_still_counts_as_present():
    components, Dependency, _ = make_components()
    add_instance(components, Dependency, "wire_dep#a")
    add_instance(components, Dependency, "wire_dep#b")

    # The question is whether such a resource exists, and it does; which one
    # is meant is decided by asking for it.
    assert components.has_resource("wire_dep")


# -- binding validation ----------------------------------------------------


def test_a_binding_target_may_name_an_instance():
    make_components()

    bindings = ComponentBindings({"wire_role": "wire_dep#b"})

    assert bindings.resolve("wire_role") == "wire_dep#b"


def test_a_binding_role_name_may_not_name_an_instance():
    make_components()

    with pytest.raises(ValueError, match="Component binding role name"):
        ComponentBindings({"wire_role#2": "wire_dep"})


def test_a_binding_target_naming_an_unregistered_component_is_rejected():
    make_components()

    with pytest.raises(ValueError, match="not a registered component"):
        ComponentBindings({"wire_role": "not_a_component#b"})


def test_a_binding_target_with_a_malformed_suffix_is_rejected():
    make_components()

    with pytest.raises(ValueError, match="invalid instance suffix"):
        ComponentBindings({"wire_role": "wire_dep#"})


def test_per_consumer_wiring_is_validated():
    make_components()

    with pytest.raises(ValueError, match="not a registered component"):
        ComponentBindings({"wire_consumer": {"wire_dep": "nope#b"}})


def test_consumer_wiring_wins_over_the_session_wide_binding():
    make_components()

    bindings = ComponentBindings({
        "wire_role": "wire_dep",
        "wire_consumer": {"wire_role": "wire_dep#b"},
    })

    assert bindings.resolve("wire_role") == "wire_dep"
    assert bindings.resolve(
        "wire_role",
        consumer="wire_consumer",
    ) == "wire_dep#b"


def test_bindings_pickled_before_per_consumer_wiring_still_resolve():
    make_components()
    bindings = ComponentBindings({"wire_role": "wire_dep"})

    state = bindings.__dict__.copy()
    del state["_instance_bindings"]
    restored = ComponentBindings.__new__(ComponentBindings)
    restored.__setstate__(state)

    assert restored.resolve("wire_role") == "wire_dep"
    assert restored.instance_bindings == {}
