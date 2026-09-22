"""Tests for component bindings whose targets name an instance.

Which instance satisfies a dependency is tested through configured sessions in
`test_multi_instance.py`; these tests cover what `ComponentBindings` itself
accepts and how it resolves.
"""

import re

import pytest

from training_framework.components import (
    ComponentBindings,
    Resource,
    requires_resource,
    resource,
)


def declare_components():
    """Register `wire_dep` and a `wire_consumer` requiring it, so bindings
    have registered targets to name."""

    @resource("wire_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("wire_dep")
    @resource("wire_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass


def test_a_binding_target_may_name_an_instance():
    declare_components()

    bindings = ComponentBindings({"wire_role": "wire_dep#b"})

    assert bindings.resolve("wire_role") == "wire_dep#b"


def test_a_binding_role_name_may_not_name_an_instance():
    declare_components()

    with pytest.raises(ValueError, match="Component binding role name"):
        ComponentBindings({"wire_role#2": "wire_dep"})


def test_a_binding_target_naming_an_unregistered_component_is_rejected():
    declare_components()

    with pytest.raises(ValueError, match="not a registered component"):
        ComponentBindings({"wire_role": "not_a_component#b"})


@pytest.mark.parametrize(
    ("target", "match"),
    (
        ("wire_dep#", "invalid instance suffix ''"),
        ("wire_dep#a.b", "invalid instance suffix 'a.b'"),
        ("#2", "no component name before '#'"),
    ),
)
def test_a_binding_target_with_a_malformed_instance_name_is_rejected(
        target,
        match,
):
    declare_components()

    with pytest.raises(ValueError, match=re.escape(match)):
        ComponentBindings({"wire_role": target})


def test_per_consumer_wiring_is_validated():
    declare_components()

    with pytest.raises(ValueError, match="not a registered component"):
        ComponentBindings({"wire_consumer": {"wire_dep": "nope#b"}})


def test_consumer_wiring_wins_over_the_session_wide_binding():
    declare_components()

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
    declare_components()
    bindings = ComponentBindings({"wire_role": "wire_dep"})

    state = bindings.__dict__.copy()
    del state["_instance_bindings"]
    restored = ComponentBindings.__new__(ComponentBindings)
    restored.__setstate__(state)

    assert restored.resolve("wire_role") == "wire_dep"
    assert restored.instance_bindings == {}
