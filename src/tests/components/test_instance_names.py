"""Tests reserving the instance separator in component names.

A session holds at most one instance of each registered component, so '#'
carries no meaning yet. It is rejected wherever a component name is written
down so the spelling stays available for per-instance names later: a name
already in use could not be given a new meaning without breaking it.

Classes are created inside test functions because the autouse registry
fixture clears the global registries before each test.
"""

import pytest

from training_framework.components import (
    ComponentBindings,
    Hook,
    Resource,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    role,
    step,
    wraps,
)
from training_framework.components.naming import (
    INSTANCE_SEPARATOR,
    validate_component_name,
)


def make_resource_class(class_name: str) -> type[Resource]:
    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass

    return type(
        class_name,
        (Resource,),
        {"setup": setup, "teardown": teardown},
    )


def test_separator_is_the_reserved_character():
    assert INSTANCE_SEPARATOR == "#"


def test_registering_a_resource_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        resource("my_resource#2")


def test_registering_a_hook_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        hook("my_hook#2")


def test_registering_a_step_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        step("my_step#2")


def test_declaring_a_role_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="Role name 'model#2'"):
        role("model#2", Resource)


def test_requiring_a_resource_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="Required Resource name 'model#b'"):
        requires_resource("model#b")


def test_requiring_a_hook_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="Required Hook name"):
        requires_hook("logger#2")


def test_requiring_a_step_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="Required Step name"):
        requires_step("train_step#2")


def test_wrapping_a_hook_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="Wrapped Hook name"):
        wraps("optimizer#2")


def test_binding_a_role_name_with_the_separator_is_rejected():
    with pytest.raises(ValueError, match="Component binding role name"):
        ComponentBindings({"model#2": "trained_model"})


# A binding *target* may name an instance -- that is how a consumer is
# pointed at one. See tests/components/test_instance_wiring.py.


def test_the_rejection_explains_why_the_character_is_reserved():
    with pytest.raises(ValueError) as error:
        resource("my_resource#2")

    message = str(error.value)
    assert "my_resource#2" in message
    assert "per-instance component names" in message


def test_a_non_string_name_keeps_its_own_error():
    # The separator check has nothing to say about a wrong type, so the
    # caller's own reporting must be left intact.
    validate_component_name(None)
    with pytest.raises(TypeError):
        ComponentBindings({"model": 2})


def test_ordinary_names_are_unaffected():
    registered = resource("plain_resource")(make_resource_class("Plain"))
    assert registered.name == "plain_resource"

    declaration = role("plain_role", Hook)
    assert declaration.name == "plain_role"

    bindings = ComponentBindings({"some_role": "plain_resource"})
    assert bindings.resolve("some_role") == "plain_resource"


def test_ordinary_dependency_declarations_are_unaffected():
    decorated = requires_resource("plain_resource")(
        make_resource_class("Consumer")
    )
    assert "plain_resource" in decorated.required_resources
