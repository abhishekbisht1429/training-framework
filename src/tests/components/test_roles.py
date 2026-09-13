"""Tests for the abstract "role" declaration API (`role`/`role_registry`).

All decorated classes are created inside test functions.  This is intentional:
the repository's autouse registry fixture clears the global registries (and,
now, the role registries) before each test, so classes/roles declared during
pytest collection would be removed before their tests run.
"""

import pytest

from training_framework.components import (
    ComponentBindings,
    Hook,
    Resource,
    RoleDeclaration,
    Step,
    component_registry,
    hook,
    requires_resource,
    resource,
    role,
    role_registry,
    step,
    topological_sort_of_components,
)


def make_resource_class(class_name: str) -> type[Resource]:
    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass

    return type(
        class_name,
        (Resource,),
        {
            "setup": setup,
            "teardown": teardown,
        },
    )


def make_hook_class(class_name: str) -> type[Hook]:
    return type(class_name, (Hook,), {})


def make_step_class(class_name: str) -> type[Step]:
    def run(self, session) -> None:
        pass

    return type(class_name, (Step,), {"run": run})


def test_role_declares_expected_category_and_description():
    declaration = role("described_role", Resource, description="a thing")

    assert declaration == RoleDeclaration("described_role", Resource, "a thing")
    assert role_registry()["described_role"] == declaration
    assert "described_role" not in component_registry()


def test_role_rejects_non_component_category():
    with pytest.raises(TypeError, match="Resource, Hook, or Step"):
        role("bad_role", int)


def test_role_redeclaration_without_overwrite_raises():
    role("dup_role", Resource)

    with pytest.raises(ValueError, match="already declared"):
        role("dup_role", Step)


def test_role_overwrite_replaces_declaration():
    role("overwrite_role", Resource, description="old")

    updated = role("overwrite_role", Step, description="new", overwrite=True)

    assert role_registry()["overwrite_role"] == updated
    assert updated.category is Step
    assert updated.description == "new"


def test_role_rejects_category_mismatch_against_registered_component():
    resource("already_registered")(make_resource_class("AlreadyRegistered"))

    with pytest.raises(ValueError, match="already registered as a Resource"):
        role("already_registered", Step)


def test_component_registration_checks_role_registry_across_scopes():
    role("shared_declared_role", Resource)

    with pytest.raises(ValueError, match="declared as a Resource role"):
        hook("shared_declared_role", session_type="training")(
            make_hook_class("CrossScopeHook")
        )


def test_role_registry_is_session_scoped_like_component_registry():
    role("scoped_role", Resource, session_type="training")

    assert "scoped_role" in role_registry("training")
    assert "scoped_role" not in role_registry()
    assert "scoped_role" not in role_registry("analysis")


def test_declared_role_without_implementation_raises_descriptive_error():
    role("undone_role", Resource, description="needs an implementation")

    consumer = requires_resource("undone_role")(make_step_class("Consumer"))
    step("consumer_of_undone_role")(consumer)

    with pytest.raises(RuntimeError) as exc_info:
        topological_sort_of_components()

    message = str(exc_info.value)
    assert "undone_role" in message
    assert "Resource" in message
    assert "needs an implementation" in message
    assert "Consumer" in message
    assert "@resource('undone_role'" in message
    assert "component_bindings" in message


def test_undeclared_missing_dependency_still_uses_generic_message():
    consumer = requires_resource("truly_missing")(make_step_class("Consumer"))
    step("consumer_of_missing_name")(consumer)

    with pytest.raises(RuntimeError) as exc_info:
        topological_sort_of_components()

    message = str(exc_info.value)
    assert "unmet prerequisite!" in message
    assert "not registered as a Resource" in message
    assert "Implement a" not in message
    assert "declared as" not in message


def test_declared_role_satisfied_via_binding_to_differently_named_implementation():
    role("bindable_role", Resource)
    resource("bound_implementation")(make_resource_class("BoundImplementation"))

    consumer = requires_resource("bindable_role")(make_step_class("Consumer"))
    step("consumer_of_bound_role")(consumer)

    order = topological_sort_of_components(
        component_bindings={"bindable_role": "bound_implementation"},
    )

    assert order["Resource.bound_implementation"] < order["Step.consumer_of_bound_role"]


def test_component_bindings_rejects_category_mismatch_against_declared_role():
    role("resource_role", Resource)
    hook("hook_implementation")(make_hook_class("HookImplementation"))

    with pytest.raises(ValueError, match="declared as Resource"):
        ComponentBindings({"resource_role": "hook_implementation"})


def test_registering_wrong_category_under_declared_role_name_is_rejected():
    role("typed_role", Resource)

    with pytest.raises(ValueError, match="declared as a Resource role"):
        hook("typed_role")(make_hook_class("WrongCategoryHook"))


def test_role_public_api_is_importable():
    from training_framework.components import RoleDeclaration as imported_role_declaration
    from training_framework.components import role as imported_role
    from training_framework.components import role_registry as imported_role_registry

    assert imported_role is role
    assert imported_role_registry is role_registry
    assert imported_role_declaration is RoleDeclaration
