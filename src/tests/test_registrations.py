"""Tests for global component registration and dependency validation.

These tests target training-framework commit
4e07ffacf33ca2cc54d5ba1b24d7915d280c8b4c.

All decorated classes are created inside test functions.  This is intentional:
the repository's autouse registry fixture clears the global registries before
each test, so classes registered during pytest collection would be removed
before their tests run.
"""

from collections.abc import Callable

import pytest

from training_framework.training_session import (
    Component,
    Hook,
    Resource,
    Step,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
    topological_sort_of_components,
)
from training_framework.registry import _COMPONENT_REGISTRY, _component


ClassFactory = Callable[[str], type]


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


COMPONENT_CASES = (
    pytest.param(
        resource,
        make_resource_class,
        "Resource",
        id="resource",
    ),
    pytest.param(
        hook,
        make_hook_class,
        "Hook",
        id="hook",
    ),
    pytest.param(
        step,
        make_step_class,
        "Step",
        id="step",
    ),
)


@pytest.fixture(autouse=True)
def isolated_component_registries():
    """Give every test a clean registry and restore prior state afterward."""

    snapshot = dict(_COMPONENT_REGISTRY)
    _COMPONENT_REGISTRY.clear()

    try:
        yield
    finally:
        _COMPONENT_REGISTRY.clear()
        _COMPONENT_REGISTRY.update(snapshot)


@pytest.mark.parametrize(
    ("decorator", "class_factory", "kind"),
    COMPONENT_CASES,
)
def test_component_decorator_registers_class_and_sets_metadata(
    decorator,
    class_factory: ClassFactory,
    kind: str,
):
    component_class = class_factory("RegisteredComponent")

    decorated_class = decorator("registered_component")(component_class)

    assert decorated_class is component_class
    assert _COMPONENT_REGISTRY == {"registered_component": component_class}
    assert component_class.name == "registered_component"
    assert component_class.id == f"{kind}.registered_component"


@pytest.mark.parametrize(
    ("decorator", "class_factory", "kind"),
    COMPONENT_CASES,
)
def test_duplicate_component_name_is_rejected_without_overwriting_registry(
    decorator,
    class_factory: ClassFactory,
    kind: str,
):
    first_class = class_factory(f"First{kind}")
    second_class = class_factory(f"Second{kind}")
    decorator("duplicate_name")(first_class)

    with pytest.raises(ValueError, match="already registered"):
        decorator("duplicate_name")(second_class)

    assert _COMPONENT_REGISTRY == {"duplicate_name": first_class}


@pytest.mark.parametrize(
    ("decorator", "wrong_class_factory", "expected_base"),
    (
        pytest.param(resource, make_hook_class, "Resource", id="resource-with-hook"),
        pytest.param(hook, make_resource_class, "Hook", id="hook-with-resource"),
        pytest.param(step, make_hook_class, "Step", id="step-with-hook"),
    ),
)
def test_component_decorator_rejects_wrong_component_type(
    decorator,
    wrong_class_factory: ClassFactory,
    expected_base: str,
):
    wrong_class = wrong_class_factory("WrongComponent")

    with pytest.raises(TypeError, match=rf"must be subclass of {expected_base}"):
        decorator("wrong_component")(wrong_class)



def test_internal_component_decorator_rejects_missing_or_ambiguous_category():
    class BareComponent(Component):
        pass

    class AmbiguousComponent(Resource, Step):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def run(self, session):
            pass

    with pytest.raises(TypeError, match="exactly one component category"):
        _component("bare_component")(BareComponent)
    with pytest.raises(TypeError, match="abstract"):
        BareComponent()
    with pytest.raises(TypeError, match="found Resource, Step"):
        _component("ambiguous_component")(AmbiguousComponent)


def test_same_category_registration_can_be_overwritten_explicitly():
    first_class = resource("replaceable")(
        make_resource_class("FirstResource")
    )
    replacement_class = resource("replaceable", overwrite=True)(
        make_resource_class("ReplacementResource")
    )

    assert replacement_class is not first_class
    assert _COMPONENT_REGISTRY["replaceable"] is replacement_class
    assert replacement_class.id == "Resource.replaceable"


def test_component_names_are_unique_across_categories():
    shared_name = "shared_name"

    resource_class = resource(shared_name)(make_resource_class("SharedResource"))
    with pytest.raises(ValueError, match="already registered"):
        hook(shared_name)(make_hook_class("SharedHook"))
    with pytest.raises(ValueError, match="already registered"):
        step(shared_name)(make_step_class("SharedStep"))

    assert _COMPONENT_REGISTRY == {shared_name: resource_class}
    assert resource_class.id == "Resource.shared_name"


@pytest.mark.parametrize(
    ("dependency_decorator", "class_factory", "attribute"),
    (
        pytest.param(
            requires_resource,
            make_resource_class,
            "required_resources",
            id="resource-requires-resource",
        ),
        pytest.param(
            requires_resource,
            make_hook_class,
            "required_resources",
            id="hook-requires-resource",
        ),
        pytest.param(
            requires_resource,
            make_step_class,
            "required_resources",
            id="step-requires-resource",
        ),
        pytest.param(
            requires_hook,
            make_hook_class,
            "required_hooks",
            id="hook-requires-hook",
        ),
        pytest.param(
            requires_hook,
            make_step_class,
            "required_hooks",
            id="step-requires-hook",
        ),
        pytest.param(
            requires_step,
            make_step_class,
            "required_steps",
            id="step-requires-step",
        ),
    ),
)
def test_dependency_decorator_records_valid_requirement(
    dependency_decorator,
    class_factory: ClassFactory,
    attribute: str,
):
    component_class = class_factory("Consumer")

    decorated_class = dependency_decorator("dependency")(component_class)

    assert decorated_class is component_class
    assert getattr(component_class, attribute) == ["dependency"]


@pytest.mark.parametrize(
    ("dependency_decorator", "class_factory"),
    (
        pytest.param(
            requires_resource,
            lambda name: type(name, (), {}),
            id="plain-class-requires-resource",
        ),
        pytest.param(
            requires_hook,
            make_resource_class,
            id="resource-requires-hook",
        ),
        pytest.param(
            requires_hook,
            lambda name: type(name, (), {}),
            id="plain-class-requires-hook",
        ),
        pytest.param(
            requires_step,
            make_resource_class,
            id="resource-requires-step",
        ),
        pytest.param(
            requires_step,
            make_hook_class,
            id="hook-requires-step",
        ),
        pytest.param(
            requires_step,
            lambda name: type(name, (), {}),
            id="plain-class-requires-step",
        ),
    ),
)
def test_dependency_decorator_rejects_invalid_consumer_type(
    dependency_decorator,
    class_factory: ClassFactory,
):
    component_class = class_factory("InvalidConsumer")

    with pytest.raises(TypeError):
        dependency_decorator("dependency")(component_class)


@pytest.mark.parametrize(
    ("dependency_decorator", "attribute"),
    (
        pytest.param(
            requires_resource,
            "required_resources",
            id="multiple-resources",
        ),
        pytest.param(requires_hook, "required_hooks", id="multiple-hooks"),
        pytest.param(requires_step, "required_steps", id="multiple-steps"),
    ),
)
def test_multiple_requirements_preserve_decorator_application_order(
    dependency_decorator,
    attribute: str,
):
    component_class = make_step_class("Consumer")

    component_class = dependency_decorator("first")(component_class)
    component_class = dependency_decorator("second")(component_class)

    assert getattr(component_class, attribute) == ["first", "second"]


def test_resource_requirements_on_subclass_do_not_mutate_parent_class():
    parent_class = requires_resource("parent_resource")(
        make_step_class("ParentStep")
    )

    class ChildStep(parent_class):
        pass

    child_class = requires_resource("child_resource")(ChildStep)

    assert parent_class.required_resources == ["parent_resource"]
    assert child_class.required_resources == [
        "parent_resource",
        "child_resource",
    ]



def test_topological_sort_returns_empty_mapping_for_empty_registries():
    assert topological_sort_of_components() == {}



def test_topological_sort_orders_full_cross_category_dependency_chain():
    # Register consumers before their prerequisites to prove that registry
    # insertion order does not determine the resulting dependency order.
    optimizer = step("optimizer")(
        requires_step("backward")(make_step_class("OptimizerStep"))
    )
    backward = step("backward")(
        requires_hook("metrics")(make_step_class("BackwardStep"))
    )
    metrics = hook("metrics")(
        requires_hook("base_hook")(make_hook_class("MetricsHook"))
    )
    base_hook = hook("base_hook")(
        requires_resource("model")(make_hook_class("BaseHook"))
    )
    model = resource("model")(
        requires_resource("config")(make_resource_class("ModelResource"))
    )
    config = resource("config")(make_resource_class("ConfigResource"))

    order = topological_sort_of_components()
    dependency_chain = (
        config.id,
        model.id,
        base_hook.id,
        metrics.id,
        backward.id,
        optimizer.id,
    )

    assert set(order) == set(dependency_chain)
    assert sorted(order.values()) == list(range(len(dependency_chain)))
    assert all(
        order[prerequisite] < order[dependent]
        for prerequisite, dependent in zip(
            dependency_chain,
            dependency_chain[1:],
        )
    )


@pytest.mark.parametrize(
    ("dependency_decorator", "missing_name"),
    (
        pytest.param(
            requires_resource,
            "missing_resource",
            id="missing-resource",
        ),
        pytest.param(requires_hook, "missing_hook", id="missing-hook"),
        pytest.param(requires_step, "missing_step", id="missing-step"),
    ),
)
def test_topological_sort_rejects_missing_prerequisite(
    dependency_decorator,
    missing_name: str,
):
    consumer = dependency_decorator(missing_name)(
        make_step_class("ConsumerStep")
    )
    step("consumer")(consumer)

    with pytest.raises(RuntimeError, match=missing_name):
        topological_sort_of_components()


@pytest.mark.parametrize(
    (
        "existing_decorator",
        "existing_class_factory",
        "dependency_decorator",
    ),
    (
        pytest.param(
            hook,
            make_hook_class,
            requires_resource,
            id="hook-does-not-satisfy-resource",
        ),
        pytest.param(
            resource,
            make_resource_class,
            requires_hook,
            id="resource-does-not-satisfy-hook",
        ),
        pytest.param(
            hook,
            make_hook_class,
            requires_step,
            id="hook-does-not-satisfy-step",
        ),
    ),
)
def test_prerequisite_name_must_exist_in_correct_registry(
    existing_decorator,
    existing_class_factory: ClassFactory,
    dependency_decorator,
):
    shared_name = "same_name_wrong_category"
    existing_decorator(shared_name)(existing_class_factory("ExistingComponent"))

    consumer = dependency_decorator(shared_name)(
        make_step_class("ConsumerStep")
    )
    step("consumer")(consumer)

    with pytest.raises(RuntimeError, match=shared_name):
        topological_sort_of_components()



def test_overwrite_cannot_change_component_category():
    shared_name = "shared_dependency"

    shared_resource = resource(shared_name)(
        make_resource_class("SharedResource")
    )
    with pytest.raises(ValueError, match="Cannot overwrite Resource"):
        hook(shared_name, overwrite=True)(make_hook_class("SharedHook"))

    assert _COMPONENT_REGISTRY[shared_name] is shared_resource



def test_topological_sort_rejects_cyclic_dependencies():
    resource_a = requires_resource("resource_b")(
        make_resource_class("ResourceA")
    )
    resource_b = requires_resource("resource_a")(
        make_resource_class("ResourceB")
    )

    resource("resource_a")(resource_a)
    resource("resource_b")(resource_b)

    with pytest.raises(RuntimeError, match="Cyclic dependency"):
        topological_sort_of_components()