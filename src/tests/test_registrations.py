import pytest

from training_framework.training_session import (
    Hook,
    Resource,
    Step,
    TrainingSession,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
)


def make_session(tmp_path):
    return TrainingSession(
        {
            "rng_seed": 123,
            "sessions_dir": str(tmp_path),
            "max_iterations": 1,
            "device": "cpu",
        }
    )


class NoOpResource(Resource):
    def setup(self, session: "TrainingSession"):
        pass

    def teardown(self, session: "TrainingSession"):
        pass


class NoOpHook(Hook):
    pass


class NoOpStep(Step):
    def run(self, session: "TrainingSession") -> None:
        pass


def assert_unmet_prerequisites(action, *expected_names):
    """Assert that registration fails and identifies every missing dependency."""
    with pytest.raises(RuntimeError) as exc_info:
        action()

    message = str(exc_info.value)
    assert "unmet prerequisites" in message.lower()

    for name in expected_names:
        assert name in message


# ============================================================
# Decorator rule enforcement
# ============================================================

def test_resource_can_require_resource_only():
    @requires_resource("resource_dependency")
    class ValidResource(NoOpResource):
        pass

    assert ValidResource.required_resources == ["resource_dependency"]

    # A Resource cannot require a Hook.
    with pytest.raises(TypeError):
        @requires_hook("hook_dependency")
        class ResourceRequiringHook(NoOpResource):
            pass

    # A Resource cannot require a Step.
    with pytest.raises(TypeError):
        @requires_step("step_dependency")
        class ResourceRequiringStep(NoOpResource):
            pass


def test_hook_can_require_resource_or_hook_only():
    @requires_resource("resource_dependency")
    @requires_hook("hook_dependency")
    class ValidHook(NoOpHook):
        pass

    assert ValidHook.required_resources == ["resource_dependency"]
    assert ValidHook.required_hooks == ["hook_dependency"]

    # A Hook cannot require a Step.
    with pytest.raises(TypeError):
        @requires_step("step_dependency")
        class HookRequiringStep(NoOpHook):
            pass


def test_step_can_require_resource_hook_or_step():
    @requires_resource("resource_dependency")
    @requires_hook("hook_dependency")
    @requires_step("step_dependency")
    class ValidStep(NoOpStep):
        pass

    assert ValidStep.required_resources == ["resource_dependency"]
    assert ValidStep.required_hooks == ["hook_dependency"]
    assert ValidStep.required_steps == ["step_dependency"]


@pytest.mark.parametrize(
    "dependency_decorator",
    [
        requires_resource("resource_dependency"),
        requires_hook("hook_dependency"),
        requires_step("step_dependency"),
    ],
    ids=["requires-resource", "requires-hook", "requires-step"],
)
def test_dependency_decorators_reject_non_framework_classes(
    dependency_decorator,
):
    class PlainClass:
        pass

    with pytest.raises(TypeError):
        dependency_decorator(PlainClass)


def test_multiple_requirements_of_same_type_are_preserved():
    @requires_resource("second_resource")
    @requires_resource("first_resource")
    class ResourceWithMultipleRequirements(NoOpResource):
        pass

    # Decorators are applied from bottom to top.
    assert ResourceWithMultipleRequirements.required_resources == [
        "first_resource",
        "second_resource",
    ]


# ============================================================
# Resource prerequisite registration
# ============================================================

def test_register_resource_requires_registered_resource_prerequisite(tmp_path):
    session = make_session(tmp_path)

    @resource("reg_base_resource")
    class BaseResource(NoOpResource):
        pass

    @resource("reg_dependent_resource")
    @requires_resource("reg_base_resource")
    class DependentResource(NoOpResource):
        pass

    assert_unmet_prerequisites(
        lambda: session.register_resource(DependentResource()),
        "reg_base_resource",
    )

    # Failed registration must not partially modify the session.
    assert session._resources == {}

    base_id = session.register_resource(BaseResource())

    assert base_id in session._resources
    assert session.get_resource(base_id).name == "reg_base_resource"

    dependent_id = session.register_resource(DependentResource())

    assert dependent_id in session._resources
    assert session.get_resource(dependent_id).name == "reg_dependent_resource"


def test_register_resource_requires_all_declared_resources(tmp_path):
    session = make_session(tmp_path)

    @resource("first_required_resource")
    class FirstResource(NoOpResource):
        pass

    @resource("second_required_resource")
    class SecondResource(NoOpResource):
        pass

    @resource("multi_resource_consumer")
    @requires_resource("second_required_resource")
    @requires_resource("first_required_resource")
    class ResourceConsumer(NoOpResource):
        pass

    assert_unmet_prerequisites(
        lambda: session.register_resource(ResourceConsumer()),
        "first_required_resource",
        "second_required_resource",
    )

    session.register_resource(FirstResource())

    assert_unmet_prerequisites(
        lambda: session.register_resource(ResourceConsumer()),
        "second_required_resource",
    )

    session.register_resource(SecondResource())
    consumer_id = session.register_resource(ResourceConsumer())

    assert session.get_resource(consumer_id).name == "multi_resource_consumer"


# ============================================================
# Hook prerequisite registration
# ============================================================

def test_register_hook_requires_resource_and_hook_prerequisites(tmp_path):
    session = make_session(tmp_path)

    @resource("reg_hook_required_resource")
    class RequiredResource(NoOpResource):
        pass

    @hook("reg_hook_required_hook")
    class RequiredHook(NoOpHook):
        pass

    @hook("reg_dependent_hook")
    @requires_resource("reg_hook_required_resource")
    @requires_hook("reg_hook_required_hook")
    class DependentHook(NoOpHook):
        pass

    assert_unmet_prerequisites(
        lambda: session.register_hook(DependentHook()),
        "reg_hook_required_resource",
        "reg_hook_required_hook",
    )

    assert session._hooks == []

    session.register_resource(RequiredResource())

    assert_unmet_prerequisites(
        lambda: session.register_hook(DependentHook()),
        "reg_hook_required_hook",
    )

    # Failed registration still must not add the dependent hook.
    assert session._hooks == []

    session.register_hook(RequiredHook())
    session.register_hook(DependentHook())

    assert [registered_hook.name for registered_hook in session._hooks] == [
        "reg_hook_required_hook",
        "reg_dependent_hook",
    ]


# ============================================================
# Step prerequisite registration
# ============================================================

def test_add_step_requires_resource_hook_and_step_prerequisites(tmp_path):
    session = make_session(tmp_path)

    @resource("reg_step_required_resource")
    class RequiredResource(NoOpResource):
        pass

    @hook("reg_step_required_hook")
    class RequiredHook(NoOpHook):
        pass

    @step("reg_step_required_step")
    class RequiredStep(NoOpStep):
        pass

    @step("reg_dependent_step")
    @requires_resource("reg_step_required_resource")
    @requires_hook("reg_step_required_hook")
    @requires_step("reg_step_required_step")
    class DependentStep(NoOpStep):
        pass

    assert_unmet_prerequisites(
        lambda: session.add_step(DependentStep()),
        "reg_step_required_resource",
        "reg_step_required_hook",
        "reg_step_required_step",
    )

    assert session._steps == []

    session.register_resource(RequiredResource())

    assert_unmet_prerequisites(
        lambda: session.add_step(DependentStep()),
        "reg_step_required_hook",
        "reg_step_required_step",
    )

    assert session._steps == []

    session.register_hook(RequiredHook())

    assert_unmet_prerequisites(
        lambda: session.add_step(DependentStep()),
        "reg_step_required_step",
    )

    assert session._steps == []

    session.add_step(RequiredStep())
    session.add_step(DependentStep())

    assert [registered_step.name for registered_step in session._steps] == [
        "reg_step_required_step",
        "reg_dependent_step",
    ]


# ============================================================
# Dependency categories must be checked separately
# ============================================================

def test_prerequisite_name_must_exist_in_the_correct_component_category(
    tmp_path,
):
    """
    A matching name in the wrong registry must not satisfy a dependency.

    This catches implementations that combine all registered names into one
    set and then lose the distinction between resources, hooks, and steps.
    """

    shared_name = "same_name_different_component_type"

    @resource(shared_name)
    class SameNameResource(NoOpResource):
        pass

    @hook(shared_name)
    class SameNameHook(NoOpHook):
        pass

    @step(shared_name)
    class SameNameStep(NoOpStep):
        pass

    @hook("hook_requiring_same_name_resource")
    @requires_resource(shared_name)
    class HookRequiringResource(NoOpHook):
        pass

    @hook("hook_requiring_same_name_hook")
    @requires_hook(shared_name)
    class HookRequiringHook(NoOpHook):
        pass

    @step("step_requiring_same_name_step")
    @requires_step(shared_name)
    class StepRequiringStep(NoOpStep):
        pass

    # A Hook with the matching name must not satisfy a Resource dependency.
    hook_only_session = make_session(tmp_path / "hook-only")
    hook_only_session.register_hook(SameNameHook())

    assert_unmet_prerequisites(
        lambda: hook_only_session.register_hook(HookRequiringResource()),
        shared_name,
    )

    # But it must satisfy a Hook dependency.
    hook_only_session.register_hook(HookRequiringHook())

    assert [item.name for item in hook_only_session._hooks] == [
        shared_name,
        "hook_requiring_same_name_hook",
    ]

    # A Resource with the matching name must not satisfy a Hook dependency.
    resource_only_session = make_session(tmp_path / "resource-only")
    resource_only_session.register_resource(SameNameResource())

    assert_unmet_prerequisites(
        lambda: resource_only_session.register_hook(HookRequiringHook()),
        shared_name,
    )

    # But it must satisfy a Resource dependency.
    resource_only_session.register_hook(HookRequiringResource())

    assert resource_only_session._hooks[-1].name == (
        "hook_requiring_same_name_resource"
    )

    # Neither a Resource nor Hook with the matching name may satisfy a
    # Step dependency.
    no_step_session = make_session(tmp_path / "no-step")
    no_step_session.register_resource(SameNameResource())
    no_step_session.register_hook(SameNameHook())

    assert_unmet_prerequisites(
        lambda: no_step_session.add_step(StepRequiringStep()),
        shared_name,
    )

    no_step_session.add_step(SameNameStep())
    no_step_session.add_step(StepRequiringStep())

    assert [item.name for item in no_step_session._steps] == [
        shared_name,
        "step_requiring_same_name_step",
    ]