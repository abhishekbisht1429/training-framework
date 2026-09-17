from __future__ import annotations

import pytest

from training_framework.components import (
    LifecycleHook,
    Resource,
    Step,
    hook,
    requires_resource,
    resource,
    role,
    step,
    topological_sort_of_components,
)
from training_framework.components.diagnostics import explain_missing_component
from training_framework.session.components import (
    ComponentNotFoundError,
    SessionComponents,
)


class _Resource(Resource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class _Hook(LifecycleHook):
    call_every = 1

    def pre_session(self, session):
        pass

    def post_session(self, session):
        pass

    def pre_iteration_callback(self, session):
        pass

    def post_iteration_callback(self, session):
        pass


class _Step(Step):
    def run(self, session):
        pass


def _new_class(base, name):
    return type(name, (base,), {})


def _consumer(dependency, *, session_type=None, name="diag_consumer"):
    return requires_resource(dependency)(
        step(name, session_type=session_type)(_new_class(_Step, "DiagConsumer"))
    )


def _activate(config, *, session_type="training", bindings=None):
    components = SessionComponents(
        session_type=session_type,
        component_bindings=bindings,
    )
    components.register_from_config(config)
    return components


def _error(error_type, action):
    with pytest.raises(error_type) as raised:
        action()
    return str(raised.value)


def test_unknown_dependency_reports_all_registries_and_suggests_names():
    resource("diag_tensorboard")(_new_class(_Resource, "DiagTensorboard"))
    _consumer("diag_tensorbord")

    message = _error(RuntimeError, lambda: _activate({"diag_consumer": {}}))

    assert message.startswith("unmet prerequisite! Resource 'diag_tensorbord'")
    assert "Required by: Step 'diag_consumer' (DiagConsumer)" in message
    assert (
        "No component named 'diag_tensorbord' is registered in the shared "
        "registry or the 'training' registry, or for any other session type"
    ) in message
    assert "@resource('diag_tensorbord')" in message
    assert "Did you mean: 'diag_tensorboard'" in message


def test_dependency_registered_only_for_another_session_type():
    resource("diag_trainer_only", session_type="training")(
        _new_class(_Resource, "DiagTrainerOnly")
    )
    _consumer("diag_trainer_only", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: _activate({"diag_consumer": {}}, session_type="analysis"),
    )

    assert (
        "'diag_trainer_only' is not in the shared registry or the 'analysis' "
        "registry; it is registered only for session type(s) 'training' "
        "(as Resource), so it is unavailable to 'analysis' sessions"
    ) in message
    assert "@resource('diag_trainer_only', session_type='analysis')" in message
    assert "@resource('diag_trainer_only')" in message


def test_root_registered_only_for_another_session_type_names_its_category():
    hook("diag_training_hook", session_type="training")(
        _new_class(_Hook, "DiagTrainingHook")
    )

    message = _error(
        ValueError,
        lambda: _activate({"diag_training_hook": {}}, session_type="analysis"),
    )

    assert message.startswith(
        "No step, hook or resource registered with name 'diag_training_hook'!"
    )
    assert "registered only for session type(s) 'training' (as Hook)" in message
    assert "@hook('diag_training_hook', session_type='analysis')" in message


def test_dependency_registered_with_the_wrong_category():
    hook("diag_hook")(_new_class(_Hook, "DiagHook"))
    _consumer("diag_hook")

    message = _error(RuntimeError, lambda: _activate({"diag_consumer": {}}))

    assert (
        "'diag_hook' is registered in the shared registry as a Hook "
        "(DiagHook), not as a Resource"
    ) in message


def test_scoped_component_shadowing_a_shared_one_with_another_category():
    resource("diag_shadowed")(_new_class(_Resource, "SharedResource"))
    hook("diag_shadowed", session_type="analysis")(
        _new_class(_Hook, "AnalysisHook")
    )
    _consumer("diag_shadowed", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: _activate({"diag_consumer": {}}, session_type="analysis"),
    )

    assert "registered in the 'analysis' registry as a Hook" in message
    assert (
        "It overrides the shared Resource 'diag_shadowed' (SharedResource) "
        "for 'analysis' sessions"
    ) in message


def test_binding_is_named_when_the_bound_implementation_is_wrong():
    hook("diag_bound_hook")(_new_class(_Hook, "DiagBoundHook"))
    _consumer("diag_role")

    message = _error(
        RuntimeError,
        lambda: _activate(
            {"diag_consumer": {}},
            bindings={"diag_role": "diag_bound_hook"},
        ),
    )

    assert "Binding: component_bindings maps 'diag_role' to 'diag_bound_hook'" in message
    assert "registered in the shared registry as a Hook" in message


def test_binding_target_registered_only_for_another_session_type():
    resource("diag_training_impl", session_type="training")(
        _new_class(_Resource, "DiagTrainingImpl")
    )
    resource("diag_role_owner", session_type="analysis")(
        _new_class(_Resource, "DiagRoleOwner")
    )

    message = _error(
        ValueError,
        lambda: SessionComponents(
            session_type="analysis",
            component_bindings={"diag_role_owner": "diag_training_impl"},
        ),
    )

    assert message.startswith(
        "Component binding target 'diag_training_impl' is not a registered "
        "component"
    )
    assert "registered only for session type(s) 'training' (as Resource)" in message


def test_declared_role_without_implementation_keeps_role_message_and_adds_reason():
    role("diag_dataset", Resource, session_type="analysis")
    _consumer("diag_dataset", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: _activate({"diag_consumer": {}}, session_type="analysis"),
    )

    assert message.startswith("Role 'diag_dataset' (Resource)")
    assert (
        "Reason: 'diag_dataset' is declared as a Resource role in the "
        "'analysis' registry, but no implementation is registered"
    ) in message
    assert message.count("Fix:") == 0


def test_role_declared_only_for_another_session_type():
    role("diag_training_role", Resource, session_type="training")
    _consumer("diag_training_role", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: _activate({"diag_consumer": {}}, session_type="analysis"),
    )

    assert (
        "'diag_training_role' is declared as a role only for session type(s) "
        "'training'"
    ) in message
    assert "component_bindings: {'diag_training_role'" in message


def test_get_resource_registered_but_not_active():
    resource("diag_inactive")(_new_class(_Resource, "DiagInactive"))
    components = _activate({})

    with pytest.raises(KeyError) as raised:
        components.get_resource("diag_inactive")

    assert isinstance(raised.value, ComponentNotFoundError)
    message = str(raised.value)
    assert message.startswith("diag_inactive not found in resources!\n")
    assert (
        "'diag_inactive' is registered in the shared registry as a Resource "
        "(DiagInactive) but is not active in this session"
    ) in message
    assert "Add a top-level 'diag_inactive' mapping" in message


def test_get_resource_for_component_of_another_session_type():
    resource("diag_elsewhere", session_type="training")(
        _new_class(_Resource, "DiagElsewhere")
    )
    components = _activate({}, session_type="analysis")

    message = _error(KeyError, lambda: components.get_resource("diag_elsewhere"))

    assert "registered only for session type(s) 'training'" in message


def test_execution_order_reports_registered_but_unconfigured_dependency():
    resource("diag_unconfigured")(_new_class(_Resource, "DiagUnconfigured"))
    consumer_class = _consumer("diag_unconfigured")

    message = _error(
        RuntimeError,
        lambda: topological_sort_of_components(
            components=[consumer_class({})],
            session_type="training",
        ),
    )

    assert "which is not configured in this session" in message
    assert "Required by: Step 'diag_consumer'" in message
    assert "registered in the shared registry as a Resource" in message
    assert "not active in this session" in message


def test_dependency_closure_reports_unconfigured_component():
    resource("diag_closure")(_new_class(_Resource, "DiagClosure"))
    components = _activate({})

    message = _error(
        RuntimeError,
        lambda: components.dependency_closure(["diag_closure"]),
    )

    assert "is not configured in this session" in message
    assert "not active in this session" in message


def test_explanation_is_empty_when_lookup_would_succeed():
    resource("diag_present")(_new_class(_Resource, "DiagPresent"))

    assert explain_missing_component(
        "diag_present",
        "diag_present",
        expected_type=Resource,
        session_type="training",
        active_names={"diag_present"},
    ) == ""
