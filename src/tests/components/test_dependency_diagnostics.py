from __future__ import annotations

import pytest
import torch

from tests.test_utils import build_session
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
from training_framework.components.builtin import Checkpointer


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


def _checkpoint(tmp_path, *, session_type="training"):
    """Save a session holding only the defaults, and return its path."""
    path = tmp_path / "checkpoint.pt"
    torch.save(build_session(tmp_path, session_type=session_type), path)
    return path


def _error(error_type, action):
    with pytest.raises(error_type) as raised:
        action()
    return str(raised.value)


def test_unknown_dependency_reports_all_registries_and_suggests_names(tmp_path):
    resource("diag_tensorboard")(_new_class(_Resource, "DiagTensorboard"))
    _consumer("diag_tensorbord")

    message = _error(RuntimeError, lambda: build_session(tmp_path, {"diag_consumer": {}}))

    assert message.startswith("unmet prerequisite! Resource 'diag_tensorbord'")
    assert "Required by: Step 'diag_consumer' (DiagConsumer)" in message
    assert (
        "No component named 'diag_tensorbord' is registered in the shared "
        "registry or the 'training' registry, or for any other session type"
    ) in message
    assert "@resource('diag_tensorbord')" in message
    assert "Did you mean: 'diag_tensorboard'" in message


def test_dependency_registered_only_for_another_session_type(tmp_path):
    resource("diag_trainer_only", session_type="training")(
        _new_class(_Resource, "DiagTrainerOnly")
    )
    _consumer("diag_trainer_only", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: build_session(tmp_path, {"diag_consumer": {}}, session_type="analysis"),
    )

    assert (
        "'diag_trainer_only' is not in the shared registry or the 'analysis' "
        "registry; it is registered only for session type(s) 'training' "
        "(as Resource), so it is unavailable to 'analysis' sessions"
    ) in message
    assert "@resource('diag_trainer_only', session_type='analysis')" in message
    assert "@resource('diag_trainer_only')" in message


def test_root_registered_only_for_another_session_type_names_its_category(tmp_path):
    hook("diag_training_hook", session_type="training")(
        _new_class(_Hook, "DiagTrainingHook")
    )

    message = _error(
        ValueError,
        lambda: build_session(tmp_path, {"diag_training_hook": {}}, session_type="analysis"),
    )

    assert message.startswith(
        "No step, hook or resource registered with name 'diag_training_hook'!"
    )
    assert "registered only for session type(s) 'training' (as Hook)" in message
    assert "@hook('diag_training_hook', session_type='analysis')" in message


def test_dependency_registered_with_the_wrong_category(tmp_path):
    hook("diag_hook")(_new_class(_Hook, "DiagHook"))
    _consumer("diag_hook")

    message = _error(RuntimeError, lambda: build_session(tmp_path, {"diag_consumer": {}}))

    assert (
        "'diag_hook' is registered in the shared registry as a Hook "
        "(DiagHook), not as a Resource"
    ) in message


def test_scoped_component_shadowing_a_shared_one_with_another_category(tmp_path):
    resource("diag_shadowed")(_new_class(_Resource, "SharedResource"))
    hook("diag_shadowed", session_type="analysis")(
        _new_class(_Hook, "AnalysisHook")
    )
    _consumer("diag_shadowed", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: build_session(tmp_path, {"diag_consumer": {}}, session_type="analysis"),
    )

    assert "registered in the 'analysis' registry as a Hook" in message
    assert (
        "It overrides the shared Resource 'diag_shadowed' (SharedResource) "
        "for 'analysis' sessions"
    ) in message


def test_binding_is_named_when_the_bound_implementation_is_wrong(tmp_path):
    hook("diag_bound_hook")(_new_class(_Hook, "DiagBoundHook"))
    _consumer("diag_role")

    message = _error(
        RuntimeError,
        lambda: build_session(tmp_path, 
            {"diag_consumer": {}},
            bindings={"diag_role": "diag_bound_hook"},
        ),
    )

    assert "Binding: component_bindings maps 'diag_role' to 'diag_bound_hook'" in message
    assert "registered in the shared registry as a Hook" in message


def test_binding_target_registered_only_for_another_session_type(tmp_path):
    resource("diag_training_impl", session_type="training")(
        _new_class(_Resource, "DiagTrainingImpl")
    )
    resource("diag_role_owner", session_type="analysis")(
        _new_class(_Resource, "DiagRoleOwner")
    )

    message = _error(
        ValueError,
        lambda: build_session(
            tmp_path,
            session_type="analysis",
            bindings={"diag_role_owner": "diag_training_impl"},
        ),
    )

    assert message.startswith(
        "Component binding target 'diag_training_impl' is not a registered "
        "component"
    )
    assert "registered only for session type(s) 'training' (as Resource)" in message


def test_declared_role_without_implementation_keeps_role_message_and_adds_reason(tmp_path):
    role("diag_dataset", Resource, session_type="analysis")
    _consumer("diag_dataset", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: build_session(tmp_path, {"diag_consumer": {}}, session_type="analysis"),
    )

    assert message.startswith("Role 'diag_dataset' (Resource)")
    assert (
        "Reason: 'diag_dataset' is declared as a Resource role in the "
        "'analysis' registry, but no implementation is registered"
    ) in message
    assert message.count("Fix:") == 0


def test_role_declared_only_for_another_session_type(tmp_path):
    role("diag_training_role", Resource, session_type="training")
    _consumer("diag_training_role", session_type="analysis")

    message = _error(
        RuntimeError,
        lambda: build_session(tmp_path, {"diag_consumer": {}}, session_type="analysis"),
    )

    assert (
        "'diag_training_role' is declared as a role only for session type(s) "
        "'training'"
    ) in message
    assert "component_bindings: {'diag_training_role'" in message


def test_loading_a_registered_but_inactive_component(tmp_path):
    resource("diag_inactive")(_new_class(_Resource, "DiagInactive"))
    path = _checkpoint(tmp_path)

    with pytest.raises(KeyError) as raised:
        Checkpointer.load_component(path, "diag_inactive")

    message = str(raised.value)
    assert message.startswith("diag_inactive not found in resources!\n")
    assert (
        "'diag_inactive' is registered in the shared registry as a Resource "
        "(DiagInactive) but is not active in this session"
    ) in message
    assert "Add a top-level 'diag_inactive' mapping" in message


def test_loading_a_component_of_another_session_type(tmp_path):
    resource("diag_elsewhere", session_type="training")(
        _new_class(_Resource, "DiagElsewhere")
    )
    path = _checkpoint(tmp_path, session_type="analysis")

    message = _error(
        KeyError,
        lambda: Checkpointer.load_component(path, "diag_elsewhere"),
    )

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


def test_a_rank_plan_naming_an_unconfigured_component_is_reported(tmp_path):
    resource("diag_closure")(_new_class(_Resource, "DiagClosure"))
    session = build_session(tmp_path)

    message = _error(
        RuntimeError,
        lambda: session.rank_parallel_names(parallel_components=["diag_closure"]),
    )

    assert "is not configured in this session" in message
    assert "not active in this session" in message
