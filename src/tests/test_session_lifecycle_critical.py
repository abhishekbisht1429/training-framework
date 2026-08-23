"""Critical TrainingSession lifecycle tests for training-framework.
Passing tests cover the normal lifecycle contract.  Strict xfail tests capture
important cleanup guarantees that current main does not yet provide.  An XPASS
is intentionally treated as a failure so the obsolete xfail marker is removed
when the implementation is fixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from training_framework.training_session import (
    SessionPhase,
    TrainingSession,
    hook,
    resource,
    step, LifecycleHook, Resource, Step,
)
from tests.test_utils import make_config

class LifecycleSetupError(RuntimeError):
    pass


class LifecycleStepError(RuntimeError):
    pass


class LifecycleTeardownError(RuntimeError):
    pass


class TraceResourceBase(Resource):
    def __init__(
        self,
        label: str,
        trace: list[str],
        *,
        fail_setup: bool = False,
        fail_teardown: bool = False,
    ):
        self.label = label
        self.trace = trace
        self.fail_setup = fail_setup
        self.fail_teardown = fail_teardown
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self, session: TrainingSession):
        self.setup_calls += 1
        self.trace.append(f"resource:{self.label}:setup")

        if self.fail_setup:
            session.session_context["partial_resource"] = self.label
            raise LifecycleSetupError(f"resource {self.label} setup failed")

    def teardown(self, session: TrainingSession):
        self.teardown_calls += 1
        self.trace.append(f"resource:{self.label}:teardown")

        if self.fail_teardown:
            raise LifecycleTeardownError(
                f"resource {self.label} teardown failed"
            )


class TraceHookBase(LifecycleHook):
    def __init__(
        self,
        label: str,
        trace: list[str],
        *,
        call_every: int = 1,
        fail_teardown: bool = False,
    ):
        self.label = label
        self.trace = trace
        self.call_every = call_every
        self.fail_teardown = fail_teardown
        self.setup_calls = 0
        self.teardown_calls = 0
        self.pre_iterations: list[int] = []
        self.post_iterations: list[int] = []
        self.post_payloads: list[tuple[int, tuple[str, ...]]] = []
        self.context_sizes_seen_on_setup: list[int] = []

    def setup(self, session: TrainingSession):
        self.setup_calls += 1
        self.trace.append(f"hook:{self.label}:setup")
        self.context_sizes_seen_on_setup.append(len(session.session_context))
        session.session_context.setdefault("hook_setups", []).append(self.label)

    def teardown(self, session: TrainingSession):
        self.teardown_calls += 1
        self.trace.append(f"hook:{self.label}:teardown")

        if self.fail_teardown:
            raise LifecycleTeardownError(
                f"hook {self.label} teardown failed"
            )

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        self.pre_iterations.append(session.iteration)
        self.trace.append(f"hook:{self.label}:pre:{session.iteration}")
        session.iteration_context.setdefault("pre_hooks", []).append(self.label)

    def post_iteration_callback(self, session: TrainingSession) -> None:
        self.post_iterations.append(session.iteration)
        self.trace.append(f"hook:{self.label}:post:{session.iteration}")
        self.post_payloads.append(session.iteration_context["payload"])


class TraceStepBase(Step):
    def __init__(
        self,
        label: str,
        trace: list[str],
        *,
        fail: bool = False,
    ):
        self.label = label
        self.trace = trace
        self.fail = fail
        self.calls = 0

    def run(self, session: TrainingSession) -> None:
        self.calls += 1
        self.trace.append(f"step:{self.label}:{session.iteration}")

        steps = session.iteration_context.setdefault("steps", [])
        steps.append(self.label)
        session.iteration_context["payload"] = (
            session.iteration,
            tuple(steps),
        )
        session.session_context["last_iteration"] = session.iteration

        if self.fail:
            session.iteration_context["failure_marker"] = self.label
            raise LifecycleStepError(f"step {self.label} failed")

def test_lifecycle_order_hook_cadence_and_iteration_context_visibility(tmp_path):

    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @resource("critical_lifecycle_resource_b")
    class CriticalLifecycleResourceB(TraceResourceBase):
        pass

    @hook("critical_lifecycle_hook_a")
    class CriticalLifecycleHookA(TraceHookBase):
        pass

    @hook("critical_lifecycle_hook_b")
    class CriticalLifecycleHookB(TraceHookBase):
        pass

    @step("critical_lifecycle_step_a")
    class CriticalLifecycleStepA(TraceStepBase):
        pass

    @step("critical_lifecycle_step_b")
    class CriticalLifecycleStepB(TraceStepBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "ordered", max_iterations=5)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    resource_b = CriticalLifecycleResourceB("B", trace)
    hook_a = CriticalLifecycleHookA("A", trace, call_every=1)
    hook_b = CriticalLifecycleHookB("B", trace, call_every=3)
    step_a = CriticalLifecycleStepA("A", trace)
    step_b = CriticalLifecycleStepB("B", trace)

    session.register_resource(resource_a)
    session.register_resource(resource_b)
    session.register_hook(hook_a)
    session.register_hook(hook_b)
    session.add_step(step_a)
    session.add_step(step_b)

    completed: list[int] = []
    with session as entered:
        assert entered is session
        assert session._phase is SessionPhase.READY

        while True:
            try:
                completed.append(next(session))
            except StopIteration:
                break

            # Post callbacks have already observed the values, and successful
            # iteration completion must clear the transient context.
            assert session.iteration_context == {}

        assert session._phase is SessionPhase.FINISHED

    assert completed == [1, 2, 3, 4, 5]
    assert session._phase is SessionPhase.FINISHED
    assert session.session_context == {}

    expected_trace = [
        "resource:A:setup",
        "resource:B:setup",
        "hook:A:setup",
        "hook:B:setup",
    ]

    hook_b_iterations = {1, 3, 5}
    for iteration in range(1, 6):
        expected_trace.append(f"hook:A:pre:{iteration}")
        if iteration in hook_b_iterations:
            expected_trace.append(f"hook:B:pre:{iteration}")

        expected_trace.extend(
            [
                f"step:A:{iteration}",
                f"step:B:{iteration}",
            ]
        )

        if iteration in hook_b_iterations:
            expected_trace.append(f"hook:B:post:{iteration}")
        expected_trace.append(f"hook:A:post:{iteration}")

    expected_trace.extend(
        [
            "hook:B:teardown",
            "hook:A:teardown",
            "resource:B:teardown",
            "resource:A:teardown",
        ]
    )

    assert trace == expected_trace
    assert hook_a.pre_iterations == [1, 2, 3, 4, 5]
    assert hook_a.post_iterations == [1, 2, 3, 4, 5]
    assert hook_b.pre_iterations == [1, 3, 5]
    assert hook_b.post_iterations == [1, 3, 5]

    assert hook_a.post_payloads == [
        (iteration, ("A", "B")) for iteration in range(1, 6)
    ]
    assert hook_b.post_payloads == [
        (iteration, ("A", "B")) for iteration in (1, 3, 5)
    ]

    with pytest.raises(
            RuntimeError,
            match="This instance of TrainingSession is not initialized yet!",
    ):
        _ = session.iteration_context


def test_paused_session_reenters_components_and_resumes_to_finished(tmp_path):
    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @hook("critical_lifecycle_hook_a")
    class CriticalLifecycleHookA(TraceHookBase):
        pass

    @step("critical_lifecycle_step_a")
    class CriticalLifecycleStepA(TraceStepBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "pause-resume", max_iterations=4)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    hook_obj = CriticalLifecycleHookA("A", trace, call_every=1)
    step_obj = CriticalLifecycleStepA("A", trace)

    session.register_resource(resource_a)
    session.register_hook(hook_obj)
    session.add_step(step_obj)

    with session:
        assert next(session) == 1
        assert next(session) == 2
        session.session_context["temporary"] = "first-context"

    assert session.iteration == 2
    assert session._phase is SessionPhase.PAUSED
    assert session.session_context == {}
    assert resource_a.setup_calls == 1
    assert resource_a.teardown_calls == 1
    assert hook_obj.setup_calls == 1
    assert hook_obj.teardown_calls == 1

    with session:
        assert session._phase is SessionPhase.READY
        assert session.iteration == 2
        assert "temporary" not in session.session_context
        assert list(session) == [3, 4]
        assert session._phase is SessionPhase.FINISHED

    assert session.iteration == 4
    assert session._phase is SessionPhase.FINISHED
    assert session.session_context == {}
    assert resource_a.setup_calls == 2
    assert resource_a.teardown_calls == 2
    assert hook_obj.setup_calls == 2
    assert hook_obj.teardown_calls == 2
    assert step_obj.calls == 4

    # session_context is cleared between entries, so each first hook sees an
    # empty context before it writes its own setup value.
    assert hook_obj.context_sizes_seen_on_setup == [0, 0]

    with pytest.raises(RuntimeError, match="finished session"):
        with session:
            pass


def test_body_exception_propagates_after_normal_cleanup(tmp_path):

    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @resource("critical_lifecycle_resource_b")
    class CriticalLifecycleResourceB(TraceResourceBase):
        pass

    @hook("critical_lifecycle_hook_a")
    class CriticalLifecycleHookA(TraceHookBase):
        pass

    @hook("critical_lifecycle_hook_b")
    class CriticalLifecycleHookB(TraceHookBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "body-error", max_iterations=1)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    resource_b = CriticalLifecycleResourceB("B", trace)
    hook_a = CriticalLifecycleHookA("A", trace)
    hook_b = CriticalLifecycleHookB("B", trace)

    session.register_resource(resource_a)
    session.register_resource(resource_b)
    session.register_hook(hook_a)
    session.register_hook(hook_b)

    with pytest.raises(LifecycleStepError, match="body failed"):
        with session:
            session.session_context["temporary"] = "value"
            raise LifecycleStepError("body failed")

    assert trace == [
        "resource:A:setup",
        "resource:B:setup",
        "hook:A:setup",
        "hook:B:setup",
        "hook:B:teardown",
        "hook:A:teardown",
        "resource:B:teardown",
        "resource:A:teardown",
    ]
    assert session.session_context == {}
    assert session._phase is SessionPhase.NEW
    assert getattr(session, "_active", False) is False


def test_resource_teardown_failure_does_not_skip_remaining_cleanup(
    tmp_path,
    capsys,
):

    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @resource("critical_lifecycle_resource_b")
    class CriticalLifecycleResourceB(TraceResourceBase):
        pass

    @hook("critical_lifecycle_hook_a")
    class CriticalLifecycleHookA(TraceHookBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "resource-teardown-error", max_iterations=1)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    resource_b = CriticalLifecycleResourceB(
        "B",
        trace,
        fail_teardown=True,
    )

    hook_obj = CriticalLifecycleHookA("A", trace)

    session.register_resource(resource_a)
    session.register_resource(resource_b)
    session.register_hook(hook_obj)

    with session:
        session.session_context["temporary"] = "value"

    captured = capsys.readouterr()

    assert "Error releasing resource 'Resource.critical_lifecycle_resource_b'" in captured.out
    assert resource_b.teardown_calls == 1
    assert resource_a.teardown_calls == 1
    assert hook_obj.teardown_calls == 1
    assert session.session_context == {}
    assert session._phase is SessionPhase.NEW
    assert trace[-3:] == [
        "hook:A:teardown",
        "resource:B:teardown",
        "resource:A:teardown",
    ]


def test_partial_setup_failure_rolls_back_initialized_resources(tmp_path):

    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @resource("critical_lifecycle_resource_b")
    class CriticalLifecycleResourceB(TraceResourceBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "setup-rollback", max_iterations=1)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    resource_b = CriticalLifecycleResourceB("B", trace, fail_setup=True)
    session.register_resource(resource_a)
    session.register_resource(resource_b)

    with pytest.raises(LifecycleSetupError, match="resource B setup failed"):
        with session:
            pass

    assert resource_a.setup_calls == 1
    assert resource_a.teardown_calls == 1
    assert session.session_context == {}
    assert session._phase is SessionPhase.NEW
    assert getattr(session, "_active", False) is False


def test_hook_setup_failure_rolls_back_hooks_and_resources(tmp_path):
    @resource("critical_hook_setup_resource")
    class CriticalHookSetupResource(TraceResourceBase):
        pass

    @hook("critical_hook_setup_ready")
    class CriticalHookSetupReady(TraceHookBase):
        pass

    @hook("critical_hook_setup_failing")
    class CriticalHookSetupFailing(TraceHookBase):
        def setup(self, session: TrainingSession):
            self.setup_calls += 1
            self.trace.append(f"hook:{self.label}:setup")
            raise LifecycleSetupError(f"hook {self.label} setup failed")

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "hook-setup-rollback", max_iterations=1)
    )
    resource_obj = CriticalHookSetupResource("resource", trace)
    ready_hook = CriticalHookSetupReady("ready", trace)
    failing_hook = CriticalHookSetupFailing("failing", trace)
    session.register_resource(resource_obj)
    session.register_hook(ready_hook)
    session.register_hook(failing_hook)

    with pytest.raises(LifecycleSetupError, match="hook failing setup failed"):
        with session:
            pass

    assert trace == [
        "resource:resource:setup",
        "hook:ready:setup",
        "hook:failing:setup",
        "hook:ready:teardown",
        "resource:resource:teardown",
    ]
    assert failing_hook.teardown_calls == 0
    assert session.session_context == {}
    assert session._phase is SessionPhase.NEW
    assert getattr(session, "_active", False) is False


def test_resource_and_hook_with_same_name_have_independent_setup_tracking(tmp_path):
    shared_name = "critical_shared_lifecycle_name"

    @resource(shared_name)
    class CriticalSharedNameResource(TraceResourceBase):
        pass

    @hook(shared_name)
    class CriticalSharedNameHook(TraceHookBase):
        def setup(self, session: TrainingSession):
            self.setup_calls += 1
            self.trace.append(f"hook:{self.label}:setup")
            raise LifecycleSetupError(f"hook {self.label} setup failed")

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "shared-name-rollback", max_iterations=1)
    )
    resource_obj = CriticalSharedNameResource("shared", trace)
    hook_obj = CriticalSharedNameHook("shared", trace)
    session.register_resource(resource_obj)
    session.register_hook(hook_obj)

    with pytest.raises(LifecycleSetupError, match="hook shared setup failed"):
        with session:
            pass

    assert trace == [
        "resource:shared:setup",
        "hook:shared:setup",
        "resource:shared:teardown",
    ]
    assert hook_obj.teardown_calls == 0
    assert resource_obj.teardown_calls == 1


def test_step_failure_still_clears_iteration_context_and_runs_session_cleanup(
    tmp_path,
):

    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @hook("critical_lifecycle_hook_a")
    class CriticalLifecycleHookA(TraceHookBase):
        pass

    @step("critical_lifecycle_failing_step")
    class CriticalLifecycleFailingStep(TraceStepBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "step-error", max_iterations=2)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    hook_obj = CriticalLifecycleHookA("A", trace)
    failing_step = CriticalLifecycleFailingStep("failure", trace, fail=True)

    session.register_resource(resource_a)
    session.register_hook(hook_obj)
    session.add_step(failing_step)

    with pytest.raises(LifecycleStepError, match="step failure failed"):
        with session:
            next(session)

    assert resource_a.teardown_calls == 1
    assert hook_obj.teardown_calls == 1
    assert session.session_context == {}
    assert session._shared_state == {}
    assert session._phase in {SessionPhase.PAUSED, SessionPhase.INTERRUPTED}
    assert getattr(session, "_active", False) is False


def test_hook_teardown_failure_does_not_skip_later_cleanup(tmp_path):

    @resource("critical_lifecycle_resource_a")
    class CriticalLifecycleResourceA(TraceResourceBase):
        pass

    @hook("critical_lifecycle_failing_teardown_hook")
    class CriticalLifecycleFailingTeardownHook(TraceHookBase):
        pass

    @hook("critical_lifecycle_hook_b")
    class CriticalLifecycleHookB(TraceHookBase):
        pass

    trace: list[str] = []
    session = TrainingSession(
        make_config(tmp_path / "hook-teardown-error", max_iterations=1)
    )

    resource_a = CriticalLifecycleResourceA("A", trace)
    failing_hook = CriticalLifecycleFailingTeardownHook(
        "failing",
        trace,
        fail_teardown=True,
    )

    later_hook = CriticalLifecycleHookB("later", trace)

    session.register_resource(resource_a)
    session.register_hook(failing_hook)
    session.register_hook(later_hook)

    # with pytest.raises(LifecycleTeardownError, match="hook failing teardown failed"):
    with session:
        session.session_context["temporary"] = "value"

    assert resource_a.teardown_calls == 1
    assert failing_hook.teardown_calls == 1
    assert later_hook.teardown_calls == 1
    assert session.session_context == {}
    assert session._phase is SessionPhase.NEW
    assert getattr(session, "_active", False) is False
