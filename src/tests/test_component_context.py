from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from training_framework.components import LifecycleHook, Resource
from training_framework.util import requires_context

if TYPE_CHECKING:
    from training_framework.session import Session


def _assert_requires_context(instance) -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            f"This instance of {type(instance).__name__} "
            "is not initialized yet!"
        ),
    ):
        instance.guarded()


def test_resource_and_lifecycle_hook_context_is_automatic():
    class GuardedResource(Resource):
        def __init__(self):
            self.teardown_value = None

        @requires_context
        def guarded(self):
            return "resource"

        def setup(self, session: "Session"):
            pass

        def teardown(self, session: "Session"):
            self.teardown_value = self.guarded()

    class GuardedHook(LifecycleHook):
        call_every = 1

        def __init__(self):
            self.callback_values = []
            self.teardown_value = None

        @requires_context
        def guarded(self):
            return "hook"

        def pre_session(self, session: "Session"):
            pass

        def post_session(self, session: "Session"):
            self.teardown_value = self.guarded()

        def pre_iteration_callback(self, session: "Session"):
            self.callback_values.append(self.guarded())

        def post_iteration_callback(self, session: "Session"):
            self.callback_values.append(self.guarded())

    resource = GuardedResource()
    hook = GuardedHook()
    session_obj = cast("Session", object())

    _assert_requires_context(resource)
    _assert_requires_context(hook)

    resource.setup(session_obj)
    hook.pre_session(session_obj)

    assert resource.guarded() == "resource"
    assert hook.guarded() == "hook"
    hook.pre_iteration_callback(session_obj)
    hook.post_iteration_callback(session_obj)

    hook.post_session(session_obj)
    resource.teardown(session_obj)

    assert hook.callback_values == ["hook", "hook"]
    assert hook.teardown_value == "hook"
    assert resource.teardown_value == "resource"
    _assert_requires_context(resource)
    _assert_requires_context(hook)


def test_context_entry_waits_for_outermost_setup_to_finish():
    class ParentResource(Resource):
        def __init__(self):
            self.active_during_child_setup = None

        @requires_context
        def guarded(self):
            return "ready"

        def setup(self, session: "Session"):
            pass

        def teardown(self, session: "Session"):
            pass

    class FailingChildResource(ParentResource):
        def setup(self, session: "Session"):
            super().setup(session)
            self.active_during_child_setup = getattr(self, "_active", False)
            raise RuntimeError("child setup failed")

    resource = FailingChildResource()
    session_obj = cast("Session", object())

    with pytest.raises(RuntimeError, match="child setup failed"):
        resource.setup(session_obj)

    assert resource.active_during_child_setup is False
    _assert_requires_context(resource)


def test_context_exit_waits_for_outermost_teardown_and_clears_on_failure():
    class ParentResource(Resource):
        @requires_context
        def guarded(self):
            return "ready"

        def setup(self, session: "Session"):
            pass

        def teardown(self, session: "Session"):
            pass

    class FailingChildResource(ParentResource):
        def __init__(self):
            self.value_after_parent_teardown = None

        def teardown(self, session: "Session"):
            super().teardown(session)
            self.value_after_parent_teardown = self.guarded()
            raise RuntimeError("child teardown failed")

    resource = FailingChildResource()
    session_obj = cast("Session", object())
    resource.setup(session_obj)

    with pytest.raises(RuntimeError, match="child teardown failed"):
        resource.teardown(session_obj)

    assert resource.value_after_parent_teardown == "ready"
    _assert_requires_context(resource)
