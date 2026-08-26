from __future__ import annotations

import pytest

from training_framework.components import (
    Hook,
    IterationHook,
    LifecycleHook,
    Resource,
    SessionHook,
    Step,
    hook,
    resource,
    step,
    wraps,
)
from training_framework.session import TrainingSession


def _base_config(tmp_path, *, max_iterations=1):
    return {
        "base_config": {
            "rng_seed": 31,
            "sessions_dir": str(tmp_path),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
            "show-execution-graph": False,
        },
    }


def _remove_default_hooks(session: TrainingSession) -> None:
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")


def test_wraps_orders_every_shared_lifecycle_phase(tmp_path):
    events = []

    @hook("inner")
    class InnerHook(LifecycleHook):
        call_every = 1

        def __init__(self, config):
            pass

        def setup(self, session):
            events.append("inner.setup")

        def pre_iteration_callback(self, session):
            events.append("inner.pre")

        def post_iteration_callback(self, session):
            events.append("inner.post")

        def teardown(self, session):
            events.append("inner.teardown")

    @hook("outer")
    @wraps("inner")
    class OuterHook(LifecycleHook):
        call_every = 1

        def __init__(self, config):
            pass

        def setup(self, session):
            events.append("outer.setup")

        def pre_iteration_callback(self, session):
            events.append("outer.pre")

        def post_iteration_callback(self, session):
            events.append("outer.post")

        def teardown(self, session):
            events.append("outer.teardown")

    @step("body")
    class BodyStep(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            events.append("step.run")

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    # Register inner first to prove wrapping, rather than insertion order,
    # determines the outer-to-inner lifecycle.
    session.register_hook(InnerHook({}))
    session.register_hook(OuterHook({}))
    session.add_step(BodyStep({}))

    graph = session.execution_graph()
    assert graph.index("Hook.outer.setup()") < graph.index("Hook.inner.setup()")
    assert (
        graph.index("Hook.outer.pre_iteration_callback()")
        < graph.index("Hook.inner.pre_iteration_callback()")
    )
    assert (
        graph.index("Hook.inner.post_iteration_callback()")
        < graph.index("Hook.outer.post_iteration_callback()")
    )
    assert graph.index("Hook.inner.teardown()") < graph.index(
        "Hook.outer.teardown()"
    )
    assert "wraps: Hook.inner" in graph

    with session:
        assert next(session) == 1

    assert events == [
        "outer.setup",
        "inner.setup",
        "outer.pre",
        "inner.pre",
        "step.run",
        "inner.post",
        "outer.post",
        "inner.teardown",
        "outer.teardown",
    ]


def test_wraps_supports_chains_and_multiple_wrappers(tmp_path):
    def iteration_hook(component_name, *wrapped_names):
        class TestHook(IterationHook):
            call_every = 1

            def __init__(self, config):
                pass

            def pre_iteration_callback(self, session):
                pass

            def post_iteration_callback(self, session):
                pass

        component_class = TestHook
        for wrapped_name in wrapped_names:
            component_class = wraps(wrapped_name)(component_class)
        return hook(component_name)(component_class)

    outer = iteration_hook("outer", "middle")
    middle = iteration_hook("middle", "inner")
    inner = iteration_hook("inner")
    first = iteration_hook("first", "shared_inner")
    second = iteration_hook("second", "shared_inner")
    shared_inner = iteration_hook("shared_inner")

    chain_session = TrainingSession(_base_config(tmp_path / "chain"))
    _remove_default_hooks(chain_session)
    for hook_instance in (inner({}), middle({}), outer({})):
        chain_session.register_hook(hook_instance)

    chain_graph = chain_session.execution_graph()
    assert (
        chain_graph.index("Hook.outer.pre_iteration_callback()")
        < chain_graph.index("Hook.middle.pre_iteration_callback()")
        < chain_graph.index("Hook.inner.pre_iteration_callback()")
    )
    assert (
        chain_graph.index("Hook.inner.post_iteration_callback()")
        < chain_graph.index("Hook.middle.post_iteration_callback()")
        < chain_graph.index("Hook.outer.post_iteration_callback()")
    )

    multiple_session = TrainingSession(_base_config(tmp_path / "multiple"))
    _remove_default_hooks(multiple_session)
    for hook_instance in (first({}), second({}), shared_inner({})):
        multiple_session.register_hook(hook_instance)

    multiple_graph = multiple_session.execution_graph()
    assert (
        multiple_graph.index("Hook.first.pre_iteration_callback()")
        < multiple_graph.index("Hook.second.pre_iteration_callback()")
        < multiple_graph.index("Hook.shared_inner.pre_iteration_callback()")
    )
    assert (
        multiple_graph.index("Hook.shared_inner.post_iteration_callback()")
        < multiple_graph.index("Hook.second.post_iteration_callback()")
        < multiple_graph.index("Hook.first.post_iteration_callback()")
    )

def test_wraps_resolves_session_aliases(tmp_path):
    @hook("actual_inner")
    class ActualInner(IterationHook):
        def __init__(self, config):
            self.call_every = config["call_every"]

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @hook("alias_outer")
    @wraps("inner_role")
    class AliasOuter(IterationHook):
        def __init__(self, config):
            self.call_every = config["call_every"]

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    config = _base_config(tmp_path)
    config.update({
        "aliases": {"inner_role": "actual_inner"},
        "inner_role": {"call_every": 2},
        "alias_outer": {"call_every": 2},
    })
    session = TrainingSession(config)
    _remove_default_hooks(session)

    graph = session.execution_graph()
    assert "wraps: Hook.actual_inner" in graph
    assert (
        graph.index("Hook.alias_outer.pre_iteration_callback()")
        < graph.index("Hook.actual_inner.pre_iteration_callback()")
    )


def test_wraps_rejects_duplicate_declarations():
    class Wrapper(Hook):
        pass

    decorated = wraps("inner")(Wrapper)
    with pytest.raises(ValueError, match="already wraps"):
        wraps("inner")(decorated)


def test_wraps_metadata_on_subclass_does_not_mutate_parent():
    class Parent(Hook):
        pass

    parent = wraps("parent_target")(Parent)

    class Child(parent):
        pass

    child = wraps("child_target")(Child)

    assert parent.wrapped_hooks == ["parent_target"]
    assert child.wrapped_hooks == ["parent_target", "child_target"]


def test_wraps_rejects_missing_hook_target(tmp_path):
    @hook("missing_target_wrapper")
    @wraps("missing_target")
    class Wrapper(LifecycleHook):
        call_every = 1

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_hook(Wrapper())

    with pytest.raises(RuntimeError, match="missing_target.*not registered as a Hook"):
        session.execution_graph()


def test_wraps_rejects_non_hook_target(tmp_path):
    @resource("not_a_hook")
    class NotAHook(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @hook("wrong_target_wrapper")
    @wraps("not_a_hook")
    class Wrapper(LifecycleHook):
        call_every = 1

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_resource(NotAHook())
    session.register_hook(Wrapper())

    with pytest.raises(RuntimeError, match="not_a_hook.*not registered as a Hook"):
        session.execution_graph()


def test_wraps_rejects_unconfigured_hook_target(tmp_path):
    @hook("configured_wrapper")
    @wraps("registered_only_target")
    class Wrapper(IterationHook):
        call_every = 1

        def __init__(self, config):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @hook("registered_only_target")
    class Target(IterationHook):
        call_every = 1

        def __init__(self, config):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_hook(Wrapper({}))

    with pytest.raises(RuntimeError, match="not configured in this session"):
        session.execution_graph()


def test_wraps_rejects_hooks_without_a_shared_lifecycle_phase(tmp_path):
    @hook("session_wrapper")
    @wraps("iteration_target")
    class Wrapper(SessionHook):
        def __init__(self, config):
            pass

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @hook("iteration_target")
    class Target(IterationHook):
        call_every = 1

        def __init__(self, config):
            pass

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_hook(Wrapper({}))
    session.register_hook(Target({}))

    with pytest.raises(RuntimeError, match="do not share a lifecycle phase"):
        session.execution_graph()


def test_wraps_allows_wrapper_to_run_less_often_than_wrapped_hook(tmp_path):
    events = []

    @hook("cadence_wrapper")
    @wraps("cadence_target")
    class Wrapper(IterationHook):
        call_every = 4

        def pre_iteration_callback(self, session):
            events.append(("wrapper_pre", session.iteration))

        def post_iteration_callback(self, session):
            events.append(("wrapper_post", session.iteration))

    @hook("cadence_target")
    class Target(IterationHook):
        call_every = 2

        def pre_iteration_callback(self, session):
            events.append(("target_pre", session.iteration))

        def post_iteration_callback(self, session):
            events.append(("target_post", session.iteration))

    session = TrainingSession(_base_config(tmp_path, max_iterations=8))
    _remove_default_hooks(session)
    session.register_hook(Wrapper())
    session.register_hook(Target())

    with session:
        list(session)

    assert [
        iteration for event, iteration in events if event == "wrapper_pre"
    ] == [1, 4, 8]
    assert [
        iteration for event, iteration in events if event == "target_pre"
    ] == [1, 2, 4, 6, 8]
    for iteration in (1, 4, 8):
        iteration_events = [
            event for event, event_iteration in events
            if event_iteration == iteration
        ]
        assert iteration_events == [
            "wrapper_pre",
            "target_pre",
            "target_post",
            "wrapper_post",
        ]


@pytest.mark.parametrize(
    ("wrapper_cadence", "target_cadence"),
    (
        pytest.param(1, 2, id="wrapper-runs-more-often"),
        pytest.param(3, 2, id="wrapper-schedule-is-not-contained"),
        pytest.param(0, 2, id="wrapper-cadence-is-zero"),
        pytest.param(2, 0, id="wrapped-cadence-is-zero"),
        pytest.param(-2, 2, id="wrapper-cadence-is-negative"),
        pytest.param(2, -2, id="wrapped-cadence-is-negative"),
        pytest.param(True, 2, id="wrapper-cadence-is-boolean"),
        pytest.param(2, False, id="wrapped-cadence-is-boolean"),
        pytest.param(4.0, 2, id="wrapper-cadence-is-not-an-integer"),
        pytest.param(4, 2.0, id="wrapped-cadence-is-not-an-integer"),
    ),
)
def test_wraps_rejects_incompatible_iteration_cadences(
        tmp_path,
        wrapper_cadence,
        target_cadence,
):
    @hook("cadence_wrapper")
    @wraps("cadence_target")
    class Wrapper(IterationHook):
        def __init__(self, config):
            self.call_every = config["call_every"]

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @hook("cadence_target")
    class Target(IterationHook):
        def __init__(self, config):
            self.call_every = config["call_every"]

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_hook(Wrapper({"call_every": wrapper_cadence}))
    session.register_hook(Target({"call_every": target_cadence}))

    with pytest.raises(RuntimeError, match="positive multiple"):
        session.execution_graph()


def test_wraps_rejects_self_wrapping(tmp_path):
    @hook("self_wrapper")
    @wraps("self_wrapper")
    class SelfWrapper(IterationHook):
        call_every = 1

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_hook(SelfWrapper())

    with pytest.raises(RuntimeError, match="cannot wrap itself"):
        session.execution_graph()


def test_wraps_rejects_cycles(tmp_path):
    @hook("cycle_a")
    @wraps("cycle_b")
    class CycleA(IterationHook):
        call_every = 1

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    @hook("cycle_b")
    @wraps("cycle_a")
    class CycleB(IterationHook):
        call_every = 1

        def pre_iteration_callback(self, session):
            pass

        def post_iteration_callback(self, session):
            pass

    session = TrainingSession(_base_config(tmp_path))
    _remove_default_hooks(session)
    session.register_hook(CycleA())
    session.register_hook(CycleB())

    with pytest.raises(RuntimeError, match="Cyclic dependency"):
        session.execution_graph()
