"""A component takes its prerequisites for itself, at any point in its life.

Prerequisites are resolved for the component that declared them -- honouring
its own `component_bindings` wiring -- and handed to it before `__init__`
runs, so construction, `setup`, a hook callback and a running step all see the
same instance. `Session.get_resource`, which is handed only a name and so can
only resolve session-wide, is deprecated in its favour.

Every scenario wires the consumer to `rt_dep#b` while a session-wide binding
says `rt_dep#a`. That is the shape in which a session-wide lookup is silently
wrong rather than loudly ambiguous.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import warnings

import pytest

from tests.test_utils import make_config
from training_framework.components import (
    ComponentDependencyError,
    LifecycleHook,
    Resource,
    Step,
    hook,
    requires_resource,
    resource,
    step,
)
from training_framework.session import TrainingSession


def declare_dependency():
    @resource("rt_dep")
    class Dependency(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.tag = (config or {}).get("tag")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    return Dependency


def declare_resource_consumer(seen):
    @requires_resource("rt_source")
    @resource("rt_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            seen.append(("setup", self.get_dependency("rt_source").tag))

        def teardown(self, session) -> None:
            seen.append(("teardown", self.get_dependency("rt_source").tag))

    return Consumer


def wired_config(tmp_path, *consumers, max_iterations=2):
    """Two instances, a session-wide binding to one, each consumer wired to
    the other."""
    config = make_config(tmp_path, max_iterations=max_iterations)
    config["session_config"]["show_execution_graph"] = False
    config["rt_dep#a"] = {"tag": "a"}
    config["rt_dep#b"] = {"tag": "b"}
    config["component_bindings"] = {
        "rt_source": "rt_dep#a",
        **{
            consumer: {"rt_source": "rt_dep#b"}
            for consumer in consumers
        },
    }
    for consumer in consumers:
        config[consumer] = {}
    return config


# -- the reported failure ---------------------------------------------------


def test_setup_is_given_the_instance_the_consumer_is_wired_to(tmp_path):
    declare_dependency()
    seen = []
    declare_resource_consumer(seen)
    session = TrainingSession(wired_config(tmp_path, "rt_consumer"))

    with session:
        pass

    assert seen == [("setup", "b"), ("teardown", "b")]


def test_every_lifecycle_stage_is_given_the_wired_instance(tmp_path):
    declare_dependency()
    seen = []

    @requires_resource("rt_source")
    @hook("rt_hook")
    class Hook(LifecycleHook):
        call_every = 1

        def pre_session(self, session) -> None:
            seen.append(("pre_session", self.get_dependency("rt_source").tag))

        def pre_iteration_callback(self, session) -> None:
            seen.append(("pre_iteration", self.get_dependency("rt_source").tag))

        def post_iteration_callback(self, session) -> None:
            seen.append(
                ("post_iteration", self.get_dependency("rt_source").tag)
            )

        def post_session(self, session) -> None:
            seen.append(("post_session", self.get_dependency("rt_source").tag))

    @requires_resource("rt_source")
    @step("rt_step")
    class Probe(Step):
        def run(self, session) -> None:
            seen.append(("run", self.get_dependency("rt_source").tag))

    session = TrainingSession(
        wired_config(tmp_path, "rt_hook", "rt_step", max_iterations=1),
    )

    with session:
        assert list(session) == [1]

    assert [stage for stage, _ in seen] == [
        "pre_session",
        "pre_iteration",
        "run",
        "post_iteration",
        "post_session",
    ]
    assert {tag for _, tag in seen} == {"b"}


def test_the_wiring_survives_a_checkpoint_round_trip(tmp_path):
    declare_dependency()
    seen = []
    declare_resource_consumer(seen)
    source = TrainingSession(wired_config(tmp_path, "rt_consumer"))

    restored = TrainingSession.from_state(source.get_state())
    with restored:
        pass

    assert seen == [("setup", "b"), ("teardown", "b")]


# -- what a component may ask for -------------------------------------------


def test_has_dependency_answers_for_declared_names_only(tmp_path):
    declare_dependency()
    seen = []

    @requires_resource("rt_source")
    @resource("rt_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            seen.append(self.has_dependency("rt_source"))
            seen.append(self.has_dependency("rt_undeclared"))

        def teardown(self, session) -> None:
            pass

    with TrainingSession(wired_config(tmp_path, "rt_consumer")):
        pass

    assert seen == [True, False]


def test_an_undeclared_prerequisite_is_rejected_with_the_fix(tmp_path):
    declare_dependency()

    @requires_resource("rt_source")
    @resource("rt_consumer")
    class Consumer(Resource):
        def setup(self, session) -> None:
            self.get_dependency("rt_undeclared")

        def teardown(self, session) -> None:
            pass

    session = TrainingSession(wired_config(tmp_path, "rt_consumer"))

    with pytest.raises(
            ComponentDependencyError,
            match=r"does not declare it\. Add "
                  r"@requires_resource\('rt_undeclared'\)",
    ):
        with session:
            pass


def test_a_component_built_outside_a_session_has_no_prerequisites():
    Consumer = declare_resource_consumer([])

    with pytest.raises(ComponentDependencyError, match="activate_component"):
        Consumer().get_dependency("rt_source")


# -- components that join a session after it is built ----------------------


def _single_dependency_session(tmp_path, *, tag):
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["rt_dep"] = {"tag": tag}
    config["component_bindings"] = {"rt_source": "rt_dep"}
    return TrainingSession(config)


def test_a_hand_registered_component_is_given_its_prerequisites(tmp_path):
    declare_dependency()
    seen = []
    Consumer = declare_resource_consumer(seen)
    session = _single_dependency_session(tmp_path, tag="configured")

    session.register_resource(Consumer())
    with session:
        pass

    assert seen == [("setup", "configured"), ("teardown", "configured")]


def test_replacing_a_prerequisite_reaches_consumers_built_earlier(tmp_path):
    Dependency = declare_dependency()
    seen = []
    Consumer = declare_resource_consumer(seen)
    session = _single_dependency_session(tmp_path, tag="original")
    session.register_resource(Consumer())

    session.unregister_resource("rt_dep")
    session.register_resource(Dependency({"tag": "replacement"}))
    with session:
        pass

    assert seen == [("setup", "replacement"), ("teardown", "replacement")]


def test_a_replaced_prerequisite_survives_a_checkpoint_round_trip(tmp_path):
    """A replacement is registered last, so the checkpoint stores it after
    the consumer that uses it; the restore must still build it first."""
    Dependency = declare_dependency()
    seen = []
    Consumer = declare_resource_consumer(seen)
    session = _single_dependency_session(tmp_path, tag="original")
    session.register_resource(Consumer())
    session.unregister_resource("rt_dep")
    session.register_resource(Dependency({"tag": "replacement"}))

    restored = TrainingSession.from_state(session.get_state())
    with restored:
        pass

    assert seen == [("setup", "replacement"), ("teardown", "replacement")]


# -- the deprecated session-wide lookup -------------------------------------


def test_session_get_resource_warns_and_still_resolves_session_wide(tmp_path):
    declare_dependency()
    declare_resource_consumer([])
    session = TrainingSession(wired_config(tmp_path, "rt_consumer"))

    with pytest.warns(FutureWarning, match=r"self\.get_dependency\('rt_source'\)"):
        resolved = session.get_resource("rt_source")
    with pytest.warns(FutureWarning, match=r"self\.has_dependency\('rt_source'\)"):
        assert session.has_resource("rt_source")

    # Unchanged while deprecated: the session-wide binding, not the
    # consumer's own wiring. This is why it is being retired.
    assert resolved.tag == "a"


def test_the_framework_does_not_trip_its_own_deprecation(tmp_path):
    declare_dependency()
    declare_resource_consumer([])
    session = TrainingSession(wired_config(tmp_path, "rt_consumer"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with session:
            assert list(session) == [1, 2]
