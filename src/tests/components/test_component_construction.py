"""The session constructs a component the way Python would.

Prerequisites have to reach an instance before its `__init__` runs. They used
to be written between a bare `cls.__new__(cls)` and a direct `__init__` call,
which gave a custom `__new__(cls, config)` no arguments and skipped the
metaclass `__call__` altogether. Construction now goes through the class call.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

from tests.test_utils import make_config, resource_named
from training_framework.components import Resource, requires_resource, resource
from training_framework.components.base import ComponentMeta
from training_framework.session import TrainingSession


def declare_dependency():
    @resource("build_dep")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass


def build_session(tmp_path, consumer_config):
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["build_dep"] = {}
    config["build_consumer"] = consumer_config
    return TrainingSession(config)


def test_a_custom_new_receives_the_constructor_arguments(tmp_path):
    declare_dependency()

    @requires_resource("build_dep")
    @resource("build_consumer")
    class Consumer(Resource):
        def __new__(cls, config):
            instance = super().__new__(cls)
            instance.seen_by_new = dict(config)
            return instance

        def __init__(self, config):
            super().__init__()
            self.dependency_in_init = self.get_dependency("build_dep")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    session = build_session(tmp_path, {"size": 3})

    consumer = resource_named(session, "build_consumer")
    assert consumer.seen_by_new == {"size": 3}
    assert consumer.dependency_in_init is resource_named(session, "build_dep")


def test_a_metaclass_call_override_runs_and_the_prerequisites_arrive(tmp_path):
    declare_dependency()
    calls = []

    class RecordingMeta(ComponentMeta):
        def __call__(cls, *args, **kwargs):
            calls.append(cls.__name__)
            return super().__call__(*args, **kwargs)

    @requires_resource("build_dep")
    @resource("build_consumer")
    class Consumer(Resource, metaclass=RecordingMeta):
        def __init__(self, config=None):
            super().__init__(config)
            self.dependency_in_init = self.get_dependency("build_dep")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    session = build_session(tmp_path, {})

    assert calls == ["Consumer"]
    consumer = resource_named(session, "build_consumer")
    assert consumer.dependency_in_init is resource_named(session, "build_dep")


def test_the_private_keyword_never_reaches_the_saved_constructor_arguments(
        tmp_path,
):
    """A checkpoint replays the constructor arguments, so they must be the
    configuration alone; the round trip is what would break otherwise."""
    declare_dependency()

    @requires_resource("build_dep")
    @resource("build_consumer")
    class Consumer(Resource):
        def __init__(self, config):
            super().__init__()
            self.config = dict(config)
            self.get_dependency("build_dep")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    session = build_session(tmp_path, {"size": 5})

    restored = TrainingSession.from_state(session.get_state())

    assert resource_named(restored, "build_consumer").config == {"size": 5}
