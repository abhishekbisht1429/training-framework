"""Taking out an instance a consumer already holds is refused.

A consumer may keep a prerequisite wherever it likes, and nothing can reach
that reference afterwards. Replacing the instance would leave the consumer
holding the old one while `get_dependency` returned the new one; removing it
would leave the consumer using a component the session no longer sets up,
tears down or checkpoints. So both are refused once any consumer has been
handed the instance. Before that, replacement is safe and consumers are given
the new instance.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pytest

from tests.test_utils import make_config, resource_named
from training_framework.components import (
    ComponentDependencyError,
    Resource,
    requires_resource,
    resource,
)
from training_framework.session import TrainingSession


def declare(*, take_in):
    @resource("swap_dep")
    class Dependency(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.tag = (config or {}).get("tag")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("swap_dep")
    @resource("swap_consumer")
    class Consumer(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.seen = []
            if take_in == "init":
                self.held = self.get_dependency("swap_dep")

        def setup(self, session) -> None:
            self.seen.append(self.get_dependency("swap_dep").tag)

        def teardown(self, session) -> None:
            pass

    return Dependency


def build(tmp_path, *, take_in):
    Dependency = declare(take_in=take_in)
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config["swap_dep"] = {"tag": "original"}
    config["swap_consumer"] = {}
    return Dependency, TrainingSession(config)


# -- once a consumer holds it ----------------------------------------------


def test_removing_an_instance_taken_in_a_constructor_is_refused(tmp_path):
    _, session = build(tmp_path, take_in="init")

    with pytest.raises(
            ValueError,
            match=r"Cannot remove 'swap_dep'.*'swap_consumer' \(as 'swap_dep'\)",
    ):
        session.unregister_resource("swap_dep")

    assert resource_named(session, "swap_dep").tag == "original"


def test_overwriting_an_instance_taken_in_a_constructor_is_refused(tmp_path):
    Dependency, session = build(tmp_path, take_in="init")

    with pytest.raises(ValueError, match=r"Cannot replace 'swap_dep'"):
        session.register_resource(
            Dependency({"tag": "replacement"}),
            overwrite=True,
        )

    assert resource_named(session, "swap_dep").tag == "original"


def test_removing_an_instance_taken_during_a_run_is_refused(tmp_path):
    _, session = build(tmp_path, take_in="setup")
    with session:
        pass

    with pytest.raises(ValueError, match=r"Cannot remove 'swap_dep'"):
        session.unregister_resource("swap_dep")


# -- before anyone has taken it --------------------------------------------


def test_overwriting_an_untaken_instance_reaches_its_consumers(tmp_path):
    Dependency, session = build(tmp_path, take_in="setup")

    session.register_resource(Dependency({"tag": "replacement"}), overwrite=True)
    with session:
        pass

    assert resource_named(session, "swap_consumer").seen == ["replacement"]


def test_a_removed_prerequisite_is_not_handed_out(tmp_path):
    _, session = build(tmp_path, take_in="setup")
    consumer = resource_named(session, "swap_consumer")

    session.unregister_resource("swap_dep")

    with pytest.raises(ComponentDependencyError, match="no longer in its session"):
        consumer.get_dependency("swap_dep")


def test_registering_after_a_removal_gives_consumers_the_new_instance(tmp_path):
    Dependency, session = build(tmp_path, take_in="setup")
    session.unregister_resource("swap_dep")

    session.register_resource(Dependency({"tag": "replacement"}))
    with session:
        pass

    assert resource_named(session, "swap_consumer").seen == ["replacement"]
