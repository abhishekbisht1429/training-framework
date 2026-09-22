"""Tests separating a component instance's identity from its class's.

Registration writes `name` and `id` onto the class, so every instance of a
component would report the same pair. The session names the instance instead:
a component configured as `logger#2` is its own `logger#2`, while its class
stays `logger`.
"""

import pytest

from tests.test_utils import component_named, make_config
from training_framework.components import (
    Resource,
    requires_resource,
    resource,
)
from training_framework.session import TrainingSession


# -- identity on a constructed component ----------------------------------


def test_a_constructed_component_is_named_per_instance(tmp_path):
    session = TrainingSession(make_config(tmp_path / "identity"))
    logger = component_named(session, "logger")

    # Set on the instance, not inherited from the class.
    assert "name" in logger.__dict__
    assert logger.name == "logger"
    assert logger.id == "Hook.logger"


def test_an_instance_keeps_the_registered_name_of_its_class(tmp_path):
    session = TrainingSession(make_config(tmp_path / "implementation"))
    logger = component_named(session, "logger")

    assert logger.implementation_name == "logger"
    assert type(logger).name == "logger"


def test_naming_an_instance_does_not_rename_its_class(tmp_path):
    config = make_config(tmp_path / "class-untouched")
    config["logger"] = {"log_every": 1}
    config["logger#2"] = {"log_every": 5}
    session = TrainingSession(config)
    first = component_named(session, "logger")
    second = component_named(session, "logger#2")

    assert (first.name, first.id) == ("logger", "Hook.logger")
    assert (second.name, second.id) == ("logger#2", "Hook.logger#2")
    # The class is shared by every instance, so it must not have moved.
    assert type(second) is type(first)
    assert type(second).name == "logger"
    assert type(second).id == "Hook.logger"
    assert second.implementation_name == "logger"


# -- recorded wiring -------------------------------------------------------


def test_wiring_records_the_instance_that_was_handed_over(tmp_path):
    @resource("identity_dependency")
    class Dependency(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("identity_dependency")
    @resource("identity_consumer")
    class Consumer(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.dependency = self.get_dependency("identity_dependency")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    def linked(tmp_path, dependency_key):
        config = make_config(tmp_path)
        config[dependency_key] = {}
        config["identity_consumer"] = {}
        session = TrainingSession(config)
        return component_named(session, "identity_consumer").linked_components

    assert linked(tmp_path / "plain", "identity_dependency") == {
        "identity_dependency": "identity_dependency",
    }
    # The instance is what was recorded, not the class implementing it: the
    # sole instance answers the declared name and is recorded as itself.
    assert linked(tmp_path / "suffixed", "identity_dependency#2") == {
        "identity_dependency": "identity_dependency#2",
    }


# -- checkpoint state ------------------------------------------------------


def test_component_state_records_the_implementing_class(tmp_path):
    session = TrainingSession(make_config(tmp_path / "state-implementation"))
    state = session.get_state()

    assert state["components_state"]["logger"]["implementation"] == "logger"


def test_a_state_without_an_implementation_still_restores(tmp_path):
    # What every checkpoint written before instances were named looks like.
    session = TrainingSession(make_config(tmp_path / "legacy-state"))
    state = session.get_state()
    for component_info in state["components_state"].values():
        del component_info["implementation"]

    restored = TrainingSession.from_state(state)

    assert component_named(restored, "logger").name == "logger"


def test_a_state_disagreeing_with_its_own_name_is_rejected(tmp_path):
    session = TrainingSession(make_config(tmp_path / "state-disagrees"))
    state = session.get_state()
    state["components_state"]["logger"]["implementation"] = "checkpointer"

    with pytest.raises(ValueError, match="does not match its name"):
        TrainingSession.from_state(state)
