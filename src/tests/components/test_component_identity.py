"""Tests separating a component instance's identity from its class's.

Registration writes `name` and `id` onto the class, so every instance of a
component would report the same pair. The session names the instance instead.
A session still holds one instance of each component, so every name here has
no instance suffix and the two identities coincide -- these tests pin the
split itself, which later phases rely on.
"""

import pytest

from tests.test_utils import component_named, make_config
from training_framework.components import (
    Resource,
    requires_resource,
    resource,
)
from training_framework.components.naming import (
    format_instance_name,
    is_instance_name,
    parse_instance_name,
)
from training_framework.session import TrainingSession


# -- name parsing ---------------------------------------------------------


def test_a_plain_name_parses_as_a_component_without_a_suffix():
    assert parse_instance_name("logger") == ("logger", None)


def test_an_instance_name_parses_into_component_and_suffix():
    assert parse_instance_name("logger#2") == ("logger", "2")


def test_a_suffix_may_name_what_the_instance_is_for():
    assert parse_instance_name("data_manager#validation") == (
        "data_manager",
        "validation",
    )


def test_an_empty_suffix_is_rejected():
    with pytest.raises(ValueError, match="invalid instance suffix"):
        parse_instance_name("logger#")


def test_a_suffix_with_punctuation_is_rejected():
    with pytest.raises(ValueError, match="invalid instance suffix"):
        parse_instance_name("logger#a.b")


def test_a_name_without_a_component_is_rejected():
    with pytest.raises(ValueError, match="no component name before"):
        parse_instance_name("#2")


def test_an_empty_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        parse_instance_name("")


def test_a_non_string_name_is_rejected():
    with pytest.raises(TypeError, match="must be a string"):
        parse_instance_name(None)


def test_formatting_round_trips_a_parsed_name():
    for name in ("logger", "logger#2", "data_manager#validation"):
        assert format_instance_name(*parse_instance_name(name)) == name


def test_only_a_suffixed_name_is_an_instance_name():
    assert is_instance_name("logger#2")
    assert not is_instance_name("logger")


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
    session = TrainingSession(make_config(tmp_path / "class-untouched"))
    logger = component_named(session, "logger")

    session._components._stamp_identity(logger, "logger#2")

    assert logger.name == "logger#2"
    assert logger.id == "Hook.logger#2"
    # The class is shared by every instance, so it must not have moved.
    assert type(logger).name == "logger"
    assert type(logger).id == "Hook.logger"
    assert logger.implementation_name == "logger"


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

    config = make_config(tmp_path / "wiring")
    config["identity_consumer"] = {}
    session = TrainingSession(config)
    consumer = component_named(session, "identity_consumer")

    assert consumer.linked_components == {
        "identity_dependency": "identity_dependency",
    }

    # The instance is what was recorded, not the class implementing it, so
    # renaming that instance moves the record with it.
    session._components._stamp_identity(
        consumer.dependency,
        "identity_dependency#2",
    )
    assert consumer.linked_components == {
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
