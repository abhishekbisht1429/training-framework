"""Built-ins that write somewhere named after themselves.

A component that derives a path from its own identity collides with its
sibling once it can be configured twice. The single-instance path must not
move, though, or every existing session would write somewhere new.
"""

import pytest

from tests.test_utils import make_config
from training_framework.components import Resource, resource
from training_framework.session import TrainingSession


def test_a_sole_instance_has_no_suffix(tmp_path):
    session = TrainingSession(make_config(tmp_path / "no-suffix"))

    assert session._components.components["checkpointer"].instance_suffix is None


def test_an_instance_reports_its_own_suffix(tmp_path):
    config = make_config(tmp_path / "suffix")
    config["checkpointer#nightly"] = {"checkpoint_every": 5}

    session = TrainingSession(config)

    component = session._components.components["checkpointer#nightly"]
    assert component.instance_suffix == "nightly"


def test_a_component_built_outside_a_session_has_no_suffix():
    @resource("safety_loose")
    class Loose(Resource):
        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    # Never named by a session, so it falls back to its class's name.
    assert Loose().instance_suffix is None


def test_the_only_checkpointer_keeps_the_plain_directory(tmp_path):
    session = TrainingSession(make_config(tmp_path / "plain-dir"))
    checkpointer = session._components.components["checkpointer"]

    checkpointer.pre_session(session)

    assert checkpointer._checkpoints_dir.endswith("/checkpoints")


def test_a_second_checkpointer_writes_to_its_own_directory(tmp_path):
    config = make_config(tmp_path / "own-dir")
    config["checkpointer"] = {"checkpoint_every": 2}
    config["checkpointer#nightly"] = {"checkpoint_every": 10}
    session = TrainingSession(config)

    components = session._components.components
    components["checkpointer"].pre_session(session)
    components["checkpointer#nightly"].pre_session(session)

    assert components["checkpointer"]._checkpoints_dir.endswith("/checkpoints")
    assert components["checkpointer#nightly"]._checkpoints_dir.endswith(
        "/checkpoints_nightly"
    )


def test_an_explicit_checkpoints_dir_still_wins(tmp_path):
    config = make_config(tmp_path / "explicit-dir")
    config["checkpointer#nightly"] = {
        "checkpoint_every": 10,
        "checkpoints_dir": str(tmp_path / "somewhere-else"),
    }
    session = TrainingSession(config)

    checkpointer = session._components.components["checkpointer#nightly"]
    checkpointer.pre_session(session)

    assert checkpointer._checkpoints_dir == str(tmp_path / "somewhere-else")
