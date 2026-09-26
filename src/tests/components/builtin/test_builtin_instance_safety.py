"""Built-ins that write somewhere named after themselves.

A component that derives a path from its own identity collides with its
sibling once it can be configured twice. The single-instance path must not
move, though, or every existing session would write somewhere new.
"""

from pathlib import Path


from tests.test_utils import component_named, make_config
from training_framework.components import Resource, resource
from training_framework.session import TrainingSession


def test_a_sole_instance_has_no_suffix(tmp_path):
    session = TrainingSession(make_config(tmp_path / "no-suffix"))

    assert component_named(session, "checkpointer").instance_suffix is None


def test_an_instance_reports_its_own_suffix(tmp_path):
    config = make_config(tmp_path / "suffix")
    config["checkpointer#nightly"] = {"checkpoint_every": 5}

    session = TrainingSession(config)

    component = component_named(session, "checkpointer#nightly")
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


def checkpoints_written(session, directory):
    """Run `session` for its iterations and list what `directory` received."""
    with session:
        list(session)
    path = Path(directory)
    if not path.is_absolute():
        path = Path(session.session_config.session_dir) / path
    return sorted(entry.name for entry in path.iterdir()) if path.is_dir() else []


def test_the_only_checkpointer_keeps_the_plain_directory(tmp_path):
    session = TrainingSession(make_config(tmp_path / "plain-dir", max_iterations=1))

    assert checkpoints_written(session, "checkpoints")


def test_a_second_checkpointer_writes_to_its_own_directory(tmp_path):
    config = make_config(tmp_path / "own-dir", max_iterations=1)
    config["checkpointer"] = {"checkpoint_every": 2}
    config["checkpointer#nightly"] = {"checkpoint_every": 10}
    session = TrainingSession(config)

    with session:
        list(session)

    session_dir = Path(session.session_config.session_dir)
    assert len(list((session_dir / "checkpoints").iterdir())) == 1
    assert len(list((session_dir / "checkpoints_nightly").iterdir())) == 1


def test_an_explicit_checkpoints_dir_still_wins(tmp_path):
    config = make_config(tmp_path / "explicit-dir", max_iterations=1)
    config["checkpointer#nightly"] = {
        "checkpoint_every": 10,
        "checkpoints_dir": str(tmp_path / "somewhere-else"),
    }
    session = TrainingSession(config)

    assert checkpoints_written(session, tmp_path / "somewhere-else")
    assert not (Path(session.session_config.session_dir) / "checkpoints_nightly").exists()
