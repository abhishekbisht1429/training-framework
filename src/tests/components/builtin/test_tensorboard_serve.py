from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from training_framework.components.builtin import Tensorboard
from training_framework.components.builtin import observability


def _session(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    return SimpleNamespace(
        session_config=SimpleNamespace(session_dir=str(session_dir))
    ), session_dir


def _forbid_server(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("serve: false must not start a server")

    monkeypatch.setattr(observability.subprocess, "Popen", fail_if_called)
    monkeypatch.setattr(observability.time, "sleep", fail_if_called)


def _event_files(directory):
    return list(directory.glob("events.out.tfevents.*"))


def test_serve_false_writes_events_without_starting_a_server(
        tmp_path,
        monkeypatch,
):
    _forbid_server(monkeypatch)
    session, session_dir = _session(tmp_path)
    component = Tensorboard({"serve": False})

    component.setup(session)
    component.summary_writer.add_scalar("train/loss", 1.0, 0)
    component.teardown(session)

    assert _event_files(session_dir / "Tensorboard_tensorboard")
    assert component.summary_writer is None


def test_pickled_serve_false_component_restores_and_writes_events(
        tmp_path,
        monkeypatch,
):
    _forbid_server(monkeypatch)
    session, session_dir = _session(tmp_path)
    restored = pickle.loads(pickle.dumps(Tensorboard({"serve": False})))

    restored.setup(session)
    restored.summary_writer.add_scalar("train/loss", 1.0, 0)
    restored.teardown(session)

    assert _event_files(session_dir / "Tensorboard_tensorboard")


@pytest.mark.parametrize(
    "server_keys",
    [
        {"host": "127.0.0.1"},
        {"port": 6006},
        {"logdir": "./runs"},
        {"host": "127.0.0.1", "port": 6006},
    ],
)
def test_server_keys_are_refused_without_a_server(server_keys):
    with pytest.raises(ValueError, match="serve: false") as error:
        Tensorboard({"serve": False, **server_keys})

    for key in server_keys:
        assert key in str(error.value)


@pytest.mark.parametrize("serve", ["yes", 1, 0, None])
def test_serve_must_be_a_boolean(serve):
    with pytest.raises(ValueError, match="serve must be a boolean"):
        Tensorboard({"serve": serve, "host": "127.0.0.1", "port": 6006})


def test_explicit_serve_true_behaves_like_the_default(tmp_path, monkeypatch):
    launches = []

    class FakeProcess:
        def __init__(self, args):
            launches.append(args)

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(observability.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(observability.time, "sleep", lambda seconds: None)
    server = {"host": "127.0.0.1", "port": 6006}

    writer_dirs = []
    for config in (dict(server), {"serve": True, **server}):
        session, session_dir = _session(tmp_path / str(len(writer_dirs)))
        component = Tensorboard(config)
        component.setup(session)
        writer_dirs.append(
            component.summary_writer.log_dir.replace(str(session_dir), "")
        )
        component.teardown(session)

    # The logdir argument differs only because each run has its own session.
    without_logdir = [args[:2] + args[3:] for args in launches]
    assert without_logdir[0] == without_logdir[1]
    assert writer_dirs[0] == writer_dirs[1]
