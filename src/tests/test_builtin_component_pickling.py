from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from training_framework.components import StatefulResource, resource
from training_framework.components.builtin import (
    AnalysisLogger,
    Checkpointer,
    Logger,
    Tensorboard,
    TrainedModel,
)
from training_framework.components.builtin import observability
from training_framework.session import TrainingSession
from training_framework.session.io import ConfigDumper


class _PickleRoundTripModel(nn.Module, StatefulResource):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(
            torch.tensor(float(config["initial_weight"]))
        )

    def forward(self, value):
        return self.weight * value

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return {"weight": self.weight.detach().clone()}

    def set_state(self, state):
        with torch.no_grad():
            self.weight.copy_(state["weight"])


def _session_config(tmp_path):
    return {
        "rng_seed": 17,
        "sessions_dir": str(tmp_path),
        "max_iterations": 1,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def test_pickled_checkpointer_creates_a_loadable_checkpoint(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpointer = pickle.loads(pickle.dumps(Checkpointer({
        "checkpoint_every": 1,
        "checkpoints_dir": str(checkpoints_dir),
    })))
    session = SimpleNamespace(
        iteration=2,
        session_config=SimpleNamespace(
            max_iterations=3,
            session_dir=str(tmp_path),
        ),
    )

    checkpointer.pre_session(session)
    checkpointer.post_iteration_callback(session)
    checkpointer.post_session(session)

    checkpoint_paths = list(checkpoints_dir.iterdir())
    assert len(checkpoint_paths) == 1
    loaded = Checkpointer.load_checkpoint(checkpoint_paths[0])
    assert loaded.iteration == 2


@pytest.mark.parametrize(
    ("component_type", "expected_line"),
    [
        (Logger, "Iteration 2/5\n"),
        (AnalysisLogger, "Analysis iteration 2/5\n"),
    ],
)
def test_pickled_logger_writes_progress(
        tmp_path,
        component_type,
        expected_line,
):
    log_path = tmp_path / f"{component_type.__name__}.log"
    restored = pickle.loads(pickle.dumps(component_type({
        "log_every": 1,
        "log_file": str(log_path),
    })))
    session = SimpleNamespace(
        iteration=2,
        session_config=SimpleNamespace(max_iterations=5),
    )

    restored.pre_session(session)
    restored.pre_iteration_callback(session)
    restored.post_session(session)

    assert log_path.read_text() == expected_line


def test_pickled_tensorboard_can_start_and_release_runtime_handles(
        tmp_path,
        monkeypatch,
):
    processes = []
    writers = []

    class FakeProcess:
        def __init__(self, args):
            self.args = args
            self.terminated = False
            processes.append(self)

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    class FakeSummaryWriter:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.closed = False
            writers.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(observability.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(observability, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(observability.time, "sleep", lambda seconds: None)

    restored = pickle.loads(pickle.dumps(Tensorboard({
        "host": "127.0.0.1",
        "port": 6007,
    })))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session = SimpleNamespace(
        session_config=SimpleNamespace(session_dir=str(session_dir))
    )

    restored.setup(session)
    assert processes[0].args == [
        "tensorboard",
        "--logdir",
        str(session_dir),
        "--host",
        "127.0.0.1",
        "--port",
        "6007",
    ]
    assert restored.summary_writer is writers[0]

    restored.teardown(session)
    assert writers[0].closed
    assert processes[0].terminated
    assert restored.summary_writer is None


def test_tensorboard_rolls_back_process_when_writer_creation_fails(
    tmp_path,
    monkeypatch,
):
    processes = []

    class FakeProcess:
        def __init__(self, args):
            self.terminated = False
            processes.append(self)

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    def fail_writer_creation(*args, **kwargs):
        raise RuntimeError("writer creation failed")

    monkeypatch.setattr(observability.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        observability,
        "SummaryWriter",
        fail_writer_creation,
    )

    component = Tensorboard({
        "host": "127.0.0.1",
        "port": 6008,
    })
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session = SimpleNamespace(
        session_config=SimpleNamespace(session_dir=str(session_dir))
    )

    with pytest.raises(RuntimeError, match="writer creation failed"):
        component.setup(session)

    component.rollback_setup(session)

    assert processes[0].terminated
    assert component.summary_writer is None


def test_pickled_trained_model_loads_an_evaluation_model(tmp_path):
    resource("pickle_round_trip_model")(_PickleRoundTripModel)
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "source"),
        "aliases": {"model": "pickle_round_trip_model"},
        "model": {"initial_weight": 2.0},
    })
    checkpoint_path = tmp_path / "training-session.pt"
    torch.save(source, checkpoint_path)

    restored = pickle.loads(pickle.dumps(TrainedModel({
        "model_checkpoint_path": str(checkpoint_path),
    })))
    analysis_session = SimpleNamespace(device=torch.device("cpu"))

    restored.setup(analysis_session)
    assert not restored.model.training
    torch.testing.assert_close(
        restored.model(torch.tensor(3.0)),
        torch.tensor(6.0),
    )

    restored.teardown(analysis_session)
    with pytest.raises(RuntimeError, match="not initialized yet"):
        _ = restored.model


def test_pickled_config_dumper_writes_the_session_configuration(tmp_path):
    restored = pickle.loads(pickle.dumps(ConfigDumper()))
    session = SimpleNamespace(
        session_config=SimpleNamespace(session_dir=str(tmp_path)),
        full_config={"session_type": "training", "model": {"width": 8}},
    )

    restored.pre_session(session)
    restored.post_session(session)

    with open(tmp_path / "config.yaml") as config_file:
        assert yaml.safe_load(config_file) == session.full_config
