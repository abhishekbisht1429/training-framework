from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from training_framework.components import (
    Resource,
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)
from training_framework.components.builtin import (
    AnalysisDataManager,
    DataManager,
)
from training_framework.components.registry import component_registry
from training_framework.session import AnalysisSession, Session, TrainingSession


class _SourceModel(nn.Module, StatefulResource):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(1.0))

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return self.state_dict()

    def set_state(self, state):
        self.load_state_dict(state)


class _IndexDataset(Dataset, Resource):
    def __init__(self, config):
        self._size = int(config["size"])

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return torch.tensor(index)

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class _CollatingDataset(_IndexDataset):
    def collate_fn(self, batch):
        return [int(item) for item in batch]


class _BatchRecorder(Step):
    def __init__(self, config):
        self.batches = []

    def run(self, session):
        batch = next(session.get_resource("data_manager").data_iter)
        self.batches.append(
            batch if isinstance(batch, list) else batch.tolist()
        )


def _register_components():
    # conftest resets registries around every test, so register per test.
    resource("analysis_data_source_model")(_SourceModel)
    resource("analysis_index_dataset", session_type="analysis")(_IndexDataset)
    resource("analysis_collating_dataset", session_type="analysis")(
        _CollatingDataset
    )
    requires_resource("data_manager")(
        step("analysis_batch_recorder", session_type="analysis")(_BatchRecorder)
    )


def _session_config(root, max_iterations):
    return {
        "rng_seed": 3,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def _checkpoint(tmp_path):
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training", 1),
        "component_bindings": {"model": "analysis_data_source_model"},
        "analysis_data_source_model": {},
    })
    path = tmp_path / "training-session.pt"
    torch.save(source, path)
    return path


def _analysis_session(
        tmp_path,
        *,
        size=5,
        max_iterations=10,
        dataset="analysis_index_dataset",
        data_manager=None,
):
    _register_components()
    return AnalysisSession({
        "session_config": _session_config(
            tmp_path / "analysis", max_iterations
        ),
        "trained_model": {"model_checkpoint_path": str(_checkpoint(tmp_path))},
        "component_bindings": {"dataset": dataset},
        dataset: {"size": size},
        "data_manager": data_manager or {"batch_size": 2},
        "analysis_batch_recorder": {},
    })


def _recorder(session):
    [recorder] = [
        s for s in session.get_all_steps()
        if s.name == "analysis_batch_recorder"
    ]
    return recorder


def _run(session):
    with session:
        for _ in session:
            pass
    return _recorder(session)


def test_data_manager_resolves_per_session_type():
    assert component_registry("analysis")["data_manager"] is AnalysisDataManager
    assert component_registry("training")["data_manager"] is DataManager


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"batch_size": True}, ValueError, "batch_size"),
        ({"batch_size": 2, "num_workers": -1}, ValueError, "num_workers"),
        ({"batch_size": 2, "pin_memory": "yes"}, TypeError, "pin_memory"),
        ({"batch_size": 2, "drop_last": 1}, TypeError, "drop_last"),
    ],
)
def test_analysis_data_manager_validates_config(config, error, match):
    with pytest.raises(error, match=match):
        AnalysisDataManager(config)


def test_analysis_session_reads_dataset_once_in_order_without_ddp(tmp_path):
    session = _analysis_session(tmp_path)

    recorder = _run(session)

    assert recorder.batches == [[0, 1], [2, 3], [4]]
    # Exhausting the data ends the session before max_iterations.
    assert session.iteration == 3


def test_analysis_data_manager_honors_drop_last(tmp_path):
    session = _analysis_session(
        tmp_path,
        data_manager={"batch_size": 2, "drop_last": True},
    )

    assert _run(session).batches == [[0, 1], [2, 3]]


def test_max_iterations_still_bounds_analysis(tmp_path):
    session = _analysis_session(tmp_path, max_iterations=2)

    assert _run(session).batches == [[0, 1], [2, 3]]


def test_analysis_data_manager_uses_dataset_collate_function(tmp_path):
    session = _analysis_session(
        tmp_path,
        dataset="analysis_collating_dataset",
        data_manager={"batch_size": 3},
    )

    assert _run(session).batches == [[0, 1, 2], [3, 4]]


def test_analysis_data_manager_releases_loader_on_teardown(tmp_path):
    session = _analysis_session(tmp_path, max_iterations=1)

    with session:
        next(session)
        data_manager = session.get_resource("data_manager")
        assert data_manager.dataloader is not None

    assert data_manager.dataloader is None
    assert data_manager.data_iter is None


def test_analysis_session_with_data_manager_round_trips_through_state(tmp_path):
    session = _analysis_session(tmp_path)

    restored = Session.from_state(session.get_state())

    assert isinstance(restored.get_resource("data_manager"), AnalysisDataManager)
    assert _run(restored).batches == [[0, 1], [2, 3], [4]]
