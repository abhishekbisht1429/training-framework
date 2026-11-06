"""A session's length given in epochs, resolved to `max_iterations`."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from training_framework.components import (
    EpochSource,
    Resource,
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)
from training_framework.hpc.services import resume as resuming
from training_framework.session import AnalysisSession, TrainingSession
from training_framework.session.checkpoint_format import (
    read_manifest,
    write_checkpoint,
)
from tests.test_utils import resource_named


class _Context(Resource):
    """A process-group-free stand-in for `ddp`."""

    def __init__(self, config):
        self.rank = int(config.get("rank", 0))
        self.world_size = int(config["world_size"])

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


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


class _BatchDrawer(Step):
    def __init__(self, config):
        self.batches = []

    def run(self, session):
        batch = next(self.get_dependency("data_manager").data_iter)
        self.batches.append(batch.tolist())


class _AnalysisIndexDataset(_IndexDataset):
    pass


class _AnalysisBatchDrawer(_BatchDrawer):
    pass


class _PlainDataManager(Resource):
    """A data manager that does not say how long an epoch is."""

    def __init__(self, config):
        pass

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class _CountingDataManager(_PlainDataManager, EpochSource):
    def iterations_for_epochs(self, epochs):
        return 7 * epochs


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


def _register_components():
    # conftest resets registries around every test, so register per test.
    resource("run_length_context")(_Context)
    resource("run_length_dataset")(_IndexDataset)
    resource("run_length_plain_data_manager")(_PlainDataManager)
    resource("run_length_counting_data_manager")(_CountingDataManager)
    resource("run_length_source_model")(_SourceModel)
    resource("run_length_analysis_dataset", session_type="analysis")(
        _AnalysisIndexDataset
    )
    requires_resource("data_manager")(step("run_length_drawer")(_BatchDrawer))
    requires_resource("data_manager")(
        step("run_length_analysis_drawer", session_type="analysis")(
            _AnalysisBatchDrawer
        )
    )


def _session_settings(root, **length):
    return {
        "rng_seed": 5,
        "sessions_dir": str(root),
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
        **length,
    }


def _training_config(
        tmp_path,
        *,
        size=10,
        batch_size=4,
        world_size=1,
        **length,
):
    return {
        "session_config": _session_settings(tmp_path / "training", **length),
        "component_bindings": {
            "ddp": "run_length_context",
            "dataset": "run_length_dataset",
        },
        "run_length_context": {"world_size": world_size},
        "run_length_dataset": {"size": size},
        "data_manager": {
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": False,
        },
        "run_length_drawer": {},
    }


def _training_session(tmp_path, **options):
    _register_components()
    session = TrainingSession(_training_config(tmp_path, **options))
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    return session


def _run(session):
    with session:
        for _ in session:
            pass
    return session


# Configuration ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "match"),
    [
        ({}, "exactly one of max_iterations, epochs$"),
        (
            {"max_iterations": 2, "epochs": 1},
            "exactly one of max_iterations, epochs; it gives "
            "max_iterations, epochs",
        ),
        ({"epochs": -1}, "epochs must be a non-negative integer"),
        ({"epochs": 1.5}, "epochs must be a non-negative integer"),
        ({"epochs": True}, "epochs must be a non-negative integer"),
    ],
)
def test_session_config_takes_exactly_one_valid_run_length(
        tmp_path, length, match,
):
    _register_components()
    config = _training_config(tmp_path)
    config["session_config"] = _session_settings(tmp_path, **length)

    with pytest.raises(ValueError, match=match):
        TrainingSession(config)


def test_max_iterations_is_taken_as_given(tmp_path):
    session = _training_session(tmp_path, max_iterations=3)

    assert session.session_config.max_iterations == 3
    assert session.session_config.epochs is None


# Training -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "batch_size", "world_size", "epochs", "expected"),
    [
        (10, 4, 1, 3, 8),     # 30 samples in batches of 4
        (8, 4, 1, 2, 4),      # whole batches per epoch
        (10, 4, 4, 2, 6),     # padded to 12 per epoch on 4 ranks
        (10, 4, 2, 0, 0),
    ],
)
def test_training_epochs_resolve_against_the_padded_epoch(
        tmp_path, size, batch_size, world_size, epochs, expected,
):
    session = _training_session(
        tmp_path,
        size=size,
        batch_size=batch_size,
        world_size=world_size,
        epochs=epochs,
    )

    assert session.session_config.max_iterations == expected
    assert session.session_config.epochs == epochs
    assert session.full_config["session_config"]["epochs"] == epochs
    assert "max_iterations" not in session.full_config["session_config"]


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_training_run_ends_when_the_sampler_completes_the_epochs(
        tmp_path, world_size,
):
    session = _run(_training_session(
        tmp_path, size=10, batch_size=4, world_size=world_size, epochs=3,
    ))

    assert session.iteration == session.session_config.max_iterations
    sampler_state = resource_named(session, "data_manager").get_state()[
        "sampler_state"
    ]
    # Every sample was drawn three times; the last batch may start the next.
    assert sampler_state["epoch"] == 3


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"size": 0}, "non-empty dataset"),
        ({"batch_size": 3, "world_size": 2}, "divisible by the DDP world_size"),
    ],
)
def test_training_epochs_refuse_unrunnable_data_when_built(
        tmp_path, options, match,
):
    with pytest.raises(ValueError, match=match):
        _training_session(tmp_path, epochs=1, **options)


def test_epochs_need_a_data_manager(tmp_path):
    _register_components()
    config = {
        "session_config": _session_settings(tmp_path, epochs=1),
    }

    with pytest.raises(ValueError, match="needs a 'data_manager' resource"):
        TrainingSession(config)


def test_epochs_need_a_data_manager_that_counts_epochs(tmp_path):
    _register_components()
    config = _training_config(tmp_path, epochs=1)
    config["component_bindings"]["data_manager"] = (
        "run_length_plain_data_manager"
    )
    config["run_length_plain_data_manager"] = {}
    del config["data_manager"]

    with pytest.raises(TypeError, match="is not an EpochSource"):
        TrainingSession(config)


def test_a_data_manager_bound_to_the_role_counts_the_epochs(tmp_path):
    _register_components()
    config = _training_config(tmp_path, epochs=2)
    config["component_bindings"]["data_manager"] = (
        "run_length_counting_data_manager"
    )
    config["run_length_counting_data_manager"] = {}
    del config["data_manager"]
    del config["run_length_drawer"]

    session = TrainingSession(config)

    assert session.session_config.max_iterations == 14


def test_execution_graph_shows_the_epochs_behind_max_iterations(tmp_path):
    by_epochs = _training_session(tmp_path, epochs=3)
    by_iterations = TrainingSession(
        _training_config(tmp_path, max_iterations=3)
    )

    assert "Max iterations: 8 (from 3 epochs)\n" in by_epochs.execution_graph()
    assert "Max iterations: 3\n" in by_iterations.execution_graph()


# Checkpoints ------------------------------------------------------------------------


def test_resolved_length_survives_a_checkpoint_round_trip(tmp_path):
    session = _training_session(tmp_path, epochs=3)

    restored = TrainingSession.from_state(session.get_state())

    assert restored.session_config.max_iterations == 8
    assert restored.session_config.epochs == 3


def test_a_checkpoint_from_before_epochs_still_loads(tmp_path):
    state = _training_session(tmp_path, max_iterations=4).get_state()
    del state["session_config"]["epochs"]

    restored = TrainingSession.from_state(state)

    assert restored.session_config.max_iterations == 4
    assert restored.session_config.epochs is None


def test_manifest_records_the_resolved_total(tmp_path):
    session = _training_session(tmp_path, epochs=3)

    path = write_checkpoint(session.get_state(), tmp_path / "checkpoint")

    assert read_manifest(path)["total_iterations"] == 8


# Extension --------------------------------------------------------------------------


def test_extending_by_epochs_recounts_the_epochs(tmp_path):
    session = _training_session(tmp_path, epochs=3)

    session.apply_extension_overrides(("session_config.epochs=5",))

    assert session.session_config.max_iterations == 13   # ceil(50 / 4)
    assert session.session_config.epochs == 5
    assert session.full_config["session_config"]["epochs"] == 5


def test_extending_with_the_other_run_length_replaces_it(tmp_path):
    session = _training_session(tmp_path, max_iterations=2)

    session.apply_extension_overrides(("session_config.epochs=2",))

    assert session.session_config.max_iterations == 5    # ceil(20 / 4)
    assert "max_iterations" not in session.full_config["session_config"]

    session.apply_extension_overrides(("session_config.max_iterations=9",))

    assert session.session_config.max_iterations == 9
    assert session.session_config.epochs is None
    assert "epochs" not in session.full_config["session_config"]
    restored = TrainingSession.from_state(session.get_state())
    assert restored.session_config.max_iterations == 9


def test_extension_refuses_both_run_lengths_at_once(tmp_path):
    _register_components()
    session = TrainingSession(_training_config(tmp_path, max_iterations=2))
    before = session.full_config

    with pytest.raises(ValueError, match="sets the run length one way"):
        session.apply_extension_overrides((
            "session_config.epochs=2",
            "session_config.max_iterations=9",
            "logger.log_every=3",
        ))

    # Refused before anything was applied.
    assert session.session_config.max_iterations == 2
    assert session.full_config == before


def test_update_max_iters_replaces_a_length_given_in_epochs(tmp_path):
    session = _training_session(tmp_path, epochs=3)

    session.update_max_iters(20)

    assert session.session_config.max_iterations == 20
    assert session.session_config.epochs is None
    assert session.full_config["session_config"]["max_iterations"] == 20
    assert "epochs" not in session.full_config["session_config"]


# Analysis ---------------------------------------------------------------------------


def _analysis_session(tmp_path, *, size=5, data_manager=None, **length):
    _register_components()
    source = TrainingSession({
        "session_config": _session_settings(
            tmp_path / "training", max_iterations=1,
        ),
        "component_bindings": {"model": "run_length_source_model"},
        "run_length_source_model": {},
    })
    checkpoint = tmp_path / "training-session.pt"
    torch.save(source, checkpoint)
    return AnalysisSession({
        "session_config": _session_settings(tmp_path / "analysis", **length),
        "trained_model": {"model_checkpoint_path": str(checkpoint)},
        "component_bindings": {"dataset": "run_length_analysis_dataset"},
        "run_length_analysis_dataset": {"size": size},
        "data_manager": data_manager or {"batch_size": 2},
        "run_length_analysis_drawer": {},
    })


@pytest.mark.parametrize(
    ("size", "data_manager", "expected"),
    [
        (5, {"batch_size": 2}, 3),
        (5, {"batch_size": 2, "drop_last": True}, 2),
        (4, {"batch_size": 2}, 2),
    ],
)
def test_one_analysis_epoch_is_one_full_pass(
        tmp_path, size, data_manager, expected,
):
    session = _analysis_session(
        tmp_path, size=size, data_manager=data_manager, epochs=1,
    )

    assert session.session_config.max_iterations == expected
    with session:
        assert len(resource_named(session, "data_manager").dataloader) == (
            expected
        )
        for _ in session:
            pass
    assert session.iteration == expected


def test_analysis_runs_only_one_epoch(tmp_path):
    with pytest.raises(ValueError, match="exactly 1 epoch, not 2"):
        _analysis_session(tmp_path, epochs=2)


@pytest.mark.parametrize(
    ("size", "data_manager", "match"),
    [
        (0, {"batch_size": 2}, "non-empty dataset"),
        (1, {"batch_size": 2, "drop_last": True}, "delivers no batch"),
    ],
)
def test_analysis_epoch_refuses_data_with_no_batch(
        tmp_path, size, data_manager, match,
):
    with pytest.raises(ValueError, match=match):
        _analysis_session(
            tmp_path, size=size, data_manager=data_manager, epochs=1,
        )


# HPC --------------------------------------------------------------------------------


def test_hpc_resume_does_not_guess_a_length_given_in_epochs():
    # Without a manifest total, as for a checkpoint written before it.
    configs = {
        "by_epochs": {"sessions": [{"session_config": {"epochs": 3}}]},
        "by_iterations": {
            "sessions": [{"session_config": {"max_iterations": 300}}],
        },
    }
    checkpoints = {
        "by_epochs": [resuming.FoundCheckpoint("/batch/a", 300)],
        "by_iterations": [resuming.FoundCheckpoint("/batch/b", 300)],
    }
    by_epochs = resuming.Candidate("by_epochs", "g", 0, "interrupted")
    by_iterations = resuming.Candidate("by_iterations", "g", 1, "interrupted")

    resuming.choose_checkpoints([by_epochs, by_iterations], checkpoints, configs)

    assert by_epochs.max_iterations is None and by_epochs.reason is None
    assert by_iterations.max_iterations == 300
    assert "nothing is left to run" in by_iterations.reason
