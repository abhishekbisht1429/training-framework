from __future__ import annotations

import importlib
import os
import pickle
import sys
import time

import pytest
import torch

from training_framework.session import TrainingSession


_COMPONENTS_PACKAGE = "tests.integration_training_components"


def _register_integration_components() -> None:
    existing = sys.modules.get(_COMPONENTS_PACKAGE)
    if existing is None:
        importlib.import_module(_COMPONENTS_PACKAGE)
    else:
        importlib.reload(existing)


def _data_manager_config(
        tmp_path,
        *,
        batch_size: int = 2,
        dataset_size: int = 8,
        dataset_name: str = "integration_worker_dataset",
        num_workers: int = 0,
        rank: int = 0,
        world_size: int = 1,
) -> dict:
    return {
        "session_config": {
            "rng_seed": 23,
            "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": 1,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show_execution_graph": False,
        },
        "aliases": {
            "ddp": "integration_data_context",
            "dataset": dataset_name,
        },
        "ddp": {
            "rank": rank,
            "world_size": world_size,
        },
        "dataset": {
            "dataset_size": dataset_size,
        },
        "data_manager": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": False,
        },
    }


def _new_session(
        tmp_path,
        *,
        register_components: bool = True,
        **config_overrides,
) -> TrainingSession:
    if register_components:
        _register_integration_components()
    session = TrainingSession(
        _data_manager_config(tmp_path, **config_overrides)
    )
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    return session


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize(
    ("batch_size", "world_size", "dataset_size", "message"),
    [
        (0, 1, 8, "positive integer"),
        (1, 2, 8, "at least the DDP world_size"),
        (3, 2, 8, "divisible by the DDP world_size"),
        (2, 2, 0, "non-empty dataset"),
    ],
)
def test_data_manager_rejects_invalid_batch_or_dataset_configuration(
        tmp_path,
        batch_size,
        world_size,
        dataset_size,
        message,
):
    if batch_size == 0:
        _register_integration_components()
        with pytest.raises(ValueError, match=message):
            TrainingSession(_data_manager_config(
                tmp_path,
                batch_size=batch_size,
                world_size=world_size,
                dataset_size=dataset_size,
            ))
        return

    session = _new_session(
        tmp_path,
        batch_size=batch_size,
        world_size=world_size,
        dataset_size=dataset_size,
    )
    with pytest.raises(ValueError, match=message):
        with session:
            pass


def test_data_manager_uses_dataset_collate_function_after_restore(tmp_path):
    source = _new_session(
        tmp_path / "source",
        dataset_name="integration_collating_dataset",
    )
    restored = TrainingSession.from_state(source.get_state())

    with restored:
        batch = next(restored.get_resource("data_manager").data_iter)

    assert batch["collated"] is True
    assert batch["indices"].shape == (2,)
    assert batch["indices"].dtype is torch.int64


def test_data_manager_rejects_non_callable_dataset_collate_function(tmp_path):
    session = _new_session(
        tmp_path,
        dataset_name="integration_invalid_collate_dataset",
    )

    with pytest.raises(TypeError, match="collate_fn must be callable"):
        with session:
            pass


def test_data_manager_resume_returns_the_exact_next_logical_batch(tmp_path):
    paused = _new_session(tmp_path, num_workers=1)
    with paused:
        data_manager = paused.get_resource("data_manager")
        first_batch = next(data_manager.data_iter)
        checkpoint_state = paused.get_state()
        expected_next_batch = next(data_manager.data_iter)

    restored = TrainingSession.from_state(checkpoint_state)
    with restored:
        actual_next_batch = next(
            restored.get_resource("data_manager").data_iter
        )

    assert not torch.equal(
        first_batch[:, 0],
        expected_next_batch[:, 0],
    )
    torch.testing.assert_close(
        actual_next_batch[:, 0],
        expected_next_batch[:, 0],
    )


def test_pickled_data_manager_resumes_at_the_exact_next_batch(tmp_path):
    source = _new_session(tmp_path / "source")
    with source:
        source_manager = source.get_resource("data_manager")
        first_batch = next(source_manager.data_iter)
        restored_manager = pickle.loads(pickle.dumps(source_manager))
        expected_next_batch = next(source_manager.data_iter)

    target = _new_session(
        tmp_path / "target",
        register_components=False,
    )
    target.unregister_resource("data_manager")
    target.register_resource(restored_manager)
    with target:
        actual_next_batch = next(restored_manager.data_iter)

    assert not torch.equal(
        first_batch[:, 0],
        expected_next_batch[:, 0],
    )
    torch.testing.assert_close(
        actual_next_batch[:, 0],
        expected_next_batch[:, 0],
    )


def test_data_manager_restores_the_same_position_for_each_current_rank(
        tmp_path,
):
    rank_zero = _new_session(
        tmp_path / "rank-zero",
        batch_size=2,
        rank=0,
        world_size=2,
    )
    with rank_zero:
        rank_zero_manager = rank_zero.get_resource("data_manager")
        next(rank_zero_manager.data_iter)
        rank_zero_state = rank_zero_manager.get_state()

    rank_one_baseline = _new_session(
        tmp_path / "rank-one-baseline",
        register_components=False,
        batch_size=2,
        rank=1,
        world_size=2,
    )
    with rank_one_baseline:
        baseline_manager = rank_one_baseline.get_resource("data_manager")
        next(baseline_manager.data_iter)
        expected_rank_one_batch = next(baseline_manager.data_iter)

    restored_rank_one = _new_session(
        tmp_path / "rank-one-restored",
        register_components=False,
        batch_size=2,
        rank=1,
        world_size=2,
    )
    restored_manager = restored_rank_one.get_resource("data_manager")
    restored_manager.set_state(rank_zero_state)
    with restored_rank_one:
        actual_rank_one_batch = next(restored_manager.data_iter)

    torch.testing.assert_close(
        actual_rank_one_batch[:, 0],
        expected_rank_one_batch[:, 0],
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dataset_size": 10}, "different dataset size"),
        ({"world_size": 2}, "different DDP world_size"),
    ],
)
def test_data_manager_rejects_incompatible_sampler_state(
        tmp_path,
        override,
        message,
):
    source = _new_session(tmp_path / "source")
    with source:
        source_manager = source.get_resource("data_manager")
        next(source_manager.data_iter)
        manager_state = source_manager.get_state()

    target = _new_session(
        tmp_path / "target",
        register_components=False,
        **override,
    )
    target.get_resource("data_manager").set_state(manager_state)
    with pytest.raises(ValueError, match=message):
        with target:
            pass


def test_data_manager_rejects_state_from_a_different_batch_size(tmp_path):
    source = _new_session(tmp_path / "source", batch_size=2)
    manager_state = source.get_resource("data_manager").get_state()

    target = _new_session(
        tmp_path / "target",
        register_components=False,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="different batch_size"):
        target.get_resource("data_manager").set_state(manager_state)


def test_data_manager_stops_worker_processes_during_teardown(tmp_path):
    session = _new_session(tmp_path, batch_size=1, num_workers=1)
    data_manager = session.get_resource("data_manager")

    with session:
        batch = next(data_manager.data_iter)
        worker_pid = int(batch[0, 1].item())
        assert worker_pid != os.getpid()
        assert _process_exists(worker_pid)

    assert data_manager.data_iter is None
    deadline = time.monotonic() + 5
    while _process_exists(worker_pid) and time.monotonic() < deadline:
        time.sleep(0.05)

    assert not _process_exists(worker_pid)
