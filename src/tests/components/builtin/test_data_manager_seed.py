"""The training DataManager shuffles with the session's rng_seed, and a
restored state keeps the seed it was saved with."""

from __future__ import annotations

from training_framework.session import TrainingSession
from tests.components.builtin.test_sampler_resharding import (  # noqa: F401 (fixture)
    _COMPONENTS_PACKAGE,
    integration_components,
)
from tests.test_utils import resource_named


def _session(tmp_path, rng_seed: int) -> TrainingSession:
    return TrainingSession({
        "session_config": {
            "rng_seed": rng_seed,
            "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": 1,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show_execution_graph": False,
        },
        "component_bindings": {
            "ddp": "integration_data_context",
            "dataset": "integration_worker_dataset",
        },
        "integration_data_context": {"rank": 0, "world_size": 1},
        "integration_worker_dataset": {"dataset_size": 64},
        "data_manager": {"batch_size": 8, "num_workers": 0, "pin_memory": False},
    })


def _order(session: TrainingSession, batches: int = 4) -> list[int]:
    with session:
        manager = resource_named(session, "data_manager")
        return [int(row[0]) for _ in range(batches) for row in next(manager.data_iter)]


def test_the_data_order_follows_rng_seed(tmp_path, integration_components):
    first = _order(_session(tmp_path / "a", rng_seed=1))
    again = _order(_session(tmp_path / "b", rng_seed=1))
    other = _order(_session(tmp_path / "c", rng_seed=2))
    assert first == again
    assert first != other
    assert sorted(first) == sorted(set(first))          # still a permutation


def test_a_saved_state_keeps_its_seed_whatever_the_new_rng_seed(
        tmp_path, integration_components,
):
    source = _session(tmp_path / "source", rng_seed=0)
    with source:
        manager = resource_named(source, "data_manager")
        next(manager.data_iter)
        saved = manager.get_state()
    assert saved["sampler_state"]["seed"] == 0
    continued = _order(_session(tmp_path / "expected", rng_seed=0), batches=2)[8:]

    target = _session(tmp_path / "target", rng_seed=99)
    restored = resource_named(target, "data_manager")
    restored.set_state(saved)
    with target:
        batch = next(restored.data_iter)
    assert [int(row[0]) for row in batch] == continued
