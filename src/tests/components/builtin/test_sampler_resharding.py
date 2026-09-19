from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from training_framework.dataloader import (
    SAMPLER_STATE_VERSION,
    DistributedInfiniteSampler,
    consumed_in_epoch,
)
from training_framework.session import TrainingSession


_COMPONENTS_PACKAGE = "tests.integration.integration_training_components"


def _sampler(num_samples: int, rank: int, world_size: int, **kwargs):
    return DistributedInfiniteSampler(
        num_samples=num_samples,
        rank=rank,
        world_size=world_size,
        **kwargs,
    )


def _take(sampler: DistributedInfiniteSampler, count: int) -> list[int]:
    iterator = iter(sampler)
    return [next(iterator) for _ in range(count)]


def _run_ranks(
        num_samples: int,
        world_size: int,
        per_rank: int,
        state: dict[str, Any] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Advance every rank of a world in lockstep, as a real run does."""
    delivered: list[int] = []
    states: list[dict[str, Any]] = []
    for rank in range(world_size):
        sampler = _sampler(num_samples, rank, world_size)
        if state is not None:
            sampler.set_state(state)
        delivered.extend(_take(sampler, per_rank))
        states.append(sampler.get_state())
    return delivered, states


# ---------------------------------------------------------------------------
# The saved position no longer belongs to a world size
# ---------------------------------------------------------------------------


def test_the_saved_position_counts_the_whole_epoch():
    _, states = _run_ranks(32, 8, 2)

    assert states[0]["state_version"] == SAMPLER_STATE_VERSION
    assert states[0]["consumed_in_epoch"] == 16


def test_every_rank_agrees_on_the_position():
    """Ranks advance in lockstep, which is what makes the count meaningful."""
    _, states = _run_ranks(32, 8, 2)

    assert {state["consumed_in_epoch"] for state in states} == {16}


@pytest.mark.parametrize(
    ("before_size", "after_size"),
    [(8, 4), (4, 8), (4, 1), (1, 4), (8, 8), (3, 6)],
)
def test_a_resumed_epoch_delivers_the_rest_and_nothing_twice(
        before_size,
        after_size,
):
    num_samples = 24
    per_rank = (num_samples // before_size) // 2

    before, states = _run_ranks(num_samples, before_size, per_rank)
    checkpoint = states[0]

    resumed = _sampler(num_samples, 0, after_size)
    resumed.set_state(checkpoint)
    remaining = resumed.num_samples_per_rank - resumed.index_within_epoch

    after, _ = _run_ranks(
        num_samples,
        after_size,
        remaining,
        state=checkpoint,
    )

    assert set(before).isdisjoint(after)

    # Every rank has to resume at the same offset into its own slice, so the
    # position rounds up to a multiple of the new world size. What it rounds
    # past is skipped rather than delivered a second time -- and nothing is
    # skipped when the position already divides.
    missed = set(range(num_samples)) - set(before) - set(after)
    assert len(missed) < after_size
    if consumed_in_epoch(checkpoint) % after_size == 0:
        assert not missed


def test_the_epoch_holds_the_same_samples_at_any_world_size():
    """Only the slicing is topological; the permutation is not."""
    epochs = {
        world_size: sorted(
            _run_ranks(24, world_size, 24 // world_size)[0]
        )
        for world_size in (1, 2, 3, 4, 6, 8)
    }

    assert all(epoch == list(range(24)) for epoch in epochs.values())


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_a_shrink_at_the_epoch_boundary_carries_into_the_next_epoch():
    """Rebasing can land past the end of a shorter epoch.

    Without the carry the sampler would slice its epoch to nothing and
    silently skip a whole pass over the data.
    """
    state = {
        "state_version": SAMPLER_STATE_VERSION,
        "epoch": 3,
        "consumed_in_epoch": 12,
        "num_samples": 10,
        "seed": 0,
        "shuffle": True,
        "drop_last": False,
    }
    sampler = _sampler(10, rank=0, world_size=8)

    sampler.set_state(state)

    assert (sampler.epoch, sampler.index_within_epoch) == (4, 0)
    assert len(_take(sampler, 2)) == 2


def test_a_rebased_position_never_precedes_what_was_consumed():
    for world_size in range(1, 9):
        state = {
            "state_version": SAMPLER_STATE_VERSION,
            "epoch": 0,
            "consumed_in_epoch": 13,
            "num_samples": 64,
            "seed": 0,
            "shuffle": True,
            "drop_last": False,
        }
        sampler = _sampler(64, rank=0, world_size=world_size)

        sampler.set_state(state)

        assert sampler.index_within_epoch * world_size >= 13


def test_a_dataset_too_small_to_split_is_rejected():
    sampler = _sampler(2, rank=0, world_size=4, drop_last=True)

    with pytest.raises(ValueError, match="cannot be split across 4 ranks"):
        sampler.set_state({
            "epoch": 0,
            "consumed_in_epoch": 0,
            "num_samples": 2,
            "drop_last": True,
        })


# ---------------------------------------------------------------------------
# Version 1 states
# ---------------------------------------------------------------------------


def test_a_version_one_position_is_read_as_a_whole_epoch_count():
    legacy = {
        "epoch": 2,
        "index_within_epoch": 2,
        "world_size": 8,
        "num_samples": 32,
        "seed": 0,
        "shuffle": True,
        "drop_last": False,
    }

    assert consumed_in_epoch(legacy) == 16


def test_a_version_one_state_resumes_on_a_different_world_size():
    num_samples = 24
    before, _ = _run_ranks(num_samples, 8, 1)
    legacy = {
        "epoch": 0,
        "index_within_epoch": 1,
        "world_size": 8,
        "num_samples": num_samples,
        "seed": 0,
        "shuffle": True,
        "drop_last": False,
    }

    after, _ = _run_ranks(num_samples, 4, 4, state=legacy)

    assert set(before).isdisjoint(after)
    assert sorted(before + after) == list(range(num_samples))


# ---------------------------------------------------------------------------
# Through the DataManager
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_components():
    """Register the integration components once for the test.

    `reset_registries` clears them before every test, and re-running the
    decorators twice in one test collides, so this happens exactly once.
    """
    existing = sys.modules.get(_COMPONENTS_PACKAGE)
    if existing is None:
        importlib.import_module(_COMPONENTS_PACKAGE)
    else:
        importlib.reload(existing)


def _session(tmp_path, *, rank: int, world_size: int, dataset_size: int = 16):
    return TrainingSession({
        "session_config": {
            "rng_seed": 23,
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
        "integration_data_context": {
            "rank": rank,
            "world_size": world_size,
        },
        "integration_worker_dataset": {"dataset_size": dataset_size},
        "data_manager": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
    })


def test_a_data_manager_resumes_on_a_smaller_world(
        tmp_path,
        integration_components,
):
    source = _session(tmp_path / "source", rank=0, world_size=2)
    with source:
        manager = source.get_resource("data_manager")
        next(manager.data_iter)
        saved = manager.get_state()

    # Two ranks each took half of one global batch of four.
    assert saved["sampler_state"]["consumed_in_epoch"] == 4

    target = _session(tmp_path / "target", rank=0, world_size=1)
    restored = target.get_resource("data_manager")
    restored.set_state(saved)

    with target:
        batch = next(restored.data_iter)

    assert len(batch) == 4
    # The single rank now takes the whole global batch, and the epoch
    # position carries on from where two ranks left it.
    assert restored.get_state()["sampler_state"]["consumed_in_epoch"] == 8
