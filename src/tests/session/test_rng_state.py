from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from training_framework.components import StatefulResource, resource
from training_framework.components.builtin import Checkpointer
from training_framework.session import AnalysisSession, Session, TrainingSession
from training_framework.session.state import (
    CHECKPOINT_VERSION,
    capture_rng_state,
    restore_rng_state,
    rng_restore_suppressed,
)
from tests.test_utils import resource_named


class _FakeCuda:
    """A stand-in for `torch.cuda` so these run on a machine without one."""

    def __init__(self, *, device_count: int, initialized: bool):
        self.device_count = device_count
        self.initialized = initialized
        self.live_stream = torch.tensor([9, 9, 9], dtype=torch.uint8)
        self.applied: list[Any] = []
        self.seeded: list[int] = []

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: self.device_count > 0
        )
        monkeypatch.setattr(
            torch.cuda, "device_count", lambda: self.device_count
        )
        monkeypatch.setattr(
            torch.cuda, "is_initialized", lambda: self.initialized
        )
        monkeypatch.setattr(
            torch.cuda, "get_rng_state", lambda *a, **k: self.live_stream
        )
        monkeypatch.setattr(
            torch.cuda,
            "set_rng_state",
            lambda state, *a, **k: self.applied.append(state),
        )
        monkeypatch.setattr(
            torch.cuda,
            "manual_seed",
            lambda seed: self.seeded.append(seed),
        )


@pytest.fixture
def fake_cuda(monkeypatch):
    def _install(*, device_count: int = 2, initialized: bool = True):
        cuda = _FakeCuda(device_count=device_count, initialized=initialized)
        cuda.install(monkeypatch)
        return cuda

    return _install


def _rng_state(cuda_rng_state: Any) -> dict[str, Any]:
    state = capture_rng_state()
    state["cuda_rng_state"] = cuda_rng_state
    return state


def _streams(count: int) -> list[torch.Tensor]:
    return [
        torch.tensor([index], dtype=torch.uint8) for index in range(count)
    ]


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _next_draws():
    return (
        torch.rand(4).tolist(),
        random.random(),
        np.random.rand(4).tolist(),
    )


# ---------------------------------------------------------------------------
# Restoring
# ---------------------------------------------------------------------------


def test_a_checkpoint_from_more_gpus_restores_on_fewer(fake_cuda):
    """Eight GPUs' worth of streams, restored on a four-GPU machine."""
    cuda = fake_cuda(device_count=4)
    streams = _streams(8)

    carried = restore_rng_state(_rng_state(streams), rng_seed=11)

    assert carried is None
    assert len(cuda.applied) == 1
    assert torch.equal(cuda.applied[0], streams[0])


def test_a_legacy_state_restores_the_writers_stream(fake_cuda):
    """Entry 0 is the live one; the rest were frozen when the run started."""
    cuda = fake_cuda(device_count=8)
    streams = _streams(8)

    restore_rng_state(_rng_state(streams), rng_seed=11)

    assert torch.equal(cuda.applied[0], streams[0])


def test_a_recorded_stream_is_restored(fake_cuda):
    cuda = fake_cuda()
    stream = torch.tensor([42], dtype=torch.uint8)

    carried = restore_rng_state(_rng_state(stream), rng_seed=11)

    assert carried is None
    assert torch.equal(cuda.applied[0], stream)


def test_without_a_recorded_stream_the_device_is_seeded(fake_cuda):
    """So `rng_seed` still governs CUDA randomness on a fresh run."""
    cuda = fake_cuda()

    carried = restore_rng_state(_rng_state(None), rng_seed=11)

    assert carried is None
    assert cuda.applied == []
    assert cuda.seeded == [11]


def test_an_unpinned_process_carries_the_stream_instead(fake_cuda):
    """The parent restores a session only to hand its state to the workers."""
    cuda = fake_cuda(device_count=4, initialized=False)
    stream = torch.tensor([42], dtype=torch.uint8)

    carried = restore_rng_state(_rng_state(stream), rng_seed=11)

    assert carried is stream
    assert cuda.applied == []
    assert cuda.seeded == []


def test_a_machine_without_cuda_carries_the_stream(fake_cuda):
    cuda = fake_cuda(device_count=0, initialized=False)
    stream = torch.tensor([42], dtype=torch.uint8)

    assert restore_rng_state(_rng_state(stream), rng_seed=11) is stream
    assert cuda.applied == []


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="exercises the real CUDA generators",
)
def test_a_real_legacy_state_restores_without_indexing_past_the_devices():
    """The reported crash, against the real generators.

    `set_rng_state_all` walked every recorded entry and indexed
    `default_generators` by position, which is what raised IndexError when a
    checkpoint held more entries than the machine has devices.
    """
    with torch.cuda.device(0):
        original = torch.cuda.get_rng_state()
        try:
            torch.cuda.manual_seed(1)
            legacy = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed(2)

            carried = restore_rng_state(_rng_state(legacy), rng_seed=11)

            assert carried is None
            assert torch.equal(torch.cuda.get_rng_state(), legacy[0])
        finally:
            torch.cuda.set_rng_state(original)


def test_an_empty_legacy_list_seeds_the_device(fake_cuda):
    cuda = fake_cuda()

    restore_rng_state(_rng_state([]), rng_seed=11)

    assert cuda.applied == []
    assert cuda.seeded == [11]


# ---------------------------------------------------------------------------
# Capturing
# ---------------------------------------------------------------------------


def test_a_pinned_process_records_its_live_stream(fake_cuda):
    cuda = fake_cuda()
    carried = torch.tensor([1], dtype=torch.uint8)

    captured = capture_rng_state(carried)

    assert torch.equal(captured["cuda_rng_state"], cuda.live_stream)


def test_an_unpinned_process_passes_on_what_it_carries(fake_cuda):
    fake_cuda(device_count=4, initialized=False)
    carried = torch.tensor([1], dtype=torch.uint8)

    captured = capture_rng_state(carried)

    assert captured["cuda_rng_state"] is carried


def test_a_captured_state_records_nothing_when_there_is_nothing(fake_cuda):
    fake_cuda(device_count=0, initialized=False)

    assert capture_rng_state()["cuda_rng_state"] is None


# ---------------------------------------------------------------------------
# The streams that are not CUDA
# ---------------------------------------------------------------------------


def test_the_cpu_streams_resume_exactly():
    state = capture_rng_state()
    expected = _next_draws()

    restore_rng_state(state)

    assert _next_draws() == expected


# ---------------------------------------------------------------------------
# Through a session
# ---------------------------------------------------------------------------


def _session_config(root) -> dict[str, Any]:
    return {
        "rng_seed": 19,
        "sessions_dir": str(root),
        "max_iterations": 2,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def test_a_session_records_the_checkpoint_version(tmp_path):
    session = TrainingSession({"session_config": _session_config(tmp_path)})

    assert session.get_state()["checkpoint_version"] == CHECKPOINT_VERSION


def test_a_version_one_state_still_loads(tmp_path, fake_cuda):
    """A checkpoint written before this change, on a machine with 8 GPUs."""
    cuda = fake_cuda(device_count=4)
    session = TrainingSession({"session_config": _session_config(tmp_path)})
    state = session.get_state()
    del state["checkpoint_version"]
    streams = _streams(8)
    state["cuda_rng_state"] = streams

    restored = Session.from_state(state)

    assert restored.iteration == session.iteration
    assert torch.equal(cuda.applied[0], streams[0])


def test_a_carried_stream_reaches_the_workers(tmp_path, fake_cuda):
    """A resume must not drop the stream while passing through the parent."""
    fake_cuda(device_count=4, initialized=False)
    session = TrainingSession({"session_config": _session_config(tmp_path)})
    state = session.get_state()
    stream = torch.tensor([42], dtype=torch.uint8)
    state["cuda_rng_state"] = stream

    parent = Session.from_state(state)

    assert parent.get_state()["cuda_rng_state"] is stream


def test_a_suppressed_restore_leaves_the_rng_alone(tmp_path):
    session = TrainingSession({"session_config": _session_config(tmp_path)})
    state = session.get_state()

    _seed_everything(1234)
    expected = _next_draws()

    _seed_everything(1234)
    with rng_restore_suppressed():
        Session.from_state(state)

    assert _next_draws() == expected


# ---------------------------------------------------------------------------
# Loading a checkpoint for its weights
# ---------------------------------------------------------------------------


class _SourceModel(nn.Module, StatefulResource):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(float(config["weight"])))

    def forward(self, value):
        return self.weight * value

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return self.state_dict()

    def set_state(self, state):
        self.load_state_dict(state)


def _training_checkpoint(tmp_path):
    resource("rng_state_source_model")(_SourceModel)
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training"),
        "component_bindings": {"model": "rng_state_source_model"},
        "rng_state_source_model": {"weight": 3.5},
    })
    checkpoint_path = tmp_path / "training-session.pt"
    torch.save(source, checkpoint_path)
    return checkpoint_path


def test_loading_for_weights_does_not_reseed_the_caller(tmp_path):
    checkpoint_path = _training_checkpoint(tmp_path)

    _seed_everything(4321)
    expected = _next_draws()

    _seed_everything(4321)
    Checkpointer.load_checkpoint(checkpoint_path, restore_rng=False)

    assert _next_draws() == expected


def test_loading_a_checkpoint_still_restores_the_rng_by_default(tmp_path):
    checkpoint_path = _training_checkpoint(tmp_path)

    _seed_everything(4321)
    baseline = _next_draws()

    _seed_everything(4321)
    Checkpointer.load_checkpoint(checkpoint_path)

    assert _next_draws() != baseline


def test_the_trained_model_resource_leaves_the_rng_alone(tmp_path):
    checkpoint_path = _training_checkpoint(tmp_path)
    session = AnalysisSession({
        "session_config": _session_config(tmp_path / "analysis"),
        "trained_model": {"model_checkpoint_path": str(checkpoint_path)},
    })

    _seed_everything(4321)
    expected = _next_draws()

    _seed_everything(4321)
    resource_named(session, "trained_model").setup(session)

    assert _next_draws() == expected


class _DrawingModel(nn.Module, StatefulResource):
    """A model whose rebuild draws from every generator, as weight
    initialisation does, before its saved state overwrites the result."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self._config = config
        self.weight = nn.Parameter(torch.rand(1))
        random.random()
        np.random.rand()

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return {
            "weights": self.state_dict(),
            "fail_on_restore": self._config.get("fail_on_restore", False),
        }

    def set_state(self, state):
        torch.rand(1)
        random.random()
        np.random.rand()
        if state["fail_on_restore"]:
            raise RuntimeError("restore failed")
        self.load_state_dict(state["weights"])


def _drawing_checkpoint(tmp_path, **model_config):
    resource("rng_state_drawing_model")(_DrawingModel)
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training"),
        "component_bindings": {"model": "rng_state_drawing_model"},
        "rng_state_drawing_model": model_config,
    })
    checkpoint_path = tmp_path / "drawing-session.pt"
    torch.save(source, checkpoint_path)
    return checkpoint_path


def test_loading_a_component_leaves_the_callers_sequence_unchanged(tmp_path):
    checkpoint_path = _drawing_checkpoint(tmp_path)

    _seed_everything(4321)
    expected = _next_draws()

    _seed_everything(4321)
    Checkpointer.load_component(checkpoint_path, "model")

    assert _next_draws() == expected


def test_loading_without_the_rng_undoes_what_rebuilding_drew(tmp_path):
    checkpoint_path = _drawing_checkpoint(tmp_path)

    _seed_everything(4321)
    expected = _next_draws()

    _seed_everything(4321)
    Checkpointer.load_checkpoint(checkpoint_path, restore_rng=False)

    assert _next_draws() == expected


def test_a_failed_load_still_leaves_the_rng_alone(tmp_path):
    checkpoint_path = _drawing_checkpoint(tmp_path, fail_on_restore=True)

    _seed_everything(4321)
    expected = _next_draws()

    _seed_everything(4321)
    with pytest.raises(RuntimeError, match="restore failed"):
        Checkpointer.load_component(checkpoint_path, "model")

    assert _next_draws() == expected
