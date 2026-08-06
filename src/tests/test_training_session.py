"""Tests for config-driven training sessions and ``TrainingEngine``.

The current interface registers a complete session configuration with the
engine.  ``create_session_from_config`` then constructs the ``TrainingSession``
and its registered steps, hooks, resources, and optional DDP hook inside the
worker process.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import importlib
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training_framework.configurator import Configurator
from training_framework.dataloader import InfiniteSampler
from training_framework.resources import Checkpointer, Logger, Tensorboard
from training_framework.training_session import SessionPhase, Step, TrainingSession, step
import training_framework.training_engine as engine_module


factory_module = importlib.import_module(
    engine_module.create_session_from_config.__module__
)


class DummyClassificationDataset(Dataset):
    """Small deterministic classification dataset used by integration tests."""

    def __init__(
        self,
        num_samples: int = 100,
        num_features: int = 5,
        num_classes: int = 2,
    ) -> None:
        generator = torch.Generator().manual_seed(0)
        self.features = torch.randn(
            num_samples,
            num_features,
            generator=generator,
        )
        self.labels = torch.randint(
            0,
            num_classes,
            (num_samples,),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, labels = zip(*batch)
        return torch.stack(features), torch.stack(labels)


@step("sample_step")
class SampleStep(Step):
    """Step instantiated by ``create_session_from_config(step_config)``."""

    def __init__(self, config: dict[str, Any]) -> None:
        dataset = DummyClassificationDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 4)),
            sampler=InfiniteSampler(len(dataset)),
            collate_fn=dataset.collate_fn,
        )
        self._dataloader_iter = iter(dataloader)
        self._model = nn.Sequential(nn.Linear(5, 2))

    def run(self, session: TrainingSession) -> None:
        feature_batch, label_batch = next(self._dataloader_iter)
        feature_batch = feature_batch.to(device=session.device)
        label_batch = label_batch.to(device=session.device)

        output = self._model(feature_batch)
        loss = F.cross_entropy(output, label_batch)
        loss.backward()
        session.iteration_context["loss"] = loss.item()


def _make_session_config(tmp_path: Path, index: int) -> dict[str, Any]:
    suffix = "" if index == 0 else str(index + 1)
    return {
        "base_config": {
            "max_iterations": 12,
            "batch_size": 4,
            "sessions_dir": str(tmp_path / f"sessions{suffix}"),
            "device": "cpu",
            "rng_seed": index,
        },
        "sample_step": {
            "batch_size": 4,
        },
        "logger": {
            "log_every": 1,
            "log_file": str(tmp_path / f"log{suffix}.txt"),
        },
        "checkpointer": {
            "checkpoint_every": 10,
            "checkpoints_dir": str(tmp_path / f"checkpoints{suffix}"),
        },
        "tensorboard": {
            "host": "0.0.0.0",
            "port": 16032 + index,
        },
    }


@pytest.fixture
def sample_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "sessions": [
            _make_session_config(tmp_path, 0),
            _make_session_config(tmp_path, 1),
            _make_session_config(tmp_path, 2),
        ]
    }


def _config_with_components(
    session_config: dict[str, Any],
    *component_names: str,
) -> dict[str, Any]:
    """Return an isolated full config containing only selected components."""

    keys = ("base_config", *component_names)
    return {key: deepcopy(session_config[key]) for key in keys}


def _run_session_to_completion(
    config: dict[str, Any],
    *,
    rank: int = 0,
) -> TrainingSession:
    """Build and run a finite session without spawning a child process."""

    session = engine_module.create_session_from_config(config, rank=rank)
    max_iterations = int(config["base_config"]["max_iterations"])

    with session:
        for _ in range(max_iterations + 1):
            try:
                next(session)
            except StopIteration:
                break
        else:
            pytest.fail(
                "TrainingSession did not stop after max_iterations plus one call"
            )

    return session


def _registered_values(registry: Any) -> list[Any]:
    if hasattr(registry, "values"):
        return list(registry.values())
    return list(registry)


# ---------------------------------------------------------------------------
# Configurator coverage retained from the original tests
# ---------------------------------------------------------------------------


def test_configurator_returns_full_session_config(
    sample_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(sample_config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["", str(config_path)])

    configurator = Configurator()

    assert configurator.get_base_config(0) == sample_config["sessions"][0]
    assert configurator.get_component_config(0, "logger") == (
        sample_config["sessions"][0]["logger"]
    )


def test_configurator_override_updates_nested_component_configs(
    sample_config: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(sample_config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "",
            str(config_path),
            "--override",
            "sessions[0].checkpointer.checkpoint_every=5",
            "sessions.1.checkpointer.checkpoint_every=20",
        ],
    )

    configurator = Configurator()
    checkpointer_config_0 = configurator.get_component_config(0, "checkpointer")
    checkpointer_config_1 = configurator.get_component_config(1, "checkpointer")

    assert checkpointer_config_0["checkpoint_every"] == 5
    assert checkpointer_config_1["checkpoint_every"] == 20
    assert checkpointer_config_0 != sample_config["sessions"][0]["checkpointer"]
    assert checkpointer_config_1 != sample_config["sessions"][1]["checkpointer"]


# ---------------------------------------------------------------------------
# create_session_from_config behavior
# ---------------------------------------------------------------------------


class RecordingSession:
    def __init__(self, session_config: dict[str, Any]) -> None:
        self.session_config = session_config
        self.steps: list[Any] = []
        self.hooks: list[Any] = []
        self.resources: list[Any] = []

    def add_step(self, step_object: Any) -> None:
        self.steps.append(step_object)

    def register_hook(self, hook_object: Any) -> None:
        self.hooks.append(hook_object)

    def register_resource(self, resource_object: Any) -> None:
        self.resources.append(resource_object)


class RecordingComponent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config


class RecordingDDPHook(RecordingComponent):
    def __init__(self, config: dict[str, Any], *, rank: int) -> None:
        super().__init__(config)
        self.rank = rank


def _install_recording_session_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "TrainingSession", RecordingSession)
    monkeypatch.setattr(
        factory_module,
        "STEP_REGISTRY",
        {"train_step": RecordingComponent},
    )
    monkeypatch.setattr(
        factory_module,
        "HOOK_REGISTRY",
        {"metrics_hook": RecordingComponent},
    )
    monkeypatch.setattr(
        factory_module,
        "RESOURCE_REGISTRY",
        {"cache_resource": RecordingComponent},
    )
    monkeypatch.setattr(factory_module, "DDPHook", RecordingDDPHook)


def test_create_session_from_config_registers_all_component_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_recording_session_factory(monkeypatch)
    config = {
        "base_config": {"max_iterations": 3},
        "train_step": {"value": "step"},
        "metrics_hook": {"value": "hook"},
        "cache_resource": {"value": "resource"},
        "ddp": {"n_proc": 2},
    }

    session = engine_module.create_session_from_config(config)

    assert isinstance(session, RecordingSession)
    assert session.session_config is config["base_config"]
    assert [component.config for component in session.steps] == [
        config["train_step"]
    ]
    assert [component.config for component in session.resources] == [
        config["cache_resource"]
    ]
    assert len(session.hooks) == 2
    assert session.hooks[0].config is config["metrics_hook"]
    assert isinstance(session.hooks[1], RecordingDDPHook)
    assert session.hooks[1].config is config["ddp"]
    assert session.hooks[1].rank == 0


def test_create_session_from_config_secondary_rank_keeps_only_parallel_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_recording_session_factory(monkeypatch)
    monkeypatch.setattr(
        factory_module,
        "STEP_REGISTRY",
        {
            "parallel_step": RecordingComponent,
            "main_process_step": RecordingComponent,
        },
    )
    monkeypatch.setattr(
        factory_module,
        "HOOK_REGISTRY",
        {"implicit_main_process_hook": RecordingComponent},
    )
    monkeypatch.setattr(factory_module, "RESOURCE_REGISTRY", {})

    config = {
        "base_config": {"max_iterations": 3},
        "parallel_step": {"parallel": True, "value": "parallel"},
        "main_process_step": {"parallel": False, "value": "rank-zero-only"},
        "implicit_main_process_hook": {"value": "rank-zero-by-default"},
        "ddp": {"n_proc": 4},
    }

    session = engine_module.create_session_from_config(config, rank=2)

    assert [component.config for component in session.steps] == [
        config["parallel_step"]
    ]
    assert len(session.hooks) == 1
    assert isinstance(session.hooks[0], RecordingDDPHook)
    assert session.hooks[0].rank == 2


def test_create_session_from_config_rejects_unknown_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_recording_session_factory(monkeypatch)
    config = {
        "base_config": {"max_iterations": 1},
        "not_registered": {},
    }

    with pytest.raises(
        ValueError,
        match="No step, hook or resource registered with name 'not_registered'",
    ):
        engine_module.create_session_from_config(config)


# ---------------------------------------------------------------------------
# Config-driven integration tests for built-in resources
# ---------------------------------------------------------------------------


def test_logger_is_created_from_config_and_writes_each_iteration(
    sample_config: dict[str, Any],
) -> None:
    full_config = sample_config["sessions"][0]
    config = _config_with_components(full_config, "sample_step", "logger")

    session = _run_session_to_completion(config)

    log_path = Path(config["logger"]["log_file"])
    lines = log_path.read_text(encoding="utf-8").splitlines()
    max_iterations = config["base_config"]["max_iterations"]

    assert session._phase is SessionPhase.FINISHED
    assert len(lines) == max_iterations
    assert lines == [
        f"Iteration {iteration}/{max_iterations}"
        for iteration in range(1, max_iterations + 1)
    ]


def test_checkpointer_is_created_from_config_and_restores_sessions(
    sample_config: dict[str, Any],
) -> None:
    full_config = sample_config["sessions"][0]
    config = _config_with_components(full_config, "sample_step", "checkpointer")

    session = _run_session_to_completion(config)

    checkpoints_dir = Path(config["checkpointer"]["checkpoints_dir"])
    checkpoint_paths = sorted(path for path in checkpoints_dir.iterdir() if path.is_file())
    assert checkpoint_paths

    reloaded_sessions = [
        Checkpointer.load_checkpoint(str(checkpoint_path))
        for checkpoint_path in checkpoint_paths
    ]
    sessions_by_iteration = {
        reloaded_session.iteration: reloaded_session
        for reloaded_session in reloaded_sessions
    }

    assert {1, 10}.issubset(sessions_by_iteration)
    for iteration in (1, 10):
        reloaded_session = sessions_by_iteration[iteration]
        assert reloaded_session.session_config == session.session_config

        reloaded_hooks = _registered_values(reloaded_session._hooks)
        original_hooks = _registered_values(session._hooks)
        assert len(reloaded_hooks) == 1
        assert len(original_hooks) == 1
        assert reloaded_hooks[0].call_every == original_hooks[0].call_every


def test_tensorboard_resource_is_created_and_cleaned_up_from_config(
    sample_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_config = sample_config["sessions"][0]
    config = _config_with_components(full_config, "sample_step", "tensorboard")

    class DummyProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            self.killed = True

    class DummySummaryWriter:
        def __init__(self, log_dir: str) -> None:
            self.log_dir = log_dir
            self.closed = False
            self.scalars: list[tuple[Any, ...]] = []

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            self.scalars.append((*args, kwargs))

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    dummy_process = DummyProcess()
    monkeypatch.setattr(
        "training_framework.resources.subprocess.Popen",
        lambda *args, **kwargs: dummy_process,
    )
    monkeypatch.setattr(
        "training_framework.resources.SummaryWriter",
        DummySummaryWriter,
    )
    monkeypatch.setattr(
        "training_framework.resources.time.sleep",
        lambda *_args, **_kwargs: None,
    )

    session = engine_module.create_session_from_config(config)
    resources = _registered_values(session._resources)
    assert len(resources) == 1
    tensorboard = resources[0]
    assert isinstance(tensorboard, Tensorboard)
    assert tensorboard.summary_writer is None

    max_iterations = config["base_config"]["max_iterations"]
    with session:
        for _ in range(max_iterations + 1):
            try:
                next(session)
            except StopIteration:
                break
        else:
            pytest.fail("TrainingSession did not stop at max_iterations")

    assert tensorboard.summary_writer is None
    assert dummy_process.terminated is True
    assert session._phase is SessionPhase.FINISHED


# ---------------------------------------------------------------------------
# Worker and process-engine tests without real multiprocessing
# ---------------------------------------------------------------------------


class ScriptedQueue:
    def __init__(
        self,
        maxsize: int = 1,
        *,
        items: Iterable[Any] = (),
        empty_results: Iterable[bool] = ()
    ) -> None:
        self.maxsize = maxsize
        self.items = deque(items)
        self.empty_results = deque(empty_results)
        self.put_calls: list[Any] = []
        self.get_calls = 0

    def empty(self) -> bool:
        if self.empty_results:
            return self.empty_results.popleft()
        return not self.items

    def put(self, value: Any, block=False) -> None:
        self.put_calls.append(value)
        self.items.append(value)

    def get(self) -> Any:
        self.get_calls += 1
        if not self.items:
            raise AssertionError("Worker attempted to read an empty test queue")
        return self.items.popleft()


class IteratorSession:
    def __init__(self, values: Iterable[Any] = ()) -> None:
        self._values = iter(values)
        self.entered = False
        self.exited = False
        self.next_calls = 0

    def __enter__(self) -> "IteratorSession":
        self.entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exited = True

    def __iter__(self) -> "IteratorSession":
        return self

    def __next__(self) -> Any:
        self.next_calls += 1
        return next(self._values)


class FakeProcess:
    def __init__(
        self,
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.alive = False
        self.finish_on_join = True
        self.join_timeouts: list[float | None] = []
        self.terminated = False

    def start(self) -> None:
        self.started = True
        self.alive = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if self.finish_on_join:
            self.alive = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


def _install_multiprocessing_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[ScriptedQueue], list[FakeProcess]]:
    queues: list[ScriptedQueue] = []
    processes: list[FakeProcess] = []

    def queue_factory(maxsize: int = 1) -> ScriptedQueue:
        queue = ScriptedQueue(maxsize=maxsize)
        queues.append(queue)
        return queue

    def process_factory(
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...],
    ) -> FakeProcess:
        process = FakeProcess(target=target, args=args)
        processes.append(process)
        return process

    monkeypatch.setattr(engine_module, "Queue", queue_factory)
    monkeypatch.setattr(engine_module, "Process", process_factory)
    return queues, processes


def test_proc_worker_builds_session_from_full_config_and_runs_until_signal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "base_config": {"max_iterations": 2},
        "sample_step": {"batch_size": 4},
    }
    queue = ScriptedQueue(
        items=["pause"],
        empty_results=[True, True, False],
    )
    session = IteratorSession(values=["iteration-1", "iteration-2"])
    create_calls: list[dict[str, Any]] = []

    def create_session(received_config: dict[str, Any]) -> IteratorSession:
        create_calls.append(received_config)
        return session

    monkeypatch.setattr(engine_module, "create_session_from_config", create_session)

    engine_module.proc_worker(config, session_id=7, queue=queue)

    assert create_calls == [config]
    assert session.entered is True
    assert session.exited is True
    assert session.next_calls == 2
    assert queue.get_calls == 1
    output = capsys.readouterr().out
    assert "Starting session 7" in output
    assert "signal pause received." in output


def test_ddp_proc_worker_passes_rank_to_session_factory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "base_config": {"max_iterations": 1},
        "ddp": {"n_proc": 2},
    }
    queue = ScriptedQueue(items=["stop"], empty_results=[True, False])
    session = IteratorSession(values=["iteration-1"])
    create_calls: list[tuple[dict[str, Any], int]] = []

    def create_session(
        received_config: dict[str, Any],
        *,
        rank: int,
    ) -> IteratorSession:
        create_calls.append((received_config, rank))
        return session

    monkeypatch.setattr(engine_module, "create_session_from_config", create_session)

    engine_module.ddp_proc_worker(config, rank=1, queue=queue)

    assert create_calls == [(config, 1)]
    assert session.next_calls == 1
    assert session.exited is True
    assert queue.get_calls == 1
    assert "Parallel Session 1: signal stop received." in capsys.readouterr().out


def test_parallel_engine_registers_full_configs_and_starts_selected_process(
    sample_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queues, processes = _install_multiprocessing_fakes(monkeypatch)
    training_engine = engine_module.TrainingEngine({})
    first_config, second_config = sample_config["sessions"][:2]

    first_id = training_engine.register_session(first_config)
    second_id = training_engine.register_session(second_config)

    assert (first_id, second_id) == (0, 1)
    assert len(queues) == 2
    assert len(processes) == 2
    assert processes[0].target is engine_module.proc_worker
    assert processes[0].args == (first_config, first_id, queues[0])
    assert processes[1].args == (second_config, second_id, queues[1])

    with training_engine:
        training_engine.start_session(second_id)

    assert processes[0].started is False
    assert processes[1].started is True



# def test_configurator_configs_can_be_registered_and_started(
#     sample_config: dict[str, Any],
#     tmp_path: Path,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     config_path = tmp_path / "config.yaml"
#     config_path.write_text(yaml.safe_dump(sample_config), encoding="utf-8")
#     monkeypatch.setattr(sys, "argv", ["", str(config_path)])
#     queues, processes = _install_multiprocessing_fakes(monkeypatch)
#
#     configurator = Configurator()
#     session_configs = [
#         configurator.get_base_config(index)
#         for index in range(len(sample_config["sessions"]))
#     ]
#     training_engine = engine_module.TrainingEngine({})
#
#     with training_engine:
#         session_ids = [
#             training_engine.register_session(config)
#             for config in session_configs
#         ]
#         for session_id in session_ids:
#             training_engine.start_session(session_id)
#
#         assert len(training_engine._session_processes) == len(session_configs)
#         assert len(training_engine._signal_queues) == len(session_configs)
#         assert all(process.started for process in processes)
#
#     assert [queue.put_calls for queue in queues] == [[1], [1], [1]]
#     # assert all(process.join_timeouts == [2.0] for process in processes)
#     assert all(not process.terminated for process in processes)


# @pytest.mark.xfail(
#     strict=False,
#     reason=(
#         "The supplied DDP branch creates recursive proc_worker children and "
#         "does not start them. It should create/start ddp_proc_worker children."
#     ),
# )
def test_proc_worker_starts_one_ddp_worker_for_each_nonzero_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, processes = _install_multiprocessing_fakes(monkeypatch)
    monkeypatch.setattr(
        engine_module,
        "create_session_from_config",
        lambda config: IteratorSession(),
    )
    config = {
        "base_config": {"max_iterations": 1},
        "ddp": {"n_proc": 4},
    }

    engine_module.proc_worker(
        config,
        session_id=0,
        queue=ScriptedQueue(items=["stop"], empty_results=[False]),
    )

    assert len(processes) == 3
    assert all(process.target is engine_module.ddp_proc_worker for process in processes)
    assert [process.args[1] for process in processes] == [1, 2, 3]
    assert all(process.started for process in processes)