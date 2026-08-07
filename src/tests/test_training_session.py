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
from training_framework.components import Checkpointer, Logger, Tensorboard
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


class RecordingDDPResource(RecordingComponent):
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
    monkeypatch.setattr(factory_module, "DDPResource", RecordingDDPResource)


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
        config["cache_resource"],
        config["ddp"]
    ]
    assert len(session.hooks) == 1
    assert session.hooks[0].config is config["metrics_hook"]
    assert isinstance(session.resources[1], RecordingDDPResource)
    assert session.resources[1].config is config["ddp"]
    assert session.resources[1].rank == 0


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
    assert len(session.resources) == 1
    assert isinstance(session.resources[0], RecordingDDPResource)
    assert session.resources[0].rank == 2


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
        "training_framework.components.subprocess.Popen",
        lambda *args, **kwargs: dummy_process,
    )
    monkeypatch.setattr(
        "training_framework.components.SummaryWriter",
        DummySummaryWriter,
    )
    monkeypatch.setattr(
        "training_framework.components.time.sleep",
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