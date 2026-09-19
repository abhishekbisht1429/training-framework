from __future__ import annotations

import importlib
import json
import socket
import sys
from pathlib import Path

import pytest
import torch
import yaml

from training_framework.engine import Configurator, TrainingEngine


_BASE_COMPONENTS = "tests.integration.integration_training_components"
_COMPONENTS_PACKAGE = "tests.integration.integration_resize_components"


pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="Resizing a real run requires PyTorch Gloo support",
)


def _register_integration_components() -> None:
    # Both, in order: `reset_registries` clears the base module's
    # registrations too, and importing the resize module will not re-run them
    # while it is already in sys.modules.
    for name in (_BASE_COMPONENTS, _COMPONENTS_PACKAGE):
        existing = sys.modules.get(name)
        if existing is None:
            importlib.import_module(name)
        else:
            importlib.reload(existing)


def _available_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _session_config(tmp_path, output_dir, *, world_size: int):
    return {
        "session_config": {
            "rng_seed": 17,
            "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show_execution_graph": False,
        },
        "component_bindings": {
            "model": "integration_ddp_model",
            "dataset": "integration_dataset",
        },
        "integration_ddp_model": {"initial_weight": 0.0},
        # Large enough that two iterations leave half the epoch unconsumed,
        # so the resumed run has somewhere to carry on to.
        "integration_dataset": {"dataset_size": 8},
        "ddp": {
            "world_size": world_size,
            "backend": "gloo",
            "parallel_components": [
                "model",
                "dataset",
                "data_manager",
                "integration_data",
                "integration_train",
                "integration_loss",
                "optimizer",
                "integration_results",
            ],
            "master_addr": "127.0.0.1",
            "master_port": _available_local_port(),
        },
        "data_manager": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "checkpointer": {"checkpoint_every": 2},
        "integration_data": {},
        "integration_train": {},
        "integration_loss": {},
        "optimizer": {
            "optimizer": {
                "name": "AdamW",
                "kwargs": {"lr": 0.1, "weight_decay": 0.0},
            },
        },
        "integration_results": {"output_dir": str(output_dir)},
    }


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    with TrainingEngine(Configurator()) as engine:
        engine.start_session()


def _only_checkpoint(tmp_path) -> Path:
    checkpoints = sorted(
        (tmp_path / "sessions").glob("session_*/checkpoints/*")
    )
    assert checkpoints, "the first run wrote no checkpoint"
    return checkpoints[-1]


def test_a_run_trained_on_two_ranks_continues_on_one(tmp_path, monkeypatch):
    """The whole point, end to end: train on two processes, finish on one.

    Exercises every piece together -- the launch resolving a new topology,
    the RNG stream restoring onto this process, and the sampler rebasing its
    position onto a world size it was not written for.
    """
    _register_integration_components()
    output_dir = tmp_path / "rank-results"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "sessions": [
                _session_config(tmp_path, output_dir, world_size=2),
            ],
        }),
        encoding="utf-8",
    )

    _run(
        [
            "training-framework",
            "--config", str(config_path),
            "--heartbeat-timeout", "30",
            "--process_timeout_on_join", "10",
        ],
        monkeypatch,
    )

    assert sorted(path.name for path in output_dir.glob("rank_*.json")) == [
        "rank_0.json",
        "rank_1.json",
    ]
    first_run = json.loads(
        (output_dir / "rank_0.json").read_text(encoding="utf-8")
    )
    assert first_run["world_size"] == 2
    assert [entry["iteration"] for entry in first_run["observations"]] == [1, 2]
    seen_before = {
        index
        for entry in first_run["observations"]
        for index in entry["sample_indices"]
    } | {
        index
        for entry in json.loads(
            (output_dir / "rank_1.json").read_text(encoding="utf-8")
        )["observations"]
        for index in entry["sample_indices"]
    }

    # Now continue the same session on a single process.
    _run(
        [
            "training-framework",
            "--extend-session", str(_only_checkpoint(tmp_path)),
            "--override",
            "session_config.max_iterations=4",
            "ddp.world_size=1",
            "--heartbeat-timeout", "30",
            "--process_timeout_on_join", "10",
        ],
        monkeypatch,
    )

    resumed = json.loads(
        (output_dir / "rank_0.json").read_text(encoding="utf-8")
    )
    assert resumed["world_size"] == 1
    assert [entry["iteration"] for entry in resumed["observations"]] == [3, 4]

    # The single rank now takes the whole global batch, and picks up where
    # the pair left off rather than starting the epoch again.
    assert all(
        len(entry["sample_indices"]) == 2
        for entry in resumed["observations"]
    )
    seen_after = {
        index
        for entry in resumed["observations"]
        for index in entry["sample_indices"]
    }
    assert seen_before.isdisjoint(seen_after)
