from __future__ import annotations

import importlib
import json
import os
import socket
import sys

import pytest
import torch
import yaml

from training_framework.engine import Configurator
from training_framework.engine import TrainingEngine


_COMPONENTS_PACKAGE = "tests.integration_training_components"


pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="The real DDP integration test requires PyTorch Gloo support",
)


def _register_integration_components() -> None:
    existing = sys.modules.get(_COMPONENTS_PACKAGE)
    if existing is None:
        importlib.import_module(_COMPONENTS_PACKAGE)
    else:
        importlib.reload(existing)


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_full_spawned_ddp_training_flow(tmp_path, monkeypatch):
    _register_integration_components()
    output_dir = tmp_path / "rank-results"
    session_config = {
        "base_config": {
            "rng_seed": 17,
            "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": 3,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show-execution-graph": False,
        },
        "aliases": {
            "model": "integration_ddp_model",
            "dataset": "integration_dataset",
        },
        "model": {
            "initial_weight": 0.0,
        },
        "dataset": {
            "dataset_size": 4,
        },
        "ddp": {
            "world_size": 2,
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
            "master_port": str(_available_local_port()),
        },
        "data_manager": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "integration_data": {},
        "integration_train": {},
        "integration_loss": {},
        "optimizer": {
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "warmup_iters": 1,
        },
        "integration_results": {
            "output_dir": str(output_dir),
        },
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({"sessions": [session_config]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training-framework",
            "--config",
            str(config_path),
            "--heartbeat-timeout",
            "30",
            "--process_timeout_on_join",
            "10",
        ],
    )

    with TrainingEngine(Configurator()) as engine:
        engine.start_session()

    results = [
        json.loads(
            (output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ]

    assert [result["rank"] for result in results] == [0, 1]
    child_pids = {result["pid"] for result in results}
    assert len(child_pids) == 2
    assert os.getpid() not in child_pids

    rank_zero_observations = results[0]["observations"]
    rank_one_observations = results[1]["observations"]
    assert [
        observation["iteration"] for observation in rank_zero_observations
    ] == [1, 2, 3]
    assert [
        observation["iteration"] for observation in rank_one_observations
    ] == [1, 2, 3]
    rank_zero_indices = [
        observation["sample_index"] for observation in rank_zero_observations
    ]
    rank_one_indices = [
        observation["sample_index"] for observation in rank_one_observations
    ]
    assert set(rank_zero_indices[:2]).isdisjoint(rank_one_indices[:2])
    assert set(rank_zero_indices[:2] + rank_one_indices[:2]) == {0, 1, 2, 3}
    assert all(
        rank_zero_index != rank_one_index
        for rank_zero_index, rank_one_index in zip(
            rank_zero_indices,
            rank_one_indices,
            strict=True,
        )
    )

    for rank_observations in (
        rank_zero_observations,
        rank_one_observations,
    ):
        for observation in rank_observations:
            assert observation["target"] == observation["sample_index"] + 1
            assert observation["loss"] == pytest.approx(
                (observation["prediction"] - observation["target"]) ** 2,
                rel=1e-6,
                abs=1e-6,
            )

    for rank_zero, rank_one in zip(
        rank_zero_observations,
        rank_one_observations,
        strict=True,
    ):
        assert rank_zero["prediction"] == pytest.approx(
            rank_one["prediction"],
            rel=1e-6,
            abs=1e-6,
        )

    assert results[0]["final_weight"] == pytest.approx(
        results[1]["final_weight"],
        rel=1e-6,
        abs=1e-6,
    )
    assert results[0]["final_weight"] > 0.0
