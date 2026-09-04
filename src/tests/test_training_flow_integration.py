from __future__ import annotations

import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from training_framework.engine import Configurator
from training_framework.engine import SessionProcessWrapper, TrainingEngine


_COMPONENTS_PACKAGE = "tests.integration_training_components"
_INTEGRATION_RUNNER = "tests.integration_training_runner"


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


def _wait_for_paths(paths, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for DDP workers to become ready")
        time.sleep(0.01)


def _wait_for_process_paths(
    process,
    paths,
    log_path,
    timeout: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        returncode = process.poll()
        if returncode is not None:
            pytest.fail(
                f"Training process exited with code {returncode} before "
                f"workers made progress:\n{log_path.read_text(encoding='utf-8')}"
            )
        if time.monotonic() >= deadline:
            pytest.fail(
                "Timed out waiting for training workers to make progress:\n"
                f"{log_path.read_text(encoding='utf-8')}"
            )
        time.sleep(0.01)


def _ddp_session_config(
    tmp_path,
    output_dir,
    *,
    max_iterations: int = 3,
    ready_dir=None,
):
    session_config = {
        "session_config": {
            "rng_seed": 17,
            "sessions_dir": str(tmp_path / "sessions"),
            "max_iterations": max_iterations,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show_execution_graph": False,
        },
        "component_bindings": {
            "model": "integration_ddp_model",
            "dataset": "integration_dataset",
        },
        "integration_ddp_model": {
            "initial_weight": 0.0,
        },
        "integration_dataset": {
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
    if ready_dir is not None:
        session_config["ddp"]["parallel_components"].append(
            "integration_ready"
        )
        session_config["integration_ready"] = {
            "ready_dir": str(ready_dir),
        }
    return session_config


def _run_sigint_training_flow(tmp_path, *, process_group: bool) -> None:
    max_iterations = 100_000
    output_dir = tmp_path / "rank-results"
    progress_dir = tmp_path / "rank-progress"
    session_config = _ddp_session_config(
        tmp_path,
        output_dir,
        max_iterations=max_iterations,
    )
    session_config["integration_results"]["progress_dir"] = str(progress_dir)
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({"sessions": [session_config]}),
        encoding="utf-8",
    )
    log_path = tmp_path / "training.log"
    command = [
        sys.executable,
        "-m",
        _INTEGRATION_RUNNER,
        "--config",
        str(config_path),
        "--heartbeat-timeout",
        "30",
        "--process_timeout_on_join",
        "10",
    ]
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[1])
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (source_root, inherited_pythonpath)
        if path
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_process_paths(
                process,
                [
                    progress_dir / "rank_0.progress",
                    progress_dir / "rank_1.progress",
                ],
                log_path,
            )
            if process_group:
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.send_signal(signal.SIGINT)
            returncode = process.wait(timeout=30.0)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5.0)

    process_log = log_path.read_text(encoding="utf-8")
    assert returncode == 0, process_log
    results = [
        json.loads(
            (output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ]
    completed_iterations = [
        [observation["iteration"] for observation in result["observations"]]
        for result in results
    ]
    assert completed_iterations[0]
    assert completed_iterations[0] == completed_iterations[1]
    assert len(completed_iterations[0]) < max_iterations
    assert results[0]["final_weight"] == pytest.approx(
        results[1]["final_weight"],
        rel=1e-6,
        abs=1e-6,
    )


@pytest.mark.skipif(
    os.name != "posix",
    reason="SIGINT process integration requires POSIX process groups",
)
def test_sigint_to_supervisor_gracefully_stops_full_training(tmp_path):
    _run_sigint_training_flow(tmp_path, process_group=False)


@pytest.mark.skipif(
    os.name != "posix",
    reason="SIGINT process integration requires POSIX process groups",
)
def test_sigint_to_process_group_gracefully_stops_full_training(tmp_path):
    _run_sigint_training_flow(tmp_path, process_group=True)


def test_full_spawned_ddp_training_flow(tmp_path, monkeypatch):
    _register_integration_components()
    output_dir = tmp_path / "rank-results"
    session_config = _ddp_session_config(tmp_path, output_dir)
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


def test_staggered_stop_request_keeps_ddp_ranks_iteration_aligned(
    tmp_path,
    monkeypatch,
):
    _register_integration_components()
    output_dir = tmp_path / "stopped-rank-results"
    ready_dir = tmp_path / "rank-ready"
    session_config = _ddp_session_config(
        tmp_path,
        output_dir,
        max_iterations=100_000,
        ready_dir=ready_dir,
    )
    config_path = tmp_path / "stopped-training.yaml"
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

    original_request_stop = SessionProcessWrapper.request_stop
    delayed_ranks = set()

    def staggered_request_stop(wrapper):
        original_request_stop(wrapper)
        if wrapper.rank == 0 and wrapper.rank not in delayed_ranks:
            delayed_ranks.add(wrapper.rank)
            time.sleep(1.0)

    monkeypatch.setattr(
        SessionProcessWrapper,
        "request_stop",
        staggered_request_stop,
    )

    with TrainingEngine(Configurator()) as engine:
        engine.start_session()
        _wait_for_paths([
            ready_dir / "rank_0.ready",
            ready_dir / "rank_1.ready",
        ])
        engine.request_stop_all()

    results = [
        json.loads(
            (output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(2)
    ]
    completed_iterations = [
        [observation["iteration"] for observation in result["observations"]]
        for result in results
    ]

    assert completed_iterations[0] == completed_iterations[1]
    assert len(completed_iterations[0]) < 100_000
    assert results[0]["final_weight"] == pytest.approx(
        results[1]["final_weight"],
        rel=1e-6,
        abs=1e-6,
    )
