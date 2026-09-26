"""End-to-end analysis flows through TrainingEngine and spawned workers."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from training_framework.engine import SessionProcessWrapper, TrainingEngine
from training_framework.session import AnalysisSession, TrainingSession


_COMPONENTS_PACKAGE = "tests.integration.integration_analysis_components"
_INITIAL_WEIGHT = 1.5
_DELTA = 0.25


@dataclass
class _EngineConfig:
    mode: str
    session_configs: tuple[dict[str, Any], ...] = ()
    process_timeout_on_join: float = 10.0
    heartbeat_timeout: float = 30.0
    stop_sync_grace_period: float = 0.01
    stop_sync_poll_interval: float = 0.005
    debug: bool = False
    checkpoint_path: str | None = None
    new_max_iters: int | None = None


def _register_components() -> None:
    # conftest resets registries per test; re-run the module's decorators.
    existing = sys.modules.get(_COMPONENTS_PACKAGE)
    if existing is None:
        importlib.import_module(_COMPONENTS_PACKAGE)
    else:
        importlib.reload(existing)


def _session_config(root: Path, max_iterations: int) -> dict[str, Any]:
    return {
        "rng_seed": 17,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": _COMPONENTS_PACKAGE,
        "show_execution_graph": False,
    }


def _training_session_config(tmp_path: Path, max_iterations: int) -> dict:
    return {
        "session_config": _session_config(
            tmp_path / "training", max_iterations
        ),
        "component_bindings": {"model": "integration_analysis_model"},
        "integration_analysis_model": {"weight": _INITIAL_WEIGHT},
        "integration_analysis_weight_update": {"delta": _DELTA},
        "checkpointer": {
            "checkpoint_every": 1000,
            "checkpoints_dir": str(tmp_path / "checkpoints"),
        },
    }


def _saved_training_checkpoint(tmp_path: Path) -> Path:
    """Write a checkpoint in the same format Checkpointer produces."""
    session = TrainingSession(_training_session_config(tmp_path, 1))
    path = tmp_path / "training-session.pt"
    torch.save(session, path)
    return path


def _analysis_definition(
        tmp_path: Path,
        checkpoint_path: Path,
        *,
        max_iterations: int,
        **components: dict,
) -> dict[str, Any]:
    return {
        "session_type": "analysis",
        "session_config": _session_config(
            tmp_path / "analysis", max_iterations
        ),
        "trained_model": {"model_checkpoint_path": str(checkpoint_path)},
        "integration_analysis_teardown_marker": {
            "marker_path": str(tmp_path / "teardown.json"),
        },
        **components,
    }


def _run_engine(*definitions: dict[str, Any], **engine_options) -> None:
    config = _EngineConfig(
        mode="new",
        session_configs=tuple(definitions),
        **engine_options,
    )
    with TrainingEngine(config) as engine:
        engine.start_session()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _teardown_marker(tmp_path: Path) -> dict[str, Any]:
    marker = tmp_path / "teardown.json"
    assert marker.exists(), "analysis worker did not tear down its resources"
    return json.loads(marker.read_text(encoding="utf-8"))


def test_trained_checkpoint_flows_into_spawned_analysis_session(tmp_path):
    _register_components()
    training_iterations = 3
    _run_engine(
        {
            "session_type": "training",
            **_training_session_config(tmp_path, training_iterations),
        }
    )
    [checkpoint] = sorted((tmp_path / "checkpoints").iterdir())

    output_path = tmp_path / "probe.jsonl"
    _run_engine(
        _analysis_definition(
            tmp_path,
            checkpoint,
            max_iterations=4,
            integration_analysis_probe={"output_path": str(output_path)},
        )
    )

    records = _read_jsonl(output_path)
    trained_weight = _INITIAL_WEIGHT + training_iterations * _DELTA
    assert [record["iteration"] for record in records] == [1, 2, 3, 4]
    assert [record["prediction"] for record in records] == pytest.approx(
        [2.0 * trained_weight] * 4
    )
    assert all(record["training"] is False for record in records)

    marker = _teardown_marker(tmp_path)
    assert marker["iteration"] == 4
    assert marker["pid"] != os.getpid()


def test_spawned_analysis_consumes_dataset_and_stops_when_exhausted(tmp_path):
    _register_components()
    checkpoint = _saved_training_checkpoint(tmp_path)
    output_path = tmp_path / "batches.jsonl"

    _run_engine(
        _analysis_definition(
            tmp_path,
            checkpoint,
            max_iterations=100,
            component_bindings={"dataset": "integration_analysis_dataset"},
            integration_analysis_dataset={"size": 5},
            data_manager={"batch_size": 2},
            integration_analysis_batch_probe={
                "output_path": str(output_path),
            },
        )
    )

    records = _read_jsonl(output_path)
    assert [record["batch"] for record in records] == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0],
    ]
    for record in records:
        assert record["predictions"] == pytest.approx(
            [_INITIAL_WEIGHT * value for value in record["batch"]]
        )
    # The worker ended on data exhaustion, well before max_iterations, and
    # still tore down cleanly.
    assert _teardown_marker(tmp_path)["iteration"] == 3


def test_spawned_analysis_failure_reports_rank_and_tears_down(tmp_path, capfd):
    _register_components()
    checkpoint = _saved_training_checkpoint(tmp_path)

    with pytest.raises(RuntimeError) as raised:
        _run_engine(
            _analysis_definition(
                tmp_path,
                checkpoint,
                max_iterations=5,
                integration_analysis_fail={"fail_at": 2},
            )
        )

    message = str(raised.value)
    assert "analysis step exploded" in message
    assert "'rank': 0" in message
    assert "KeyError" not in message
    # The failed iteration is rolled back, so teardown sees the last
    # completed one.
    assert _teardown_marker(tmp_path)["iteration"] == 1
    # The worker reports once; a second send used to hit the closed pipe.
    assert "BrokenPipeError" not in capfd.readouterr().err


def test_spawned_analysis_rejects_non_training_checkpoint(tmp_path):
    _register_components()
    training_checkpoint = _saved_training_checkpoint(tmp_path)
    analysis_checkpoint = tmp_path / "analysis-session.pt"
    torch.save(
        AnalysisSession({
            "session_config": _session_config(tmp_path / "source", 1),
            "trained_model": {
                "model_checkpoint_path": str(training_checkpoint),
            },
        }),
        analysis_checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="must contain a training session",
    ):
        _run_engine(
            _analysis_definition(
                tmp_path,
                analysis_checkpoint,
                max_iterations=1,
            )
        )


def test_spawned_analysis_worker_honors_stop_request(tmp_path):
    _register_components()
    checkpoint = _saved_training_checkpoint(tmp_path)
    output_path = tmp_path / "probe.jsonl"
    max_iterations = 1_000_000
    definition = _analysis_definition(
        tmp_path,
        checkpoint,
        max_iterations=max_iterations,
        integration_analysis_probe={"output_path": str(output_path)},
    )
    definition.pop("session_type")
    wrapper = SessionProcessWrapper(
        AnalysisSession(definition),
        rank=0,
        heartbeat_timeout=30.0,
    )

    wrapper.start()
    try:
        wrapper.request_stop()
        wrapper.process.join(timeout=30.0)
        assert not wrapper.process.is_alive(), "worker ignored stop request"
    finally:
        if wrapper.process.is_alive():
            wrapper.process.kill()
            wrapper.process.join(timeout=5.0)
    assert wrapper.process.exitcode == 0
    wrapper.join()

    marker = _teardown_marker(tmp_path)
    assert marker["iteration"] < max_iterations
