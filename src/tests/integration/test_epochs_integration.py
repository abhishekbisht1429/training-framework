"""A run given in epochs, through the engine and spawned DDP workers."""

from __future__ import annotations

import json
import sys

import yaml

from training_framework.engine import Configurator, TrainingEngine
from tests.integration.test_training_flow_integration import (
    _ddp_session_config,
    _register_integration_components,
)


def test_spawned_ddp_run_given_in_epochs_runs_the_resolved_iterations(
        tmp_path, monkeypatch,
):
    _register_integration_components()
    output_dir = tmp_path / "rank-results"
    session_config = _ddp_session_config(tmp_path, output_dir)
    del session_config["session_config"]["max_iterations"]
    session_config["session_config"]["epochs"] = 2
    # 5 samples pad to 6 on 2 ranks; 2 epochs of 6 in global batches of 2.
    session_config["integration_dataset"]["dataset_size"] = 5
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({"sessions": [session_config]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "training-framework",
        "--config", str(config_path),
        "--heartbeat-timeout", "30",
        "--process_timeout_on_join", "10",
    ])

    with TrainingEngine(Configurator()) as engine:
        engine.start_session()

    for rank in range(2):
        result = json.loads(
            (output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")
        )
        assert [
            observation["iteration"] for observation in result["observations"]
        ] == [1, 2, 3, 4, 5, 6]
