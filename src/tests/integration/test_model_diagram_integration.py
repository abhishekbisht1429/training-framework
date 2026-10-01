"""`model_diagram` in a real spawned two-rank gloo run: rank 0 draws the
model from its first training forward call; rank 1 draws nothing."""

from __future__ import annotations

import sys

import pytest
import torch
import yaml

from tests.components.builtin.test_model_diagram import read_diagram
from tests.integration.test_training_flow_integration import (
    _ddp_session_config,
    _register_integration_components,
)
from training_framework.engine import Configurator, TrainingEngine

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="The real DDP integration test requires PyTorch Gloo support",
)


def test_a_ddp_run_draws_the_model_once(tmp_path, monkeypatch):
    _register_integration_components()
    config = _ddp_session_config(tmp_path, tmp_path / "rank-results")
    config["model_diagram"] = {"formats": ["svg"]}
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump({"sessions": [config]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "training-framework", "--config", str(config_path),
        "--heartbeat-timeout", "30", "--process_timeout_on_join", "10",
    ])

    with TrainingEngine(Configurator()) as engine:
        engine.start_session()

    (session_dir,) = (tmp_path / "sessions").iterdir()
    names = sorted(path.name for path in session_dir.glob("model_diagram*"))
    assert names == [
        f"model_diagram{part}.{ext}"
        for part in ("", "_components", "_session")
        for ext in ("dot", "mmd", "svg")
    ]
    model = read_diagram(session_dir / "model_diagram.dot")
    assert model.named("output")
    # The session wiring: data from writer to reader, dashed, by key; a
    # companion dotted; a requirement labelled with its role.
    wiring = read_diagram(session_dir / "model_diagram_session.dot")
    assert wiring.edges_between("integration_loss", "backward") == [("loss", "dashed")]
    assert wiring.edges_between("optimizer", "optimizer_step") == [("activates", "dotted")]
    assert wiring.edges_between("data_manager", "integration_dataset") == [("dataset", "solid")]
    assert {"Resources", "Hooks", "Steps"} <= set(wiring.clusters)
