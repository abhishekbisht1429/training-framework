"""DINO and BYOL on two real gloo ranks, spawned by the engine.

Each rank reads its own share of the synthetic multi-view data. DDP must
keep the students identical across ranks, the teachers must follow them
identically (the update runs on every rank, from the same student), and
training must move the weights.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys

import pytest
import torch
import yaml

from tests.test_utils import resource_named
from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import TrainingSession

_COMPONENTS_PACKAGE = "tests.integration.integration_ssl_components"
ITERATIONS = 3

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="The real DDP integration test requires PyTorch Gloo support",
)


def _components():
    existing = sys.modules.get(_COMPONENTS_PACKAGE)
    if existing is None:
        return importlib.import_module(_COMPONENTS_PACKAGE)
    return importlib.reload(existing)


@pytest.mark.parametrize("recipe", ["dino", "byol"])
def test_self_supervised_training_on_two_ranks(tmp_path, monkeypatch, recipe):
    components = _components()
    build = components.dino_config if recipe == "dino" else components.byol_config
    output_dir = tmp_path / "rank-results"
    config = build(tmp_path / "sessions", max_iterations=ITERATIONS, world_size=2)
    del config["ddp"]["master_port"]
    config["issl_results"] = {"output_dir": str(output_dir)}

    # The initial weights, from the same configuration built here:
    # construction is seeded, so every rank builds the same ones.
    local = copy.deepcopy(config)
    local["ddp"]["master_port"] = "12355"
    initial = {
        name: value.detach().clone()
        for name, value in resource_named(
            TrainingSession(local), "ema_teacher",
        ).state_dict().items()
    }

    config_path = tmp_path / "ssl.yaml"
    config_path.write_text(yaml.safe_dump({"sessions": [config]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "training-framework", "--config", str(config_path),
        "--heartbeat-timeout", "60", "--process_timeout_on_join", "10",
    ])
    with TrainingEngine(Configurator()) as engine:
        engine.start_session()

    ranks = [
        json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8"))
        ["parameters"]
        for rank in range(2)
    ]

    assert ranks[0].keys() == initial.keys()
    for name in initial:
        rank0 = torch.tensor(ranks[0][name])
        assert torch.isfinite(rank0.float()).all(), name
        if "running_" in name or name.endswith("num_batches_tracked"):
            # BatchNorm statistics are per rank without SyncBatchNorm: DDP
            # copies rank 0's at each forward, then each rank's own batch
            # updates them.
            continue
        torch.testing.assert_close(rank0, torch.tensor(ranks[1][name]), msg=name)
    for side in ("student.", "teacher."):
        assert any(
            not torch.equal(torch.tensor(ranks[0][name]), initial[name])
            for name in initial
            if name.startswith(side) and initial[name].is_floating_point()
        ), f"the {side[:-1]} did not train"
