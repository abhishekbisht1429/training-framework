"""Fine-tuning on two real gloo ranks, spawned by the engine.

A pretraining run is checkpointed; the fine-tuning run imports its model
(`import_components`), takes the encoder through `module_part`, freezes the
first block, and trains a head on two ranks that see different data. Both
ranks must end with the same weights, equal to the same fine-tuning written
in plain torch with the gradient averaged over the ranks, and the frozen
block must be untouched. The source checkpoint is moved away once the
engine has built the sessions and before it starts the ranks, so they run
without it.
"""

from __future__ import annotations

import copy
import importlib
import json
import shutil
import sys

import pytest
import torch
import yaml
from torch.nn import functional

from tests.test_utils import resource_named
from training_framework.components.builtin import Checkpointer
from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import TrainingSession

_COMPONENTS_PACKAGE = "tests.integration.integration_fine_tuning_components"
ITERATIONS = 3
LR = 0.05
FROZEN = [
    "backbone.module.0.weight", "backbone.module.0.bias",
    "backbone.module.1.weight", "backbone.module.1.bias",
]

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


def _session_settings(tmp_path, name):
    return {
        "rng_seed": 31,
        "sessions_dir": str(tmp_path / name),
        "max_iterations": ITERATIONS,
        "device": "cpu",
        "components_package": _COMPONENTS_PACKAGE,
        "show_execution_graph": False,
    }


def _pretrain(tmp_path):
    session = TrainingSession({
        "session_config": _session_settings(tmp_path, "pretrain"),
        "component_bindings": {"model": "ift_pretrain_model"},
        "ift_pretrain_model": {},
    })
    return Checkpointer.save_checkpoint(session, tmp_path / "pretrained")


def _fine_tuning_config(tmp_path, checkpoint, output_dir):
    return {
        "session_config": _session_settings(tmp_path, "fine-tune"),
        "import_components": {
            "pretrained": {"checkpoint": str(checkpoint), "role": "source"},
        },
        "component_bindings": {
            "model": "fine_tuned_model",
            "backbone": "module_part",
            "head": "ift_head",
        },
        "module_part": {"submodule": "encoder"},
        "fine_tuned_model": {"frozen": ["module.0.*", "module.1.*"]},
        "ift_head": {},
        "ift_loss": {},
        "ift_results": {"output_dir": str(output_dir), "source": str(checkpoint)},
        "ddp": {"world_size": 2, "backend": "gloo", "master_addr": "127.0.0.1"},
        "optimizer": {"optimizer": {"name": "AdamW", "kwargs": {"lr": LR}}},
    }


def test_fine_tuning_on_two_ranks(tmp_path, monkeypatch):
    components = _components()
    checkpoint = _pretrain(tmp_path)
    output_dir = tmp_path / "rank-results"
    config = _fine_tuning_config(tmp_path, checkpoint, output_dir)

    # Initial weights, from the same configuration built in this process:
    # construction is seeded, so the workers build the same ones.
    # Never set up, so its placeholder port is never bound.
    local = copy.deepcopy(config)
    local["ddp"]["master_port"] = "12355"
    initial = resource_named(TrainingSession(local), "fine_tuned_model")
    backbone = copy.deepcopy(initial.backbone.module)
    head = copy.deepcopy(initial.head.linear)
    running_mean = backbone[1].running_mean.clone()

    config_path = tmp_path / "fine-tune.yaml"
    config_path.write_text(yaml.safe_dump({"sessions": [config]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "training-framework", "--config", str(config_path),
        "--heartbeat-timeout", "30", "--process_timeout_on_join", "10",
    ])
    with TrainingEngine(Configurator()) as engine:
        # The sessions are built; the ranks have not started.
        shutil.move(checkpoint, tmp_path / "moved-source")
        engine.start_session()

    results = [
        json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]

    # The ranks ran without the source checkpoint.
    assert not any(result["source_exists"] for result in results)

    # Both ranks hold the same model, with the first block frozen.
    assert results[0]["parameters"] == results[1]["parameters"]
    assert [name for name, trainable in results[0]["trainable"].items()
            if not trainable] == FROZEN
    assert results[0]["running_mean"] == pytest.approx(running_mean.tolist())

    # Plain torch: DDP averages gradients, i.e. the mean of the ranks' losses.
    for parameter in [*backbone[0].parameters(), *backbone[1].parameters()]:
        parameter.requires_grad_(False)
    backbone.train()
    backbone[0].eval()
    backbone[1].eval()
    trainable = [
        p for p in [*backbone.parameters(), *head.parameters()] if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    for _ in range(ITERATIONS):
        optimizer.zero_grad()
        loss = sum(
            functional.cross_entropy(
                head(backbone(components.rank_inputs(rank))), components.TARGETS,
            )
            for rank in range(2)
        ) / 2
        loss.backward()
        optimizer.step()

    expected = {
        **{f"backbone.module.{name}": value
           for name, value in backbone.named_parameters()},
        **{f"head.linear.{name}": value for name, value in head.named_parameters()},
    }
    assert results[0]["parameters"].keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(
            torch.tensor(results[0]["parameters"][name]), value.detach(),
        )
    for name in FROZEN:
        torch.testing.assert_close(
            torch.tensor(results[0]["parameters"][name]),
            initial.get_parameter(name).detach(),
        )
