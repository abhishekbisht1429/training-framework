"""Components for the spawned fine-tuning integration test."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from training_framework.components import (
    ModuleResource,
    SessionHook,
    Step,
    hook,
    requires_resource,
    resource,
    step,
    writes,
)

INPUTS = torch.tensor([
    [1.0, -0.5, 0.25, 2.0],
    [-1.0, 0.5, 1.5, -0.25],
    [0.5, 0.5, -1.0, 1.0],
])
TARGETS = torch.tensor([0, 2, 1])


def rank_inputs(rank: int) -> torch.Tensor:
    """Each rank sees different data, so DDP's averaging matters."""
    return INPUTS * (rank + 1)


@resource("ift_pretrain_model")
class PretrainModel(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(4, 6),     # encoder.0
            nn.BatchNorm1d(6),   # encoder.1
            nn.ReLU(),
            nn.Linear(6, 5),     # encoder.3
        )
        self.pretext_head = nn.Linear(5, 7)

    def forward(self, x):
        return self.pretext_head(self.encoder(x))


@resource("ift_head")
class Head(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(5, 3)

    def forward(self, x):
        return self.linear(x)


@step("ift_loss")
@requires_resource("ddp")
@writes("loss")
class FineTuneLoss(Step):
    def run(self, session):
        ddp = self.get_dependency("ddp")
        return functional.cross_entropy(
            ddp.wrapped_model(rank_inputs(ddp.rank)), TARGETS,
        )


@hook("ift_results")
@requires_resource("ddp")
class RankResults(SessionHook):
    """Write each rank's final weights, trainability and BatchNorm statistics."""

    def __init__(self, config):
        self._output_dir = Path(config["output_dir"])
        self._source = Path(config["source"])

    def pre_session(self, session):
        return None

    def post_session(self, session):
        ddp = self.get_dependency("ddp")
        model = ddp.wrapped_model.module
        batch_norm = model.backbone.module[1]
        payload = {
            "rank": ddp.rank,
            "parameters": {
                name: parameter.detach().tolist()
                for name, parameter in model.named_parameters()
            },
            "trainable": {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            },
            "running_mean": batch_norm.running_mean.tolist(),
            "source_exists": self._source.exists(),
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / f"rank_{ddp.rank}.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
