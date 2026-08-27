"""Importable components for spawned analysis-session integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from training_framework.components import (
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)


@resource("integration_analysis_model")
class IntegrationAnalysisModel(nn.Module, StatefulResource):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(float(config["weight"])))

    def forward(self, value):
        return self.weight * value

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return self.state_dict()

    def set_state(self, state):
        self.load_state_dict(state)


@step("integration_analysis_probe", session_type="analysis")
@requires_resource("trained_model")
class IntegrationAnalysisProbe(Step):
    def __init__(self, config):
        self._output_path = Path(config["output_path"])

    def run(self, session):
        model = session.get_resource("trained_model").model
        payload = {
            "iteration": session.iteration,
            "prediction": float(model(torch.tensor(2.0)).detach()),
            "training": model.training,
            "grad_enabled": torch.is_grad_enabled(),
        }
        with self._output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, sort_keys=True) + "\n")
