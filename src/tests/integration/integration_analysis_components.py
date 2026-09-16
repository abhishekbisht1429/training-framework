"""Importable components for spawned analysis-session integration tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from training_framework.components import (
    Resource,
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


@step("integration_analysis_weight_update", session_type="training")
@requires_resource("model")
class IntegrationWeightUpdate(Step):
    """Deterministically shift the model weight once per training iteration."""

    def __init__(self, config):
        self._delta = float(config["delta"])

    def run(self, session):
        model = session.get_resource("model")
        with torch.no_grad():
            model.weight.add_(self._delta)


@resource("integration_analysis_dataset", session_type="analysis")
class IntegrationAnalysisDataset(Dataset, Resource):
    def __init__(self, config):
        self._size = int(config["size"])

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return torch.tensor(float(index))

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@step("integration_analysis_batch_probe", session_type="analysis")
@requires_resource("data_manager")
@requires_resource("trained_model")
class IntegrationAnalysisBatchProbe(Step):
    """Record each batch drawn from the analysis data manager."""

    def __init__(self, config):
        self._output_path = Path(config["output_path"])

    def run(self, session):
        batch = next(session.get_resource("data_manager").data_iter)
        model = session.get_resource("trained_model").model
        with torch.no_grad():
            predictions = model(batch)
        payload = {
            "iteration": session.iteration,
            "batch": batch.tolist(),
            "predictions": predictions.tolist(),
        }
        with self._output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, sort_keys=True) + "\n")


@resource("integration_analysis_teardown_marker", session_type="analysis")
class IntegrationTeardownMarker(Resource):
    """Write a marker file when the worker tears the session down."""

    def __init__(self, config):
        self._marker_path = Path(config["marker_path"])

    def setup(self, session):
        pass

    def teardown(self, session):
        self._marker_path.write_text(
            json.dumps({"iteration": session.iteration, "pid": os.getpid()}),
            encoding="utf-8",
        )


@step("integration_analysis_fail", session_type="analysis")
class IntegrationAnalysisFail(Step):
    def __init__(self, config):
        self._fail_at = int(config["fail_at"])

    def run(self, session):
        if session.iteration == self._fail_at:
            raise RuntimeError("analysis step exploded")
