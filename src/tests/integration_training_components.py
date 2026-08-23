"""Deterministic components for the real spawned DDP integration test."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, override

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training_framework.dataloader import DistributedInfiniteSampler
from training_framework.training_session import (
    LifecycleHook,
    StatefulResource,
    Step,
    TrainingSession,
    hook,
    requires_resource,
    requires_step,
    resource,
    step,
)


@resource("integration_ddp_model")
class TinyLinearModel(nn.Module, StatefulResource):
    """A one-parameter model whose predictions are easy to inspect."""

    def __init__(self, config: dict):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(
            torch.tensor([[float(config.get("initial_weight", 0.0))]])
        )

    @override
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.weight

    @override
    def setup(self, session: TrainingSession) -> None:
        self.to(session.device)

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None

    @override
    def get_state(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight.detach().clone()}

    @override
    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])


@step("integration_data")
@requires_resource("ddp")
class DistributedDataLoadingStep(Step):
    """Load one deterministic rank-local sample per training iteration."""

    def __init__(self, config: dict):
        self._dataset_size = int(config["dataset_size"])
        self._loader_iterator: Any = None

    def _build_loader(self, session: TrainingSession) -> None:
        ddp = session.get_resource("ddp")
        sample_indices = torch.arange(self._dataset_size, dtype=torch.int64)
        inputs = torch.ones((self._dataset_size, 1), dtype=torch.float32)
        targets = (sample_indices + 1).to(torch.float32).unsqueeze(1)
        dataset = TensorDataset(sample_indices, inputs, targets)
        sampler = DistributedInfiniteSampler(
            num_samples=len(dataset),
            rank=ddp.rank,
            world_size=ddp.world_size,
            shuffle=False,
        )
        self._loader_iterator = iter(
            DataLoader(dataset, batch_size=1, sampler=sampler)
        )

    @override
    def run(self, session: TrainingSession) -> None:
        if self._loader_iterator is None:
            self._build_loader(session)

        sample_indices, inputs, targets = next(self._loader_iterator)
        session.iteration_context["sample_index"] = int(sample_indices.item())
        session.iteration_context["inputs"] = inputs.to(session.device)
        session.iteration_context["targets"] = targets.to(session.device)


@step("integration_train")
@requires_resource("ddp")
@requires_step("integration_data")
class DDPTrainingStep(Step):
    def __init__(self, config: dict):
        pass

    @override
    def run(self, session: TrainingSession) -> None:
        model = session.get_resource("ddp").wrapped_model
        session.iteration_context["prediction"] = model(
            session.iteration_context["inputs"]
        )


@step("integration_loss")
@requires_step("integration_train")
class MeanSquaredLossStep(Step):
    def __init__(self, config: dict):
        pass

    @override
    def run(self, session: TrainingSession) -> None:
        prediction = session.iteration_context["prediction"]
        target = session.iteration_context["targets"]
        session.iteration_context["loss"] = torch.nn.functional.mse_loss(
            prediction,
            target,
        )


@hook("integration_results")
@requires_resource("ddp")
class RankResultHook(LifecycleHook):
    """Write observable child-process results after distributed training."""

    def __init__(self, config: dict):
        self.call_every = 1
        self._output_dir = Path(config["output_dir"])
        self._observations: list[dict[str, float | int]] = []

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def pre_iteration_callback(self, session: TrainingSession) -> None:
        return None

    @override
    def post_iteration_callback(self, session: TrainingSession) -> None:
        self._observations.append({
            "iteration": session.iteration,
            "sample_index": session.iteration_context["sample_index"],
            "prediction": float(
                session.iteration_context["prediction"].detach().item()
            ),
            "target": float(session.iteration_context["targets"].item()),
            "loss": float(session.iteration_context["loss"].detach().item()),
        })

    @override
    def teardown(self, session: TrainingSession) -> None:
        ddp = session.get_resource("ddp")
        model = ddp.wrapped_model.module
        payload = {
            "rank": ddp.rank,
            "pid": os.getpid(),
            "final_weight": float(model.weight.detach().item()),
            "observations": self._observations,
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / f"rank_{ddp.rank}.json"
        output_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
