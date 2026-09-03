"""Deterministic components for the real spawned DDP integration test."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import override

import torch
from torch import nn
from torch.utils.data import Dataset
from training_framework.components import (
    LifecycleHook,
    Resource,
    StatefulResource,
    Step,
    hook,
    requires_resource,
    requires_step,
    resource,
    step,
)
from training_framework.session import TrainingSession


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


@resource("integration_dataset")
class DeterministicDataset(Dataset, Resource):
    """Return stackable records containing index, input, and target."""

    def __init__(self, config: dict):
        self._dataset_size = int(config["dataset_size"])

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(
            [float(index), 1.0, float(index + 1)],
            dtype=torch.float32,
        )

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None


@resource("integration_data_context")
class IntegrationDataContext(Resource):
    """A process-group-free DDP-shaped resource for DataManager tests."""

    def __init__(self, config: dict):
        self._rank = int(config["rank"])
        self._world_size = int(config["world_size"])

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None


@resource("integration_worker_dataset")
class WorkerReportingDataset(Dataset, Resource):
    """Return each sample index and the process that loaded it."""

    def __init__(self, config: dict):
        self._dataset_size = int(config["dataset_size"])

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([index, os.getpid()], dtype=torch.int64)

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None


@resource("integration_collating_dataset")
class CollatingDataset(Dataset, Resource):
    def __init__(self, config: dict):
        self._dataset_size = int(config["dataset_size"])

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: int) -> int:
        return index

    def collate_fn(self, batch: list[int]) -> dict[str, object]:
        return {
            "indices": torch.tensor(batch, dtype=torch.int64),
            "collated": True,
        }

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None


@resource("integration_invalid_collate_dataset")
class InvalidCollateDataset(CollatingDataset):
    collate_fn = None


@step("integration_data")
@requires_resource("data_manager")
class DistributedDataLoadingStep(Step):
    """Publish the current DataManager batch to iteration context."""

    def __init__(self, config: dict):
        pass

    @override
    def run(self, session: TrainingSession) -> None:
        data_manager = session.get_resource("data_manager")
        batch = next(data_manager.data_iter)
        session.iteration_context["sample_index"] = int(batch[:, 0].item())
        session.iteration_context["inputs"] = batch[:, 1:2].to(session.device)
        session.iteration_context["targets"] = batch[:, 2:3].to(session.device)


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
    def pre_session(self, session: TrainingSession) -> None:
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
    def post_session(self, session: TrainingSession) -> None:
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
