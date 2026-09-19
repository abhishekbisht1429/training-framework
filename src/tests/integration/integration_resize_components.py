"""Fixtures for resuming a real run on a different number of processes.

Everything here builds on `integration_training_components`; only the two
components that assume a local batch of exactly one are replaced. Changing
the world size changes the local batch size, so those assumptions are what
stand in the way of resizing a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import override

from training_framework.components import (
    LifecycleHook,
    Step,
    requires_resource,
    hook,
    step,
)
from training_framework.session import TrainingSession

# Imported for its registrations: the model, dataset, training and loss
# steps are reused as they are.
from tests.integration import integration_training_components  # noqa: F401


@step("integration_data", overwrite=True)
@requires_resource("data_manager")
class ResizableDataLoadingStep(Step):
    """Publish the current batch, whatever local size the world gives it."""

    def __init__(self, config: dict):
        pass

    @override
    def run(self, session: TrainingSession) -> None:
        data_manager = session.get_resource("data_manager")
        batch = next(data_manager.data_iter)
        session.iteration_context["sample_indices"] = [
            int(value) for value in batch[:, 0].tolist()
        ]
        session.iteration_context["inputs"] = batch[:, 1:2].to(session.device)
        session.iteration_context["targets"] = batch[:, 2:3].to(session.device)


@hook("integration_results", overwrite=True)
@requires_resource("ddp")
class ResizableRankResultHook(LifecycleHook):
    """Record what each rank saw, without assuming one sample per batch."""

    def __init__(self, config: dict):
        self.call_every = 1
        self._output_dir = Path(config["output_dir"])
        self._observations: list[dict] = []

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
            "sample_indices": session.iteration_context["sample_indices"],
        })

    @override
    def post_session(self, session: TrainingSession) -> None:
        ddp = session.get_resource("ddp")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / f"rank_{ddp.rank}.json").write_text(
            json.dumps({
                "rank": ddp.rank,
                "world_size": ddp.world_size,
                "observations": self._observations,
            }),
            encoding="utf-8",
        )
