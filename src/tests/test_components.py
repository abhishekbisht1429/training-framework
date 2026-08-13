"""Small, real components used by the commit-3d45 integration tests.

The module must remain importable by name because ``spawn`` starts a fresh
interpreter and ``TrainingSession`` imports ``components_package`` there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, override

import torch

from training_framework.training_session import (
    LifecycleHook,
    Resource,
    StatefulLifeCycleHook,
    StatefulResource,
    StatefulStep,
    Step,
    TrainingSession,
    hook,
    requires_resource,
    resource,
    step,
)


def _append_event(path: str | None, event: str, **payload: Any) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "pid": os.getpid(), **payload}
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


@resource("it_3d45_model")
class StatefulModelResource(StatefulResource):
    """A scalar model plus a momentum optimizer, both checkpointable."""

    def __init__(self, config: dict):
        self.config = dict(config)
        dtype = torch.float64
        self.weight = torch.nn.Parameter(
            torch.tensor(float(self.config.get("initial_weight", 0.25)), dtype=dtype)
        )
        self.optimizer = torch.optim.SGD(
            [self.weight],
            lr=float(self.config.get("learning_rate", 0.08)),
            momentum=float(self.config.get("momentum", 0.9)),
        )
        self.setup_count = 0
        self.teardown_count = 0

    @override
    def setup(self, session: TrainingSession) -> None:
        self.setup_count += 1
        _append_event(
            self.config.get("event_path"),
            "model_setup",
            iteration=session.iteration,
            setup_count=self.setup_count,
        )

    @override
    def teardown(self, session: TrainingSession) -> None:
        self.teardown_count += 1
        _append_event(
            self.config.get("event_path"),
            "model_teardown",
            iteration=session.iteration,
            teardown_count=self.teardown_count,
        )

    @override
    def get_state(self) -> dict[str, Any]:
        return {
            "weight": self.weight.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "setup_count": self.setup_count,
            "teardown_count": self.teardown_count,
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.setup_count = int(state["setup_count"])
        self.teardown_count = int(state["teardown_count"])


@step("it_3d45_train")
@requires_resource("it_3d45_model")
class StatefulTrainingStep(StatefulStep):
    """A genuine autograd/optimizer step with a stochastic target."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.weight_history: list[float] = []
        self.noise_history: list[float] = []

    @override
    def run(self, session: TrainingSession) -> None:
        model: StatefulModelResource = session.get_resource("it_3d45_model")
        noise_scale = float(self.config.get("noise_scale", 0.05))
        noise = float(torch.rand((), dtype=model.weight.dtype).item()) * noise_scale
        target = torch.tensor(
            float(session.iteration) + noise,
            dtype=model.weight.dtype,
            device=model.weight.device,
        )

        model.optimizer.zero_grad(set_to_none=True)
        loss = (model.weight - target).square()
        loss.backward()
        model.optimizer.step()

        weight = float(model.weight.detach().item())
        loss_value = float(loss.detach().item())
        self.weight_history.append(weight)
        self.noise_history.append(noise)
        session.iteration_context["weight"] = weight
        session.iteration_context["loss"] = loss_value
        session.session_context["last_weight"] = weight

        _append_event(
            self.config.get("event_path"),
            "iteration",
            iteration=session.iteration,
            weight=weight,
            loss=loss_value,
            noise=noise,
        )

    @override
    def get_state(self) -> dict[str, Any]:
        return {
            "weight_history": list(self.weight_history),
            "noise_history": list(self.noise_history),
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:
        self.weight_history = list(state["weight_history"])
        self.noise_history = list(state["noise_history"])


@hook("it_3d45_metrics")
@requires_resource("it_3d45_model")
class StatefulMetricsHook(StatefulLifeCycleHook):
    def __init__(self, config: dict):
        self.config = dict(config)
        self.call_every = int(self.config.get("call_every", 1))
        self.setup_count = 0
        self.teardown_count = 0
        self.observations: list[dict[str, float | int]] = []

    @override
    def setup(self, session: TrainingSession) -> None:
        self.setup_count += 1
        _append_event(
            self.config.get("event_path"),
            "metrics_setup",
            iteration=session.iteration,
            setup_count=self.setup_count,
        )

    @override
    def teardown(self, session: TrainingSession) -> None:
        self.teardown_count += 1
        _append_event(
            self.config.get("event_path"),
            "metrics_teardown",
            iteration=session.iteration,
            teardown_count=self.teardown_count,
        )

    @override
    def pre_iteration_callback(self, session: TrainingSession) -> None:
        return None

    @override
    def post_iteration_callback(self, session: TrainingSession) -> None:
        observation = {
            "iteration": session.iteration,
            "weight": float(session.iteration_context["weight"]),
            "loss": float(session.iteration_context["loss"]),
        }
        self.observations.append(observation)
        _append_event(
            self.config.get("event_path"),
            "metrics",
            **observation,
        )

    @override
    def get_state(self) -> dict[str, Any]:
        return {
            "setup_count": self.setup_count,
            "teardown_count": self.teardown_count,
            "observations": list(self.observations),
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:
        self.setup_count = int(state["setup_count"])
        self.teardown_count = int(state["teardown_count"])
        self.observations = list(state["observations"])


@step("it_3d45_fail")
class FailingStep(Step):
    def __init__(self, config: dict):
        self.config = dict(config)

    @override
    def run(self, session: TrainingSession) -> None:
        fail_at = int(self.config.get("fail_at", 1))
        if session.iteration == fail_at:
            raise RuntimeError(self.config.get("message", "intentional worker failure"))


@resource("it_3d45_rank0_resource")
class RankZeroOnlyResource(Resource):
    def __init__(self, config: dict):
        self.config = dict(config)

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None


@step("it_3d45_rank0_step")
class RankZeroOnlyStep(Step):
    def __init__(self, config: dict):
        self.config = dict(config)

    @override
    def run(self, session: TrainingSession) -> None:
        return None


@hook("it_3d45_rank0_hook")
class RankZeroOnlyHook(LifecycleHook):
    def __init__(self, config: dict):
        self.config = dict(config)
        self.call_every = int(self.config.get("call_every", 1))

    @override
    def setup(self, session: TrainingSession) -> None:
        return None

    @override
    def teardown(self, session: TrainingSession) -> None:
        return None

    @override
    def pre_iteration_callback(self, session: TrainingSession) -> None:
        return None

    @override
    def post_iteration_callback(self, session: TrainingSession) -> None:
        return None
