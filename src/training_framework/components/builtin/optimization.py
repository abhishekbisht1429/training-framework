from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch
from torch import nn, optim

from training_framework.components import StatefulLifeCycleHook
from training_framework.components import hook, requires_resource

if TYPE_CHECKING:
    from training_framework.session import Session


@hook("optimizer", session_type="training")
@requires_resource("ddp")
class OptimizerHook(StatefulLifeCycleHook):

    def __init__(self, config):
        self.call_every = 1
        self._learning_rate = config["learning_rate"]
        self._weight_decay = config["weight_decay"]
        self._warmup_iters = config["warmup_iters"]
        self._optimizer = None
        self._lr_scheduler = None
        self._restored_state = None

    def _prepare_scheduler(self, optimizer, max_iter):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            total_iters=self._warmup_iters,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iter - self._warmup_iters,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self._warmup_iters],
        )

    @override
    def setup(self, session: Session):
        ddp_model: nn.Module = session.get_resource("ddp")
        self._optimizer = optim.AdamW(
            ddp_model.wrapped_model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        self._lr_scheduler = self._prepare_scheduler(
            self._optimizer,
            session.session_config.max_iterations,
        )
        self._restore_state()

    def _restore_state(self):
        if self._restored_state is None:
            return

        optimizer_state = self._restored_state.get("optimizer_state")
        if optimizer_state is not None:
            self._optimizer.load_state_dict(optimizer_state)

        scheduler_state = self._restored_state.get("lr_scheduler_state")
        if scheduler_state is not None:
            self._lr_scheduler.load_state_dict(scheduler_state)

        self._restored_state = None

    @override
    def pre_iteration_callback(self, session: Session) -> None:
        self._optimizer.zero_grad()

    @override
    def post_iteration_callback(self, session: Session) -> None:
        loss = session.iteration_context["loss"]
        loss.backward()
        self._optimizer.step()
        self._lr_scheduler.step()

    @override
    def teardown(self, session: Session):
        self._restored_state = self.get_state()
        self._optimizer = None
        self._lr_scheduler = None

    @override
    def set_state(self, state: Any) -> None:
        self._restored_state = deepcopy(state)
        if self._optimizer is not None and self._lr_scheduler is not None:
            self._restore_state()

    @override
    def get_state(self) -> Any:
        if self._optimizer is None and self._lr_scheduler is None:
            if self._restored_state is not None:
                return deepcopy(self._restored_state)
            return {
                "optimizer_state": None,
                "lr_scheduler_state": None,
            }
        return {
            "optimizer_state": (
                self._optimizer.state_dict()
                if self._optimizer
                else None
            ),
            "lr_scheduler_state": (
                self._lr_scheduler.state_dict()
                if self._lr_scheduler
                else None
            ),
        }
