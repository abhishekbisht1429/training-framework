"""`dino_distillation_loss`: DINO's self-distillation loss, and the centre it keeps.

The teacher's outputs are centred (a running mean of them is subtracted)
and sharpened (a low temperature) before they become the student's targets;
centring keeps one dimension from dominating, sharpening keeps the targets
from going uniform. The centre is state: checkpointed, and averaged over
every rank's batch so all ranks keep the same one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import torch
from torch import distributed

from training_framework.components import (
    TRAINING_SESSION_TYPE,
    StatefulStep,
    requires_resource,
    step,
)
from training_framework.components.builtin.self_supervised.losses import (
    dino_cross_entropy,
)
from training_framework.components.builtin.self_supervised.schedules import (
    Schedule,
    check_range,
    parse_schedule,
)
from training_framework.components.builtin.transformer.modules import (
    _positive_float,
    _positive_int,
    _probability,
)

if TYPE_CHECKING:
    from training_framework.session import Session


def _key(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{name} must be a non-empty iteration_context key; got {value!r}"
        )
    return value


@dataclass
class DINOLossConfig:
    """`out_dim` is the head's output width (the centre's size).
    `num_student_views` and `num_teacher_views` say how the stacked outputs
    split into views; they are checked against every batch, never inferred.

    `teacher_temp` is a schedule (see `schedules`); DINO warms it up
    linearly, e.g. `{start: 0.04, end: 0.07, kind: linear, steps: <30
    epochs of steps>}`. `center_momentum` is the centre's EMA momentum.

    `metrics_key` (default: none) also writes, under that key, a mapping of
    `teacher_temp`, `center_norm` and `teacher_entropy` (the mean entropy
    of the teacher's target distributions: near 0 or near log(out_dim)
    means collapse) -- for `tensorboard_summaries` to plot.
    """

    out_dim: int
    num_student_views: int
    num_teacher_views: int = 2
    student_temp: float = 0.1
    teacher_temp: Any = 0.04
    center_momentum: float = 0.9
    student_key: str = "student"
    teacher_key: str = "teacher"
    loss_key: str = "loss"
    metrics_key: Any = None

    def __post_init__(self):
        self.out_dim = _positive_int(self.out_dim, "out_dim")
        self.num_student_views = _positive_int(
            self.num_student_views, "num_student_views",
        )
        self.num_teacher_views = _positive_int(
            self.num_teacher_views, "num_teacher_views",
        )
        if self.num_teacher_views > self.num_student_views:
            raise ValueError(
                f"num_teacher_views ({self.num_teacher_views}) cannot exceed "
                f"num_student_views ({self.num_student_views}): the teacher's "
                "views are the student's first ones"
            )
        if self.num_student_views < 2:
            raise ValueError("num_student_views must be at least 2")
        self.student_temp = _positive_float(self.student_temp, "student_temp")
        self.teacher_temp = check_range(
            parse_schedule(self.teacher_temp, "teacher_temp"),
            "teacher_temp", low=0.0, high=float("inf"), low_open=True,
        )
        self.center_momentum = _probability(
            self.center_momentum, "center_momentum",
        )
        for name in ("student_key", "teacher_key", "loss_key"):
            setattr(self, name, _key(getattr(self, name), name))
        if self.metrics_key is not None:
            self.metrics_key = _key(self.metrics_key, "metrics_key")


@requires_resource("optimizer")
@step("dino_distillation_loss", session_type=TRAINING_SESSION_TYPE)
class DINOLoss(StatefulStep):
    """Read the student's and teacher's head outputs; write DINO's loss.

    After the loss, the centre moves toward the mean teacher output of this
    iteration's batch -- the mean over every rank's batch when
    `torch.distributed` is initialized.
    """

    config_schema = DINOLossConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self._schedule: Schedule = self._cfg.teacher_temp
        self._length: int | None = None
        self._center = torch.zeros(1, self._cfg.out_dim)
        self.last_teacher_temp: float | None = None

    @property
    def center(self) -> torch.Tensor:
        return self._center.detach().clone()

    @override
    def context_reads(self) -> dict[str, str]:
        return {"student": self._cfg.student_key, "teacher": self._cfg.teacher_key}

    @override
    def context_writes(self) -> dict[str, str]:
        writes = {"loss": self._cfg.loss_key}
        if self._cfg.metrics_key is not None:
            writes["metrics"] = self._cfg.metrics_key
        return writes

    @override
    def run(
            self,
            session: Session,
            /,
            *args: Any,
            student: torch.Tensor,
            teacher: torch.Tensor,
            **reads: Any,
    ) -> torch.Tensor:
        cfg = self._cfg
        for name, value in (("student", student), ("teacher", teacher)):
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or (
                    value.shape[1] != cfg.out_dim
            ):
                raise ValueError(
                    f"{self.name} reads (views * B, {cfg.out_dim}) {name} "
                    "outputs; got "
                    + (
                        str(tuple(value.shape))
                        if isinstance(value, torch.Tensor)
                        else type(value).__name__
                    )
                )
        if student.shape[0] * cfg.num_teacher_views != (
                teacher.shape[0] * cfg.num_student_views
        ):
            raise ValueError(
                f"{self.name}: {student.shape[0]} student rows and "
                f"{teacher.shape[0]} teacher rows are not "
                f"{cfg.num_student_views} and {cfg.num_teacher_views} views of "
                "one batch"
            )
        optimizer = self.get_dependency("optimizer")
        if self._length is None:
            self._length = self._schedule.length(
                optimizer.optimizer_steps(session.session_config.max_iterations)
            )
        # Scheduled over the optimizer steps taken before this iteration's.
        teacher_temp = self._schedule.value(optimizer.completed_steps, self._length)
        self.last_teacher_temp = teacher_temp
        if self._center.device != teacher.device:
            self._center = self._center.to(teacher.device)
        loss = dino_cross_entropy(
            student,
            teacher,
            self._center,
            num_student_views=cfg.num_student_views,
            num_teacher_views=cfg.num_teacher_views,
            student_temp=cfg.student_temp,
            teacher_temp=teacher_temp,
        )
        entropy = (
            self._teacher_entropy(teacher, teacher_temp)
            if cfg.metrics_key is not None else None
        )
        self._update_center(teacher)
        if cfg.metrics_key is None:
            return loss
        return {
            "loss": loss,
            "metrics": {
                "teacher_temp": teacher_temp,
                "center_norm": self._center.norm(),
                "teacher_entropy": entropy,
            },
        }

    @torch.no_grad()
    def _teacher_entropy(self, teacher: torch.Tensor, teacher_temp: float) -> torch.Tensor:
        """Mean entropy of the centred, sharpened teacher distributions --
        the targets this iteration's loss used."""
        log_probs = torch.log_softmax(
            (teacher.detach().float() - self._center) / teacher_temp, dim=-1,
        )
        return -(log_probs.exp() * log_probs).sum(dim=-1).mean()

    @torch.no_grad()
    def _update_center(self, teacher: torch.Tensor) -> None:
        batch_sum = teacher.detach().float().sum(dim=0, keepdim=True)
        rows = teacher.shape[0]
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(batch_sum)
            rows *= distributed.get_world_size()
        momentum = self._cfg.center_momentum
        self._center.mul_(momentum).add_(batch_sum / rows, alpha=1 - momentum)

    @override
    def get_state(self) -> dict[str, Any]:
        return {
            "center": self._center.detach().cpu().clone(),
            "schedule_steps": self._length,
        }

    @override
    def set_state(self, state: Any) -> None:
        if state is None:
            return
        center = state["center"]
        if tuple(center.shape) != (1, self._cfg.out_dim):
            raise ValueError(
                f"{self.name} state holds a centre of shape "
                f"{tuple(center.shape)}, but out_dim is {self._cfg.out_dim}"
            )
        self._center = center.detach().clone().to(self._center.device)
        self._length = state.get("schedule_steps")


__all__ = ["DINOLoss", "DINOLossConfig"]
