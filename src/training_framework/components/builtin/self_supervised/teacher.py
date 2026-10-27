"""A student and a teacher that follows it: `ema_teacher` and `ema_update`.

DINO and BYOL train a student and keep a teacher (BYOL: target) that is an
exponential moving average of it. `ema_teacher` holds both and is the model
`ddp` wraps: its forward is the student's, so DDP trains the student only;
`teacher_forward` runs the teacher without gradients. `ema_update` moves the
teacher toward the student after every optimizer step.

The teacher is not a copy made here: a component that declares
prerequisites cannot be held privately (`ModuleResource` checkpoints every
tensor exactly once, through the component that created it). It is a second
configured instance tree -- the same blocks under another suffix, e.g.
`#teacher` -- bound to the `teacher` role. Construction copies the student's
weights into it and freezes it. On restore the teacher's own saved weights
are loaded after construction, so a teacher that had diverged from the
student comes back as it was saved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import torch
from torch import nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ModuleResource,
    Resource,
    StatefulStep,
    requires_resource,
    requires_step,
    resource,
    role,
    step,
)
from training_framework.components.builtin.self_supervised.schedules import (
    Schedule,
    check_range,
    parse_schedule,
)

if TYPE_CHECKING:
    from training_framework.session import Session


role(
    "student",
    Resource,
    description="the nn.Module resource ema_teacher trains through DDP",
)
role(
    "teacher",
    Resource,
    description=(
        "an nn.Module resource with the student's architecture, kept as a "
        "moving average of it"
    ),
)


def _attribute_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or (value and not all(value.split("."))):
        raise ValueError(
            f"{name} must be an attribute path such as 'network.backbone', "
            f"or '' for the whole module; got {value!r}"
        )
    return value


@dataclass
class EMATeacherConfig:
    """`student_part` and `teacher_part` (default: the whole module) are the
    attribute paths of the parts that mirror each other: the teacher part
    follows the student part, and both must have the same parameters.

    BYOL's online network ends in a predictor its target does not have, so
    only the online network's encoder and projector are mirrored: with both
    wrapped in `multi_view`, `student_part: network.backbone` and
    `teacher_part: network`."""

    student_part: Any = ""
    teacher_part: Any = ""

    def __post_init__(self):
        self.student_part = _attribute_path(self.student_part, "student_part")
        self.teacher_part = _attribute_path(self.teacher_part, "teacher_part")


@requires_resource("student")
@requires_resource("teacher")
@resource("ema_teacher", session_type=TRAINING_SESSION_TYPE)
@resource("ema_teacher", session_type=ANALYSIS_SESSION_TYPE)
class EMATeacher(ModuleResource):
    """A student trained through DDP and a teacher that follows it.

    Both are attached as the submodules `student` and `teacher`, so
    parameter names read `student.<...>` and `teacher.<...>`, and each
    checkpoints its own weights; this component owns none. The teacher's
    parameters never require gradients: DDP and the optimizer leave them
    alone, and only `update_teacher` changes them.
    """

    config_schema = EMATeacherConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        name = self._component_name()
        student = self.get_dependency("student")
        teacher = self.get_dependency("teacher")
        for role_name, part in (("student", student), ("teacher", teacher)):
            if not isinstance(part, nn.Module):
                raise TypeError(
                    f"{name} role {role_name!r} is filled by "
                    f"{type(part).__name__}, which is not an nn.Module"
                )
        if student is teacher:
            raise ValueError(
                f"{name}: the same instance fills 'student' and 'teacher'; "
                "the teacher must be its own instance (configure the blocks "
                "a second time, e.g. under a #teacher suffix)"
            )
        self.student = student
        self.teacher = teacher
        self._check_not_shared()
        self._check_same_architecture()
        with torch.no_grad():
            for mine, theirs in self._paired_tensors():
                mine.copy_(theirs)
        self.teacher.requires_grad_(False)

    def _part(self, side: str) -> nn.Module:
        """The mirrored part of the `side` ('student' or 'teacher')."""
        module = getattr(self, side)
        path = getattr(self._cfg, f"{side}_part")
        if not path:
            return module
        try:
            return module.get_submodule(path)
        except AttributeError as error:
            raise ValueError(
                f"{self._component_name()}.{side}_part: the {side} has no "
                f"submodule {path!r} ({error})"
            ) from None

    def mirrored_parts(self) -> tuple[nn.Module, nn.Module]:
        """The teacher part and the student part it follows."""
        return self._part("teacher"), self._part("student")

    def _check_not_shared(self) -> None:
        student = {id(tensor) for tensor in self.student.parameters()}
        shared = [
            parameter_name
            for parameter_name, parameter in self.teacher.named_parameters()
            if id(parameter) in student
        ]
        if shared:
            raise ValueError(
                f"{self._component_name()}: the teacher shares parameters with "
                f"the student ({shared[:3]}...); bind the teacher's blocks to "
                "instances of their own"
            )

    def _check_same_architecture(self) -> None:
        teacher, student = self.mirrored_parts()
        for kind, mine, theirs in (
                ("parameter", teacher.named_parameters(), student.named_parameters()),
                ("buffer", teacher.named_buffers(), student.named_buffers()),
        ):
            mine = [(key, tuple(value.shape)) for key, value in mine]
            theirs = [(key, tuple(value.shape)) for key, value in theirs]
            if mine == theirs:
                continue
            for index in range(max(len(mine), len(theirs))):
                left = mine[index] if index < len(mine) else None
                right = theirs[index] if index < len(theirs) else None
                if left != right:
                    raise ValueError(
                        f"{self._component_name()}: the teacher part "
                        f"{self._cfg.teacher_part!r} and the student part "
                        f"{self._cfg.student_part!r} differ at {kind} "
                        f"{index}: teacher has {left}, student has {right}"
                    )

    def _paired_tensors(self, *, buffers: bool = True):
        teacher, student = self.mirrored_parts()
        pairs = list(zip(teacher.parameters(), student.parameters()))
        if buffers:
            pairs += list(zip(teacher.buffers(), student.buffers()))
        return pairs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.student(*args, **kwargs)

    def teacher_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the teacher, without gradients."""
        with torch.no_grad():
            return self.teacher(*args, **kwargs)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """`teacher = momentum * teacher + (1 - momentum) * student`, for
        every parameter. Buffers (BatchNorm statistics) are the teacher's own,
        kept up to date by its forward passes, and are not averaged."""
        teacher, student = zip(*self._paired_tensors(buffers=False))
        torch._foreach_mul_(teacher, momentum)
        torch._foreach_add_(teacher, student, alpha=1.0 - momentum)


# -- ema_update -------------------------------------------------------------


def _default_momentum() -> dict[str, Any]:
    return {"start": 0.996, "end": 1.0, "kind": "cosine"}


@dataclass
class EMAUpdateConfig:
    """`momentum` is a schedule (see `schedules`) over optimizer steps;
    default cosine 0.996 -> 1.0 over the run, as DINO and BYOL use."""

    momentum: Any = field(default_factory=_default_momentum)

    def __post_init__(self):
        self.momentum = check_range(
            parse_schedule(self.momentum, "momentum"),
            "momentum", low=0.0, high=1.0,
        )


@requires_resource("model")
@requires_resource("optimizer")
@requires_step("optimizer_step")
@step("ema_update", session_type=TRAINING_SESSION_TYPE)
class EMAUpdate(StatefulStep):
    """Move the teacher toward the student after each optimizer step.

    Runs after `optimizer_step` and only when it changed the weights
    (`optimizer.stepped`: not inside an accumulation group, not on a step
    fp16 skipped), and calls the model's `update_teacher(momentum)`:
    `ema_teacher`'s, or any model that has one. The momentum is scheduled
    over the optimizer steps actually taken (`optimizer.completed_steps`);
    the schedule's length is kept in the state; see `schedules`.
    """

    config_schema = EMAUpdateConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self._schedule: Schedule = self._cfg.momentum
        self._length: int | None = None
        self.last_momentum: float | None = None

    @override
    def run(self, session: Session, /, *args: Any, **reads: Any) -> None:
        optimizer = self.get_dependency("optimizer")
        if not optimizer.stepped:
            # Inside an accumulation group, or fp16 skipped the step: the
            # student did not change, so neither does the teacher.
            return None
        model = self.get_dependency("model")
        update = getattr(model, "update_teacher", None)
        if not callable(update):
            raise TypeError(
                f"{self.name} updates a teacher, but the model "
                f"({type(model).__name__}) has no update_teacher method; bind "
                "model to an ema_teacher"
            )
        if self._length is None:
            self._length = self._schedule.length(
                optimizer.optimizer_steps(session.session_config.max_iterations)
            )
        # The step just taken is the `completed_steps`-th.
        momentum = self._schedule.value(optimizer.completed_steps - 1, self._length)
        update(momentum)
        self.last_momentum = momentum
        return None

    @override
    def get_state(self) -> dict[str, Any]:
        return {"schedule_steps": self._length}

    @override
    def set_state(self, state: Any) -> None:
        if state is None:
            return
        self._length = state.get("schedule_steps")


__all__ = ["EMATeacher", "EMATeacherConfig", "EMAUpdate", "EMAUpdateConfig"]
