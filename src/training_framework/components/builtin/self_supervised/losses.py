"""The self-supervised losses, as plain functions of tensors.

Views arrive stacked, view by view: a `(V * B, D)` tensor holds view `v` at
rows `v*B:(v+1)*B`, as `multi_view` returns them. A teacher sees the first
`num_teacher_views` views (the global crops); a student sees them all. Every
loss compares a teacher view with each *other* student view.

`byol_loss` is meant for the `compute` step::

    compute#loss:
      function: training_framework.components.builtin.self_supervised.losses.byol_loss
      args: [student, teacher]
      constants: {num_views: 2}
      outputs: loss

`dino_cross_entropy` is what the `dino_distillation_loss` step computes; that step keeps
the centre, which is state.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def view_chunks(
        stacked: torch.Tensor, num_views: int, name: str,
) -> tuple[torch.Tensor, ...]:
    """Split `(V * B, ...)` into its `num_views` view batches, refusing a
    tensor that does not hold a whole number of them."""
    if (
            isinstance(num_views, bool)
            or not isinstance(num_views, int)
            or num_views <= 0
    ):
        raise ValueError(f"the number of {name} views must be positive; got {num_views!r}")
    if stacked.ndim < 1 or stacked.shape[0] % num_views:
        raise ValueError(
            f"{name} holds {stacked.shape[0] if stacked.ndim else 0} rows, "
            f"which is not {num_views} views of one batch"
        )
    return stacked.chunk(num_views)


def cross_view_pairs(
        num_teacher_views: int, num_student_views: int,
) -> list[tuple[int, int]]:
    """Every (teacher view, student view) pair of different views."""
    if num_teacher_views > num_student_views:
        raise ValueError(
            f"the teacher sees {num_teacher_views} views but the student only "
            f"{num_student_views}; the teacher's views are the student's first"
        )
    return [
        (teacher_view, student_view)
        for teacher_view in range(num_teacher_views)
        for student_view in range(num_student_views)
        if student_view != teacher_view
    ]


def dino_cross_entropy(
        student: torch.Tensor,
        teacher: torch.Tensor,
        center: torch.Tensor,
        *,
        num_student_views: int,
        num_teacher_views: int,
        student_temp: float,
        teacher_temp: float,
) -> torch.Tensor:
    """DINO's loss: the cross-entropy between the centred, sharpened teacher
    distribution of each teacher view and the student's distribution of each
    other view, averaged over the pairs.

    `student` is `(Vs * B, K)` logits, `teacher` `(Vt * B, K)` logits and
    `center` `(1, K)`. Computed in float32 whatever the inputs' precision.
    """
    student_views = view_chunks(
        student.float() / student_temp, num_student_views, "student",
    )
    targets = view_chunks(
        functional.softmax((teacher.float() - center) / teacher_temp, dim=-1)
        .detach(),
        num_teacher_views, "teacher",
    )
    if student_views[0].shape != targets[0].shape:
        raise ValueError(
            f"student views are {tuple(student_views[0].shape)} and teacher "
            f"views {tuple(targets[0].shape)}; they must match"
        )
    pairs = cross_view_pairs(num_teacher_views, num_student_views)
    if not pairs:
        raise ValueError("DINO needs at least two views")
    total = student.new_zeros((), dtype=torch.float32)
    for teacher_view, student_view in pairs:
        total = total + torch.sum(
            -targets[teacher_view]
            * functional.log_softmax(student_views[student_view], dim=-1),
            dim=-1,
        ).mean()
    return total / len(pairs)


def byol_loss(
        online: torch.Tensor,
        target: torch.Tensor,
        *,
        num_views: int = 2,
) -> torch.Tensor:
    """BYOL's loss: `2 - 2 cos(p_i, z_j)` between the online prediction of
    each view and the target projection of each other view, averaged over
    the batch and summed over the pairs (both directions, as BYOL
    symmetrizes it). The target gets no gradient."""
    predictions = view_chunks(online.float(), num_views, "online")
    projections = view_chunks(target.float().detach(), num_views, "target")
    if predictions[0].shape != projections[0].shape:
        raise ValueError(
            f"online views are {tuple(predictions[0].shape)} and target "
            f"views {tuple(projections[0].shape)}; they must match"
        )
    pairs = cross_view_pairs(num_views, num_views)
    if not pairs:
        raise ValueError("BYOL needs at least two views")
    total = online.new_zeros((), dtype=torch.float32)
    for target_view, online_view in pairs:
        p = functional.normalize(predictions[online_view], dim=-1)
        z = functional.normalize(projections[target_view], dim=-1)
        total = total + (2 - 2 * (p * z).sum(dim=-1)).mean()
    return total


__all__ = [
    "byol_loss",
    "cross_view_pairs",
    "dino_cross_entropy",
    "view_chunks",
]
