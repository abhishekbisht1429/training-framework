"""The DINO and BYOL loss functions against hand-written references."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from training_framework.components.builtin.self_supervised import (
    byol_loss,
    cross_view_pairs,
    dino_cross_entropy,
)


def test_cross_view_pairs_skip_a_view_compared_with_itself():
    assert cross_view_pairs(2, 4) == [
        (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3),
    ]
    with pytest.raises(ValueError, match="teacher sees 3 views but the student only 2"):
        cross_view_pairs(3, 2)


def _dino_reference(student, teacher, center, vs, vt, ts, tt):
    """DINO's `DINOLoss.forward`, written out view by view."""
    batch = student.shape[0] // vs
    total, terms = 0.0, 0
    for i in range(vt):
        target = functional.softmax(
            (teacher[i * batch:(i + 1) * batch] - center) / tt, dim=-1,
        )
        for j in range(vs):
            if i == j:
                continue
            log_probs = functional.log_softmax(
                student[j * batch:(j + 1) * batch] / ts, dim=-1,
            )
            total += (-(target * log_probs).sum(dim=-1)).mean()
            terms += 1
    return total / terms


def test_dino_cross_entropy_matches_the_reference():
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(4 * 3, 7, generator=generator)
    teacher = torch.randn(2 * 3, 7, generator=generator)
    center = torch.randn(1, 7, generator=generator)

    loss = dino_cross_entropy(
        student, teacher, center,
        num_student_views=4, num_teacher_views=2,
        student_temp=0.1, teacher_temp=0.04,
    )

    expected = _dino_reference(student, teacher, center, 4, 2, 0.1, 0.04)
    torch.testing.assert_close(loss, expected)


def test_dino_cross_entropy_sends_no_gradient_to_the_teacher():
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5, requires_grad=True)

    dino_cross_entropy(
        student, teacher, torch.zeros(1, 5),
        num_student_views=2, num_teacher_views=2,
        student_temp=0.1, teacher_temp=0.04,
    ).backward()

    assert student.grad is not None
    assert teacher.grad is None


@pytest.mark.parametrize(
    ("rows", "message"),
    [(7, "7 rows, which is not 2 views"), (6, "student views are")],
)
def test_dino_cross_entropy_refuses_rows_that_are_not_whole_views(rows, message):
    with pytest.raises(ValueError, match=message):
        dino_cross_entropy(
            torch.randn(rows, 5), torch.randn(4, 5), torch.zeros(1, 5),
            num_student_views=2, num_teacher_views=2,
            student_temp=0.1, teacher_temp=0.04,
        )


def test_byol_loss_matches_the_reference_and_is_symmetric():
    generator = torch.Generator().manual_seed(1)
    p1, p2, z1, z2 = (torch.randn(3, 6, generator=generator) for _ in range(4))

    def regression(p, z):
        return (2 - 2 * functional.cosine_similarity(p, z, dim=-1)).mean()

    loss = byol_loss(torch.cat([p1, p2]), torch.cat([z1, z2]), num_views=2)
    swapped = byol_loss(torch.cat([p2, p1]), torch.cat([z2, z1]), num_views=2)

    torch.testing.assert_close(loss, regression(p1, z2) + regression(p2, z1))
    torch.testing.assert_close(loss, swapped)


def test_byol_loss_trains_only_the_online_network():
    online = torch.randn(4, 3, requires_grad=True)
    target = torch.randn(4, 3, requires_grad=True)

    byol_loss(online, target).backward()

    assert online.grad is not None
    assert target.grad is None


def test_byol_loss_refuses_a_batch_that_is_not_two_views():
    with pytest.raises(ValueError, match="3 rows, which is not 2 views"):
        byol_loss(torch.randn(3, 4), torch.randn(4, 4))
