"""The `dino_distillation_loss` step: its loss, centre, temperature schedule and state."""

from __future__ import annotations

import pytest
import torch
from torch import distributed

from tests.integration.integration_ssl_components import OUT_DIM, dino_config
from tests.test_utils import component_named
from training_framework.components import Step, reads, step
from training_framework.components.builtin import Checkpointer
from training_framework.components.builtin.self_supervised import (
    dino_cross_entropy,
)

from .helpers import quiet_session, run


@reads("student", "teacher")
class Recorder(Step):
    """Keeps what the loss step read, iteration by iteration."""

    seen: list = []

    def run(self, session, /, *args, student, teacher):
        type(self).seen.append((student.detach().clone(), teacher.detach().clone()))


@pytest.fixture
def recorded(tmp_path):
    Recorder.seen = []
    step("ssl_recorder", overwrite=True)(Recorder)
    return Recorder.seen


def _session(tmp_path, **dino_distillation_loss):
    config = dino_config(tmp_path)
    config["dino_distillation_loss"].update(dino_distillation_loss)
    config["ssl_recorder"] = {}
    return quiet_session(config)


def test_the_centre_follows_the_teacher_outputs(tmp_path, recorded):
    session = _session(tmp_path, center_momentum=0.9)
    loss_step = component_named(session, "dino_distillation_loss")
    centers = []

    with session:
        for _ in range(2):
            before = loss_step.center
            next(session)
            centers.append((before, loss_step.center))

    assert torch.count_nonzero(centers[0][0]) == 0
    for (_, teacher), (before, after) in zip(recorded, centers):
        torch.testing.assert_close(
            after, 0.9 * before + 0.1 * teacher.mean(dim=0, keepdim=True),
        )


def test_the_loss_written_is_the_cross_entropy_of_the_views(tmp_path, recorded):
    session = _session(tmp_path)
    loss_step = component_named(session, "dino_distillation_loss")
    student = torch.randn(4 * 2, OUT_DIM)
    teacher = torch.randn(2 * 2, OUT_DIM)

    with session:
        next(session)
        center = loss_step.center
        loss = loss_step.run(session, student=student, teacher=teacher)

    torch.testing.assert_close(loss, dino_cross_entropy(
        student, teacher, center, num_student_views=4, num_teacher_views=2,
        student_temp=0.1, teacher_temp=loss_step.last_teacher_temp,
    ))


def test_the_teacher_temperature_warms_up_over_optimizer_steps(tmp_path, recorded):
    session = _session(tmp_path)
    loss_step = component_named(session, "dino_distillation_loss")
    temperatures = []

    with session:
        for _ in range(3):
            next(session)
            temperatures.append(loss_step.last_teacher_temp)

    assert temperatures == pytest.approx([0.04, 0.055, 0.07])


def test_the_centre_is_averaged_over_every_ranks_batch(tmp_path, monkeypatch, recorded):
    session = _session(tmp_path)
    loss_step = component_named(session, "dino_distillation_loss")
    teacher = torch.randn(2 * 2, OUT_DIM)

    with session:
        next(session)
        before = loss_step.center
        # A second rank whose teacher outputs sum to zero: the mean over both
        # ranks' batches is half this rank's mean.
        monkeypatch.setattr(distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(distributed, "all_reduce", lambda tensor: None)
        loss_step.run(session, student=torch.randn(4 * 2, OUT_DIM), teacher=teacher)
        monkeypatch.undo()

    torch.testing.assert_close(
        loss_step.center, 0.9 * before + 0.1 * teacher.mean(dim=0, keepdim=True) / 2,
    )


def test_outputs_that_are_not_the_configured_views_are_refused(tmp_path, recorded):
    session = _session(tmp_path, num_student_views=3)

    with pytest.raises(ValueError, match="are not 3 and 2 views of one batch"):
        run(session, 1)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"num_teacher_views": 3, "num_student_views": 2}, "cannot exceed"),
        ({"num_student_views": 1, "num_teacher_views": 1}, "at least 2"),
        ({"teacher_temp": 0}, r"teacher_temp values must be in \(0.0"),
        ({"center_momentum": 1.5}, "center_momentum must be a number between 0 and 1"),
        ({"loss_key": ""}, "loss_key must be a non-empty"),
    ],
)
def test_invalid_dino_loss_configuration_is_reported(tmp_path, recorded, config, message):
    with pytest.raises(ValueError, match=message):
        _session(tmp_path, **config)


def test_the_centre_and_schedule_length_survive_a_checkpoint(tmp_path, recorded):
    session = _session(tmp_path)
    run(session, 2)
    saved = component_named(session, "dino_distillation_loss")
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")

    restored = component_named(Checkpointer.load_checkpoint(path, restore_rng=False), "dino_distillation_loss")

    assert torch.count_nonzero(saved.center) > 0
    torch.testing.assert_close(restored.center, saved.center)
    assert restored.get_state()["schedule_steps"] == saved.get_state()["schedule_steps"] == 2
