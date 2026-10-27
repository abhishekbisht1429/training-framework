"""`ema_teacher` and `ema_update`: the teacher starts as the student, never
trains, follows it after each optimizer step, and is checkpointed as it is."""

from __future__ import annotations

import pytest
import torch

from tests.integration.integration_ssl_components import byol_config, dino_config
from tests.test_utils import component_named, resource_named
from training_framework.components import Step, reads, step, writes
from training_framework.components.builtin import Checkpointer

from .helpers import quiet_session, run, snapshot

CONSTANT_HALF = {"momentum": 0.5}


def _model(session):
    return resource_named(session, "ema_teacher")


def _teacher_part(model):
    return model.mirrored_parts()[0]


def _student_part(model):
    return model.mirrored_parts()[1]


@pytest.mark.parametrize("build", [dino_config, byol_config])
def test_the_teacher_starts_as_the_student_and_never_trains(tmp_path, build):
    model = _model(quiet_session(build(tmp_path)))

    teacher = snapshot(_teacher_part(model))
    student = snapshot(_student_part(model))
    assert list(teacher) == list(student)
    for name in teacher:
        torch.testing.assert_close(teacher[name], student[name], msg=name)
    assert not any(p.requires_grad for p in model.teacher.parameters())
    assert any(p.requires_grad for p in model.student.parameters())


def test_forward_is_the_students_and_teacher_forward_has_no_gradient(tmp_path):
    model = _model(quiet_session(dino_config(tmp_path)))
    views = [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)]

    student = model(views)
    teacher = model.teacher_forward(views)

    assert student.requires_grad and not teacher.requires_grad
    torch.testing.assert_close(student, model.student(views))


def test_the_teacher_moves_toward_the_student_after_the_optimizer_step(tmp_path):
    # Not 0.5, where mixing up momentum and 1 - momentum would not show.
    session = quiet_session(dino_config(tmp_path, ema_update={"momentum": 0.75}))
    model = _model(session)
    before = snapshot(_teacher_part(model))

    with session:
        next(session)
        after_step = {
            name: value.detach().clone()
            for name, value in _student_part(model).named_parameters()
        }
        teacher = dict(_teacher_part(model).named_parameters())

    for name, student in after_step.items():
        # The student *after* this iteration's step: updating before the
        # step would mix in the student's previous weights instead.
        torch.testing.assert_close(
            teacher[name], 0.75 * before[name] + 0.25 * student, msg=name,
        )


def test_teacher_buffers_are_its_own_and_not_averaged(tmp_path):
    session = quiet_session(byol_config(tmp_path, ema_update=CONSTANT_HALF))
    model = _model(session)
    run(session, 2)

    teacher_bn = model.teacher.network.head.mlp[1]
    student_bn = model.student.network.backbone.head.mlp[1]
    # Both ran forward passes in train mode, on different inputs' features.
    assert not torch.equal(teacher_bn.running_mean, student_bn.running_mean)


def test_the_teacher_is_updated_only_when_the_optimizer_steps(tmp_path):
    config = dino_config(tmp_path, max_iterations=4, ema_update=CONSTANT_HALF)
    config["optimizer"]["accumulate_steps"] = 2
    session = quiet_session(config)
    model = _model(session)
    update = component_named(session, "ema_update")
    initial = snapshot(_teacher_part(model))

    with session:
        next(session)
        unchanged = snapshot(_teacher_part(model))
        assert update.last_momentum is None
        next(session)
        assert update.last_momentum == 0.5

    for name in initial:
        torch.testing.assert_close(unchanged[name], initial[name], msg=name)
    assert any(
        not torch.equal(value, initial[name])
        for name, value in snapshot(_teacher_part(model)).items()
    )


def test_the_momentum_follows_its_schedule_over_optimizer_steps(tmp_path):
    session = quiet_session(dino_config(
        tmp_path, max_iterations=4,
        ema_update={"momentum": {"start": 0.9, "end": 1.0, "kind": "linear"}},
    ))
    update = component_named(session, "ema_update")
    seen = []

    with session:
        for _ in range(4):
            next(session)
            seen.append(update.last_momentum)

    assert seen == pytest.approx([0.9, 0.925, 0.95, 0.975])
    assert update.get_state() == {"schedule_steps": 4}


def test_a_restored_schedule_keeps_its_length_when_the_run_is_longer(tmp_path):
    session = quiet_session(dino_config(
        tmp_path / "a", max_iterations=6,
        ema_update={"momentum": {"start": 0.9, "end": 1.0, "kind": "linear"}},
    ))
    update = component_named(session, "ema_update")
    update.set_state({"schedule_steps": 2})

    with session:
        run_values = []
        for _ in range(3):
            next(session)
            run_values.append(update.last_momentum)

    assert run_values == pytest.approx([0.9, 0.95, 1.0])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["ema_teacher"]["dependencies_role_bindings"].update(
            teacher="multi_view#student"), "same instance fills"),
        (lambda c: c["projection_head#teacher"].update(hidden_dim=24),
         "differ at parameter"),
        (lambda c: c["multi_view#teacher"]["dependencies_role_bindings"].update(
            network="backbone_with_head#student"), "shares parameters"),
        (lambda c: c["ema_teacher"].update(student_part="network.trunk"),
         "student has no submodule 'network.trunk'"),
        (lambda c: c["ema_teacher"].update(teacher_part=".x"),
         "teacher_part must be an attribute path"),
    ],
)
def test_a_teacher_that_cannot_follow_the_student_is_refused(tmp_path, change, message):
    config = dino_config(tmp_path)
    change(config)

    with pytest.raises(ValueError, match=message):
        quiet_session(config)


def test_ema_update_refuses_a_model_without_a_teacher(tmp_path):
    config = dino_config(tmp_path)
    config["role_bindings"]["model"] = "multi_view#student"
    del config["forward#teacher"], config["dino_distillation_loss"], config["ema_teacher"]
    del config["multi_view#teacher"], config["freeze_gradients"]
    config["compute#loss"] = {
        "function": "mean", "args": ["student"], "outputs": "loss",
    }
    session = quiet_session(config)

    with pytest.raises(TypeError, match="has no update_teacher method"):
        run(session, 1)


def test_a_diverged_teacher_is_restored_as_saved(tmp_path):
    session = quiet_session(dino_config(tmp_path, max_iterations=2))
    run(session)
    saved = _model(session)
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")

    restored = Checkpointer.load_component(path, "ema_teacher")

    teacher = snapshot(_teacher_part(restored))
    student = snapshot(_student_part(restored))
    assert any(not torch.equal(teacher[name], student[name]) for name in teacher)
    for name, value in snapshot(_teacher_part(saved)).items():
        torch.testing.assert_close(teacher[name], value, msg=name)
    assert not any(p.requires_grad for p in restored.teacher.parameters())


# -- a step fp16 skips ---------------------------------------------------------------


@reads("raw_loss")
@writes("loss")
class OverflowOnIteration2(Step):
    """Passes the loss on, infinite on iteration 2 so the fp16 scaler finds
    non-finite gradients and skips that optimizer step. Otherwise scaled
    down, so no other step overflows at the scaler's initial scale (2**16)."""

    def run(self, session, /, *args, raw_loss):
        return raw_loss * (float("inf") if session.iteration == 2 else 1e-4)


def _fp16_session(tmp_path, **ema_update):
    step("ssl_overflow", overwrite=True)(OverflowOnIteration2)
    config = dino_config(
        tmp_path, max_iterations=4,
        ema_update=ema_update or {
            "momentum": {"start": 0.9, "end": 1.0, "kind": "linear"},
        },
    )
    config["optimizer"]["precision"] = "fp16"
    config["dino_distillation_loss"]["loss_key"] = "raw_loss"
    config["ssl_overflow"] = {}
    return quiet_session(config)


def test_a_step_fp16_skips_moves_neither_the_teacher_nor_its_schedule(tmp_path):
    session = _fp16_session(tmp_path)
    model = _model(session)
    optimizer = resource_named(session, "optimizer")
    update = component_named(session, "ema_update")
    momenta, steps, stepped = [], [], []

    with session:
        for _ in range(4):
            before = snapshot(_teacher_part(model))
            next(session)
            momenta.append(update.last_momentum)
            steps.append(optimizer.completed_steps)
            stepped.append(optimizer.stepped)
            if session.iteration == 2:
                after = snapshot(_teacher_part(model))
                for name in before:
                    assert torch.equal(after[name], before[name]), name

    assert stepped == [True, False, True, True]
    assert steps == [1, 1, 2, 3]
    # Iteration 3 takes the schedule's second value, not its third.
    assert momenta == pytest.approx([0.9, 0.9, 0.925, 0.95])


def test_the_teacher_temperature_follows_the_steps_taken(tmp_path):
    session = _fp16_session(tmp_path)
    loss_step = component_named(session, "dino_distillation_loss")
    temperatures = []

    with session:
        for _ in range(4):
            next(session)
            temperatures.append(loss_step.last_teacher_temp)

    # Steps taken before each iteration: 0, 1, 1 (iteration 2 skipped), 2.
    assert temperatures == pytest.approx([0.04, 0.055, 0.055, 0.07])


def test_the_count_of_steps_taken_survives_a_checkpoint(tmp_path):
    session = _fp16_session(tmp_path)
    run(session, 3)
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")
    resumed = Checkpointer.load_checkpoint(path, restore_rng=False)
    optimizer = resource_named(resumed, "optimizer")

    with resumed:
        # Two steps taken in three iterations, not the three planned.
        assert optimizer.completed_steps == 2
        next(resumed)
        assert optimizer.completed_steps == 3


def _count_restored_from_old_state(session, tmp_path, iterations):
    """Run `iterations`, checkpoint, drop `completed_steps` from the saved
    optimizer state (as a checkpoint written before it existed), resume, and
    return the count the optimizer restored."""
    run(session, iterations)
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")
    resumed = Checkpointer.load_checkpoint(path, restore_rng=False)
    optimizer = resource_named(resumed, "optimizer")
    old_state = dict(optimizer.get_state())
    del old_state["completed_steps"]
    optimizer.set_state(old_state)
    with resumed:
        return optimizer.completed_steps


SGD = {"name": "SGD", "kwargs": {"lr": 0.01}}


@pytest.mark.parametrize(
    ("optimizer", "accumulate", "max_iterations", "saved_after", "expected"),
    [
        # From the optimizer's own per-parameter step counts.
        (None, 2, 5, 3, 1),        # mid-group: one step, not ceil(3 / 2) = 2
        (None, 1, 5, 3, 3),
        # SGD keeps no step count: complete groups, plus a final partial one.
        (SGD, 2, 5, 3, 1),         # mid-group
        (SGD, 2, 3, 3, 2),         # the run's last iteration ended a group of 1
        (SGD, 1, 5, 3, 3),
    ],
)
def test_a_state_saved_before_the_count_existed_restores_the_steps_taken(
        tmp_path, optimizer, accumulate, max_iterations, saved_after, expected,
):
    config = dino_config(tmp_path, max_iterations=max_iterations)
    config["optimizer"]["accumulate_steps"] = accumulate
    if optimizer is not None:
        config["optimizer"]["optimizer"] = optimizer

    restored = _count_restored_from_old_state(
        quiet_session(config), tmp_path, saved_after,
    )

    assert restored == expected


def test_an_old_state_does_not_count_a_step_fp16_skipped(tmp_path):
    # Three iterations, the second skipped: two steps, as AdamW counted.
    assert _count_restored_from_old_state(_fp16_session(tmp_path), tmp_path, 3) == 2
