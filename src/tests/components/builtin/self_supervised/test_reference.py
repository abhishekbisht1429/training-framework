"""DINO and BYOL built from the components, against the same recipes in
plain torch.

The framework session trains for a few iterations; the reference repeats
them on the batches the session read, with the loss written out by hand
(`test_losses`), the optimizer, last-layer freezing, gradient clipping, the
teacher's moving average and DINO's centre. Every weight of student and
teacher must agree at the end. A resumed run must match an uninterrupted
one.
"""

from __future__ import annotations

import copy
import fnmatch
import math

import pytest
import torch
from torch.nn import functional

from tests.integration.integration_ssl_components import byol_config, dino_config
from tests.test_utils import component_named, resource_named
from training_framework.components import Step, reads, step
from training_framework.components.builtin import Checkpointer

from .helpers import quiet_session, run, snapshot
from .test_losses import _dino_reference

ITERATIONS = 4


@reads("views")
class BatchRecorder(Step):
    batches: list = []

    def run(self, session, /, *args, views):
        type(self).batches.append([view.clone() for view in views])


@pytest.fixture
def batches():
    BatchRecorder.batches = []
    step("ssl_batch_recorder", overwrite=True)(BatchRecorder)
    return BatchRecorder.batches


def _train(config):
    config["ssl_batch_recorder"] = {}
    session = quiet_session(config)
    model = resource_named(session, "ema_teacher")
    initial = copy.deepcopy(model)
    run(session)
    return session, model, initial


def _param_groups(model, weight_decay):
    no_decay = [
        parameter for name, parameter in model.named_parameters()
        if any(fnmatch.fnmatchcase(name, p) for p in ("*.bias", "*norm*"))
    ]
    ids = {id(p) for p in no_decay}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": rest, "weight_decay": weight_decay},
    ]


def _ema(model, student_part, teacher_part, momentum):
    with torch.no_grad():
        for teacher, student in zip(
                model.get_submodule(teacher_part).parameters(),
                model.get_submodule(student_part).parameters(),
        ):
            teacher.mul_(momentum).add_(student, alpha=1 - momentum)


def _cosine(start, end, step, total):
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * min(step / total, 1)))


def _assert_same(trained, reference):
    trained, reference = snapshot(trained), snapshot(reference)
    assert list(trained) == list(reference)
    for name in trained:
        torch.testing.assert_close(trained[name], reference[name], msg=name)


def test_dino_matches_plain_torch(tmp_path, batches):
    session, model, reference = _train(dino_config(tmp_path, max_iterations=ITERATIONS))

    optimizer = torch.optim.AdamW(_param_groups(reference, 0.04), lr=0.01)
    last_layer = [
        p for name, p in reference.named_parameters()
        if name.startswith("student.network.head.last_layer.")
    ]
    center = torch.zeros(1, 32)
    for iteration, views in enumerate(batches, start=1):
        optimizer.zero_grad()
        student = reference(views)
        teacher = reference.teacher_forward(views)
        teacher_temp = 0.04 + (0.07 - 0.04) * min((iteration - 1) / 2, 1)
        loss = _dino_reference(student, teacher, center, 4, 2, 0.1, teacher_temp)
        loss.backward()
        if iteration <= 1:
            for parameter in last_layer:
                parameter.grad = None
        torch.nn.utils.clip_grad_norm_(
            [p for p in reference.parameters() if p.grad is not None], 3.0,
        )
        optimizer.step()
        _ema(reference, "student", "teacher",
             _cosine(0.9, 1.0, iteration - 1, ITERATIONS))
        center = 0.9 * center + 0.1 * teacher.mean(dim=0, keepdim=True)

    assert len(batches) == ITERATIONS
    _assert_same(model, reference)
    torch.testing.assert_close(component_named(session, "dino_distillation_loss").center, center)


def test_byol_matches_plain_torch(tmp_path, batches):
    _, model, reference = _train(byol_config(tmp_path, max_iterations=ITERATIONS))

    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=1e-6)
    for iteration, views in enumerate(batches, start=1):
        optimizer.zero_grad()
        online = reference(views)
        target = reference.teacher_forward(views)
        p1, p2 = online.chunk(2)
        z1, z2 = target.chunk(2)
        loss = sum(
            (2 - 2 * functional.cosine_similarity(p, z, dim=-1)).mean()
            for p, z in ((p1, z2), (p2, z1))
        )
        loss.backward()
        optimizer.step()
        _ema(reference, "student.network.backbone", "teacher.network",
             _cosine(0.9, 1.0, iteration - 1, ITERATIONS))

    _assert_same(model, reference)


@pytest.mark.parametrize(
    ("build", "views"),
    [
        (dino_config, {"num_global": 2, "num_local": 2, "num_samples": 1}),
        (byol_config, {"num_global": 2, "num_local": 0, "num_samples": 1}),
    ],
)
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, build, views, batches):
    # One sample, so every batch is the same whatever order the sampler
    # resumes in: this checks what these components restore (teacher,
    # centre, schedule lengths, optimizer), not the data manager's position.
    whole = quiet_session(build(
        tmp_path / "whole", max_iterations=ITERATIONS, issl_views=views,
    ))
    run(whole)

    first = quiet_session(build(
        tmp_path / "first", max_iterations=ITERATIONS, issl_views=views,
    ))
    with first:
        for _ in range(2):
            next(first)
        path = Checkpointer.save_checkpoint(first, tmp_path / "checkpoint")
    resumed = Checkpointer.load_checkpoint(path)
    run(resumed)

    _assert_same(resource_named(resumed, "ema_teacher"), resource_named(whole, "ema_teacher"))
    if build is dino_config:
        torch.testing.assert_close(
            component_named(resumed, "dino_distillation_loss").center,
            component_named(whole, "dino_distillation_loss").center,
        )
