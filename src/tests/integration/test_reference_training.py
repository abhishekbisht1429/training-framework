"""Well-known trainings run through the framework and, side by side, as plain
torch loops from the same seeds, data and initial weights.

Every iteration is compared -- loss, learning rate, gradient norm -- and the
weights at the end, so any difference the framework introduces (what runs
when, what the optimizer or schedule sees, what a restore brings back) shows
up at the first iteration it affects. Every value must also stay finite.

The optimizer is the recipe a real run uses: AdamW with weight decay, a
linear warmup handing over to a cosine decay at a milestone.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from copy import deepcopy

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional

from tests.test_utils import (
    component_named,
    make_config,
    resource_named,
    stub_process_group,
)
from training_framework.components import (
    StatefulResource,
    Step,
    reads,
    requires_resource,
    requires_step,
    resource,
    step,
    writes,
)
from training_framework.components.builtin import Checkpointer
from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import TrainingSession

_COMPONENTS = "tests.integration.integration_reference_components"

ITERATIONS = 40
BATCH_SIZE = 8
MODEL_SEED = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_START = 1e-3
ETA_MIN = 1e-5


def _register_components():
    """Register the shared components again: the registry is reset around
    every test, as importing a components package does in a run."""
    existing = sys.modules.get(_COMPONENTS)
    if existing is None:
        importlib.import_module(_COMPONENTS)
    else:
        importlib.reload(existing)


def _components():
    """The shared components module, for its model and data."""
    return importlib.import_module(_COMPONENTS)


def _optimizer_config(*, milestone, accumulate=1):
    config = {
        "optimizer": {
            "name": "AdamW",
            "kwargs": {"lr": LR, "weight_decay": WEIGHT_DECAY},
        },
        "lr_scheduler": {
            "stages": [
                {"name": "LinearLR", "kwargs": {
                    "start_factor": WARMUP_START,
                    "total_iters": "$stage_iterations",
                }},
                {"name": "CosineAnnealingLR", "kwargs": {
                    "T_max": "$stage_iterations", "eta_min": ETA_MIN,
                }},
            ],
            "milestones": [milestone],
        },
    }
    if accumulate > 1:
        config["accumulate_steps"] = accumulate
    return config


def _reference_optimizer(parameters, *, steps, milestone):
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=WARMUP_START, total_iters=milestone,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps - milestone, eta_min=ETA_MIN,
            ),
        ],
        milestones=[milestone],
    )
    return optimizer, scheduler


def _in_process(config) -> TrainingSession:
    """A session run in this process: DDP is the recording stand-in, and
    `ddp` gets rank 0, which the engine would assign."""
    session = TrainingSession(config)
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    placeholder = resource_named(session, "ddp")
    session.unregister_resource("ddp")
    session.register_resource(type(placeholder)(config=placeholder.config, rank=0))
    return session


def _assert_iterations_match(got, expected):
    assert len(got) == len(expected)
    for record, reference in zip(got, expected):
        where = f"iteration {record['iteration']}"
        for key in ("loss", "lr", "grad_norm"):
            value = record[key]
            if value is not None:
                assert math.isfinite(value), f"{where}: {key} is {value}"
        assert record["loss"] == pytest.approx(
            reference["loss"], rel=1e-5, abs=1e-7,
        ), f"{where}: loss"
        assert record["lr"] == pytest.approx(
            reference["lr"], rel=1e-9, abs=1e-12,
        ), f"{where}: learning rate after the step"
        if reference["grad_norm"] is None:
            assert record["grad_norm"] is None, f"{where}: stepped unexpectedly"
        else:
            assert record["grad_norm"] == pytest.approx(
                reference["grad_norm"], rel=1e-5, abs=1e-7,
            ), f"{where}: gradient norm"


def _assert_same_parameters(module, reference):
    for (name, value), expected in zip(
            module.named_parameters(), reference.parameters(),
    ):
        torch.testing.assert_close(value, expected, msg=f"parameter {name}")


# -- 1. supervised ViT classification -----------------------------------------------


def _classification_config(
        tmp_path, *, iterations=ITERATIONS, milestone=10, clip=None, accumulate=1,
        output_path=None,
):
    config = make_config(tmp_path, max_iterations=iterations)
    config["session_config"].update({
        "components_package": _COMPONENTS,
        "show_execution_graph": False,
    })
    config.update({
        "component_bindings": {"model": "ref_vit", "dataset": "ref_dataset"},
        "ref_vit": {"seed": MODEL_SEED},
        "ref_dataset": {},
        "ddp": {
            "world_size": 1, "backend": "gloo",
            "master_addr": "127.0.0.1", "master_port": "12355",
        },
        "data_manager": {
            "batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": False,
        },
        "load_batch": {"fields": ["images", "labels", "indices"]},
        "forward": {"args": ["images"], "outputs": "logits"},
        "compute#loss": {
            "function": "torch.nn.functional.cross_entropy",
            "args": ["logits", "labels"],
            "outputs": "loss",
        },
        "optimizer": _optimizer_config(milestone=milestone, accumulate=accumulate),
        "clip_gradients": {
            "track_norm": True,
            **({} if clip is None else {"max_norm": clip}),
        },
        "ref_recorder": (
            {} if output_path is None else {"output_path": str(output_path)}
        ),
    })
    return config


def _reference_classification(records, *, milestone=10, clip=None, accumulate=1):
    """The classification loop in plain torch, fed the batches the
    framework drew (by the recorded indices)."""
    components = _components()
    images, labels = components.dataset_tensors()
    model = components.TinyViT(seed=MODEL_SEED)
    iterations = len(records)
    optimizer, scheduler = _reference_optimizer(
        model.parameters(),
        steps=math.ceil(iterations / accumulate),
        milestone=milestone,
    )
    expected = []
    for iteration, record in enumerate(records, start=1):
        index = torch.tensor(record["indices"])
        loss = functional.cross_entropy(model(images[index]), labels[index])
        # The mean over the micro-batches of this iteration's group; only
        # the run's last group can be shorter than `accumulate`.
        group_start = (iteration - 1) // accumulate * accumulate + 1
        group_size = min(group_start + accumulate - 1, iterations) - group_start + 1
        (loss / group_size if group_size > 1 else loss).backward()
        norm = None
        if accumulate == 1 or iteration % accumulate == 0 or iteration == iterations:
            gradients = [p for p in model.parameters() if p.grad is not None]
            if clip is None:
                norm = nn.utils.get_total_norm([p.grad for p in gradients])
            else:
                norm = nn.utils.clip_grad_norm_(gradients, clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        expected.append({
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": None if norm is None else float(norm),
        })
    return model, expected


@pytest.mark.parametrize(
    ("clip", "accumulate", "milestone", "iterations"),
    [
        (None, 1, 10, ITERATIONS),
        (0.5, 1, 10, ITERATIONS),
        (None, 2, 5, ITERATIONS),
        # 41 = 13 groups of 3 and a last group of 2.
        (None, 3, 5, 41),
    ],
    ids=["plain", "clipped", "accumulated", "accumulated-short-last-group"],
)
def test_vit_classification_matches_plain_torch(
        tmp_path, monkeypatch, clip, accumulate, milestone, iterations,
):
    stub_process_group(monkeypatch)
    _register_components()
    session = _in_process(_classification_config(
        tmp_path, iterations=iterations, milestone=milestone, clip=clip,
        accumulate=accumulate,
    ))
    with session:
        list(session)
    records = component_named(session, "ref_recorder").records

    reference, expected = _reference_classification(
        records, milestone=milestone, clip=clip, accumulate=accumulate,
    )

    _assert_iterations_match(records, expected)
    _assert_same_parameters(resource_named(session, "ref_vit"), reference)
    # Every sample is drawn, the loss moves, and the schedule is crossed.
    assert {i for r in records for i in r["indices"]} == set(range(32))
    assert records[-1]["loss"] < records[0]["loss"]
    if clip is not None:
        assert any(r["grad_norm"] > clip for r in records), "clipping never bit"


# -- 2 and 4. DINO-style self-distillation -----------------------------------------------

DINO_OUT = 16
STUDENT_TEMPERATURE = 0.1
TEACHER_TEMPERATURE = 0.04
CENTER_MOMENTUM = 0.9
EMA_MOMENTUM = 0.99


def _two_views():
    """Two fixed "augmented" views of each image: noise on the image, and
    noise on its mirror image."""
    images, _ = _components().dataset_tensors()
    generator = torch.Generator().manual_seed(99)
    first = images + 0.1 * torch.randn(images.shape, generator=generator)
    second = images.flip(-1) + 0.1 * torch.randn(images.shape, generator=generator)
    return first, second


def dino_loss(s1, s2, t1, t2, center):
    """Cross-entropy between each view's centred, sharpened teacher output
    and the student's output on the other view."""
    q1 = functional.softmax((t1 - center) / TEACHER_TEMPERATURE, dim=-1)
    q2 = functional.softmax((t2 - center) / TEACHER_TEMPERATURE, dim=-1)
    p1 = functional.log_softmax(s1 / STUDENT_TEMPERATURE, dim=-1)
    p2 = functional.log_softmax(s2 / STUDENT_TEMPERATURE, dim=-1)
    return (-(q1 * p2).sum(-1).mean() - (q2 * p1).sum(-1).mean()) / 2


def _updated_center(center, t1, t2):
    batch_center = torch.cat([t1, t2]).mean(dim=0, keepdim=True)
    return CENTER_MOMENTUM * center + (1 - CENTER_MOMENTUM) * batch_center


def _ema(teacher: nn.Module, student: nn.Module) -> None:
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters()):
            t.mul_(EMA_MOMENTUM).add_(s.detach(), alpha=1 - EMA_MOMENTUM)


def _register_dino(*, pinned: bool):
    """The DINO pieces a user writes. `pinned` places the center update after
    the loss and the EMA after the optimizer step, as the docs say to; without
    it they declare only what they read."""
    from training_framework.components import Resource

    views = _two_views()

    @resource("dino_views", overwrite=True)
    class Views(Resource):
        def __len__(self):
            return len(views[0])

        def __getitem__(self, index):
            return views[0][index], views[1][index], index

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    @requires_resource("model")
    @resource("dino_teacher", overwrite=True)
    class Teacher(StatefulResource):
        """A copy of the student, updated only by EMA."""

        def __init__(self, config=None):
            self.network = _components().TinyViT(
                seed=MODEL_SEED, out_dim=DINO_OUT,
            ).requires_grad_(False)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {k: v.clone() for k, v in self.network.state_dict().items()}

        def set_state(self, state):
            self.network.load_state_dict(state)

    @resource("dino_center", overwrite=True)
    class Center(StatefulResource):
        def __init__(self, config=None):
            self.value = torch.zeros(1, DINO_OUT)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {"value": self.value.clone()}

        def set_state(self, state):
            self.value = state["value"].clone()

    @requires_resource("dino_teacher")
    @reads("view1", "view2")
    @writes("t1", "t2")
    @step("dino_teacher_forward", overwrite=True)
    class TeacherForward(Step):
        def run(self, session, *, view1, view2):
            teacher = self.get_dependency("dino_teacher").network
            with torch.no_grad():
                return teacher(view1), teacher(view2)

    @requires_resource("dino_center")
    @reads("s1", "s2", "t1", "t2")
    @writes("loss")
    @step("dino_loss", overwrite=True)
    class Loss(Step):
        def run(self, session, *, s1, s2, t1, t2):
            return dino_loss(s1, s2, t1, t2, self.get_dependency("dino_center").value)

    @requires_resource("dino_center")
    @reads("t1", "t2")
    class CenterUpdate(Step):
        def run(self, session, *, t1, t2):
            center = self.get_dependency("dino_center")
            center.value = _updated_center(center.value, t1, t2)

    @requires_resource("dino_teacher")
    @requires_resource("model")
    class EmaUpdate(Step):
        def run(self, session):
            _ema(
                self.get_dependency("dino_teacher").network,
                self.get_dependency("model"),
            )

    if pinned:
        CenterUpdate = requires_step("dino_loss")(CenterUpdate)
        EmaUpdate = requires_step("optimizer_step")(EmaUpdate)
    step("dino_center_update", overwrite=True)(CenterUpdate)
    step("dino_ema", overwrite=True)(EmaUpdate)


def _dino_config(tmp_path, *, milestone=10):
    config = make_config(tmp_path, max_iterations=ITERATIONS)
    config["session_config"].update({
        "components_package": _COMPONENTS,
        "show_execution_graph": False,
    })
    # Listed in the order a DINO loop runs them.
    config.update({
        "component_bindings": {"model": "ref_vit", "dataset": "dino_views"},
        "ref_vit": {"seed": MODEL_SEED, "out_dim": DINO_OUT},
        "dino_views": {},
        "dino_teacher": {},
        "dino_center": {},
        "ddp": {
            "world_size": 1, "backend": "gloo",
            "master_addr": "127.0.0.1", "master_port": "12355",
        },
        "data_manager": {
            "batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": False,
        },
        "load_batch": {"fields": ["view1", "view2", "indices"]},
        "forward#s1": {"args": ["view1"], "outputs": "s1"},
        "forward#s2": {"args": ["view2"], "outputs": "s2"},
        "dino_teacher_forward": {},
        "dino_loss": {},
        "dino_center_update": {},
        "optimizer": _optimizer_config(milestone=milestone),
        "clip_gradients": {"track_norm": True},
        "dino_ema": {},
        "ref_recorder": {},
    })
    return config


def _reference_dino(records, *, milestone=10):
    """The canonical DINO iteration: student and teacher forward, loss,
    center update, backward, step, then the teacher's EMA."""
    first, second = _two_views()
    components = _components()
    student = components.TinyViT(seed=MODEL_SEED, out_dim=DINO_OUT)
    teacher = components.TinyViT(seed=MODEL_SEED, out_dim=DINO_OUT)
    teacher.requires_grad_(False)
    center = torch.zeros(1, DINO_OUT)
    optimizer, scheduler = _reference_optimizer(
        student.parameters(), steps=len(records), milestone=milestone,
    )
    expected = []
    for record in records:
        index = torch.tensor(record["indices"])
        view1, view2 = first[index], second[index]
        s1, s2 = student(view1), student(view2)
        with torch.no_grad():
            t1, t2 = teacher(view1), teacher(view2)
        loss = dino_loss(s1, s2, t1, t2, center)
        center = _updated_center(center, t1, t2)
        loss.backward()
        norm = nn.utils.get_total_norm(
            [p.grad for p in student.parameters() if p.grad is not None]
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        _ema(teacher, student)
        expected.append({
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": float(norm),
        })
    return student, teacher, center, expected


def _run_dino(tmp_path, monkeypatch, *, pinned):
    stub_process_group(monkeypatch)
    _register_components()
    _register_dino(pinned=pinned)
    session = _in_process(_dino_config(tmp_path))
    with session:
        list(session)
    records = component_named(session, "ref_recorder").records
    student, teacher, center, expected = _reference_dino(records)

    _assert_iterations_match(records, expected)
    _assert_same_parameters(resource_named(session, "ref_vit"), student)
    _assert_same_parameters(resource_named(session, "dino_teacher").network, teacher)
    torch.testing.assert_close(resource_named(session, "dino_center").value, center)


def test_dino_self_distillation_matches_plain_torch(tmp_path, monkeypatch):
    _run_dino(tmp_path, monkeypatch, pinned=True)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known hazard: a step with no declared link to the chain is placed "
        "by the sort, not by its config position. The EMA step, listed after "
        "the optimizer, runs first in each iteration: an extra update before "
        "the first forward and none after the last step. The teacher drifts "
        "from the reference, and AdamW amplifies the resulting loss "
        "differences into the student's weights. Pin such steps with "
        "@requires_step until the framework places or rejects them."
    ),
)
def test_dino_with_unplaced_update_steps_matches_plain_torch(tmp_path, monkeypatch):
    _run_dino(tmp_path, monkeypatch, pinned=False)


# -- 3. the real launch path, and a resume --------------------------------------------


def test_an_engine_launched_run_matches_plain_torch(tmp_path, monkeypatch):
    _register_components()
    output_path = tmp_path / "records.json"
    config = _classification_config(tmp_path, output_path=output_path)
    # The engine picks the rendezvous port itself.
    del config["ddp"]["master_port"]
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump({"sessions": [config]}), encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "training-framework", "--config", str(config_path),
        "--heartbeat-timeout", "60", "--process_timeout_on_join", "10",
    ])

    with TrainingEngine(Configurator()) as engine:
        engine.start_session()

    records = json.loads(output_path.read_text(encoding="utf-8"))
    _, expected = _reference_classification(records)
    _assert_iterations_match(records, expected)


def test_a_run_resumed_from_a_checkpoint_matches_plain_torch(tmp_path, monkeypatch):
    stub_process_group(monkeypatch)
    _register_components()
    paused = _in_process(_classification_config(tmp_path / "run"))
    with paused:
        for _ in range(ITERATIONS // 2):
            next(paused)
    path = Checkpointer.save_checkpoint(paused, tmp_path / "checkpoint")

    resumed = Checkpointer.load_checkpoint(path)
    with resumed:
        list(resumed)
    records = component_named(resumed, "ref_recorder").records

    reference, expected = _reference_classification(records)
    assert len(records) == ITERATIONS
    _assert_iterations_match(records, expected)
    _assert_same_parameters(resource_named(resumed, "ref_vit"), reference)
