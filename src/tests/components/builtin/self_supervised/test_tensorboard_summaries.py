"""`tensorboard_summaries` on a DINO run, read back from the event files,
and the metrics the DINO steps publish for it."""

from __future__ import annotations

import glob
import pickle

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.integration.integration_ssl_components import dino_config
from tests.test_utils import component_named
from training_framework.components import Step, reads, step, writes
from training_framework.components.builtin import Checkpointer

from .helpers import quiet_session, run


def _config(tmp_path, *, max_iterations=3, **summaries):
    config = dino_config(tmp_path, max_iterations=max_iterations)
    config["dino_distillation_loss"]["metrics_key"] = "dino_metrics"
    config["ema_update"]["momentum"] = {"start": 0.9, "end": 1.0, "kind": "linear"}
    config["ema_update"]["momentum_key"] = "ema_momentum"
    config["tensorboard"] = {"serve": False}
    config["tensorboard_summaries"] = summaries or {
        "scalars": {
            "train/loss": "loss",
            "dino": "dino_metrics",
            "ema/momentum": "ema_momentum",
        },
        "optimizer": True,
    }
    return config


def _events(session) -> dict[str, list[tuple[int, float]]]:
    (directory,) = glob.glob(f"{session.session_config.session_dir}/*_tensorboard")
    events = EventAccumulator(directory)
    events.Reload()
    return {
        tag: [(event.step, event.value) for event in events.Scalars(tag)]
        for tag in events.Tags()["scalars"]
    }


def test_the_run_writes_its_loss_metrics_and_optimizer_values(tmp_path):
    session = quiet_session(_config(tmp_path))
    run(session)

    events = _events(session)

    assert sorted(events) == sorted([
        "train/loss",
        "dino/teacher_temp", "dino/center_norm", "dino/teacher_entropy",
        "ema/momentum",
        "optimizer/lr/group_0", "optimizer/lr/group_1",
        "optimizer/grad_norm", "optimizer/last_loss", "optimizer/completed_steps",
    ])
    assert [step for step, _ in events["train/loss"]] == [1, 2, 3]
    assert [v for _, v in events["dino/teacher_temp"]] == pytest.approx([0.04, 0.055, 0.07])
    assert [v for _, v in events["ema/momentum"]] == pytest.approx([0.9, 0.9 + 0.1 / 3, 0.9 + 0.2 / 3])
    assert [v for _, v in events["optimizer/completed_steps"]] == [1, 2, 3]
    last_losses, losses = events["optimizer/last_loss"], events["train/loss"]
    assert [step for step, _ in last_losses] == [step for step, _ in losses]
    assert [v for _, v in last_losses] == pytest.approx([v for _, v in losses])
    entropies = [v for _, v in events["dino/teacher_entropy"]]
    assert all(0 < v < torch.log(torch.tensor(32.0)).item() for v in entropies)
    center = component_named(session, "dino_distillation_loss").center
    assert events["dino/center_norm"][-1][1] == pytest.approx(center.norm().item(), rel=1e-5)


def test_summaries_are_written_on_their_cadence(tmp_path):
    session = quiet_session(_config(
        tmp_path, max_iterations=5,
        call_every=2, scalars={"train/loss": "loss"},
    ))
    run(session)

    # The first iteration, every second one, and the last.
    assert [step for step, _ in _events(session)["train/loss"]] == [1, 2, 4, 5]


def test_a_key_no_step_writes_is_refused_when_the_session_is_built(tmp_path):
    with pytest.raises(Exception, match="lossy"):
        quiet_session(_config(tmp_path, scalars={"train/loss": "lossy"}))


@pytest.mark.parametrize(
    ("summaries", "message"),
    [
        ({}, "writes nothing"),
        ({"scalars": {"": "loss"}}, "non-empty tags"),
        ({"scalars": ["loss"]}, "must be a mapping"),
        ({"scalars": {"a": "loss"}, "call_every": 0}, "call_every must be a positive"),
        ({"optimizer": "yes"}, "optimizer must be true or false"),
    ],
)
def test_invalid_summaries_configuration_is_reported(tmp_path, summaries, message):
    config = _config(tmp_path)
    config["tensorboard_summaries"] = summaries
    with pytest.raises(ValueError, match=message):
        quiet_session(config)


@writes("views_mean")
class ViewsMean(Step):
    def run(self, session):
        return torch.ones(2, 2)


def test_a_value_that_is_not_a_scalar_is_refused_naming_its_tag(tmp_path):
    step("ssl_views_mean", overwrite=True)(ViewsMean)
    config = _config(tmp_path, scalars={"views/mean": "views_mean"})
    config["ssl_views_mean"] = {}

    with pytest.raises(TypeError, match="'views/mean' is a tensor of shape \\(2, 2\\)"):
        run(quiet_session(config), 1)


def test_the_summaries_hook_runs_on_rank_zero_only():
    from training_framework.components.builtin import TensorboardSummaries

    assert TensorboardSummaries.rank_zero_only is True


def test_the_summaries_hook_survives_a_pickle_and_a_checkpoint(tmp_path):
    session = quiet_session(_config(tmp_path))
    run(session, 2)
    hook = component_named(session, "tensorboard_summaries")

    copy = pickle.loads(pickle.dumps(hook))
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")
    restored = component_named(
        Checkpointer.load_checkpoint(path, restore_rng=False), "tensorboard_summaries",
    )

    for other in (copy, restored):
        assert other.call_every == hook.call_every
        assert other.context_reads() == hook.context_reads()


def test_without_metric_keys_the_dino_steps_write_only_what_they_did(tmp_path):
    config = dino_config(tmp_path)
    session = quiet_session(config)

    assert component_named(session, "dino_distillation_loss").context_writes() == {"loss": "loss"}
    assert component_named(session, "ema_update").context_writes() == {}


def test_the_logger_prints_the_loss_backward_last_received(tmp_path):
    from training_framework.session import TrainingSession

    config = dino_config(tmp_path, max_iterations=3)
    config["logger"] = {"log_every": 1, "log_file": str(tmp_path / "log.txt")}
    config["ssl_loss_recorder"] = {}
    step("ssl_loss_recorder", overwrite=True)(LossRecorder)
    LossRecorder.losses = []
    session = TrainingSession(config)
    session.unregister_hook("checkpointer")
    run(session)

    lines = (tmp_path / "log.txt").read_text().splitlines()
    assert " | loss: " not in lines[0]          # nothing backpropagated yet
    # Written before each iteration: the loss of the iteration before.
    for line, loss in zip(lines[1:], LossRecorder.losses):
        assert line.endswith(f" | loss: {loss:.4e}")


@reads("loss")
class LossRecorder(Step):
    losses: list = []

    def run(self, session, /, *args, loss):
        type(self).losses.append(float(loss))
