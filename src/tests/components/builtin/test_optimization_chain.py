"""The optimizer resource and the steps that drive it, through real sessions.

Each session is built from configuration and run through the public session
API. DDP is replaced by a stand-in that records `no_sync`, and the process
group calls are stubbed, as the other built-in tests do.
"""

from __future__ import annotations

import math
import pickle

import pytest
import torch
from torch import nn

from tests.test_utils import (
    component_named,
    make_config,
    resource_named,
    stub_process_group,
)
from training_framework.components import (
    StatefulResource,
    Step,
    requires_resource,
    requires_step,
    resource,
    step,
    writes,
)
from training_framework.components.builtin import GradientProcessor
from training_framework.session import TrainingSession


@pytest.fixture(autouse=True)
def _stub_process_group(monkeypatch):
    stub_process_group(monkeypatch)


INITIAL_WEIGHT = [[0.5, -0.25]]
INITIAL_BIAS = [0.1]


def batch(index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The deterministic micro-batch for micro-batch `index` (from 1)."""
    x = torch.tensor([[float(index), 1.0], [1.0, -float(index)]])
    y = torch.tensor([[1.0], [0.0]])
    return x, y


def fresh_model() -> nn.Linear:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(INITIAL_WEIGHT))
        model.bias.copy_(torch.tensor(INITIAL_BIAS))
    return model


def _register_components(records: dict):
    @resource("chain_model", overwrite=True)
    class ChainModel(nn.Module, StatefulResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self.linear = fresh_model()

        def forward(self, x):
            records.setdefault("forward_autocast", []).append(
                torch.is_autocast_enabled("cpu")
            )
            return self.linear(x)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {k: v.detach().clone() for k, v in self.state_dict().items()}

        def set_state(self, state):
            self.load_state_dict(state)

    @writes("loss")
    @requires_resource("ddp")
    @step("chain_loss", overwrite=True)
    class ChainLoss(Step):
        """Mean squared error over `merge` consecutive micro-batches."""

        def __init__(self, config=None):
            self._merge = int((config or {}).get("merge", 1))
            self._fail_on = (config or {}).get("fail_on")

        def run(self, session):
            if session.iteration == self._fail_on:
                self._fail_on = None
                raise RuntimeError("loss step failed on purpose")
            wrapped = self.get_dependency("ddp").wrapped_model
            records.setdefault("synced", []).append(wrapped.syncing)
            first = (session.iteration - 1) * self._merge + 1
            parts = [batch(first + offset) for offset in range(self._merge)]
            x = torch.cat([p[0] for p in parts])
            y = torch.cat([p[1] for p in parts])
            prediction = wrapped(x)
            prediction.register_hook(
                lambda grad: records.setdefault("backward_autocast", []).append(
                    torch.is_autocast_enabled("cpu")
                )
            )
            session.iteration_context["loss"] = (
                (prediction.float() - y) ** 2
            ).mean()


def _config(tmp_path, *, max_iterations=3, optimizer=None, loss=None, **extra):
    config = make_config(tmp_path, max_iterations=max_iterations)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {"model": "chain_model"},
        "chain_model": {},
        "chain_loss": loss or {},
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "master_addr": "localhost",
            "master_port": "12355",
        },
        "optimizer": optimizer or {
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        },
    })
    config.update(extra)
    return config


def _session(tmp_path, records=None, **config):
    _register_components({} if records is None else records)
    session = TrainingSession(_config(tmp_path, **config))
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    return session


def _run(session, iterations=None):
    with session:
        if iterations is None:
            return list(session)
        return [next(session) for _ in range(iterations)]


def _linear(session) -> nn.Linear:
    return resource_named(session, "chain_model").linear


def _reference(
        optimizer_factory,
        iterations,
        *,
        merge=1,
        scheduler_factory=None,
        clip=None,
):
    """The same training written directly in torch."""
    model = fresh_model()
    optimizer = optimizer_factory(model.parameters())
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    for iteration in range(1, iterations + 1):
        first = (iteration - 1) * merge + 1
        parts = [batch(first + offset) for offset in range(merge)]
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])
        optimizer.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model


# -- the chain -----------------------------------------------------------------


@pytest.mark.parametrize("name, kwargs, factory", [
    ("SGD", {"lr": 0.1, "momentum": 0.9},
     lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
    ("AdamW", {"lr": 0.05, "weight_decay": 0.1},
     lambda p: torch.optim.AdamW(p, lr=0.05, weight_decay=0.1)),
])
def test_training_matches_plain_torch(tmp_path, name, kwargs, factory):
    session = _session(tmp_path, max_iterations=4, optimizer={
        "optimizer": {"name": name, "kwargs": kwargs},
        "lr_scheduler": {"stages": [{
            "name": "StepLR", "kwargs": {"step_size": 1, "gamma": 0.5},
        }]},
    })
    _run(session)

    expected = _reference(
        factory, 4,
        scheduler_factory=lambda o: torch.optim.lr_scheduler.StepLR(
            o, step_size=1, gamma=0.5,
        ),
    )
    torch.testing.assert_close(_linear(session).weight, expected.weight)
    torch.testing.assert_close(_linear(session).bias, expected.bias)


def test_configuring_the_optimizer_brings_in_the_whole_chain(tmp_path):
    session = _session(tmp_path)
    graph = session.execution_graph()

    order = [
        graph.index(f"Step.{name}.run()")
        for name in (
            "chain_loss", "backward", "freeze_gradients",
            "clip_gradients", "optimizer_step",
        )
    ]
    assert order == sorted(order)
    assert "activates: Step.optimizer_step" in graph
    assert "Hook.forward_context.pre_iteration_callback()" in graph


def test_a_loss_nobody_writes_is_a_start_up_error_naming_the_key(tmp_path):
    _register_components({})
    config = _config(tmp_path)
    config["backward"] = {"loss_key": "objective"}

    with pytest.raises(
            RuntimeError,
            match=r"Step.backward reads iteration_context key 'objective', "
                  r"which no step or hook writes.*Keys written in this "
                  r"session: loss",
    ):
        TrainingSession(config)


def test_a_metric_schedule_whose_metric_nobody_writes_fails_at_start_up(
        tmp_path,
):
    _register_components({})

    with pytest.raises(RuntimeError, match="reads iteration_context key 'val_loss'"):
        TrainingSession(_config(tmp_path, optimizer={
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
            "lr_scheduler": {
                "stages": [{"name": "ReduceLROnPlateau"}],
                "metric_key": "val_loss",
            },
        }))


def test_a_checkpoint_of_the_former_optimizer_hook_does_not_restore(tmp_path):
    session = _session(tmp_path)
    state = session.get_state()
    # What a checkpoint written before the optimizer became a resource holds.
    state["components_state"]["optimizer"]["component_type"] = "Hook"

    with pytest.raises(ValueError, match="stored as a Hook"):
        TrainingSession.from_state(state)


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path):
    optimizer = {
        "optimizer": {"name": "AdamW", "kwargs": {"lr": 0.05}},
        "lr_scheduler": {"stages": [{
            "name": "CosineAnnealingLR", "kwargs": {"T_max": "$max_iterations"},
        }]},
        "precision": "fp16",
    }
    uninterrupted = _session(tmp_path / "a", max_iterations=4, optimizer=optimizer)
    _run(uninterrupted)

    paused = _session(tmp_path / "b", max_iterations=4, optimizer=optimizer)
    _run(paused, 2)
    resumed = TrainingSession.from_state(paused.get_state())
    with resumed:
        assert list(resumed) == [3, 4]

    torch.testing.assert_close(_linear(resumed).weight, _linear(uninterrupted).weight)
    resumed_state = resource_named(resumed, "optimizer").get_state()
    reference_state = resource_named(uninterrupted, "optimizer").get_state()
    assert resumed_state["lr_scheduler_state"]["last_epoch"] == 4
    assert resumed_state["grad_scaler_state"] == reference_state["grad_scaler_state"]


# -- accumulation -----------------------------------------------------------------


def test_accumulated_micro_batches_match_one_larger_batch(tmp_path):
    accumulated = _session(tmp_path / "a", max_iterations=4, optimizer={
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        "accumulate_steps": 2,
    })
    _run(accumulated)

    expected = _reference(
        lambda p: torch.optim.SGD(p, lr=0.1), 2, merge=2,
    )
    torch.testing.assert_close(_linear(accumulated).weight, expected.weight)
    torch.testing.assert_close(_linear(accumulated).bias, expected.bias)


def test_accumulation_steps_at_boundaries_and_the_final_iteration(tmp_path):
    records = {}
    session = _session(tmp_path, records, max_iterations=5, optimizer={
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        "lr_scheduler": {"stages": [{
            "name": "CosineAnnealingLR", "kwargs": {"T_max": "$max_iterations"},
        }]},
        "accumulate_steps": 2,
    })
    _run(session)

    scheduler_state = resource_named(session, "optimizer").get_state()[
        "lr_scheduler_state"
    ]
    # Steps at iterations 2, 4 and 5; the schedule counts those three.
    assert scheduler_state["last_epoch"] == 3
    assert scheduler_state["T_max"] == math.ceil(5 / 2)
    # Gradients are synchronised only on the iterations that step.
    assert records["synced"] == [False, True, False, True, True]


# -- precision --------------------------------------------------------------------


def test_bf16_runs_the_forward_pass_under_autocast_and_backward_outside(tmp_path):
    records = {}
    session = _session(tmp_path, records, max_iterations=2, optimizer={
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        "precision": "bf16",
    })
    _run(session)

    assert records["forward_autocast"] == [True, True]
    assert records["backward_autocast"] == [False, False]
    assert not torch.is_autocast_enabled("cpu")


def test_a_failed_iteration_leaves_no_autocast_behind(tmp_path):
    session = _session(
        tmp_path,
        max_iterations=2,
        optimizer={
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
            "precision": "bf16",
        },
        loss={"fail_on": 1},
    )
    with session:
        with pytest.raises(RuntimeError, match="failed on purpose"):
            next(session)
        assert next(session) == 1
        assert next(session) == 2
    assert not torch.is_autocast_enabled("cpu")


# -- gradient processors -------------------------------------------------------------


def test_frozen_parameters_do_not_move_even_under_weight_decay(tmp_path):
    session = _session(
        tmp_path,
        max_iterations=3,
        optimizer={"optimizer": {
            "name": "AdamW", "kwargs": {"lr": 0.1, "weight_decay": 0.5},
        }},
        freeze_gradients={"rules": [{"match": "*bias", "until_iteration": 2}]},
    )
    with session:
        next(session)
        next(session)
        torch.testing.assert_close(
            _linear(session).bias, torch.tensor(INITIAL_BIAS),
        )
        assert not torch.equal(
            _linear(session).weight.detach(), torch.tensor(INITIAL_WEIGHT),
        )
        next(session)
        assert not torch.equal(
            _linear(session).bias.detach(), torch.tensor(INITIAL_BIAS),
        )


def test_a_freeze_pattern_that_matches_nothing_is_reported(tmp_path):
    session = _session(
        tmp_path,
        freeze_gradients={"rules": [{"match": "*head*", "until_iteration": 2}]},
    )
    with session:
        with pytest.raises(ValueError, match=r"\['\*head\*'\] match no parameter"):
            next(session)


def test_clipping_caps_the_norm_and_reports_it_before_clipping(tmp_path):
    session = _session(
        tmp_path,
        max_iterations=1,
        clip_gradients={"max_norm": 0.5},
    )
    _run(session)

    expected = _reference(lambda p: torch.optim.SGD(p, lr=0.1), 1, clip=0.5)
    torch.testing.assert_close(_linear(session).weight, expected.weight)
    model = fresh_model()
    x, y = batch(1)
    ((model(x) - y) ** 2).mean().backward()
    unclipped = nn.utils.get_total_norm([p.grad for p in model.parameters()])
    assert unclipped > 0.5
    assert resource_named(session, "optimizer").grad_norm == pytest.approx(
        float(unclipped)
    )


def test_tracking_the_norm_does_not_change_training(tmp_path):
    session = _session(
        tmp_path, max_iterations=2, clip_gradients={"track_norm": True},
    )
    _run(session)

    expected = _reference(lambda p: torch.optim.SGD(p, lr=0.1), 2)
    torch.testing.assert_close(_linear(session).weight, expected.weight)
    assert resource_named(session, "optimizer").grad_norm is not None


def test_clipping_can_be_changed_by_an_extension(tmp_path):
    session = _session(tmp_path, max_iterations=1)
    _run(session)

    session.apply_extension_overrides([
        "session_config.max_iterations=2",
        "clip_gradients.max_norm=0.5",
    ])

    assert component_named(session, "clip_gradients")._cfg.max_norm == 0.5


def _declare_doubling_stage(*, after: str):
    @requires_step(after)
    @step("chain_doubling")
    class Doubling(GradientProcessor):
        def process(self, session, named_parameters):
            for _, parameter in named_parameters:
                parameter.grad.mul_(2)


def test_a_bound_custom_stage_runs_between_clipping_and_the_step(tmp_path):
    _declare_doubling_stage(after="clip_gradients")
    session = _session(
        tmp_path,
        max_iterations=1,
        component_bindings={
            "model": "chain_model",
            "optimizer_step": {"clip_gradients": "chain_doubling"},
        },
        chain_doubling={},
    )
    _run(session)

    expected = _reference(lambda p: torch.optim.SGD(p, lr=0.2), 1)
    torch.testing.assert_close(_linear(session).weight, expected.weight)


def test_a_stage_that_would_run_after_the_step_is_refused_before_it(tmp_path):
    _declare_doubling_stage(after="optimizer_step")
    session = _session(tmp_path, max_iterations=1, chain_doubling={})

    with session:
        with pytest.raises(RuntimeError, match=r"\['chain_doubling'\] had not run"):
            next(session)
    torch.testing.assert_close(
        _linear(session).weight, torch.tensor(INITIAL_WEIGHT),
    )


# -- param groups and extension -------------------------------------------------------


def test_param_groups_apply_their_kwargs_first_match_first(tmp_path):
    session = _session(tmp_path, max_iterations=1, optimizer={
        "optimizer": {"name": "AdamW", "kwargs": {"lr": 0.1, "weight_decay": 0.3}},
        "param_groups": [
            {"match": ["*bias"], "kwargs": {"weight_decay": 0.0}},
            {"match": ["*bias", "*weight"], "kwargs": {"lr": 0.01}},
        ],
    })
    _run(session)

    groups = resource_named(session, "optimizer").get_state()[
        "optimizer_state"
    ]["param_groups"]
    assert [len(group["params"]) for group in groups] == [1, 1, 0]
    assert groups[0]["weight_decay"] == 0.0 and groups[0]["lr"] == 0.1
    assert groups[1]["weight_decay"] == 0.3 and groups[1]["lr"] == 0.01


def test_a_param_group_pattern_that_matches_nothing_is_reported(tmp_path):
    session = _session(tmp_path, optimizer={
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        "param_groups": [{"match": "*head*", "kwargs": {"lr": 0.0}}],
    })

    with pytest.raises(ValueError, match="match no parameter"):
        with session:
            pass


def test_an_extended_kwarg_leaves_a_group_that_sets_its_own(tmp_path):
    session = _session(tmp_path, max_iterations=1, optimizer={
        "optimizer": {"name": "AdamW", "kwargs": {"lr": 0.1, "weight_decay": 0.3}},
        "param_groups": [{"match": ["*bias"], "kwargs": {"weight_decay": 0.0}}],
    })
    _run(session)

    session.apply_extension_overrides([
        "session_config.max_iterations=2",
        "optimizer.optimizer.kwargs.weight_decay=0.2",
    ])

    groups = resource_named(session, "optimizer").get_state()[
        "optimizer_state"
    ]["param_groups"]
    assert [group["weight_decay"] for group in groups] == [0.0, 0.2]


@pytest.mark.parametrize("override", [
    "optimizer.precision=bf16",
    "optimizer.accumulate_steps=2",
])
def test_settings_that_change_the_state_cannot_be_extended(tmp_path, override):
    session = _session(tmp_path, max_iterations=1)
    _run(session)

    with pytest.raises(ValueError, match="does not allow changes"):
        session.apply_extension_overrides([override])


def _learning_rates(session, iterations=None) -> list[float]:
    """The lr after each iteration run."""
    optimizer = resource_named(session, "optimizer")
    with session:
        run = (
            session if iterations is None
            else (next(session) for _ in range(iterations))
        )
        return [optimizer.current_lrs[0] for _ in run]


def _extended(session, overrides) -> TrainingSession:
    """The session restored from its state and extended, as
    `--extend-session` does from a checkpoint."""
    extended = TrainingSession.from_state(session.get_state())
    extended.apply_extension_overrides(overrides)
    return extended


COSINE_TO_A_FLOOR = {
    "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
    "lr_scheduler": {"stages": [{
        "name": "CosineAnnealingLR",
        "kwargs": {"T_max": "$max_iterations", "eta_min": 0.01},
    }]},
}


def test_an_extension_holds_a_finished_schedule_at_its_final_lr(tmp_path):
    session = _session(tmp_path, max_iterations=4, optimizer=COSINE_TO_A_FLOOR)
    assert _learning_rates(session)[-1] == pytest.approx(0.01)

    session = _extended(session, ["session_config.max_iterations=8"])
    extended = _learning_rates(session, 2)
    resumed = TrainingSession.from_state(session.get_state())
    extended += _learning_rates(resumed)

    # Without the hold, the cosine climbs back towards 0.1.
    assert extended == pytest.approx([0.01] * 4)


def test_an_extension_past_a_one_cycle_schedule_does_not_fail(tmp_path):
    session = _session(tmp_path, max_iterations=3, optimizer={
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        "lr_scheduler": {"stages": [{
            "name": "OneCycleLR",
            "kwargs": {"max_lr": 0.1, "total_steps": "$max_iterations"},
        }]},
    })
    final_lr = _learning_rates(session)[-1]

    session = _extended(session, ["session_config.max_iterations=5"])

    assert _learning_rates(session) == pytest.approx([final_lr] * 2)


def test_a_replacement_schedule_runs_over_the_steps_left(tmp_path):
    session = _session(tmp_path, max_iterations=4, optimizer=COSINE_TO_A_FLOOR)
    _run(session)

    session = _extended(session, [
        "session_config.max_iterations=8",
        "optimizer.lr_scheduler.stages=[{name: CosineAnnealingLR, "
        "kwargs: {T_max: $max_iterations, eta_min: 0.02}}]",
    ])
    lrs = _learning_rates(session)

    scheduler_state = resource_named(session, "optimizer").get_state()[
        "lr_scheduler_state"
    ]
    assert scheduler_state["T_max"] == 4
    # It restarts from the base lr and reaches its own floor at the end.
    assert lrs[0] > 0.05
    assert lrs[-1] == pytest.approx(0.02)


def test_a_replacement_schedule_counts_optimizer_steps_left(tmp_path):
    session = _session(tmp_path, max_iterations=5, optimizer={
        **COSINE_TO_A_FLOOR, "accumulate_steps": 2,
    })
    _run(session)

    session = _extended(session, [
        "session_config.max_iterations=10",
        "optimizer.lr_scheduler.stages=[{name: CosineAnnealingLR, "
        "kwargs: {T_max: $max_iterations, eta_min: 0.02}}]",
    ])
    lrs = _learning_rates(session)

    # Iterations 6 to 10 step at 6, 8 and 10.
    scheduler_state = resource_named(session, "optimizer").get_state()[
        "lr_scheduler_state"
    ]
    assert scheduler_state["T_max"] == 3
    assert scheduler_state["last_epoch"] == 3
    assert lrs[-1] == pytest.approx(0.02)


# -- reporting ------------------------------------------------------------------------


def test_the_logger_reports_learning_rates_and_the_gradient_norm(tmp_path, capsys):
    _register_components({})
    config = _config(
        tmp_path, max_iterations=2, clip_gradients={"track_norm": True},
    )
    config["logger"] = {"log_every": 1}
    session = TrainingSession(config)
    session.unregister_hook("checkpointer")
    _run(session)

    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Iteration")
    ]
    assert lines[0] == "Iteration 1/2 | lr: 1.000e-01"
    assert lines[1].startswith("Iteration 2/2 | lr: 1.000e-01 | grad_norm: ")


# -- serialization ------------------------------------------------------------------------


EVERY_STAGE = {
    "optimizer": {
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1, "momentum": 0.9}},
        "precision": "bf16",
        "accumulate_steps": 2,
    },
    "freeze_gradients": {"rules": [{"match": "*bias", "until_iteration": 2}]},
    "clip_gradients": {"max_norm": 0.5},
}


def _pickled(component):
    return pickle.loads(pickle.dumps(component))


def test_pickled_chain_components_train_like_fresh_ones(tmp_path):
    fresh = _session(tmp_path / "fresh", max_iterations=6, **EVERY_STAGE)
    _run(fresh)

    session = _session(tmp_path / "copies", max_iterations=6, **EVERY_STAGE)
    hook = next(h for h in session.get_all_hooks() if h.name == "forward_context")
    session.unregister_hook("forward_context")
    session.register_hook(_pickled(hook))
    for name in ("backward", "freeze_gradients", "clip_gradients", "optimizer_step"):
        original = next(s for s in session.get_all_steps() if s.name == name)
        session.remove_step(name)
        session.add_step(_pickled(original))
    _run(session)

    torch.testing.assert_close(_linear(session).weight, _linear(fresh).weight)
    torch.testing.assert_close(_linear(session).bias, _linear(fresh).bias)
    assert (
        resource_named(session, "optimizer").grad_norm
        == resource_named(fresh, "optimizer").grad_norm
    )


def test_a_run_with_every_stage_configured_resumes_exactly(tmp_path):
    uninterrupted = _session(tmp_path / "a", max_iterations=8, **EVERY_STAGE)
    _run(uninterrupted)

    paused = _session(tmp_path / "b", max_iterations=8, **EVERY_STAGE)
    # Resumed at a group boundary: a partly accumulated group is not saved.
    _run(paused, 4)
    resumed = TrainingSession.from_state(paused.get_state())
    with resumed:
        assert list(resumed) == [5, 6, 7, 8]

    torch.testing.assert_close(_linear(resumed).weight, _linear(uninterrupted).weight)
    torch.testing.assert_close(_linear(resumed).bias, _linear(uninterrupted).bias)
