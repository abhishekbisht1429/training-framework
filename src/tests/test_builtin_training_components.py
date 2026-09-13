from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training_framework.components.builtin import OptimizerHook, Timer
from training_framework.components.builtin import distributed, observability
from training_framework.components import (
    Resource,
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)
from training_framework.session import TrainingSession


class FakeDistributedDataParallel(nn.Module):
    def __init__(self, module, device_ids):
        super().__init__()
        self.module = module
        self.device_ids = device_ids

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _session_config(tmp_path, *, max_iterations=3):
    return {
        "rng_seed": 41,
        "sessions_dir": str(tmp_path),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def _register_training_components():
    @resource("public_test_model")
    class PublicTestModel(nn.Module, StatefulResource):
        def __init__(self, config):
            nn.Module.__init__(self)
            self.weight = nn.Parameter(
                torch.tensor(float(config["initial_weight"]))
            )

        def forward(self, value):
            return self.weight * value

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {"weight": self.weight.detach().clone()}

        def set_state(self, state):
            with torch.no_grad():
                self.weight.copy_(state["weight"])

    @step("public_test_loss")
    @requires_resource("ddp")
    class PublicTestLoss(Step):
        def __init__(self, config):
            self.target = float(config["target"])

        def run(self, session):
            wrapped_model = session.get_resource("ddp").wrapped_model
            prediction = wrapped_model(torch.tensor(1.0))
            session.iteration_context["loss"] = (
                prediction - self.target
            ).square()


def _training_config(tmp_path, *, max_iterations=3):
    return {
        "session_config": _session_config(
            tmp_path,
            max_iterations=max_iterations,
        ),
        "component_bindings": {
            "model": "public_test_model",
        },
        "public_test_model": {"initial_weight": 1.0},
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "parallel_components": [],
            "master_addr": "localhost",
            "master_port": "12355",
        },
        "optimizer": {
            "optimizer": {
                "name": "AdamW",
                "kwargs": {
                    "lr": 0.1,
                    "weight_decay": 0.0,
                },
            },
            "lr_scheduler": {
                "stages": [
                    {
                        "name": "LinearLR",
                        "kwargs": {
                            "start_factor": 0.001,
                            "total_iters": "$stage_iterations",
                        },
                    },
                    {
                        "name": "CosineAnnealingLR",
                        "kwargs": {"T_max": "$stage_iterations"},
                    },
                ],
                "milestones": [1],
            },
        },
        "public_test_loss": {"target": 0.0},
    }


def _patch_distributed_boundaries(monkeypatch):
    calls = {
        "initializations": [],
        "destroy_count": 0,
    }

    def init_process_group(**kwargs):
        calls["initializations"].append(kwargs)

    def destroy_process_group():
        calls["destroy_count"] += 1

    monkeypatch.setattr(
        distributed,
        "DDP",
        FakeDistributedDataParallel,
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        init_process_group,
    )
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        destroy_process_group,
    )
    return calls


def _remove_default_hooks(session):
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")


def _optimizer_hook(session):
    return next(
        hook
        for hook in session.get_all_hooks()
        if hook.name == "optimizer"
    )


def test_pickled_timer_formats_iteration_and_elapsed_durations(
        monkeypatch,
        capsys,
):
    timestamps = iter((100, 130, 190))
    monkeypatch.setattr(
        observability.time,
        "time_ns",
        timestamps.__next__,
    )
    timer = pickle.loads(pickle.dumps(
        Timer({"call_every": 1})
    ))
    session = SimpleNamespace(iteration=2)

    timer.pre_session(session)
    timer.pre_iteration_callback(session)
    timer.post_iteration_callback(session)

    assert capsys.readouterr().out == (
        "Time taken for the iteration 2: 60 ns\n"
        "Elapsed time: 90 ns\n\n"
    )


def test_optional_builtins_do_not_break_unrelated_sessions(tmp_path):
    session = TrainingSession({
        "session_config": _session_config(tmp_path, max_iterations=1),
    })
    _remove_default_hooks(session)

    assert "TRAINING SESSION EXECUTION GRAPH" in session.execution_graph()
    with session:
        assert list(session) == [1]



def test_data_manager_runs_through_public_session_lifecycle(
        tmp_path,
        monkeypatch,
):
    _register_training_components()
    _patch_distributed_boundaries(monkeypatch)

    @resource("public_test_dataset")
    class PublicTestDataset(Resource):
        def __init__(self, config):
            self._size = int(config["size"])

        def __len__(self):
            return self._size

        def __getitem__(self, index):
            return torch.tensor([float(index), float(index + 10)])

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    config = _training_config(tmp_path)
    config["component_bindings"]["dataset"] = "public_test_dataset"
    config["public_test_dataset"] = {"size": 4}
    config["data_manager"] = {
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
    }
    config["ddp"]["world_size"] = 2
    del config["optimizer"]
    del config["public_test_loss"]

    session = TrainingSession(config)
    _remove_default_hooks(session)
    placeholder_ddp = session.get_resource("ddp")
    ranked_ddp = type(placeholder_ddp)(
        config=placeholder_ddp.config,
        rank=0,
    )
    session.unregister_resource("ddp")
    session.register_resource(ranked_ddp)

    data_manager = session.get_resource("data_manager")
    graph = session.execution_graph()
    manager_setup = graph.index("Resource.data_manager.setup()")
    assert graph.index("Resource.public_test_dataset.setup()") < manager_setup
    assert graph.index("Resource.ddp.setup()") < manager_setup
    assert data_manager.batch_size == 4
    assert data_manager.data_iter is None

    with session:
        assert data_manager.data_iter is not None
        batch = next(data_manager.data_iter)
        assert batch.shape == (2, 2)
        torch.testing.assert_close(
            batch[:, 1] - batch[:, 0],
            torch.full((2,), 10.0),
        )

    assert data_manager.data_iter is None


def test_ddp_resource_activates_its_model_dependency(tmp_path):
    @resource("model")
    class UnconfiguredModel(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    session = TrainingSession({
        "session_config": _session_config(tmp_path),
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "parallel_components": [],
            "master_addr": "localhost",
            "master_port": "12355",
        },
    })

    assert session.has_resource("model")
    graph = session.execution_graph()
    assert graph.index("Resource.model.setup()") < graph.index(
        "Resource.ddp.setup()"
    )


def test_pickled_ddp_resource_and_optimizer_run_through_public_session_api(
        tmp_path,
        monkeypatch,
):
    _register_training_components()
    distributed_calls = _patch_distributed_boundaries(monkeypatch)
    session = TrainingSession(_training_config(tmp_path))
    _remove_default_hooks(session)

    ddp = session.get_resource("ddp")
    ddp = pickle.loads(pickle.dumps(ddp))
    session.unregister_resource("ddp")
    session.register_resource(ddp)
    model = session.get_resource("model")
    optimizer = _optimizer_hook(session)
    graph = session.execution_graph()

    ddp_setup = graph.index("Resource.ddp.setup()")
    assert graph.index("Resource.public_test_model.setup()") < ddp_setup
    assert ddp_setup < graph.index("Hook.optimizer.pre_session()")
    assert "requires: Resource.public_test_model" in graph
    assert "requires: Resource.ddp" in graph

    initial_weight = model.weight.detach().clone()
    with session:
        assert isinstance(
            ddp.wrapped_model,
            FakeDistributedDataParallel,
        )
        assert ddp.wrapped_model.module is model
        assert ddp.wrapped_model.device_ids is None
        assert distributed_calls["initializations"] == [{
            "backend": "gloo",
            "rank": -1,
            "world_size": 1,
        }]
        assert next(session) == 1

        optimizer_state = optimizer.get_state()
        assert optimizer_state["optimizer_state"]["state"]
        assert optimizer_state["lr_scheduler_state"]["last_epoch"] == 1

    assert not torch.equal(model.weight.detach(), initial_weight)
    with pytest.raises(
            RuntimeError,
            match="This instance of DDPResource is not initialized yet!",
    ):
        _ = ddp.wrapped_model
    assert distributed_calls["destroy_count"] == 1
    assert optimizer.get_state()["optimizer_state"]["state"]



def test_ddp_resource_moves_model_to_rank_local_cuda_before_wrapping(
        tmp_path,
        monkeypatch,
):
    _register_training_components()
    _patch_distributed_boundaries(monkeypatch)
    config = _training_config(tmp_path)
    config["ddp"]["backend"] = "nccl"
    config["ddp"]["world_size"] = 2
    del config["optimizer"]
    del config["public_test_loss"]

    session = TrainingSession(config)
    _remove_default_hooks(session)
    model = session.get_resource("model")
    placeholder_ddp = session.get_resource("ddp")
    ranked_ddp = type(placeholder_ddp)(
        config=placeholder_ddp.config,
        rank=1,
    )
    session.unregister_resource("ddp")
    session.register_resource(ranked_ddp)

    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda rank: events.append(("set_device", rank)),
    )

    def move_model(model_instance, device):
        events.append(("model_to", device))
        return model_instance

    monkeypatch.setattr(nn.Module, "to", move_model)

    class RecordingDistributedDataParallel(FakeDistributedDataParallel):
        def __init__(self, module, device_ids):
            events.append(("ddp", list(device_ids)))
            super().__init__(module, device_ids)

    monkeypatch.setattr(
        distributed,
        "DDP",
        RecordingDistributedDataParallel,
    )

    with session:
        assert session.device == torch.device("cuda", 1)
        assert ranked_ddp.wrapped_model.device_ids == [1]

    assert events == [
        ("set_device", 1),
        ("model_to", torch.device("cuda", 1)),
        ("ddp", [1]),
    ]


def test_ddp_resource_cleans_up_when_model_wrapping_fails(
        tmp_path,
        monkeypatch,
):
    _register_training_components()
    distributed_calls = _patch_distributed_boundaries(monkeypatch)

    class FailingDistributedDataParallel:
        def __init__(self, module, device_ids):
            raise RuntimeError("could not wrap model")

    monkeypatch.setattr(
        distributed,
        "DDP",
        FailingDistributedDataParallel,
    )
    config = _training_config(tmp_path)
    del config["optimizer"]
    del config["public_test_loss"]
    session = TrainingSession(config)
    _remove_default_hooks(session)
    ddp = session.get_resource("ddp")

    with pytest.raises(RuntimeError, match="could not wrap model"):
        with session:
            pass

    assert distributed_calls["destroy_count"] == 1
    with pytest.raises(
            RuntimeError,
            match="This instance of DDPResource is not initialized yet!",
    ):
        _ = ddp.wrapped_model


def test_pickled_optimizer_state_matches_uninterrupted_training(
        tmp_path,
        monkeypatch,
):
    _register_training_components()
    _patch_distributed_boundaries(monkeypatch)
    config = _training_config(tmp_path, max_iterations=3)

    uninterrupted = TrainingSession(config)
    _remove_default_hooks(uninterrupted)
    with uninterrupted:
        assert list(uninterrupted) == [1, 2, 3]

    paused = TrainingSession(config)
    _remove_default_hooks(paused)
    with paused:
        assert next(paused) == 1

    restored_optimizer = pickle.loads(pickle.dumps(
        _optimizer_hook(paused)
    ))
    restored = TrainingSession.from_state(paused.get_state())
    restored.unregister_hook("optimizer")
    restored.register_hook(restored_optimizer)

    with restored:
        assert list(restored) == [2, 3]

    torch.testing.assert_close(
        restored.get_resource("model").weight,
        uninterrupted.get_resource("model").weight,
    )
    assert (
        _optimizer_hook(restored).get_state()["lr_scheduler_state"][
            "last_epoch"
        ]
        == _optimizer_hook(uninterrupted).get_state()[
            "lr_scheduler_state"
        ]["last_epoch"]
        == 3
    )


def _optimizer_test_session(model, *, max_iterations, iteration_context=None):
    ddp = SimpleNamespace(wrapped_model=model)
    return SimpleNamespace(
        get_resource=lambda name: ddp,
        session_config=SimpleNamespace(max_iterations=max_iterations),
        iteration_context=(iteration_context or {}),
    )


def test_optimizer_can_be_configured_without_a_scheduler():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {
            "name": "SGD",
            "kwargs": {"lr": 0.25},
        },
    })
    session = _optimizer_test_session(model, max_iterations=1)
    initial_weight = model.weight.detach().clone()

    hook.pre_session(session)
    hook.pre_iteration_callback(session)
    session.iteration_context["loss"] = (
        model(torch.ones(1, 1)).square().sum()
    )
    hook.post_iteration_callback(session)

    assert not torch.equal(model.weight.detach(), initial_weight)
    state = hook.get_state()
    assert state["optimizer_state"]["param_groups"][0]["lr"] == 0.25
    assert state["lr_scheduler_state"] is None

    hook.post_session(session)
    restored = pickle.loads(pickle.dumps(hook))
    restored_state = restored.get_state()
    assert restored_state["optimizer_state"]["param_groups"][0]["lr"] == 0.25
    assert restored_state["lr_scheduler_state"] is None


def test_scheduler_pipeline_resolves_runtime_stage_lengths():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {
            "name": "SGD",
            "kwargs": {"lr": 0.2},
        },
        "lr_scheduler": {
            "stages": [
                {
                    "name": "LinearLR",
                    "kwargs": {
                        "start_factor": 0.5,
                        "total_iters": "$stage_iterations",
                    },
                },
                {
                    "name": "CosineAnnealingLR",
                    "kwargs": {"T_max": "$stage_iterations"},
                },
            ],
            "milestones": [2],
        },
    })
    session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(session)

    for _ in range(5):
        hook.pre_iteration_callback(session)
        session.iteration_context["loss"] = (
            model(torch.ones(1, 1)).square().sum()
        )
        hook.post_iteration_callback(session)

    assert hook.get_state()["lr_scheduler_state"]["last_epoch"] == 5


def test_metric_scheduler_reads_the_configured_iteration_value():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {
            "name": "SGD",
            "kwargs": {"lr": 0.2},
        },
        "lr_scheduler": {
            "stages": [{
                "name": "ReduceLROnPlateau",
                "kwargs": {
                    "mode": "min",
                    "patience": 0,
                    "factor": 0.5,
                },
            }],
            "metric_key": "validation_loss",
        },
    })
    session = _optimizer_test_session(model, max_iterations=2)
    hook.pre_session(session)

    for metric in (1.0, 2.0):
        hook.pre_iteration_callback(session)
        session.iteration_context.update({
            "loss": model(torch.ones(1, 1)).square().sum(),
            "validation_loss": metric,
        })
        hook.post_iteration_callback(session)

    assert (
        hook.get_state()["optimizer_state"]["param_groups"][0]["lr"]
        == 0.1
    )


def test_metric_scheduler_reports_a_missing_iteration_value():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
        "lr_scheduler": {
            "stages": [{"name": "ReduceLROnPlateau"}],
            "metric_key": "validation_loss",
        },
    })
    session = _optimizer_test_session(model, max_iterations=1)
    hook.pre_session(session)
    session.iteration_context["loss"] = (
        model(torch.ones(1, 1)).square().sum()
    )

    with pytest.raises(KeyError, match="validation_loss"):
        hook.post_iteration_callback(session)


def test_scheduler_milestones_are_bounded_by_the_session():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
        "lr_scheduler": {
            "stages": [
                {"name": "LinearLR"},
                {"name": "CosineAnnealingLR", "kwargs": {"T_max": 1}},
            ],
            "milestones": [2],
        },
    })
    session = _optimizer_test_session(model, max_iterations=2)

    with pytest.raises(ValueError, match="less than"):
        hook.pre_session(session)


def test_optimizer_extension_preserves_state_and_replaces_hyperparameters():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    optimizer_state = optimizer.state_dict()

    hook = OptimizerHook({
        "optimizer": {
            "name": "SGD",
            "kwargs": {"lr": 0.2, "momentum": 0.9},
        },
    })
    hook.set_state({
        "optimizer_state": optimizer_state,
        "lr_scheduler_state": None,
    })
    original_momentum = optimizer_state["state"][0]["momentum_buffer"].clone()

    hook.apply_extension_config(
        {
            "optimizer": {
                "name": "SGD",
                "kwargs": {"lr": 0.05, "momentum": 0.8},
            },
        },
        frozenset({
            ("optimizer", "kwargs", "lr"),
            ("optimizer", "kwargs", "momentum"),
        }),
    )
    extended_state = pickle.loads(pickle.dumps(hook)).get_state()

    assert extended_state["optimizer_state"]["param_groups"][0]["lr"] == 0.05
    assert extended_state["optimizer_state"]["param_groups"][0]["momentum"] == 0.8
    torch.testing.assert_close(
        extended_state["optimizer_state"]["state"][0]["momentum_buffer"],
        original_momentum,
    )


def test_optimizer_extension_rebase_scales_base_lr_by_current_multiplier():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
        "lr_scheduler": {
            "stages": [{
                "name": "ConstantLR",
                "kwargs": {"factor": 0.5, "total_iters": "$max_iterations"},
            }],
        },
    })
    session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(session)
    hook.pre_iteration_callback(session)
    session.iteration_context["loss"] = model(torch.ones(1, 1)).square().sum()
    hook.post_iteration_callback(session)
    # Still inside the constant phase: current lr is base(0.2) * factor(0.5).
    assert (
        hook.get_state()["optimizer_state"]["param_groups"][0]["lr"]
        == pytest.approx(0.1)
    )
    hook.post_session(session)

    hook.apply_extension_config(
        {
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.05}},
            "lr_scheduler": {
                "stages": [{
                    "name": "ConstantLR",
                    "kwargs": {"factor": 0.5, "total_iters": "$max_iterations"},
                }],
            },
        },
        frozenset({("optimizer", "kwargs", "lr")}),
    )

    # The override (0.05) is the desired *current* effective lr, so the
    # rebased base must be 0.1 (0.05 / factor 0.5), not a flat 0.05 that
    # would ignore the factor the schedule is currently applying.
    assert hook.get_state()["lr_scheduler_state"]["base_lrs"] == pytest.approx(
        [0.1]
    )


def test_optimizer_extension_lr_rebase_does_not_leak_into_a_later_scheduler_stage():
    # A SequentialLR stage that hasn't started yet has no meaningful
    # "current lr" of its own, so an lr override applied mid-earlier-stage
    # must not distort it: it should reach its transition using its own
    # originally configured base, exactly as if no override had happened.
    model = nn.Linear(1, 1, bias=False)
    config = {
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
        "lr_scheduler": {
            "stages": [
                {
                    "name": "LinearLR",
                    "kwargs": {"start_factor": 0.5, "total_iters": 4},
                },
                {"name": "CosineAnnealingLR", "kwargs": {"T_max": 4}},
            ],
            "milestones": [4],
        },
    }
    hook = OptimizerHook(config)
    session = _optimizer_test_session(model, max_iterations=8)
    hook.pre_session(session)
    for _ in range(2):
        hook.pre_iteration_callback(session)
        session.iteration_context["loss"] = (
            model(torch.ones(1, 1)).square().sum()
        )
        hook.post_iteration_callback(session)
    hook.post_session(session)

    # Override lr mid-warmup (still stage 0); lr_scheduler is unchanged.
    hook.apply_extension_config(
        {**config, "optimizer": {"name": "SGD", "kwargs": {"lr": 0.05}}},
        frozenset({("optimizer", "kwargs", "lr")}),
    )

    new_session = _optimizer_test_session(model, max_iterations=8)
    hook.pre_session(new_session)
    for _ in range(2):  # reaches the milestone (stage 1 activates).
        hook.pre_iteration_callback(new_session)
        new_session.iteration_context["loss"] = (
            model(torch.ones(1, 1)).square().sum()
        )
        hook.post_iteration_callback(new_session)

    # At the transition, CosineAnnealingLR uses its own base (0.2), not
    # something derived from the warmup-stage override.
    assert (
        hook.get_state()["optimizer_state"]["param_groups"][0]["lr"]
        == pytest.approx(0.2)
    )


def test_optimizer_extension_allows_scheduler_replacement_and_resets_progress():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2, "momentum": 0.9}},
        "lr_scheduler": {
            "stages": [{
                "name": "CosineAnnealingLR",
                "kwargs": {"T_max": "$max_iterations"},
            }],
        },
    })
    session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(session)
    for _ in range(3):
        hook.pre_iteration_callback(session)
        session.iteration_context["loss"] = (
            model(torch.ones(1, 1)).square().sum()
        )
        hook.post_iteration_callback(session)
    momentum_buffer = hook.get_state()["optimizer_state"]["state"][0][
        "momentum_buffer"
    ].clone()
    assert hook.get_state()["lr_scheduler_state"]["last_epoch"] == 3
    hook.post_session(session)

    hook.apply_extension_config(
        {
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2, "momentum": 0.9}},
            "lr_scheduler": {
                "stages": [{
                    "name": "LinearLR",
                    "kwargs": {
                        "start_factor": 0.1,
                        "total_iters": "$max_iterations",
                    },
                }],
            },
        },
        frozenset({("lr_scheduler", "stages")}),
    )

    reset_state = hook.get_state()
    assert reset_state["lr_scheduler_state"] is None
    torch.testing.assert_close(
        reset_state["optimizer_state"]["state"][0]["momentum_buffer"],
        momentum_buffer,
    )

    new_session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(new_session)
    state = hook.get_state()
    assert state["lr_scheduler_state"]["last_epoch"] == 0
    # Clean restart: LinearLR's start_factor must scale the configured base
    # lr (0.2), not wherever the discarded CosineAnnealingLR schedule had
    # progressed the optimizer's current lr to. LinearLR scales *current*
    # group['lr'] at construction time, so that stale progressed value
    # would otherwise silently compound into the replacement's own factor.
    assert state["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(
        0.02
    )
    assert state["lr_scheduler_state"]["base_lrs"] == pytest.approx([0.2])


def test_optimizer_extension_lr_override_survives_scheduler_replacement():
    # Combining an lr override with a scheduler replacement in the same
    # extension must not let a stale `initial_lr` (stamped by the old
    # scheduler on the optimizer's param_groups) resurface: the freshly
    # constructed scheduler would otherwise setdefault onto it and silently
    # revert the override on its first step.
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
        "lr_scheduler": {
            "stages": [{
                "name": "CosineAnnealingLR",
                "kwargs": {"T_max": "$max_iterations"},
            }],
        },
    })
    session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(session)
    hook.post_session(session)

    hook.apply_extension_config(
        {
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.05}},
            "lr_scheduler": {
                "stages": [{
                    "name": "CosineAnnealingWarmRestarts",
                    "kwargs": {"T_0": "$max_iterations"},
                }],
            },
        },
        frozenset({
            ("optimizer", "kwargs", "lr"),
            ("lr_scheduler", "stages"),
        }),
    )

    new_session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(new_session)

    state = hook.get_state()
    assert (
        state["optimizer_state"]["param_groups"][0]["lr"]
        == pytest.approx(0.05)
    )
    assert (
        state["optimizer_state"]["param_groups"][0]["initial_lr"]
        == pytest.approx(0.05)
    )
    assert state["lr_scheduler_state"]["base_lrs"] == pytest.approx([0.05])


def test_optimizer_extension_clean_restart_uses_optimizer_default_lr():
    # `lr` is never configured here -- AdamW falls back to its own PyTorch
    # default (0.001). A scheduler replacement's clean restart must derive
    # the base from that resolved default, not merely skip the reset
    # because no explicit `lr` kwarg exists in the config.
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "AdamW", "kwargs": {}},
        "lr_scheduler": {
            "stages": [{
                "name": "CosineAnnealingLR",
                "kwargs": {"T_max": "$max_iterations"},
            }],
        },
    })
    session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(session)
    for _ in range(3):
        hook.pre_iteration_callback(session)
        session.iteration_context["loss"] = (
            model(torch.ones(1, 1)).square().sum()
        )
        hook.post_iteration_callback(session)
    hook.post_session(session)

    hook.apply_extension_config(
        {
            "optimizer": {"name": "AdamW", "kwargs": {}},
            "lr_scheduler": {
                "stages": [{
                    "name": "LinearLR",
                    "kwargs": {
                        "start_factor": 0.1,
                        "total_iters": "$max_iterations",
                    },
                }],
            },
        },
        frozenset({("lr_scheduler", "stages")}),
    )

    new_session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(new_session)
    state = hook.get_state()
    assert state["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(
        0.0001
    )
    assert state["lr_scheduler_state"]["base_lrs"] == pytest.approx([0.001])


def test_optimizer_extension_can_add_or_remove_a_scheduler():
    model = nn.Linear(1, 1, bias=False)
    hook = OptimizerHook({
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
    })
    session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(session)
    hook.pre_iteration_callback(session)
    session.iteration_context["loss"] = model(torch.ones(1, 1)).square().sum()
    hook.post_iteration_callback(session)
    hook.post_session(session)

    hook.apply_extension_config(
        {
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}},
            "lr_scheduler": {
                "stages": [{
                    "name": "ConstantLR",
                    "kwargs": {"factor": 1.0, "total_iters": "$max_iterations"},
                }],
            },
        },
        frozenset({("lr_scheduler", "stages")}),
    )

    new_session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(new_session)
    assert hook.get_state()["lr_scheduler_state"] is not None

    hook.post_session(new_session)
    hook.apply_extension_config(
        {"optimizer": {"name": "SGD", "kwargs": {"lr": 0.2}}},
        frozenset({("lr_scheduler",)}),
    )
    another_session = _optimizer_test_session(model, max_iterations=5)
    hook.pre_session(another_session)
    assert hook.get_state()["lr_scheduler_state"] is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"learning_rate": 0.1},
            "Legacy optimizer fields are no longer supported",
        ),
        (
            {"optimizer": {"name": "NotAnOptimizer"}},
            "Unknown optimizer class",
        ),
        (
            {
                "optimizer": {"name": "AdamW"},
                "lr_scheduler": {
                    "stages": [
                        {"name": "LinearLR"},
                        {"name": "CosineAnnealingLR"},
                    ],
                    "milestones": [],
                },
            },
            "exactly one entry between each pair of stages",
        ),
        (
            {
                "optimizer": {"name": "AdamW"},
                "lr_scheduler": {
                    "stages": [
                        {"name": "LinearLR"},
                        {"name": "CosineAnnealingLR"},
                    ],
                    "milestones": [1],
                    "metric_key": "loss",
                },
            },
            "supported only for a single scheduler stage",
        ),
    ],
)
def test_optimizer_configuration_errors_are_actionable(config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        OptimizerHook(config)
