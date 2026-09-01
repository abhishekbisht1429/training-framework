from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training_framework.components.builtin import Timer
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
        "aliases": {
            "model": "public_test_model",
        },
        "model": {"initial_weight": 1.0},
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "parallel_components": [],
            "master_addr": "localhost",
            "master_port": "12355",
        },
        "optimizer": {
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "warmup_iters": 1,
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
    config["aliases"]["dataset"] = "public_test_dataset"
    config["dataset"] = {"size": 4}
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
        def __init__(self, config):
            self.config = dict(config)

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

    model = session.get_resource("model")
    assert model.config == {}
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
