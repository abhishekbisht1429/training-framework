from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import training_framework.configurator as configurator_sut
import training_framework.training_engine as engine_sut
from training_framework import training_engine
from training_framework.builtin_components import DDPResource


def _parse_args(monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(sys, "argv", ["training-framework", *args])
    return configurator_sut.Configurator()


class IdleSession:
    """Minimal context-managed iterator used by worker tests."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

    def __next__(self):
        raise AssertionError("The stop event should prevent session iteration")


# ---------------------------------------------------------------------------
# Configurator modes
# ---------------------------------------------------------------------------


def test_configurator_resume_mode_exposes_checkpoint_mode_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurator = _parse_args(
        monkeypatch,
        "--resume-session",
        "checkpoints/session.pkl",
        "--process_timeout_on_join",
        "7.5",
    )

    assert configurator.mode == "resume"
    assert configurator.checkpoint_path == "checkpoints/session.pkl"
    assert configurator.process_timeout_on_join == pytest.approx(7.5)


def test_configurator_modes_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            monkeypatch,
            "--config",
            "config.yaml",
            "--resume-session",
            "session.pkl",
        )


def test_configurator_requires_exactly_one_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse_args(monkeypatch)


def test_configurator_extension_parses_max_iterations_as_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurator = _parse_args(
        monkeypatch,
        "--extend-session",
        "session.pkl",
        "250",
    )

    assert configurator.mode == "extend"
    assert configurator.checkpoint_path == "session.pkl"
    assert configurator.new_max_iters == 250
    assert isinstance(configurator.new_max_iters, int)


def test_configurator_extension_rejects_non_integer_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError):
        _parse_args(
            monkeypatch,
            "--extend-session",
            "session.pkl",
            "not-an-integer",
        )


def test_resume_mode_rejects_new_session_accessors_with_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurator = _parse_args(
        monkeypatch,
        "--resume-session",
        "session.pkl",
    )

    with pytest.raises(KeyError, match="current mode"):
        _ = configurator.session_configs

    with pytest.raises(KeyError, match="current mode"):
        configurator.get_base_config(0)

    with pytest.raises(KeyError, match="current mode"):
        configurator.get_component_config(0, "logger")


# ---------------------------------------------------------------------------
# copy_and_modify_session_for_worker
# ---------------------------------------------------------------------------


@dataclass
class Component:
    name: str


class FakeSession:
    def __init__(
        self,
        *,
        ddp_config: dict[str, Any] | None = None,
        hooks: tuple[str, ...] = (),
        resources: tuple[str, ...] = (),
        steps: tuple[str, ...] = (),
    ) -> None:
        self.hooks = {name: Component(name) for name in hooks}
        self.resources = {name: Component(name) for name in resources}
        self.steps = {name: Component(name) for name in steps}
        self.max_iteration_updates: list[int] = []

        if ddp_config is not None:
            self.resources["ddp"] = DDPResource(ddp_config)

    def update_max_iters(self, value: int) -> None:
        self.max_iteration_updates.append(value)

    def has_resource(self, name: str) -> bool:
        return name in self.resources

    def get_resource(self, name: str):
        return self.resources[name]

    def get_all_hooks(self) -> list[Any]:
        return list(self.hooks.values())

    def get_all_resources(self) -> list[Any]:
        return list(self.resources.values())

    def get_all_steps(self) -> list[Any]:
        return list(self.steps.values())

    def unregister_hook(self, name: str) -> None:
        del self.hooks[name]

    def unregister_resource(self, name: str) -> None:
        del self.resources[name]

    def register_resource(self, resource) -> None:
        self.resources[resource.name] = resource

    def remove_step(self, name: str) -> None:
        del self.steps[name]


def _ddp_config(
    *,
    world_size: int = 4,
    parallel_components: list[str] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "world_size": world_size,
        "backend": "gloo",
    }
    if parallel_components is not None:
        config["parallel_components"] = parallel_components
    return config


def test_copy_and_modify_returns_a_deep_copy_and_updates_iteration_limit() -> None:
    source = FakeSession(hooks=("metrics",), resources=("cache",), steps=("train",))

    result = training_engine.load_session_for_worker(
        source,
        rank=0,
        session_update_params={"max_iterations": 300},
    )

    assert result is not source
    assert result.max_iteration_updates == [300]
    assert source.max_iteration_updates == []
    assert result.hooks is not source.hooks
    assert result.resources is not source.resources
    assert result.steps is not source.steps


def test_copy_and_modify_without_updates_preserves_iteration_configuration() -> None:
    source = FakeSession()

    result = training_engine.load_session_for_worker(source, rank=0)

    assert result is not source
    assert result.max_iteration_updates == []


def test_rank_zero_worker_replaces_placeholder_ddp_resource_with_rank_zero() -> None:
    """Every worker, including rank zero, needs a concrete runtime DDP rank."""
    source = FakeSession(
        ddp_config=_ddp_config(parallel_components=["ddp"]),
    )
    assert source.get_resource("ddp").rank == -1

    result = training_engine.load_session_for_worker(source, rank=0)

    assert result.get_resource("ddp").rank == 0
    assert source.get_resource("ddp").rank == -1


def test_secondary_rank_replaces_ddp_resource_and_keeps_only_parallel_components() -> None:
    source = FakeSession(
        ddp_config=_ddp_config(
            parallel_components=[
                "ddp",
                "parallel_hook",
                "parallel_resource",
                "parallel_step",
            ]
        ),
        hooks=("parallel_hook", "rank_zero_hook"),
        resources=("parallel_resource", "rank_zero_resource"),
        steps=("parallel_step", "rank_zero_step"),
    )

    result = training_engine.load_session_for_worker(source, rank=2)

    assert result.get_resource("ddp").rank == 2
    assert set(result.hooks) == {"parallel_hook"}
    assert set(result.resources) == {"ddp", "parallel_resource"}
    assert set(result.steps) == {"parallel_step"}

    # The parent-owned source session is authoritative and remains unchanged.
    assert source.get_resource("ddp").rank == -1
    assert set(source.hooks) == {"parallel_hook", "rank_zero_hook"}
    assert set(source.resources) == {
        "ddp",
        "parallel_resource",
        "rank_zero_resource",
    }
    assert set(source.steps) == {"parallel_step", "rank_zero_step"}


def test_non_ddp_worker_copy_does_not_prune_components() -> None:
    source = FakeSession(
        hooks=("hook",),
        resources=("resource",),
        steps=("step",),
    )

    result = training_engine.load_session_for_worker(source, rank=5)

    assert set(result.hooks) == {"hook"}
    assert set(result.resources) == {"resource"}
    assert set(result.steps) == {"step"}


def test_ddp_resource_defaults_to_placeholder_rank_and_empty_parallel_list() -> None:
    resource = DDPResource(_ddp_config(world_size=2))

    assert resource.rank == -1
    assert resource.world_size == 2
    assert resource.parallel_components == []


def test_ddp_resource_properties_return_defensive_copies() -> None:
    config = _ddp_config(
        world_size=3,
        parallel_components=["ddp", "train"],
    )
    resource = DDPResource(config)

    returned_config = resource.config
    returned_components = resource.parallel_components
    returned_config["world_size"] = 99
    returned_components.append("mutated")

    assert resource.world_size == 3
    assert resource.parallel_components == ["ddp", "train"]


# ---------------------------------------------------------------------------
# Parent-side session creation
# ---------------------------------------------------------------------------


class RecordingSession:
    def __init__(self, base_config: dict[str, Any]) -> None:
        self.base_config = base_config
        self.steps: list[Any] = []
        self.hooks: list[Any] = []
        self.resources: list[Any] = []

    def add_step(self, component) -> None:
        self.steps.append(component)

    def register_hook(self, component) -> None:
        self.hooks.append(component)

    def register_resource(self, component) -> None:
        self.resources.append(component)


class RecordingComponent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config


class RecordingDDPResource(RecordingComponent):
    name = "ddp"

    def __init__(self, config: dict[str, Any], rank: int = -1) -> None:
        super().__init__(config)
        self.rank = rank
        self.world_size = config["world_size"]
        self.parallel_components = list(config.get("parallel_components", []))


def _install_recording_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(configurator_sut, "TrainingSession", RecordingSession)
    monkeypatch.setattr(
        configurator_sut,
        "STEP_REGISTRY",
        {"train_step": RecordingComponent},
    )
    monkeypatch.setattr(
        configurator_sut,
        "HOOK_REGISTRY",
        {"metrics_hook": RecordingComponent},
    )
    monkeypatch.setattr(
        configurator_sut,
        "RESOURCE_REGISTRY",
        {
            "cache_resource": RecordingComponent,
            "ddp": RecordingDDPResource,
        },
    )


def test_create_session_from_config_builds_complete_parent_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_recording_registries(monkeypatch)
    config = {
        "base_config": {"max_iterations": 3},
        "train_step": {"value": "step"},
        "metrics_hook": {"value": "hook"},
        "cache_resource": {"value": "resource"},
        "ddp": {
            "world_size": 2,
            "backend": "gloo",
            "parallel_components": ["ddp", "train_step"],
        },
    }

    session = configurator_sut.create_session_from_config(config)

    assert isinstance(session, RecordingSession)
    assert session.base_config is config["base_config"]
    assert [component.config for component in session.steps] == [
        config["train_step"]
    ]
    assert [component.config for component in session.hooks] == [
        config["metrics_hook"]
    ]
    assert [component.config for component in session.resources] == [
        config["cache_resource"],
        config["ddp"],
    ]
    assert session.resources[-1].rank == -1


def test_create_session_from_config_does_not_prune_rank_zero_only_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_recording_registries(monkeypatch)
    monkeypatch.setattr(
        configurator_sut,
        "STEP_REGISTRY",
        {
            "parallel_step": RecordingComponent,
            "rank_zero_step": RecordingComponent,
        },
    )
    config = {
        "base_config": {},
        "parallel_step": {"kind": "parallel"},
        "rank_zero_step": {"kind": "rank-zero"},
        "ddp": {
            "world_size": 4,
            "backend": "gloo",
            "parallel_components": ["ddp", "parallel_step"],
        },
    }

    session = configurator_sut.create_session_from_config(config)

    assert [component.config for component in session.steps] == [
        config["parallel_step"],
        config["rank_zero_step"],
    ]
    assert session.resources[0].rank == -1


def test_create_session_from_config_allows_ddp_without_parallel_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(configurator_sut, "TrainingSession", RecordingSession)
    monkeypatch.setattr(configurator_sut, "STEP_REGISTRY", {})
    monkeypatch.setattr(configurator_sut, "HOOK_REGISTRY", {})
    monkeypatch.setattr(
        configurator_sut,
        "RESOURCE_REGISTRY",
        {"ddp": DDPResource},
    )

    result = configurator_sut.create_session_from_config(
        {
            "base_config": {},
            "ddp": {
                "world_size": 2,
                "backend": "gloo",
            },
        }
    )

    assert len(result.resources) == 1
    assert result.resources[0].rank == -1
    assert result.resources[0].parallel_components == []


def test_create_session_from_config_rejects_unknown_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(configurator_sut, "TrainingSession", RecordingSession)
    monkeypatch.setattr(configurator_sut, "STEP_REGISTRY", {})
    monkeypatch.setattr(configurator_sut, "HOOK_REGISTRY", {})
    monkeypatch.setattr(configurator_sut, "RESOURCE_REGISTRY", {})

    with pytest.raises(
        ValueError,
        match="No step, hook or resource registered with name 'missing'",
    ):
        configurator_sut.create_session_from_config(
            {"base_config": {}, "missing": {}}
        )


# ---------------------------------------------------------------------------
# Worker receives a parent-created session, not config/checkpoint source data.
# ---------------------------------------------------------------------------


def test_worker_modifies_supplied_session_for_its_rank(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_session = object()
    worker_session = IdleSession()
    calls: list[tuple[Any, int, Any]] = []

    def copy_for_worker(session, rank, session_update_params=None):
        calls.append((session, rank, session_update_params))
        return worker_session

    monkeypatch.setattr(engine_sut.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        engine_sut,
        "copy_and_modify_session_for_worker",
        copy_for_worker,
    )

    updates = {"max_iterations": 500}
    engine_sut.session_process_worker(
        source_session,
        session_id=7,
        rank=2,
        stop_event=SimpleNamespace(is_set=lambda: True),
        session_update_params=updates,
    )

    assert calls == [(source_session, 2, updates)]
    assert worker_session.entered is True
    assert worker_session.exited is True
    assert capsys.readouterr().out == "Session 7[2] exiting.\n"


def test_worker_allows_no_session_update_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_session = IdleSession()
    calls: list[tuple[Any, int, Any]] = []

    def copy_for_worker(session, rank, session_update_params=None):
        calls.append((session, rank, session_update_params))
        return worker_session

    source_session = object()
    monkeypatch.setattr(engine_sut.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        engine_sut,
        "copy_and_modify_session_for_worker",
        copy_for_worker,
    )

    engine_sut.session_process_worker(
        source_session,
        session_id=1,
        rank=0,
        stop_event=SimpleNamespace(is_set=lambda: True),
    )

    assert calls == [(source_session, 0, None)]


def test_worker_does_not_create_or_load_the_session_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_session = IdleSession()
    monkeypatch.setattr(engine_sut.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        engine_sut,
        "create_session_from_config",
        lambda *args, **kwargs: pytest.fail(
            "worker must not construct a new session from config"
        ),
    )
    monkeypatch.setattr(
        engine_sut,
        "copy_and_modify_session_for_worker",
        lambda session, rank, session_update_params=None: worker_session,
    )

    engine_sut.session_process_worker(
        object(),
        session_id=1,
        rank=0,
        stop_event=SimpleNamespace(is_set=lambda: True),
    )


# ---------------------------------------------------------------------------
# TrainingEngine.load_session uses the loaded parent session for all wrappers.
# ---------------------------------------------------------------------------


class RecordingWrapper:
    instances: list["RecordingWrapper"] = []

    def __init__(self, session, session_id: int, rank: int, **kwargs) -> None:
        self.session = session
        self.session_id = session_id
        self.rank = rank
        self.kwargs = kwargs
        type(self).instances.append(self)


class MinimalConfigurator:
    mode = "new"
    session_configs: list[dict[str, Any]] = []
    process_timeout_on_join = 3.0


def test_engine_load_session_creates_one_worker_for_non_ddp_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoadedSession:
        def has_resource(self, name: str) -> bool:
            return False

    loaded_session = LoadedSession()
    RecordingWrapper.instances.clear()
    monkeypatch.setattr(engine_sut, "SessionProcessWrapper", RecordingWrapper)
    monkeypatch.setattr(
        engine_sut.Checkpointer,
        "load_checkpoint",
        lambda path: loaded_session,
    )

    engine = engine_sut.TrainingEngine(MinimalConfigurator())
    session_id = engine.load_session("session.pkl")

    assert session_id == 0
    assert [wrapper.rank for wrapper in RecordingWrapper.instances] == [0]
    assert RecordingWrapper.instances[0].session is loaded_session


def test_engine_load_session_uses_world_size_from_ddp_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ddp_resource = SimpleNamespace(world_size=4)

    class LoadedSession:
        def has_resource(self, name: str) -> bool:
            return name == "ddp"

        def get_resource(self, name: str):
            assert name == "ddp"
            return ddp_resource

    RecordingWrapper.instances.clear()
    monkeypatch.setattr(engine_sut, "SessionProcessWrapper", RecordingWrapper)
    monkeypatch.setattr(
        engine_sut.Checkpointer,
        "load_checkpoint",
        lambda path: LoadedSession(),
    )

    engine = engine_sut.TrainingEngine(MinimalConfigurator())
    engine.load_session("ddp-session.pkl")

    assert [wrapper.rank for wrapper in RecordingWrapper.instances] == [
        0,
        1,
        2,
        3,
    ]


def test_engine_load_session_forwards_extension_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoadedSession:
        def has_resource(self, name: str) -> bool:
            return False

    RecordingWrapper.instances.clear()
    monkeypatch.setattr(engine_sut, "SessionProcessWrapper", RecordingWrapper)
    monkeypatch.setattr(
        engine_sut.Checkpointer,
        "load_checkpoint",
        lambda path: LoadedSession(),
    )

    updates = {"max_iterations": 800}
    engine = engine_sut.TrainingEngine(MinimalConfigurator())
    engine.load_session("session.pkl", session_update_params=updates)

    assert RecordingWrapper.instances[0].kwargs == {
        "session_update_params": updates,
    }