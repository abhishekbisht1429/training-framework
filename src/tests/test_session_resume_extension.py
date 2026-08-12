from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import training_framework.configurator as configurator_sut
import training_framework.training_engine as engine_sut
import training_framework.training_session as session_sut
import training_framework.util as util_sut


COMMIT = "c7f3b225150648daf8cddbabd8b0e8d4557b52c0"


def _parse_args(monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(sys, "argv", ["training-framework", *args])
    return configurator_sut.Configurator()


class _IdleSession:
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
# Configurator: config / resume / extension modes
# ---------------------------------------------------------------------------


def test_configurator_resume_mode_exposes_checkpoint_and_timeout(monkeypatch):
    configurator = _parse_args(
        monkeypatch,
        "--resume-session",
        "checkpoints/session.pkl",
        "--process_timeout_on_join",
        "7.5",
    )

    assert (
        configurator.checkpoint_path
        == "checkpoints/session.pkl"
    )
    assert configurator.process_timeout_on_join == pytest.approx(7.5)


def test_configurator_modes_are_mutually_exclusive(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_args(
            monkeypatch,
            "--config",
            "config.yaml",
            "--resume-session",
            "session.pkl",
        )

def test_configurator_requires_exactly_one_mode(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_args(monkeypatch)


def test_configurator_extension_parses_max_iterations_as_integer(monkeypatch):
    configurator = _parse_args(
        monkeypatch,
        "--extend-session",
        "session.pkl",
        "250",
    )

    assert configurator.checkpoint_path == "session.pkl"
    assert configurator.new_max_iters == 250
    assert isinstance(configurator.new_max_iters, int)


def test_resume_mode_rejects_config_accessors_with_key_error(monkeypatch):
    configurator = _parse_args(
        monkeypatch,
        "--resume-session",
        "session.pkl",
    )

    with pytest.raises(KeyError, match="current mode"):
        _ = configurator.session_configs


# ---------------------------------------------------------------------------
# Checkpoint session construction
# ---------------------------------------------------------------------------


def test_create_session_from_checkpoint_applies_max_iteration_update(
    monkeypatch,
):
    class FakeSession:
        def __init__(self) -> None:
            self.max_iteration_updates: list[int] = []

        def update_max_iters(self, value: int) -> None:
            self.max_iteration_updates.append(value)

        def has_resource(self, name: str) -> bool:
            return False

    loaded_session = FakeSession()
    loaded_paths: list[str] = []

    monkeypatch.setattr(
        configurator_sut,
        "Checkpointer",
        SimpleNamespace(
            load_checkpoint=lambda *, path: (
                loaded_paths.append(path) or loaded_session
            )
        ),
    )

    result = configurator_sut.create_session_from_checkpoint(
        "session.pkl",
        session_update_params={"max_iterations": 300},
        rank=0,
    )

    assert result is loaded_session
    assert loaded_paths == ["session.pkl"]
    assert loaded_session.max_iteration_updates == [300]


def test_secondary_checkpoint_rank_keeps_only_parallel_components(
    monkeypatch,
):
    class Component:
        def __init__(self, name: str) -> None:
            self.name = name

    class DDPComponent(Component):
        def __init__(self) -> None:
            super().__init__("ddp")
            self.world_size = 3
            self._rank = 0
            self.parallel_components = [
                "ddp",
                "parallel_hook",
                "parallel_resource",
                "parallel_step",
            ]
            self.config = {
                'world_size': self.world_size,
                'parallel_components': self.parallel_components,
                'backend': 'nccl'
            }

        @property
        def rank(self) -> int:
            return self._rank

    class FakeSession:
        def __init__(self) -> None:
            self.ddp = DDPComponent()
            self.hooks = {
                "parallel_hook": Component("parallel_hook"),
                "rank_zero_hook": Component("rank_zero_hook"),
            }
            self.resources = {
                "ddp": self.ddp,
                "parallel_resource": Component("parallel_resource"),
                "rank_zero_resource": Component("rank_zero_resource"),
            }
            self.steps = {
                "parallel_step": Component("parallel_step"),
                "rank_zero_step": Component("rank_zero_step"),
            }

        # Both APIs are supplied so the test remains useful after the DDP
        # lookup is corrected from hook to resource.
        def has_hook(self, name: str) -> bool:
            return name in self.hooks

        def get_hook(self, name: str):
            return self.hooks[name]

        def has_resource(self, name: str) -> bool:
            return name in self.resources

        def get_resource(self, name: str):
            return self.resources[name]

        def get_all_hooks(self):
            return list(self.hooks.values())

        def get_all_resources(self):
            return list(self.resources.values())

        def get_all_steps(self):
            return list(self.steps.values())

        def unregister_hook(self, name: str) -> None:
            del self.hooks[name]

        def unregister_resource(self, name: str) -> None:
            if not isinstance(name, str):
                raise TypeError("resource name must be a string")
            del self.resources[name]

        def register_resource(self, resource):
            self.resources[resource.name] = resource

        def remove_step(self, name: str) -> None:
            del self.steps[name]

    loaded_session = FakeSession()
    monkeypatch.setattr(
        configurator_sut,
        "Checkpointer",
        SimpleNamespace(load_checkpoint=lambda *, path: loaded_session),
    )

    result = configurator_sut.create_session_from_checkpoint(
        "session.pkl",
        rank=2,
    )

    assert set(result.hooks) == {"parallel_hook"}
    assert set(result.resources) == {"ddp", "parallel_resource"}
    assert set(result.steps) == {"parallel_step"}


def test_checkpoint_factory_updates_ddp_resource_to_worker_rank(monkeypatch):
    class DDPComponent:
        name = "ddp"
        world_size = 4
        parallel_components = ["ddp"]

        def __init__(self) -> None:
            self._rank = 0

        @property
        def rank(self) -> int:
            return self._rank

        @property
        def config(self):
            return {
                "world_size": self.world_size,
                "parallel_components": self.parallel_components,
                "backend": "nccl",
            }

    class FakeSession:
        def __init__(self) -> None:
            self.resources = {'ddp': DDPComponent()}

        def has_hook(self, name: str) -> bool:
            return False

        def has_resource(self, name: str) -> bool:
            return name == "ddp"

        def get_resource(self, name: str):
            assert name == "ddp"
            return self.resources['ddp']

        def get_all_hooks(self):
            return []

        def get_all_resources(self):
            return self.resources.values()

        def get_all_steps(self):
            return []

        def unregister_resource(self, name: str) -> None:
            if not isinstance(name, str):
                raise TypeError("resource name must be a string")
            del self.resources[name]

        def register_resource(self, resource):
            self.resources[resource.name] = resource

    loaded_session = FakeSession()
    monkeypatch.setattr(
        configurator_sut,
        "Checkpointer",
        SimpleNamespace(load_checkpoint=lambda *, path: loaded_session),
    )

    result = configurator_sut.create_session_from_checkpoint(
        "session.pkl",
        rank=3,
    )

    assert result.get_resource("ddp").rank == 3


def test_ddp_configuration_may_omit_parallel_components(monkeypatch):
    created_sessions: list[Any] = []

    class FakeSession:
        def __init__(self, config: dict) -> None:
            self.config = config
            self.resources: list[Any] = []
            created_sessions.append(self)

        def register_resource(self, resource) -> None:
            self.resources.append(resource)

    monkeypatch.setattr(configurator_sut, "TrainingSession", FakeSession)

    result = configurator_sut.create_session_from_config(
        {
            "base_config": {},
            "ddp": {
                "world_size": 2,
                "backend": "gloo",
            },
        },
        rank=0,
    )

    assert result is created_sessions[0]
    assert len(result.resources) == 1
    assert result.resources[0].parallel_components == []


# ---------------------------------------------------------------------------
# Worker configuration and process wrappers
# ---------------------------------------------------------------------------


def test_worker_builds_session_from_checkpoint(monkeypatch, capsys):
    session = _IdleSession()
    calls: list[tuple[str, dict[str, int] | None, int]] = []

    def checkpoint_factory(
        checkpoint_path: str,
        session_update_params=None,
        *,
        rank: int,
    ):
        calls.append((checkpoint_path, session_update_params, rank))
        return session

    monkeypatch.setattr(engine_sut.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        engine_sut,
        "create_session_from_checkpoint",
        checkpoint_factory,
    )

    engine_sut.session_process_worker(
        {
            "checkpoint_path": "session.pkl",
            "session_update_params": {"max_iterations": 500},
        },
        session_id=7,
        rank=2,
        stop_event=SimpleNamespace(is_set=lambda: True),
    )

    assert calls == [
        ("session.pkl", {"max_iterations": 500}, 2)
    ]
    assert session.entered is True
    assert session.exited is True
    assert capsys.readouterr().out == "Session 7[2] exiting.\n"


def test_worker_allows_checkpoint_without_update_parameters(monkeypatch):
    session = _IdleSession()
    calls: list[tuple[str, Any, int]] = []

    def checkpoint_factory(
        checkpoint_path: str,
        session_update_params=None,
        *,
        rank: int,
    ):
        calls.append((checkpoint_path, session_update_params, rank))
        return session

    monkeypatch.setattr(engine_sut.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        engine_sut,
        "create_session_from_checkpoint",
        checkpoint_factory,
    )

    engine_sut.session_process_worker(
        {"checkpoint_path": "session.pkl"},
        session_id=1,
        rank=0,
        stop_event=SimpleNamespace(is_set=lambda: True),
    )

    assert calls == [("session.pkl", None, 0)]


def test_worker_rejects_ambiguous_session_source(monkeypatch):
    monkeypatch.setattr(engine_sut.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        engine_sut,
        "create_session_from_config",
        lambda config, *, rank: _IdleSession(),
    )
    monkeypatch.setattr(
        engine_sut,
        "create_session_from_checkpoint",
        lambda *args, **kwargs: _IdleSession(),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        engine_sut.session_process_worker(
            {
                "session_config": {},
                "checkpoint_path": "session.pkl",
                "session_update_params": None,
            },
            session_id=1,
            rank=0,
            stop_event=SimpleNamespace(is_set=lambda: True),
        )


def test_process_wrapper_deep_copies_nested_worker_configuration(monkeypatch):
    class FakeEvent:
        pass

    class FakeProcess:
        def __init__(self, *, name, target, args):
            self.name = name
            self.target = target
            self.args = args

    monkeypatch.setattr(engine_sut, "Event", FakeEvent)
    monkeypatch.setattr(engine_sut, "Process", FakeProcess)

    worker_config = {
        "checkpoint_path": "session.pkl",
        "session_update_params": {"max_iterations": 10},
    }

    wrapper = engine_sut.SessionProcessWrapper(
        worker_config=worker_config,
        session_id=4,
        rank=1,
    )

    worker_config["session_update_params"]["max_iterations"] = 999

    copied_config = wrapper.process.args[0]
    assert copied_config == {
        "checkpoint_path": "session.pkl",
        "session_update_params": {"max_iterations": 10},
    }
    assert copied_config is not worker_config
    assert (
        copied_config["session_update_params"]
        is not worker_config["session_update_params"]
    )


# ---------------------------------------------------------------------------
# TrainingEngine.load_session
# ---------------------------------------------------------------------------


class _RecordingWrapper:
    instances: list["_RecordingWrapper"] = []

    def __init__(self, worker_config, session_id: int, rank: int):
        self.worker_config = worker_config
        self.session_id = session_id
        self.rank = rank
        type(self).instances.append(self)


def test_engine_load_session_creates_one_worker_for_non_ddp_checkpoint(
    monkeypatch,
):
    class LoadedSession:
        def has_resource(self, name: str) -> bool:
            return False

    _RecordingWrapper.instances.clear()
    monkeypatch.setattr(engine_sut, "SessionProcessWrapper", _RecordingWrapper)
    monkeypatch.setattr(
        engine_sut,
        "Checkpointer",
        SimpleNamespace(load_checkpoint=lambda *, path: LoadedSession()),
    )

    engine = engine_sut.TrainingEngine(
        SimpleNamespace(process_timeout_on_join=3.0)
    )

    session_id = engine.load_session(
        "session.pkl",
        session_update_params={"max_iterations": 20},
    )

    assert session_id == 0
    assert [wrapper.rank for wrapper in _RecordingWrapper.instances] == [0]
    assert _RecordingWrapper.instances[0].worker_config == {
        "checkpoint_path": "session.pkl",
        "session_update_params": {"max_iterations": 20},
    }


# @pytest.mark.xfail(
#     strict=True,
#     reason=(
#         f"Known draft issue in {COMMIT}: load_session() looks for DDP in "
#         "the hook collection even though DDPResource is a resource, so a "
#         "DDP checkpoint is resumed with only one process."
#     ),
# )
def test_engine_load_session_uses_world_size_from_ddp_resource(monkeypatch):
    ddp_resource = SimpleNamespace(name="ddp", world_size=4)

    class LoadedSession:
        def has_hook(self, name: str) -> bool:
            return False

        def has_resource(self, name: str) -> bool:
            return name == "ddp"

        def get_resource(self, name: str):
            assert name == "ddp"
            return ddp_resource

        def get_all_resources(self):
            return [ddp_resource]

    _RecordingWrapper.instances.clear()
    monkeypatch.setattr(engine_sut, "SessionProcessWrapper", _RecordingWrapper)
    monkeypatch.setattr(
        engine_sut,
        "Checkpointer",
        SimpleNamespace(load_checkpoint=lambda *, path: LoadedSession()),
    )

    engine = engine_sut.TrainingEngine(
        SimpleNamespace(process_timeout_on_join=3.0)
    )
    engine.load_session("ddp-session.pkl")

    assert [wrapper.rank for wrapper in _RecordingWrapper.instances] == [
        0,
        1,
        2,
        3,
    ]


# ---------------------------------------------------------------------------
# TrainingSession additions
# ---------------------------------------------------------------------------


def _bare_training_session() -> session_sut.TrainingSession:
    session = object.__new__(session_sut.TrainingSession)
    session._hooks = {}
    session._resources = {}
    session._steps = {}
    return session


def test_component_collection_accessors_return_snapshots():
    session = _bare_training_session()
    hook = SimpleNamespace(name="hook")
    resource = SimpleNamespace(name="resource")
    step = SimpleNamespace(name="step")
    session._hooks[hook.name] = hook
    session._resources[resource.name] = resource
    session._steps[step.name] = step

    hooks = session.get_all_hooks()
    resources = session.get_all_resources()
    steps = session.get_all_steps()

    hooks.clear()
    resources.clear()
    steps.clear()

    assert session.get_all_hooks() == [hook]
    assert session.get_all_resources() == [resource]
    assert session.get_all_steps() == [step]


def test_component_removal_methods_remove_by_registered_name(monkeypatch):
    session = _bare_training_session()
    hook = SimpleNamespace(name="hook")
    resource = SimpleNamespace(name="resource")
    step = SimpleNamespace(name="step")

    monkeypatch.setitem(session_sut.HOOK_REGISTRY, hook.name, object())
    monkeypatch.setitem(
        session_sut.RESOURCE_REGISTRY,
        resource.name,
        object(),
    )
    monkeypatch.setitem(session_sut.STEP_REGISTRY, step.name, object())

    session._hooks[hook.name] = hook
    session._resources[resource.name] = resource
    session._steps[step.name] = step

    session.unregister_hook(hook.name)
    session.unregister_resource(resource.name)
    session.remove_step(step.name)

    assert session.get_all_hooks() == []
    assert session.get_all_resources() == []
    assert session.get_all_steps() == []


@pytest.mark.parametrize(
    ("method_name", "registry", "component_name"),
    [
        ("unregister_hook", "HOOK_REGISTRY", "missing_hook"),
        ("unregister_resource", "RESOURCE_REGISTRY", "missing_resource"),
        ("remove_step", "STEP_REGISTRY", "missing_step"),
    ],
)
def test_component_removal_rejects_names_absent_from_global_registry(
    method_name,
    registry,
    component_name,
):
    session = _bare_training_session()
    assert component_name not in getattr(session_sut, registry)

    with pytest.raises(ValueError, match=component_name):
        getattr(session, method_name)(component_name)


def test_update_max_iters_preserves_seed_and_session_directory():
    session = _bare_training_session()
    session._session_config = session_sut.SessionConfig(
        rng_seed=42,
        session_dir="/tmp/example-session",
        max_iterations=10,
    )

    session.update_max_iters(25)

    assert session.session_config == session_sut.SessionConfig(
        rng_seed=42,
        session_dir="/tmp/example-session",
        max_iterations=25,
    )


def test_transient_initialization_imports_components_before_sorting(
    monkeypatch,
):
    events: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        session_sut,
        "import_all_modules",
        lambda package: events.append(("import", package)),
    )
    monkeypatch.setattr(
        session_sut,
        "topological_sort_of_components",
        lambda: events.append(("sort", None)) or {"component": 0},
    )

    session = _bare_training_session()
    session._config = {
        "components_package": "example.components",
        "device": "cpu",
    }

    session._init_transient_infra()

    assert events == [
        ("import", "example.components"),
        ("sort", None),
    ]
    assert session._shared_state == {}
    assert session._order_of_components == {"component": 0}


# ---------------------------------------------------------------------------
# Package/module discovery and DDPResource metadata
# ---------------------------------------------------------------------------


def _write_module(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_import_all_modules_imports_root_package_and_descendants(
    tmp_path,
    monkeypatch,
):
    package_name = "component_discovery_fixture"
    sink_name = "component_discovery_sink"

    sink = ModuleType(sink_name)
    sink.events = []
    monkeypatch.setitem(sys.modules, sink_name, sink)
    monkeypatch.syspath_prepend(str(tmp_path))

    body = (
        f"from {sink_name} import events\n"
        "events.append(__name__)\n"
    )

    package_dir = tmp_path / package_name
    _write_module(package_dir / "__init__.py", body)
    _write_module(package_dir / "model.py", body)
    _write_module(package_dir / "nested" / "__init__.py", body)
    _write_module(package_dir / "nested" / "step.py", body)

    try:
        util_sut.import_all_modules(package_name)

        assert set(sink.events) == {
            package_name,
            f"{package_name}.model",
            f"{package_name}.nested",
            f"{package_name}.nested.step",
        }
    finally:
        for module_name in list(sys.modules):
            if (
                module_name == package_name
                or module_name.startswith(package_name + ".")
            ):
                sys.modules.pop(module_name, None)


def test_import_all_modules_accepts_a_plain_module(tmp_path, monkeypatch):
    module_name = "single_component_module"
    sink_name = "single_component_sink"

    sink = ModuleType(sink_name)
    sink.events = []
    monkeypatch.setitem(sys.modules, sink_name, sink)
    monkeypatch.syspath_prepend(str(tmp_path))

    _write_module(
        tmp_path / f"{module_name}.py",
        f"from {sink_name} import events\nevents.append(__name__)\n",
    )

    try:
        util_sut.import_all_modules(module_name)
        assert sink.events == [module_name]
    finally:
        sys.modules.pop(module_name, None)


def test_ddp_parallel_components_property_returns_a_defensive_copy():
    resource = configurator_sut.DDPResource(
        {
            "world_size": 2,
            "backend": "gloo",
            "parallel_components": ["model", "optimizer"],
        },
        rank=1,
    )

    returned = resource.parallel_components
    returned.append("late_mutation")

    assert resource.parallel_components == ["model", "optimizer"]
    assert resource.rank == 1
    assert resource.world_size == 2
    assert resource.backend == "gloo"