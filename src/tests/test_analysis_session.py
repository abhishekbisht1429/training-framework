from __future__ import annotations

import sys
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from training_framework.components.builtin import (
    AnalysisLogger,
    Checkpointer,
    TrainedModel,
)
from training_framework.engine import Configurator
from training_framework.engine import TrainingEngine, load_session_for_worker
from training_framework.components.registry import (
    _ANALYSIS_COMPONENT_REGISTRY,
    _COMPONENT_REGISTRY,
    component_registry,
)
from training_framework.components import (
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)
from training_framework.session import AnalysisSession, Session, TrainingSession


def _session_config(root, *, max_iterations=2):
    return {
        "rng_seed": 19,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def _write_training_checkpoint(tmp_path):
    @resource("analysis_source_model")
    class AnalysisSourceModel(nn.Module, StatefulResource):
        def __init__(self, config):
            nn.Module.__init__(self)
            self.weight = nn.Parameter(torch.tensor(float(config["weight"])))

        def forward(self, value):
            return self.weight * value

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return self.state_dict()

        def set_state(self, state):
            self.load_state_dict(state)

    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training", max_iterations=4),
        "aliases": {"model": "analysis_source_model"},
        "model": {"weight": 3.5},
    })
    checkpoint_path = tmp_path / "model-session.pt"
    torch.save(source, checkpoint_path)
    return checkpoint_path


def test_training_session_is_fixed_mode_compatibility_subclass(tmp_path):
    session = TrainingSession({"session_config": _session_config(tmp_path)})

    assert isinstance(session, Session)
    assert session.session_type == "training"
    assert not hasattr(session, "model_checkpoint_path")
    assert hasattr(session, "update_max_iters")
    assert "model_checkpoint_path" not in session.get_state()
    assert isinstance(Session.from_state(session.get_state()), TrainingSession)

    invalid_state = session.get_state()
    invalid_state.pop("session_type")
    with pytest.raises(ValueError, match="required 'session_type'"):
        Session.from_state(invalid_state)


def test_session_is_an_abstract_base(tmp_path):
    with pytest.raises(TypeError, match="abstract class"):
        Session({"session_config": _session_config(tmp_path)})


def test_mode_scoped_registries_allow_same_component_name():
    @step("shared_name")
    class TrainingStep(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    @step("shared_name", session_type="analysis")
    class AnalysisStep(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    assert _COMPONENT_REGISTRY["shared_name"] is TrainingStep
    assert _ANALYSIS_COMPONENT_REGISTRY["shared_name"] is AnalysisStep


def test_trained_model_is_registered_as_a_shared_component():
    assert _COMPONENT_REGISTRY["trained_model"] is TrainedModel
    assert "trained_model" not in _ANALYSIS_COMPONENT_REGISTRY
    assert component_registry("training")["trained_model"] is TrainedModel
    assert component_registry("analysis")["trained_model"] is TrainedModel
    assert component_registry("evaluation")["trained_model"] is TrainedModel


def test_trained_model_loads_session_checkpoint(tmp_path):
    checkpoint_path = _write_training_checkpoint(tmp_path)
    trained_model = TrainedModel({})
    session = SimpleNamespace(
        device="cpu",
        model_checkpoint_path=str(checkpoint_path),
    )

    with pytest.raises(RuntimeError, match="not initialized yet"):
        getattr(trained_model, "model")

    trained_model.setup(session)
    assert float(
        trained_model.model(torch.tensor(2.0)).detach(),
    ) == 7.0

    trained_model.teardown(session)
    with pytest.raises(RuntimeError, match="not initialized yet"):
        getattr(trained_model, "model")


def test_analysis_session_loads_model_and_runs_analysis_components(tmp_path):
    checkpoint_path = _write_training_checkpoint(tmp_path)

    @step("analysis_probe", session_type="analysis")
    @requires_resource("trained_model")
    class AnalysisProbe(Step):
        def __init__(self, config):
            self.observations = []

        def run(self, session):
            model = session.get_resource("trained_model").model
            value = model(torch.tensor(2.0))
            self.observations.append({
                "value": float(value.detach()),
                "training": model.training,
                "grad_enabled": torch.is_grad_enabled(),
            })

    session = AnalysisSession(
        {
            "session_config": _session_config(tmp_path / "analysis"),
            "model_checkpoint_path": checkpoint_path,
            "analysis_probe": {},
        },
    )

    hooks = {hook.name: hook for hook in session.get_all_hooks()}
    resources = {
        item.name: item for item in session.get_all_resources()
    }
    assert isinstance(hooks["logger"], AnalysisLogger)
    assert "checkpointer" not in hooks
    assert isinstance(resources["trained_model"], TrainedModel)
    assert session.execution_graph().startswith("ANALYSIS SESSION EXECUTION GRAPH")

    with session:
        assert list(session) == [1, 2]

    probe = next(item for item in session.get_all_steps() if item.name == "analysis_probe")
    assert probe.observations == [
        {"value": 7.0, "training": False, "grad_enabled": True},
        {"value": 7.0, "training": False, "grad_enabled": True},
    ]


def test_analysis_state_restores_with_concrete_subclass(tmp_path):
    checkpoint_path = _write_training_checkpoint(tmp_path)
    session = AnalysisSession(
        {
            "session_config": _session_config(tmp_path / "analysis"),
            "model_checkpoint_path": checkpoint_path,
        },
    )

    restored = Session.from_state(session.get_state())

    assert type(restored) is AnalysisSession
    assert restored.session_type == "analysis"
    assert restored.model_checkpoint_path == str(checkpoint_path)
    assert restored.full_config["model_checkpoint_path"] == str(
        checkpoint_path
    )
    assert "session_kwargs" not in restored.full_config
    assert not hasattr(restored, "update_max_iters")
    with pytest.raises(ValueError, match="cannot restore analysis session-type state"):
        TrainingSession.from_state(session.get_state())
    training = TrainingSession({"session_config": _session_config(tmp_path / "training")})
    with pytest.raises(ValueError, match="cannot restore training session-type state"):
        AnalysisSession.from_state(training.get_state())
    with pytest.raises(TypeError, match="require a TrainingSession"):
        load_session_for_worker(
            session.get_state(),
            rank=0,
            session_update_params={"max_iterations": 5},
        )

    analysis_checkpoint = tmp_path / "analysis-session.pt"
    torch.save(session, analysis_checkpoint)
    loaded = Checkpointer.load_checkpoint(analysis_checkpoint)
    assert isinstance(loaded, AnalysisSession)
    assert loaded.model_checkpoint_path == str(checkpoint_path)


def test_analysis_requires_existing_checkpoint(tmp_path):
    config = {"session_config": _session_config(tmp_path)}

    with pytest.raises(ValueError, match="model_checkpoint_path"):
        AnalysisSession(config)
    with pytest.raises(TypeError, match="string or path-like"):
        AnalysisSession({
            **config,
            "model_checkpoint_path": 42,
        })
    with pytest.raises(FileNotFoundError, match="does not exist"):
        AnalysisSession({
            **config,
            "model_checkpoint_path": tmp_path / "missing.pt",
        })
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AnalysisSession(
            {
                **config,
                "model_checkpoint_path": tmp_path / "missing.pt",
            },
            model_checkpoint_path=tmp_path / "missing.pt",
        )


def test_analysis_rejects_non_session_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "payload.pt"
    torch.save({"model": "not a session"}, checkpoint_path)
    session = AnalysisSession(
        {
            "session_config": _session_config(tmp_path / "analysis"),
            "model_checkpoint_path": checkpoint_path,
        },
    )

    with pytest.raises(TypeError, match="framework Session"):
        with session:
            pass


def test_analysis_rejects_training_checkpoint_without_model(tmp_path):
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training"),
    })
    checkpoint_path = tmp_path / "no-model.pt"
    torch.save(source, checkpoint_path)
    session = AnalysisSession(
        {
            "session_config": _session_config(tmp_path / "analysis"),
            "model_checkpoint_path": checkpoint_path,
        },
    )

    with pytest.raises(ValueError, match="resolvable 'model' resource"):
        with session:
            pass


def test_configurator_parses_analysis_checkpoint_path(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "source.pt"
    checkpoint_path.touch()
    config_path = tmp_path / "analysis.yaml"
    definition = {
        "session_type": "analysis",
        "model_checkpoint_path": str(checkpoint_path),
        "session_config": _session_config(tmp_path / "run"),
    }
    config_path.write_text(
        yaml.safe_dump({"sessions": [definition]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "analyze",
        "--config",
        str(config_path),
    ])

    configurator = Configurator()

    assert configurator.mode == "new"
    assert configurator.session_configs == [definition]
    assert configurator.get_all_component_configs(0) == {}


def test_configurator_rejects_removed_analysis_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "analyze",
        "--analysis-config",
        "analysis.yaml",
    ])

    with pytest.raises(SystemExit):
        Configurator()


def test_engine_runs_analysis_in_spawned_worker(tmp_path):
    module_name = "tests.integration_analysis_components"
    existing = sys.modules.get(module_name)
    if existing is None:
        importlib.import_module(module_name)
    else:
        importlib.reload(existing)
    checkpoint_path = tmp_path / "spawn-source.pt"
    source = TrainingSession({
        "session_config": {
            **_session_config(tmp_path / "training"),
            "components_package": "tests.integration_analysis_components",
        },
        "aliases": {"model": "integration_analysis_model"},
        "model": {"weight": 4.0},
    })
    torch.save(source, checkpoint_path)

    output_path = tmp_path / "analysis.jsonl"
    analysis_config = {
        "session_config": {
            **_session_config(tmp_path / "analysis"),
            "components_package": "tests.integration_analysis_components",
        },
        "model_checkpoint_path": str(checkpoint_path),
        "integration_analysis_probe": {"output_path": str(output_path)},
    }
    configurator = SimpleNamespace(
        mode="new",
        session_configs=[{
            "session_type": "analysis",
            **analysis_config,
        }],
        heartbeat_timeout=10.0,
        process_timeout_on_join=5.0,
    )

    with TrainingEngine(configurator) as engine:
        engine.start_session()

    observations = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert observations == [
        {
            "grad_enabled": True,
            "iteration": 1,
            "prediction": 8.0,
            "training": False,
        },
        {
            "grad_enabled": True,
            "iteration": 2,
            "prediction": 8.0,
            "training": False,
        },
    ]
