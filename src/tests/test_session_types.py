from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from training_framework.components import Step, step
from training_framework.components.registry import component_registry
from training_framework.engine import TrainingEngine
from training_framework.session import (
    Session,
    TrainingSession,
    normalize_session_config,
    register_session_type,
)


def _config(tmp_path, name: str = "session") -> dict:
    return {
        "session_config": {
            "rng_seed": 7,
            "sessions_dir": str(tmp_path / name),
            "max_iterations": 1,
            "components_package": "training_framework.components.builtin",
        },
    }


def _evaluation_session_class():
    @register_session_type("evaluation")
    class EvaluationSession(Session):
        @classmethod
        def _default_component_configs(cls) -> Mapping[str, Mapping]:
            return {}

        def __init__(self, config: dict, *, label: str):
            self.label = label
            super().__init__(config)

        def _get_session_type_state(self):
            return {"label": self.label}

        def _restore_session_type_state(self, state):
            self.label = state["label"]

    return EvaluationSession


def test_shared_and_scoped_components_use_effective_session_registry(tmp_path):
    EvaluationSession = _evaluation_session_class()

    @step("probe")
    class SharedProbe(Step):
        def __init__(self, config):
            self.scope = "shared"

        def run(self, session):
            pass

    @step("probe", session_type="evaluation")
    class EvaluationProbe(Step):
        def __init__(self, config):
            self.scope = "evaluation"

        def run(self, session):
            pass

    training = TrainingSession({**_config(tmp_path, "training"), "probe": {}})
    evaluation = EvaluationSession(
        {**_config(tmp_path, "evaluation"), "probe": {}},
        label="report",
    )

    assert training.get_all_steps()[0].scope == "shared"
    assert evaluation.get_all_steps()[0].scope == "evaluation"
    assert component_registry("training")["probe"] is SharedProbe
    assert component_registry("evaluation")["probe"] is EvaluationProbe
    assert component_registry("future")["probe"] is SharedProbe


def test_component_scoped_to_another_session_type_is_inaccessible(tmp_path):
    @step("analysis_only", session_type="analysis")
    class AnalysisOnly(Step):
        def __init__(self, config):
            pass

        def run(self, session):
            pass

    with pytest.raises(ValueError, match="analysis_only"):
        TrainingSession({
            **_config(tmp_path),
            "analysis_only": {},
        })


def test_custom_session_type_round_trips_through_state_registry(tmp_path):
    EvaluationSession = _evaluation_session_class()
    session = EvaluationSession(_config(tmp_path), label="quality")

    state = session.get_state()
    restored = Session.from_state(state)

    assert state["session_type"] == "evaluation"
    assert type(restored) is EvaluationSession
    assert restored.session_type == "evaluation"
    assert restored.label == "quality"
    assert session.full_config["session_kwargs"] == {"label": "quality"}


def test_full_config_excludes_keyword_config_argument(tmp_path):
    session = TrainingSession(config=_config(tmp_path))

    assert "session_kwargs" not in session.full_config


def test_engine_dispatches_mixed_session_types_and_constructor_kwargs(
        tmp_path,
        monkeypatch,
):
    EvaluationSession = _evaluation_session_class()
    captured_sessions = []

    class Wrapper:
        def __init__(self, *, session, rank, heartbeat_timeout):
            self.session = session
            captured_sessions.append(session)

    monkeypatch.setattr(
        "training_framework.engine.core.SessionProcessWrapper",
        Wrapper,
    )
    configurator = SimpleNamespace(
        mode="new",
        session_configs=[
            _config(tmp_path, "training"),
            {
                "session_type": "evaluation",
                "session_kwargs": {"label": "metrics"},
                **_config(tmp_path, "evaluation"),
            },
        ],
        heartbeat_timeout=10.0,
        process_timeout_on_join=5.0,
    )

    engine = TrainingEngine(configurator)
    engine.__enter__()

    assert [session.session_type for session in captured_sessions] == [
        "training",
        "evaluation",
    ]
    assert isinstance(captured_sessions[1], EvaluationSession)
    assert captured_sessions[1].label == "metrics"


def test_session_config_defaults_and_validation(tmp_path):
    config = _config(tmp_path)["session_config"]

    normalized = normalize_session_config(config)

    assert normalized["device"] == "cpu"
    assert normalized["show_execution_graph"] is True
    with pytest.raises(ValueError, match="Missing required"):
        normalize_session_config({})
    with pytest.raises(ValueError, match="Unknown session_config fields"):
        normalize_session_config({**config, "batch_size": 4})
    with pytest.raises(ValueError, match="must contain 'session_config'"):
        TrainingSession({"base_config": config})


def test_duplicate_session_type_registration_is_rejected():
    _evaluation_session_class()

    with pytest.raises(ValueError, match="already registered"):
        _evaluation_session_class()
