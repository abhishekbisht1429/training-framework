from __future__ import annotations

import pytest

from training_framework.training_session import SessionPhase, TrainingSession
from tests.test_utils import read_events, register_test_components, session_config


def _component(session: TrainingSession, name: str):
    components = session.get_all_resources() + session.get_all_steps() + session.get_all_hooks()
    return next(component for component in components if component.name == name)


def test_config_driven_session_runs_training_and_lifecycle_end_to_end(tmp_path):
    """Exercise construction, hooks, autograd, optimizer state, and teardown together."""

    register_test_components()
    event_path = tmp_path / "lifecycle.jsonl"
    session = TrainingSession(
        session_config(
            tmp_path,
            max_iterations=3,
            event_path=event_path,
            include_metrics=True,
        )
    )

    with session:
        completed_iterations = list(session)

    model = _component(session, "it_3d45_model")
    training_step = _component(session, "it_3d45_train")
    metrics = _component(session, "it_3d45_metrics")

    assert completed_iterations == [1, 2, 3]
    assert session.iteration == 3
    assert session._phase is SessionPhase.FINISHED
    assert session.session_context == {}

    assert model.setup_count == 1
    assert model.teardown_count == 1
    assert len(training_step.weight_history) == 3
    assert len(training_step.noise_history) == 3
    assert metrics.setup_count == 1
    assert metrics.teardown_count == 1
    assert [item["iteration"] for item in metrics.observations] == [1, 2, 3]
    assert [item["weight"] for item in metrics.observations] == pytest.approx(
        training_step.weight_history
    )

    events = read_events(event_path)
    assert [event["event"] for event in events] == [
        "model_setup",
        "metrics_setup",
        "iteration",
        "metrics",
        "iteration",
        "metrics",
        "iteration",
        "metrics",
        "model_teardown",
        "metrics_teardown",
    ]
