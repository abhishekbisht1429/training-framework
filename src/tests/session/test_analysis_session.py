from __future__ import annotations

import pytest

from training_framework.session import Session, TrainingSession


def _session_config(root, *, max_iterations=2):
    return {
        "rng_seed": 19,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


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
