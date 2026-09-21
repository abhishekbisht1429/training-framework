from __future__ import annotations

import pytest
import torch
from torch import nn

from training_framework.components import Resource, StatefulResource, resource
from training_framework.components.builtin import TrainedModel
from training_framework.session import AnalysisSession, Session, TrainingSession
from tests.test_utils import resource_named


class _AnalysisSourceModel(nn.Module, StatefulResource):
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
    resource("analysis_session_source_model")(_AnalysisSourceModel)
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training"),
        "component_bindings": {
            "model": "analysis_session_source_model",
        },
        "analysis_session_source_model": {"weight": 3.5},
    })
    checkpoint_path = tmp_path / "training-session.pt"
    torch.save(source, checkpoint_path)
    return checkpoint_path


def _analysis_config(tmp_path, checkpoint_path):
    return {
        "session_config": _session_config(tmp_path / "analysis"),
        "trained_model": {
            "model_checkpoint_path": str(checkpoint_path),
        },
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


def test_analysis_session_loads_component_configured_model(tmp_path):
    checkpoint_path = _write_training_checkpoint(tmp_path)
    session = AnalysisSession(_analysis_config(tmp_path, checkpoint_path))

    assert not hasattr(session, "model_checkpoint_path")
    assert isinstance(resource_named(session, "trained_model"), TrainedModel)
    assert "model_checkpoint_path" not in session.get_state()

    with session:
        model = resource_named(session, "trained_model").model
        assert not model.training
        torch.testing.assert_close(
            model(torch.tensor(2.0)),
            torch.tensor(7.0),
        )


def test_analysis_session_round_trips_component_owned_checkpoint_path(tmp_path):
    checkpoint_path = _write_training_checkpoint(tmp_path)
    session = AnalysisSession(_analysis_config(tmp_path, checkpoint_path))

    restored = Session.from_state(session.get_state())

    assert type(restored) is AnalysisSession
    assert not hasattr(restored, "model_checkpoint_path")
    with restored:
        model = resource_named(restored, "trained_model").model
        torch.testing.assert_close(
            model(torch.tensor(2.0)),
            torch.tensor(7.0),
        )


def test_analysis_session_rejects_legacy_top_level_checkpoint_path(tmp_path):
    checkpoint_path = tmp_path / "training-session.pt"
    checkpoint_path.touch()

    with pytest.raises(
            ValueError,
            match=r"trained_model\.model_checkpoint_path",
    ):
        AnalysisSession({
            "session_config": _session_config(tmp_path),
            "model_checkpoint_path": checkpoint_path,
        })


def test_analysis_session_requires_default_trained_model_config(tmp_path):
    with pytest.raises(
            ValueError,
            match=r"trained_model\.model_checkpoint_path is required",
    ):
        AnalysisSession({
            "session_config": _session_config(tmp_path),
        })


def test_analysis_session_allows_bound_trained_model_implementation(tmp_path):
    @resource("analysis_model_replacement")
    class AnalysisModelReplacement(Resource):
        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    session = AnalysisSession({
        "session_config": _session_config(tmp_path),
        "component_bindings": {
            "trained_model": "analysis_model_replacement",
        },
    })

    assert isinstance(
        resource_named(session, "trained_model"),
        AnalysisModelReplacement,
    )
