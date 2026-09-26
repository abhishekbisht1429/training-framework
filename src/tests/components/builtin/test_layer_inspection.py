from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

from training_framework.components import (
    StatefulResource,
    Step,
    requires_resource,
    resource,
    step,
)
from training_framework.components.builtin import LayerInspector
from training_framework.session import AnalysisSession, TrainingSession
from tests.test_utils import inject_dependencies


class _AttentionBlock(nn.Module):
    """A minimal attention-shaped layer: scales input, optionally adds a mask."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, mask=None):
        if mask is None:
            return x * self.scale
        return x * self.scale + mask


class _InspectionModel(nn.Module, StatefulResource):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.attention = _AttentionBlock()
        self.projection = nn.Linear(1, 1)

    def forward(self, x, mask=None):
        attended = self.attention(x, mask=mask)
        return self.projection(attended.reshape(1)).reshape(())

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return self.state_dict()

    def set_state(self, state):
        self.load_state_dict(state)


class _CaptureCountingStep(Step):
    """Test-only Step: runs a forward pass (unless skipped) and records how
    many 'attention' captures are visible each iteration."""

    def __init__(self, config):
        self._skip_iterations = set(config.get("skip_iterations", ()))
        self.capture_counts: list[int] = []

    def run(self, session):
        model = self.get_dependency("trained_model").model
        if session.iteration not in self._skip_iterations:
            model(torch.tensor(float(session.iteration)))
        inspector = self.get_dependency("layer_inspector")
        self.capture_counts.append(
            len(inspector.captures.get("attention", []))
        )


class _FakeTrainedModel:
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model


class _FakeSession:
    """Minimal stand-in for Session, enough for LayerInspector's own
    lifecycle: the iteration generation that scopes its captures, plus the
    `trained_model` it serves to the inspector through `serve`."""

    def __init__(self, model):
        self._trained_model = _FakeTrainedModel(model)
        # One iteration for the whole test: captures accumulate.
        self._iteration_generation = 0

    def serve(self, inspector):
        """Inject this session's trained model, as a real session would."""
        return inject_dependencies(inspector, trained_model=self._trained_model)


def _session_config(root, *, max_iterations=2):
    return {
        "rng_seed": 5,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def _write_training_checkpoint(tmp_path):
    resource("layer_inspection_source_model")(_InspectionModel)
    source = TrainingSession({
        "session_config": _session_config(tmp_path / "training"),
        "component_bindings": {"model": "layer_inspection_source_model"},
        "layer_inspection_source_model": {},
    })
    checkpoint_path = tmp_path / "training-session.pt"
    torch.save(source, checkpoint_path)
    return checkpoint_path


def _register_capture_counting_step():
    cls = step("layer_inspection_probe", session_type="analysis")(
        _CaptureCountingStep
    )
    cls = requires_resource("trained_model")(cls)
    cls = requires_resource("layer_inspector")(cls)
    return cls


def _analysis_config(
        tmp_path,
        checkpoint_path,
        layer_inspector_config,
        *,
        max_iterations=2,
        extra=None,
):
    config = {
        "session_config": _session_config(
            tmp_path / "analysis", max_iterations=max_iterations
        ),
        "trained_model": {"model_checkpoint_path": str(checkpoint_path)},
        "layer_inspector": layer_inspector_config,
    }
    if extra:
        config.update(extra)
    return config


# -- constructor / selection validation --------------------------------


def test_layer_inspector_requires_at_least_one_selector():
    with pytest.raises(TypeError, match="config must be a mapping"):
        LayerInspector(None)
    with pytest.raises(TypeError, match="name_patterns must be a list"):
        LayerInspector({"name_patterns": "attention"})
    with pytest.raises(TypeError, match="module_types must be a list"):
        LayerInspector({"module_types": "torch.nn.Linear"})
    with pytest.raises(ValueError, match="at least one of"):
        LayerInspector({})
    with pytest.raises(ValueError, match="at least one of"):
        LayerInspector({"name_patterns": [], "module_types": []})


def test_layer_inspector_rejects_invalid_regex():
    with pytest.raises(ValueError, match="invalid regex"):
        LayerInspector({"name_patterns": ["("]})


def test_layer_inspector_resolves_module_types_from_dotted_paths():
    LayerInspector({"module_types": ["torch.nn.Linear"]})

    with pytest.raises(ValueError, match="dotted paths"):
        LayerInspector({"module_types": ["Linear"]})
    with pytest.raises(ImportError):
        LayerInspector({
            "module_types": ["definitely_not_a_real_module_xyz.Thing"],
        })
    with pytest.raises(ValueError, match="no attribute"):
        LayerInspector({"module_types": ["torch.nn.NotARealLayer"]})
    with pytest.raises(TypeError, match="does not resolve to an nn.Module"):
        LayerInspector({"module_types": ["torch.optim.SGD"]})


# -- selection / hook lifecycle, driven directly (no full Session) -----


def test_layer_inspector_selects_union_of_name_and_type_matches_without_duplication():
    model = _InspectionModel({})
    inspector = LayerInspector({
        "name_patterns": [r"^attention$"],
        "module_types": [
            "tests.components.builtin.test_layer_inspection._AttentionBlock",
        ],
    })
    session = _FakeSession(model)
    session.serve(inspector)

    inspector.setup(session)
    try:
        assert inspector.matched_layer_names == ("attention",)
        # Matched twice, hooked once: one forward pass, one capture.
        model(torch.tensor(1.0))
        assert len(inspector.captures["attention"]) == 1
    finally:
        inspector.teardown(session)


def test_layer_inspector_exposes_selected_layers_for_weight_inspection():
    model = _InspectionModel({})
    inspector = LayerInspector({
        "name_patterns": [r"^attention$", r"^projection$"],
    })
    session = _FakeSession(model)
    session.serve(inspector)
    assert dict(inspector.layers) == {}

    inspector.setup(session)
    try:
        layers = inspector.layers
        assert list(layers) == list(inspector.matched_layer_names)
        assert layers["attention"] is model.attention
        assert layers["projection"] is model.projection
        # Parameters are readable without a forward pass.
        assert layers["attention"].scale is model.attention.scale
        assert dict(layers["projection"].named_parameters()).keys() == {
            "weight",
            "bias",
        }
        with pytest.raises(TypeError):
            layers["attention"] = model.projection  # type: ignore[index]
    finally:
        inspector.teardown(session)

    assert dict(inspector.layers) == {}


def test_layer_inspector_raises_when_no_layer_matches():
    model = _InspectionModel({})
    inspector = LayerInspector({"name_patterns": [r"^no_such_layer$"]})
    session = _FakeSession(model)
    session.serve(inspector)

    with pytest.raises(ValueError, match="matched no layers"):
        inspector.setup(session)
    assert dict(inspector.layers) == {}


def test_last_capture_returns_latest_call_none_or_rejects_unknown_layer():
    model = _InspectionModel({})
    inspector = LayerInspector({
        "name_patterns": [r"^attention$", r"^projection$"],
    })
    session = _FakeSession(model)
    session.serve(inspector)
    inspector.setup(session)
    try:
        assert inspector.last_capture("attention") is None

        model(torch.tensor(1.0))
        model(torch.tensor(2.0))

        latest = inspector.last_capture("attention")
        assert latest is inspector.captures["attention"][-1]
        assert latest.input_args == (torch.tensor(2.0),)

        with pytest.raises(KeyError, match="'atention' is not a layer selected"):
            inspector.last_capture("atention")
    finally:
        inspector.teardown(session)


def test_layer_inspector_captures_input_args_kwargs_and_output_per_forward_pass():
    model = _InspectionModel({})
    inspector = LayerInspector({"name_patterns": [r"^attention$"]})
    session = _FakeSession(model)
    session.serve(inspector)
    inspector.setup(session)
    try:
        mask = torch.tensor(0.5)
        model(torch.tensor(3.0), mask=mask)

        capture = inspector.captures["attention"][0]
        assert capture.layer_name == "attention"
        assert capture.module_type == "_AttentionBlock"
        assert capture.input_args == (torch.tensor(3.0),)
        assert capture.input_kwargs == {"mask": mask}
        torch.testing.assert_close(capture.output, torch.tensor(6.5))
        assert capture.module is model.attention
        assert capture.module is inspector.layers["attention"]
        assert capture.module.scale is model.attention.scale
    finally:
        inspector.teardown(session)


def test_layer_inspector_accumulates_multiple_forward_passes_within_one_iteration():
    model = _InspectionModel({})
    inspector = LayerInspector({"name_patterns": [r"^attention$"]})
    session = _FakeSession(model)
    session.serve(inspector)
    inspector.setup(session)
    try:
        model(torch.tensor(1.0))
        model(torch.tensor(2.0))

        captures = inspector.captures["attention"]
        assert len(captures) == 2
        assert [c.input_args[0].item() for c in captures] == [1.0, 2.0]
    finally:
        inspector.teardown(session)


def test_layer_inspector_removes_hooks_on_teardown():
    model = _InspectionModel({})
    inspector = LayerInspector({"name_patterns": [r"^attention$"]})
    session = _FakeSession(model)
    session.serve(inspector)
    inspector.setup(session)
    inspector.teardown(session)

    # A hook left behind would capture alongside the fresh one.
    inspector.setup(session)
    try:
        model(torch.tensor(1.0))
        assert len(inspector.captures["attention"]) == 1
    finally:
        inspector.teardown(session)


def test_layer_inspector_rolls_back_partial_hook_registration_on_setup_failure(
        monkeypatch,
):
    model = _InspectionModel({})
    inspector = LayerInspector({
        "name_patterns": [r"^attention$", r"^projection$"],
    })
    session = _FakeSession(model)
    session.serve(inspector)

    def failing_register(self, *args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(nn.Linear, "register_forward_hook", failing_register)

    with pytest.raises(RuntimeError, match="registration failed"):
        inspector.setup(session)

    assert dict(inspector.layers) == {}
    # The hook registered before the failure is gone: a working setup now
    # captures each forward pass once.
    monkeypatch.undo()
    inspector.setup(session)
    try:
        model(torch.tensor(1.0))
        assert len(inspector.captures["attention"]) == 1
    finally:
        inspector.teardown(session)


# -- per-iteration clearing, driven through a real AnalysisSession -----


def test_layer_inspector_clears_captures_between_analysis_iterations(tmp_path):
    _register_capture_counting_step()
    checkpoint_path = _write_training_checkpoint(tmp_path)
    session = AnalysisSession(_analysis_config(
        tmp_path,
        checkpoint_path,
        {"name_patterns": [r"^attention$"]},
        max_iterations=3,
        extra={"layer_inspection_probe": {}},
    ))

    with session:
        for _ in session:
            pass

    [probe] = [
        s for s in session.get_all_steps()
        if s.name == "layer_inspection_probe"
    ]
    # Every iteration ran exactly one forward pass, so each should see
    # exactly 1 capture -- never an accumulating count across iterations.
    assert probe.capture_counts == [1, 1, 1]


def test_layer_inspector_captures_are_empty_when_no_forward_pass_occurs_in_an_iteration(
        tmp_path,
):
    _register_capture_counting_step()
    checkpoint_path = _write_training_checkpoint(tmp_path)
    session = AnalysisSession(_analysis_config(
        tmp_path,
        checkpoint_path,
        {"name_patterns": [r"^attention$"]},
        max_iterations=3,
        extra={
            "layer_inspection_probe": {"skip_iterations": [2]},
        },
    ))

    with session:
        for _ in session:
            pass

    [probe] = [
        s for s in session.get_all_steps()
        if s.name == "layer_inspection_probe"
    ]
    # Iteration 2 skipped its forward pass; its capture count must be 0,
    # not a stale count carried over from iteration 1.
    assert probe.capture_counts == [1, 0, 1]
