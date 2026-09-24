"""The generic steps -- load_batch, forward, compute -- through real sessions.

Each paradigm is only configuration: the tests build sessions from config
alone and compare training against the same loop written in plain torch.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_utils import (
    make_config,
    resource_named,
    stub_process_group,
)
from training_framework.components import (
    Resource,
    StatefulResource,
    Step,
    reads,
    resource,
    step,
)
from training_framework.components.builtin import Compute
from training_framework.functions import weighted_sum
from training_framework.session import AnalysisSession, TrainingSession


@pytest.fixture(autouse=True)
def _stub_process_group(monkeypatch):
    stub_process_group(monkeypatch)


INPUTS = torch.tensor([[1.0, 2.0], [0.5, -1.0], [-2.0, 0.0], [3.0, 1.0]])
TARGETS = torch.tensor([[1.0], [0.0], [-1.0], [2.0]])
LABELS = torch.tensor([0, 2, 1, 2])

class _InertResource(Resource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def _register(*, outputs: int = 1, sample: str = "tuple"):
    """A four-sample dataset and a small linear model, as resources."""

    @resource("cmp_dataset", overwrite=True)
    class Dataset(_InertResource):
        def __len__(self):
            return len(INPUTS)

        def __getitem__(self, index):
            if sample == "dict":
                return {"x": INPUTS[index], "y": TARGETS[index], "id": f"s{index}"}
            if sample == "labels":
                return INPUTS[index], int(LABELS[index])
            return INPUTS[index], TARGETS[index]

    @resource("cmp_model", overwrite=True)
    class Model(nn.Module, StatefulResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self.linear = _fresh_linear(outputs)

        def forward(self, x):
            return self.linear(x)

        def encode(self, x):
            return self.linear(x) * 0

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {k: v.detach().clone() for k, v in self.state_dict().items()}

        def set_state(self, state):
            self.load_state_dict(state)


def _fresh_linear(outputs: int = 1) -> nn.Linear:
    linear = nn.Linear(2, outputs)
    with torch.no_grad():
        linear.weight.copy_(torch.linspace(-0.5, 0.5, 2 * outputs).reshape(outputs, 2))
        linear.bias.copy_(torch.linspace(0.0, 0.2, outputs))
    return linear


def _config(tmp_path, *, max_iterations=3, **components):
    config = make_config(tmp_path, max_iterations=max_iterations)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {"model": "cmp_model", "dataset": "cmp_dataset"},
        "cmp_model": {},
        "cmp_dataset": {},
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "master_addr": "localhost",
            "master_port": "12355",
        },
        # The whole dataset each iteration, so the order it comes in does not
        # change a mean loss.
        "data_manager": {"batch_size": 4, "num_workers": 0, "pin_memory": False},
        "optimizer": {"optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}}},
    })
    config.update(components)
    return config


def _session(config):
    """Build a session from `config`, with `ddp` given rank 0.

    A session driven by hand keeps `ddp`'s placeholder rank, which the
    sampler would read; the engine assigns the real one.
    """
    session_type = AnalysisSession if "trained_model" in config.get(
        "component_bindings", {}) else TrainingSession
    session = session_type(config)
    session.unregister_hook("logger")
    if session_type is TrainingSession:
        session.unregister_hook("checkpointer")
        placeholder = resource_named(session, "ddp")
        session.unregister_resource("ddp")
        session.register_resource(type(placeholder)(config=placeholder.config, rank=0))
    return session


def _train(tmp_path, **config):
    session = _session(_config(tmp_path, **config))
    with session:
        list(session)
    return session


def _recorder(*keys):
    """A step that keeps, per iteration, the values of `keys` it reads."""
    seen: list[dict] = []

    @reads(*keys)
    @step("cmp_recorder", overwrite=True)
    class Recorder(Step):
        def run(self, session, **values):
            seen.append(values)

    return seen


def _reference(loss_fn, iterations=3, outputs=1):
    linear = _fresh_linear(outputs)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    for _ in range(iterations):
        optimizer.zero_grad()
        loss_fn(linear).backward()
        optimizer.step()
    return linear


def _trained_linear(session) -> nn.Linear:
    return resource_named(session, "cmp_model").linear


# -- whole paradigms, configuration only -------------------------------------------


def test_supervised_regression_from_configuration_alone(tmp_path):
    _register()
    session = _train(
        tmp_path,
        load_batch={"fields": ["inputs", "targets"]},
        forward={"args": ["inputs"], "outputs": "prediction"},
        **{"compute#loss": {
            "function": "mse_loss",
            "args": ["prediction", "targets"],
            "outputs": "loss",
        }},
    )

    expected = _reference(
        lambda linear: nn.functional.mse_loss(linear(INPUTS), TARGETS)
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)
    torch.testing.assert_close(_trained_linear(session).bias, expected.bias)


def test_keys_named_like_the_step_parameters_are_ordinary_keys(tmp_path):
    _register()
    session = _train(
        tmp_path,
        load_batch={"fields": ["session", "self"]},
        forward={"args": ["session"], "outputs": "prediction"},
        **{"compute#loss": {
            "function": "mse_loss",
            "args": ["prediction", "self"],
            "outputs": "loss",
        }},
    )

    expected = _reference(
        lambda linear: nn.functional.mse_loss(linear(INPUTS), TARGETS)
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)


def test_classification_with_a_loss_class_and_labels_collated_by_default(tmp_path):
    _register(outputs=3, sample="labels")
    session = _train(
        tmp_path,
        load_batch={"fields": ["inputs", "labels"]},
        forward={"args": ["inputs"], "outputs": "logits"},
        **{"compute#loss": {
            "function": "CrossEntropyLoss",
            "init": {"label_smoothing": 0.1},
            "args": ["logits", "labels"],
            "outputs": "loss",
        }},
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    expected = _reference(
        lambda linear: loss_fn(linear(INPUTS), LABELS), outputs=3,
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)


def test_several_loss_terms_combined_by_weight(tmp_path):
    _register()
    session = _train(
        tmp_path,
        load_batch={"fields": ["inputs", "targets"]},
        forward={"args": ["inputs"], "outputs": "prediction"},
        **{
            "compute#doubled": {
                "function": "torch.mul",
                "args": ["prediction"],
                "constants": {"other": 2.0},
                "outputs": "doubled",
            },
            "compute#fit": {
                "function": "mse_loss",
                "args": ["prediction", "targets"],
                "outputs": "fit",
            },
            "compute#spread": {
                "function": "l1_loss",
                "args": ["doubled", "targets"],
                "outputs": "spread",
            },
            "compute#loss": {
                "function": "weighted_sum",
                "kwargs": {"fit": "fit", "spread": "spread"},
                "constants": {"weights": {"spread": 0.5}},
                "outputs": "loss",
            },
        },
    )

    def loss_fn(linear):
        prediction = linear(INPUTS)
        return (
            nn.functional.mse_loss(prediction, TARGETS)
            + 0.5 * nn.functional.l1_loss(prediction * 2.0, TARGETS)
        )

    expected = _reference(loss_fn)
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)


def test_a_teacher_without_gradients_distils_into_the_model(tmp_path):
    _register()

    @resource("cmp_teacher")
    class Teacher(nn.Module, _InertResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self.linear = nn.Linear(2, 1)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.zero_()

        def forward(self, x):
            return self.linear(x)

    config = dict(
        load_batch={"fields": ["inputs", "targets"]},
        cmp_teacher={},
        **{
            "forward#student": {"args": ["inputs"], "outputs": "student"},
            "forward#teacher": {
                "args": ["inputs"], "outputs": "teacher", "no_grad": True,
            },
            "compute#loss": {
                "function": "mse_loss",
                "args": ["student", "teacher"],
                "outputs": "loss",
            },
        },
    )
    session = _session({
        **_config(tmp_path, **config),
        "component_bindings": {
            "model": "cmp_model",
            "dataset": "cmp_dataset",
            "forward#teacher": {"model": "cmp_teacher"},
        },
    })
    with session:
        list(session)

    teacher_targets = INPUTS.sum(dim=1, keepdim=True)
    expected = _reference(
        lambda linear: nn.functional.mse_loss(linear(INPUTS), teacher_targets)
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)
    assert torch.equal(
        resource_named(session, "cmp_teacher").linear.weight,
        torch.ones(1, 2),
    )


# -- load_batch ------------------------------------------------------------------------


def test_a_dict_batch_is_picked_apart_and_other_values_pass_through(tmp_path):
    _register(sample="dict")
    session = _train(
        tmp_path,
        max_iterations=1,
        load_batch={"fields": {"inputs": "x", "targets": "y", "ids": "id"}},
        forward={"args": ["inputs"], "outputs": "prediction"},
        **{"compute#loss": {
            "function": "mse_loss",
            "args": ["prediction", "targets"],
            "outputs": "loss",
        }},
    )

    expected = _reference(
        lambda linear: nn.functional.mse_loss(linear(INPUTS), TARGETS), 1,
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)


def test_fields_that_do_not_fit_the_batch_are_reported(tmp_path):
    _register()
    session = _session(_config(
        tmp_path,
        load_batch={"fields": ["inputs", "targets", "weights"]},
        forward={"args": ["inputs"], "outputs": "prediction"},
        **{"compute#loss": {
            "function": "mse_loss",
            "args": ["prediction", "targets"],
            "outputs": "loss",
        }},
    ))

    with session:
        with pytest.raises(ValueError, match="names 3 parts, but the batch is a list of 2"):
            next(session)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_batch_is_moved_to_the_session_device(tmp_path):
    _register(sample="dict")
    seen = _recorder("batch")
    config = _config(
        tmp_path, max_iterations=1, load_batch={"key": "batch"}, cmp_recorder={},
    )
    config["session_config"]["device"] = "cuda"
    del config["optimizer"]
    with _session(config) as session:
        next(session)

    batch = seen[0]["batch"]
    assert batch["x"].device.type == "cuda"
    assert batch["y"].device.type == "cuda"
    assert sorted(batch["id"]) == ["s0", "s1", "s2", "s3"]


# -- forward ---------------------------------------------------------------------------


def test_the_trained_model_is_called_through_the_ddp_wrapper(tmp_path):
    _register()
    seen = _recorder("encoded")
    session = _session(_config(
        tmp_path,
        max_iterations=2,
        load_batch={"fields": ["inputs", "targets"]},
        forward={"args": ["inputs"], "outputs": "prediction"},
        cmp_recorder={},
        **{
            "forward#encoded": {
                "args": ["inputs"], "outputs": "encoded", "method": "encode",
            },
            "compute#loss": {
                "function": "mse_loss",
                "args": ["prediction", "targets"],
                "outputs": "loss",
            },
        },
    ))
    with session:
        list(session)
        wrapper = resource_named(session, "ddp").wrapped_model
        # `forward` goes through the wrapper; `method` calls the model itself.
        assert wrapper.calls == 2
    assert all(torch.equal(entry["encoded"], torch.zeros(4, 1)) for entry in seen)


def test_outputs_pick_fields_of_a_dict_or_an_attribute_result(tmp_path):
    _register()
    seen = _recorder("logits", "pooled", "both")

    @resource("cmp_structured")
    class Structured(nn.Module, _InertResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)

        def forward(self, x):
            return {"logits": x.sum(dim=1), "extra": x}

        def as_namespace(self, x):
            return SimpleNamespace(pooled=x.mean(dim=1))

    config = _config(
        tmp_path,
        max_iterations=1,
        cmp_structured={},
        cmp_recorder={},
        load_batch={"fields": ["inputs", "targets"]},
        **{
            "forward#dict": {"args": ["inputs"], "outputs": {"logits": "logits"}},
            "forward#attr": {
                "args": ["inputs"], "outputs": {"pooled": "pooled"},
                "method": "as_namespace",
            },
            "compute#both": {
                "function": "stack",
                "args": [["logits", "pooled"]],
                "outputs": "both",
            },
        },
    )
    del config["optimizer"]
    config["component_bindings"]["forward#dict"] = {"model": "cmp_structured"}
    config["component_bindings"]["forward#attr"] = {"model": "cmp_structured"}
    with _session(config) as session:
        next(session)

    values = seen[0]
    assert values["logits"].shape == (4,) and values["pooled"].shape == (4,)
    torch.testing.assert_close(
        values["both"], torch.stack([values["logits"], values["pooled"]]),
    )


# -- compute ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, expected", [
    ("mse_loss", nn.functional.mse_loss),
    ("torch.nn.functional.l1_loss", nn.functional.l1_loss),
    ("cat", torch.cat),
    ("weighted_sum", weighted_sum),
])
def test_function_names_resolve_in_order(name, expected):
    compute = Compute({"function": name, "outputs": "out"})
    assert compute._function is expected


def test_a_plain_class_is_called_like_a_function():
    compute = Compute({"function": "builtins.int", "outputs": "count"})
    assert compute._function is int


class ZeroArgumentCallable:
    """A callable class whose constructor takes nothing."""

    def __call__(self, value):
        return value


def test_an_empty_init_still_constructs_the_class():
    compute = Compute({
        "function": f"{__name__}.ZeroArgumentCallable",
        "init": {},
        "outputs": "same",
    })
    assert isinstance(compute._function, ZeroArgumentCallable)


def test_a_class_is_constructed_once_with_init():
    compute = Compute({
        "function": "CrossEntropyLoss",
        "init": {"label_smoothing": 0.2},
        "outputs": "loss",
    })
    assert isinstance(compute._function, nn.CrossEntropyLoss)
    assert compute._function.label_smoothing == 0.2


@pytest.mark.parametrize("config, error, message", [
    ({"function": "no_such_function", "outputs": "x"}, ValueError, "is not in"),
    ({"function": "mse_loss", "init": {"a": 1}, "outputs": "x"}, ValueError, "is for a class"),
    ({"function": "CrossEntropyLoss", "init": {"nope": 1}, "outputs": "x"}, ValueError, "does not fit"),
    ({"function": "mse_loss", "kwargs": {"a": "k"}, "constants": {"a": 1}, "outputs": "x"},
     ValueError, "both by kwargs and constants"),
    ({"function": "mse_loss"}, (TypeError, ValueError), "outputs"),
    ({"function": "mse_loss", "outputs": ["a", "a"]}, ValueError, "more than once"),
    ({"function": "mse_loss", "constants": {1: 2}, "outputs": "x"},
     ValueError, "constants parameter names must be non-empty strings"),
    ({"function": "mse_loss", "kwargs": {3: "k"}, "outputs": "x"},
     ValueError, "kwargs parameter names must be non-empty strings"),
])
def test_compute_configuration_errors_are_reported(config, error, message):
    with pytest.raises(error, match=message):
        Compute(config)


def test_a_call_declares_what_it_reads_and_writes():
    compute = Compute({
        "function": "weighted_sum",
        "args": ["a", ["b", "c"]],
        "kwargs": {"x": "d", "y": "a"},
        "outputs": {"total": "sum"},
    })
    assert compute.context_reads() == {"a": "a", "b": "b", "c": "c", "d": "d"}
    assert compute.context_writes() == {"total": "total"}


def test_weighted_sum_defaults_weights_and_rejects_unknown_ones():
    assert weighted_sum(a=torch.tensor(1.0), b=torch.tensor(2.0),
                        weights={"b": 0.5}) == torch.tensor(2.0)
    with pytest.raises(ValueError, match=r"weights for \['c'\]"):
        weighted_sum(a=torch.tensor(1.0), weights={"c": 1.0})


# -- analysis --------------------------------------------------------------------------


def test_analysis_forward_runs_the_trained_model_without_gradients(tmp_path):
    _register()
    model = _fresh_linear()

    @resource("cmp_trained", session_type="analysis")
    class Trained(_InertResource):
        @property
        def model(self):
            return model

    seen = []

    @reads("prediction")
    @step("cmp_probe", session_type="analysis")
    class Probe(Step):
        def run(self, session, prediction):
            seen.append(prediction)

    config = make_config(tmp_path, max_iterations=1)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {
            "trained_model": "cmp_trained", "dataset": "cmp_dataset",
        },
        "cmp_trained": {},
        "cmp_dataset": {},
        "cmp_probe": {},
        "data_manager": {"batch_size": 4},
        "load_batch": {"fields": ["inputs", "targets"]},
        "forward": {"args": ["inputs"], "outputs": "prediction"},
    })
    with _session(config) as session:
        next(session)

    assert not seen[0].requires_grad
    with torch.no_grad():
        torch.testing.assert_close(seen[0], model(INPUTS))


# -- serialization ------------------------------------------------------------------------


SUPERVISED = {
    "load_batch": {"fields": ["inputs", "targets"]},
    "forward": {"args": ["inputs"], "outputs": "prediction"},
    "compute#loss": {
        "function": "mse_loss",
        "args": ["prediction", "targets"],
        "outputs": "loss",
    },
}


def _pickled(component):
    return pickle.loads(pickle.dumps(component))


def _swap_in_pickled_copies(session, names):
    """Replace each named step with a copy that went through pickle."""
    for name in names:
        original = next(s for s in session.get_all_steps() if s.name == name)
        session.remove_step(name)
        session.add_step(_pickled(original))


def test_pickled_generic_steps_train_like_fresh_ones(tmp_path):
    _register()
    session = _session(_config(tmp_path, **SUPERVISED))
    _swap_in_pickled_copies(session, ["load_batch", "forward", "compute#loss"])

    with session:
        list(session)
        # The copy still calls the trained model through the DDP wrapper.
        assert resource_named(session, "ddp").wrapped_model.calls == 3

    expected = _reference(
        lambda linear: nn.functional.mse_loss(linear(INPUTS), TARGETS)
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)


def test_a_pickled_compute_keeps_the_instance_it_built(tmp_path):
    _register(outputs=3, sample="labels")
    session = _session(_config(
        tmp_path,
        load_batch={"fields": ["inputs", "labels"]},
        forward={"args": ["inputs"], "outputs": "logits"},
        **{"compute#loss": {
            "function": "CrossEntropyLoss",
            "init": {"label_smoothing": 0.1},
            "args": ["logits", "labels"],
            "outputs": "loss",
        }},
    ))
    _swap_in_pickled_copies(session, ["compute#loss"])

    with session:
        list(session)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    expected = _reference(
        lambda linear: loss_fn(linear(INPUTS), LABELS), outputs=3,
    )
    torch.testing.assert_close(_trained_linear(session).weight, expected.weight)


def test_a_run_of_generic_steps_resumes_from_a_checkpoint_exactly(tmp_path):
    _register()
    config = _config(
        tmp_path,
        max_iterations=4,
        optimizer={"optimizer": {
            "name": "SGD", "kwargs": {"lr": 0.1, "momentum": 0.9},
        }},
        **SUPERVISED,
    )
    uninterrupted = _session(config)
    with uninterrupted:
        list(uninterrupted)

    paused = _session(config)
    with paused:
        next(paused)
        next(paused)
    resumed = TrainingSession.from_state(paused.get_state())
    with resumed:
        assert list(resumed) == [3, 4]

    torch.testing.assert_close(
        _trained_linear(resumed).weight, _trained_linear(uninterrupted).weight,
    )


def test_a_pickled_analysis_forward_still_runs_without_gradients(tmp_path):
    _register()
    model = _fresh_linear()

    @resource("cmp_trained", session_type="analysis")
    class Trained(_InertResource):
        @property
        def model(self):
            return model

    seen = []

    @reads("prediction")
    @step("cmp_probe", session_type="analysis")
    class Probe(Step):
        def run(self, session, prediction):
            seen.append(prediction)

    config = make_config(tmp_path, max_iterations=1)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {
            "trained_model": "cmp_trained", "dataset": "cmp_dataset",
        },
        "cmp_trained": {},
        "cmp_dataset": {},
        "cmp_probe": {},
        "data_manager": {"batch_size": 4},
        "load_batch": {"fields": ["inputs", "targets"]},
        "forward": {"args": ["inputs"], "outputs": "prediction"},
    })
    session = _session(config)
    _swap_in_pickled_copies(session, ["load_batch", "forward"])
    with session:
        next(session)

    assert not seen[0].requires_grad
    with torch.no_grad():
        torch.testing.assert_close(seen[0], model(INPUTS))
