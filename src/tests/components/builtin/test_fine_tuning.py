"""Fine-tuning: `checkpoint_module` and `fine_tuned_model`, through real sessions.

A pretraining session is checkpointed to disk; a fine-tuning session takes
part of its model through `checkpoint_module`, puts a head on it through
`fine_tuned_model`, and trains with the built-in optimizer chain. DDP is
replaced by the recording stand-in, as the other built-in tests do.
"""

from __future__ import annotations

import copy
import pickle
import shutil

import pytest
import torch
from torch import nn
from torch.nn import functional

from tests.test_utils import make_config, resource_named, stub_process_group
from training_framework.components import (
    ComponentDependencyError,
    ModuleResource,
    Resource,
    Step,
    requires_resource,
    resource,
    step,
    writes,
)
from training_framework.components.builtin import Checkpointer
from training_framework.session import TrainingSession


@pytest.fixture(autouse=True)
def _stub_process_group(monkeypatch):
    stub_process_group(monkeypatch)


# -- components ------------------------------------------------------------------
# At module scope, so the checkpointed classes can be rebuilt.


class PretrainModel(ModuleResource):
    """An encoder with BatchNorm and dropout, and a pretext head."""

    def __init__(self, config=None):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(4, 6),     # encoder.0
            nn.BatchNorm1d(6),   # encoder.1
            nn.ReLU(),
            nn.Dropout(self.config.get("dropout", 0.5)),  # encoder.3
            nn.Linear(6, 5),     # encoder.4
        )
        self.pretext_head = nn.Linear(5, 7)

    def features(self, x):
        encoded = self.encoder(x)
        return {"pooled": encoded, "twice": 2 * encoded}

    def forward(self, x):
        return self.pretext_head(self.encoder(x))


class DrivenModel(ModuleResource):
    """A model the session sets up: it cannot be held as a plain module."""

    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(4, 5)

    def setup(self, session):
        super().setup(session)


class ProbeHead(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(self.config.get("in_features", 5), 3)

    def forward(self, x):
        return self.linear(x)


class ToyDataset(Resource):
    num_classes = 3

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("dataset")
class DatasetSizedHead(ModuleResource):
    """The user's pattern: a head that takes its class count from `dataset`."""

    def __init__(self, config=None):
        super().__init__(config)
        num_classes = self.get_dependency("dataset").num_classes
        self.linear = nn.Linear(5, num_classes)

    def forward(self, x):
        return self.linear(x)


class PlainBackbone(ModuleResource):
    """A backbone of the user's own, not from a checkpoint."""

    def __init__(self, config=None):
        super().__init__(config)
        self.layer = nn.Linear(4, 5)

    def forward(self, x):
        return self.layer(x)


class ResettingBackbone(PlainBackbone):
    """A backbone whose own setup puts every submodule back in train mode."""

    def setup(self, session):
        super().setup(session)
        self.train()


class NotAModule(Resource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass


INPUTS = torch.tensor([
    [1.0, -0.5, 0.25, 2.0],
    [-1.0, 0.5, 1.5, -0.25],
    [0.5, 0.5, -1.0, 1.0],
    [2.0, -1.5, 0.0, 0.75],
])
TARGETS = torch.tensor([0, 2, 1, 2])


@writes("loss")
@requires_resource("ddp")
class FineTuneLoss(Step):
    def run(self, session):
        return functional.cross_entropy(
            self.get_dependency("ddp").wrapped_model(INPUTS), TARGETS,
        )


def _register():
    resource("ft_pretrain_model", overwrite=True)(PretrainModel)
    resource("ft_driven_model", overwrite=True)(DrivenModel)
    resource("ft_probe_head", overwrite=True)(ProbeHead)
    resource("ft_toy_dataset", overwrite=True)(ToyDataset)
    resource("ft_dataset_head", overwrite=True)(DatasetSizedHead)
    resource("ft_plain_backbone", overwrite=True)(PlainBackbone)
    resource("ft_resetting_backbone", overwrite=True)(ResettingBackbone)
    resource("ft_not_a_module", overwrite=True)(NotAModule)
    step("ft_loss", overwrite=True)(FineTuneLoss)


@pytest.fixture(autouse=True)
def _components():
    _register()


# -- helpers -----------------------------------------------------------------------


def pretrain_checkpoint(tmp_path, model="ft_pretrain_model", **model_config):
    """Write a pretraining run whose `model` is `model`; return its path and
    the pretrained model."""
    config = make_config(tmp_path / "pretrain", seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"model": model}
    config[model] = model_config
    session = TrainingSession(config)
    path = Checkpointer.save_checkpoint(session, tmp_path / "pretrained")
    return path, resource_named(session, model)


def fine_tuning_config(
        tmp_path,
        checkpoint,
        *,
        source=None,
        model=None,
        bindings=None,
        optimizer=None,
        max_iterations=3,
        **extra,
):
    config = make_config(tmp_path, max_iterations=max_iterations)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {
        "model": "fine_tuned_model",
        "backbone": "checkpoint_module",
        "head": "ft_probe_head",
        **(bindings or {}),
    }
    config.update({
        "checkpoint_module": {
            "checkpoint": str(checkpoint),
            "submodule": "encoder",
            **(source or {}),
        },
        "fine_tuned_model": model or {},
        "ft_probe_head": {},
        "ft_loss": {},
        "ddp": {
            "world_size": 1,
            "backend": "gloo",
            "master_addr": "localhost",
            "master_port": "12355",
        },
        "optimizer": optimizer or {
            "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        },
    })
    config.update(extra)
    return config


def fine_tuning_session(tmp_path, checkpoint, **kwargs):
    session = TrainingSession(fine_tuning_config(tmp_path, checkpoint, **kwargs))
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    return session


def run(session, iterations=None):
    with session:
        if iterations is None:
            return list(session)
        return [next(session) for _ in range(iterations)]


def model_of(session) -> nn.Module:
    return resource_named(session, "fine_tuned_model")


def parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


FROZEN_FIRST_BLOCK = {"frozen": ["module.0.*", "module.1.*"]}


# -- checkpoint_module ----------------------------------------------------------------


def test_checkpoint_module_keeps_only_the_selected_submodule(tmp_path):
    path, pretrained = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path)
    source = resource_named(session, "checkpoint_module")

    names = [name for name, _ in source.named_parameters()]
    assert names and all(name.startswith("module.") for name in names)
    assert not any("pretext_head" in name for name in names)
    for name, parameter in pretrained.encoder.named_parameters():
        torch.testing.assert_close(source.module.get_parameter(name), parameter)


def test_checkpoint_module_defaults_to_the_whole_model_resource(tmp_path):
    path, pretrained = pretrain_checkpoint(tmp_path)
    config = fine_tuning_config(tmp_path / "run", path, bindings={"head": "ft_probe_head"})
    config["checkpoint_module"] = {"checkpoint": str(path)}
    config["ft_probe_head"] = {"in_features": 7}
    session = TrainingSession(config)

    held = resource_named(session, "checkpoint_module").module
    assert isinstance(held, PretrainModel)
    torch.testing.assert_close(
        held.pretext_head.weight, pretrained.pretext_head.weight,
    )


def test_checkpoint_module_calls_the_configured_method(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        source={"submodule": "", "method": "features"},
        model={"backbone_output": "pooled"},
    )

    backbone = resource_named(session, "checkpoint_module").eval()
    output = backbone(INPUTS)
    assert set(output) == {"pooled", "twice"}
    torch.testing.assert_close(output["pooled"], backbone.module.encoder(INPUTS))


def test_a_missing_submodule_names_the_ones_that_exist(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match=r"no attribute 'encoderr'.*'encoder', 'pretext_head'"):
        fine_tuning_session(tmp_path / "run", path, source={"submodule": "encoderr"})


def test_a_submodule_that_is_not_a_module_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(TypeError, match="model.training is a bool, not an nn.Module"):
        fine_tuning_session(tmp_path / "run", path, source={"submodule": "training"})


def test_a_missing_method_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="has no method 'embed'"):
        fine_tuning_session(tmp_path / "run", path, source={"method": "embed"})


def test_a_resource_the_checkpoint_does_not_hold_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="has no single resource 'teacher'"):
        fine_tuning_session(tmp_path / "run", path, source={"resource": "teacher"})


def test_a_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint_module.checkpoint does not exist"):
        fine_tuning_session(tmp_path / "run", tmp_path / "nowhere")


def test_a_session_driven_source_must_be_narrowed_with_submodule(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path, model="ft_driven_model")

    with pytest.raises(ComponentDependencyError, match="Pick a part of it with `submodule`"):
        fine_tuning_session(tmp_path / "run", path, source={"submodule": ""})

    # A part of it is a plain module, and is accepted.
    session = fine_tuning_session(tmp_path / "run2", path, source={"submodule": "linear"})
    assert isinstance(resource_named(session, "checkpoint_module").module, nn.Linear)


def test_checkpoint_module_survives_a_pickle_round_trip(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, max_iterations=2)
    run(session)
    source = resource_named(session, "checkpoint_module")

    copied = pickle.loads(pickle.dumps(source))

    original = parameters(source)
    assert parameters(copied).keys() == original.keys()
    for name, value in parameters(copied).items():
        torch.testing.assert_close(value, original[name])


# -- fine_tuned_model: wiring -----------------------------------------------------------


def test_parameters_are_named_after_the_roles(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path)

    names = [name for name, _ in model_of(session).named_parameters()]
    assert {name.split(".", 1)[0] for name in names} == {"backbone", "head"}
    assert "backbone.module.0.weight" in names
    assert "head.linear.weight" in names


def test_a_role_filled_by_something_that_is_not_a_module_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(TypeError, match="role 'head' is filled by NotAModule"):
        fine_tuning_session(
            tmp_path / "run", path,
            bindings={"head": "ft_not_a_module"}, ft_not_a_module={},
        )


def test_a_backbone_of_the_users_own_fills_the_role(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    config = fine_tuning_config(
        tmp_path / "run", path,
        bindings={"backbone": "ft_plain_backbone"},
        model={"frozen": ["layer.bias"]},
        ft_plain_backbone={},
    )
    del config["checkpoint_module"]
    session = TrainingSession(config)
    backbone = resource_named(session, "ft_plain_backbone")
    before = parameters(backbone)

    run(session)

    assert not backbone.layer.bias.requires_grad
    torch.testing.assert_close(backbone.layer.bias, before["layer.bias"])
    assert not torch.equal(backbone.layer.weight, before["layer.weight"])


def test_a_head_can_take_its_class_count_from_the_dataset(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        bindings={"head": "ft_dataset_head", "dataset": "ft_toy_dataset"},
        ft_dataset_head={}, ft_toy_dataset={},
    )

    assert model_of(session).head.linear.out_features == ToyDataset.num_classes
    run(session)


# -- fine_tuned_model: freezing ---------------------------------------------------------


def test_fine_tuning_matches_plain_torch(tmp_path):
    # No dropout: its masks would differ between the two runs.
    path, _ = pretrain_checkpoint(tmp_path, dropout=0.0)
    session = fine_tuning_session(
        tmp_path / "run", path, model=FROZEN_FIRST_BLOCK, max_iterations=4,
        optimizer={"optimizer": {"name": "AdamW", "kwargs": {"lr": 0.05}}},
    )
    model = model_of(session)
    backbone = copy.deepcopy(model.backbone.module)
    head = copy.deepcopy(model.head.linear)
    frozen_before = parameters(model.backbone.module[0])

    run(session)

    # The same fine-tuning written directly in torch: first block frozen and
    # in eval mode (frozen_eval defaults to true), the rest trained.
    for parameter in [*backbone[0].parameters(), *backbone[1].parameters()]:
        parameter.requires_grad_(False)
    backbone.train()
    backbone[0].eval()
    backbone[1].eval()
    trainable = [p for p in [*backbone.parameters(), *head.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    for _ in range(4):
        optimizer.zero_grad()
        functional.cross_entropy(head(backbone(INPUTS)), TARGETS).backward()
        optimizer.step()

    for name, parameter in model.backbone.module.named_parameters():
        torch.testing.assert_close(parameter, backbone.get_parameter(name))
    torch.testing.assert_close(model.head.linear.weight, head.weight)
    for name, parameter in model.backbone.module[0].named_parameters():
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter, frozen_before[name])


def test_frozen_parameters_get_no_gradient(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, model=FROZEN_FIRST_BLOCK)
    model = model_of(session)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert frozen == [
        "backbone.module.0.weight", "backbone.module.0.bias",
        "backbone.module.1.weight", "backbone.module.1.bias",
    ]
    model(INPUTS).sum().backward()
    assert all(model.get_parameter(name).grad is None for name in frozen)


def test_a_frozen_pattern_that_matches_nothing_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match=r"fine_tuned_model.frozen patterns \['encoder.0.\*'\] match no parameter"):
        fine_tuning_session(tmp_path / "run", path, model={"frozen": ["encoder.0.*"]})


def test_frozen_blocks_stay_in_eval_mode(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, model=FROZEN_FIRST_BLOCK)
    model = model_of(session)
    batch_norm = model.backbone.module[1]
    running_mean = batch_norm.running_mean.clone()

    run(session)
    model.train()

    assert not batch_norm.training
    assert model.backbone.module[4].training  # trainable: follows the model
    assert model.backbone.module[3].training  # dropout outside the frozen block
    torch.testing.assert_close(batch_norm.running_mean, running_mean)


def test_frozen_blocks_are_in_eval_mode_once_built(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, model=FROZEN_FIRST_BLOCK)

    assert not model_of(session).backbone.module[1].training


def test_frozen_blocks_stay_in_eval_mode_when_the_backbone_resets_itself(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    config = fine_tuning_config(
        tmp_path / "run", path,
        bindings={"backbone": "ft_resetting_backbone"},
        model={"frozen": ["layer.*"]},
        ft_resetting_backbone={},
    )
    del config["checkpoint_module"]
    session = TrainingSession(config)
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")

    with session:
        assert not model_of(session).backbone.layer.training


def test_frozen_eval_false_leaves_frozen_blocks_training(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path, model={**FROZEN_FIRST_BLOCK, "frozen_eval": False},
    )
    batch_norm = model_of(session).backbone.module[1]
    running_mean = batch_norm.running_mean.clone()

    run(session)

    assert batch_norm.training
    assert not torch.equal(batch_norm.running_mean, running_mean)


def test_frozen_eval_without_frozen_parameters_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="frozen_eval is true but nothing is frozen"):
        fine_tuning_session(tmp_path / "run", path, model={"frozen_eval": True})


@pytest.mark.parametrize("frozen", [False, 0, "", {}, None])
def test_a_frozen_value_that_is_not_a_list_of_patterns_is_refused(tmp_path, frozen):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="frozen must be a non-empty list of parameter name patterns"):
        fine_tuning_session(tmp_path / "run", path, model={"frozen": frozen})


def test_an_empty_frozen_list_freezes_nothing(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, model={"frozen": []})

    assert all(p.requires_grad for p in model_of(session).parameters())


def test_param_groups_give_backbone_and_head_their_own_rate(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, optimizer={
        "optimizer": {"name": "SGD", "kwargs": {"lr": 0.1}},
        # The leading `*` covers the DDP stand-in's `module.` prefix.
        "param_groups": [{"match": ["*backbone.*"], "kwargs": {"lr": 0.0}}],
    })
    model = model_of(session)
    backbone_before = parameters(model.backbone)
    head_before = parameters(model.head)

    run(session)

    for name, value in parameters(model.backbone).items():
        torch.testing.assert_close(value, backbone_before[name])
    assert not torch.equal(model.head.linear.weight, head_before["linear.weight"])


# -- fine_tuned_model: backbone_output --------------------------------------------------


@pytest.mark.parametrize("selector, expected", [
    ("pooled", lambda encoded: encoded),
    ("twice", lambda encoded: 2 * encoded),
])
def test_backbone_output_picks_a_key(tmp_path, selector, expected):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        source={"submodule": "", "method": "features"},
        model={"backbone_output": selector},
    )
    model = model_of(session).eval()
    encoded = model.backbone.module.encoder(INPUTS)

    torch.testing.assert_close(model(INPUTS), model.head(expected(encoded)))


def test_backbone_output_picks_an_index(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        bindings={"backbone": "ft_plain_backbone"},
        model={"backbone_output": 1},
        ft_plain_backbone={},
    )
    model = model_of(session)
    model.backbone.forward = lambda x: (x, model.backbone.layer(x))

    torch.testing.assert_close(model(INPUTS), model.head(model.backbone.layer(INPUTS)))


class CountingOutput:
    """A backbone output whose part is a property, counting its reads."""

    def __init__(self, pooled):
        self._pooled = pooled
        self.reads = 0

    @property
    def pooled(self):
        self.reads += 1
        return self._pooled


def test_backbone_output_reads_an_attribute_once(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        bindings={"backbone": "ft_plain_backbone"},
        model={"backbone_output": "pooled"},
        ft_plain_backbone={},
    )
    model = model_of(session)
    outputs = []

    def forward(x):
        outputs.append(CountingOutput(model.backbone.layer(x)))
        return outputs[-1]

    model.backbone.forward = forward

    torch.testing.assert_close(model(INPUTS), model.head(model.backbone.layer(INPUTS)))
    assert outputs[0].reads == 1


@pytest.mark.parametrize("selector, error, message", [
    ("pooled", TypeError, "backbone returned a Tensor without such a key"),
    # A tensor attribute is never a part of the output.
    ("shape", TypeError, "backbone returned a Tensor without such a key"),
    (0, TypeError, "is the index 0, but the backbone returned a Tensor"),
])
def test_backbone_output_that_does_not_fit_is_refused(tmp_path, selector, error, message):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path, model={"backbone_output": selector},
    )

    with pytest.raises(error, match=message):
        model_of(session)(INPUTS)


class ProbedTensor(torch.Tensor):
    """A tensor subclass whose `pooled` property must never be read."""

    @property
    def pooled(self):
        raise RuntimeError("a tensor attribute was read")


def test_backbone_output_never_looks_up_a_tensor_attribute(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        bindings={"backbone": "ft_plain_backbone"},
        model={"backbone_output": "pooled"},
        ft_plain_backbone={},
    )
    model = model_of(session)
    model.backbone.forward = lambda x: model.backbone.layer(x).as_subclass(ProbedTensor)

    with pytest.raises(TypeError, match="backbone returned a ProbedTensor without such a key"):
        model(INPUTS)


def test_backbone_output_names_the_keys_it_had(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        source={"submodule": "", "method": "features"},
        model={"backbone_output": "hidden"},
    )

    with pytest.raises(KeyError, match=r"keys \['pooled', 'twice'\]"):
        model_of(session)(INPUTS)


# -- checkpoints ------------------------------------------------------------------------


def test_each_part_checkpoints_its_own_weights(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, model=FROZEN_FIRST_BLOCK)
    run(session)
    saved = Checkpointer.save_checkpoint(session, tmp_path / "fine-tuned")

    backbone = Checkpointer.load_component_state(saved, "checkpoint_module")
    head = Checkpointer.load_component_state(saved, "ft_probe_head")
    composite = Checkpointer.load_component_state(saved, "fine_tuned_model")

    assert set(backbone["state_dict"]) == set(
        model_of(session).backbone.state_dict()
    )
    assert set(head["state_dict"]) == {"linear.weight", "linear.bias"}
    assert composite["state_dict"] == {}


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    optimizer = {"optimizer": {"name": "AdamW", "kwargs": {"lr": 0.05}}}
    uninterrupted = fine_tuning_session(
        tmp_path / "a", path, model=FROZEN_FIRST_BLOCK,
        optimizer=optimizer, max_iterations=4,
    )
    run(uninterrupted)

    paused = fine_tuning_session(
        tmp_path / "b", path, model=FROZEN_FIRST_BLOCK,
        optimizer=optimizer, max_iterations=4,
    )
    run(paused, 2)
    saved = Checkpointer.save_checkpoint(paused, tmp_path / "paused")
    resumed = Checkpointer.load_checkpoint(saved)
    model = model_of(resumed)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert frozen == [
        "backbone.module.0.weight", "backbone.module.0.bias",
        "backbone.module.1.weight", "backbone.module.1.bias",
    ]
    with resumed:
        assert list(resumed) == [3, 4]
    for name, value in parameters(model).items():
        torch.testing.assert_close(value, model_of(uninterrupted).get_parameter(name))


def test_resuming_needs_the_source_checkpoint(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, max_iterations=2)
    run(session, 1)
    saved = Checkpointer.save_checkpoint(session, tmp_path / "paused")
    shutil.move(path, tmp_path / "moved")

    with pytest.raises(FileNotFoundError, match="checkpoint_module.checkpoint does not exist"):
        Checkpointer.load_checkpoint(saved)
