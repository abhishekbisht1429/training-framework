"""Fine-tuning: `import_components`, `module_part` and `fine_tuned_model`,
through real sessions.

A pretraining session is checkpointed to disk; a fine-tuning session imports
its model (`import_components`), takes a part of it through `module_part`,
puts a head on it through `fine_tuned_model`, and trains with the built-in
optimizer chain. DDP is replaced by the recording stand-in, as the other
built-in tests do.
"""

from __future__ import annotations

import copy
import shutil

import pytest
import torch
from torch import nn
from torch.nn import functional

from tests.test_utils import make_config, resource_named, stub_process_group
from training_framework.components import (
    ModuleResource,
    Resource,
    StatefulResource,
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
    """A model the session must set up."""

    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(4, 5)
        self.set_up = False

    def setup(self, session):
        super().setup(session)
        self.set_up = True


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
    """A backbone of the user's own, not from another run."""

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


class Block(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x)


@requires_resource("block")
class Composite(ModuleResource):
    """Wired to a block, with one weight of its own, as a class token is."""

    def __init__(self, config=None):
        super().__init__(config)
        self.block = self.get_dependency("block")
        self.scale = nn.Parameter(torch.ones(5))

    def forward(self, x):
        return self.block(x) * self.scale


@requires_resource("composite")
class Outer(ModuleResource):
    """Holds a wired composite, which a module_part can take out of it."""

    def __init__(self, config=None):
        super().__init__(config)
        self.inner = self.get_dependency("composite")
        self.extra = nn.Linear(5, 5)

    def forward(self, x):
        return self.extra(self.inner(x))


class Counter(StatefulResource):
    """A prerequisite that is not a module and has state of its own."""

    def __init__(self, config=None):
        super().__init__(config)
        self.count = 0

    def get_state(self):
        return {"count": self.count}

    def set_state(self, state):
        self.count = state["count"]

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


@requires_resource("counter")
class CountedModel(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.counter = self.get_dependency("counter")
        self.linear = nn.Linear(4, 5)

    def forward(self, x):
        self.counter.count += 1
        return self.linear(x)


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
    resource("ft_block", overwrite=True)(Block)
    resource("ft_composite", overwrite=True)(Composite)
    resource("ft_outer", overwrite=True)(Outer)
    resource("ft_counter", overwrite=True)(Counter)
    resource("ft_counted_model", overwrite=True)(CountedModel)
    step("ft_loss", overwrite=True)(FineTuneLoss)


@pytest.fixture(autouse=True)
def _components():
    _register()


# -- helpers -----------------------------------------------------------------------


def wired_checkpoint(tmp_path, bindings, **components):
    """Write a pretraining run wired as `bindings` says, `components` giving
    each implementation's config; return its path and its `model`."""
    config = make_config(tmp_path / "pretrain", seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = bindings
    config.update(components)
    session = TrainingSession(config)
    path = Checkpointer.save_checkpoint(session, tmp_path / "pretrained")
    return path, resource_named(session, bindings["model"])


def pretrain_checkpoint(tmp_path, model="ft_pretrain_model", **model_config):
    """Write a pretraining run whose `model` is `model`; return its path and
    the pretrained model."""
    return wired_checkpoint(tmp_path, {"model": model}, **{model: model_config})


def fine_tuning_config(
        tmp_path,
        checkpoint,
        *,
        part=None,
        imports=None,
        model=None,
        bindings=None,
        optimizer=None,
        max_iterations=3,
        **extra,
):
    """Import the pretrained `model` as `source`, and take its encoder
    through `module_part` as the backbone."""
    config = make_config(tmp_path, max_iterations=max_iterations)
    config["session_config"]["show_execution_graph"] = False
    config["import_components"] = {
        "pretrained": {
            "checkpoint": str(checkpoint), "role": "source", **(imports or {}),
        },
    }
    config["component_bindings"] = {
        "model": "fine_tuned_model",
        "backbone": "module_part",
        "head": "ft_probe_head",
        **(bindings or {}),
    }
    config.update({
        "module_part": {"submodule": "encoder", **(part or {})},
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


def imported_backbone_config(tmp_path, checkpoint, *, imports=None, **kwargs):
    """The imported resource fills `backbone` itself, with no module_part."""
    config = fine_tuning_config(
        tmp_path, checkpoint,
        imports={"role": "backbone", **(imports or {})}, **kwargs,
    )
    del config["component_bindings"]["backbone"]
    del config["module_part"]
    return config


def without_import(config):
    del config["import_components"]
    del config["module_part"]
    return config


def quiet(session):
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    return session


def fine_tuning_session(tmp_path, checkpoint, **kwargs):
    return quiet(TrainingSession(fine_tuning_config(tmp_path, checkpoint, **kwargs)))


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


# -- module_part -----------------------------------------------------------------------


def test_module_part_holds_the_selected_part_of_the_import(tmp_path):
    path, pretrained = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path)
    part = resource_named(session, "module_part")
    imported = resource_named(session, "ft_pretrain_model")

    assert part.module is imported.encoder
    names = [name for name, _ in part.named_parameters()]
    assert names and all(name.startswith("module.") for name in names)
    assert not any("pretext_head" in name for name in names)
    for name, parameter in pretrained.encoder.named_parameters():
        torch.testing.assert_close(part.module.get_parameter(name), parameter)


def test_module_part_owns_no_weights(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path)

    assert resource_named(session, "module_part").get_state()["state_dict"] == {}
    owned = resource_named(session, "ft_pretrain_model").get_state()["state_dict"]
    assert {name.split(".")[0] for name in owned} == {"encoder", "pretext_head"}


def test_module_part_defaults_to_the_whole_source(tmp_path):
    path, pretrained = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path, part={"submodule": ""},
        ft_probe_head={"in_features": 7},
    )

    held = resource_named(session, "module_part").module
    assert held is resource_named(session, "ft_pretrain_model")
    torch.testing.assert_close(
        held.pretext_head.weight, pretrained.pretext_head.weight,
    )


def test_module_part_calls_the_configured_method(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(
        tmp_path / "run", path,
        part={"submodule": "", "method": "features"},
        model={"backbone_output": "pooled"},
    )

    backbone = resource_named(session, "module_part").eval()
    output = backbone(INPUTS)
    assert set(output) == {"pooled", "twice"}
    torch.testing.assert_close(output["pooled"], backbone.module.encoder(INPUTS))


def test_a_missing_submodule_names_the_ones_that_exist(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match=r"no attribute 'encoderr'.*'encoder', 'pretext_head'"):
        fine_tuning_session(tmp_path / "run", path, part={"submodule": "encoderr"})


def test_a_submodule_that_is_not_a_module_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(TypeError, match="ft_pretrain_model.training is a bool, not an nn.Module"):
        fine_tuning_session(tmp_path / "run", path, part={"submodule": "training"})


def test_a_missing_method_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="has no method 'embed'"):
        fine_tuning_session(tmp_path / "run", path, part={"method": "embed"})


def test_a_source_that_is_not_a_module_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    config = without_import(fine_tuning_config(
        tmp_path / "run", path,
        bindings={"source": "ft_not_a_module"}, ft_not_a_module={},
    ))
    config["module_part"] = {}

    with pytest.raises(TypeError, match="role 'source' is filled by NotAModule"):
        TrainingSession(config)


# -- the import -------------------------------------------------------------------------


def test_the_imported_model_is_set_up_by_the_session(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path, model="ft_driven_model")
    session = fine_tuning_session(
        tmp_path / "run", path, part={"submodule": "linear"},
    )
    imported = resource_named(session, "ft_driven_model")
    assert not imported.set_up

    with session:
        assert imported.set_up


def test_a_resource_the_checkpoint_does_not_hold_is_refused(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="import_components.pretrained.resource: The checkpoint has no component 'teacher'"):
        fine_tuning_session(tmp_path / "run", path, imports={"resource": "teacher"})


def test_a_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="import_components.pretrained.checkpoint does not exist"):
        fine_tuning_session(tmp_path / "run", tmp_path / "nowhere")


def composite_checkpoint(tmp_path):
    return wired_checkpoint(
        tmp_path, {"model": "ft_composite", "block": "ft_block"},
        ft_composite={}, ft_block={},
    )


def test_a_wired_composite_is_imported_as_components(tmp_path):
    path, pretrained = composite_checkpoint(tmp_path)
    session = quiet(TrainingSession(imported_backbone_config(tmp_path / "run", path)))
    composite = resource_named(session, "ft_composite")
    model = model_of(session)

    assert composite.block is resource_named(session, "ft_block")
    assert model.backbone is composite
    assert sorted(name for name, _ in model.backbone.named_parameters()) == [
        "block.linear.bias", "block.linear.weight", "scale",
    ]
    for name, parameter in pretrained.named_parameters():
        torch.testing.assert_close(composite.get_parameter(name), parameter)

    before = parameters(composite)
    run(session)

    # The prerequisite's weights and the composite's own are both trained.
    for name, value in parameters(composite).items():
        assert not torch.equal(value, before[name]), name


def test_module_part_can_take_a_wired_component_out_of_its_source(tmp_path):
    path, _ = wired_checkpoint(
        tmp_path,
        {"model": "ft_outer", "composite": "ft_composite", "block": "ft_block"},
        ft_outer={}, ft_composite={}, ft_block={},
    )
    session = fine_tuning_session(tmp_path / "run", path, part={"submodule": "inner"})
    part = resource_named(session, "module_part")

    # A component that declares prerequisites, held inside the source: it
    # is the source's (and its own), never the part's.
    assert part.module is resource_named(session, "ft_composite")
    assert part.get_state()["state_dict"] == {}
    run(session)


def test_an_imported_stateful_prerequisite_keeps_its_state(tmp_path):
    # The source run's counter has moved on when it is checkpointed.
    config = make_config(tmp_path / "source", seed=7)
    config["session_config"]["show_execution_graph"] = False
    config["component_bindings"] = {"model": "ft_counted_model", "counter": "ft_counter"}
    config.update(ft_counted_model={}, ft_counter={})
    source = TrainingSession(config)
    resource_named(source, "ft_counter").count = 5
    path = Checkpointer.save_checkpoint(source, tmp_path / "counted")

    session = quiet(TrainingSession(imported_backbone_config(tmp_path / "run", path)))
    counter = resource_named(session, "ft_counter")

    assert counter.count == 5
    assert model_of(session).backbone.counter is counter
    run(session)
    saved = Checkpointer.save_checkpoint(session, tmp_path / "fine-tuned")
    restored = resource_named(Checkpointer.load_checkpoint(saved), "ft_counter")
    assert restored.count == counter.count > 5


def test_a_bound_prerequisite_is_this_sessions_own(tmp_path):
    path, _ = wired_checkpoint(
        tmp_path, {"model": "ft_dataset_head", "dataset": "ft_toy_dataset"},
        ft_dataset_head={}, ft_toy_dataset={},
    )
    config = imported_backbone_config(
        tmp_path / "run", path,
        imports={"overwritten_dependencies": {"dataset": "dataset"}},
        bindings={"dataset": "ft_toy_dataset"},
        ft_toy_dataset={},
        ft_probe_head={"in_features": ToyDataset.num_classes},
    )
    session = quiet(TrainingSession(config))
    head = resource_named(session, "ft_dataset_head")

    assert head.get_dependency("dataset") is resource_named(session, "ft_toy_dataset")
    # Saved and restored with the wiring it was given here.
    saved = Checkpointer.save_checkpoint(session, tmp_path / "fine-tuned")
    Checkpointer.load_checkpoint(saved)


EMBED_DIM = 8
POOLED_BLOCKS = {
    "conv_patch_embedding": {"in_channels": 3, "patch_size": 4, "embed_dim": EMBED_DIM},
    "learned_positional_embedding_2d": {"grid_size": [2, 2], "embed_dim": EMBED_DIM},
    "torch_transformer_encoder": {
        "embed_dim": EMBED_DIM, "num_heads": 2, "num_layers": 1,
        "dim_feedforward": 16, "dropout": 0.0,
    },
    "attention_pooling": {"embed_dim": EMBED_DIM, "num_heads": 2},
    "learned_pooling_query": {"embed_dim": EMBED_DIM, "num_queries": 2},
}


def test_a_pooled_patch_transformer_is_imported_as_its_blocks(tmp_path):
    path, pretrained = wired_checkpoint(
        tmp_path,
        {
            "model": "pooled_patch_transformer",
            "patch_embedding": "conv_patch_embedding",
            "positional_embedding": "learned_positional_embedding_2d",
            "sequence_encoder": "torch_transformer_encoder",
            "pooling": "attention_pooling",
            "pooling_query": "learned_pooling_query",
        },
        pooled_patch_transformer={"class_token": True},
        **POOLED_BLOCKS,
    )
    session = quiet(TrainingSession(imported_backbone_config(
        tmp_path / "run", path, ft_probe_head={"in_features": EMBED_DIM},
    )))
    backbone = model_of(session).backbone.eval()

    for block in POOLED_BLOCKS:
        assert resource_named(session, block) in list(backbone.children())
    prefixes = {name.split(".")[0] for name, _ in backbone.named_parameters()}
    assert prefixes == {
        "patch_embedding", "positional_embedding", "sequence_encoder",
        "pooling", "pooling_query", "class_token",
    }
    images = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(backbone(images), pretrained.eval()(images))


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
    without_import(config)
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
    without_import(config)
    session = quiet(TrainingSession(config))

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
        part={"submodule": "", "method": "features"},
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
        part={"submodule": "", "method": "features"},
        model={"backbone_output": "hidden"},
    )

    with pytest.raises(KeyError, match=r"keys \['pooled', 'twice'\]"):
        model_of(session)(INPUTS)


# -- checkpoints ------------------------------------------------------------------------


def test_each_component_checkpoints_its_own_weights(tmp_path):
    path, _ = pretrain_checkpoint(tmp_path)
    session = fine_tuning_session(tmp_path / "run", path, model=FROZEN_FIRST_BLOCK)
    run(session)
    saved = Checkpointer.save_checkpoint(session, tmp_path / "fine-tuned")

    imported = Checkpointer.load_component_state(saved, "ft_pretrain_model")
    part = Checkpointer.load_component_state(saved, "module_part")
    head = Checkpointer.load_component_state(saved, "ft_probe_head")
    composite = Checkpointer.load_component_state(saved, "fine_tuned_model")

    # The whole imported model, including the part no one uses.
    assert set(imported["state_dict"]) == set(
        resource_named(session, "ft_pretrain_model").state_dict()
    )
    assert part["state_dict"] == {}
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


def test_resuming_does_not_need_the_source_checkpoint(tmp_path):
    path, _ = composite_checkpoint(tmp_path)
    paused = quiet(TrainingSession(imported_backbone_config(
        tmp_path / "run", path, max_iterations=4,
    )))
    run(paused, 2)
    trained = parameters(model_of(paused))
    saved = Checkpointer.save_checkpoint(paused, tmp_path / "paused")
    shutil.move(path, tmp_path / "moved")

    resumed = Checkpointer.load_checkpoint(saved)

    for name, value in parameters(model_of(resumed)).items():
        torch.testing.assert_close(value, trained[name])
    with resumed:
        assert list(resumed) == [3, 4]
