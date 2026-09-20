from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

from training_framework.components import (
    ModuleResource,
    Step,
    component_registry,
    requires_resource,
    role_registry,
    step,
    topological_sort_of_components,
)
from training_framework.components.builtin import TrainedModel
from training_framework.components.builtin.transformer import (
    AttentionPooling,
    ClassToken,
    ConditionedPoolingQuery,
    ConvPatchEmbedding,
    LearnedPoolingQuery,
    LearnedPositionalEmbedding2D,
    PatchTransformer,
    PooledPatchTransformer,
    SinusoidalPositionalEmbedding2D,
    TokenReduction,
    TorchTransformerEncoder,
)
from training_framework.session import AnalysisSession, TrainingSession


EMBED_DIM = 8

_PATCH_TOKENS = (
    "tests.components.builtin.transformer.test_components.PatchTokens"
)


class PatchTokens(nn.Module):
    """A user-style encoder producing (B, N, D) tokens from images.

    Declared here, at module scope, because conditioning encoders are resolved
    from a dotted path and must be plain modules rather than components.
    """

    def __init__(self, in_channels=3, patch_size=4, embed_dim=EMBED_DIM):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.embed_dim = embed_dim

    def forward(self, images):
        return self.projection(images).flatten(2).transpose(1, 2)


BLOCK_CONFIGS = {
    "conv_patch_embedding": {"in_channels": 3, "patch_size": 4, "embed_dim": EMBED_DIM},
    "learned_positional_embedding_2d": {"grid_size": [2, 2], "embed_dim": EMBED_DIM},
    "sinusoidal_positional_embedding_2d": {"embed_dim": EMBED_DIM},
    "torch_transformer_encoder": {
        "embed_dim": EMBED_DIM,
        "num_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
    },
    "attention_pooling": {"embed_dim": EMBED_DIM, "num_heads": 2},
    "learned_pooling_query": {"embed_dim": EMBED_DIM, "num_queries": 2},
    "conditioned_pooling_query": {
        "embed_dim": EMBED_DIM,
        "inputs": {
            "obj_patch": {"module": _PATCH_TOKENS, "reduce": "mean"},
            "obj_patch_location": {
                "module": "torch.nn.Linear",
                "in_features": 2,
                "out_features": EMBED_DIM,
            },
        },
        "hidden_dims": [EMBED_DIM],
    },
}

BLOCK_CLASSES = {
    "conv_patch_embedding": ConvPatchEmbedding,
    "learned_positional_embedding_2d": LearnedPositionalEmbedding2D,
    "sinusoidal_positional_embedding_2d": SinusoidalPositionalEmbedding2D,
    "torch_transformer_encoder": TorchTransformerEncoder,
    "attention_pooling": AttentionPooling,
    "learned_pooling_query": LearnedPoolingQuery,
    "conditioned_pooling_query": ConditionedPoolingQuery,
}


def _block(name, **overrides):
    return BLOCK_CLASSES[name]({**BLOCK_CONFIGS[name], **overrides})


def _reduced_patch_query(**overrides):
    """A conditioned query whose patch encoder reduces tokens to one vector."""
    config = {
        "embed_dim": EMBED_DIM,
        "inputs": {
            "obj_patch": {"module": _PATCH_TOKENS, "reduce": "mean"},
            "obj_patch_location": {
                "module": "torch.nn.Linear",
                "in_features": 2,
                "out_features": EMBED_DIM,
            },
        },
        "hidden_dims": [16],
    }
    config.update(overrides)
    return ConditionedPoolingQuery(config)


def _session_config(root, *, max_iterations=1):
    return {
        "rng_seed": 3,
        "sessions_dir": str(root),
        "max_iterations": max_iterations,
        "device": "cpu",
        "components_package": "training_framework.components.builtin",
        "show_execution_graph": False,
    }


def _pooled_config(tmp_path, *, positional="learned_positional_embedding_2d",
                   query="learned_pooling_query", model="pooled_patch_transformer",
                   max_iterations=1, extra=None):
    bindings = {
        "model": model,
        "patch_embedding": "conv_patch_embedding",
        "positional_embedding": positional,
        "sequence_encoder": "torch_transformer_encoder",
    }
    blocks = ["conv_patch_embedding", positional, "torch_transformer_encoder"]
    if model == "pooled_patch_transformer":
        bindings.update({"pooling": "attention_pooling", "pooling_query": query})
        blocks += ["attention_pooling", query]
    config = {
        "session_config": _session_config(tmp_path / "training", max_iterations=max_iterations),
        "component_bindings": bindings,
        model: {},
        **{name: BLOCK_CONFIGS[name] for name in blocks},
    }
    config.update(extra or {})
    return config


def _conditioning(batch_size=2):
    return {
        "obj_patch": torch.randn(batch_size, 3, 4, 4),
        "obj_patch_location": torch.rand(batch_size, 2),
    }


# -- registration -------------------------------------------------------


def test_transformer_components_and_roles_are_registered():
    registry = component_registry()
    for name, block_class in BLOCK_CLASSES.items():
        assert registry[name] is block_class
    for session_type in ("training", "analysis"):
        scoped = component_registry(session_type)
        assert scoped["patch_transformer"] is PatchTransformer
        assert scoped["pooled_patch_transformer"] is PooledPatchTransformer
    # Composites need roles without defaults, so they stay out of the shared
    # scope, which must remain sortable on its own.
    assert "patch_transformer" not in registry
    topological_sort_of_components()
    for role_name in (
        "patch_embedding",
        "positional_embedding",
        "sequence_encoder",
        "pooling",
        "pooling_query",
    ):
        assert role_name in role_registry()
    assert set(PatchTransformer.required_resources) == {
        "patch_embedding", "positional_embedding", "sequence_encoder",
    }
    assert set(PooledPatchTransformer.required_resources) == set(
        PooledPatchTransformer.blocks()
    )


def test_every_block_is_a_module_resource_owning_its_weights():
    for name in BLOCK_CLASSES:
        block = _block(name)
        assert isinstance(block, ModuleResource)
        # No link step and no session: a block is complete once constructed.
        state = block.get_state()
        assert state["linked"] == {}
        assert set(state["state_dict"]) == set(
            dict(block.state_dict(keep_vars=True))
        )
        assert all(not p.is_meta for p in block.parameters())


def test_a_block_reports_invalid_config_at_construction():
    with pytest.raises(TypeError, match="config must be a mapping"):
        ConvPatchEmbedding([1, 2])
    with pytest.raises(ValueError, match="missing required keys"):
        ConvPatchEmbedding({"in_channels": 3, "patch_size": 4})
    with pytest.raises(ValueError, match="unknown keys"):
        _block("torch_transformer_encoder", depth=2)
    with pytest.raises(
        ValueError,
        match="Invalid torch_transformer_encoder config: .*num_heads",
    ):
        _block("torch_transformer_encoder", num_heads=3)


def test_a_block_config_is_isolated_from_callers():
    config = {"in_channels": 3, "patch_size": [4, 4], "embed_dim": EMBED_DIM}

    block = ConvPatchEmbedding(config)

    config["patch_size"][0] = 2
    block.config["patch_size"] = "nonsense"
    assert block.patch_size == (4, 4)


# -- patch embedding ----------------------------------------------------


def test_patch_embedding_tokenizes_square_and_non_square_patches():
    square = _block("conv_patch_embedding")
    assert square(torch.randn(2, 3, 8, 12)).shape == (2, 6, EMBED_DIM)
    assert square.grid_size(8, 12) == (2, 3)

    rectangular = ConvPatchEmbedding(
        {"in_channels": 1, "patch_size": [2, 4], "embed_dim": EMBED_DIM}
    )
    assert rectangular(torch.randn(2, 1, 4, 8)).shape == (2, 4, EMBED_DIM)
    assert rectangular.patch_size == (2, 4)


def test_patch_embedding_rejects_bad_inputs():
    block = _block("conv_patch_embedding")
    with pytest.raises(ValueError, match=r"Expected \[B, C, H, W\]"):
        block(torch.randn(3, 8, 8))
    with pytest.raises(ValueError, match="channels but expected 3"):
        block(torch.randn(1, 1, 8, 8))
    with pytest.raises(ValueError, match="must be divisible"):
        block(torch.randn(1, 3, 8, 10))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"in_channels": 0}, "in_channels"),
        ({"patch_size": [4]}, "patch_size"),
        ({"patch_size": True}, "patch_size"),
        ({"embed_dim": -1}, "embed_dim"),
    ],
)
def test_patch_embedding_validates_config(overrides, match):
    with pytest.raises(ValueError, match=match):
        _block("conv_patch_embedding", **overrides)


# -- positional embeddings ----------------------------------------------


def test_learned_positional_embedding_adds_table_at_trained_grid():
    block = LearnedPositionalEmbedding2D(
        {"grid_size": [2, 3], "embed_dim": 4, "init": "trunc_normal"}
    )
    tokens = torch.zeros(5, 6, 4)

    output = block(tokens, (2, 3))

    torch.testing.assert_close(output, block.table.expand(5, -1, -1))
    assert block.embeddings((2, 3)) is block.table


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_learned_positional_embedding_resizes_for_other_grids(mode):
    block = LearnedPositionalEmbedding2D({
        "grid_size": 2,
        "embed_dim": 4,
        "init": "trunc_normal",
        "interpolation_mode": mode,
    })
    assert block(torch.zeros(1, 12, 4), (3, 4)).shape == (1, 12, 4)


def test_learned_positional_embedding_defaults_to_zero_init():
    block = LearnedPositionalEmbedding2D({"grid_size": 2, "embed_dim": 4})
    assert torch.count_nonzero(block.table) == 0


@pytest.mark.parametrize(
    "block_name, config",
    [
        ("learned_positional_embedding_2d", {"grid_size": 2, "embed_dim": 4}),
        ("sinusoidal_positional_embedding_2d", {"embed_dim": 4}),
    ],
    ids=["learned", "sinusoidal"],
)
@pytest.mark.parametrize(
    "shape",
    [(3, 1, 4), (3, 4, 1), (3, 5, 4), (3, 4)],
    ids=["one_token", "one_feature", "wrong_token_count", "missing_batch_dim"],
)
def test_positional_embeddings_reject_token_shapes_that_would_broadcast(
        block_name, config, shape,
):
    # A 2x2 grid needs exactly (B, 4, 4); singleton dimensions would otherwise
    # broadcast silently into the wrong shape.
    block = BLOCK_CLASSES[block_name](config)
    with pytest.raises(ValueError, match="tokens for this grid size"):
        block(torch.zeros(*shape), (2, 2))


def test_learned_positional_embedding_validates_config():
    with pytest.raises(ValueError, match="init must be one of"):
        LearnedPositionalEmbedding2D({"grid_size": 2, "embed_dim": 4, "init": "ones"})
    with pytest.raises(ValueError, match="interpolation_mode"):
        LearnedPositionalEmbedding2D(
            {"grid_size": 2, "embed_dim": 4, "interpolation_mode": "area"}
        )


def test_sinusoidal_positional_embedding_has_no_parameters_and_any_grid():
    block = _block("sinusoidal_positional_embedding_2d")

    assert list(block.parameters()) == []
    small = block.embeddings((2, 3))
    large = block.embeddings((4, 5))
    assert small.shape == (1, 6, EMBED_DIM)
    assert large.shape == (1, 20, EMBED_DIM)
    # Positions are distinct and the first row/column pattern is stable.
    assert torch.unique(small[0], dim=0).shape[0] == 6
    torch.testing.assert_close(small[0, 0], large[0, 0])
    assert block(torch.zeros(2, 6, EMBED_DIM), (2, 3)).shape == (2, 6, EMBED_DIM)


def test_sinusoidal_positional_embedding_validates_config():
    with pytest.raises(ValueError, match="divisible by 4"):
        SinusoidalPositionalEmbedding2D({"embed_dim": 6})
    with pytest.raises(ValueError, match="temperature"):
        SinusoidalPositionalEmbedding2D({"embed_dim": 8, "temperature": 1})


# -- numeric validation shared by the blocks ----------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(ValueError, match="finite non-negative number"):
        LearnedPositionalEmbedding2D(
            {"grid_size": 2, "embed_dim": 4, "init_std": value}
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        _block("attention_pooling", dropout=value)
    with pytest.raises(ValueError, match="finite non-negative number"):
        SinusoidalPositionalEmbedding2D({"embed_dim": 8, "temperature": value})
    with pytest.raises(ValueError, match="between 0 and 1"):
        _block("torch_transformer_encoder", dropout=value)
    with pytest.raises(ValueError, match="finite positive number"):
        _block("torch_transformer_encoder", layer_norm_eps=value)


@pytest.mark.parametrize("dropout", [-0.1, 1.5, 5])
def test_dropout_outside_zero_to_one_is_rejected(dropout):
    with pytest.raises(ValueError, match="between 0 and 1"):
        _block("attention_pooling", dropout=dropout)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _block("torch_transformer_encoder", dropout=dropout)


@pytest.mark.parametrize("eps", [0, -1e-5])
def test_non_positive_layer_norm_eps_is_rejected(eps):
    with pytest.raises(ValueError, match="layer_norm_eps must be a finite positive"):
        _block("torch_transformer_encoder", layer_norm_eps=eps)
    with pytest.raises(ValueError, match="layer_norm_eps must be a finite positive"):
        _block("torch_transformer_encoder", layer_norm_eps=eps, final_norm=True)


# -- encoder / pooling / queries ----------------------------------------


def test_transformer_encoder_encodes_tokens_and_honours_padding_mask():
    torch.manual_seed(0)
    block = _block(
        "torch_transformer_encoder",
        num_layers=2,
        activation="gelu",
        norm_first=True,
        final_norm=True,
    ).eval()
    tokens = torch.randn(1, 4, EMBED_DIM)
    mask = torch.tensor([[False, False, False, True]])

    output = block(tokens, key_padding_mask=mask)
    changed_padding = tokens.clone()
    changed_padding[0, 3] = 100.0

    assert output.shape == (1, 4, EMBED_DIM)
    torch.testing.assert_close(
        block(changed_padding, key_padding_mask=mask)[0, :3],
        output[0, :3],
    )
    assert isinstance(block.encoder.norm, nn.LayerNorm)


def test_transformer_encoder_validates_config():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        _block("torch_transformer_encoder", num_heads=3)
    with pytest.raises(ValueError, match="activation"):
        _block("torch_transformer_encoder", activation="tanh")


def test_attention_pooling_pools_to_query_count():
    block = _block("attention_pooling")
    assert block(
        torch.randn(3, 10, EMBED_DIM), torch.randn(3, 2, EMBED_DIM)
    ).shape == (3, 2, EMBED_DIM)


def test_attention_pooling_exposes_weights_to_forward_hooks_when_requested():
    block = _block("attention_pooling", need_weights=True)
    captured = []
    block.attention.register_forward_hook(
        lambda m, args, output: captured.append(output)
    )

    block(torch.randn(3, 10, EMBED_DIM), torch.randn(3, 1, EMBED_DIM))

    assert captured[0][1].shape == (3, 1, 10)


def test_learned_query_expands_over_batch_and_rejects_conditioning():
    block = _block("learned_pooling_query", num_queries=3)
    assert block(4).shape == (4, 3, EMBED_DIM)
    with pytest.raises(TypeError, match="takes no conditioning"):
        block(4, obj_patch=torch.zeros(4, 3, 4, 4))


# -- conditioned pooling query ------------------------------------------


def test_conditioned_query_builds_encoders_from_dotted_paths():
    query = _reduced_patch_query()

    assert isinstance(query.encoders["obj_patch"], TokenReduction)
    assert isinstance(query.encoders["obj_patch"].module, PatchTokens)
    assert query.encoders["obj_patch"].reduce == "mean"
    assert isinstance(query.encoders["obj_patch_location"], nn.Linear)
    assert query.input_names == ("obj_patch", "obj_patch_location")
    # 8x8 crops give 4 patch tokens; they are reduced to one vector.
    assert query(2, obj_patch=torch.randn(2, 3, 8, 8),
                 obj_patch_location=torch.randn(2, 2)).shape == (2, 1, EMBED_DIM)


def test_conditioned_query_accepts_a_block_component_as_an_encoder():
    query = ConditionedPoolingQuery({
        "embed_dim": EMBED_DIM,
        "inputs": {
            "obj_patch": {
                "module": (
                    "training_framework.components.builtin.transformer."
                    "ConvPatchEmbedding"
                ),
                "in_channels": 3,
                "patch_size": 4,
                "embed_dim": EMBED_DIM,
                "reduce": "mean",
            },
        },
    })

    # The block needs nothing from the session, so the query simply owns it.
    assert isinstance(query.encoders["obj_patch"].module, ConvPatchEmbedding)
    assert query(2, obj_patch=torch.randn(2, 3, 8, 8)).shape == (2, 1, EMBED_DIM)
    assert any(
        key.startswith("encoders.obj_patch.module.")
        for key in query.get_state()["state_dict"]
    )


def test_conditioned_query_accepts_any_nn_module_encoder():
    query = ConditionedPoolingQuery({
        "embed_dim": EMBED_DIM,
        "inputs": {
            "sequence": {
                "module": "torch.nn.Embedding",
                "num_embeddings": 5,
                "embedding_dim": EMBED_DIM,
                "reduce": "max",
            },
        },
    })

    assert query(2, sequence=torch.tensor([[0, 1, 2], [3, 4, 0]])).shape == (
        2, 1, EMBED_DIM,
    )


def test_conditioned_query_mlp_is_non_linear():
    query = _reduced_patch_query(hidden_dims=[16, 16], activation="relu")

    assert [type(layer) for layer in query.projection] == [
        nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear,
    ]
    assert query.projection[0].in_features == 16  # two encoded inputs


def test_conditioned_query_rejects_missing_extra_or_mis_batched_inputs():
    query = _reduced_patch_query()
    patch = torch.randn(2, 3, 4, 4)
    with pytest.raises(TypeError, match="expects inputs"):
        query(2, obj_patch=patch)
    with pytest.raises(TypeError, match="expects inputs"):
        query(2, obj_patch=patch, obj_patch_location=torch.randn(2, 2), extra=patch)
    with pytest.raises(ValueError, match="batch size 2, expected 3"):
        query(3, obj_patch=patch, obj_patch_location=torch.randn(2, 2))


def test_conditioned_query_rejects_encoders_with_the_wrong_output_size():
    with pytest.raises(ValueError, match="out_features=4, but must produce 8"):
        ConditionedPoolingQuery({
            "embed_dim": EMBED_DIM,
            "inputs": {
                "x": {"module": "torch.nn.Linear", "in_features": 2, "out_features": 4},
            },
        })

    # A module that declares nothing is checked on its first forward pass.
    undeclared = ConditionedPoolingQuery({
        "embed_dim": EMBED_DIM,
        "inputs": {"x": {"module": "torch.nn.Identity"}},
    })
    with pytest.raises(ValueError, match=r"must return \(3, 8\) features; got \(3, 4\)"):
        undeclared(3, x=torch.randn(3, 4))


@pytest.mark.parametrize(
    "inputs, error, match",
    [
        ({}, ValueError, "inputs must be a non-empty mapping"),
        ({"x": "nope"}, ValueError, "inputs.x must be a mapping"),
        ({"x": {"in_features": 2}}, ValueError, "inputs.x.module must be a fully-qualified"),
        ({"x": {"module": "Linear"}}, ValueError, "fully-qualified dotted path"),
        ({"x": {"module": "no_such_module_xyz.Thing"}}, ImportError, "could not be imported"),
        ({"x": {"module": "torch.nn.NotAReal"}}, ValueError, "has no attribute"),
        ({"x": {"module": "torch.optim.SGD"}}, TypeError, "does not resolve to an nn.Module"),
        (
            {"x": {"module": (
                "training_framework.components.builtin.transformer."
                "PooledPatchTransformer"
            )}},
            TypeError,
            "declares prerequisites or a lifecycle the session drives",
        ),
        ({"x": {"module": "torch.nn.Linear"}}, TypeError, "Invalid .*inputs.x config"),
        ({"bad name": {"module": "torch.nn.Identity"}}, ValueError, "identifiers"),
        ({"forward": {"module": "torch.nn.Identity"}}, ValueError, "clashes with an attribute"),
        ({"keys": {"module": "torch.nn.Identity"}}, ValueError, "clashes with an attribute"),
    ],
)
def test_conditioned_query_reports_bad_input_specs(inputs, error, match):
    with pytest.raises(error, match=match):
        ConditionedPoolingQuery({"embed_dim": EMBED_DIM, "inputs": inputs})


@pytest.mark.parametrize("activation", [None, "none"])
def test_conditioned_query_activations_are_optional(activation):
    query = _reduced_patch_query(hidden_dims=[16], activation=activation)

    assert [type(layer) for layer in query.projection] == [nn.Linear, nn.Linear]
    assert query(2, obj_patch=torch.randn(2, 3, 4, 4),
                 obj_patch_location=torch.randn(2, 2)).shape == (2, 1, EMBED_DIM)


def test_conditioned_query_validates_hidden_dims_and_activation():
    with pytest.raises(ValueError, match="hidden_dims"):
        _reduced_patch_query(hidden_dims=8)
    with pytest.raises(ValueError, match="activation must be one of"):
        _reduced_patch_query(activation="swish")


# -- composite model ----------------------------------------------------


def _composite(model_class, tmp_path, **overrides):
    """Build a composite through a real session, which is how it is wired."""
    model = "pooled_patch_transformer" if model_class is PooledPatchTransformer \
        else "patch_transformer"
    config = _pooled_config(tmp_path, model=model, **overrides)
    return TrainingSession(config)


def test_patch_transformer_config_only_accepts_class_token(tmp_path):
    assert not _composite(PatchTransformer, tmp_path).get_resource(
        "model"
    ).has_class_token

    with_token = _composite(
        PatchTransformer, tmp_path,
        extra={"patch_transformer": {"class_token": True}},
    )
    assert with_token.get_resource("model").has_class_token

    with pytest.raises(ValueError, match="unknown keys"):
        _composite(
            PatchTransformer, tmp_path,
            extra={"patch_transformer": {"embed_dim": 8}},
        )
    with pytest.raises(TypeError, match="class_token must be a boolean or a mapping"):
        _composite(
            PatchTransformer, tmp_path,
            extra={"patch_transformer": {"class_token": "yes"}},
        )
    with pytest.raises(TypeError, match="unexpected options"):
        _composite(
            PatchTransformer, tmp_path,
            extra={"patch_transformer": {"class_token": {"std": 0.1}}},
        )
    with pytest.raises(ValueError, match="init_std"):
        _composite(
            PatchTransformer, tmp_path,
            extra={"patch_transformer": {"class_token": {"init_std": -1}}},
        )


def test_a_composite_owns_only_its_class_token(tmp_path):
    session = _composite(
        PooledPatchTransformer, tmp_path,
        extra={"pooled_patch_transformer": {"class_token": True}},
    )

    state = session.get_state()["components_state"]

    assert set(state["pooled_patch_transformer"]["state"]["state_dict"]) == {
        "class_token.token",
    }
    # Every block checkpoints its own weights, exactly once.
    assert set(state["conv_patch_embedding"]["state"]["state_dict"]) == {
        "projection.weight", "projection.bias",
    }
    assert state["pooled_patch_transformer"]["state"]["linked"] == {
        role: name for role, name in zip(
            PooledPatchTransformer.blocks(),
            [
                "conv_patch_embedding",
                "learned_positional_embedding_2d",
                "torch_transformer_encoder",
                "attention_pooling",
                "learned_pooling_query",
            ],
        )
    }


def test_a_composite_without_a_class_token_owns_nothing(tmp_path):
    session = _composite(PatchTransformer, tmp_path)

    state = session.get_state()["components_state"]

    assert state["patch_transformer"]["state"]["state_dict"] == {}


def test_class_token_is_prepended_after_positional_embedding(tmp_path):
    session = _composite(
        PatchTransformer, tmp_path,
        extra={"patch_transformer": {"class_token": True}},
    )
    model = session.get_resource("model")

    assert isinstance(model.class_token, ClassToken)
    assert model.class_token.embed_dim == EMBED_DIM
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 5, EMBED_DIM)
    # The class token has no position, so resizing the positional table
    # for a larger grid still works.
    assert model(torch.randn(2, 3, 12, 16)).shape == (2, 13, EMBED_DIM)
    assert "class_token.token" in dict(model.named_parameters())


def test_class_token_widens_the_padding_mask_for_encoder_and_pooling(tmp_path):
    session = _composite(
        PooledPatchTransformer, tmp_path,
        extra={"pooled_patch_transformer": {"class_token": True}},
    )
    model = session.get_resource("model")
    seen_masks = []
    model.sequence_encoder.register_forward_hook(
        lambda module, args, kwargs, output: seen_masks.append(kwargs["key_padding_mask"]),
        with_kwargs=True,
    )
    model.pooling.register_forward_hook(
        lambda module, args, kwargs, output: seen_masks.append(kwargs["key_padding_mask"]),
        with_kwargs=True,
    )
    mask = torch.tensor([[False, False, False, True], [False, True, True, True]])

    output = model.eval()(torch.randn(2, 3, 8, 8), key_padding_mask=mask)

    assert output.shape == (2, 2, EMBED_DIM)
    expected = torch.cat([torch.zeros(2, 1, dtype=torch.bool), mask], dim=1)
    assert len(seen_masks) == 2
    for seen in seen_masks:
        torch.testing.assert_close(seen, expected)


def test_a_composite_is_usable_as_soon_as_it_is_constructed(tmp_path):
    session = _composite(PatchTransformer, tmp_path)
    model = session.get_resource("model")

    # No `with session:` -- no setup has run.
    assert model.embed_dim == EMBED_DIM
    assert isinstance(model.patch_embedding, ConvPatchEmbedding)
    assert model.patch_embedding is session.get_resource("conv_patch_embedding")
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 4, EMBED_DIM)
    # Larger inputs resize the learned positional table.
    assert model(torch.randn(2, 3, 12, 16)).shape == (2, 12, EMBED_DIM)


def test_pooled_patch_transformer_forwards_conditioning_to_query(tmp_path):
    session = _composite(
        PooledPatchTransformer, tmp_path, query="conditioned_pooling_query",
    )
    model = session.get_resource("model")

    assert model(torch.randn(2, 3, 8, 8), **_conditioning()).shape == (
        2, 1, EMBED_DIM,
    )
    with pytest.raises(TypeError, match="expects inputs"):
        model(torch.randn(2, 3, 8, 8))


def test_blocks_that_disagree_on_embed_dim_are_rejected(tmp_path):
    config = _pooled_config(tmp_path, model="patch_transformer")
    config["torch_transformer_encoder"] = {
        **BLOCK_CONFIGS["torch_transformer_encoder"],
        "embed_dim": 16,
    }

    with pytest.raises(ValueError, match="disagree on embed_dim"):
        TrainingSession(config)


def test_a_pooling_query_that_disagrees_with_the_chain_is_rejected(tmp_path):
    config = _pooled_config(tmp_path)
    config["learned_pooling_query"] = {"embed_dim": 16, "num_queries": 2}

    with pytest.raises(
        ValueError,
        match="'pooling_query' produces 16 but the encoded tokens are 8",
    ):
        TrainingSession(config)


def test_a_block_without_an_embed_dim_is_rejected(tmp_path):
    from training_framework.components import resource

    @resource("widthless_encoder")
    class Widthless(ModuleResource):
        def forward(self, tokens, key_padding_mask=None):
            return tokens

    config = _pooled_config(tmp_path, model="patch_transformer")
    config["component_bindings"]["sequence_encoder"] = "widthless_encoder"
    del config["torch_transformer_encoder"]
    config["widthless_encoder"] = {}

    with pytest.raises(TypeError, match="must expose an integer embed_dim"):
        TrainingSession(config)


# -- real sessions ------------------------------------------------------


def _register_sgd_step():
    class _SgdStep(Step):
        def __init__(self, config):
            self.losses = []

        def run(self, session):
            model = session.get_resource("model")
            model.train()
            output = model(torch.ones(2, 3, 8, 8), **_conditioning())
            loss = output.pow(2).mean()
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter -= 0.1 * parameter.grad
            self.losses.append(loss.item())

    return requires_resource("model")(step("transformer_sgd", session_type="training")(_SgdStep))


def test_training_session_trains_checkpoints_and_restores_the_model(tmp_path):
    _register_sgd_step()
    session = TrainingSession(_pooled_config(
        tmp_path,
        query="conditioned_pooling_query",
        max_iterations=2,
        extra={"transformer_sgd": {}},
    ))
    initial_state = session.get_state()["components_state"]
    # Components are built during construction, so the model already carries
    # real weights before any worker runs.
    assert initial_state["conv_patch_embedding"]["state"]["state_dict"]

    with session:
        model = session.get_resource("model")
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for _ in session:
            pass

    assert any(
        not torch.equal(before[name], p) for name, p in model.named_parameters()
    )
    assert {name.split(".")[0] for name, _ in model.named_parameters()} == set(
        PooledPatchTransformer.blocks()
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(session, checkpoint_path)
    restored_session = torch.load(checkpoint_path, weights_only=False)
    restored_model = restored_session.get_resource("model")

    # The checkpoint is self-contained: no setup() needed to use the model.
    images = torch.randn(2, 3, 8, 8)
    conditioning = _conditioning()
    torch.testing.assert_close(
        restored_model.eval()(images, **conditioning),
        model.eval()(images, **conditioning),
    )


def test_session_state_round_trip_builds_in_a_worker(tmp_path):
    session = TrainingSession(_pooled_config(
        tmp_path,
        positional="sinusoidal_positional_embedding_2d",
        query="learned_pooling_query",
    ))

    worker_session = TrainingSession.from_state(
        pickle.loads(pickle.dumps(session.get_state()))
    )
    with worker_session:
        model = worker_session.get_resource("model")
        assert isinstance(model.positional_embedding, SinusoidalPositionalEmbedding2D)
        assert model(torch.randn(3, 3, 8, 8)).shape == (3, 2, EMBED_DIM)


def test_patch_transformer_without_pooling_in_a_session(tmp_path):
    session = TrainingSession(_pooled_config(tmp_path, model="patch_transformer"))
    with session:
        model = session.get_resource("model")
        assert type(model) is PatchTransformer
        assert model(torch.randn(1, 3, 8, 8)).shape == (1, 4, EMBED_DIM)


@pytest.mark.parametrize("class_token", [False, True])
def test_trained_model_loads_transformer_checkpoint_for_analysis(tmp_path, class_token):
    config = _pooled_config(tmp_path, query="conditioned_pooling_query")
    config["pooled_patch_transformer"] = {"class_token": class_token}
    session = TrainingSession(config)
    with session:
        trained = session.get_resource("model")
    checkpoint_path = tmp_path / "training.pt"
    torch.save(session, checkpoint_path)

    analysis = AnalysisSession({
        "session_config": _session_config(tmp_path / "analysis"),
        "trained_model": {"model_checkpoint_path": str(checkpoint_path)},
    })
    with analysis:
        model = analysis.get_resource("trained_model").model
        assert isinstance(analysis.get_resource("trained_model"), TrainedModel)
        assert model.has_class_token is class_token
        images = torch.randn(2, 3, 8, 8)
        conditioning = _conditioning()
        torch.testing.assert_close(
            model(images, **conditioning),
            trained.eval()(images, **conditioning),
        )


def test_a_block_is_shared_by_two_composites_as_one_instance(tmp_path):
    config = _pooled_config(tmp_path)
    config["patch_transformer"] = {}

    session = TrainingSession(config)

    pooled = session.get_resource("pooled_patch_transformer")
    plain = session.get_resource("patch_transformer")
    assert pooled.patch_embedding is plain.patch_embedding
