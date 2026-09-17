from __future__ import annotations

import pytest
import torch
from torch import nn

from training_framework.components.builtin.transformer import (
    AttentionPooling,
    ConditionedQuery,
    LearnedPositionalEmbedding2D,
    LearnedQuery,
    PatchEmbedding,
    SinusoidalPositionalEmbedding2D,
    TransformerEncoder,
)


# -- PatchEmbedding -----------------------------------------------------


def test_patch_embedding_tokenizes_square_and_non_square_patches():
    square = PatchEmbedding(in_channels=3, patch_size=4, embed_dim=8)
    assert square(torch.randn(2, 3, 8, 12)).shape == (2, 6, 8)
    assert square.grid_size(8, 12) == (2, 3)

    rectangular = PatchEmbedding(in_channels=1, patch_size=[2, 4], embed_dim=8)
    assert rectangular(torch.randn(2, 1, 4, 8)).shape == (2, 4, 8)
    assert rectangular.patch_size == (2, 4)


def test_patch_embedding_rejects_bad_inputs():
    module = PatchEmbedding(in_channels=3, patch_size=4, embed_dim=8)
    with pytest.raises(ValueError, match=r"Expected \[B, C, H, W\]"):
        module(torch.randn(3, 8, 8))
    with pytest.raises(ValueError, match="channels but expected 3"):
        module(torch.randn(1, 1, 8, 8))
    with pytest.raises(ValueError, match="must be divisible"):
        module(torch.randn(1, 3, 8, 10))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"in_channels": 0, "patch_size": 4, "embed_dim": 8}, "in_channels"),
        ({"in_channels": 3, "patch_size": [4], "embed_dim": 8}, "patch_size"),
        ({"in_channels": 3, "patch_size": True, "embed_dim": 8}, "patch_size"),
        ({"in_channels": 3, "patch_size": 4, "embed_dim": -1}, "embed_dim"),
    ],
)
def test_patch_embedding_validates_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PatchEmbedding(**kwargs)


# -- positional embeddings ----------------------------------------------


def test_learned_positional_embedding_adds_table_at_trained_grid():
    module = LearnedPositionalEmbedding2D(grid_size=[2, 3], embed_dim=4, init="trunc_normal")
    tokens = torch.zeros(5, 6, 4)

    output = module(tokens, (2, 3))

    torch.testing.assert_close(output, module.table.expand(5, -1, -1))
    assert module.embeddings((2, 3)) is module.table


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_learned_positional_embedding_resizes_for_other_grids(mode):
    module = LearnedPositionalEmbedding2D(
        grid_size=2,
        embed_dim=4,
        init="trunc_normal",
        interpolation_mode=mode,
    )
    assert module(torch.zeros(1, 12, 4), (3, 4)).shape == (1, 12, 4)


def test_learned_positional_embedding_defaults_to_zero_init():
    module = LearnedPositionalEmbedding2D(grid_size=2, embed_dim=4)
    assert torch.count_nonzero(module.table) == 0


def test_learned_positional_embedding_validates_arguments():
    with pytest.raises(ValueError, match="init must be one of"):
        LearnedPositionalEmbedding2D(grid_size=2, embed_dim=4, init="ones")
    with pytest.raises(ValueError, match="interpolation_mode"):
        LearnedPositionalEmbedding2D(grid_size=2, embed_dim=4, interpolation_mode="area")


def test_sinusoidal_positional_embedding_has_no_parameters_and_any_grid():
    module = SinusoidalPositionalEmbedding2D(embed_dim=8)

    assert list(module.parameters()) == []
    small = module.embeddings((2, 3))
    large = module.embeddings((4, 5))
    assert small.shape == (1, 6, 8)
    assert large.shape == (1, 20, 8)
    # Positions are distinct and the first row/column pattern is stable.
    assert torch.unique(small[0], dim=0).shape[0] == 6
    torch.testing.assert_close(small[0, 0], large[0, 0])
    assert module(torch.zeros(2, 6, 8), (2, 3)).shape == (2, 6, 8)


def test_sinusoidal_positional_embedding_validates_arguments():
    with pytest.raises(ValueError, match="divisible by 4"):
        SinusoidalPositionalEmbedding2D(embed_dim=6)
    with pytest.raises(ValueError, match="temperature"):
        SinusoidalPositionalEmbedding2D(embed_dim=8, temperature=1)


# -- encoder / pooling / queries ----------------------------------------


def test_transformer_encoder_encodes_tokens_and_honours_padding_mask():
    torch.manual_seed(0)
    module = TransformerEncoder(
        embed_dim=8,
        num_heads=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        final_norm=True,
    ).eval()
    tokens = torch.randn(1, 4, 8)
    mask = torch.tensor([[False, False, False, True]])

    output = module(tokens, key_padding_mask=mask)
    changed_padding = tokens.clone()
    changed_padding[0, 3] = 100.0

    assert output.shape == (1, 4, 8)
    torch.testing.assert_close(
        module(changed_padding, key_padding_mask=mask)[0, :3],
        output[0, :3],
    )
    assert isinstance(module.encoder.norm, nn.LayerNorm)


def test_transformer_encoder_validates_arguments():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        TransformerEncoder(embed_dim=8, num_heads=3, num_layers=1)
    with pytest.raises(ValueError, match="activation"):
        TransformerEncoder(embed_dim=8, num_heads=2, num_layers=1, activation="tanh")


def test_attention_pooling_pools_to_query_count():
    module = AttentionPooling(embed_dim=8, num_heads=2)
    assert module(torch.randn(3, 10, 8), torch.randn(3, 2, 8)).shape == (3, 2, 8)


def test_attention_pooling_exposes_weights_to_forward_hooks_when_requested():
    module = AttentionPooling(embed_dim=8, num_heads=2, need_weights=True)
    captured = []
    module.attention.register_forward_hook(lambda m, args, output: captured.append(output))

    module(torch.randn(3, 10, 8), torch.randn(3, 1, 8))

    assert captured[0][1].shape == (3, 1, 10)


def test_learned_query_expands_over_batch_and_rejects_conditioning():
    module = LearnedQuery(embed_dim=8, num_queries=3)
    assert module(4).shape == (4, 3, 8)
    with pytest.raises(TypeError, match="takes no conditioning"):
        module(4, obj_patch=torch.zeros(4, 3, 4, 4))


def _object_query(**overrides):
    kwargs = {
        "embed_dim": 8,
        "inputs": {
            "obj_patch": {"type": "patch", "in_channels": 3, "patch_size": 4},
            "obj_patch_location": {"type": "linear", "in_features": 2},
        },
        "hidden_dims": [16],
    }
    kwargs.update(overrides)
    return ConditionedQuery(**kwargs)


def test_conditioned_query_encodes_any_crop_size_into_one_query():
    module = _object_query()

    # 8x8 crops give 4 patch tokens; they are reduced to one vector.
    query = module(2, obj_patch=torch.randn(2, 3, 8, 8), obj_patch_location=torch.randn(2, 2))

    assert query.shape == (2, 1, 8)
    assert module.input_names == ("obj_patch", "obj_patch_location")


def test_conditioned_query_mlp_is_non_linear():
    module = _object_query(hidden_dims=[16, 16], activation="relu")
    layer_types = [type(layer) for layer in module.projection]
    assert layer_types == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert module.projection[0].in_features == 16  # two encoded inputs


def test_conditioned_query_rejects_missing_extra_or_mis_batched_inputs():
    module = _object_query()
    patch = torch.randn(2, 3, 4, 4)
    with pytest.raises(TypeError, match="expects inputs"):
        module(2, obj_patch=patch)
    with pytest.raises(TypeError, match="expects inputs"):
        module(2, obj_patch=patch, obj_patch_location=torch.randn(2, 2), extra=patch)
    with pytest.raises(ValueError, match="batch size 2, expected 3"):
        module(3, obj_patch=patch, obj_patch_location=torch.randn(2, 2))


@pytest.mark.parametrize(
    "inputs, match",
    [
        ({}, "non-empty mapping"),
        ({"bad name": {"type": "linear", "in_features": 2}}, "identifiers"),
        ({"x": {"type": "conv"}}, "type must be one of"),
        ({"x": {"type": "linear"}}, "missing a required key"),
        ({"x": {"type": "linear", "in_features": 2, "extra": 1}}, "unknown keys"),
        ({"x": {"type": "patch", "in_channels": 3, "patch_size": 4, "reduce": "sum"}}, "reduce"),
    ],
)
def test_conditioned_query_validates_input_specs(inputs, match):
    with pytest.raises(ValueError, match=match):
        ConditionedQuery(embed_dim=8, inputs=inputs)
