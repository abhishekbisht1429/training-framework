from __future__ import annotations

import pytest
import torch
from torch import nn

from training_framework.components.builtin.transformer import (
    AttentionPooling,
    ClassToken,
    ConditionedQuery,
    LearnedPositionalEmbedding2D,
    LearnedQuery,
    PatchEmbedding,
    SinusoidalPositionalEmbedding2D,
    TokenReduction,
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


# -- class token --------------------------------------------------------


def test_class_token_prepends_one_shared_learned_token():
    module = ClassToken(embed_dim=4)
    tokens = torch.randn(3, 5, 4)

    output, mask = module(tokens)

    assert output.shape == (3, 6, 4)
    assert mask is None
    torch.testing.assert_close(output[:, 1:], tokens)
    torch.testing.assert_close(output[:, 0], module.token[0].expand(3, -1))
    assert torch.count_nonzero(module.token) > 0


def test_class_token_widens_mask_so_it_is_never_masked():
    module = ClassToken(embed_dim=4)
    mask = torch.tensor([[False, True], [True, True]])

    _, widened = module(torch.randn(2, 2, 4), mask)

    torch.testing.assert_close(
        widened,
        torch.tensor([[False, False, True], [False, True, True]]),
    )


def test_class_token_validates_arguments():
    with pytest.raises(ValueError, match="embed_dim"):
        ClassToken(embed_dim=0)
    with pytest.raises(ValueError, match="init_std"):
        ClassToken(embed_dim=4, init_std=-0.1)


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
        "encoders": {
            "obj_patch": TokenReduction(PatchEmbedding(in_channels=3, patch_size=4, embed_dim=8)),
            "obj_patch_location": nn.Linear(2, 8),
        },
        "hidden_dims": [16],
    }
    kwargs.update(overrides)
    return ConditionedQuery(**kwargs)


def test_token_reduction_turns_tokens_into_one_vector_per_sample():
    tokens = PatchEmbedding(in_channels=3, patch_size=4, embed_dim=8)
    images = torch.randn(2, 3, 8, 8)

    mean = TokenReduction(tokens, reduce="mean")
    maximum = TokenReduction(tokens, reduce="max")

    assert mean(images).shape == (2, 8)
    torch.testing.assert_close(mean(images), tokens(images).mean(dim=1))
    torch.testing.assert_close(maximum(images), tokens(images).amax(dim=1))
    assert mean.embed_dim == 8  # forwarded from the wrapped module


def test_token_reduction_validates_its_module_and_output():
    with pytest.raises(TypeError, match="expects an nn.Module"):
        TokenReduction("not a module")
    with pytest.raises(ValueError, match="reduce must be one of"):
        TokenReduction(nn.Identity(), reduce="sum")
    with pytest.raises(ValueError, match=r"expects \(B, N, D\) tokens"):
        TokenReduction(nn.Linear(4, 4))(torch.randn(2, 4))


def test_conditioned_query_encodes_any_crop_size_into_one_query():
    module = _object_query()

    # 8x8 crops give 4 patch tokens; they are reduced to one vector.
    query = module(2, obj_patch=torch.randn(2, 3, 8, 8), obj_patch_location=torch.randn(2, 2))

    assert query.shape == (2, 1, 8)
    assert module.input_names == ("obj_patch", "obj_patch_location")


def test_conditioned_query_accepts_any_module_meeting_the_output_contract():
    class Constant(nn.Module):
        def forward(self, value):
            return value.new_zeros(value.shape[0], 8)

    module = ConditionedQuery(embed_dim=8, encoders={"anything": Constant()})

    assert module(3, anything=torch.randn(3, 5)).shape == (3, 1, 8)


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


def test_conditioned_query_rejects_encoders_with_the_wrong_output_size():
    with pytest.raises(ValueError, match="out_features=4, but must produce 8"):
        ConditionedQuery(embed_dim=8, encoders={"x": nn.Linear(2, 4)})
    with pytest.raises(ValueError, match="embed_dim=4, but must produce 8"):
        ConditionedQuery(
            embed_dim=8,
            encoders={"x": TokenReduction(PatchEmbedding(3, 4, 4))},
        )

    # A module that declares nothing is checked on its first forward pass.
    undeclared = ConditionedQuery(embed_dim=8, encoders={"x": nn.Sequential(nn.Linear(2, 4))})
    with pytest.raises(ValueError, match=r"must return \(3, 8\) features; got \(3, 4\)"):
        undeclared(3, x=torch.randn(3, 2))


@pytest.mark.parametrize(
    "encoders, error, match",
    [
        ({}, ValueError, "non-empty mapping"),
        ({"bad name": nn.Linear(2, 8)}, ValueError, "identifiers"),
        ({"x": "not a module"}, TypeError, "must be an nn.Module"),
    ],
)
def test_conditioned_query_validates_encoders(encoders, error, match):
    with pytest.raises(error, match=match):
        ConditionedQuery(embed_dim=8, encoders=encoders)


def test_conditioned_query_validates_hidden_dims_and_activation():
    with pytest.raises(ValueError, match="hidden_dims"):
        _object_query(hidden_dims=8)
    with pytest.raises(ValueError, match="activation must be one of"):
        _object_query(activation="swish")
