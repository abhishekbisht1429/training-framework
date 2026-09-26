"""Tests for the plain modules the transformer blocks share.

The blocks themselves are components -- see `test_components.py`. What is
tested here is what genuinely is not a component: `ClassToken`, owned by the
composite that prepends it, and `TokenReduction`, which wraps a user-supplied
encoder.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from training_framework.components.builtin.transformer import (
    ClassToken,
    ConvPatchEmbedding,
    TokenReduction,
)


def _patch_embedding(embed_dim=8):
    return ConvPatchEmbedding({
        "in_channels": 3,
        "patch_size": 4,
        "embed_dim": embed_dim,
    })


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_class_token_rejects_non_finite_init_std(value):
    with pytest.raises(ValueError, match="finite non-negative number"):
        ClassToken(embed_dim=4, init_std=value)


# -- token reduction ----------------------------------------------------


def test_token_reduction_turns_tokens_into_one_vector_per_sample():
    tokens = _patch_embedding()
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
