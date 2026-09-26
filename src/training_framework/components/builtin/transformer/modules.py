"""Shared pieces of the transformer blocks, as plain PyTorch modules.

The blocks themselves are components -- see `components.py`, where each one
creates its own weights. What is left here is what genuinely is not a
component: `ClassToken`, whose weights belong to the composite that prepends
it, `TokenReduction`, which wraps a *user-supplied* encoder, and the
configuration validators the blocks share.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _non_negative_float(value, name: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
    ):
        raise ValueError(
            f"{name} must be a finite non-negative number; got {value!r}"
        )
    return float(value)


def _positive_float(value, name: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number; got {value!r}")
    return float(value)


def _probability(value, name: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a number between 0 and 1; got {value!r}")
    return float(value)


def _pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        pair = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        pair = tuple(value)
    else:
        raise ValueError(
            f"{name} must be a positive integer or a pair of positive "
            f"integers; got {value!r}"
        )
    return (
        _positive_int(pair[0], f"{name}[0]"),
        _positive_int(pair[1], f"{name}[1]"),
    )


def _choice(value, name: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}; got {value!r}")
    return value


def _check_token_shape(
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        embed_dim: int,
) -> None:
    """Reject token shapes that would broadcast against `embeddings` instead
    of lining up with them, e.g. a singleton token or feature dimension."""
    expected = (embeddings.shape[1], embed_dim)
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
        raise ValueError(
            f"Expected (B, {expected[0]}, {expected[1]}) tokens for this grid "
            f"size, got {tuple(tokens.shape)}"
        )


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class ClassToken(nn.Module):
    """Prepend one learned token to a `(B, N, embed_dim)` sequence.

    Returns `(B, 1 + N, embed_dim)` tokens and, when a `(B, N)` key padding
    mask is given, the mask widened so the class token is never masked.
    """

    def __init__(self, embed_dim: int, init_std: float = 0.02) -> None:
        super().__init__()
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        self.token = nn.Parameter(torch.zeros(1, 1, self._embed_dim))
        nn.init.trunc_normal_(self.token, std=_non_negative_float(init_std, "init_std"))

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
            self,
            tokens: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = tokens.shape[0]
        tokens = torch.cat([self.token.expand(batch_size, -1, -1), tokens], dim=1)
        if key_padding_mask is not None:
            unmasked = torch.zeros(
                batch_size, 1,
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            key_padding_mask = torch.cat([unmasked, key_padding_mask], dim=1)
        return tokens, key_padding_mask


class TokenReduction(nn.Module):
    """Reduce a wrapped module's `(B, N, D)` tokens to one `(B, D)` vector.

    Lets any token-producing encoder (a patch embedding, a small transformer,
    your own module) satisfy an interface that expects one vector per sample.
    """

    _REDUCTIONS = ("mean", "max")

    def __init__(self, module: nn.Module, reduce: str = "mean") -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError(f"TokenReduction expects an nn.Module; got {module!r}")
        self.module = module
        self._reduce = _choice(reduce, "reduce", self._REDUCTIONS)

    @property
    def reduce(self) -> str:
        return self._reduce

    @property
    def embed_dim(self) -> int | None:
        return getattr(self.module, "embed_dim", None)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        tokens = self.module(*args, **kwargs)
        if tokens.ndim != 3:
            raise ValueError(
                f"TokenReduction expects (B, N, D) tokens, got {tuple(tokens.shape)}"
            )
        return tokens.mean(dim=1) if self._reduce == "mean" else tokens.amax(dim=1)


# Names nn.ModuleDict already uses; a conditioning input cannot take one.
_MODULE_DICT_ATTRIBUTES = frozenset(dir(nn.ModuleDict()))



__all__ = [
    "ClassToken",
    "TokenReduction",
]
