"""Generic transformer building blocks as plain PyTorch modules.

Nothing here depends on the component framework, so every module can be used
on its own. `components.py` exposes them as pluggable resources.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _non_negative_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number; got {value!r}")
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


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class PatchEmbedding(nn.Module):
    """Split images into non-overlapping patches and project each to a token.

    Input `(B, C, H, W)`, output `(B, N, embed_dim)` with
    `N = (H / patch_h) * (W / patch_w)`.
    """

    def __init__(
            self,
            in_channels: int,
            patch_size: int | Sequence[int],
            embed_dim: int,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self._in_channels = _positive_int(in_channels, "in_channels")
        self._patch_size = _pair(patch_size, "patch_size")
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        self.projection = nn.Conv2d(
            self._in_channels,
            self._embed_dim,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            bias=bool(bias),
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._patch_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        """Number of patches along each spatial dimension for an input size."""
        patch_h, patch_w = self._patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Image dimensions ({height}, {width}) must be divisible by "
                f"patch size {self._patch_size}."
            )
        return height // patch_h, width // patch_w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != self._in_channels:
            raise ValueError(
                f"Input has {images.shape[1]} channels but expected "
                f"{self._in_channels}"
            )
        self.grid_size(images.shape[2], images.shape[3])

        tokens = self.projection(images)  # (B, D, H/ph, W/pw)
        return tokens.flatten(2).transpose(1, 2)  # (B, N, D)


class LearnedPositionalEmbedding2D(nn.Module):
    """Add a learned positional table over a 2D token grid.

    The table is trained for `grid_size`; other grid sizes are served by
    resizing the table with `interpolation_mode`.
    """

    _INTERPOLATION_MODES = ("nearest", "bilinear", "bicubic")

    def __init__(
            self,
            grid_size: int | Sequence[int],
            embed_dim: int,
            init: str = "zeros",
            init_std: float = 0.02,
            interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self._grid_size = _pair(grid_size, "grid_size")
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        _choice(init, "init", ("zeros", "trunc_normal"))
        init_std = _non_negative_float(init_std, "init_std")
        self._interpolation_mode = _choice(
            interpolation_mode,
            "interpolation_mode",
            self._INTERPOLATION_MODES,
        )

        n_tokens = self._grid_size[0] * self._grid_size[1]
        self.table = nn.Parameter(torch.zeros(1, n_tokens, self._embed_dim))
        if init == "trunc_normal":
            nn.init.trunc_normal_(self.table, std=init_std)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embeddings(self, grid_size: Sequence[int]) -> torch.Tensor:
        """The `(1, h * w, embed_dim)` table resized to `grid_size`."""
        grid_h, grid_w = _pair(tuple(grid_size), "grid_size")
        if (grid_h, grid_w) == self._grid_size:
            return self.table

        table = self.table.transpose(1, 2).reshape(
            1, self._embed_dim, *self._grid_size
        )
        align_corners = None if self._interpolation_mode == "nearest" else False
        table = functional.interpolate(
            table,
            size=(grid_h, grid_w),
            mode=self._interpolation_mode,
            align_corners=align_corners,
        )
        return table.flatten(2).transpose(1, 2)

    def forward(self, tokens: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
        return tokens + self.embeddings(grid_size)


class SinusoidalPositionalEmbedding2D(nn.Module):
    """Add fixed 2D sine-cosine positional embeddings (no parameters).

    Half of `embed_dim` encodes the row and half the column, so `embed_dim`
    must be divisible by 4. Works for any grid size.
    """

    def __init__(self, embed_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        if self._embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4; got {self._embed_dim}"
            )
        if _non_negative_float(temperature, "temperature") <= 1:
            raise ValueError(f"temperature must be greater than 1; got {temperature!r}")
        self._temperature = float(temperature)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embeddings(
            self,
            grid_size: Sequence[int],
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The `(1, h * w, embed_dim)` embeddings for `grid_size`."""
        grid_h, grid_w = _pair(tuple(grid_size), "grid_size")
        quarter = self._embed_dim // 4
        omega = torch.arange(quarter, device=device, dtype=torch.float64) / quarter
        omega = 1.0 / (self._temperature ** omega)

        rows = torch.arange(grid_h, device=device, dtype=torch.float64)
        cols = torch.arange(grid_w, device=device, dtype=torch.float64)
        rows, cols = torch.meshgrid(rows, cols, indexing="ij")
        row_angles = rows.reshape(-1, 1) * omega
        col_angles = cols.reshape(-1, 1) * omega
        embeddings = torch.cat(
            [row_angles.sin(), row_angles.cos(), col_angles.sin(), col_angles.cos()],
            dim=1,
        )
        return embeddings.unsqueeze(0).to(dtype or torch.get_default_dtype())

    def forward(self, tokens: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
        return tokens + self.embeddings(
            grid_size,
            device=tokens.device,
            dtype=tokens.dtype,
        )


class TransformerEncoder(nn.Module):
    """A stack of `nn.TransformerEncoderLayer`s over `(B, N, embed_dim)` tokens.

    Defaults follow `nn.TransformerEncoderLayer`.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
            norm_first: bool = False,
            layer_norm_eps: float = 1e-5,
            final_norm: bool = False,
    ) -> None:
        super().__init__()
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        num_heads = _positive_int(num_heads, "num_heads")
        if self._embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({self._embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=self._embed_dim,
            nhead=num_heads,
            dim_feedforward=_positive_int(dim_feedforward, "dim_feedforward"),
            dropout=_non_negative_float(dropout, "dropout"),
            activation=_choice(activation, "activation", ("relu", "gelu")),
            layer_norm_eps=float(layer_norm_eps),
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=_positive_int(num_layers, "num_layers"),
            norm=nn.LayerNorm(self._embed_dim, eps=float(layer_norm_eps)) if final_norm else None,
            enable_nested_tensor=False,
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
            self,
            tokens: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=key_padding_mask)


class AttentionPooling(nn.Module):
    """Pool `(B, N, D)` tokens into `(B, Q, D)` with multi-head cross-attention.

    The `(B, Q, D)` query is supplied by the caller. With `need_weights`, the
    inner `attention` module also returns attention weights, which
    `layer_inspector` can capture.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            need_weights: bool = False,
            average_attn_weights: bool = True,
    ) -> None:
        super().__init__()
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        num_heads = _positive_int(num_heads, "num_heads")
        if self._embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({self._embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self._need_weights = bool(need_weights)
        self._average_attn_weights = bool(average_attn_weights)
        self.attention = nn.MultiheadAttention(
            self._embed_dim,
            num_heads,
            dropout=_non_negative_float(dropout, "dropout"),
            bias=bool(bias),
            batch_first=True,
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(
            self,
            tokens: torch.Tensor,
            query: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled, _ = self.attention(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=self._need_weights,
            average_attn_weights=self._average_attn_weights,
        )
        return pooled


class LearnedQuery(nn.Module):
    """`num_queries` learned query vectors shared across the batch."""

    def __init__(
            self,
            embed_dim: int,
            num_queries: int = 1,
            init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        num_queries = _positive_int(num_queries, "num_queries")
        self.queries = nn.Parameter(torch.zeros(1, num_queries, self._embed_dim))
        nn.init.trunc_normal_(self.queries, std=_non_negative_float(init_std, "init_std"))

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, batch_size: int, **conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning:
            raise TypeError(
                "LearnedQuery takes no conditioning inputs; got "
                f"{sorted(conditioning)}"
            )
        return self.queries.expand(batch_size, -1, -1)


class ConditionedQuery(nn.Module):
    """Build one query per sample from named conditioning inputs.

    `inputs` maps each input name to an encoder spec:
    - `{"type": "patch", "in_channels", "patch_size", "reduce": "mean"|"max"}`
      encodes a `(B, C, H, W)` image crop and reduces its patch tokens.
    - `{"type": "linear", "in_features"}` encodes a `(B, in_features)` vector.

    The encodings are concatenated in `inputs` order and passed through an MLP
    with `hidden_dims` hidden layers. Output shape is `(B, 1, embed_dim)`.
    """

    _INPUT_TYPES = ("patch", "linear")

    def __init__(
            self,
            embed_dim: int,
            inputs: Mapping[str, Mapping],
            hidden_dims: Sequence[int] = (),
            activation: str = "gelu",
    ) -> None:
        super().__init__()
        self._embed_dim = _positive_int(embed_dim, "embed_dim")
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError("inputs must be a non-empty mapping of input name to encoder spec")
        if isinstance(hidden_dims, (str, bytes)) or not isinstance(hidden_dims, Sequence):
            raise ValueError(f"hidden_dims must be a list of positive integers; got {hidden_dims!r}")
        hidden_dims = [
            _positive_int(dim, f"hidden_dims[{index}]")
            for index, dim in enumerate(hidden_dims)
        ]
        make_activation = _ACTIVATIONS[
            _choice(activation, "activation", tuple(_ACTIVATIONS))
        ]

        self.encoders = nn.ModuleDict()
        self._reductions: dict[str, str] = {}
        for name, spec in inputs.items():
            self.encoders[self._input_name(name)] = self._build_encoder(name, spec)

        dims = [len(self.encoders) * self._embed_dim, *hidden_dims, self._embed_dim]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            if index > 0:
                layers.append(make_activation())
            layers.append(nn.Linear(in_dim, out_dim))
        self.projection = nn.Sequential(*layers)

    @staticmethod
    def _input_name(name) -> str:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"input names must be Python identifiers; got {name!r}")
        return name

    def _build_encoder(self, name: str, spec) -> nn.Module:
        if not isinstance(spec, Mapping):
            raise ValueError(f"inputs.{name} must be a mapping; got {spec!r}")
        spec = dict(spec)
        input_type = _choice(spec.pop("type", None), f"inputs.{name}.type", self._INPUT_TYPES)
        if input_type == "patch":
            self._reductions[name] = _choice(
                spec.pop("reduce", "mean"),
                f"inputs.{name}.reduce",
                ("mean", "max"),
            )
            allowed = {"in_channels", "patch_size", "bias"}
        else:
            allowed = {"in_features", "bias"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"inputs.{name} has unknown keys: {sorted(unknown)}")
        try:
            if input_type == "patch":
                return PatchEmbedding(embed_dim=self._embed_dim, **spec)
            return nn.Linear(
                _positive_int(spec["in_features"], f"inputs.{name}.in_features"),
                self._embed_dim,
                bias=bool(spec.get("bias", True)),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"inputs.{name} is missing a required key: {error}") from error

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self.encoders.keys())

    def forward(self, batch_size: int, **conditioning: torch.Tensor) -> torch.Tensor:
        expected = set(self.encoders.keys())
        if set(conditioning) != expected:
            raise TypeError(
                f"ConditionedQuery expects inputs {sorted(expected)}; got "
                f"{sorted(conditioning)}"
            )

        encoded = []
        for name, encoder in self.encoders.items():
            value = conditioning[name]
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Conditioning input '{name}' has batch size "
                    f"{value.shape[0]}, expected {batch_size}"
                )
            features = encoder(value)
            if name in self._reductions:
                features = (
                    features.mean(dim=1)
                    if self._reductions[name] == "mean"
                    else features.amax(dim=1)
                )
            encoded.append(features)

        return self.projection(torch.cat(encoded, dim=1)).unsqueeze(1)


__all__ = [
    "AttentionPooling",
    "ConditionedQuery",
    "LearnedPositionalEmbedding2D",
    "LearnedQuery",
    "PatchEmbedding",
    "SinusoidalPositionalEmbedding2D",
    "TransformerEncoder",
]
