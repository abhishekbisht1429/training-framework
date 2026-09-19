"""Pluggable transformer blocks, as components.

Each block is a `ModuleResource` that creates its own weights in `__init__`,
like any other PyTorch module, and declares its configuration with a
`config_schema`. The composite models (`patch_transformer`,
`pooled_patch_transformer`) depend on block *roles* and attach whatever is
bound to them during construction, so they are wiring plus a forward pass.
`component_bindings` chooses which implementation fills each role.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch
from torch import nn
from torch.nn import functional

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ModuleResource,
    Resource,
    requires_resource,
    resource,
    role,
)
from training_framework.components.builtin.transformer.modules import (
    _ACTIVATIONS,
    _MODULE_DICT_ATTRIBUTES,
    _check_token_shape,
    _choice,
    _non_negative_float,
    _pair,
    _positive_float,
    _positive_int,
    _probability,
    ClassToken,
    TokenReduction,
)

if TYPE_CHECKING:
    from training_framework.session import Session


# -- role contracts -----------------------------------------------------
#
# Protocols rather than base classes: they document what a role promises and
# are visible to a type checker, while costing nothing at runtime and leaving
# implementations free to inherit whatever they like.


class SizedBlock(Protocol):
    """Every block declares the token width it works in."""

    @property
    def embed_dim(self) -> int: ...


class PatchEmbeddingBlock(SizedBlock, Protocol):
    def grid_size(self, height: int, width: int) -> tuple[int, int]: ...

    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...


class PositionalEmbeddingBlock(SizedBlock, Protocol):
    def __call__(
            self,
            tokens: torch.Tensor,
            grid_size: Sequence[int],
    ) -> torch.Tensor: ...


class SequenceEncoderBlock(SizedBlock, Protocol):
    def __call__(
            self,
            tokens: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class PoolingBlock(SizedBlock, Protocol):
    def __call__(
            self,
            tokens: torch.Tensor,
            query: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class PoolingQueryBlock(SizedBlock, Protocol):
    def __call__(
            self,
            batch_size: int,
            **conditioning: torch.Tensor,
    ) -> torch.Tensor: ...


role(
    "patch_embedding",
    Resource,
    description=(
        "a ModuleResource mapping (B, C, H, W) images to (B, N, embed_dim) "
        "tokens and providing grid_size(height, width)"
    ),
)
role(
    "positional_embedding",
    Resource,
    description=(
        "a ModuleResource called as module(tokens, grid_size) that returns "
        "tokens with positions added"
    ),
)
role(
    "sequence_encoder",
    Resource,
    description=(
        "a ModuleResource called as module(tokens, key_padding_mask=None) "
        "that returns encoded tokens"
    ),
)
role(
    "pooling",
    Resource,
    description=(
        "a ModuleResource called as module(tokens, query, "
        "key_padding_mask=None) that returns (B, Q, embed_dim)"
    ),
)
role(
    "pooling_query",
    Resource,
    description=(
        "a ModuleResource called as module(batch_size, **conditioning) that "
        "returns a (B, Q, embed_dim) query"
    ),
)


# -- patch embedding ----------------------------------------------------


@dataclass
class ConvPatchEmbeddingConfig:
    in_channels: int
    patch_size: Any
    embed_dim: int
    bias: bool = True

    def __post_init__(self):
        self.in_channels = _positive_int(self.in_channels, "in_channels")
        self.patch_size = _pair(self.patch_size, "patch_size")
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        self.bias = bool(self.bias)


@resource("conv_patch_embedding")
class ConvPatchEmbedding(ModuleResource):
    """Split images into non-overlapping patches and project each to a token.

    Input `(B, C, H, W)`, output `(B, N, embed_dim)` with
    `N = (H / patch_h) * (W / patch_w)`.
    """

    config_schema = ConvPatchEmbeddingConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self.projection = nn.Conv2d(
            self._cfg.in_channels,
            self._cfg.embed_dim,
            kernel_size=self._cfg.patch_size,
            stride=self._cfg.patch_size,
            bias=self._cfg.bias,
        )

    @property
    def in_channels(self) -> int:
        return self._cfg.in_channels

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._cfg.patch_size

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        """Number of patches along each spatial dimension for an input size."""
        patch_h, patch_w = self._cfg.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Image dimensions ({height}, {width}) must be divisible by "
                f"patch size {self._cfg.patch_size}."
            )
        return height // patch_h, width // patch_w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != self._cfg.in_channels:
            raise ValueError(
                f"Input has {images.shape[1]} channels but expected "
                f"{self._cfg.in_channels}"
            )
        self.grid_size(images.shape[2], images.shape[3])

        tokens = self.projection(images)  # (B, D, H/ph, W/pw)
        return tokens.flatten(2).transpose(1, 2)  # (B, N, D)


# -- positional embeddings ----------------------------------------------


_INTERPOLATION_MODES = ("nearest", "bilinear", "bicubic")


@dataclass
class LearnedPositionalEmbedding2DConfig:
    grid_size: Any
    embed_dim: int
    init: str = "zeros"
    init_std: float = 0.02
    interpolation_mode: str = "bilinear"

    def __post_init__(self):
        self.grid_size = _pair(self.grid_size, "grid_size")
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        self.init = _choice(self.init, "init", ("zeros", "trunc_normal"))
        self.init_std = _non_negative_float(self.init_std, "init_std")
        self.interpolation_mode = _choice(
            self.interpolation_mode,
            "interpolation_mode",
            _INTERPOLATION_MODES,
        )


@resource("learned_positional_embedding_2d")
class LearnedPositionalEmbedding2D(ModuleResource):
    """Add a learned positional table over a 2D token grid.

    The table is trained for `grid_size`; other grid sizes are served by
    resizing the table with `interpolation_mode`.
    """

    config_schema = LearnedPositionalEmbedding2DConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        grid_h, grid_w = self._cfg.grid_size
        self.table = nn.Parameter(
            torch.zeros(1, grid_h * grid_w, self._cfg.embed_dim)
        )
        if self._cfg.init == "trunc_normal":
            nn.init.trunc_normal_(self.table, std=self._cfg.init_std)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._cfg.grid_size

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

    def embeddings(self, grid_size: Sequence[int]) -> torch.Tensor:
        """The `(1, h * w, embed_dim)` table resized to `grid_size`."""
        grid_h, grid_w = _pair(tuple(grid_size), "grid_size")
        if (grid_h, grid_w) == self._cfg.grid_size:
            return self.table

        table = self.table.transpose(1, 2).reshape(
            1, self._cfg.embed_dim, *self._cfg.grid_size
        )
        align_corners = (
            None if self._cfg.interpolation_mode == "nearest" else False
        )
        table = functional.interpolate(
            table,
            size=(grid_h, grid_w),
            mode=self._cfg.interpolation_mode,
            align_corners=align_corners,
        )
        return table.flatten(2).transpose(1, 2)

    def forward(
            self,
            tokens: torch.Tensor,
            grid_size: Sequence[int],
    ) -> torch.Tensor:
        embeddings = self.embeddings(grid_size)
        _check_token_shape(tokens, embeddings, self._cfg.embed_dim)
        return tokens + embeddings


@dataclass
class SinusoidalPositionalEmbedding2DConfig:
    embed_dim: int
    temperature: float = 10000.0

    def __post_init__(self):
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        if self.embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim must be divisible by 4; got {self.embed_dim}"
            )
        if _non_negative_float(self.temperature, "temperature") <= 1:
            raise ValueError(
                f"temperature must be greater than 1; got {self.temperature!r}"
            )
        self.temperature = float(self.temperature)


@resource("sinusoidal_positional_embedding_2d")
class SinusoidalPositionalEmbedding2D(ModuleResource):
    """Add fixed 2D sine-cosine positional embeddings (no parameters).

    Half of `embed_dim` encodes the row and half the column, so `embed_dim`
    must be divisible by 4. Works for any grid size.
    """

    config_schema = SinusoidalPositionalEmbedding2DConfig

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

    def embeddings(
            self,
            grid_size: Sequence[int],
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """The `(1, h * w, embed_dim)` embeddings for `grid_size`."""
        grid_h, grid_w = _pair(tuple(grid_size), "grid_size")
        quarter = self._cfg.embed_dim // 4
        omega = torch.arange(quarter, device=device, dtype=torch.float64) / quarter
        omega = 1.0 / (self._cfg.temperature ** omega)

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

    def forward(
            self,
            tokens: torch.Tensor,
            grid_size: Sequence[int],
    ) -> torch.Tensor:
        embeddings = self.embeddings(
            grid_size,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        _check_token_shape(tokens, embeddings, self._cfg.embed_dim)
        return tokens + embeddings


# -- sequence encoder ---------------------------------------------------


@dataclass
class TorchTransformerEncoderConfig:
    embed_dim: int
    num_heads: int
    num_layers: int
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    norm_first: bool = False
    layer_norm_eps: float = 1e-5
    final_norm: bool = False

    def __post_init__(self):
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        self.num_heads = _positive_int(self.num_heads, "num_heads")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        self.num_layers = _positive_int(self.num_layers, "num_layers")
        self.dim_feedforward = _positive_int(
            self.dim_feedforward, "dim_feedforward",
        )
        self.dropout = _probability(self.dropout, "dropout")
        self.activation = _choice(self.activation, "activation", ("relu", "gelu"))
        self.norm_first = bool(self.norm_first)
        self.layer_norm_eps = _positive_float(self.layer_norm_eps, "layer_norm_eps")
        self.final_norm = bool(self.final_norm)


@resource("torch_transformer_encoder")
class TorchTransformerEncoder(ModuleResource):
    """A stack of `nn.TransformerEncoderLayer`s over `(B, N, embed_dim)` tokens.

    Defaults follow `nn.TransformerEncoderLayer`.
    """

    config_schema = TorchTransformerEncoderConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        layer = nn.TransformerEncoderLayer(
            d_model=self._cfg.embed_dim,
            nhead=self._cfg.num_heads,
            dim_feedforward=self._cfg.dim_feedforward,
            dropout=self._cfg.dropout,
            activation=self._cfg.activation,
            layer_norm_eps=self._cfg.layer_norm_eps,
            batch_first=True,
            norm_first=self._cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self._cfg.num_layers,
            norm=(
                nn.LayerNorm(self._cfg.embed_dim, eps=self._cfg.layer_norm_eps)
                if self._cfg.final_norm else None
            ),
            enable_nested_tensor=False,
        )

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

    def forward(
            self,
            tokens: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(tokens, src_key_padding_mask=key_padding_mask)


# -- pooling ------------------------------------------------------------


@dataclass
class AttentionPoolingConfig:
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    need_weights: bool = False
    average_attn_weights: bool = True

    def __post_init__(self):
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        self.num_heads = _positive_int(self.num_heads, "num_heads")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        self.dropout = _probability(self.dropout, "dropout")
        self.bias = bool(self.bias)
        self.need_weights = bool(self.need_weights)
        self.average_attn_weights = bool(self.average_attn_weights)


@resource("attention_pooling")
class AttentionPooling(ModuleResource):
    """Pool `(B, N, D)` tokens into `(B, Q, D)` with multi-head cross-attention.

    The `(B, Q, D)` query is supplied by the caller. With `need_weights`, the
    inner `attention` module also returns attention weights, which
    `layer_inspector` can capture.
    """

    config_schema = AttentionPoolingConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self.attention = nn.MultiheadAttention(
            self._cfg.embed_dim,
            self._cfg.num_heads,
            dropout=self._cfg.dropout,
            bias=self._cfg.bias,
            batch_first=True,
        )

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

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
            need_weights=self._cfg.need_weights,
            average_attn_weights=self._cfg.average_attn_weights,
        )
        return pooled


# -- pooling queries ----------------------------------------------------


@dataclass
class LearnedPoolingQueryConfig:
    embed_dim: int
    num_queries: int = 1
    init_std: float = 0.02

    def __post_init__(self):
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        self.num_queries = _positive_int(self.num_queries, "num_queries")
        self.init_std = _non_negative_float(self.init_std, "init_std")


@resource("learned_pooling_query")
class LearnedPoolingQuery(ModuleResource):
    """`num_queries` learned query vectors shared across the batch."""

    config_schema = LearnedPoolingQueryConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self.queries = nn.Parameter(
            torch.zeros(1, self._cfg.num_queries, self._cfg.embed_dim)
        )
        nn.init.trunc_normal_(self.queries, std=self._cfg.init_std)

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

    def forward(
            self,
            batch_size: int,
            **conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if conditioning:
            raise TypeError(
                "learned_pooling_query takes no conditioning inputs; got "
                f"{sorted(conditioning)}"
            )
        return self.queries.expand(batch_size, -1, -1)


def _resolve_module_class(dotted_path: str, context: str) -> type[nn.Module]:
    """Import an `nn.Module` subclass from a fully-qualified dotted path.

    `layer_inspector.module_types` resolves dotted paths the same way; the two
    could share a helper if a third caller appears.
    """
    if not isinstance(dotted_path, str) or "." not in dotted_path:
        raise ValueError(
            f"{context} must be a fully-qualified dotted path (e.g. "
            f"'torch.nn.Linear'); got {dotted_path!r}"
        )
    module_path, _, attr_name = dotted_path.rpartition(".")
    try:
        imported = importlib.import_module(module_path)
    except ImportError as error:
        raise ImportError(
            f"{context} {dotted_path!r} could not be imported: {error}"
        ) from error
    try:
        resolved = getattr(imported, attr_name)
    except AttributeError as error:
        raise ValueError(
            f"{context} {dotted_path!r} has no attribute {attr_name!r} in "
            f"module {module_path!r}"
        ) from error
    if not isinstance(resolved, type) or not issubclass(resolved, nn.Module):
        raise TypeError(
            f"{context} {dotted_path!r} does not resolve to an nn.Module subclass"
        )
    return resolved


@dataclass
class ConditionedPoolingQueryConfig:
    embed_dim: int
    inputs: Any
    hidden_dims: tuple = ()
    activation: Any = "gelu"

    def __post_init__(self):
        self.embed_dim = _positive_int(self.embed_dim, "embed_dim")
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError(
                "conditioned_pooling_query.inputs must be a non-empty mapping "
                "of input name to encoder spec"
            )
        if isinstance(self.hidden_dims, (str, bytes)) or not isinstance(
                self.hidden_dims, Sequence,
        ):
            raise ValueError(
                "hidden_dims must be a list of positive integers; got "
                f"{self.hidden_dims!r}"
            )
        self.hidden_dims = tuple(
            _positive_int(dim, f"hidden_dims[{index}]")
            for index, dim in enumerate(self.hidden_dims)
        )
        if self.activation is not None and self.activation != "none":
            _choice(self.activation, "activation", (*_ACTIVATIONS, "none"))


@resource("conditioned_pooling_query")
class ConditionedPoolingQuery(ModuleResource):
    """Build one query per sample from named conditioning inputs.

    Each entry of `inputs` describes one conditioning input: `module` is a
    dotted path to any `nn.Module` subclass, including a block component that
    the session does not need to drive, the optional `reduce` wraps it in
    `TokenReduction`, and every other key configures it. Each encoder must
    produce `embed_dim` features.

    The encodings are concatenated in `inputs` order and passed through an MLP
    with `hidden_dims` hidden layers. `activation` may be `None` (or "none")
    for a purely linear projection. Output shape is `(B, 1, embed_dim)`.

    `inputs` is hand-parsed rather than described by the schema: its entries
    are arbitrary user-supplied constructor calls, which no generic parser can
    validate.
    """

    config_schema = ConditionedPoolingQueryConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        embed_dim = self._cfg.embed_dim

        self.encoders = nn.ModuleDict()
        for name, spec in self._cfg.inputs.items():
            checked_name = self._checked_name(name)
            self.encoders[checked_name] = self._checked_encoder(
                checked_name,
                self._build_encoder(checked_name, spec),
            )

        activation = self._cfg.activation
        make_activation = (
            None
            if activation is None or activation == "none"
            else _ACTIVATIONS[activation]
        )
        dims = [
            len(self.encoders) * embed_dim,
            *self._cfg.hidden_dims,
            embed_dim,
        ]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            if index > 0 and make_activation is not None:
                layers.append(make_activation())
            layers.append(nn.Linear(in_dim, out_dim))
        self.projection = nn.Sequential(*layers)

    @staticmethod
    def _build_encoder(name: str, spec) -> nn.Module:
        context = f"conditioned_pooling_query.inputs.{name}"
        if not isinstance(spec, Mapping):
            raise ValueError(f"{context} must be a mapping; got {spec!r}")
        kwargs = dict(spec)
        reduce = kwargs.pop("reduce", None)
        module_class = _resolve_module_class(
            kwargs.pop("module", None),
            f"{context}.module",
        )
        is_component = issubclass(module_class, ModuleResource)
        if is_component and not ModuleResource.usable_as_plain_module(module_class):
            # This encoder is owned privately by the query, so the session
            # never wires or sets it up. A component that needs either has to
            # be bound to a role instead.
            raise TypeError(
                f"{context}.module cannot be {module_class.__name__}: it "
                "declares prerequisites or a lifecycle the session drives, "
                "which would never run for an encoder owned here. Bind it to "
                "a role, or use a plain nn.Module."
            )
        try:
            # A component takes its configuration as one mapping; a plain
            # module takes keyword arguments.
            encoder = module_class(kwargs) if is_component else module_class(**kwargs)
        except (TypeError, ValueError) as error:
            raise type(error)(f"Invalid {context} config: {error}") from error
        return encoder if reduce is None else TokenReduction(encoder, reduce=reduce)

    @staticmethod
    def _checked_name(name) -> str:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"input names must be Python identifiers; got {name!r}")
        if name in _MODULE_DICT_ATTRIBUTES:
            raise ValueError(
                f"input name {name!r} clashes with an attribute of the "
                "nn.ModuleDict holding the encoders; choose another name"
            )
        return name

    def _checked_encoder(self, name: str, encoder) -> nn.Module:
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"encoders.{name} must be an nn.Module; got {encoder!r}")
        # Modules that declare their own output size are checked up front;
        # the rest are checked on their first forward pass.
        for attribute in ("embed_dim", "out_features"):
            declared = getattr(encoder, attribute, None)
            if isinstance(declared, int) and declared != self._cfg.embed_dim:
                raise ValueError(
                    f"encoders.{name} has {attribute}={declared}, but must "
                    f"produce {self._cfg.embed_dim} features"
                )
        return encoder

    @property
    def embed_dim(self) -> int:
        return self._cfg.embed_dim

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self.encoders.keys())

    def forward(
            self,
            batch_size: int,
            **conditioning: torch.Tensor,
    ) -> torch.Tensor:
        expected = set(self.encoders.keys())
        if set(conditioning) != expected:
            raise TypeError(
                f"conditioned_pooling_query expects inputs {sorted(expected)}; "
                f"got {sorted(conditioning)}"
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
            if features.shape != (batch_size, self._cfg.embed_dim):
                raise ValueError(
                    f"Encoder for conditioning input '{name}' must return "
                    f"({batch_size}, {self._cfg.embed_dim}) features; got "
                    f"{tuple(features.shape)}"
                )
            encoded.append(features)

        return self.projection(torch.cat(encoded, dim=1)).unsqueeze(1)


# -- composites ---------------------------------------------------------


def _block_widths(block: nn.Module) -> tuple[int | None, int | None]:
    """Return a block's `(input, output)` token width.

    Every block currently preserves width, so one `embed_dim` describes both.
    A block that changes width only has to expose `in_embed_dim` and
    `out_embed_dim`, and the compatibility check below keeps working.
    """
    width = getattr(block, "embed_dim", None)
    return (
        getattr(block, "in_embed_dim", width),
        getattr(block, "out_embed_dim", width),
    )


@dataclass
class PatchTransformerConfig:
    class_token: Any = False

    def __post_init__(self):
        value = self.class_token
        if value is False or value is None:
            self.class_token = None
            return
        if value is True:
            self.class_token = {}
            return
        if not isinstance(value, Mapping):
            raise TypeError(
                "class_token must be a boolean or a mapping of ClassToken "
                f"options; got {value!r}"
            )
        options = dict(value)
        unknown = set(options) - {"init_std"}
        if unknown:
            raise TypeError(
                f"class_token got unexpected options {sorted(unknown)}; "
                "only 'init_std' is accepted"
            )
        if "init_std" in options:
            _non_negative_float(options["init_std"], "class_token.init_std")
        self.class_token = options


# Composites depend on roles with no default implementation, so, like the other
# role-consuming built-ins, they are registered per session type rather than
# in the shared scope.
@requires_resource("patch_embedding")
@requires_resource("positional_embedding")
@requires_resource("sequence_encoder")
@resource("patch_transformer", session_type=TRAINING_SESSION_TYPE)
@resource("patch_transformer", session_type=ANALYSIS_SESSION_TYPE)
class PatchTransformer(ModuleResource):
    """Encode images as tokens: patch embedding, positions, sequence encoder.

    The blocks bound to `block_chain` are attached during construction and
    become submodules named after their role, each owning and checkpointing
    its own weights.

    Config: `class_token` (default false) is `true` or a mapping of
    `ClassToken` options (e.g. `{init_std: 0.02}`). When enabled, a learned
    token is prepended after positional embedding, so outputs gain one
    leading token. It is the only weight a composite owns.
    """

    config_schema = PatchTransformerConfig

    block_chain: ClassVar[tuple[str, ...]] = (
        "patch_embedding",
        "positional_embedding",
        "sequence_encoder",
    )
    """Roles that pass tokens along, in order."""

    parallel_blocks: ClassVar[tuple[str, ...]] = ()
    """Roles that feed the chain sideways and must match its width."""

    linked_modules = block_chain + parallel_blocks

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        embed_dim = self._check_block_widths()
        if self._cfg.class_token is not None:
            self.class_token = ClassToken(embed_dim, **self._cfg.class_token)

    @property
    def has_class_token(self) -> bool:
        return self._cfg.class_token is not None

    @property
    def embed_dim(self) -> int:
        return self.patch_embedding.embed_dim

    def _check_block_widths(self) -> int:
        """Check every block agrees on token width, and return it."""
        widths = {
            role: _block_widths(getattr(self, role))
            for role in type(self).linked_modules
        }
        missing = sorted(
            role for role, (in_dim, out_dim) in widths.items()
            if not isinstance(in_dim, int) or not isinstance(out_dim, int)
        )
        if missing:
            raise TypeError(
                f"{self._component_name()} blocks must expose an integer "
                f"embed_dim; missing for {missing}"
            )

        chain = type(self).block_chain
        for producer, consumer in zip(chain, chain[1:]):
            produced = widths[producer][1]
            accepted = widths[consumer][0]
            if produced != accepted:
                raise ValueError(
                    f"{self._component_name()} blocks disagree on embed_dim: "
                    f"'{producer}' produces {produced} but '{consumer}' "
                    f"accepts {accepted}"
                )

        chain_width = widths[chain[-1]][1]
        for role in type(self).parallel_blocks:
            produced = widths[role][1]
            if produced != chain_width:
                raise ValueError(
                    f"{self._component_name()} blocks disagree on embed_dim: "
                    f"'{role}' produces {produced} but the encoded tokens are "
                    f"{chain_width}"
                )
        return widths[chain[0]][0]

    def encode(
            self,
            images: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return encoded tokens: `(B, N, embed_dim)`, or `(B, 1 + N, embed_dim)`
        with the class token at index 0 when `class_token` is enabled.

        `key_padding_mask` covers the `N` patch tokens only.
        """
        tokens, _ = self._encode(images, key_padding_mask)
        return tokens

    def _encode(
            self,
            images: torch.Tensor,
            key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        tokens = self.patch_embedding(images)
        grid_size = self.patch_embedding.grid_size(
            images.shape[-2], images.shape[-1],
        )
        tokens = self.positional_embedding(tokens, grid_size)
        if self.has_class_token:
            tokens, key_padding_mask = self.class_token(tokens, key_padding_mask)
        tokens = self.sequence_encoder(tokens, key_padding_mask=key_padding_mask)
        return tokens, key_padding_mask

    def forward(
            self,
            images: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(images, key_padding_mask=key_padding_mask)


@requires_resource("pooling")
@requires_resource("pooling_query")
@resource("pooled_patch_transformer", session_type=TRAINING_SESSION_TYPE)
@resource("pooled_patch_transformer", session_type=ANALYSIS_SESSION_TYPE)
class PooledPatchTransformer(PatchTransformer):
    """A `PatchTransformer` whose tokens are pooled by attention.

    `forward(images, **conditioning)` passes `conditioning` to the
    `pooling_query` block, so the query can be learned or built per sample.
    Returns `(B, Q, embed_dim)`. With `class_token` enabled, pooling attends
    over the class token as well as the patch tokens.
    """

    block_chain: ClassVar[tuple[str, ...]] = (
        *PatchTransformer.block_chain,
        "pooling",
    )
    parallel_blocks: ClassVar[tuple[str, ...]] = ("pooling_query",)

    linked_modules = block_chain + parallel_blocks

    def forward(
            self,
            images: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            **conditioning: torch.Tensor,
    ) -> torch.Tensor:
        tokens, key_padding_mask = self._encode(images, key_padding_mask)
        query = self.pooling_query(tokens.shape[0], **conditioning)
        return self.pooling(tokens, query, key_padding_mask=key_padding_mask)


__all__ = [
    "AttentionPooling",
    "ConditionedPoolingQuery",
    "ConvPatchEmbedding",
    "LearnedPoolingQuery",
    "LearnedPositionalEmbedding2D",
    "PatchEmbeddingBlock",
    "PatchTransformer",
    "PooledPatchTransformer",
    "PoolingBlock",
    "PoolingQueryBlock",
    "PositionalEmbeddingBlock",
    "SequenceEncoderBlock",
    "SinusoidalPositionalEmbedding2D",
    "SizedBlock",
    "TorchTransformerEncoder",
]
