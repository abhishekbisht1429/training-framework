"""Pluggable transformer resources.

Each building block is a `ModuleFactory`: a config-only resource that checks
its configuration and builds a fresh `nn.Module` on request. The composite
models (`patch_transformer`, `pooled_patch_transformer`) depend on block
*roles*, build their blocks in `setup()`, and own every resulting weight, so
DDP, optimizers, and checkpoints see a single module. `component_bindings`
chooses which factory fills each role.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    Resource,
    StatefulResource,
    requires_resource,
    resource,
    role,
)
from training_framework.components.builtin.transformer.modules import (
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

if TYPE_CHECKING:
    from training_framework.session import Session


role(
    "patch_embedding",
    Resource,
    description=(
        "a ModuleFactory whose module maps (B, C, H, W) images to "
        "(B, N, embed_dim) tokens and provides grid_size(height, width)"
    ),
)
role(
    "positional_embedding",
    Resource,
    description=(
        "a ModuleFactory whose module is called as module(tokens, grid_size) "
        "and returns tokens with positions added"
    ),
)
role(
    "sequence_encoder",
    Resource,
    description=(
        "a ModuleFactory whose module is called as "
        "module(tokens, key_padding_mask=None) and returns encoded tokens"
    ),
)
role(
    "pooling",
    Resource,
    description=(
        "a ModuleFactory whose module is called as "
        "module(tokens, query, key_padding_mask=None) and returns (B, Q, embed_dim)"
    ),
)
role(
    "pooling_query",
    Resource,
    description=(
        "a ModuleFactory whose module is called as "
        "module(batch_size, **conditioning) and returns a (B, Q, embed_dim) query"
    ),
)


class ModuleFactory(Resource):
    """A config-only resource that builds one `nn.Module`.

    Subclasses set `module_class`; the config mapping is passed to it as
    keyword arguments. The config is checked at construction by building the
    module on the meta device, so mistakes surface before any session runs
    without allocating real weights. The factory owns no weights: whoever
    calls `build()` owns the returned module.
    """

    module_class: ClassVar[type[nn.Module]]

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        component_name = getattr(type(self), "name", type(self).__name__)
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError(f"{component_name} config must be a mapping")
        self._config = deepcopy(dict(config))
        try:
            with torch.device("meta"):
                self.build()
        except (TypeError, ValueError) as error:
            raise type(error)(f"Invalid {component_name} config: {error}") from error

    @property
    def config(self) -> dict[str, Any]:
        return deepcopy(self._config)

    def build(self) -> nn.Module:
        """Return a new module built from this factory's config."""
        return self.module_class(**deepcopy(self._config))

    def setup(self, session: Session) -> None:
        pass

    def teardown(self, session: Session) -> None:
        pass


@resource("conv_patch_embedding")
class ConvPatchEmbeddingFactory(ModuleFactory):
    module_class = PatchEmbedding


@resource("learned_positional_embedding_2d")
class LearnedPositionalEmbedding2DFactory(ModuleFactory):
    module_class = LearnedPositionalEmbedding2D


@resource("sinusoidal_positional_embedding_2d")
class SinusoidalPositionalEmbedding2DFactory(ModuleFactory):
    module_class = SinusoidalPositionalEmbedding2D


@resource("torch_transformer_encoder")
class TorchTransformerEncoderFactory(ModuleFactory):
    module_class = TransformerEncoder


@resource("attention_pooling")
class AttentionPoolingFactory(ModuleFactory):
    module_class = AttentionPooling


@resource("learned_pooling_query")
class LearnedPoolingQueryFactory(ModuleFactory):
    module_class = LearnedQuery


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
        raise ImportError(f"{context} {dotted_path!r} could not be imported: {error}") from error
    try:
        resolved = getattr(imported, attr_name)
    except AttributeError as error:
        raise ValueError(
            f"{context} {dotted_path!r} has no attribute {attr_name!r} in "
            f"module {module_path!r}"
        ) from error
    if not isinstance(resolved, type) or not issubclass(resolved, nn.Module):
        raise TypeError(f"{context} {dotted_path!r} does not resolve to an nn.Module subclass")
    return resolved


@resource("conditioned_pooling_query")
class ConditionedPoolingQueryFactory(ModuleFactory):
    """Build a `ConditionedQuery` whose encoders come from config.

    Each entry of `inputs` describes one conditioning input: `module` is a
    dotted path to any `nn.Module` subclass, the optional `reduce` wraps it in
    `TokenReduction`, and every other key is passed to its constructor. Each
    encoder must produce `embed_dim` features.
    """

    module_class = ConditionedQuery

    def build(self) -> nn.Module:
        config = deepcopy(self._config)
        inputs = config.pop("inputs", None)
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError(
                "conditioned_pooling_query.inputs must be a non-empty mapping "
                "of input name to encoder spec"
            )
        encoders = {
            name: self._build_encoder(name, spec) for name, spec in inputs.items()
        }
        return ConditionedQuery(encoders=encoders, **config)

    @staticmethod
    def _build_encoder(name, spec) -> nn.Module:
        context = f"conditioned_pooling_query.inputs.{name}"
        if not isinstance(spec, Mapping):
            raise ValueError(f"{context} must be a mapping; got {spec!r}")
        kwargs = deepcopy(dict(spec))
        reduce = kwargs.pop("reduce", None)
        module_class = _resolve_module_class(kwargs.pop("module", None), f"{context}.module")
        try:
            encoder = module_class(**kwargs)
        except (TypeError, ValueError) as error:
            raise type(error)(f"Invalid {context} config: {error}") from error
        return encoder if reduce is None else TokenReduction(encoder, reduce=reduce)


# Composites depend on roles with no default implementation, so, like the other
# role-consuming built-ins, they are registered per session type rather than
# in the shared scope.
@requires_resource("patch_embedding")
@requires_resource("positional_embedding")
@requires_resource("sequence_encoder")
@resource("patch_transformer", session_type=TRAINING_SESSION_TYPE)
@resource("patch_transformer", session_type=ANALYSIS_SESSION_TYPE)
class PatchTransformer(nn.Module, StatefulResource):
    """Encode images as tokens: patch embedding, positions, sequence encoder.

    Blocks are built in `setup()` from the factories bound to `block_roles`
    and become submodules named after their role. The checkpointed state holds
    the built modules themselves, so a restored model works without calling
    `setup()` (as `trained_model` requires); `setup()` then keeps them rather
    than building new ones.

    Config: `class_token` (default false) is `true` or a mapping of
    `ClassToken` options (e.g. `{init_std: 0.02}`). When enabled, a learned
    token is prepended after positional embedding, so outputs gain one
    leading token.
    """

    block_roles: ClassVar[tuple[str, ...]] = (
        "patch_embedding",
        "positional_embedding",
        "sequence_encoder",
    )

    def __init__(self, config: Mapping | None = None) -> None:
        nn.Module.__init__(self)
        if config is not None and not isinstance(config, Mapping):
            raise TypeError(f"{self._component_name()} config must be a mapping")
        config = dict(config or {})
        unknown = set(config) - {"class_token"}
        if unknown:
            raise ValueError(
                f"{self._component_name()} only accepts 'class_token'; "
                f"configure its blocks instead. Got keys: {sorted(unknown)}"
            )
        self._class_token_options = self._parse_class_token(config.get("class_token", False))
        self._built = False

    @classmethod
    def _parse_class_token(cls, value) -> dict[str, Any] | None:
        if value is False or value is None:
            return None
        if value is True:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{cls._component_name()}.class_token must be a boolean or a "
                f"mapping of ClassToken options; got {value!r}"
            )
        options = deepcopy(dict(value))
        try:
            with torch.device("meta"):
                ClassToken(embed_dim=1, **options)
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"Invalid {cls._component_name()}.class_token config: {error}"
            ) from error
        return options

    @classmethod
    def _component_name(cls) -> str:
        return getattr(cls, "name", cls.__name__)

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def has_class_token(self) -> bool:
        return self._class_token_options is not None

    @property
    def embed_dim(self) -> int:
        self._require_built()
        return self.patch_embedding.embed_dim

    def setup(self, session: Session) -> None:
        if not self._built:
            blocks = {}
            for block_role in self.block_roles:
                factory = session.get_resource(block_role)
                if not isinstance(factory, ModuleFactory):
                    raise TypeError(
                        f"{self._component_name()} requires '{block_role}' to "
                        f"be a ModuleFactory; got {type(factory).__name__}"
                    )
                blocks[block_role] = factory.build()
            self._attach_blocks(blocks)
        self.to(session.device)

    def teardown(self, session: Session) -> None:
        pass

    def get_state(self) -> dict[str, Any] | None:
        if not self._built:
            return None
        modules = {
            block_role: getattr(self, block_role)
            for block_role in self.block_roles
        }
        if self.has_class_token:
            modules["class_token"] = self.class_token
        return {"modules": modules}

    def set_state(self, state: Mapping[str, Any] | None) -> None:
        if state is None:
            return
        modules = dict(state["modules"])
        class_token = modules.pop("class_token", None)
        if (class_token is not None) != self.has_class_token:
            raise ValueError(
                f"{self._component_name()} state "
                f"{'has' if class_token is not None else 'lacks'} a class token "
                f"but class_token is {'enabled' if self.has_class_token else 'disabled'}"
            )
        self._attach_blocks(modules, class_token=class_token)

    def _attach_blocks(
            self,
            blocks: dict[str, nn.Module],
            class_token: ClassToken | None = None,
    ) -> None:
        if set(blocks) != set(self.block_roles):
            raise ValueError(
                f"{self._component_name()} expects blocks "
                f"{list(self.block_roles)}; got {sorted(blocks)}"
            )
        embed_dims = {
            block_role: getattr(block, "embed_dim", None)
            for block_role, block in blocks.items()
        }
        missing = [
            block_role for block_role, dim in embed_dims.items()
            if not isinstance(dim, int)
        ]
        if missing:
            raise TypeError(
                f"{self._component_name()} blocks must expose an integer "
                f"embed_dim; missing for {missing}"
            )
        if len(set(embed_dims.values())) != 1:
            raise ValueError(
                f"{self._component_name()} blocks disagree on embed_dim: "
                f"{embed_dims}"
            )
        if self.has_class_token:
            [embed_dim] = set(embed_dims.values())
            if class_token is None:
                class_token = ClassToken(embed_dim, **deepcopy(self._class_token_options))
            if class_token.embed_dim != embed_dim:
                raise ValueError(
                    f"{self._component_name()} class token embed_dim "
                    f"{class_token.embed_dim} does not match blocks ({embed_dim})"
                )
        for block_role in self.block_roles:
            setattr(self, block_role, blocks[block_role])
        if class_token is not None:
            self.class_token = class_token
        self._built = True

    def _require_built(self) -> None:
        # Not `requires_context`: that checks for an active session, but the
        # model must also work after teardown and when restored from a
        # checkpoint without setup() (the `trained_model` path).
        if not self._built:
            raise RuntimeError(
                f"{self._component_name()} has no blocks yet; they are built "
                "in setup() or restored from a checkpoint"
            )

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
        self._require_built()
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        tokens = self.patch_embedding(images)
        grid_size = self.patch_embedding.grid_size(images.shape[-2], images.shape[-1])
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

    block_roles: ClassVar[tuple[str, ...]] = (
        *PatchTransformer.block_roles,
        "pooling_query",
        "pooling",
    )

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
    "AttentionPoolingFactory",
    "ConditionedPoolingQueryFactory",
    "ConvPatchEmbeddingFactory",
    "LearnedPoolingQueryFactory",
    "LearnedPositionalEmbedding2DFactory",
    "ModuleFactory",
    "PatchTransformer",
    "PooledPatchTransformer",
    "SinusoidalPositionalEmbedding2DFactory",
    "TorchTransformerEncoderFactory",
]
