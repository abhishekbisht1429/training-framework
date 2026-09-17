"""Pluggable transformer resources.

Each building block is a `ModuleFactory`: a config-only resource that checks
its configuration and builds a fresh `nn.Module` on request. The composite
models (`patch_transformer`, `pooled_patch_transformer`) depend on block
*roles*, build their blocks in `setup()`, and own every resulting weight, so
DDP, optimizers, and checkpoints see a single module. `component_bindings`
chooses which factory fills each role.
"""

from __future__ import annotations

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
    ConditionedQuery,
    LearnedPositionalEmbedding2D,
    LearnedQuery,
    PatchEmbedding,
    SinusoidalPositionalEmbedding2D,
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


@resource("conditioned_pooling_query")
class ConditionedPoolingQueryFactory(ModuleFactory):
    module_class = ConditionedQuery


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
        if config:
            raise ValueError(
                f"{self._component_name()} takes no configuration; configure "
                f"its blocks instead. Got keys: {sorted(config)}"
            )
        self._built = False

    @classmethod
    def _component_name(cls) -> str:
        return getattr(cls, "name", cls.__name__)

    @property
    def is_built(self) -> bool:
        return self._built

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
        return {
            "modules": {
                block_role: getattr(self, block_role)
                for block_role in self.block_roles
            },
        }

    def set_state(self, state: Mapping[str, Any] | None) -> None:
        if state is None:
            return
        self._attach_blocks(dict(state["modules"]))

    def _attach_blocks(self, blocks: dict[str, nn.Module]) -> None:
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
        for block_role in self.block_roles:
            setattr(self, block_role, blocks[block_role])
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
        """Return `(B, N, embed_dim)` encoded patch tokens."""
        self._require_built()
        if images.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")
        tokens = self.patch_embedding(images)
        grid_size = self.patch_embedding.grid_size(images.shape[-2], images.shape[-1])
        tokens = self.positional_embedding(tokens, grid_size)
        return self.sequence_encoder(tokens, key_padding_mask=key_padding_mask)

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
    Returns `(B, Q, embed_dim)`.
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
        tokens = self.encode(images, key_padding_mask=key_padding_mask)
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
