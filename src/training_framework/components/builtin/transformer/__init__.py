"""Transformer blocks as components, plus the pieces they share."""

from training_framework.components.builtin.transformer.components import (
    AttentionPooling,
    ConditionedPoolingQuery,
    ConvPatchEmbedding,
    LearnedPoolingQuery,
    LearnedPositionalEmbedding2D,
    PatchEmbeddingBlock,
    PatchTransformer,
    PooledPatchTransformer,
    PoolingBlock,
    PoolingQueryBlock,
    PositionalEmbeddingBlock,
    SequenceEncoderBlock,
    SinusoidalPositionalEmbedding2D,
    SizedBlock,
    TorchTransformerEncoder,
)
from training_framework.components.builtin.transformer.modules import (
    ClassToken,
    TokenReduction,
)

__all__ = [
    "AttentionPooling",
    "ClassToken",
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
    "TokenReduction",
    "TorchTransformerEncoder",
]
