"""Generic transformer blocks and their pluggable resource components."""

from training_framework.components.builtin.transformer.components import (
    AttentionPoolingFactory,
    ConditionedPoolingQueryFactory,
    ConvPatchEmbeddingFactory,
    LearnedPoolingQueryFactory,
    LearnedPositionalEmbedding2DFactory,
    ModuleFactory,
    PatchTransformer,
    PooledPatchTransformer,
    SinusoidalPositionalEmbedding2DFactory,
    TorchTransformerEncoderFactory,
)
from training_framework.components.builtin.transformer.modules import (
    AttentionPooling,
    ClassToken,
    ConditionedQuery,
    LearnedPositionalEmbedding2D,
    LearnedQuery,
    PatchEmbedding,
    SinusoidalPositionalEmbedding2D,
    TransformerEncoder,
)

__all__ = [
    "AttentionPooling",
    "AttentionPoolingFactory",
    "ClassToken",
    "ConditionedPoolingQueryFactory",
    "ConditionedQuery",
    "ConvPatchEmbeddingFactory",
    "LearnedPoolingQueryFactory",
    "LearnedPositionalEmbedding2D",
    "LearnedPositionalEmbedding2DFactory",
    "LearnedQuery",
    "ModuleFactory",
    "PatchEmbedding",
    "PatchTransformer",
    "PooledPatchTransformer",
    "SinusoidalPositionalEmbedding2D",
    "SinusoidalPositionalEmbedding2DFactory",
    "TorchTransformerEncoderFactory",
    "TransformerEncoder",
]
