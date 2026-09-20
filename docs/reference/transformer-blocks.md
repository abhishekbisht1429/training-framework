# Transformer Blocks

[← Docs](../README.md) · [Project README](../../README.md)

Generic, swappable transformer building blocks, plus the two composite models
that wire them together. This page lists each block's configuration keys and
the role it fills.

Every block is a [`ModuleResource`](../concepts/module-resource.md), so read
that page first if you want to add a block of your own.

## The blocks

`training_framework.components.builtin.transformer` provides generic
transformer building blocks you can swap in and out. Each block is a
`ModuleResource`: it creates its own weights in `__init__`, owns and
checkpoints them, and is configured, bound and shared like any other
component. `transformer/modules.py` keeps only what is not a component --
`ClassToken`, whose weights belong to the composite that prepends it, and
`TokenReduction`, which wraps a user-supplied encoder.

| Block | Config keys | Fills role |
|---|---|---|
| `conv_patch_embedding` | `in_channels, patch_size, embed_dim, bias=True` | `patch_embedding` |
| `learned_positional_embedding_2d` | `grid_size, embed_dim, init="zeros", init_std=0.02, interpolation_mode="bilinear"` | `positional_embedding` |
| `sinusoidal_positional_embedding_2d` | `embed_dim, temperature=10000.0` | `positional_embedding` |
| `torch_transformer_encoder` | `embed_dim, num_heads, num_layers, dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False, layer_norm_eps=1e-5, final_norm=False` | `sequence_encoder` |
| `attention_pooling` | `embed_dim, num_heads, dropout=0.0, bias=True, need_weights=False, average_attn_weights=True` | `pooling` |
| `learned_pooling_query` | `embed_dim, num_queries=1, init_std=0.02` | `pooling_query` |
| `conditioned_pooling_query` | `embed_dim, inputs, hidden_dims=[], activation="gelu"` | `pooling_query` |

Every block declares its keys with a [`config_schema`](../concepts/component-model.md#declaring-a-configuration-schema), so an
unknown key, a missing one or an impossible combination is reported by name
when the component is constructed -- before any worker starts.

`patch_size` and `grid_size` accept an integer or an `[h, w]` pair. A learned
positional table is resized for inputs whose patch grid differs from
`grid_size`.

The role contracts are also declared as `typing.Protocol`s
(`PatchEmbeddingBlock`, `PositionalEmbeddingBlock`, `SequenceEncoderBlock`,
`PoolingBlock`, `PoolingQueryBlock`), so a type checker can verify your own
block against the role it fills.

## Conditioned queries

`conditioned_pooling_query` builds one query per
sample from named conditioning inputs. Each input has its own encoder module,
which must return `(B, embed_dim)` features; the encodings are concatenated
and passed through an MLP with activations between its layers
(`activation: null` or `none` makes the projection purely linear).

```yaml
conditioned_pooling_query:
  embed_dim: 256
  inputs:
    obj_patch:                 # the keyword this input is passed as
      module: training_framework.components.builtin.transformer.ConvPatchEmbedding
      in_channels: 3           # any other key goes to the module constructor
      patch_size: 16
      embed_dim: 256
      reduce: mean             # optional: wrap in TokenReduction (mean|max)
    obj_patch_location:
      module: torch.nn.Linear
      in_features: 2
      out_features: 256
  hidden_dims: [256]
```

`module` is a dotted path to any `nn.Module` subclass, including your own and
including a block component the session does not need to drive -- a component
encoder takes its keys as its configuration, and its weights are checkpointed
inside the query that owns it. A component that declares prerequisites or a
lifecycle is rejected, since neither would run here; bind that one to a role
instead. `reduce` wraps the module in `TokenReduction`,
which turns `(B, N, D)` tokens into one `(B, D)` vector -- that is how a patch
encoder, or any other token-producing module, meets the contract. An encoder
that declares `embed_dim` or `out_features` is checked when the query is
built; any other module is checked on its first forward pass.

## Models

Two composite resources, registered for both training and analysis
sessions, depend on the roles above. Their only config key is `class_token`
(see below); otherwise use `{}`:

- `patch_transformer` requires `patch_embedding`, `positional_embedding` and
  `sequence_encoder`. `model(images, key_padding_mask=None)` returns
  `(B, N, embed_dim)` tokens.
- `pooled_patch_transformer` also requires `pooling_query` and `pooling`.
  `model(images, key_padding_mask=None, **conditioning)` passes
  `conditioning` to the query block and returns `(B, Q, embed_dim)`.

A composite is wiring: it attaches the blocks bound to its roles as submodules
named after those roles (`model.patch_embedding`, `model.sequence_encoder`,
...) while it is constructed, and checks that each block's output width
matches the next block's input width. Because the blocks are real submodules,
`ddp`, the optimizer and `layer_inspector` still see one module tree, while
each block checkpoints only the weights it created. A composite owns no
weights of its own apart from an optional class token, and is usable as soon
as it is constructed -- no `setup()` required, which is what `trained_model`
relies on.

## Class token

Set `class_token: true`, or pass `ClassToken` options such as
`class_token: {init_std: 0.02}`, on either model to prepend one learned token.
It is added after positional embedding, so it has no position of its own (a
learned token doesn't need one) and positional-table resizing is unaffected.
Outputs gain one leading token: `patch_transformer` returns
`(B, 1 + N, embed_dim)` with the class token at `tokens[:, 0]`, and
`pooled_patch_transformer` pools over the class token as well as the patch
tokens. A `key_padding_mask` still covers only the `N` patch tokens; the model
widens it so the class token is never masked. It is the one weight a composite
owns, saved as `model.class_token`, and like the rest of a model's shape it
can't change during `--extend-session`.

```yaml
patch_transformer:
  class_token: true
```

Bind the model and each role through `component_bindings`, and configure the
blocks by name. This example reproduces an image encoder whose output is
pooled by a query built from an object crop and its 2D location:

```yaml
component_bindings:
  model: pooled_patch_transformer
  patch_embedding: conv_patch_embedding
  positional_embedding: learned_positional_embedding_2d
  sequence_encoder: torch_transformer_encoder
  pooling: attention_pooling
  pooling_query: conditioned_pooling_query
pooled_patch_transformer: {}
conv_patch_embedding: {in_channels: 3, patch_size: 16, embed_dim: 256}
learned_positional_embedding_2d: {grid_size: [14, 14], embed_dim: 256}
torch_transformer_encoder: {embed_dim: 256, num_heads: 8, num_layers: 6}
attention_pooling: {embed_dim: 256, num_heads: 4}
conditioned_pooling_query:
  embed_dim: 256
  inputs:
    obj_patch:
      module: training_framework.components.builtin.transformer.ConvPatchEmbedding
      in_channels: 3
      patch_size: 16
      embed_dim: 256
      reduce: mean
    obj_patch_location:
      module: torch.nn.Linear
      in_features: 2
      out_features: 256
  hidden_dims: [256]
```

```python
pooled = model(images, obj_patch=crops, obj_patch_location=locations)  # (B, 1, 256)
```

To use a learned (CLS-style) query instead, bind
`pooling_query: learned_pooling_query` and configure it. To use fixed
positions, bind `positional_embedding: sinusoidal_positional_embedding_2d`.
To add your own block, subclass `ModuleResource`, create its weights in
`__init__`, register it with `@resource(...)`, and bind it to the role. The
block must follow the call signature in the role's description and expose
`embed_dim`.

---

**See also:** [`ModuleResource`](../concepts/module-resource.md) for the
ownership and checkpointing rules these blocks follow, and
[built-in components](builtin-components.md) for everything else the framework
ships with.
