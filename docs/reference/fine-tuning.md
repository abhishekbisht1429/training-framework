# Fine-tuning

[← Docs](../README.md) · [Built-in components](builtin-components.md)

A model trained in an earlier run comes in through
[`import_components`](import-components.md), which makes it -- and everything
it was wired to -- components of this session. Two built-in resources build on
it:

- **`module_part`** takes a part of a module resource -- the encoder of a whole
  pretrained model, say -- and calls one of its methods.
- **`backbone_with_head`** composes the `backbone` and `head` roles, freezes part
  of the backbone, and computes `head(backbone(x))`. Bind it as `model` and
  `ddp`, `forward` and `optimizer` use it like any other model. It was called
  `fine_tuned_model`; see [the former name](#fine_tuned_model-the-former-name).

The head is always yours: a `ModuleResource` bound to the `head` role. The
backbone is usually the imported model, or a `module_part` of it, but any
`nn.Module` resource can fill the role.

## Example

A classifier head on the first part of a pretrained encoder, with the first
half of the encoder frozen and a lower learning rate for the rest:

```yaml
import_components:
  pretrain_model:                                # that run's model instance
    checkpoint: runs/pretrain/checkpoints/last   # a checkpoint directory

role_bindings:
  model: backbone_with_head
  source: pretrain_model#imported                # what module_part takes a part of
  backbone: module_part
  head: cifar_head
  dataset: cifar10

module_part:
  submodule: transformer_backbone                # a part of the imported model

backbone_with_head:
  frozen: ["module._transformer_encoder.layers.[0-5].*"]

cifar_head: {embed_dim: 384, dropout: 0.1}

optimizer:
  optimizer: {name: AdamW, kwargs: {lr: 1.0e-3}}
  param_groups:
    - {match: ["backbone.*"], kwargs: {lr: 1.0e-5}}
```

The head declares what it needs itself -- here the class count, from
`dataset`:

```python
from torch import nn

from training_framework.components import (
    ModuleResource, requires_resource, resource,
)


@requires_resource("dataset")
@resource("cifar_head")
class CIFARHead(ModuleResource):
    def __init__(self, config):
        super().__init__(config)
        num_classes = self.get_dependency("dataset").num_classes
        self.dropout = nn.Dropout(self.config.get("dropout", 0.0))
        self.linear = nn.Linear(self.config["embed_dim"], num_classes)

    def forward(self, tokens):
        return self.linear(self.dropout(tokens.mean(dim=1)))
```

The head cannot ask the backbone for its width: requiring `backbone_with_head`
from the head would be a cycle. Give it in configuration, or use a lazy layer
(`nn.LazyLinear`). A wrong width is a shape error on the first forward pass.

When the pretrained model is itself the backbone -- a `pooled_patch_transformer`
wired to its blocks, say -- skip `module_part` and bind it directly
(`backbone: pooled_patch_transformer#imported`); the parameter names are then
`backbone.patch_embedding.*`,
`backbone.pooling.*` and so on.

## `module_part`

| Key | Default | Meaning |
|---|---|---|
| `submodule` | `""` | Attribute path inside the `source`, e.g. `encoder.blocks`; `""` keeps the whole source |
| `method` | `forward` | The method of the part that calling this resource calls, e.g. `forward_features` |

It requires the `source` role: an `nn.Module` resource, usually an imported
one (`source: <impl>#imported` in `role_bindings`). The part is held under the attribute
`module` (as `DistributedDataParallel` does), so its parameter names start with
`module.`.

The part stays the source's: the source trains, saves, restores and moves its
weights, and `module_part` owns none. What the source holds outside the part
-- a pretext head, say -- stays in the session and is saved with it, unused by
anything that only calls `module_part`. To avoid that for a wired model,
import just the component you want (key the import by its instance).

Errors, all when the session is built: a missing attribute (the error lists
the submodules that exist), something that is not an `nn.Module`, a missing
`method`, and a `source` that is not an `nn.Module`.

## `backbone_with_head`

| Key | Default | Meaning |
|---|---|---|
| `frozen` | `[]` | Glob patterns over the backbone's parameter names; each must match. `[]` or omitted: nothing frozen; any other non-list value (`false`, `null`, ...) is an error |
| `frozen_eval` | `true` when `frozen` is set | Keep frozen backbone blocks in eval mode |
| `backbone_output` | `null` | Which part of the backbone's output the head gets |
| `pool` | `null` | `cls` or `mean`: reduce `(B, N, D)` tokens to `(B, D)` before the head |

It requires the `backbone` and `head` roles; either one filled by something
that is not an `nn.Module` is an error. Both are attached as submodules named
after their role, so the model's parameter names are `backbone.<...>` and
`head.<...>`. Each part checkpoints its own weights; `backbone_with_head` owns
none.

**`frozen`** patterns are matched against the backbone's own parameter names
(through `module_part`, they start with `module.`), with the same rules as
[`optimizer.param_groups`](optimization.md#optimizer); a pattern that matches
nothing is an error. Matching parameters get `requires_grad=False` when the
model is built, before `ddp` wraps it: they receive no gradient, the backward
pass stops before them, DDP does not synchronise them and the optimizer never
updates them. `requires_grad` is not part of any checkpoint, so the freezing is
applied again on every build (resume, workers).

**`frozen_eval`**: every backbone block whose parameters are all frozen stays
in eval mode, also after `model.train()` and after the backbone's own `setup`.
Without it, a frozen BatchNorm still updates its running statistics and a
frozen block's dropout stays on -- the weights are frozen, but what the block
computes still changes. `frozen_eval: true` with nothing frozen is an error;
set `false` to let frozen blocks follow the model's mode.

**`backbone_output`** selects from the backbone's output: an integer indexes a
tuple or list; a string is a key of a mapping (a Hugging Face model output) or
an attribute of another object. A string is never looked up on a tensor. An
output that does not fit is an error on the first call, naming what the
backbone returned.

**`pool`** reduces the backbone's tokens to one vector per sample: `cls`
takes the class token, `mean` averages the other tokens (all of them when
there is no class token), leaving out the ones the padding mask marks. The
tokens are read through the backbone's `encode_with_mask` -- the
token-encoder contract `patch_transformer` follows: `(B, N, D)` tokens, the
class token first exactly when `has_class_token`, and the mask that covers
them -- never through `forward`, which a subclass may change:
`pooled_patch_transformer`'s `forward` returns pooled queries, so with
`pool` its tokens are pooled and its own pooling blocks go unused (which DDP
refuses, as for [`attention_pooled_encoder`](transformer-blocks.md#filling-patch_transformer-with-pooled_patch_transformer)).
A backbone without `encode_with_mask` and a boolean `has_class_token`, `cls`
with no class token, and `pool` together with `backbone_output` are errors
when the session is built. The call's arguments go to `encode_with_mask`, so
`model(images, key_padding_mask=mask)` pools only the unpadded tokens.
With `pool`, a linear probe or a fine-tuned classifier on a
`patch_transformer` needs no head of its own making that pools:

```yaml
role_bindings:
  model: backbone_with_head
  backbone: patch_transformer#imported
  head: projection_head          # num_layers: 1 is a linear layer
backbone_with_head:
  pool: cls
  frozen: ["*"]                  # a linear probe: only the head trains
projection_head: {in_dim: 384, out_dim: 1000, num_layers: 1}
```

### Freezing for part of the run

`frozen` holds for the whole run: DDP decides which parameters it synchronises
when it wraps the model. To unfreeze at some iteration, leave the parameters
out of `frozen` and use [`freeze_gradients`](optimization.md#freeze_gradients),
which drops their gradients until an iteration. That costs the backward pass
through them and does not change their train/eval mode.

### Learning rates per part

`optimizer.param_groups` patterns see the model's names, so `backbone.*` and
`head.*` select the two parts (see the example above).

## A backbone of your own

Any `nn.Module` resource can fill `backbone` -- a model built from
configuration, or one from another library:

```python
@resource("timm_backbone")
class TimmBackbone(ModuleResource):
    def __init__(self, config):
        super().__init__(config)
        self.net = timm.create_model(
            self.config["name"], pretrained=True, num_classes=0,
        )

    def forward(self, images):
        return self.net(images)
```

```yaml
role_bindings:
  backbone: timm_backbone
backbone_with_head:
  frozen: ["net.patch_embed.*", "net.blocks.[0-5].*"]
```

## `fine_tuned_model`, the former name

`fine_tuned_model` is `backbone_with_head` under its former name, kept so
configurations and checkpoints that use it still build and restore: same
roles, same keys (without `pool`), same parameter names and saved state.
Building it raises a `DeprecationWarning`. A run saved as `fine_tuned_model`
resumes under that name -- the name is part of every instance's checkpoint
entry -- so moving an existing run to `backbone_with_head` means starting a
new one (or [importing](import-components.md) its parts into one). New
configurations use `backbone_with_head`.
