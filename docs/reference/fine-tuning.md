# Fine-tuning

[← Docs](../README.md) · [Built-in components](builtin-components.md)

A model trained in an earlier run comes in through
[`import_components`](import-components.md), which makes it -- and everything
it was wired to -- components of this session. Two built-in resources build on
it:

- **`module_part`** takes a part of a module resource -- the encoder of a whole
  pretrained model, say -- and calls one of its methods.
- **`fine_tuned_model`** composes the `backbone` and `head` roles, freezes part
  of the backbone, and computes `head(backbone(x))`. Bind it as `model` and
  `ddp`, `forward` and `optimizer` use it like any other model.

The head is always yours: a `ModuleResource` bound to the `head` role. The
backbone is usually the imported model, or a `module_part` of it, but any
`nn.Module` resource can fill the role.

## Example

A classifier head on the first part of a pretrained encoder, with the first
half of the encoder frozen and a lower learning rate for the rest:

```yaml
import_components:
  pretrained:
    checkpoint: runs/pretrain/checkpoints/last   # a checkpoint directory
    role: source                                 # that run's `model`

role_bindings:
  model: fine_tuned_model
  backbone: module_part
  head: cifar_head
  dataset: cifar10

module_part:
  submodule: transformer_backbone                # a part of the imported model

fine_tuned_model:
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

The head cannot ask the backbone for its width: requiring `fine_tuned_model`
from the head would be a cycle. Give it in configuration, or use a lazy layer
(`nn.LazyLinear`). A wrong width is a shape error on the first forward pass.

When the pretrained model is itself the backbone -- a `pooled_patch_transformer`
wired to its blocks, say -- skip `module_part` and give the import
`role: backbone`; the parameter names are then `backbone.patch_embedding.*`,
`backbone.pooling.*` and so on.

## `module_part`

| Key | Default | Meaning |
|---|---|---|
| `submodule` | `""` | Attribute path inside the `source`, e.g. `encoder.blocks`; `""` keeps the whole source |
| `method` | `forward` | The method of the part that calling this resource calls, e.g. `forward_features` |

It requires the `source` role: an `nn.Module` resource, usually an imported
one (`role: source` on the import). The part is held under the attribute
`module` (as `DistributedDataParallel` does), so its parameter names start with
`module.`.

The part stays the source's: the source trains, saves, restores and moves its
weights, and `module_part` owns none. What the source holds outside the part
-- a pretext head, say -- stays in the session and is saved with it, unused by
anything that only calls `module_part`. To avoid that for a wired model,
import just the component you want (`resource: <its instance>`).

Errors, all when the session is built: a missing attribute (the error lists
the submodules that exist), something that is not an `nn.Module`, a missing
`method`, and a `source` that is not an `nn.Module`.

## `fine_tuned_model`

| Key | Default | Meaning |
|---|---|---|
| `frozen` | `[]` | Glob patterns over the backbone's parameter names; each must match. `[]` or omitted: nothing frozen; any other non-list value (`false`, `null`, ...) is an error |
| `frozen_eval` | `true` when `frozen` is set | Keep frozen backbone blocks in eval mode |
| `backbone_output` | `null` | Which part of the backbone's output the head gets |

It requires the `backbone` and `head` roles; either one filled by something
that is not an `nn.Module` is an error. Both are attached as submodules named
after their role, so the model's parameter names are `backbone.<...>` and
`head.<...>`. Each part checkpoints its own weights; `fine_tuned_model` owns
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
fine_tuned_model:
  frozen: ["net.patch_embed.*", "net.blocks.[0-5].*"]
```
