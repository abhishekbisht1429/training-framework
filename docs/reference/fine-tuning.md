# Fine-tuning

[← Docs](../README.md) · [Built-in components](builtin-components.md)

Two built-in resources fine-tune a model trained in an earlier run:

- **`checkpoint_module`** takes a module out of another run's checkpoint and
  owns it from then on: it is trained, and checkpointed, in this run.
- **`fine_tuned_model`** composes the `backbone` and `head` roles, freezes part
  of the backbone, and computes `head(backbone(x))`. Bind it as `model` and
  `ddp`, `forward` and `optimizer` use it like any other model.

The head is always yours: a `ModuleResource` bound to the `head` role. The
backbone is usually `checkpoint_module`, but any `nn.Module` resource can fill
the role.

## Example

A classifier head on the first part of a pretrained encoder, with the first
half of the encoder frozen and a lower learning rate for the rest:

```yaml
component_bindings:
  model: fine_tuned_model
  backbone: checkpoint_module
  head: cifar_head
  dataset: cifar10

checkpoint_module:
  checkpoint: runs/pretrain/checkpoints/last   # a checkpoint directory
  submodule: transformer_backbone              # a part of that run's `model`

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

## `checkpoint_module`

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | required | Path of the checkpoint to take the module from |
| `resource` | `model` | A resource of that run, named as that run named it: a role (resolved through its own `component_bindings`) or an instance such as `model#teacher` |
| `submodule` | `""` | Attribute path inside that resource, e.g. `encoder.blocks`; `""` keeps the whole resource |
| `method` | `forward` | The method of the module that calling this resource calls, e.g. `forward_features` |

`resource` and `submodule` are two keys, not one path: a resource name follows
the framework's naming (roles, bindings, `#` instances), a submodule is plain
attribute access.

Only the selected module is kept, under the attribute `module` (as
`DistributedDataParallel` does), so its parameter names start with `module.`;
the parts of the source model left out are neither trained nor saved. The
source is loaded on the CPU with the caller's random state left as it was,
and moved to the session's device in `setup`.

Errors, all when the session is built: a missing checkpoint file, a resource
the checkpoint does not hold (or holds several instances of, with nothing to
choose between them), a missing attribute (the error lists the submodules that
exist), something that is not an `nn.Module`, a missing `method`, and a
component that the session would drive -- one that declares prerequisites or
overrides `setup`, `teardown`, `get_state` or `set_state` -- selected as a
whole: its lifecycle would never run here, so select a part of it with
`submodule`.

**The source checkpoint must stay where `checkpoint` points.** The module's
architecture is not in this run's checkpoint -- only its weights are -- so
every build reads the source again: a resume, each DDP worker, and an analysis
session loading this run's model. The saved weights then replace the source's.
If the source has moved, the build fails with `FileNotFoundError` naming the
path. Loading this run's model elsewhere likewise rebuilds what the model was
wired to, the head's own prerequisites (its `dataset`, say) included.

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
(with `checkpoint_module`, they start with `module.`), with the same rules as
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
component_bindings:
  backbone: timm_backbone
fine_tuned_model:
  frozen: ["net.patch_embed.*", "net.blocks.[0-5].*"]
```
