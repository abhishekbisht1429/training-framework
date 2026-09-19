# Built-in Components and Samplers

[← Documentation index](README.md) · [Project README](../README.md)

## Built-in components

Importing `training_framework` initializes `training_framework.components`,
which registers all built-ins. Their classes are also importable from
`training_framework.components.builtin`.

### Training built-ins

| Name | Kind | Purpose and dependencies |
|---|---|---|
| `logger` | Hook | Prints `Iteration <current>/<maximum>`, followed by ` \| lr: <lr>` (one value per param group) when `optimizer` is active; enabled by default |
| `checkpointer` | Hook | Saves complete session checkpoints; enabled by default |
| `ddp` | Resource | Initializes distributed execution and wraps the required `model` resource |
| `data_manager` | Stateful resource | Creates a resumable distributed `DataLoader`; requires `dataset` and `ddp` (analysis sessions use a [separate implementation](#analysis-data_manager)) |
| `optimizer` | Stateful lifecycle hook | Runs a configured PyTorch optimizer and optional learning-rate schedule; requires `ddp` and reads `iteration_context["loss"]` |
| `timer` | Lifecycle hook | Reports iteration and elapsed durations; wraps `optimizer` |
| `tensorboard` | Resource | Starts TensorBoard and exposes a `SummaryWriter` |

`dataset` and `model` are declared roles (see [Component
bindings](components.md#component-bindings)) with no built-in
implementation; register a `Resource` under that name, or bind one via
`component_bindings`, before activating `data_manager` or `ddp`.

The training defaults are equivalent to:

```yaml
logger:
  log_every: 10
  # log_file: ./runs/train.log  # optional; stdout when omitted

checkpointer:
  checkpoint_every: 100
```

An explicit component mapping replaces its default mapping, so retain required
fields such as `log_every` and `checkpoint_every` when overriding defaults.
The analysis logger likewise defaults to `log_every: 10`.

The optional training built-ins use these configurations:

```yaml
ddp:
  world_size: 1
  backend: gloo
  master_addr: "127.0.0.1"
  master_port: "12355"
  parallel_components: []

data_manager:
  batch_size: 32       # global batch size; divisible by world_size
  num_workers: 0
  pin_memory: false

optimizer:
  optimizer:
    name: AdamW
    kwargs:
      lr: 0.0003
      weight_decay: 0.01
  lr_scheduler:
    stages:
      - name: LinearLR
        kwargs:
          start_factor: 0.001
          total_iters: "$stage_iterations"
      - name: CosineAnnealingLR
        kwargs:
          T_max: "$stage_iterations"
    milestones: [100]

timer:
  call_every: 10

tensorboard:
  host: "127.0.0.1"
  port: 6006
  logdir: ./runs/tensorboard  # optional TensorBoard server log directory
```

`ddp.world_size`, `ddp.master_addr` and `ddp.master_port` describe the launch
rather than the session. They are resolved on every run and may be overridden
from the command line even when resuming, which is what lets a run continue on
a different number of GPUs — see
[The launch decides the topology](distributed-training.md#the-launch-decides-the-topology).
Because `batch_size` is global, a resize changes the per-rank batch while the
global batch, and so the optimizer's step semantics, stay put.

Optimizer names are resolved from `torch.optim`; scheduler names are resolved
from `torch.optim.lr_scheduler`. Constructor options belong in each entry's
`kwargs`. Omit `lr_scheduler` to train without a learning-rate scheduler. A
single scheduler can specify `metric_key` to pass an
`iteration_context[metric_key]` value to `scheduler.step(...)`; metric-driven
schedulers cannot be used in a multi-stage pipeline.

Multiple stages are combined with `SequentialLR`, so `milestones` must contain
one strictly increasing boundary between each pair of stages. Scheduler kwargs
may use the exact values `$max_iterations` and `$stage_iterations`, which are
resolved when the session starts. The latter is the distance between the
stage's surrounding milestones (or the start/end of the session).

The former `learning_rate`, `weight_decay`, and `warmup_iters` optimizer fields
are no longer accepted. Move optimizer arguments under `optimizer.kwargs` and
describe the warmup/main schedule explicitly as shown above.

`--extend-session` overrides may change `optimizer.optimizer.kwargs` values
(the optimizer class itself cannot change) and may replace `lr_scheduler`
entirely. Overriding `optimizer.optimizer.kwargs.lr` while leaving
`lr_scheduler` unchanged scales the active stage's base learning rate(s) by
the same ratio as the override, keeping its schedule progress; a multi-stage
schedule's not-yet-reached stage is unaffected and runs its own originally
configured base once it activates. Changing `lr_scheduler` itself
(scheduler class, stages, milestones, or `metric_key`) restarts its schedule
from the extension point; optimizer tensors and step counts are unaffected.
Setting `optimizer.lr_scheduler=null` removes scheduling: training continues
at the learning rate stored in the checkpoint, held fixed (combine with
`optimizer.optimizer.kwargs.lr=<value>` to pin a different rate).
See [Session-extension configuration](components.md#session-extension-configuration).

`data_manager.data_iter` is available only while the session is active. It
divides the global batch size across ranks and checkpoints delivered-batch
progress. A dataset resource may provide a callable `collate_fn(batch)` method
to control batching (this registers the implementation directly under the
declared `dataset` role's name):

```python
from torch.utils.data import Dataset

from training_framework.components import Resource, resource


@resource("dataset")
class TokenDataset(Dataset, Resource):
    ...

    def collate_fn(self, batch):
        return pad_sequences(batch)
```

When the dataset does not define `collate_fn`, the data manager uses
`torch.stack`. Keeping the collator on the importable dataset class makes it
available after checkpoint restoration and in spawned workers.

`optimizer` expects a loss tensor in `session.iteration_context` and performs
zeroing, backward propagation, optimization, and scheduler advancement.
If its pre-session initialization fails after creating an optimizer or
scheduler, its rollback callback clears those incomplete runtime handles
without changing the persisted component-state schema.

The TensorBoard resource starts the external `tensorboard` command, creates a
PyTorch `SummaryWriter`, and exposes it through `summary_writer`:

```python
tensorboard = session.get_resource("tensorboard")
tensorboard.summary_writer.add_scalar(
    "train/loss",
    loss,
    session.iteration,
)
```

The executable must be available and the selected port must be free. Teardown
closes the writer and terminates the external process.
If setup fails after creating either handle, rollback closes the writer when
present and terminates the partially started process.

### Analysis built-ins

| Name | Kind | Purpose |
|---|---|---|
| `trained_model` | Resource | Loads the `model` role from the source training checkpoint; enabled by default |
| `logger` | Hook | Prints `Analysis iteration <current>/<maximum>`; enabled by default |
| `layer_inspector` | Resource | Captures forward-pass input/output of selected `trained_model` layers via forward hooks; requires `trained_model`; not enabled by default |
| `data_manager` | Resource | Iterates the `dataset` role once, in order, with a plain `DataLoader`; requires `dataset` (no `ddp`); not enabled by default |

The analysis logger lives in the analysis registry. `trained_model` is shared
and can be activated by any session type. Analysis sessions activate it by
default, so an unbound analysis session must provide:

```yaml
trained_model:
  model_checkpoint_path: ./runs/session_.../checkpoints/<checkpoint-name>
```

The path must reference an existing, trusted framework `TrainingSession`
checkpoint whose `model` resource provides `to(device)` and `eval()`.

### Analysis `data_manager`

The analysis registry has its own `data_manager` (`AnalysisDataManager`),
separate from the training one. It does not use `ddp`, does not shuffle or
repeat, and keeps no resumable state: each sample is delivered exactly once,
in dataset order. `dataset` is also a declared role in the analysis scope;
register an analysis-scoped or shared `Resource` under that name, or bind one
with `component_bindings`.

```yaml
data_manager:
  batch_size: 32       # required, per-process batch size
  num_workers: 0       # optional
  pin_memory: false    # optional
  drop_last: false     # optional; drop a final partial batch
```

While the session is active, `data_manager.data_iter` is an iterator over the
`DataLoader` (also exposed as `data_manager.dataloader`), and the dataset's
optional `collate_fn` is used the same way as in training. When a step calls
`next(data_iter)` after the last batch, the `StopIteration` ends the analysis
session cleanly, so a run stops at whichever comes first: the end of the data
or `max_iterations`. The partial final iteration is not counted.

```python
@requires_resource("data_manager")
@step("embed_batches", session_type="analysis")
class EmbedBatches(Step):
    def run(self, session):
        batch = next(session.get_resource("data_manager").data_iter)
        model = session.get_resource("trained_model").model
        session.iteration_context["embeddings"] = model(batch)
```

### `layer_inspector`

`layer_inspector` automates the boilerplate of analyzing individual layers of
`trained_model.model` — finding the right layers and managing forward-hook
registration/removal — so an analysis `Step` can focus on interpreting
whatever the layer produced (an attention heatmap is one example; the
component itself renders nothing). It lives in the analysis registry and
requires `trained_model`; it is not activated by default, so it needs an
explicit top-level entry:

```yaml
layer_inspector:
  name_patterns:            # optional list[str], regex via re.search against
    - "encoder\\.layer\\.\\d+\\.attention$"    # the name from model.named_modules()
  module_types:              # optional list[str], dotted paths resolved to classes
    - torch.nn.MultiheadAttention
  always_call: false          # optional, default false; forwarded to register_forward_hook
```

At least one of `name_patterns` or `module_types` is required. Layers are
selected as the union of both: any module whose `model.named_modules()` name
matches one of `name_patterns`, or whose type matches one of `module_types`
(each entry a fully-qualified dotted path, since YAML cannot hold a Python
class). A module matched by both registers exactly one hook. If nothing
matches, `setup()` raises `ValueError` rather than silently doing nothing.

A `Step` reads captures through the resource:

```python
inspector = session.get_resource("layer_inspector")
model = session.get_resource("trained_model").model

model(some_input)  # triggers the registered forward hooks

for layer_name, captures in inspector.captures.items():
    for capture in captures:
        ...  # capture.input_args, capture.input_kwargs, capture.output,
             # capture.module (the layer itself, e.g. capture.module.weight)
```

`inspector.matched_layer_names` is the full, static set of layers selected
during `setup()`. `inspector.layers` maps each of those names to its live
module (read-only mapping, emptied on teardown), so a Step can read a layer's
parameters directly without a forward pass:

```python
for layer_name, layer in inspector.layers.items():
    for param_name, param in layer.named_parameters(recurse=False):
        ...  # e.g. param.detach().cpu() for a weight histogram
```

These are the trained model's own modules, not copies; don't modify them in
place. Each capture also carries the same module as `capture.module`, so a
Step can read the weights of the layer that produced `capture.output` without
going back to the inspector. `layers` remains the way to reach weights of
layers that did not run a forward pass this iteration.

`inspector.captures` is sparse — keyed only by layers that
actually ran a forward pass — and accumulates every forward pass within the
current iteration, in call order; it is cleared automatically at each
iteration boundary (backed by `session.iteration_context`), so an iteration
that never triggers a forward pass sees an empty mapping rather than stale
data from a previous one.

Each value is a list because a layer can run several times in one iteration
(multiple forward passes, a reused module, step-by-step decoding). When you
run one forward pass per iteration, `inspector.last_capture(layer_name)`
returns the latest `LayerCapture` for that layer, or `None` if it did not run
this iteration; it raises `KeyError` for a name that is not in
`matched_layer_names`.

Captured `input_args`/`input_kwargs`/`output` are live tensor references, not
detached copies — gradients remain enabled by default in analysis sessions,
so a captured tensor still carries its autograd graph if the triggering
forward pass wasn't run under `torch.no_grad()`. This is deliberate: a
gradient/attribution-style analysis needs the live graph. A Step that doesn't
need it should run its forward pass under `torch.no_grad()` itself.

### `ModuleResource`

`ModuleResource` is an `nn.Module` that is also a `StatefulResource`, for
models assembled from other resources. Subclasses create their own parameters
in `__init__`, like any other PyTorch module, and name the resources they
attach in `linked_modules`. Those children are attached by `super().__init__()`
before the subclass body runs, so a constructor can size its own weights from
them. Components are constructed [prerequisite-first on every
path](components.md#holding-another-component), so a restored model is usable
without `setup()` -- which is what `trained_model` relies on.

```python
@resource("text_encoder")
class TextEncoder(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.embedding = nn.Embedding(self._config["vocab_size"], 256)


@requires_resource("text_encoder")
@resource("model")
class CaptionedImageModel(ModuleResource):
    linked_modules = ("text_encoder",)

    def __init__(self, config=None):
        super().__init__(config)
        self.head = nn.Linear(512, self._config["num_classes"])

    def forward(self, image, tokens):
        ...
```

| Member | Purpose |
|---|---|
| `linked_modules` | Resource names attached as submodules under the same attribute |
| `attach_dependencies()` | Override to attach conditionally or under another attribute name |
| `attach_linked_module(attribute, component)` | Attach one component; rejects non-modules and re-attaching a different child |
| `linked_components` | The attribute -> component name map |
| `config_schema` | Optional dataclass; parsed into `self._cfg` (see [components](components.md)) |
| `usable_as_plain_module(cls)` | Whether a component class may be owned privately as an ordinary submodule |

**Ownership.** An attached child is a real submodule, so `model.parameters()`
covers it and `ddp`, the optimizer and `layer_inspector` see one module tree,
with each tensor appearing once. State is split the other way: a parent
excludes everything reachable from an attached child, so each component
checkpoints only the weights it created. `set_state` loads in place, which is
what keeps a parent's references valid.

A child may be shared by several models and is restored as one shared
instance. Attaching *another component's* weights as a plain submodule
instead of through `attach_linked_module` raises `ComponentDependencyError`
when the session captures state, naming both components and the shared tensor,
because it would otherwise be stored twice.

**What is recorded.** `linked_components` is a copy of the attribute ->
component name map built by `attach_linked_module`: the key is the attribute
the child hangs on (normally the role name), the value is the registered name
of the implementation that filled it.

```python
{"patch_embedding": "conv_patch_embedding",
 "positional_embedding": "learned_positional_embedding_2d",
 "sequence_encoder": "torch_transformer_encoder"}
```

The map does three jobs: it says which tensors belong to someone else, it tells
the ownership walk which subtrees to skip, and it is saved in the checkpoint
under `linked`. `set_state` compares the saved map against the current one and
refuses state captured from a differently wired instance -- a component class
whose `linked_modules` or `attach_dependencies` changed since the checkpoint
was written, or state moved between instances by hand. It is not a
configuration guard: a restore rebuilds components from the checkpoint's own
bindings, so rebinding a role in a later config never reaches this check, and a
renamed component is caught earlier, when the checkpoint entry no longer
matches a registered name.

**Owning a component privately.** A component class may also be used as an
ordinary submodule -- constructed and owned by another component, never
registered, its weights checkpointed inside its owner. That is allowed as long
as the session drives nothing about it: it must declare no `linked_modules`
and override none of `plain_module_api` (`attach_dependencies`, `get_state`,
`set_state`, `rollback_setup`, `setup`, `teardown`). One that does is rejected
with an explanation, because the session never calls those for a module it
does not know about. `ModuleResource.usable_as_plain_module(cls)` answers the
same question in code.

### Transformer blocks

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

Every block declares its keys with a [`config_schema`](components.md), so an
unknown key, a missing one or an impossible combination is reported by name
when the component is constructed -- before any worker starts.

`patch_size` and `grid_size` accept an integer or an `[h, w]` pair. A learned
positional table is resized for inputs whose patch grid differs from
`grid_size`.

The role contracts are also declared as `typing.Protocol`s
(`PatchEmbeddingBlock`, `PositionalEmbeddingBlock`, `SequenceEncoderBlock`,
`PoolingBlock`, `PoolingQueryBlock`), so a type checker can verify your own
block against the role it fills.

**Conditioned queries.** `conditioned_pooling_query` builds one query per
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

**Models.** Two composite resources, registered for both training and analysis
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

**Class token.** Set `class_token: true`, or pass `ClassToken` options such as
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

## Infinite samplers

### `InfiniteSampler`

`InfiniteSampler` repeatedly yields random permutations of dataset indices:

```python
from torch.utils.data import DataLoader
from training_framework.dataloader import InfiniteSampler


sampler = InfiniteSampler(len(dataset))
loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
)
```

It has no natural end. Use the session's `max_iterations` to bound training.

### `DistributedInfiniteSampler`

`DistributedInfiniteSampler` creates one deterministic, rank-specific slice of a shuffled global index sequence for each logical epoch:

```python
from training_framework.dataloader import DistributedInfiniteSampler


sampler = DistributedInfiniteSampler(
    num_samples=len(dataset),
    rank=rank,
    world_size=world_size,
    shuffle=True,
    seed=42,
    drop_last=False,
)
```

When rank and world size are omitted, it resolves them from an initialized PyTorch distributed process group, or falls back to rank 0 and world size 1.

It exposes:

```python
state = sampler.get_state()
sampler.set_state(state)
```

The iterator is infinite even though `len(sampler)` reports one rank-local logical epoch.

> **Checkpointing note:** with `DataLoader(num_workers > 0)`, sampler indices may be prefetched before their batches are consumed. Treat exact mid-epoch sampler restoration as experimental and track consumed progress in the training loop when exact replay matters.
