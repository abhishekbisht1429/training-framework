# Built-in Components

[← Docs](../README.md) · [Project README](../../README.md)

Reference for the components the framework ships with: what each one does, the
configuration keys it accepts, and which are enabled by default. Look here for
a key you need to set; for how components are activated and bound in the first
place, see [Wiring components together](../guide/02-wiring-components.md).

Importing `training_framework` initializes `training_framework.components`,
which registers all built-ins. Their classes are also importable from
`training_framework.components.builtin`.

## Training built-ins

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
bindings](../guide/02-wiring-components.md#component-bindings)) with no built-in
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
  rank_zero_components: []   # optional; built on rank 0 only

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

### `ddp`

The DDP resource is a [singleton](../guide/02-wiring-components.md#components-that-must-stay-unique):
it owns the process group, and the engine, worker and session each expect
exactly one, so configuring a second instance is rejected.

`ddp.world_size`, `ddp.master_addr` and `ddp.master_port` describe the launch
rather than the session. They are resolved on every run and may be overridden
from the command line even when resuming, which is what lets a run continue on
a different number of GPUs — see
[The launch decides the topology](../guide/05-distributed-training.md#the-launch-decides-the-topology).
Because `batch_size` is global, a resize changes the per-rank batch while the
global batch, and so the optimizer's step semantics, stay put.

`ddp.rank_zero_components` is optional and lists components this session keeps
off every rank but rank 0, on top of the ones whose classes are marked with
`@rank_zero_only` — see
[What each rank builds](../guide/05-distributed-training.md#what-each-rank-builds).
The former `ddp.parallel_components`, which listed the components to keep *on*
the other ranks, is deprecated: a session that still sets it keeps the old
opt-in behaviour and warns.

### `optimizer`

`optimizer` expects a loss tensor in `session.iteration_context` and performs
zeroing, backward propagation, optimization, and scheduler advancement.
If its pre-session initialization fails after creating an optimizer or
scheduler, its rollback callback clears those incomplete runtime handles
without changing the persisted component-state schema.

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
from the extension point (`last_epoch` and any per-stage progress reset), since
old scheduler state cannot be assumed compatible with a different schedule
shape; optimizer tensors and step counts are unaffected.
Setting `optimizer.lr_scheduler=null` removes scheduling: training continues
at the learning rate stored in the checkpoint, held fixed (combine with
`optimizer.optimizer.kwargs.lr=<value>` to pin a different rate).

For the operator's view of what `--extend-session` accepts, see
[Extend](../guide/04-checkpoints-and-resume.md#extend); for the contract a
custom component implements to opt in, see
[Opting into extension](../concepts/component-model.md#opting-into-extension).

### `data_manager`


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

### `checkpointer`

Add the built-in `checkpointer` hook to YAML:

```yaml
checkpointer:
  checkpoint_every: 100
  checkpoints_dir: ./runs/checkpoints  # optional
  checkpoint_first: false              # optional; defaults to false
```

If `checkpoints_dir` is omitted, checkpoints are written to a `checkpoints`
directory under the session directory. A [second instance](../guide/02-wiring-components.md#configuring-a-component-more-than-once)
writes to `checkpoints_<suffix>` instead, so two checkpointers do not
interleave their files; an explicit `checkpoints_dir` still wins.

The checkpointer uses `torch.save(session, path)`. Because it is an iteration
hook, it saves on:

- iterations divisible by `checkpoint_every`;
- the final configured iteration; and
- the first iteration only when it is also the final iteration or
  `checkpoint_first: true`.

A component that needs something another run produced reads it with
`Checkpointer.load_component(path, name, session_type=None)`. The name is
resolved through the *checkpoint's* bindings -- `model` finds whatever that
run bound it to -- and the checkpoint's RNG is not adopted. The caller's RNG
is left exactly as it was, even when the rebuilt components draw from it, so
the caller's seed still decides what comes next. It raises `KeyError` when the checkpoint
has no such resource and `ComponentDependencyError` when several instances
answer; `session_type="training"` also rejects a checkpoint of another kind.
This is how the analysis `trained_model` gets its model.

### `tensorboard`

The TensorBoard resource starts the external `tensorboard` command, creates a
PyTorch `SummaryWriter`, and exposes it through `summary_writer`:

```python
# In a component that declares @requires_resource("tensorboard")
tensorboard = self.get_dependency("tensorboard")
tensorboard.summary_writer.add_scalar(
    "train/loss",
    loss,
    session.iteration,
)
```

The executable must be available and the selected port must be free. A
[second instance](../guide/02-wiring-components.md#configuring-a-component-more-than-once)
writes its events to a directory carrying its instance suffix, but **must be
given a `port` of its own**: two servers cannot share one, and the port is
not adjusted automatically. Teardown
closes the writer and terminates the external process.
If setup fails after creating either handle, rollback closes the writer when
present and terminates the partially started process.

## Analysis built-ins

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
@requires_resource("trained_model")
@step("embed_batches", session_type="analysis")
class EmbedBatches(Step):
    def run(self, session):
        batch = next(self.get_dependency("data_manager").data_iter)
        model = self.get_dependency("trained_model").model
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

A `Step` that declares `@requires_resource("layer_inspector")` and
`@requires_resource("trained_model")` reads captures through the resource:

```python
inspector = self.get_dependency("layer_inspector")
model = self.get_dependency("trained_model").model

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

---

**See also:** [transformer blocks](transformer-blocks.md) for the built-in
model building blocks, [infinite samplers](samplers.md), and
[`ModuleResource`](../concepts/module-resource.md) for writing your own
composable model resource.
