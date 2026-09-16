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
| `data_manager` | Stateful resource | Creates a resumable distributed `DataLoader`; requires `dataset` and `ddp` |
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

The analysis logger lives in the analysis registry. `trained_model` is shared
and can be activated by any session type. Analysis sessions activate it by
default, so an unbound analysis session must provide:

```yaml
trained_model:
  model_checkpoint_path: ./runs/session_.../checkpoints/<checkpoint-name>
```

The path must reference an existing, trusted framework `TrainingSession`
checkpoint whose `model` resource provides `to(device)` and `eval()`.

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
        ...  # capture.input_args, capture.input_kwargs, capture.output
```

`inspector.matched_layer_names` is the full, static set of layers selected
during `setup()`. `inspector.captures` is sparse — keyed only by layers that
actually ran a forward pass — and accumulates every forward pass within the
current iteration, in call order; it is cleared automatically at each
iteration boundary (backed by `session.iteration_context`), so an iteration
that never triggers a forward pass sees an empty mapping rather than stale
data from a previous one.

Captured `input_args`/`input_kwargs`/`output` are live tensor references, not
detached copies — gradients remain enabled by default in analysis sessions,
so a captured tensor still carries its autograd graph if the triggering
forward pass wasn't run under `torch.no_grad()`. This is deliberate: a
gradient/attribution-style analysis needs the live graph. A Step that doesn't
need it should run its forward pass under `torch.no_grad()` itself.

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
