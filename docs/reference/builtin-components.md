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
| `logger` | Hook | Prints `Iteration <current>/<maximum>`, followed by ` \| lr: <lr>` (one value per param group) when `optimizer` is active and ` \| grad_norm: <norm>` once one has been measured; enabled by default |
| `checkpointer` | Hook | Saves complete session checkpoints; enabled by default |
| `ddp` | Resource | Initializes distributed execution and wraps the required `model` resource |
| `data_manager` | Stateful resource | Creates a resumable distributed `DataLoader`; requires `dataset` and `ddp` (analysis sessions use a [separate implementation](#analysis-data_manager)) |
| `optimizer` | Stateful resource | Owns the PyTorch optimizer, its learning-rate schedule and the fp16 gradient scaler; requires `ddp` and activates the [optimization chain](#optimizer) |
| `forward_context` | Lifecycle hook | Opens each iteration on `optimizer`: autocast and DDP `no_sync` around the forward pass; part of the chain |
| `load_batch` | Step | Takes the next batch from `data_manager`, moves it to the device and names its parts; [generic](#generic-steps) |
| `forward` | Step | Calls a model on context keys (through the DDP wrapper for the trained model); [generic](#generic-steps) |
| `compute` | Step | Calls a function or class on context keys (losses, combinations, `torch` ops); [generic](#generic-steps) |
| `backward` | Step | Backpropagates `iteration_context["loss"]`; runs after whichever step writes it |
| `freeze_gradients` | Step | Drops the gradients of matching parameters until an iteration; does nothing unless configured |
| `clip_gradients` | Step | Clips or measures the total gradient norm; does nothing unless configured |
| `optimizer_step` | Step | Steps the optimizer and its schedule on iterations that end an accumulation group |
| `timer` | Lifecycle hook | Reports iteration and elapsed durations |
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

optimizer:                   # backward reads iteration_context["loss"]
  optimizer:
    name: AdamW
    kwargs:
      lr: 0.0003
      weight_decay: 0.01
  lr_scheduler:              # optional
    stages:
      - name: LinearLR
        kwargs:
          start_factor: 0.001
          total_iters: "$stage_iterations"
      - name: CosineAnnealingLR
        kwargs:
          T_max: "$stage_iterations"
    milestones: [100]
  param_groups:              # optional; first match wins
    - match: ["*.bias", "*norm*"]
      kwargs: {weight_decay: 0.0}
  precision: fp32            # fp32 | bf16 | fp16
  accumulate_steps: 1        # iterations per optimizer step

clip_gradients:              # optional
  max_norm: 1.0
  norm_type: 2.0
  track_norm: false          # measure the norm without clipping

freeze_gradients:            # optional
  rules:
    - match: ["head.last_layer.*"]
      until_iteration: 1000

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

`optimizer` is a resource that owns the PyTorch optimizer, its learning-rate
schedule and, for fp16, the gradient scaler. It builds them in `setup` from
the parameters of the DDP-wrapped model, drops them in `teardown`, and
checkpoints their state in between. The work of an iteration is done by
steps, each requiring the one before it:

```
<step writing "loss"> -> backward -> freeze_gradients -> clip_gradients -> optimizer_step
```

Configuring `optimizer` activates `optimizer_step` (a
[companion](../guide/02-wiring-components.md#companions)), which brings in the
rest of the chain and the `forward_context` hook. Every post-iteration hook --
`checkpointer`, `logger`, `timer` -- therefore runs after the update.

- **The loss** is whatever step [declares](../guide/02-wiring-components.md#ordering-by-dataflow)
  writing `iteration_context["loss"]` -- a built-in [`compute`](#generic-steps)
  or your own step with `@writes("loss")`. `backward` reads it, so it runs
  after that step. `backward.loss_key` reads another key.
- **`forward_context`** (hook) opens the iteration before the forward steps
  run: DDP's `no_sync` while gradients are being accumulated, and
  `torch.autocast` for `bf16` / `fp16`.
- **`backward`** leaves autocast, backpropagates the loss (divided by
  `accumulate_steps`, and scaled for fp16), and unscales fp16 gradients on an
  iteration that steps, so every later stage sees true gradients.
- **`freeze_gradients`** sets the gradients of parameters matching a rule to
  None while `iteration <= until_iteration`. The optimizer then skips them
  entirely, weight decay and moment updates included.
- **`clip_gradients`** clips the total norm to `max_norm`, or with
  `track_norm` only measures it. The norm before clipping is exposed as
  `optimizer.grad_norm`, which `logger` prints.
- **`optimizer_step`** steps the optimizer and advances the schedule, then
  clears the gradients.

`freeze_gradients` and `clip_gradients` do nothing until configured. Patterns
are `fnmatch` globs over the model's parameter names (the names of the model
DDP wraps, without a `module.` prefix); a pattern that matches no parameter is
an error.

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

`param_groups` splits the parameters into optimizer parameter groups. Each
entry's `kwargs` override the optimizer's for the parameters its `match`
patterns select; a parameter belongs to the first group that matches it, and
the rest form a final default group.

**Gradient accumulation.** With `accumulate_steps: k`, each iteration is one
micro-batch and the optimizer steps on every k-th iteration and on the final
one. The schedule advances once per optimizer step, so `$max_iterations`
resolves to `ceil(max_iterations / k)` and `milestones` count optimizer steps.
(A schedule replaced by an extension counts only the steps left; see
[Extending a session](../guide/04-checkpoints-and-resume.md).)
Gradients of an unfinished group are not checkpointed: choose a
`checkpoint_every` that is a multiple of `k`, or the first step after a resume
uses fewer micro-batches.

**Precision.** `bf16` runs the forward pass under autocast; `fp16` also scales
the loss with a `GradScaler`, whose state is checkpointed. bf16 is rejected on
a CUDA device that does not support it.

**Custom gradient stages.** Subclass `GradientProcessor` (from
`training_framework.components.builtin`), implement
`process(session, named_parameters)`, require the stage it follows, and bind
`optimizer_step` to it so it runs before the step:

```python
@requires_step("clip_gradients")
@step("scale_gradients")
class ScaleGradients(GradientProcessor):
    def process(self, session, named_parameters):
        for _, parameter in named_parameters:
            parameter.grad.mul_(0.5)
```

```yaml
component_bindings:
  optimizer_step: {clip_gradients: scale_gradients}
scale_gradients: {}
```

`process` runs only on iterations that step, with the parameters that have a
gradient. A stage that ends up after `optimizer_step` -- its binding forgotten
-- is refused on the first iteration, before any step is taken.

If `setup` fails after creating the optimizer or scheduler, its rollback
clears those incomplete runtime handles.

The former `learning_rate`, `weight_decay`, and `warmup_iters` optimizer fields
are no longer accepted. Move optimizer arguments under `optimizer.kwargs` and
describe the warmup/main schedule explicitly as shown above.

`--extend-session` overrides may change `optimizer.optimizer.kwargs` values
(the optimizer class itself cannot change) and may replace `lr_scheduler`
entirely. A changed kwarg does not replace the value a `param_groups` entry
sets for its own group. Overriding `optimizer.optimizer.kwargs.lr` while
leaving `lr_scheduler` unchanged scales the active stage's base learning
rate(s) by the same ratio as the override, keeping its schedule progress; a
multi-stage schedule's not-yet-reached stage is unaffected and runs its own
originally configured base once it activates. Changing `lr_scheduler` itself
(scheduler class, stages, milestones, or `metric_key`) restarts its schedule
from the extension point (`last_epoch` and any per-stage progress reset), since
old scheduler state cannot be assumed compatible with a different schedule
shape; optimizer tensors and step counts are unaffected.
Setting `optimizer.lr_scheduler=null` removes scheduling: training continues
at the learning rate stored in the checkpoint, held fixed (combine with
`optimizer.optimizer.kwargs.lr=<value>` to pin a different rate).
`param_groups`, `precision` and `accumulate_steps` cannot change on extension.
`clip_gradients.*` and `freeze_gradients.*` can, including for a stage that
was never configured.

#### Migrating from the `optimizer` hook (before 0.4.0)

`optimizer` used to be a hook that ran backward, step and schedule itself.

- **Configs:** the `optimizer:` block is unchanged. The step that computes the
  loss declares `@writes("loss")` (or is a built-in `compute`). A session
  where nothing writes `loss` fails when it is built, naming the key.
- **Checkpoints** written by the hook cannot be resumed: restoring one fails
  with "stored as a Hook, but is now registered as a Resource".
- **Code** that looked the hook up (`session.get_all_hooks()`) reads the
  resource instead; its `get_state()` keeps the same keys and adds
  `grad_scaler_state`.

For the operator's view of what `--extend-session` accepts, see
[Extend](../guide/04-checkpoints-and-resume.md#extend); for the contract a
custom component implements to opt in, see
[Opting into extension](../concepts/component-model.md#opting-into-extension).

### Generic steps

`load_batch`, `forward` and `compute` cover the work in front of `backward`
for any kind of training: they are used as many times as needed, through
[instances](../guide/02-wiring-components.md#configuring-a-component-more-than-once)
(`forward#teacher`, `compute#kl`), and each declares the `iteration_context`
keys it reads and writes from its configuration. Those keys are what order
them -- see [Ordering by dataflow](../guide/02-wiring-components.md#ordering-by-dataflow).

**`load_batch`** (training and analysis) takes `next(data_manager.data_iter)`,
moves every tensor in it to the session's device, and names it:

| Key | Default | Meaning |
|---|---|---|
| `key` | `batch` | Store the whole batch under this key |
| `fields` | none | A list unpacks a tuple/list batch by position (`[inputs, targets]`); a mapping picks fields of a dict batch (`{inputs: image, targets: label}`) |
| `non_blocking` | `false` | Passed to `.to()` |

**`forward`** calls the `model` resource -- or, per instance, whatever a
binding names: `component_bindings: {forward#teacher: {model: teacher}}`. In a
training session the model DDP wraps is always called through the wrapper,
so gradients are synchronised. In an analysis session it calls
`trained_model.model`, without gradients unless `no_grad: false`.

**`compute`** calls `function`: a name from `training_framework.functions`
(`weighted_sum`), `torch.nn` (`CrossEntropyLoss`), `torch.nn.functional`
(`mse_loss`) or `torch` (`cat`), searched in that order, or a dotted path to
your own. An `nn.Module` class, or any class given `init` -- even `init: {}`
-- is constructed once and the instance called (a module is moved to the
device); any other class (`builtins.int`) is called directly.

Both take the same call settings:

| Key | Meaning |
|---|---|
| `args` | Positional arguments, as context keys; a nested list passes a list of values (`[[view_a, view_b]]`) |
| `kwargs` | `{parameter: context key}` keyword arguments |
| `constants` | `{parameter: value}` literal keyword arguments |
| `outputs` | Required. A key; a list of keys to unpack a tuple result; or `{key: field}` to pick fields of a dict or attribute result |
| `no_grad` | Run without gradients (default `false`; `true` for analysis `forward`) |
| `method` | `forward` only: call this method of the model instead. It bypasses DDP, so use it for paths without gradients |
| `init` | `compute` only: constructor arguments |

`weighted_sum(weights=..., **terms)` adds named terms, each scaled by its
weight (default 1).

Supervised training:

```yaml
component_bindings: {model: my_model, dataset: my_dataset}
load_batch: {fields: [inputs, targets]}
forward: {args: [inputs], outputs: logits}
compute#loss: {function: CrossEntropyLoss, init: {label_smoothing: 0.1},
               args: [logits, targets], outputs: loss}
optimizer: {optimizer: {name: AdamW, kwargs: {lr: 3.0e-4}}}
```

Several loss terms (a VAE):

```yaml
load_batch: {fields: [images, labels]}
forward: {args: [images], outputs: [reconstruction, mu, logvar]}
compute#recon: {function: mse_loss, args: [reconstruction, images], outputs: recon}
compute#kl: {function: my_project.losses.kl_divergence, args: [mu, logvar], outputs: kl}
compute#loss: {function: weighted_sum, kwargs: {recon: recon, kl: kl},
               constants: {weights: {kl: 0.1}}, outputs: loss}
```

Distillation, with a teacher that gets no gradients:

```yaml
component_bindings: {forward#teacher: {model: teacher_model}}
forward#student: {args: [inputs], outputs: student_logits}
forward#teacher: {args: [inputs], outputs: teacher_logits, no_grad: true}
compute#loss: {function: my_project.losses.distill,
               args: [student_logits, teacher_logits, targets], outputs: loss}
```

Contrastive, with two views in one pass through DDP (calling the same
DDP-wrapped model twice before one backward is fragile, so concatenate):

```yaml
load_batch: {fields: [view_a, view_b]}
compute#views: {function: cat, args: [[view_a, view_b]], outputs: views}
forward: {args: [views], outputs: embeddings}
compute#loss: {function: my_project.losses.nt_xent, args: [embeddings], outputs: loss}
```

State that must survive a checkpoint (a running centre, a queue of negatives)
does not belong in `compute`, which is stateless: write a `StatefulStep` and
declare what it reads and writes. The optimizer is one resource over the
model's parameters, so GANs and other multi-optimizer setups are not covered.

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
`torch.utils.data.default_collate`: tensor samples are stacked, and tuple,
dict and number samples are collated field by field. Keeping the collator on the importable dataset class makes it
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

Each checkpoint is a directory named after its timestamp, written with
`Checkpointer.save_checkpoint(session, path)`; see
[a checkpoint on disk](../guide/04-checkpoints-and-resume.md#a-checkpoint-on-disk).
Because it is an iteration hook, it saves on:

- iterations divisible by `checkpoint_every`;
- the final configured iteration; and
- the first iteration only when it is also the final iteration or
  `checkpoint_first: true`.

A component that needs something another run produced reads it with
`Checkpointer.load_component(path, name, *, with_dependencies=True,
session_type=None)`, which rebuilds only that resource and the instances it
was wired to. The name is
resolved through the *checkpoint's* bindings -- `model` finds whatever that
run bound it to -- and the checkpoint's RNG is not adopted. The caller's RNG
is left exactly as it was, even when the rebuilt components draw from it, so
the caller's seed still decides what comes next. It raises `KeyError` when the checkpoint
has no such resource and `ComponentDependencyError` when several instances
answer; `session_type="training"` also rejects a checkpoint of another kind.
This is how the analysis `trained_model` gets its model.
`Checkpointer.load_component_state(path, name)` returns a component's saved
state without building anything, and `Checkpointer.read_manifest(path)`
describes the checkpoint; see
[reading part of a checkpoint](../guide/04-checkpoints-and-resume.md#reading-part-of-a-checkpoint).

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
