# Optimization

[← Docs](../README.md) · [Project README](../../README.md)

Reference for the `optimizer` resource and the steps that turn a loss into a
parameter update: their configuration keys, gradient accumulation, mixed
precision, custom gradient stages, what `--extend-session` may change, and the
migration from the pre-0.4.0 hook. For how these fit into a whole iteration
with the steps in front of them, see
[Building an iteration](../guide/03-building-an-iteration.md).

## The chain

`optimizer` owns the PyTorch optimizer, its learning-rate schedule and, for
fp16, the gradient scaler. The work of an iteration is done by steps, each
requiring the one before it:

```
<step writing "loss"> -> backward -> freeze_gradients -> clip_gradients -> optimizer_step
```

Configuring `optimizer` activates `optimizer_step` (a
[companion](../guide/02-wiring-components.md#companions)), which brings in the
rest of the chain and the `forward_context` hook. You never list `backward`,
`freeze_gradients`, `clip_gradients` or `optimizer_step` yourself unless you
want to configure one. Every post-iteration hook -- `checkpointer`, `logger`,
`timer` -- runs after the update.

| Component | Kind | Does | Configuration |
|---|---|---|---|
| `optimizer` | Stateful resource | Builds the optimizer, schedule and scaler in `setup` from the parameters of the DDP-wrapped model; checkpoints their state | [`optimizer`](#optimizer) |
| `forward_context` | Lifecycle hook | Opens each iteration before any step: DDP `no_sync` while gradients accumulate, `torch.autocast` for `bf16` / `fp16` | none |
| `backward` | Step | Leaves autocast and backpropagates the loss (averaged over its accumulation group, scaled for fp16); unscales fp16 gradients on an iteration that steps | [`backward`](#backward) |
| `freeze_gradients` | Step | Sets matching parameters' gradients to None until an iteration | [`freeze_gradients`](#freeze_gradients) |
| `clip_gradients` | Step | Clips, or only measures, the total gradient norm | [`clip_gradients`](#clip_gradients) |
| `optimizer_step` | Step | Steps the optimizer and the schedule, then clears the gradients | none |

`freeze_gradients` and `clip_gradients` run, and edit gradients, only on
iterations that step (see [gradient accumulation](#gradient-accumulation)),
so they always see complete, unscaled gradients.

## `optimizer`

```yaml
optimizer:
  optimizer:                 # required
    name: AdamW
    kwargs:
      lr: 0.0003
      weight_decay: 0.01
  lr_scheduler:              # optional; constant lr when omitted
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
```

| Key | Default | Meaning |
|---|---|---|
| `optimizer.name` | required | A class in `torch.optim` |
| `optimizer.kwargs` | `{}` | Its constructor arguments (not `params`) |
| `lr_scheduler` | none | See [learning-rate schedule](#learning-rate-schedule) |
| `param_groups` | none | A list of `{match, kwargs}`: parameters matching a `match` pattern get the entry's `kwargs` on top of the optimizer's. A parameter belongs to the first entry that matches it; the rest form a final default group |
| `precision` | `fp32` | `fp32`, `bf16` (autocast) or `fp16` (autocast and a `GradScaler`, whose state is checkpointed). Under `fp16` a step whose gradients overflow is skipped, and the schedule does not advance for it. `bf16` is rejected on a CUDA device that does not support it |
| `accumulate_steps` | `1` | Iterations per optimizer step; see [gradient accumulation](#gradient-accumulation) |

The former `learning_rate`, `weight_decay` and `warmup_iters` fields are
rejected: move optimizer arguments under `optimizer.kwargs` and describe the
warmup/main schedule explicitly, as above.

**Patterns.** `match` patterns here and in `freeze_gradients` are `fnmatch`
globs over the model's parameter names -- the names of the model DDP wraps,
without a `module.` prefix. A pattern that matches no parameter is an error.

### Learning-rate schedule

| Key | Default | Meaning |
|---|---|---|
| `stages` | required | A non-empty list of `{name, kwargs}`; `name` is a class in `torch.optim.lr_scheduler`, `kwargs` its arguments (not `optimizer`) |
| `milestones` | `[]` | One boundary between each pair of stages: positive, strictly increasing optimizer-step counts. Several stages are combined with `SequentialLR` |
| `metric_key` | none | Single-stage schedules only: pass `iteration_context[metric_key]` to `scheduler.step(...)` (e.g. `ReduceLROnPlateau`). `optimizer_step` then reads that key, so some step or hook [must write it](../guide/02-wiring-components.md#ordering-by-dataflow) |

Scheduler kwargs may use the exact string values `$max_iterations` and
`$stage_iterations`, resolved when the session starts. The latter is the
distance between the stage's surrounding milestones (or the start/end of the
session). Both count optimizer steps, not iterations.

A schedule runs for the optimizer steps it was built for and then holds its
final learning rate, so a cosine never wraps and `OneCycleLR` never overruns.
Metric-driven schedules are never held.

## `backward`

| Key | Default | Meaning |
|---|---|---|
| `loss_key` | `loss` | The `iteration_context` key holding the scalar loss |

`backward` declares that it reads `loss_key`, so it runs after whichever step
declares writing it -- a built-in [`compute`](generic-steps.md#compute) with
`outputs: loss`, or your own step with `@writes("loss")`. A session in which
nothing writes that key fails when it is built, naming the key.

## `freeze_gradients`

```yaml
freeze_gradients:
  rules:
    - match: ["head.last_layer.*"]
      until_iteration: 1000
```

| Key | Default | Meaning |
|---|---|---|
| `rules` | `[]` (does nothing) | A list of `{match, until_iteration}` |
| `rules[].match` | required | Parameter-name [patterns](#optimizer) |
| `rules[].until_iteration` | required | Non-negative integer; the gradients are dropped while `iteration <= until_iteration` |

A parameter whose gradient is None is skipped by the optimizer entirely,
weight decay and moment updates included.

## `clip_gradients`

```yaml
clip_gradients:
  max_norm: 1.0
  norm_type: 2.0
  track_norm: false
```

| Key | Default | Meaning |
|---|---|---|
| `max_norm` | none | Clip the total norm to this finite positive value |
| `norm_type` | `2.0` | The norm's order; `.inf` for the largest absolute gradient (NaN is rejected) |
| `track_norm` | `false` | Measure the norm without clipping |

With neither `max_norm` nor `track_norm` it does nothing. The norm before
clipping is exposed as `optimizer.grad_norm`, which `logger` prints: the norm
of the step the last iteration took, or none if that iteration did not step
(accumulating).

## Gradient accumulation

With `accumulate_steps: k`, each iteration is one micro-batch and the
optimizer steps on every k-th iteration and on the final one. `backward`
divides the loss by the size of its group -- k, except for a shorter last
group when `max_iterations` is not a multiple of k -- so every step applies
the mean over its micro-batches. DDP skips its gradient all-reduce
(`no_sync`) on the iterations in between. The schedule advances once per optimizer step, so
`$max_iterations` resolves to `ceil(max_iterations / k)` and `milestones`
count optimizer steps.

Gradients of an unfinished group are not checkpointed: choose a
`checkpoint_every` that is a multiple of `k`, or the first step after a resume
uses fewer micro-batches.

An iteration that raises can be run again (the session rolls its counter
back). If it had already run `backward`, its gradients are discarded first
when it starts a group -- always so with `k = 1`; in the middle of a group they
cannot be told apart from the group's earlier micro-batches, so running it
again is refused: resume from the last checkpoint.

## Custom gradient stages

Subclass `GradientProcessor` (from `training_framework.components.builtin`),
implement `process(session, named_parameters)`, require the stage it follows,
and bind `optimizer_step` to it so it runs before the step:

```python
from training_framework.components import requires_step, step
from training_framework.components.builtin import GradientProcessor


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
gradient, and edits `parameter.grad` in place. A stage that ends up after
`optimizer_step` -- its binding forgotten -- is refused on the first
iteration, before any step is taken. A subclass with a `config_schema` can be
reconfigured by `--extend-session`.

## Extending a session

`--extend-session` overrides may change:

- **`optimizer.optimizer.kwargs` values** (the optimizer class cannot change).
  A changed kwarg does not replace the value a `param_groups` entry sets for
  its own group. Overriding `optimizer.optimizer.kwargs.lr` while leaving
  `lr_scheduler` unchanged scales the active stage's base learning rate(s) by
  the same ratio, keeping its progress; a stage not reached yet runs with its
  own originally configured base once it activates. A group whose
  `param_groups` entry sets its own `lr` keeps its lr and its schedule as
  they were.
- **`lr_scheduler`**, replaced entirely. A changed schedule (class, stages,
  milestones or `metric_key`) restarts from the extension point and runs over
  the optimizer steps left; optimizer tensors and step counts are unaffected.
  `optimizer.lr_scheduler=null` removes scheduling: training continues at the
  learning rate stored in the checkpoint, held fixed (combine with
  `optimizer.optimizer.kwargs.lr=<value>` to pin a different one).
- **`clip_gradients.*` and `freeze_gradients.*`**, including for a stage that
  was never configured.

`param_groups`, `precision` and `accumulate_steps` cannot change. For the
operator's view of what `--extend-session` accepts, see
[Extend](../guide/05-checkpoints-and-resume.md#extend); for the contract a
custom component implements to opt in, see
[Opting into extension](../concepts/component-model.md#opting-into-extension).

If `setup` fails after creating the optimizer or scheduler, its rollback
clears those incomplete runtime handles.

## Migrating from the `optimizer` hook (before 0.4.0)

`optimizer` used to be a hook that ran backward, step and schedule itself.

- **Configs:** the `optimizer:` block is unchanged. The step that computes the
  loss declares `@writes("loss")` (or is a built-in `compute`). A session
  where nothing writes `loss` fails when it is built, naming the key.
- **Checkpoints** written by the hook cannot be resumed: restoring one fails
  with "stored as a Hook, but is now registered as a Resource".
- **Code** that looked the hook up (`session.get_all_hooks()`) reads the
  resource instead; its `get_state()` keeps the same keys and adds
  `grad_scaler_state`.

---

**See also:** [generic steps](generic-steps.md) for the steps in front of
`backward`, and [built-in components](builtin-components.md) for everything
else the framework ships.
