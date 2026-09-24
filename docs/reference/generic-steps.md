# Generic Steps

[← Docs](../README.md) · [Project README](../../README.md)

Reference for `load_batch`, `forward` and `compute`: every configuration key,
how inputs are taken from and results put into `iteration_context`, and the
errors each one raises. For a walkthrough of putting them together into a
training iteration, with recipes, see
[Building an iteration](../guide/03-building-an-iteration.md).

| Step | Session types | Requires | Reads | Writes |
|---|---|---|---|---|
| [`load_batch`](#load_batch) | training, analysis | `data_manager` | nothing | `key`, or the keys in `fields` |
| [`forward`](#forward) | training | `model`, `ddp` | the keys in `args` / `kwargs` | the keys in `outputs` |
| [`forward`](#forward-in-an-analysis-session) | analysis | `trained_model` | the keys in `args` / `kwargs` | the keys in `outputs` |
| [`compute`](#compute) | any | nothing | the keys in `args` / `kwargs` | the keys in `outputs` |

Each step declares the keys it reads and writes from its own configuration,
and those declarations order the steps: a step runs after the step that writes
what it reads (see [Ordering by
dataflow](../guide/02-wiring-components.md#ordering-by-dataflow)). Each can be
configured
[more than once](../guide/02-wiring-components.md#configuring-a-component-more-than-once)
-- `forward#teacher`, `compute#kl` -- and every instance is ordered by its own
keys. All configuration is checked when the session is built, before any
worker starts; the errors below that mention the batch or the result are
raised when the step runs.

## `load_batch`

Takes `next(data_manager.data_iter)`, moves every tensor in it to the
session's device -- through nested lists, tuples, named tuples and dicts;
anything else is left as it is -- and stores it.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `key` | string | `batch` | Store the whole batch under this key. Ignored when `fields` is given |
| `fields` | list or mapping | none | Name the parts of the batch instead; see below |
| `non_blocking` | bool | `false` | Passed to `.to()` |

`fields` takes one of two forms:

| Form | Batch | Effect |
|---|---|---|
| A list of keys, `[inputs, targets]` | A tuple or list of the same length | Element *i* is stored under the *i*-th key |
| A mapping `{context key: batch field}`, `{inputs: image, targets: label}` | A dict | `batch["image"]` is stored under `inputs`, and so on |

A list must not repeat a key; neither form may be empty. A batch of the wrong
shape -- a dict for a list, a tuple of another length, a missing field -- is
an error naming what the batch actually is.

In an analysis session `load_batch` reads the analysis `data_manager`, which
delivers each sample once: the `StopIteration` after the last batch ends the
session cleanly.

## Call settings

`forward` and `compute` both call something, and share these keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `outputs` | string, list or mapping | required | Where the result goes; see below |
| `args` | list | `[]` | Positional arguments, each a context key. A nested list passes one list of values: `[[view_a, view_b]]` calls `f([view_a, view_b])` |
| `kwargs` | mapping | `{}` | `{parameter: context key}` keyword arguments |
| `constants` | mapping | `{}` | `{parameter: value}` keyword arguments taken literally from the configuration |
| `no_grad` | bool | `false` (`true` for the analysis `forward`) | Run the call under `torch.no_grad()` |

The keys in `args` and `kwargs` are what the step reads; the keys in
`outputs` are what it writes. A parameter may not be given by both `kwargs`
and `constants`, and parameter names must be non-empty strings.

`outputs` takes one of three forms:

| Form | Result | Effect |
|---|---|---|
| A key, `logits` | anything | The result is stored under `logits` |
| A list of keys, `[reconstruction, mu, logvar]` | A tuple or list of the same length | Element *i* is stored under the *i*-th key |
| A mapping `{context key: field}`, `{logits: logits, hidden: last_hidden_state}` | A dict, or an object with those attributes | `result["last_hidden_state"]` (or `result.last_hidden_state`) is stored under `hidden`, and so on |

## `forward`

Calls a model on context keys.

| Key | Type | Default | Meaning |
|---|---|---|---|
| [call settings](#call-settings) | | | |
| `method` | string | none | Call this method of the model instead of the model itself |

The model is the `model` resource or, for one instance, whatever its
[binding](../guide/02-wiring-components.md#naming-the-instance-a-component-should-use)
names:

```yaml
component_bindings:
  forward#teacher: {model: teacher_model}
```

When the model is the one `ddp` wraps, `forward` calls it **through the DDP
wrapper**, so its gradients are synchronised across ranks. Any other model --
a teacher, a frozen encoder -- is called directly. `method` always calls the
model directly, bypassing DDP, so use it only for paths without gradients
(`method: encode` with `no_grad: true`).

Calling the DDP-wrapped model more than once before one `backward` is
fragile under DDP; to pass two inputs through it, concatenate them first (see
the [contrastive recipe](../guide/03-building-an-iteration.md#two-views-in-one-pass)).

### `forward` in an analysis session

The analysis `forward` calls `trained_model.model` (no `ddp` involved) and
takes the same keys, including `method`, but `no_grad` defaults to `true`.
Set `no_grad: false` for gradient-based attributions.

## `compute`

Calls a function, or an instance of a class, on context keys.

| Key | Type | Default | Meaning |
|---|---|---|---|
| [call settings](#call-settings) | | | |
| `function` | string | required | What to call; see below |
| `init` | mapping | none | Constructor arguments; see below |

**Finding `function`.** A name without a dot is looked up in these modules, in
order, and the first match wins:

1. `training_framework.functions` -- `weighted_sum`
2. `torch.nn` -- `CrossEntropyLoss`, `MSELoss`, ...
3. `torch.nn.functional` -- `cross_entropy`, `mse_loss`, ...
4. `torch` -- `cat`, `stack`, ...

A name with a dot is imported as a dotted path:
`my_project.losses.kl_divergence`, `torch.nn.functional.mse_loss`. A name
found nowhere is an error listing the modules searched.

**Classes.** What happens to a class depends on `init`:

| `function` is | `init` | Result |
|---|---|---|
| a function | absent | Called on every run |
| a function | given | Error: `init` is for a class |
| an `nn.Module` class | absent or given | Constructed once with `init` (or no arguments); the instance is moved to the session's device and called on every run |
| any other class | given, even `{}` | Constructed once with `init`; the instance is called on every run |
| any other class | absent | The class itself is called on every run, like a function (`builtins.int`) |

`init: {}` and no `init` therefore differ for a class that is not a module.
Constructor arguments that do not fit are reported when the session is built.

`compute` is meant for **stateless** callables: a function, or a class whose
output depends only on its inputs (`nn.CrossEntropyLoss`, `nn.MSELoss`). A
class instance is built once and called on every run, so anything it changes
on itself -- a counter, a running statistic, a module buffer such as a
`BatchNorm` running mean -- carries over from one iteration to the next. None
of it is checkpointed: only `compute`'s configuration is, so a restore rebuilds
the instance from `init`, that state starts over, and a resumed run quietly
differs from an uninterrupted one. Parameters of a module built here are not
trained either: the optimizer takes its parameters from the `model` resource,
and DDP does not synchronise them.

Put such state where the framework manages it:

- **learnable weights** (a projection head, a learned temperature) in the
  model, or in a [`ModuleResource`](../concepts/module-resource.md) attached to
  it, so the optimizer updates them, DDP synchronises them and the checkpoint
  saves them;
- **state that is not learned** (a running centre, a queue of negatives) in a
  [`StatefulStep`](../guide/01-resources-hooks-steps.md#stateful-components)
  that declares what it reads and writes.

### `training_framework.functions`

Small functions for `compute` to call by name:

| Function | Meaning |
|---|---|
| `weighted_sum(*, weights=None, **terms)` | Sum the keyword `terms`, each multiplied by `weights[name]` (default 1). At least one term is required, and a weight for a name that is not a term is an error |

```yaml
compute#loss:
  function: weighted_sum
  kwargs: {recon: recon, kl: kl}          # terms, from the context
  constants: {weights: {kl: 0.1}}         # recon is weighted 1
  outputs: loss
```

---

**See also:** [optimization](optimization.md) for `backward` and the steps
after it, and [built-in components](builtin-components.md) for everything else
the framework ships.
