# Building an Iteration

[← Docs](../README.md) · [Project README](../../README.md)

Most training loops are the same few operations wired differently: take a
batch, run a model, compute a loss, update the weights. The framework ships
each of them as a built-in step, so an ordinary training run needs a model, a
dataset and configuration -- no step code. This page builds a supervised run
from those steps, shows how they are put in order, and then varies it: several
loss terms, a second model, two views of one input.

It assumes you have read [Resources, hooks, and steps](01-resources-hooks-steps.md)
and [Wiring components together](02-wiring-components.md). Every key used here
is listed in the reference: [generic steps](../reference/generic-steps.md) for
`load_batch`, `forward` and `compute`, and [optimization](../reference/optimization.md)
for `optimizer` and the steps after the loss.

## What runs in an iteration

An iteration runs every iteration hook's pre-callback, then every step, then
every post-callback. A training run built from the built-ins looks like this:

```
pre-iteration   forward_context          opens autocast / DDP no_sync
steps           load_batch               batch -> inputs, targets
                forward                  inputs -> logits
                compute#loss             logits, targets -> loss
                backward                 loss -> gradients
                freeze_gradients         (only if configured)
                clip_gradients           (only if configured)
                optimizer_step           gradients -> updated weights
post-iteration  checkpointer, logger, ...  see the updated weights
```

The steps hand values to each other through the iteration context, a set of
named values that is cleared after every iteration. You choose the keys: each
generic step's configuration says which keys it reads and which it writes,
and the session puts a step after the step that writes what it reads. You never
state the order of the steps yourself.

The part after the loss is one unit. Configuring `optimizer` brings in
`backward`, `freeze_gradients`, `clip_gradients`, `optimizer_step` and the
`forward_context` hook on its own; `backward` reads the key `loss`, so it runs
after whichever step writes `loss`.

## A supervised run

Two resources are all the code a run needs: the dataset and the model. The
dataset returns one sample per index; the model is an `nn.Module`
([`ModuleResource`](../concepts/module-resource.md) makes one a resource):

```python
# my_project/components/supervised.py
import torch
from torch import nn

from training_framework.components import ModuleResource, Resource, resource


@resource("toy_dataset")
class ToyDataset(Resource):
    def __init__(self, config=None):
        generator = torch.Generator().manual_seed(0)
        self.inputs = torch.randn(256, 8, generator=generator)
        self.targets = (self.inputs.sum(dim=1) > 0).long()

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


@resource("classifier")
class Classifier(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, inputs):
        return self.net(inputs)
```

The rest is configuration:

```yaml
sessions:
  - session_config:
      rng_seed: 42
      sessions_dir: ./runs
      max_iterations: 20
      components_package: my_project.components
      device: cpu
      show_execution_graph: true

    component_bindings:
      model: classifier          # the roles data_manager and forward use
      dataset: toy_dataset

    classifier: {}               # they have constructors, so they are listed
    toy_dataset: {}
    ddp: {world_size: 1, backend: gloo}

    data_manager:
      batch_size: 32
      num_workers: 0
      pin_memory: false

    load_batch:
      fields: [inputs, targets]  # the sample is a pair: name its two parts

    forward:
      args: [inputs]             # classifier(inputs) ...
      outputs: logits            # ... stored as "logits"

    compute#loss:
      function: cross_entropy    # torch.nn.functional.cross_entropy
      args: [logits, targets]
      outputs: loss              # what backward reads

    optimizer:
      optimizer:
        name: AdamW
        kwargs: {lr: 0.001}
```

Reading it top to bottom:

- **`component_bindings`** says which of your resources fills the `model` and
  `dataset` roles. `ddp` is required because a training `forward` calls the
  model through the DDP wrapper, which synchronises gradients when you later
  run on several processes; with `world_size: 1` it is a single process.
- **`load_batch`** takes the next batch from `data_manager` -- a collated pair
  of tensors -- moves it to the session's device and stores its two parts as
  `inputs` and `targets`. Without `fields` the whole batch would be stored
  under `batch`.
- **`forward`** calls the model with the value of `inputs` and stores what it
  returns as `logits`.
- **`compute#loss`** calls `cross_entropy(logits, targets)` and stores the
  result as `loss`. `compute` is configured as an
  [instance](02-wiring-components.md#configuring-a-component-more-than-once),
  `#loss`, because a run usually has more than one of them; a plain `compute:`
  works as well.
- **`optimizer`** activates the rest of the chain. `backward` reads `loss`.

`show_execution_graph: true` prints what the session made of this. The
`DATAFLOW` section lists every key with its writer and readers, and the steps
come out in that order:

```
DATAFLOW
  inputs: Step.load_batch -> Step.forward
  logits: Step.forward -> Step.compute#loss
  loss: Step.compute#loss -> Step.backward
  targets: Step.load_batch -> Step.compute#loss
...
  +-- STEPS
  |   01. Step.load_batch.run() [requires: Resource.data_manager; writes: inputs, targets]
  |   02. Step.forward.run() [requires: Resource.classifier, Resource.ddp; writes: logits; reads: inputs]
  |   03. Step.compute#loss.run() [writes: loss; reads: logits, targets]
  |   04. Step.backward.run() [requires: Resource.optimizer, Hook.forward_context; reads: loss]
  |   05. Step.freeze_gradients.run() [requires: Resource.optimizer, Step.backward]
  |   06. Step.clip_gradients.run() [requires: Resource.optimizer, Step.freeze_gradients]
  |   07. Step.optimizer_step.run() [requires: Resource.optimizer, Step.clip_gradients]
```

### When the keys do not line up

The keys are checked when the session is built -- in the parent process,
before any worker starts. A typo in `outputs: los` leaves `backward` without
its input:

```
Step.backward reads iteration_context key 'loss', which no step or hook
writes. [...] Keys written in this session: inputs, logits, los, targets.
```

Two steps writing one key are rejected too, since their order would be a
guess:

```
iteration_context key 'loss' is written by both Step.compute#loss and
Step.compute#other. [...]
```

A step may not read and write the same key, and steps that read each other's
keys in a circle are reported as a cycle with the chain. The full set of rules
is in [Ordering by dataflow](02-wiring-components.md#ordering-by-dataflow).

## Variations

Each recipe below shows only the entries that change from the supervised run.

### Several loss terms

A VAE's model returns three values, and its loss is a weighted sum of two
terms. A list in `outputs` unpacks a tuple result; each term is its own
`compute`; the built-in `weighted_sum` adds them:

```yaml
load_batch: {fields: [images, labels]}
forward: {args: [images], outputs: [reconstruction, mu, logvar]}
compute#recon: {function: mse_loss, args: [reconstruction, images], outputs: recon}
compute#kl: {function: my_project.losses.kl_divergence, args: [mu, logvar], outputs: kl}
compute#loss: {function: weighted_sum, kwargs: {recon: recon, kl: kl},
               constants: {weights: {kl: 0.1}}, outputs: loss}
```

`kwargs` maps parameter names to context keys; `constants` passes values
from the configuration as they are. `compute#recon` and `compute#kl` run in
either order -- neither reads what the other writes -- and `compute#loss`
after both.

### A loss that is a module

A name that is an `nn.Module` class, such as `CrossEntropyLoss`, is built
once and then called; `init` gives its constructor arguments:

```yaml
compute#loss: {function: CrossEntropyLoss, init: {label_smoothing: 0.1},
               args: [logits, targets], outputs: loss}
```

Your own functions and classes are given by dotted path. When `compute`
builds a class and when it calls it directly is set out in
[the reference](../reference/generic-steps.md#compute).

### A second model

Distillation calls a student and a teacher. Each is its own `forward`
instance, and a
[per-instance binding](02-wiring-components.md#naming-the-instance-a-component-should-use)
points the teacher's at another resource. The student is the model DDP
wraps and gets gradients; the teacher is called directly, and `no_grad: true`
keeps it out of the graph:

```yaml
component_bindings:
  model: student_model
  forward#teacher: {model: teacher_model}
forward#student: {args: [inputs], outputs: student_logits}
forward#teacher: {args: [inputs], outputs: teacher_logits, no_grad: true}
compute#loss: {function: my_project.losses.distill,
               args: [student_logits, teacher_logits, targets], outputs: loss}
```

The optimizer only ever updates the model DDP wraps, so the teacher's
weights stay as they are.

### Two views in one pass

A contrastive loss compares two augmented views of each sample. Calling the
DDP-wrapped model twice before one `backward` is fragile, so concatenate the
views and make one call. A nested list in `args` passes a list of values --
`torch.cat([view_a, view_b])`:

```yaml
load_batch: {fields: [view_a, view_b]}
compute#views: {function: cat, args: [[view_a, view_b]], outputs: views}
forward: {args: [views], outputs: embeddings}
compute#loss: {function: my_project.losses.nt_xent, args: [embeddings], outputs: loss}
```

### Dict batches and structured outputs

When samples are dicts, `fields` is a mapping from the key you want to the
field of the batch. When the model returns a dict or an object with
attributes, `outputs` is a mapping the same way round:

```yaml
load_batch: {fields: {inputs: pixel_values, targets: label}}
forward: {args: [inputs], outputs: {logits: logits, features: pooler_output}}
```

## In an analysis session

`load_batch` and `forward` also work in an
[analysis session](07-analysis-sessions.md#writing-an-analysis-step).
There `forward` calls `trained_model.model`, without gradients unless
`no_grad: false`, and `load_batch` reads the analysis `data_manager`, which
ends the session after the last batch. `compute` is available in both.

## When to write your own step

The generic steps are stateless calls on context keys. Write a step when you
need something else:

- **State that survives a checkpoint** -- a running centre, a queue of
  negatives, an EMA teacher. Use a
  [`StatefulStep`](01-resources-hooks-steps.md#stateful-components) and
  declare its keys with `@reads` / `@writes`, so it is ordered among the
  generic steps like any of them:

  ```python
  @reads("teacher_logits")
  @writes("centred_logits")
  @step("centre")
  class Centre(StatefulStep):
      def run(self, session, teacher_logits):
          ...
          return teacher_logits - self.centre
  ```

  Declared reads arrive as arguments and declared writes are returned; see
  [Ordering by dataflow](02-wiring-components.md#ordering-by-dataflow).

- **Control flow** -- work that runs only on some iterations or chooses
  between paths.
- **More than one optimizer** -- `optimizer` is one resource over the model's
  parameters, so GANs and similar setups need their own update steps.

A step you write that produces the loss declares `@writes("loss")` and
returns the loss from `run`; `backward` picks it up exactly as it does a
`compute`.

---

**Next:** [Configuration](04-configuration.md) — the full `sessions` YAML
structure and `session_config` fields.

**Reference:** [generic steps](../reference/generic-steps.md) and
[optimization](../reference/optimization.md) list every key used on this page.
