# Built-in Components and Samplers

[← Documentation index](README.md) · [Project README](../README.md)

## Built-in components

Importing `training_framework` initializes `training_framework.components`,
which registers all built-ins. Their classes are also importable from
`training_framework.components.builtin`.

### Training built-ins

| Name | Kind | Purpose and dependencies |
|---|---|---|
| `logger` | Hook | Prints `Iteration <current>/<maximum>`; enabled by default |
| `checkpointer` | Hook | Saves complete session checkpoints; enabled by default |
| `ddp` | Resource | Initializes distributed execution and wraps the required `model` resource |
| `data_manager` | Stateful resource | Creates a resumable distributed `DataLoader`; requires `dataset` and `ddp` |
| `optimizer` | Stateful lifecycle hook | Runs AdamW and a warmup/cosine scheduler around each iteration; requires `ddp` and reads `iteration_context["loss"]` |
| `timer` | Lifecycle hook | Reports iteration and elapsed durations; wraps `optimizer` |
| `tensorboard` | Resource | Starts TensorBoard and exposes a `SummaryWriter` |

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
  learning_rate: 0.0003
  weight_decay: 0.01
  warmup_iters: 100

timer:
  call_every: 10

tensorboard:
  host: "127.0.0.1"
  port: 6006
  logdir: ./runs/tensorboard  # optional TensorBoard server log directory
```

`data_manager.data_iter` is available only while the session is active. It
divides the global batch size across ranks and checkpoints delivered-batch
progress. A dataset resource may provide a callable `collate_fn(batch)` method
to control batching:

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

### Analysis built-ins

| Name | Kind | Purpose |
|---|---|---|
| `trained_model` | Resource | Loads the `model` role from the source training checkpoint; enabled by default |
| `logger` | Hook | Prints `Analysis iteration <current>/<maximum>`; enabled by default |

These names live in the analysis registry and therefore do not conflict with
training components that use the same names.

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