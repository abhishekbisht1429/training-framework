# Infinite Samplers

[← Docs](../README.md) · [Project README](../../README.md)

`training_framework.dataloader` provides two samplers that never exhaust, so a
run is bounded by `max_iterations` rather than by the end of the dataset. The
built-in `data_manager` uses `DistributedInfiniteSampler`; use these directly
only when building your own data pipeline.

## `InfiniteSampler`

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

## `DistributedInfiniteSampler`

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

---

**See also:** the [`data_manager` reference](builtin-components.md#data_manager),
which wraps `DistributedInfiniteSampler` and checkpoints its position, and
[Resuming on a different number of GPUs](../guide/04-checkpoints-and-resume.md#resuming-on-a-different-number-of-gpus).
