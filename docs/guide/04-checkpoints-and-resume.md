# Checkpoints, Resume, and Extend

[← Docs](../README.md) · [Project README](../../README.md)

A checkpoint captures a running session so it can be continued later — on the
same machine or a different one, with the same hyperparameters or changed ones.
This page covers what a checkpoint holds, the two ways to continue from one
(`--resume-session` and `--extend-session`), and what may safely change in
each.

To turn checkpointing on, configure the built-in `checkpointer` hook; its
options are in the
[built-in component reference](../reference/builtin-components.md#checkpointer).

## Stored session state

`TrainingSession.get_state()` includes:

- current iteration;
- immutable `SessionConfig`;
- component constructor arguments;
- state from `Stateful` resources, hooks, and steps;
- `session_context`;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state; and
- the CUDA RNG stream of the device the writing process was using.

Each component's state holds only what that component owns. A `ModuleResource`
excludes every tensor reachable from a resource it attached, so weights shared
between components are stored once, by the component that created them, and
restored as one shared instance. A component held *privately* by another --
constructed by it, never registered -- is not shared, so its weights are
checkpointed inside its owner.

Capturing session state checks this rather than assuming it: if two components
would save the same tensor, the capture is rejected and the error names both
components and the tensor.

Transient infrastructure, such as the selected device, iteration context, error pipe, and progress beacon, is recreated in each worker.

## What a checkpoint does not record

A checkpoint records what was learned and how far it got, never the machine it
ran on. The DDP world size, master address and port are resolved on every
launch and injected into the `ddp` component as the workers are built, so a
stored value is only a default. The CUDA device a rank uses is decided the
same way, and each rank pins it before any component is constructed.

The state that *is* recorded is kept in a form that does not depend on the
number of processes. Only one CUDA RNG stream is stored, not one per visible
device, and the data sampler records how far the epoch got across all ranks
rather than how far one rank got. Both are restored onto whatever topology the
resuming launch resolved.


## Resume

```bash
python -m my_project.train --resume-session <checkpoint-path>
```

The engine loads the session in the parent, resolves the launch topology, sends session state to workers, and continues from the saved iteration.

The only overrides `--resume-session` accepts are the launch topology —
`ddp.world_size`, `ddp.master_addr` and `ddp.master_port` — because those
describe the machine rather than the session:

```bash
python -m my_project.train \
  --resume-session ./runs/session_.../checkpoints/<checkpoint-name> \
  --override ddp.world_size=4
```

Any other override is rejected with a pointer to `--extend-session`, rather
than being silently ignored. See
[Resuming on a different number of GPUs](#resuming-on-a-different-number-of-gpus).

## Resuming on a different number of GPUs

A run trained on eight GPUs resumes on four with the same command. If the
stored world size no longer fits the visible CUDA devices, the launch warns
and falls back to the device count. Choose a size explicitly with
`--override ddp.world_size=N`, which is the one kind of override
`--resume-session` accepts.

`data_manager.batch_size` is the *global* batch size, and each rank takes
`batch_size // world_size` of it, so the global batch, the number of
optimizer steps per epoch, and the learning-rate schedule all mean the same
thing before and after a resize. `batch_size` must stay divisible by the new
world size.

Two details are worth knowing:

- **The sampler position rounds up.** Every rank has to resume at the same
  offset into its own slice, so the position moves to a multiple of the new
  world size. Nothing is delivered twice; up to `world_size - 1` samples from
  the tail of that epoch are skipped instead. The permutation is drawn from
  the seed and the epoch alone, so which samples an epoch contains does not
  change — only the padding at its end.
- **Bit-exact continuation is not preserved across a resize**, and is not
  available across different GPU models either. What is preserved is
  determinism: the same checkpoint, world size and seed produce the same run.

Checkpoints written before this behaviour existed still load. Their per-device
RNG list restores the writing rank's own stream, and their per-rank sampler
position is converted to the topology-independent form.

## Extend

To restore one training checkpoint and change extension-safe hyperparameters,
use overrides relative to that session (without a `sessions[0]` prefix):

```bash
python -m my_project.train \
  --extend-session ./runs/session_.../checkpoints/<checkpoint-name> \
  --override \
  session_config.max_iterations=5000 \
  optimizer.optimizer.kwargs.lr=0.0001 \
  logger.log_every=25
```

The built-in mutable settings are `session_config.max_iterations`, optimizer
constructor values under `optimizer.optimizer.kwargs`, `logger.log_every`,
and `checkpointer.checkpoint_every` / `checkpoint_first`. Optimizer state such
as momentum buffers and step counters is retained; only explicitly overridden
parameter-group values are replaced. The optimizer class cannot change, and
existing optimizer kwargs cannot be removed. Model, DDP, data-manager,
component-binding, and other session changes are rejected unless a custom
component explicitly opts into extension.

The launch-topology keys `ddp.world_size`, `ddp.master_addr` and
`ddp.master_port` may be given alongside these. They are not session
configuration, so they bypass the extension rules and are applied when the
workers are built — an extend can resize the run and change hyperparameters
in one command.

`optimizer.lr_scheduler` may be replaced entirely; the new schedule restarts
from the extension point. To drop scheduling and continue at a fixed learning
rate, set it to `null`:

```bash
python -m my_project.train \
  --extend-session ./runs/session_.../checkpoints/<checkpoint-name> \
  --override \
  session_config.max_iterations=5000 \
  optimizer.lr_scheduler=null
```

Without a scheduler, training continues at the learning rate stored in the
checkpoint (the last value the scheduler set) and it stays fixed. Add
`optimizer.optimizer.kwargs.lr=<value>` to pin a different fixed rate. Keys
must be removed with `=null`; the `~key` deletion syntax is not supported.

The positional form `--extend-session CHECKPOINT NEW_MAX_ITERATIONS` remains
available with a deprecation warning.

Rank zero rewrites `config.yaml` with the effective configuration and also
creates `config_extension_<timestamp>.yaml` in the session directory.

The session is restored as in the resume operation, then safe overrides are
applied before worker state is captured. Extension is training-specific; the
checkpoint must contain a `TrainingSession`. Components reject configuration changes by default and must
implement `ExtendableComponent` to opt in — see
[Opting into extension](../concepts/component-model.md#opting-into-extension)
for the author-side contract, and the
[`optimizer` reference](../reference/builtin-components.md#optimizer) for
exactly how the built-in optimizer rescales its learning rate.

## Checkpoint safety

Checkpoint loading uses `torch.load(..., weights_only=False)`, which can execute arbitrary code through Python deserialization. Load only checkpoints from trusted sources.

Exact training continuation also depends on application state. Persist model, optimizer, scheduler, scaler, sampler, and any data-pipeline state that affects the next batch.

---

**Next:** [Distributed training](05-distributed-training.md) — running a
session across several GPUs.
