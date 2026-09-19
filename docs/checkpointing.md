# Checkpointing, Resume, and Extension

[← Documentation index](README.md) · [Project README](../README.md)

## Checkpointing, resume, and extension

### Built-in checkpointer

Add the built-in `checkpointer` hook to YAML:

```yaml
checkpointer:
  checkpoint_every: 100
  checkpoints_dir: ./runs/checkpoints  # optional
  checkpoint_first: false              # optional; defaults to false
```

If `checkpoints_dir` is omitted, checkpoints are written to a `checkpoints` directory under the session directory.

The checkpointer uses `torch.save(session, path)`. Because it is an iteration
hook, it saves on:

- iterations divisible by `checkpoint_every`;
- the final configured iteration; and
- the first iteration only when it is also the final iteration or
  `checkpoint_first: true`.

### Stored session state

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

### What a checkpoint does not record

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

### Resume

```bash
python -m my_project.train --resume-session <checkpoint-path>
```

The engine loads the session in the parent, resolves the launch topology, sends session state to workers, and continues from the saved iteration.

### Resuming on a different number of GPUs

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

### Extend

```bash
python -m my_project.train \
  --extend-session <checkpoint-path> \
  --override \
  session_config.max_iterations=5000 \
  optimizer.optimizer.kwargs.lr=0.0001
```

The session is restored as in the resume operation, then safe overrides are
applied before worker state is captured. Extension is training-specific; the
checkpoint must contain a `TrainingSession`. Components reject changes by
default and must implement `ExtendableComponent` to opt in. The built-in
optimizer, logger, and checkpointer allow their documented training/cadence
settings while model, DDP, and data-manager configuration remains immutable.

`ddp.world_size`, `ddp.master_addr` and `ddp.master_port` are the exception,
because they are not session configuration at all: they describe the launch,
and are applied when the workers are built rather than through the extension
machinery. An extend may therefore resize the run at the same time as it
changes hyperparameters.

The effective configuration replaces `config.yaml` and is also preserved in a
timestamped `config_extension_*.yaml` file. The legacy positional maximum
iteration argument remains temporarily available with a deprecation warning.

### Checkpoint safety

Checkpoint loading uses `torch.load(..., weights_only=False)`, which can execute arbitrary code through Python deserialization. Load only checkpoints from trusted sources.

Exact training continuation also depends on application state. Persist model, optimizer, scheduler, scaler, sampler, and any data-pipeline state that affects the next batch.
