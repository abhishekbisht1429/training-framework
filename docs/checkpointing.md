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
- CUDA RNG state.

Transient infrastructure, such as the selected device, iteration context, manager pipe, and heartbeat timer, is recreated in each worker.

### Resume

```bash
python -m my_project.train --resume-session <checkpoint-path>
```

The engine loads the session in the parent, recovers the saved DDP world size when present, sends session state to workers, and continues from the saved iteration.

### Extend

```bash
python -m my_project.train --extend-session <checkpoint-path> <new-max-iterations>
```

The session is restored as in the resume operation, then each worker receives the new maximum iteration count before training starts.
Extension is training-specific; the checkpoint must contain a `TrainingSession`.

### Checkpoint safety

Checkpoint loading uses `torch.load(..., weights_only=False)`, which can execute arbitrary code through Python deserialization. Load only checkpoints from trusted sources.

Exact training continuation also depends on application state. Persist model, optimizer, scheduler, scaler, sampler, and any data-pipeline state that affects the next batch.