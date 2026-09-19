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

### Resume

```bash
python -m my_project.train --resume-session <checkpoint-path>
```

The engine loads the session in the parent, recovers the saved DDP world size when present, sends session state to workers, and continues from the saved iteration.

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

The effective configuration replaces `config.yaml` and is also preserved in a
timestamped `config_extension_*.yaml` file. The legacy positional maximum
iteration argument remains temporarily available with a deprecation warning.

### Checkpoint safety

Checkpoint loading uses `torch.load(..., weights_only=False)`, which can execute arbitrary code through Python deserialization. Load only checkpoints from trusted sources.

Exact training continuation also depends on application state. Persist model, optimizer, scheduler, scaler, sampler, and any data-pipeline state that affects the next batch.
