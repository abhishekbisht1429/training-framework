# Current Behavior and Limitations

[← Docs](../README.md) · [Project README](../../README.md)

Known constraints of the current implementation, grouped by the area they
affect. These are deliberate scope decisions or accepted trade-offs, not open
bugs; each says what the behavior is so you can design around it.


## Processes and supervision

### Configured sessions start as one concurrent batch

Every `sessions[]` entry contributes worker wrappers to the run; the engine does not provide sequencing or dependency ordering between sessions.

### Spawn requires importable and serializable definitions

Define worker targets and component classes at module scope. Constructor arguments, state returned by `get_state()`, and checkpointed session-context values must be serializable.

### Progress is detected between framework stages

A single long-running component call can exceed the deadline without marking progress. Call `session.send_heartbeat(...)` periodically inside long loops — it is a cheap shared-memory write, so once per batch is fine — or set `--heartbeat-timeout` above the longest uninterrupted setup, hook, step, or teardown operation:

```python
def run(self, session):
    for batch_index, batch in enumerate(validation_loader):
        session.send_heartbeat(f"validate batch {batch_index}")
        ...
```

### Graceful stopping occurs between iterations

Non-DDP workers use their local stop event. DDP workers make a rank-wide stop decision before each iteration, so every rank may complete one final synchronized iteration when a request races with that decision. Work already inside a component or collective must finish or wait until the join timeout causes termination.

## Distributed training

### DDP is currently single-node oriented

A rank runs on CUDA ordinal `rank % <visible devices>`, and the engine spawns every rank itself, so all of them share one `CUDA_VISIBLE_DEVICES` and take the first `world_size` entries of it. Multi-node execution, where a rank's global index and its local device index genuinely diverge, is not exposed by the framework.

### The built-in DDP resource requires a compatible model resource

The `model` role must resolve to a module accepted by PyTorch DDP. Distributed forward passes should use `session.get_resource("ddp").wrapped_model` while the session is active.

### Only the DDP root's module tree is gradient-synchronised

`ddp` wraps the single `model` resource, so parameters owned by an attached `ModuleResource` are synchronised only while that component is reachable from the `model` root. A `ModuleResource` that is active but not reachable from `model` is also invisible to `OptimizerHook`, which collects parameters from the wrapped model, so its weights are never updated and never all-reduced while `Checkpointer` still saves them. Nothing raises: the same shape is legitimate weight sharing between two models.

### Rank-zero-only work is opt-out

Ranks greater than zero build every configured component except those marked `@rank_zero_only` or named in `ddp.rank_zero_components`, so a component that should run once per run rather than once per rank has to say so. Nothing is inferred from the dependency graph: a component that takes part in the collectives without declaring `ddp` as a prerequisite is still built everywhere, and one that writes a file per run is duplicated across ranks until it is declared.

## Components and registration

### Model shape is fixed once a component is constructed

Components are wired to each other as they are built, so `--extend-session` can change a component's configuration but not the set of parameters it owns.

### Components are constructed in the parent process

The engine builds the session before spawning workers, so a model's weights are allocated in the parent and shipped to each rank inside the session state. Every rank therefore starts from identical weights, at the cost of one model's memory in the parent.

### A component that declares dependencies cannot be constructed by hand

Wiring happens during construction, so such a component must be activated by the session -- through configuration or `session.activate_component(name, config)`. Replacing an already-constructed component is not supported.

### Component registration is global per interpreter

Resource, hook, and step names share one namespace within each shared or session-specific scope. Duplicate names in one scope fail; a matching scoped component overrides a shared component with the same name. Test suites that reset registration must account for Python's module import cache before expecting decorators to run again.

## Checkpoints and data

### Checkpoint files are trusted-code artifacts

The built-in loader uses unrestricted Python deserialization. Never load an untrusted checkpoint.

### Exact data-pipeline replay is application-dependent

DataLoader prefetching can move a sampler's issued position ahead of consumed batches. Persist and restore committed batch progress when exact continuation is required.

### Resuming on a different world size is deterministic, not bit-exact

Every rank must resume at the same offset into its own slice, so the sampler position rounds up to a multiple of the new world size: nothing is delivered twice, but up to `world_size - 1` samples from the tail of that epoch are skipped. One CUDA RNG stream is restored onto every rank rather than a per-rank stream, and `batch_size` must stay divisible by the new world size. The same checkpoint, world size and seed still reproduce the same run.

### The data order does not follow `rng_seed`

`DataManager` builds its sampler without a seed, so the shuffle always uses the sampler default of `0`. Two runs that differ only in `rng_seed` see the same data order.

### Legacy analysis checkpoint-path ownership is unsupported

Analysis configurations must place the source checkpoint at `trained_model.model_checkpoint_path`. The removed top-level `model_checkpoint_path` entry and `AnalysisSession.model_checkpoint_path` property are not compatibility aliases. Analysis-session checkpoints that rely only on the removed session-level state must be recreated or explicitly migrated before loading.

## Environment

### An unavailable CUDA device currently falls back to CPU

Validate the final `session.device` in application code when silent fallback is undesirable.

### TensorBoard is an external process

Starting it requires an available executable and port, and including it in DDP parallel components would start one server per retained rank.
