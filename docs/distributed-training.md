# Distributed Training

[← Documentation index](README.md) · [Project README](../README.md)

## Distributed training with DDP

Adding a top-level `ddp` resource makes the engine create `world_size` worker processes.

```yaml
sessions:
  - session_config:
      rng_seed: 42
      sessions_dir: ./runs
      max_iterations: 1000
      components_package: my_project.components
      device: cpu

    ddp:
      world_size: 4
      backend: nccl
      master_addr: "127.0.0.1"
      master_port: "12355"
      parallel_components:
        - model
        - train

    model: {}
    train: {}

    logger:
      log_every: 10

    checkpointer:
      checkpoint_every: 100
```

`master_port` should be a string because it is assigned to the `MASTER_PORT` environment variable.

### The launch decides the topology

`world_size`, `master_addr` and `master_port` describe the machine a run is
launched on, not the run itself, so they are resolved once per launch and
injected into the `ddp` component as the workers are built. A checkpoint never
imposes its own.

Each is taken from the first of these that provides it:

1. `--override ddp.<key>=<value>`;
2. the environment (`WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), which
   outranks a stored value only when that value came from a checkpoint — a
   config file is the launch's own statement and keeps precedence over an
   ambient variable;
3. the configuration, whether from the file or the checkpoint;
4. a default: the visible CUDA device count when resuming, and a free local
   port.

A new session must still state its `world_size`. Asking for more ranks than
there are visible CUDA devices is an error when the run is new, and a warning
with a fallback to the device count when it is resumed — so a checkpoint
written on eight GPUs resumes on four without being told to.

`--resume-session` accepts these three overrides and rejects any other, which
belongs to `--extend-session`.

### Rank-specific session construction

The parent session holds a placeholder DDP resource with rank `-1`. Each child
process settles its own configuration *before* it builds anything: it pins its
CUDA device, records its rank and the launch's topology against the `ddp`
entry in the session state it received, and, on a secondary rank, drops the
components it does not need from that state. Only then is the session
reconstructed. Components are wired to each other as they are constructed, so
a worker never builds a session and then rewires it.

- Rank 0 keeps every configured component.
- Ranks greater than 0 keep `ddp`, roots listed in `parallel_components`, and
  their recursive dependency and wrapping-target closure. Because the
  dependency graph is declared on the component classes, that closure is
  resolved from the state alone -- nothing a secondary rank will discard is
  ever constructed.
- Non-parallel logging, checkpointing, and other rank-zero-only work can
  therefore remain off secondary ranks by omitting those roots.

### What the DDP resource does

During setup, the built-in DDP resource:

- sets `MASTER_ADDR` and `MASTER_PORT`;
- selects CUDA device `rank` for the NCCL backend;
- updates `session.device` to that CUDA device;
- initializes the process group;
- retrieves the `model` resource, moving it to the rank-local CUDA device when
  using NCCL; and
- wraps the model with `torch.nn.parallel.DistributedDataParallel`.

Before each DDP iteration, every rank contributes its local cooperative-stop
state to an `all_reduce(MAX)` decision. A stop request received by any rank
therefore causes all ranks to leave the training loop at the same iteration
boundary instead of allowing another rank to enter a mismatched forward or
backward collective. Gloo uses a CPU control tensor; NCCL uses the session's
CUDA device.

A request that arrives just after this decision may allow one additional
iteration, but that iteration is admitted for every rank. The consensus does
not interrupt a collective already in progress. Worker failures and unhealthy
process groups still rely on the supervisor's configured graceful-join timeout
and terminate/kill fallback. Non-DDP workers keep using their local stop event.

The wrapped module is available only inside the active session context:

```python
ddp_resource = session.get_resource("ddp")
prediction = ddp_resource.wrapped_model(batch)
```

The built-in DDP resource declares a dependency on the `model` role, so model
setup runs before DDP setup. Do not also declare that model as requiring `ddp`,
because the two requirements would form a cycle. During teardown, the wrapped
reference is cleared and the process group is destroyed.

### Devices and ranks

A rank runs on CUDA ordinal `rank % <visible devices>`. Ordinals are relative
to `CUDA_VISIBLE_DEVICES`, so which physical GPUs a run uses is controlled
there: `CUDA_VISIBLE_DEVICES=4,5,6,7` puts rank 0 on physical GPU 4.

The worker pins that device before it constructs anything, so no component is
ever built or restored against the wrong one, and the session's CUDA RNG
stream has a definite device to land on. A session that asks for a CUDA
device without using `nccl` — a single-process run, or a `gloo` group over
CUDA tensors — pins one too. A `gloo` run over CPU tensors pins nothing and
never claims a GPU.

### Current DDP scope

Treat the implementation as a single-node design: the engine spawns every
rank itself, so all of them share one `CUDA_VISIBLE_DEVICES` and take the
first `world_size` entries of it. Multi-node execution, where a rank's global
index and its local device index genuinely diverge, is not exposed by the
framework.