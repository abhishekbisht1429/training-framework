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

    components:
      - model
      - train

    logger:
      log_every: 10

    checkpointer:
      checkpoint_every: 100
```

`master_port` should be a string because it is assigned to the `MASTER_PORT` environment variable.

### Rank-specific session construction

The parent session contains a placeholder DDP resource with rank `-1`. In each child process, the framework replaces it with a resource configured for that worker's rank.

- Rank 0 keeps every configured component.
- Ranks greater than 0 keep `ddp`, roots listed in `parallel_components`, and their recursive dependency and wrapping-target closure.
- Non-parallel logging, checkpointing, and other rank-zero-only work can therefore remain off secondary ranks by omitting those roots.

### What the DDP resource does

During setup, the built-in DDP resource:

- sets `MASTER_ADDR` and `MASTER_PORT`;
- selects CUDA device `rank` for the NCCL backend;
- updates `session.device` to that CUDA device;
- initializes the process group;
- retrieves the `model` resource, moving it to the rank-local CUDA device when
  using NCCL; and
- wraps the model with `torch.nn.parallel.DistributedDataParallel`.

The wrapped module is available only inside the active session context:

```python
ddp_resource = session.get_resource("ddp")
prediction = ddp_resource.wrapped_model(batch)
```

The built-in DDP resource declares a dependency on the `model` role, so model
setup runs before DDP setup. Do not also declare that model as requiring `ddp`,
because the two requirements would form a cycle. During teardown, the wrapped
reference is cleared and the process group is destroyed.

### Current DDP scope

The current CUDA mapping uses the process rank directly as the local CUDA device index. Treat the implementation as a single-node, one-process-per-GPU design. Multi-node execution requires separate global-rank and local-rank handling that is not currently exposed by the framework.