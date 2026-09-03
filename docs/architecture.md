# Architecture and Process Model

[← Documentation index](README.md) · [Project README](../README.md)

## Architecture

```text
Parent process

Configurator
    |
    v
TrainingEngine
    |
    +-- resolve each --config entry through its registered session_type
    +-- restore a checkpointed Session for --resume-session
    +-- restore and extend TrainingSession for --extend-session
    |
    +-- capture session state
    |
    +-- spawn worker process rank 0
    |       +-- reconstruct the concrete Session subtype from state
    |       +-- configure rank-specific DDP resource, when enabled
    |       +-- run resources, hooks, and steps
    |       +-- send heartbeats and errors to the parent
    |
    +-- spawn worker processes rank 1..N-1 for DDP
    |       +-- reconstruct the same session state
    |       +-- keep only configured parallel components
    |       +-- run the rank-specific session
    |
    +-- monitor worker pipes and process sentinels
            +-- propagate worker failures
            +-- detect heartbeat timeouts
            +-- coordinate graceful shutdown
            +-- terminate or kill unresponsive workers
```

Even a non-DDP run is executed in one spawned worker process. Child processes ignore `SIGINT`; the parent process handles interruption and coordinates shutdown.

A concrete `Session` contains:

- **Resources**: objects with session-scoped setup and teardown, such as models, optimizers, datasets, writers, and distributed infrastructure.
- **Hooks**: callbacks around session setup/teardown and/or iterations.
- **Steps**: ordered units of work performed during every iteration.
- **Session context**: shared state for the lifetime of an active session.
- **Iteration context**: temporary shared state that is cleared after every iteration.

## Process model and supervision

The engine uses `torch.multiprocessing.get_context("spawn")`.

For each worker, the parent:

1. calls `get_state()` on the concrete session;
2. passes the state to a new interpreter;
3. reconstructs the correct subtype with `Session.from_state()`;
4. starts the training or analysis lifecycle in that child process; and
5. watches both the worker's message pipe and process sentinel.

The worker sends:

- **heartbeat messages**, including PID, iteration, and the current component stage; and
- **error messages**, including rank, exception type, message, and traceback.

The worker heartbeat interval is currently fixed at 10 seconds. The parent timeout is configurable with `--heartbeat-timeout`.

On interruption, worker failure, or heartbeat timeout, the engine:

1. sets each worker's cooperative stop event;
2. waits for the configured graceful-shutdown period;
3. terminates surviving processes;
4. waits briefly again; and
5. kills processes that still remain alive.

Non-DDP workers check their local stop event between iterations. DDP workers
instead combine their local event states with `all_reduce(MAX)` before each
iteration, so all ranks either admit the next iteration or leave the loop
together. This prevents one rank from stopping while another enters a DDP
forward or backward collective.

A stop request that races with a completed DDP decision may permit one
additional synchronized iteration. A step, hook, setup, teardown, or collective
that is already running is not interrupted cooperatively; the supervisor's
timeout and process termination remain the fallback for unresponsive workers.