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
    +-- restore TrainingSession and validate opt-in overrides for --extend-session
    |
    +-- capture session state
    |
    +-- spawn worker process rank 0
    |       +-- adjust the state for this rank, before anything is built
    |       +-- reconstruct the concrete Session subtype from state
    |       +-- run resources, hooks, and steps
    |       +-- mark stage progress in shared memory
    |       +-- send errors to the parent
    |
    +-- spawn worker processes rank 1..N-1 for DDP
    |       +-- drop components this rank does not need, before building
    |       +-- reconstruct the reduced session state
    |       +-- run the rank-specific session
    |
    +-- monitor worker pipes and process sentinels
            +-- propagate worker failures
            +-- poll worker progress and detect heartbeat timeouts
            +-- coordinate graceful shutdown
            +-- terminate or kill unresponsive workers
```

Even a non-DDP run is executed in one spawned worker process. Child processes ignore `SIGINT`; the parent process handles interruption and coordinates shutdown.

Components are constructed prerequisite-first, so each one can take hold of
whatever it declared while it is being built. That happens on every path --
fresh configuration, checkpoint restore and worker start-up -- which is what
lets one component keep a live reference to another across a checkpoint or a
spawn. A worker therefore settles its rank-specific configuration and its
component set *before* reconstructing the session, rather than building a
session and then modifying it.

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
3. adjusts that state for the worker's rank, then reconstructs the correct
   subtype with `Session.from_state()`;
4. starts the training or analysis lifecycle in that child process; and
5. watches the worker's progress beacon, error pipe, and process sentinel.

Each worker reports through two channels:

- **Progress beacon** — a small lock-free shared-memory record (sequence counter, iteration, stage label). The framework marks it before every resource setup/teardown, session hook, iteration hook, and step, and on every `session.send_heartbeat(stage)` call. Marking is a few shared-memory writes with no pickling or syscall, so it happens at every stage.
- **Error pipe** — error messages including rank, exception type, message, and traceback.

The parent polls each beacon at least once per second (more often for short timeouts). Any sequence change counts as progress and resets that worker's deadline. When no progress is seen for `--heartbeat-timeout` seconds, the parent raises a `TimeoutError` naming the rank, PID, iteration, and the stage the worker was stuck in, for example:

```text
Worker rank=0 pid=4242 made no progress for 30.4s (iteration 17, stage 'Running Step.validate')
```

While workers progress, the parent prints at most one status line per rank every `min(10, heartbeat_timeout / 3)` seconds.

A single long component call, such as a full validation pass, does not change stages on its own. Call `session.send_heartbeat("…")` periodically inside such loops, or raise `--heartbeat-timeout` above the longest uninterrupted operation.

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
