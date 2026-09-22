# Distributed Training

[← Docs](../README.md) · [Project README](../../README.md)

Adding a `ddp` component makes the engine spawn one worker process per rank and
wrap your model in PyTorch `DistributedDataParallel`. This page covers the
configuration, how a run's topology is decided at launch, what each rank builds,
and how ranks stop together.

The framework is single-node by design; see
[Limitations](../concepts/limitations.md#distributed-training) for the scope.

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

    model: {}
    train: {}

    logger:
      log_every: 10

    checkpointer:
      checkpoint_every: 100
```

`master_port` may be left out, and usually should be: the engine then asks
the operating system for a free port. Either way, the engine binds the port
itself before it starts any worker and holds it until every worker has
finished; the ranks meet through that rendezvous. Nothing else on the machine
can take the port between the moment it is chosen and the moment the ranks
use it.

A session driven by hand, outside the engine, has no launcher to hold the
port: its DDP resource hands `master_addr` and `master_port` to the process
group as a `tcp://` address, and rank 0 binds it. Either way the resource
never writes `MASTER_ADDR` or `MASTER_PORT` into the environment, so nothing
it sets outlives the session. An IPv6 `master_addr` is bracketed for you.

## The launch decides the topology

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
   port, which the engine binds and holds for the run.

A new session must still state its `world_size`. Asking for more ranks than
there are visible CUDA devices is an error when the run is new, and a warning
with a fallback to the device count when it is resumed — so a checkpoint
written on eight GPUs resumes on four without being told to.

A port that is already in use is reported when the run is launched, before
any worker starts. A port you configured, by file, environment or override,
is an error: you asked for that port. A port that came from a checkpoint
belonged to the machine that wrote it, so it is replaced with a free one and
a warning.
`master_addr` has to be an address of the machine running the engine, since
every rank runs there; any other address is reported at launch too.

`--resume-session` accepts these three overrides and rejects any other, which
belongs to `--extend-session`.

## What each rank builds

Rank 0 builds every configured component. Every other rank builds all of them
too, except the ones declared **rank-zero-only** — work that should happen once
per run rather than once per rank.

Nothing has to be listed for a component to run on every rank. That is the
safe default in both directions: a component left off a rank stops taking part
in the collectives and the run hangs, while one built on a rank that had no
need for it costs a constructor call.

There are two ways to declare a component rank-zero-only. Mark the class, and
every session that uses it inherits the decision:

```python
@rank_zero_only
@hook("my_reporter")
class MyReporter(LifecycleHook):
    ...
```

Or name it in the session, for a component whose class you do not own or whose
role differs per run:

```yaml
ddp:
  world_size: 4
  backend: nccl
  rank_zero_components:
    - my_reporter
```

The built-in `logger`, `checkpointer`, `tensorboard` and `timer` are marked
already, so a default configuration keeps logging and checkpointing on rank 0
without being told to.

The two declarations are not treated alike. A class-level mark is its author's
settled decision and is taken at face value — `timer` reaches `ddp` through
`optimizer` and is rank-zero-only on purpose. A name in `rank_zero_components`
is a per-run override, so naming a component whose prerequisites include `ddp`
warns: that is the shape of a component that takes part in the collectives,
and excluding one leaves the other ranks waiting. Every name is resolved
through the bindings and checked against the session before any worker is
spawned, so a typo fails the launch rather than one rank — including on a
single-rank launch, which has no ranks to prune for but would otherwise carry
the mistake until the day the same configuration is scaled up.

A rank-zero-only component that a component this rank *does* build declares as
a prerequisite is built anyway, with a warning: a prerequisite has to exist
wherever its consumer does. Nothing else is inferred — a component is left off
a rank because it was declared rank-zero-only, never because the framework
decided it was needed only there.

### How a rank settles its component set

The parent session holds a placeholder DDP resource with rank `-1`. Each child
process settles its own configuration *before* it builds anything: it pins its
CUDA device, records its rank and the launch's topology against the `ddp`
entry in the session state it received, and, on a secondary rank, drops the
components it does not need from that state. Only then is the session
reconstructed. Components are wired to each other as they are constructed, so
a worker never builds a session and then rewires it.

Because the dependency graph is declared on the component classes, the set a
rank needs is resolved from the state alone — nothing a secondary rank will
discard is ever constructed.

### The deprecated `parallel_components` list

`ddp.parallel_components` was the opposite declaration: an opt-in list of the
roots to keep *on* the other ranks, with everything else dropped. A session
that still sets it keeps exactly that behaviour and warns, so existing
configurations are unaffected; it also warns when the list prunes a component
that requires `ddp`, which is the mistake the list invited. Delete the key to
get the behaviour above, and declare whatever was deliberately absent from the
list as rank-zero-only instead.

## What the DDP resource does

During setup, the built-in DDP resource:

- selects CUDA device `rank` for the NCCL backend;
- updates `session.device` to that CUDA device;
- initializes the process group through the rendezvous the engine holds, or
  at `tcp://<master_addr>:<master_port>` when the session is driven by hand;
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

The wrapped module is available only inside the active session context, to a
component that declares `@requires_resource("ddp")`:

```python
ddp_resource = self.get_dependency("ddp")
prediction = ddp_resource.wrapped_model(batch)
```

The built-in DDP resource declares a dependency on the `model` role, so model
setup runs before DDP setup. Do not also declare that model as requiring `ddp`,
because the two requirements would form a cycle. During teardown, the wrapped
reference is cleared and the process group is destroyed.

## Devices and ranks

A rank runs on CUDA ordinal `rank % <visible devices>`. Ordinals are relative
to `CUDA_VISIBLE_DEVICES`, so which physical GPUs a run uses is controlled
there: `CUDA_VISIBLE_DEVICES=4,5,6,7` puts rank 0 on physical GPU 4.

The worker pins that device before it constructs anything, so no component is
ever built or restored against the wrong one, and the session's CUDA RNG
stream has a definite device to land on. A session that asks for a CUDA
device without using `nccl` — a single-process run, or a `gloo` group over
CUDA tensors — pins one too. A `gloo` run over CPU tensors pins nothing and
never claims a GPU.


---

**Next:** [Analysis sessions](06-analysis-sessions.md) — driving a trained
checkpoint through an analysis workflow.
