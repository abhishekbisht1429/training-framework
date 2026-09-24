# Checkpoints, Resume, and Extend

[← Docs](../README.md) · [Project README](../../README.md)

A checkpoint captures a running session so it can be continued later — on the
same machine or a different one, with the same hyperparameters or changed ones.
This page covers what a checkpoint holds, how it is laid out on disk, reading
part of one, what happens when a component changes between saving and
loading, the two ways to continue from one (`--resume-session` and
`--extend-session`), and what may safely change in each.

To turn checkpointing on, configure the built-in `checkpointer` hook; its
options are in the
[built-in component reference](../reference/builtin-components.md#checkpointer).

## Stored session state

`TrainingSession.get_state()` includes:

- current iteration;
- the `SessionConfig` fields;
- component constructor arguments;
- each component's wiring -- the instance it was given for every resource it
  asked for -- and its `state_version`;
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

## A checkpoint on disk

The checkpointer writes each checkpoint as a directory named after its
timestamp:

```text
checkpoints/<timestamp>/
  manifest.json            what the checkpoint holds
  session.pt               the session's own state, and every component's
                           record: implementation, kind, constructor
                           arguments, wiring, state_version
  components/<name>.pt     one component's state, one file per instance
```

`manifest.json` is plain JSON: the session type, iteration, configuration, and
for every component its implementation, kind, `state_version`, the instances
it was wired to, its file and that file's SHA-256. Anything can read it,
without the framework; `Checkpointer.read_manifest(path)` returns it.

Every `.pt` file holds plain data only -- tensors, dicts, lists, tuples, sets,
numbers, strings and `None` -- and is read with
`torch.load(weights_only=True)`. Reading a checkpoint therefore imports no
class and runs no pickled code, so renaming or moving a class no longer makes
a checkpoint unreadable. A component whose state or constructor arguments hold
anything else (an object, a function, a class) is refused *when the checkpoint
is written*, with an error naming the component and where the value is. Store
the object's state, or the arguments that rebuild it, instead.

A checkpoint is written under a temporary name (`<timestamp>.tmp`) and renamed
into place only when complete, so a save interrupted by a crash never looks
like a checkpoint, and loading one is refused. A component file that no longer
matches its checksum is reported by component name.

To write a checkpoint yourself, call
`Checkpointer.save_checkpoint(session, path)`.

### Single-file checkpoints from before 0.5.0

A checkpoint written before 0.5.0 is one file holding the pickled session.
`--resume-session`, `--extend-session`, `load_checkpoint` and `load_component`
still read it in 0.5.0, with a `FutureWarning`; the next release does not.
Convert one by loading and saving it:

```python
from training_framework.components.builtin import Checkpointer

Checkpointer.save_checkpoint(
    Checkpointer.load_checkpoint("old-checkpoint.pt"),
    "converted-checkpoint",
)
```

## Reading part of a checkpoint

Loading a whole checkpoint rebuilds every component in it, so one that no
longer builds -- its class was removed, or its constructor changed -- stops
the whole load. Three calls read less:

| Call | Builds | Needs the component's class |
|---|---|---|
| `Checkpointer.read_manifest(path)` | nothing | no |
| `Checkpointer.load_component_state(path, name)` | nothing: returns the saved state, e.g. a model's weights | no |
| `Checkpointer.load_component(path, name, with_dependencies=True)` | that resource and the instances it was wired to | yes |

`load_component` rebuilds the resource and, transitively, the resources it was
wired to when the checkpoint was written -- nothing else in the checkpoint is
read, so another component that no longer loads does not stand in the way.
With `with_dependencies=False` it builds a resource that was wired to nothing,
and refuses one that was, since it could not be constructed without them; use
`load_component_state` to get that one's saved state alone.

`load_component` resolves `name` through the checkpoint's own bindings, as the
session that wrote it did. `load_component_state` imports nothing, so it
accepts an instance name, a name the checkpoint's top-level
`component_bindings` bind, or a name one of its components asked for as a
dependency (which resolves to the instance that component was given); a role
declared only in your component package needs `load_component`.

Neither adopts the checkpoint's RNG, and both leave the caller's RNG as it
was.

## When a component changes

A restore rebuilds each component from its recorded constructor arguments and
hands it the instances it was wired to, then gives it its saved state. Every
checkpointed component is checked before any is built -- still registered,
still the same kind, every prerequisite present, its state version one this
code can take -- and every problem found is reported in one error, not only
the first. When several components' `set_state` fail, those are reported
together too.

A component that changes what it checkpoints raises its `state_version`
(a class attribute, `1` by default) and says how to bring an older state
forward:

```python
@resource("encoder")
class Encoder(StatefulResource):
    state_version = 2

    @classmethod
    def migrate_state(cls, from_version, state):
        # version 1 stored a single "weight" tensor
        return {"weights": [state["weight"]]}

    @classmethod
    def migrate_init_args(cls, from_version, init_args):
        # only needed when the constructor changed too
        return init_args
```

The recorded version is compared with the class's when the checkpoint is
loaded. An older one is passed through `migrate_init_args` (the default keeps
the arguments) and `migrate_state` (the default refuses); a newer one --
written by a later version of the component -- is always refused.

A component whose saved state cannot be taken fails the load by default. To
continue without it, pass `on_mismatch`:

```python
session = Checkpointer.load_checkpoint(path, on_mismatch="reinit")
session = Checkpointer.load_checkpoint(path, on_mismatch={"encoder"})
```

`"reinit"` keeps every such component as freshly built from its recorded
constructor arguments, and a collection of instance names does that for those
only; both warn with the names and reasons. Nothing that cannot be *built* --
an unregistered component, a missing prerequisite -- is ever skipped.

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
constructor values under `optimizer.optimizer.kwargs`, `optimizer.lr_scheduler`,
the `clip_gradients` and `freeze_gradients` settings, `logger.log_every`,
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

A schedule left unchanged keeps the length it was built for: raising
`max_iterations` does not stretch it. Once it has run its steps the learning
rate stays at the schedule's final value (a cosine stays at `eta_min`) for the
rest of the extension.

`optimizer.lr_scheduler` may be replaced entirely; the new schedule restarts
from the extension point, and `$max_iterations` in it counts only the
optimizer steps still to come. To drop scheduling and continue at a fixed
learning rate, set it to `null`:

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
[optimization reference](../reference/optimization.md#extending-a-session) for
exactly how the built-in optimizer rescales its learning rate.

## Checkpoint safety

A checkpoint directory is read with `torch.load(..., weights_only=True)`, which
rebuilds tensors and plain containers only and runs no pickled code. A
[single-file checkpoint from before 0.5.0](#single-file-checkpoints-from-before-050)
is read with `weights_only=False`, which can execute arbitrary code: load one
only from a trusted source.

Exact training continuation also depends on application state. Persist model, optimizer, scheduler, scaler, sampler, and any data-pipeline state that affects the next batch.

---

**Next:** [Distributed training](06-distributed-training.md) — running a
session across several GPUs.
