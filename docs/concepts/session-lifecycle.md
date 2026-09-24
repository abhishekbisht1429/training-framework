# Session Lifecycle

[← Docs](../README.md) · [Project README](../../README.md)

The exact order in which a session builds, sets up, iterates, and tears down
its components, what happens when one of those phases fails, and the two
dictionaries components use to pass values to each other. Read this when you
need to know precisely when your callback runs relative to someone else's.

## Phases

Both `TrainingSession` and `AnalysisSession` are context managers and iterators.

```text
Construct session
    |
    +-- import component package
    +-- construct configured components, prerequisite-first,
    |   each one taking hold of what it declared
    |
Enter session
    +-- resource.setup() in dependency order
    +-- SessionHook.pre_session() in dependency order
    +-- create session directory and write config.yaml for rank 0
    |
Each iteration
    +-- selected IterationHook.pre_iteration_callback()
    +-- Step.run() in dependency order
    +-- selected IterationHook.post_iteration_callback() in reverse order
    +-- clear iteration_context
    |
Exit session
    +-- SessionHook.post_session() in reverse dependency order
    +-- resource.teardown() in reverse dependency order
    +-- clear session_context
```

The main phases are:

```text
NEW -> READY -> RUNNING -> FINISHED
                  |
                  +-> PAUSED when the context exits before max_iterations
```

If an iteration fails, its iteration counter is rolled back and its iteration context is cleared before the exception propagates.

If a resource's `setup()` fails, its `rollback_setup()` callback runs before
previously initialized resources are torn down in reverse order. If a session
hook's `pre_session()` fails, its `rollback_pre_session()` callback runs
before earlier hooks receive `post_session()` and resources are torn down.
The component whose initialization failed does not receive its normal
`teardown()` or `post_session()` callback.

Rollback errors are reported without replacing the original initialization
exception or preventing the remaining cleanup. Session context is cleared
before that original exception propagates.

## Shared contexts

### `iteration_context`

`session.iteration_context` is a dictionary for communication among hooks and steps during one iteration.

```python
session.iteration_context["batch"] = batch
loss = session.iteration_context["loss"]
```

It is:

- available only while the session context is active;
- visible to pre-hooks, steps, and post-hooks;
- cleared after every iteration; and
- not included in session checkpoints.

### `session_context`

`session.session_context` is a dictionary shared for the active session lifetime.

```python
session.session_context["best_loss"] = best_loss
```

It is included in `Session.get_state()` and restored with the concrete session. It is cleared when the session context exits. Values that exist when a checkpoint is created must therefore be serializable.

## Direct session execution

For single-process development or unit tests, either concrete session can be
driven without `TrainingEngine`. A training session can be run directly as:

```python
import yaml
from training_framework.session import TrainingSession


with open("my_project/config.yaml") as config_file:
    config = yaml.safe_load(config_file)["sessions"][0]

session = TrainingSession(config)

with session:
    for iteration in session:
        print(iteration)
```

Construct `AnalysisSession` in the same way shown in
[Analysis sessions](../guide/07-analysis-sessions.md), then enter and iterate it
with the same pattern. Direct execution bypasses spawned-worker supervision, error pipes,
heartbeat monitoring, and rank-specific DDP reconstruction. Use
`TrainingEngine` for the normal managed execution path.

---

**See also:** [Architecture and process model](architecture.md) for what the
parent process does around this lifecycle, and
[The component model](component-model.md) for what a component may do while it
is being constructed.
