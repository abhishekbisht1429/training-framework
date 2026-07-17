# Training Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package version](https://img.shields.io/badge/version-0.3.4-blue.svg)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)

A lightweight, component-based framework for organizing PyTorch training workflows into reusable **sessions**, **resources**, **hooks**, and **steps**.

The framework provides:

- a deterministic session lifecycle;
- registry-based component discovery;
- explicit component prerequisites;
- per-iteration and per-session shared contexts;
- stateful checkpoint save and restore;
- Python, NumPy, PyTorch, and CUDA RNG restoration;
- configuration through YAML and OmegaConf overrides;
- one worker thread per training session;
- built-in logging, checkpointing, TensorBoard, and infinite sampling utilities.

> **Project status:** the framework is under active development. Review the [current behavior and limitations](#current-behavior-and-limitations) before using it for long-running or production workloads.

## Table of contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Component registration](#component-registration)
- [Component prerequisites](#component-prerequisites)
- [Shared contexts](#shared-contexts)
- [State and checkpointing](#state-and-checkpointing)
- [Built-in components](#built-in-components)
- [YAML configuration](#yaml-configuration)
- [Running multiple sessions](#running-multiple-sessions)
- [InfiniteSampler](#infinitesampler)
- [API summary](#api-summary)
- [Testing](#testing)
- [Current behavior and limitations](#current-behavior-and-limitations)
- [License](#license)

## Architecture

```text
TrainingEngine
│
├── worker thread ──> TrainingSession A
│                    ├── Resources
│                    ├── Hooks
│                    ├── Steps
│                    ├── session_context
│                    └── iteration_context
│
├── worker thread ──> TrainingSession B
│                    ├── Resources
│                    ├── Hooks
│                    └── Steps
│
└── ...
```

A `TrainingSession` owns one complete training workflow. A `TrainingEngine` can execute multiple sessions concurrently, using one Python thread for each registered session.

Within a session:

- **resources** manage objects whose lifetime follows the session context;
- **hooks** observe session or iteration lifecycle events;
- **steps** perform the ordered work of each iteration;
- **session context** shares data across the active session;
- **iteration context** shares transient data within one iteration.

## Requirements

- Python 3.12 or newer
- PyTorch 2.11 or newer
- NumPy 2.4.4 or newer
- OmegaConf 2.3 or newer
- TensorBoard 2.20 or newer

The package metadata contains the complete dependency list.

## Installation

### Install directly from GitHub

```bash
python -m pip install "git+https://github.com/abhishekbisht1429/training-framework.git@main"
```

To pin the exact version documented here:

```bash
python -m pip install \
  "git+https://github.com/abhishekbisht1429/training-framework.git@c4e09ea4a75d92baf965c72c786c9c874e376e7b"
```

### Add it to another `pyproject.toml`

```toml
[project]
dependencies = [
    "training-framework @ git+https://github.com/abhishekbisht1429/training-framework.git@main",
]
```

### Development installation

```bash
git clone https://github.com/abhishekbisht1429/training-framework.git
cd training-framework
python -m pip install --upgrade pip
python -m pip install -e .
pytest
```

## Quick start

The following example creates a resource, a lifecycle hook, and a step; declares their dependencies; and runs the session through `TrainingEngine`.

```python
from training_framework.training_engine import TrainingEngine
from training_framework.training_session import (
    LifecycleHook,
    Resource,
    Step,
    TrainingSession,
    hook,
    requires_hook,
    requires_resource,
    resource,
    step,
)


@resource("example_metric_store")
class MetricStore(Resource):
    def __init__(self):
        self.losses: list[float] = []

    def setup(self, session: TrainingSession):
        self.losses.clear()

    def teardown(self, session: TrainingSession):
        pass


@hook("example_loss_printer")
@requires_resource("example_metric_store")
class LossPrinter(LifecycleHook):
    def __init__(self, call_every: int = 1):
        self.call_every = call_every

    def setup(self, session: TrainingSession):
        pass

    def teardown(self, session: TrainingSession):
        pass

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        pass

    def post_iteration_callback(self, session: TrainingSession) -> None:
        loss = session.iteration_context["loss"]
        print(f"iteration={session.iteration} loss={loss:.3f}")


@step("example_train_step")
@requires_resource("example_metric_store")
@requires_hook("example_loss_printer")
class TrainStep(Step):
    def __init__(self, metric_store_id: str):
        self.metric_store_id = metric_store_id

    def run(self, session: TrainingSession) -> None:
        # Replace this with a real forward/backward/optimizer operation.
        loss = 1.0 / session.iteration

        # Visible to later steps and post-iteration hooks in this iteration.
        session.iteration_context["loss"] = loss

        # Visible for the lifetime of the active session context.
        session.session_context["last_loss"] = loss

        metric_store = session.get_resource(self.metric_store_id)
        metric_store.losses.append(loss)


config = {
    "rng_seed": 42,
    "sessions_dir": "./runs",
    "max_iterations": 5,
    "device": "cpu",
}

session = TrainingSession(config)

# Dependencies must be registered before their consumers.
metric_store_id = session.register_resource(MetricStore())
session.register_hook(LossPrinter(call_every=1))
session.add_step(TrainStep(metric_store_id))

engine = TrainingEngine({})
engine.register_session(session)

# TrainingEngine.run_all() must be called inside the engine context.
with engine:
    engine.run_all(wait=True)
```

Alternatively, a session can be driven directly instead of registering it with an engine:

```python
with session:
    for iteration in session:
        print(iteration)
```

A finished session cannot be entered again. A session that exits before reaching `max_iterations` is paused and can be re-entered.

## Core concepts

### TrainingSession

`TrainingSession` is an iterator and context manager. Its required configuration fields are:

```python
config = {
    "rng_seed": 42,
    "sessions_dir": "./runs",
    "max_iterations": 100,
    "device": "cpu",       # or an available value such as "cuda:0"
}
```

At construction, the session:

1. creates a timestamped session directory path under `sessions_dir`;
2. stores an immutable `SessionConfig`;
3. seeds Python, NumPy, PyTorch, and CUDA RNGs;
4. validates the requested device;
5. initializes resource, hook, step, and context collections.

The session phases are:

```text
NEW ──enter──> READY ──next()──> RUNNING
                                │
                                ├── context exits early ──> PAUSED
                                └── iteration limit reached ──> FINISHED
```

`SessionPhase.INTERRUPTED` is defined but is not currently assigned by the execution path.

### Resource

A resource owns infrastructure or data used by a session, such as a dataset, writer, process, logger backend, or model service.

```python
@resource("my_resource")
class MyResource(Resource):
    def setup(self, session: TrainingSession):
        ...

    def teardown(self, session: TrainingSession):
        ...
```

Resources are set up in registration order and torn down in reverse registration order.

Registering a resource returns an instance-specific ID:

```python
resource_id = session.register_resource(MyResource())
resource = session.get_resource(resource_id)
```

The registry name identifies the component type; the returned resource ID identifies one registered instance.

### Hook

Hooks observe session or iteration events.

The framework provides three hook interfaces:

| Interface | Required methods |
|---|---|
| `SessionHook` | `setup(session)`, `teardown(session)` |
| `IterationHook` | `pre_iteration_callback(session)`, `post_iteration_callback(session)` |
| `LifecycleHook` | all four methods above |

An iteration hook must expose a positive integer `call_every` value.

```python
@hook("metrics_hook")
class MetricsHook(LifecycleHook):
    def __init__(self, call_every: int = 10):
        self.call_every = call_every

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def pre_iteration_callback(self, session):
        pass

    def post_iteration_callback(self, session):
        pass
```

Iteration callbacks run when at least one of the following is true:

- the current iteration is the first iteration;
- the current iteration is the last configured iteration;
- `session.iteration % hook.call_every == 0`.

Therefore, the first and last iterations always invoke every iteration hook, regardless of `call_every`.

For hooks registered as `A`, `B`, and `C`, callback order is:

```text
A.pre -> B.pre -> C.pre -> steps -> C.post -> B.post -> A.post
```

### Step

A step is one ordered unit of iteration work.

```python
@step("forward")
class ForwardStep(Step):
    def run(self, session: TrainingSession) -> None:
        ...
```

Steps execute in the order in which they were added:

```python
session.add_step(LoadBatchStep())
session.add_step(ForwardStep())
session.add_step(BackwardStep())
session.add_step(OptimizerStep())
```

## Component registration

Component decorators assign a globally unique registry name:

```python
@resource("dataset")
class DatasetResource(Resource):
    ...


@hook("metrics")
class MetricsHook(SessionHook):
    ...


@step("train")
class TrainStep(Step):
    ...
```

The corresponding registries are:

```python
RESOURCE_REGISTRY
HOOK_REGISTRY
STEP_REGISTRY
```

Registering two classes under the same name in the same registry raises `ValueError`.

Registration is required for session checkpoint reconstruction. Before loading a checkpoint, import every module that declares custom resource, hook, and step classes so that their decorators populate the registries.

## Component prerequisites

Dependencies are declared by registry name:

```python
@step("optimizer")
@requires_resource("model")
@requires_hook("metrics")
@requires_step("backward")
class OptimizerStep(Step):
    ...
```

The allowed dependency types are:

| Consumer | May require resources | May require hooks | May require steps |
|---|:---:|:---:|:---:|
| `Resource` | Yes | No | No |
| `Hook` | Yes | Yes | No |
| `Step` | Yes | Yes | Yes |

The decorators enforce those type rules:

- `@requires_resource(...)` may decorate a `Resource`, `Hook`, or `Step` subclass;
- `@requires_hook(...)` may decorate a `Hook` or `Step` subclass;
- `@requires_step(...)` may decorate only a `Step` subclass.

Prerequisites must already be present in the same session before the dependent component is registered:

```python
session.register_resource(ModelResource())

session.register_hook(MetricsHook())

session.add_step(BackwardStep())
session.add_step(OptimizerStep())
```

Missing prerequisites raise `RuntimeError` and identify missing resources, hooks, or steps. Names are checked by category, so a hook named `model` does not satisfy a required resource named `model`.

Multiple prerequisite decorators are supported. Python applies stacked decorators from bottom to top:

```python
@step("consumer")
@requires_resource("second")
@requires_resource("first")
class ConsumerStep(Step):
    ...
```

Registration order is significant. The framework currently validates direct prerequisites but does not resolve, auto-register, topologically sort, or detect circular dependency graphs.

## Shared contexts

### `iteration_context`

`session.iteration_context` is a dictionary for communication within one iteration.

```python
class ForwardStep(Step):
    def run(self, session):
        session.iteration_context["output"] = output


class LossStep(Step):
    def run(self, session):
        output = session.iteration_context["output"]
        session.iteration_context["loss"] = compute_loss(output)
```

Properties:

- available only while the session context is active;
- shared by steps and hooks in the current iteration;
- available to post-iteration hooks;
- cleared after post-iteration callbacks complete;
- not saved in session checkpoints.

Access outside `with session:` raises `RuntimeError`.

### `session_context`

`session.session_context` is a dictionary shared for the active session lifetime.

```python
class ProducerHook(SessionHook):
    def setup(self, session):
        session.session_context["run_name"] = "experiment-1"

    def teardown(self, session):
        pass


class ConsumerHook(SessionHook):
    def setup(self, session):
        print(session.session_context["run_name"])

    def teardown(self, session):
        pass
```

Properties:

- persists across iterations while the session remains active;
- is shared by session components;
- is included in `TrainingSession.get_state()`;
- is restored by `TrainingSession.set_state()`;
- is cleared when the session context exits.

Because it is checkpointed, values stored in `session_context` must be serializable if the session is saved.

## State and checkpointing

### Stateful components

A component that must preserve mutable state across checkpoints should implement `Stateful` or inherit one of the convenience abstract classes:

- `StatefulResource`
- `StatefulStep`
- `StatefulSessionHook`
- `StatefulIterationHook`
- `StatefulLifeCycleHook`

```python
from typing import Any

from training_framework.training_session import StatefulStep, step


@step("counter")
class CounterStep(StatefulStep):
    def __init__(self, increment: int = 1):
        self.increment = increment
        self.value = 0

    def run(self, session) -> None:
        self.value += self.increment

    def get_state(self) -> Any:
        return {"value": self.value}

    def set_state(self, state: Any) -> None:
        self.value = state["value"]
```

The framework automatically captures exact constructor `args` and `kwargs` through `CaptureInitMeta`. During restoration it:

1. finds each class by its registry name;
2. reconstructs it with the captured constructor call;
3. calls `set_state()` for `Stateful` components.

Constructor arguments and returned state must therefore be serializable.

### Session state

`TrainingSession.get_state()` includes:

- original session configuration;
- current iteration;
- immutable `SessionConfig`;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- PyTorch CUDA RNG state;
- resource definitions and state;
- hook definitions and state;
- step definitions and state;
- captured constructor arguments;
- `session_context`.

Transient objects such as `device` and `iteration_context` are recreated instead of restored directly.

### Save and restore directly

```python
import torch

# Save the complete session object.
torch.save(session, "checkpoint.pt")

# Ensure modules containing all custom registered classes are imported first.
restored_session = torch.load(
    "checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)
```

Python `pickle` can also be used because `TrainingSession` implements the state protocol:

```python
import pickle

payload = pickle.dumps(session)
restored_session = pickle.loads(payload)
```

### Built-in Checkpointer

```python
from training_framework.resources import Checkpointer

checkpointer = Checkpointer(
    {
        "checkpoint_every": 100,
        "checkpoints_dir": "./runs/checkpoints",
    }
)

session.register_hook(checkpointer)
```

Load a saved checkpoint with:

```python
restored_session = Checkpointer.load_checkpoint(
    "./runs/checkpoints/<checkpoint-name>",
    map_location="cpu",
)
```

The built-in checkpointer is an iteration hook, so it saves on the first iteration, the last iteration, and iterations divisible by `checkpoint_every`.

## Built-in components

### Logger

`Logger` is registered under the hook name `logger`.

```python
from training_framework.resources import Logger

session.register_hook(
    Logger(
        {
            "log_every": 10,
            "log_file": "./runs/training.log",  # optional
        }
    )
)
```

It writes:

```text
Iteration <current>/<maximum>
```

If `log_file` is omitted, output is written to standard output. The parent directory for a configured log file must already exist.

### Checkpointer

`Checkpointer` is registered under the hook name `checkpointer`.

Required configuration:

```python
{
    "checkpoint_every": 100,
    "checkpoints_dir": "./runs/checkpoints",  # optional
}
```

When `checkpoints_dir` is omitted, checkpoints are written under the session directory. The directory is created automatically.

### Tensorboard

`Tensorboard` is registered under the resource name `tensorboard`.

```python
from training_framework.resources import Tensorboard

resource_id = session.register_resource(
    Tensorboard(
        {
            "host": "127.0.0.1",
            "port": 6006,
            "logdir": "./runs/tensorboard",  # optional server log directory
        }
    )
)

# The writer is created during resource setup and is therefore available
# only after the session has entered its context, including from steps/hooks.
with session:
    tensorboard = session.get_resource(resource_id)
    writer = tensorboard.summary_writer
    writer.add_scalar("example/value", 1.0, session.iteration)
```

On setup, the resource starts a TensorBoard subprocess and creates a PyTorch `SummaryWriter`. On teardown, it closes the writer and terminates the subprocess.

The `tensorboard` executable must be available in the active environment, and the configured port must be free.

## YAML configuration

`Configurator` reads a YAML file from the first positional command-line argument. The root must contain a `sessions` list.

```yaml
sessions:
  - rng_seed: 42
    sessions_dir: ./runs
    max_iterations: 100
    device: cpu

    logger:
      log_every: 10
      log_file: ./runs/training.log

    checkpointer:
      checkpoint_every: 25
      checkpoints_dir: ./runs/checkpoints

    tensorboard:
      host: 127.0.0.1
      port: 6006
      logdir: ./runs/tensorboard
```

Create sessions from the configuration:

```python
from training_framework.configurator import Configurator
from training_framework.training_engine import TrainingEngine

configurator = Configurator()
sessions = configurator.create_sessions()

# Configurator attaches configured built-in Logger, Checkpointer, and
# Tensorboard components. Attach application-specific steps and components here.
for session in sessions:
    session.add_step(...)

engine = TrainingEngine({})
for session in sessions:
    engine.register_session(session)

with engine:
    engine.run_all()
```

Run the program:

```bash
python train.py config.yaml
```

### Command-line overrides

Pass OmegaConf dot-list overrides after `--override`:

```bash
python train.py config.yaml --override \
  'sessions[0].max_iterations=25' \
  'sessions[0].logger.log_every=5'
```

### Configurator API

```python
configurator.get_session_config(index)
configurator.get_sub_config(session_index, key)
configurator.create_sessions()
```

`get_sub_config()` returns a deep copy of a mapping. It raises `KeyError` for a missing key and `ValueError` when the selected value is not a mapping.

## Running multiple sessions

```python
engine = TrainingEngine({})

engine.register_session(session_a)
engine.register_session(session_b)

with engine:
    engine.run_all(wait=True)
```

`wait=True` starts every registered session and joins all worker threads before returning from `run_all()`.

For non-blocking startup:

```python
with engine:
    engine.run_all(wait=False)
    # The worker threads are active here.
    do_other_work()

# Leaving the context clears each per-session active flag and joins every
# still-running worker thread.
```

Shutdown is cooperative at iteration boundaries. A worker that has started checks its active flag before beginning the next iteration. If it is already inside `next(session)`, the current iteration is allowed to finish before the flag is checked again.

Registering the same `TrainingSession` object twice in one engine raises `RuntimeError`.

## InfiniteSampler

`InfiniteSampler` repeatedly yields a new random permutation of indices from `0` through `n_samples - 1`.

```python
from torch.utils.data import DataLoader

from training_framework.dataloader import InfiniteSampler

sampler = InfiniteSampler(len(dataset))
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

The sampler has no natural end. Session iteration limits, rather than sampler exhaustion, should determine training length.

## API summary

### TrainingSession

| Member | Purpose |
|---|---|
| `TrainingSession(config)` | Create a session and seed its RNGs |
| `register_resource(resource)` | Validate and register a resource; returns its instance ID |
| `get_resource(resource_id)` | Retrieve a registered resource |
| `register_hook(hook)` | Validate and register a hook |
| `add_step(step)` | Validate and append a step |
| `session_config` | Immutable `SessionConfig` |
| `iteration` | Current iteration number |
| `device` | Validated `torch.device` |
| `session_context` | Session-lifetime shared dictionary |
| `iteration_context` | Current-iteration shared dictionary; context-only |
| `get_state()` | Return serializable session state |
| `set_state(state)` | Restore session state |

### TrainingEngine

| Member | Purpose |
|---|---|
| `TrainingEngine(config)` | Create an engine |
| `register_session(session)` | Register one session and create its worker thread |
| `run_all(wait=True)` | Start all registered session threads |

`run_all()` requires an active engine context.

### Registries and decorators

| API | Purpose |
|---|---|
| `@resource(name)` | Register a resource class |
| `@hook(name)` | Register a hook class |
| `@step(name)` | Register a step class |
| `@requires_resource(name)` | Declare a resource prerequisite |
| `@requires_hook(name)` | Declare a hook prerequisite |
| `@requires_step(name)` | Declare a step prerequisite |

## Testing

Install the project and run:

```bash
pytest
```

The repository test suite covers:

- session lifecycle and iteration ordering;
- component registries and validation;
- prerequisite decorators and category-aware dependency checks;
- configuration parsing and overrides;
- built-in Logger, Checkpointer, and TensorBoard behavior;
- session-context sharing, persistence, and cleanup;
- state and constructor-argument restoration;
- Python, NumPy, and PyTorch training-state behavior;
- threaded execution and cooperative shutdown;
- duplicate session registration;
- `InfiniteSampler` behavior.

The GitHub Actions workflow runs the suite on Python 3.12 and 3.13.

## Current behavior and limitations

The following points describe the current `0.3.4` implementation:

1. **Registration order matters.** A prerequisite must be registered before its consumer. Dependencies are validated but are not automatically resolved or sorted.
2. **Circular dependencies are not detected.** Avoid dependency cycles between components.
3. **Component registry names are global within a Python process.** Importing two classes with the same name in the same registry raises `ValueError`.
4. **Custom component modules must be imported before checkpoint loading.** Restoration looks up classes in the global registries.
5. **Constructor arguments and persistent state must be serializable.** This includes values stored in `session_context` at checkpoint time.
6. **Thread shutdown is cooperative.** A long-running or blocked step can delay engine context exit because Python threads are not forcefully interrupted.
7. **Session ownership is enforced only within one engine.** Do not register the same session object with multiple engines concurrently.
8. **An engine's worker threads are created during registration and are intended for one start.** Create a new engine when a completely new execution run is required.
9. **In version 0.3.4, session exit calls component `teardown` methods with `None`.** Implement the required parameter, but do not rely on it containing the session until the call site is changed to pass the active `TrainingSession`.
10. **The built-in TensorBoard resource launches an external process and waits during setup.** Use an available port and ensure the executable can be started.

## Project layout

```text
training-framework/
├── .github/workflows/python-tests.yaml
├── pyproject.toml
├── README.md
├── LICENSE
└── src/
    ├── training_framework/
    │   ├── configurator.py
    │   ├── dataloader.py
    │   ├── resources.py
    │   ├── training_engine.py
    │   ├── training_session.py
    │   └── util.py
    └── tests/
        ├── test_registrations.py
        ├── test_session_context.py
        ├── test_state_loading.py
        ├── test_threads.py
        ├── test_training_framework_additional.py
        └── test_training_session.py
```

## License

This repository is distributed under the [Apache License 2.0](./LICENSE).


---

This README was generated with the assistance of ChatGPT. In case you come across some errors please report it by creating a new issue.