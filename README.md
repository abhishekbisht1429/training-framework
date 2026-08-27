# Training Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package version](https://img.shields.io/badge/version-0.3.4-blue.svg)](./pyproject.toml)
[![Python tests](https://github.com/abhishekbisht1429/training-framework/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/abhishekbisht1429/training-framework/actions/workflows/python-tests.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)

A component-based framework for building, running, checkpointing, and supervising PyTorch training and trained-model analysis workflows.

Workflow code is organized into reusable **resources**, **hooks**, and **steps**. `TrainingSession` and `AnalysisSession` share the lifecycle implemented by the abstract `Session` base, while using separate component registries and defaults. `TrainingEngine` constructs or restores the appropriate session in the parent process, serializes its state, launches one or more spawned workers, and monitors them for completion, errors, interrupts, and missed heartbeats.

> **Project status:** This project is under active development. The current API is suitable for experimentation and framework development, but review [Current behavior and limitations](#current-behavior-and-limitations) before using it for long-running or production workloads.

## Features

- Configuration-driven session construction with YAML and OmegaConf overrides
- Decorator-based registration and recursive component-package discovery
- Shared components with optional per-session-type registry overrides
- Explicit resource, hook, and step dependencies
- Topological execution ordering and dependency-cycle detection
- Session-level and iteration-level shared contexts
- Stateful component checkpointing and session restoration
- Restoration of Python, NumPy, PyTorch, and CUDA RNG state
- Spawn-based worker processes with error forwarding and heartbeat monitoring
- Single-node PyTorch DistributedDataParallel process-group setup
- Resume and extend operations for saved sessions
- Analysis sessions driven by a trained-session checkpoint
- Built-in model loading, logging, checkpointing, DDP, data management, optimization, timing, TensorBoard, and infinite samplers

## Table of contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Component registration and discovery](#component-registration-and-discovery)
- [Component dependencies](#component-dependencies)
- [Session lifecycle](#session-lifecycle)
- [Shared contexts](#shared-contexts)
- [Configuration and command-line interface](#configuration-and-command-line-interface)
- [Analysis sessions](#analysis-sessions)
- [Process-monitoring options](#process-monitoring-options)
- [Process model and supervision](#process-model-and-supervision)
- [Distributed training with DDP](#distributed-training-with-ddp)
- [Checkpointing, resume, and extension](#checkpointing-resume-and-extension)
- [Built-in components](#built-in-components)
- [Infinite samplers](#infinite-samplers)
- [Direct session execution](#direct-session-execution)
- [API summary](#api-summary)
- [Testing](#testing)
- [Current behavior and limitations](#current-behavior-and-limitations)
- [Project layout](#project-layout)
- [License](#license)

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

## Requirements

The package currently declares:

- Python 3.12 or newer
- PyTorch 2.11 or newer
- NumPy 2.4.4 or newer
- OmegaConf 2.3 or newer
- Matplotlib 3.10.9 or newer
- TensorBoard 2.20 or newer
- PyYAML
- pytest

See [`pyproject.toml`](./pyproject.toml) for the complete dependency list.

## Installation

### Install from GitHub

```bash
python -m pip install \
  "git+https://github.com/abhishekbisht1429/training-framework.git@main"
```

### Development installation

```bash
git clone https://github.com/abhishekbisht1429/training-framework.git
cd training-framework
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quick start

This example defines a stateful resource, a training step, and a lifecycle hook. The framework discovers the component module, creates the components from YAML, spawns a worker, and runs five iterations.

### 1. Create a project package

```text
my_project/
├── __init__.py
├── config.yaml
├── train.py
└── components/
    ├── __init__.py
    └── demo.py
```

The package containing decorated components must be importable by both the parent process and spawned worker processes.

### 2. Define components

Create `my_project/components/demo.py`:

```python
from training_framework.components import (
    LifecycleHook,
    StatefulResource,
    Step,
    hook,
    requires_resource,
    resource,
    step,
)
from training_framework.session import TrainingSession


@resource("counter")
class CounterResource(StatefulResource):
    def __init__(self, config: dict):
        self.value = int(config.get("start", 0))

    def setup(self, session: TrainingSession) -> None:
        pass

    def teardown(self, session: TrainingSession) -> None:
        pass

    def get_state(self) -> dict[str, int]:
        return {"value": self.value}

    def set_state(self, state: dict[str, int]) -> None:
        self.value = state["value"]


@step("increment")
@requires_resource("counter")
class IncrementStep(Step):
    def __init__(self, config: dict):
        self.amount = int(config.get("amount", 1))

    def run(self, session: TrainingSession) -> None:
        counter = session.get_resource("counter")
        counter.value += self.amount
        session.iteration_context["counter_value"] = counter.value


@hook("progress")
@requires_resource("counter")
class ProgressHook(LifecycleHook):
    def __init__(self, config: dict):
        self.call_every = int(config.get("call_every", 1))

    def setup(self, session: TrainingSession) -> None:
        pass

    def teardown(self, session: TrainingSession) -> None:
        pass

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        pass

    def post_iteration_callback(self, session: TrainingSession) -> None:
        value = session.iteration_context["counter_value"]
        print(f"iteration={session.iteration}, counter={value}")
```

Each configured component class receives its YAML mapping as one `config` argument.

### 3. Create the YAML configuration

Create `my_project/config.yaml`:

```yaml
sessions:
  - session_config:
      rng_seed: 42
      sessions_dir: ./runs
      max_iterations: 5
      components_package: my_project.components
      device: cpu

    counter:
      start: 0

    increment:
      amount: 2

    progress:
      call_every: 1

    logger:
      log_every: 1

    checkpointer:
      checkpoint_every: 5
```

Every top-level key inside a session, other than the reserved `session_type`,
`session_config`, `session_kwargs`, `components`, and `aliases` entries, must
match a component visible to the active session type.

### 4. Create the entry point

Create `my_project/train.py`:

```python
from training_framework.engine import Configurator
from training_framework.engine import TrainingEngine


def main() -> None:
    configurator = Configurator()

    with TrainingEngine(configurator) as engine:
        engine.start_session()


if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__"` guard is required for safe process spawning.

### 5. Run training

From the directory containing `my_project`:

```bash
python -m my_project.train --config my_project/config.yaml
```

The parent process creates the session and worker configuration. The worker process reconstructs the session and executes the configured components.

## Core concepts

### Resource

A resource owns an object or service whose lifecycle follows the session context.

```python
from training_framework.components import Resource, resource
from training_framework.session import TrainingSession


@resource("dataset")
class DatasetResource(Resource):
    def __init__(self, config: dict):
        self.config = config
        self.dataset = None

    def setup(self, session: TrainingSession) -> None:
        self.dataset = build_dataset(self.config)

    def teardown(self, session: TrainingSession) -> None:
        self.dataset = None
```

Resources are set up before session hooks and torn down in reverse resource order.

### Hook

The framework provides three hook interfaces:

| Interface | Methods |
|---|---|
| `SessionHook` | `setup(session)`, `teardown(session)` |
| `IterationHook` | `pre_iteration_callback(session)`, `post_iteration_callback(session)` |
| `LifecycleHook` | All four methods |

An iteration hook must expose a positive `call_every` integer. An iteration hook is selected when:

- the current iteration is the first iteration;
- the current iteration is the configured final iteration; or
- `session.iteration % hook.call_every == 0`.

The first and final iterations therefore invoke every iteration hook, regardless of `call_every`.

### Step

A step performs one unit of work during every iteration:

```python
from training_framework.components import Step, step
from training_framework.session import TrainingSession


@step("train")
class TrainStep(Step):
    def __init__(self, config: dict):
        self.config = config

    def run(self, session: TrainingSession) -> None:
        # Forward pass, loss, backward pass, optimizer update, and so on.
        ...
```

Steps are executed in dependency order.

### Stateful components

Components with mutable state that must survive checkpointing can inherit one of:

- `StatefulResource`
- `StatefulStep`
- `StatefulSessionHook`
- `StatefulIterationHook`
- `StatefulLifecycleHook`

They must implement:

```python
def get_state(self):
    ...


def set_state(self, state) -> None:
    ...
```

The framework captures each component's constructor arguments and uses them to reconstruct the component before calling `set_state()`.

## Component registration and discovery

Register classes with decorators:

```python
@resource("model")
class ModelResource(Resource):
    ...


@hook("metrics")
class MetricsHook(LifecycleHook):
    ...


@step("optimizer_step")
class OptimizerStep(Step):
    ...
```

Registration is global within each Python interpreter. Omitting
`session_type` registers a shared component; providing it registers a scoped
component visible only to matching sessions. A scoped registration overrides a
shared component with the same name for that session type. Duplicate names
within the same scope raise `ValueError`. Application code should register
components through the public decorators and interact with active instances
through a concrete `Session`.

`session_config.components_package` identifies the package that contains application components. During session initialization, the framework:

1. imports that package;
2. recursively discovers its submodules with `pkgutil.walk_packages()`;
3. imports every discovered module; and
4. relies on module-level decorators to populate the component registry.

For reliable spawn and checkpoint behavior:

- define component classes at module scope;
- return the original class from custom decorators;
- avoid registration that depends on process ID, rank, or other process-specific state;
- make the component package importable from a fresh Python interpreter; and
- keep component names stable across checkpoint save and restore.

### Selecting components

A top-level component mapping both activates a component and supplies its
constructor configuration. Existing mappings, including empty mappings, remain
valid:

```yaml
train:
  gradient_accumulation: 4
metrics: {}
```

For components that need no configuration, list their names under the special
`components` entry instead:

```yaml
components:
  - model
  - dataloader
  - train
```

The list must contain unique, non-empty component names or virtual alias roles.
Every name must resolve to a registered component. If a name appears in both
places, its explicit mapping wins. The framework recursively activates
resources, hooks, steps, and wrapped hooks required by the selected roots, using
an empty configuration for each automatically activated component. Unrelated
registered components stay inactive. If an automatically activated component
requires constructor settings, add its top-level mapping.

`TrainingSession` activates `logger` and `checkpointer` by default.
`AnalysisSession` instead activates `trained_model` and its analysis-specific
`logger`. Both special entries and component dependencies support aliases.

### Component aliases

Use the session-level `aliases` mapping to substitute a registered implementation
for the component name expected by configuration and dependency decorators:

```yaml
aliases:
  optimizer: my_custom_optimizer

optimizer:
  learning_rate: 0.001
```

The mapping direction is `expected_name: registered_name`. In this example,
`my_custom_optimizer` must be registered, while `optimizer` may be either a
registered name or a virtual role. When configuration is needed, it remains
under `optimizer`; otherwise the role may be selected through `components`, a
dependency, or a built-in default without a separate mapping. Dependencies such
as `@requires_step("optimizer")` resolve to `my_custom_optimizer`. The actual
registered name is used in component state and
shown in the execution graph, which also includes an `ALIASES` section.

Aliases are session-scoped and one-to-one. Alias chains, cycles, unknown or
ambiguous targets, category changes, and configuring both names are rejected.
Built-in defaults such as `logger` and `checkpointer` can be replaced through the
same mechanism. `ddp.parallel_components` may contain either expected or actual
names; an aliased DDP resource must support the same `config` and `rank`
construction interface as the built-in resource.

## Component dependencies

Dependencies are declared using registry names:

```python
from training_framework.components import (
    LifecycleHook,
    Step,
    hook,
    requires_hook,
    requires_resource,
    requires_step,
    step,
    wraps,
)


@step("optimizer_step")
@requires_step("backward")
@requires_hook("metrics")
@requires_resource("optimizer")
class OptimizerStep(Step):
    ...
```

Supported dependency directions are:

| Consumer | May require a resource | May require a hook | May require a step |
|---|:---:|:---:|:---:|
| Resource | Yes | No | No |
| Hook | Yes | No | No |
| Step | Yes | Yes | Yes |

A Hook declares nesting separately:

```python
@hook("outer")
@wraps("inner")
class OuterHook(LifecycleHook):
    ...
```

If `outer` wraps `inner`, setup and pre-iteration callbacks run outer
then inner, while post-iteration callbacks and teardown run inner then outer.
Wrapping names support session aliases. Hooks must share a lifecycle phase. For
two iteration-capable hooks, the wrapper's `call_every` must be a positive
multiple of the wrapped hook's value. This lets the wrapper run less often while
ensuring it never runs on an iteration where the wrapped hook does not run.

The framework builds a registry-wide ordering graph and performs a topological
sort. It rejects missing, incorrectly typed, or unconfigured targets; wrapping
relationships without a shared lifecycle phase; cadence mismatches; and cycles.

The resulting order controls resource setup, hook callbacks, and step execution.
Teardown and post-iteration hook callbacks use reverse order.

Selecting a component automatically activates its recursive dependency and
wrapping-target closure with empty configurations. Activation follows dependency
edges outward: selecting a wrapped hook alone does not select hooks that wrap it.
For DDP, secondary ranks retain the same closure for each root named in
`ddp.parallel_components`.

## Session lifecycle

Both `TrainingSession` and `AnalysisSession` are context managers and iterators.

```text
Construct session
    |
    +-- import component package
    +-- instantiate configured components
    |
Enter session
    +-- resource.setup() in dependency order
    +-- SessionHook.setup() in dependency order
    +-- create session directory and write config.yaml for rank 0
    |
Each iteration
    +-- selected IterationHook.pre_iteration_callback()
    +-- Step.run() in dependency order
    +-- selected IterationHook.post_iteration_callback() in reverse order
    +-- clear iteration_context
    |
Exit session
    +-- SessionHook.teardown() in reverse dependency order
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

## Configuration and command-line interface

### YAML structure

The configuration root must contain a `sessions` list:

```yaml
sessions:
  - session_config:
      rng_seed: 42
      sessions_dir: ./runs
      max_iterations: 1000
      components_package: my_project.components
      device: cuda:0

    model:
      hidden_size: 512

    train:
      learning_rate: 0.0003

    logger:
      log_every: 10
```

`session_config` fields:

| Field | Required | Meaning |
|---|:---:|---|
| `rng_seed` | Yes | Seed used for Python, NumPy, PyTorch, and CUDA RNG initialization |
| `sessions_dir` | Yes | Parent directory for timestamped session directories |
| `max_iterations` | Yes | Number of training or analysis iterations |
| `components_package` | Yes | Importable package recursively scanned for decorated components |
| `device` | No | Requested device string; defaults to CPU, and unavailable CUDA requests currently fall back to CPU |
| `show_execution_graph` | No | Print the resolved lifecycle graph on entry; defaults to `true` |

A session directory is created as:

```text
<sessions_dir>/session_YYYYMMDD_HHMMSS/
```

The resolved session configuration is written to `config.yaml` in that directory.

Every `sessions[]` entry is registered for the same engine run. Entries may use different registered session types; all resulting worker wrappers start together.

### New session

```bash
python -m my_project.train --config my_project/config.yaml
```

### OmegaConf overrides

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --override \
  'sessions[0].session_config.max_iterations=2000' \
  'sessions[0].logger.log_every=25'
```

Overrides are applied to `--config` session definitions.

### Resume a checkpoint

```bash
python -m my_project.train \
  --resume-session ./runs/session_.../checkpoints/<checkpoint-name>
```

### Extend a checkpoint

To restore a checkpoint and replace its maximum iteration count:

```bash
python -m my_project.train \
  --extend-session ./runs/session_.../checkpoints/<checkpoint-name> 5000
```

## Analysis sessions

Analysis sessions use the same resource, hook, step, context, and iteration
lifecycle as training sessions. Their effective registry combines shared
components with components registered using `session_type="analysis"`:

```python
from training_framework.components import Step, requires_resource, step


@step("report", session_type="analysis")
@requires_resource("trained_model")
class ReportStep(Step):
    def __init__(self, config):
        self.output_path = config["output_path"]

    def run(self, session):
        model = session.get_resource("trained_model").model
        # Analyze the model and write the configured report.
        ...
```

An analysis configuration uses the same `sessions` structure. The built-in
`trained_model` resource and analysis logger are default roots, so only
application analysis components need to be selected explicitly:

```yaml
sessions:
  - session_type: analysis

    session_config:
      rng_seed: 42
      sessions_dir: ./analysis-runs
      max_iterations: 1
      components_package: my_project.analysis_components
      device: cpu
      show_execution_graph: true

    session_kwargs:
      model_checkpoint_path: ./runs/session_.../checkpoints/<checkpoint-name>

    report:
      output_path: ./analysis-runs/report.json
```

For direct execution, construct the concrete analysis subclass:

```python
from training_framework.session import AnalysisSession


session = AnalysisSession(
    analysis_config,
    model_checkpoint_path="./runs/session_.../checkpoints/<checkpoint-name>",
)
```

Run the analysis entry through the same generic config path:

```bash
python -m my_project.train --config my_project/analysis.yaml
```

The checkpoint must contain a framework `TrainingSession`, not a standalone
model state dictionary. The source session must expose a model through the
`model` resource role, directly or through an alias, and that resource must
provide `to(device)` and `eval()`. During analysis setup, `trained_model` loads
the source session on CPU, moves the recovered model to the analysis device,
places it in evaluation mode, and exposes it through
`session.get_resource("trained_model").model`. Gradients remain enabled for
attribution-style analyses. Analysis does not activate the training
checkpointer by default.

Only load trusted checkpoints because session loading uses unrestricted Python
deserialization.

## Process-monitoring options

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --heartbeat-timeout 60 \
  --process_timeout_on_join 30
```

| Argument | Default | Meaning |
|---|---:|---|
| `--heartbeat-timeout` | `30.0` | Maximum seconds a live worker may go without a heartbeat |
| `--process_timeout_on_join` | `30.0` | Graceful-shutdown period before surviving workers are terminated |

The three operations, `--config`, `--resume-session`, and
`--extend-session`, are mutually exclusive and one is required. New-session
type selection and constructor-specific arguments belong to each
`sessions[]` entry.

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

The stop event is checked between iterations. A step, hook, setup, or teardown call that is already running is not interrupted cooperatively.

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

Transient infrastructure, such as the selected device, iteration context, manager pipe, and heartbeat timer, is recreated in each worker.

### Resume

```bash
python -m my_project.train --resume-session <checkpoint-path>
```

The engine loads the session in the parent, recovers the saved DDP world size when present, sends session state to workers, and continues from the saved iteration.

### Extend

```bash
python -m my_project.train --extend-session <checkpoint-path> <new-max-iterations>
```

The session is restored as in the resume operation, then each worker receives the new maximum iteration count before training starts.
Extension is training-specific; the checkpoint must contain a `TrainingSession`.

### Checkpoint safety

Checkpoint loading uses `torch.load(..., weights_only=False)`, which can execute arbitrary code through Python deserialization. Load only checkpoints from trusted sources.

Exact training continuation also depends on application state. Persist model, optimizer, scheduler, scaler, sampler, and any data-pipeline state that affects the next batch.

## Built-in components

Importing `training_framework` initializes `training_framework.components`,
which registers all built-ins. Their classes are also importable from
`training_framework.components.builtin`.

### Training built-ins

| Name | Kind | Purpose and dependencies |
|---|---|---|
| `logger` | Hook | Prints `Iteration <current>/<maximum>`; enabled by default |
| `checkpointer` | Hook | Saves complete session checkpoints; enabled by default |
| `ddp` | Resource | Initializes distributed execution and wraps the required `model` resource |
| `data_manager` | Stateful resource | Creates a resumable distributed `DataLoader`; requires `dataset` and `ddp` |
| `optimizer` | Stateful lifecycle hook | Runs AdamW and a warmup/cosine scheduler around each iteration; requires `ddp` and reads `iteration_context["loss"]` |
| `timer` | Lifecycle hook | Reports iteration and elapsed durations; wraps `optimizer` |
| `tensorboard` | Resource | Starts TensorBoard and exposes a `SummaryWriter` |

The training defaults are equivalent to:

```yaml
logger:
  log_every: 10
  # log_file: ./runs/train.log  # optional; stdout when omitted

checkpointer:
  checkpoint_every: 100
```

An explicit component mapping replaces its default mapping, so retain required
fields such as `log_every` and `checkpoint_every` when overriding defaults.
The analysis logger likewise defaults to `log_every: 10`.

The optional training built-ins use these configurations:

```yaml
ddp:
  world_size: 1
  backend: gloo
  master_addr: "127.0.0.1"
  master_port: "12355"
  parallel_components: []

data_manager:
  batch_size: 32       # global batch size; divisible by world_size
  num_workers: 0
  pin_memory: false

optimizer:
  learning_rate: 0.0003
  weight_decay: 0.01
  warmup_iters: 100

timer:
  call_every: 10

tensorboard:
  host: "127.0.0.1"
  port: 6006
  logdir: ./runs/tensorboard  # optional TensorBoard server log directory
```

`data_manager.data_iter` is available only while the session is active. It
divides the global batch size across ranks and checkpoints delivered-batch
progress. `optimizer` expects a loss tensor in `session.iteration_context` and
performs zeroing, backward propagation, optimization, and scheduler advancement.

The TensorBoard resource starts the external `tensorboard` command, creates a
PyTorch `SummaryWriter`, and exposes it through `summary_writer`:

```python
tensorboard = session.get_resource("tensorboard")
tensorboard.summary_writer.add_scalar(
    "train/loss",
    loss,
    session.iteration,
)
```

The executable must be available and the selected port must be free. Teardown
closes the writer and terminates the external process.

### Analysis built-ins

| Name | Kind | Purpose |
|---|---|---|
| `trained_model` | Resource | Loads the `model` role from the source training checkpoint; enabled by default |
| `logger` | Hook | Prints `Analysis iteration <current>/<maximum>`; enabled by default |

These names live in the analysis registry and therefore do not conflict with
training components that use the same names.

## Infinite samplers

### `InfiniteSampler`

`InfiniteSampler` repeatedly yields random permutations of dataset indices:

```python
from torch.utils.data import DataLoader
from training_framework.dataloader import InfiniteSampler


sampler = InfiniteSampler(len(dataset))
loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
)
```

It has no natural end. Use the session's `max_iterations` to bound training.

### `DistributedInfiniteSampler`

`DistributedInfiniteSampler` creates one deterministic, rank-specific slice of a shuffled global index sequence for each logical epoch:

```python
from training_framework.dataloader import DistributedInfiniteSampler


sampler = DistributedInfiniteSampler(
    num_samples=len(dataset),
    rank=rank,
    world_size=world_size,
    shuffle=True,
    seed=42,
    drop_last=False,
)
```

When rank and world size are omitted, it resolves them from an initialized PyTorch distributed process group, or falls back to rank 0 and world size 1.

It exposes:

```python
state = sampler.get_state()
sampler.set_state(state)
```

The iterator is infinite even though `len(sampler)` reports one rank-local logical epoch.

> **Checkpointing note:** with `DataLoader(num_workers > 0)`, sampler indices may be prefetched before their batches are consumed. Treat exact mid-epoch sampler restoration as experimental and track consumed progress in the training loop when exact replay matters.

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
[Analysis sessions](#analysis-sessions), then enter and iterate it with the same
pattern. Direct execution bypasses spawned-worker supervision, error pipes,
heartbeat monitoring, and rank-specific DDP reconstruction. Use
`TrainingEngine` for the normal managed execution path.

## API summary

The supported public imports are grouped by responsibility:

```python
from training_framework.components import Resource, Step, resource, step
from training_framework.components.builtin import Checkpointer, TrainedModel
from training_framework.engine import Configurator, TrainingEngine
from training_framework.session import (
    AnalysisSession,
    TrainingSession,
    register_session_type,
)
```

`training_framework.dataloader` remains the public home of the infinite
samplers.

### `Configurator`

| Member | Purpose |
|---|---|
| `Configurator()` | Parse command-line operation and options |
| `operation` | `new`, `resume`, or `extend` |
| `session_configs` | Deep copy of parsed YAML session definitions in the new operation |
| `checkpoint_path` | Checkpoint path in resume or extend operations |
| `new_max_iters` | New iteration limit in the extend operation |
| `heartbeat_timeout` | Worker heartbeat deadline |
| `process_timeout_on_join` | Graceful process-join timeout |
| `get_component_config(session_index, key)` | Return a deep copy of one component mapping, or `{}` for a listed no-config component |
| `get_all_component_configs(session_index)` | Return all selected component configs, excluding special entries |

### `TrainingEngine`

| Member | Purpose |
|---|---|
| `TrainingEngine(configurator)` | Create a process manager from CLI configuration |
| `start_session()` | Start all worker ranks for the active session; requires engine context |
| `register_session(config, *, session_type="training", session_kwargs=None)` | Construct a registered session type and its worker wrappers |
| `load_session(path, session_update_params=None)` | Load a checkpoint and prepare worker wrappers |
| `request_stop_all()` | Request cooperative shutdown of started workers |

Normal usage is:

```python
with TrainingEngine(Configurator()) as engine:
    engine.start_session()
```

The engine monitors workers while leaving the context.

### `Session`, `TrainingSession`, and `AnalysisSession`

| Member | Purpose |
|---|---|
| `Session` | Abstract base implementing the shared component and iteration lifecycle |
| `TrainingSession(config)` | Concrete training session with logger/checkpointer defaults and extension support |
| `AnalysisSession(config, *, model_checkpoint_path)` | Concrete analysis session with trained-model/logger defaults |
| `session_type` | Registered string identifying the concrete session workflow |
| `AnalysisSession.model_checkpoint_path` | Source training checkpoint used by analysis |
| `session_config` | Frozen `SessionConfig` containing seed, directory, and max iterations |
| `iteration` | Current completed/in-progress iteration counter |
| `device` | Active `torch.device` |
| `session_context` | Session-lifetime shared dictionary |
| `iteration_context` | Current-iteration shared dictionary; context-only |
| `component_aliases` | Copy of the session's expected-to-actual alias bindings |
| `resolve_component_name(name)` | Resolve an expected or actual component name to its registered name |
| `get_resource(name)` | Retrieve a configured resource |
| `has_resource(name)` | Test whether a resource is present |
| `get_all_resources()` | Return configured resources |
| `get_all_hooks()` | Return configured hooks |
| `get_all_steps()` | Return configured steps |
| `register_resource(resource)` | Add a registered resource instance |
| `register_hook(hook)` | Add a registered hook instance |
| `add_step(step)` | Add a registered step instance |
| `unregister_resource(name)` | Remove a resource from the session |
| `unregister_hook(name)` | Remove a hook from the session |
| `remove_step(name)` | Remove a step from the session |
| `get_state()` | Capture serializable session state |
| `set_state(state)` | Restore state into a session |
| `Session.from_state(state)` | Reconstruct and dispatch to the concrete session class recorded in state |
| `TrainingSession.update_max_iters(value)` | Replace a training session's maximum iteration count |

### Registration decorators

| API | Purpose |
|---|---|
| `@resource(name, session_type=None)` | Register a shared Resource, or scope it to one session type |
| `@hook(name, session_type=None)` | Register a shared Hook, or scope it to one session type |
| `@step(name, session_type=None)` | Register a shared Step, or scope it to one session type |
| `@requires_resource(name)` | Declare a resource prerequisite |
| `@requires_hook(name)` | Declare a Hook prerequisite for a Step |
| `@requires_step(name)` | Declare a Step prerequisite for a Step |
| `@wraps(name)` | Declare that a Hook wraps another Hook |
| `component_registry(session_type)` | Return shared components overlaid by the matching scoped registry |
| `topological_sort_of_components(..., session_type=...)` | Validate and order the selected session type's component graph |
| `@register_session_type(name)` | Register a concrete Session subclass for engine and checkpoint dispatch |

## Testing

Install the package in editable mode and run:

```bash
python -m pip install -e .
python -m pytest
```

Run a focused file with:

```bash
python -m pytest src/tests/test_engine.py -q
```

The GitHub Actions workflow runs the suite on Python 3.12 and Python 3.13.

The current tests cover areas including:

- config-driven session execution;
- component registration and dependency validation;
- lifecycle and shared-context behavior;
- state and RNG restoration;
- checkpoint resume and extension;
- analysis session-type registration, model recovery, and spawned execution;
- real spawned-worker continuation and error propagation;
- rank-specific DDP session reconstruction; and
- heartbeat/process-monitoring behavior.

## Current behavior and limitations

1. **Configured sessions start as one concurrent batch.** Every `sessions[]` entry contributes worker wrappers to the run; the engine does not provide sequencing or dependency ordering between sessions.

2. **DDP is currently single-node oriented.** The process rank is used directly as the CUDA device index. There is no separate local-rank abstraction for multi-node execution.

3. **The built-in DDP resource requires a compatible model resource.** The `model` role must resolve to a module accepted by PyTorch DDP. Distributed forward passes should use `session.get_resource("ddp").wrapped_model` while the session is active.

4. **Secondary-rank component roots are opt-in.** Ranks greater than zero retain roots in `ddp.parallel_components`, their recursive dependencies and wrapping targets, plus the DDP resource.

5. **Component registration is global per interpreter.** Resource, hook, and step names share one namespace within each shared or session-specific scope. Duplicate names in one scope fail; a matching scoped component overrides a shared component with the same name. Test suites that reset registration must account for Python's module import cache before expecting decorators to run again.

6. **Spawn requires importable and serializable definitions.** Define worker targets and component classes at module scope. Constructor arguments, state returned by `get_state()`, and checkpointed session-context values must be serializable.

7. **Heartbeat detection happens between framework stages.** A single long-running component call can exceed the deadline without sending another heartbeat. Set `--heartbeat-timeout` above the longest expected uninterrupted setup, hook, step, or teardown operation.

8. **Graceful stopping occurs between iterations.** A worker already inside a component call will finish or block there until the join timeout causes termination.

9. **Checkpoint files are trusted-code artifacts.** The built-in loader uses unrestricted Python deserialization. Never load an untrusted checkpoint.

10. **Exact data-pipeline replay is application-dependent.** DataLoader prefetching can move a sampler's issued position ahead of consumed batches. Persist and restore committed batch progress when exact continuation is required.

11. **An unavailable CUDA device currently falls back to CPU.** Validate the final `session.device` in application code when silent fallback is undesirable.

12. **TensorBoard is an external process.** Starting it requires an available executable and port, and including it in DDP parallel components would start one server per retained rank.

## Project layout

```text
training-framework/
├── .github/
│   └── workflows/
│       └── python-tests.yaml
├── src/
│   ├── training_framework/
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── config.py
│   │   │   ├── graph.py
│   │   │   ├── registry.py
│   │   │   └── builtin/
│   │   │       ├── __init__.py
│   │   │       ├── analysis.py
│   │   │       ├── checkpointing.py
│   │   │       ├── data.py
│   │   │       ├── distributed.py
│   │   │       ├── observability.py
│   │   │       └── optimization.py
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── core.py
│   │   │   ├── supervision.py
│   │   │   └── worker.py
│   │   ├── session/
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── base.py
│   │   │   ├── components.py
│   │   │   ├── config.py
│   │   │   ├── io.py
│   │   │   ├── runtime.py
│   │   │   ├── state.py
│   │   │   └── training.py
│   │   ├── dataloader.py
│   │   └── util.py
│   └── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```

## License

This repository is licensed under the [Apache License 2.0](./LICENSE).

---
**Author Note**
This README was generated using ChatGPT. Although, I have done an overview of it, please open an issue if you find anything missing and inconsistent.
