# Training Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package version](https://img.shields.io/badge/version-0.5.0-blue.svg)](./pyproject.toml)
[![Python tests](https://github.com/abhishekbisht1429/training-framework/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/abhishekbisht1429/training-framework/actions/workflows/python-tests.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)

A component-based framework for building, running, checkpointing, and supervising PyTorch training and trained-model analysis workflows.

Workflow code is organized into reusable **resources**, **hooks**, and **steps**. `TrainingSession` and `AnalysisSession` share the lifecycle implemented by the abstract `Session` base, while using separate component registries and defaults. `TrainingEngine` constructs or restores the appropriate session in the parent process, serializes its state, launches one or more spawned workers, and monitors them for completion, errors, interrupts, and missed heartbeats.

> **Project status:** This project is under active development. The current API is suitable for experimentation and framework development, but review [Current behavior and limitations](docs/concepts/limitations.md) before using it for long-running or production workloads.


## Features

- Configuration-driven session construction with YAML and OmegaConf overrides
- Decorator-based registration and recursive component-package discovery
- Shared components with optional per-session-type registry overrides
- Explicit resource, hook, and step dependencies
- Opt-in rollback for partially initialized resources and session hooks
- Topological execution ordering and dependency-cycle detection
- Built-in batch, forward and loss steps configured in YAML, ordered by the `iteration_context` keys they read and write
- Session-level and iteration-level shared contexts
- Stateful component checkpointing and session restoration
- Restoration of Python, NumPy, PyTorch, and CUDA RNG state
- Spawn-based worker processes with error forwarding and heartbeat monitoring
- Single-node PyTorch DistributedDataParallel setup with coordinated stopping
- Resume and extend operations for saved sessions
- Analysis sessions driven by a trained-session checkpoint
- Built-in model loading, logging, checkpointing, DDP, data management, optimization, timing, TensorBoard, and infinite samplers

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
    reads,
    requires_resource,
    resource,
    step,
    writes,
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
@writes("counter_value")
class IncrementStep(Step):
    def __init__(self, config: dict):
        self.amount = int(config.get("amount", 1))

    def run(self, session: TrainingSession) -> int:
        counter = self.get_dependency("counter")
        counter.value += self.amount
        return counter.value


@hook("progress")
@reads("counter_value")
class ProgressHook(LifecycleHook):
    def __init__(self, config: dict):
        self.call_every = int(config.get("call_every", 1))

    def pre_session(self, session: TrainingSession) -> None:
        pass

    def post_session(self, session: TrainingSession) -> None:
        pass

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        pass

    def post_iteration_callback(
        self, session: TrainingSession, counter_value: int
    ) -> None:
        print(f"iteration={session.iteration}, counter={counter_value}")
```

Each configured component class receives its YAML mapping as one `config` argument.
The step and the hook share a value through the iteration's context: `@writes`
stores what `run` returns under `counter_value`, and `@reads` passes it to
`post_iteration_callback` as the `counter_value` argument (see
[Ordering by dataflow](docs/guide/02-wiring-components.md#ordering-by-dataflow)).

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
`session_config`, `session_kwargs`, and `component_bindings` entries, must
match a component visible to the active session type. A binding maps a role name
to a registered implementation, and component configuration belongs under that
implementation name. The former `aliases` key remains accepted with a
deprecation warning. Use an empty mapping for a config-free root.
Required components that inherit `Component.__init__` are activated without a
mapping; required components with a custom constructor must have one. The former
top-level `components` list is rejected with migration guidance.

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

For debugger-managed worker processes, pass `--debug`. Workers are spawned
normally, and the parent waits for them using plain process joins without
heartbeat, failure, timeout, or termination monitoring:

```bash
python -m my_project.train --config my_project/config.yaml --debug
```

## Training a real model

The quick start writes its own step. For ordinary training you do not have to:
the built-in `load_batch`, `forward` and `compute` steps and the `optimizer`
resource do the work from configuration, so the only code is a dataset and a
model, registered as resources:

```python
import torch
from torch import nn

from training_framework.components import ModuleResource, Resource, resource


@resource("toy_dataset")
class ToyDataset(Resource):
    def __init__(self, config=None):
        self.inputs = torch.randn(256, 8)
        self.targets = (self.inputs.sum(dim=1) > 0).long()

    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


@resource("classifier")
class Classifier(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, inputs):
        return self.net(inputs)
```

In the session's YAML, next to `session_config`:

```yaml
    component_bindings: {model: classifier, dataset: toy_dataset}
    classifier: {}
    toy_dataset: {}
    ddp: {world_size: 1, backend: gloo}
    data_manager: {batch_size: 32, num_workers: 0, pin_memory: false}

    load_batch: {fields: [inputs, targets]}           # batch -> inputs, targets
    forward: {args: [inputs], outputs: logits}        # classifier(inputs)
    compute#loss: {function: cross_entropy, args: [logits, targets], outputs: loss}
    optimizer: {optimizer: {name: AdamW, kwargs: {lr: 0.001}}}
```

Each step names the `iteration_context` keys it reads and writes, and the
session orders the steps by them; `optimizer` brings in `backward` (which reads
`loss`) and the update after it. [Building an
iteration](docs/guide/03-building-an-iteration.md) walks through this run and
varies it: several loss terms, a second model, two views of one input.

## Documentation

The README covers installation and a complete first run. Everything else is in
the [documentation index](docs/README.md), which is organized in three tracks:

- **[Guide](docs/README.md#learn-it)** — a task-ordered path starting from
  [resources, hooks, and steps](docs/guide/01-resources-hooks-steps.md) and
  ending at [analysis sessions](docs/guide/07-analysis-sessions.md).
- **[Reference](docs/README.md#look-it-up)** —
  [built-in components](docs/reference/builtin-components.md),
  [generic steps](docs/reference/generic-steps.md),
  [optimization](docs/reference/optimization.md),
  [transformer blocks](docs/reference/transformer-blocks.md),
  [samplers](docs/reference/samplers.md),
  [CLI](docs/reference/cli.md) and [API](docs/reference/api.md).
- **[Concepts](docs/README.md#understand-it)** —
  [architecture](docs/concepts/architecture.md),
  [session lifecycle](docs/concepts/session-lifecycle.md),
  [the component model](docs/concepts/component-model.md),
  [`ModuleResource`](docs/concepts/module-resource.md) and
  [limitations](docs/concepts/limitations.md).

Working on the framework itself? See
[development and testing](docs/development.md).

## License

This repository is licensed under the [Apache License 2.0](./LICENSE).

---
**Author Note**
This README was generated using ChatGPT. Although, I have done an overview of it, please open an issue if you find anything missing and inconsistent.
