# Training Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Package version](https://img.shields.io/badge/version-0.3.4-blue.svg)](./pyproject.toml)
[![Python tests](https://github.com/abhishekbisht1429/training-framework/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/abhishekbisht1429/training-framework/actions/workflows/python-tests.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)

A component-based framework for building, running, checkpointing, and supervising PyTorch training and trained-model analysis workflows.

Workflow code is organized into reusable **resources**, **hooks**, and **steps**. `TrainingSession` and `AnalysisSession` share the lifecycle implemented by the abstract `Session` base, while using separate component registries and defaults. `TrainingEngine` constructs or restores the appropriate session in the parent process, serializes its state, launches one or more spawned workers, and monitors them for completion, errors, interrupts, and missed heartbeats.

> **Project status:** This project is under active development. The current API is suitable for experimentation and framework development, but review [Current behavior and limitations](docs/limitations.md) before using it for long-running or production workloads.


## Features

- Configuration-driven session construction with YAML and OmegaConf overrides
- Decorator-based registration and recursive component-package discovery
- Shared components with optional per-session-type registry overrides
- Explicit resource, hook, and step dependencies
- Opt-in rollback for partially initialized resources and session hooks
- Topological execution ordering and dependency-cycle detection
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

    def pre_session(self, session: TrainingSession) -> None:
        pass

    def post_session(self, session: TrainingSession) -> None:
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

## Documentation

The README covers installation and a complete first run. Detailed guides are
available in the [documentation index](docs/README.md):

- [Architecture and process model](docs/architecture.md)
- [Components](docs/components.md)
- [Sessions](docs/sessions.md)
- [Configuration and CLI](docs/configuration.md)
- [Distributed training](docs/distributed-training.md)
- [Checkpointing, resume, and extension](docs/checkpointing.md)
- [Built-in components and samplers](docs/built-in-components.md)
- [API summary](docs/api.md)
- [Development and testing](docs/development.md)
- [Current behavior and limitations](docs/limitations.md)

## License

This repository is licensed under the [Apache License 2.0](./LICENSE).

---
**Author Note**
This README was generated using ChatGPT. Although, I have done an overview of it, please open an issue if you find anything missing and inconsistent.
