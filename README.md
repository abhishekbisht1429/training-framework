# How to add dependency for this project

In your `pyproject.toml` add the following

dependencies = [
    "training-framework @ git+ssh://git@github.com:abhishekbisht1429/training-framework.git"
]
# Training Framework Documentation

## Overview

The Training Framework is a modular, extensible framework for building machine learning training pipelines. It separates training into four core concepts:

- **TrainingSession** – Encapsulates a single training run.
- **TrainingEngine** – Executes one or more sessions, optionally in parallel.
- **Resources** – Long-lived objects such as loggers, TensorBoard, datasets, or checkpoints.
- **Steps** – Individual units of work executed once per iteration.
- **Hooks** – Callbacks executed before and/or after iterations.

The framework emphasizes:

- Modular composition
- Configurable execution
- Automatic component registration
- Session serialization/restoration
- Multi-threaded execution

---

# Architecture

```
TrainingEngine
│
├── TrainingSession 1
│   ├── Resources
│   ├── Hooks
│   └── Steps
│
├── TrainingSession 2
│   ├── Resources
│   ├── Hooks
│   └── Steps
│
└── ...
```

Each session is independent and owns all state required for one training experiment.

---

# Core Components

## TrainingEngine

Responsible for:

- registering sessions
- spawning one worker thread per session
- coordinating lifecycle
- graceful shutdown

### Responsibilities

- Register sessions
- Execute sessions concurrently
- Stop sessions cleanly
- Wait for worker completion on shutdown

### Public API

```python
engine.register_session(session)
engine.run_all(wait=True)
```

---

## TrainingSession

Represents a complete training workflow.

A session owns:

- configuration
- resources
- hooks
- steps
- shared state
- iteration counter

Typical lifecycle:

```
NEW
 ↓
READY
 ↓
RUNNING
 ↓
PAUSED / FINISHED
```

### Responsibilities

- Execute one training iteration
- Manage resources
- Invoke hooks
- Execute steps
- Share values between components
- Serialize and restore itself

---

# Resources

Resources are long-lived objects whose lifetime matches the session.

Examples:

- Logger
- TensorBoard
- Checkpoint manager
- Dataset

Lifecycle:

```
setup()
   │
training...
   │
teardown()
```

Resources are initialized exactly once.

---

# Steps

Steps perform the actual work of an iteration.

Examples:

- Forward pass
- Backward pass
- Optimizer step
- Scheduler step
- Validation

Execution order:

```
Hook (pre)
 ↓
Step 1
 ↓
Step 2
 ↓
...
 ↓
Hook (post)
```

Steps may publish values into the session shared state.

Example:

```python
session.iteration_context["loss"] = loss
```

---

# Hooks

Hooks provide callback functionality.

Typical uses:

- logging
- checkpointing
- metric computation
- visualization
- validation

Hooks execute before and/or after an iteration depending on configuration.

---

# Component Registration

Components are registered using decorators.

Example:

```python
@step("optimizer")
class OptimizerStep(Step):
    ...
```

Registries map names from configuration files to Python classes.

---

# Configuration

The Configurator constructs sessions from YAML configuration.

Typical flow:

```
YAML
 ↓
Configurator
 ↓
TrainingSession
 ↓
Resources
Hooks
Steps
```

---

# Shared Session State

Steps and hooks communicate using the shared session state.

```
session.iteration_context[key] = value
value = session.iteration_context[key]
```

The shared state exists only during the current iteration and is cleared afterwards.

---

# Stateful Components

Components requiring persistence inherit from `Stateful`.

```
class MyStep(Step, Stateful):
```

They implement:

```python
get_state()
set_state(state)
```

This enables checkpointing and restoration.

---

# Automatic Constructor Capture

The framework uses `CaptureInitMeta` to automatically capture constructor arguments.

When a component is instantiated:

```python
Logger(config)
```

its initialization arguments are stored as:

```python
{
    "args": (...),
    "kwargs": {...}
}
```

This allows reconstruction without requiring every class to manually implement constructor serialization.

---

# Serialization

Session serialization stores:

- session configuration
- resources
- hooks
- steps
- constructor arguments
- state of Stateful objects

Restoration reconstructs each component using its stored constructor arguments before applying any saved state.

---

# Threading Model

Each registered session executes on its own worker thread.

```
TrainingEngine
│
├── Thread 1 → Session A
├── Thread 2 → Session B
└── Thread 3 → Session C
```

Each session has an independent active flag.

Engine shutdown:

1. Signal every session to stop.
2. Allow the current iteration to complete.
3. Join all worker threads.
4. Exit the engine context.

This guarantees that no worker thread remains active after engine shutdown.

---

# Extension Guide

## Creating a Resource

```python
@resource("logger")
class MyLogger(Resource):
    ...
```

## Creating a Step

```python
@step("optimizer")
class OptimizerStep(Step):
    ...
```

## Creating a Hook

```python
@hook("metrics")
class MetricsHook(Hook):
    ...
```

## Persistent Components

```python
class MyStep(Step, Stateful):

    def get_state(self):
        ...

    def set_state(self, state):
        ...
```

---

# Design Principles

- Separation of responsibilities
- Configuration-driven composition
- Automatic registration
- Automatic constructor capture
- Explicit persistence through Stateful
- Deterministic resource lifecycle
- Thread-safe concurrent session execution
- Extensibility via decorators

---

# Typical Execution Flow

```
Load YAML
        │
Configurator
        │
TrainingEngine
        │
Register Sessions
        │
Run Threads
        │
Session
        │
Resources.setup()
        │
Iteration
    │
    ├── Hook (pre)
    ├── Step 1
    ├── Step 2
    ├── ...
    └── Hook (post)
        │
Resources.teardown()
        │
Session Finished
```
