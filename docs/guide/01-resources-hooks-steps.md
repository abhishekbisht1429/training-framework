# Resources, Hooks, and Steps

[← Docs](../README.md) · [Project README](../../README.md)

Workflow code in this framework is written as **components**: resources, hooks,
and steps. This page introduces the three kinds, how to give one state that
survives a checkpoint, and how the framework finds your classes. It picks up
where the quick start in the project README leaves off, and assumes you have
run that example once.

## Resource

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

    def rollback_setup(self, session: TrainingSession) -> None:
        self.dataset = None

    def teardown(self, session: TrainingSession) -> None:
        self.dataset = None
```

`dataset` above is an example of a declared role: the built-in `DataManager`
requires a `dataset` Resource but the framework ships no implementation for
it. See [Component bindings](02-wiring-components.md#component-bindings) for how such roles are
declared and satisfied.

Resources are set up before session hooks and torn down in reverse resource order.

If a resource's own `setup()` raises, the framework calls its
`rollback_setup()` before tearing down resources that completed setup earlier.
The rollback method has a no-op default, so existing resource subclasses remain
valid. Override it when setup can leave partial external state, and make it safe
for every point at which setup can fail. Because setup did not complete,
rollback must not depend on context-guarded resource members.

## Hook

The framework provides three hook interfaces:

| Interface | Methods |
|---|---|
| `SessionHook` | `pre_session(session)`, `rollback_pre_session(session)`, `post_session(session)` |
| `IterationHook` | `pre_iteration_callback(session)`, `post_iteration_callback(session)` |
| `LifecycleHook` | All session and iteration hook methods |

An iteration hook must expose a positive `call_every` integer. An iteration hook is selected when:

- the current iteration is the first iteration;
- the current iteration is the configured final iteration; or
- `session.iteration % hook.call_every == 0`.

The first and final iterations therefore invoke every iteration hook, regardless of `call_every`.

If a session hook's own `pre_session()` raises, the framework calls its
`rollback_pre_session()`. This method also has a no-op default for backward
compatibility and should reverse only effects created by the incomplete
callback. The failing hook does not receive `post_session()`; hooks that
completed `pre_session()` earlier still receive normal reverse-order
post-session cleanup.

## Step

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

Steps are executed in dependency order: after the steps they require, and
after the steps that write the `iteration_context` keys they declare reading
(see [Ordering by dataflow](02-wiring-components.md#ordering-by-dataflow)).

For the common work -- taking a batch, calling a model, computing a loss --
you do not need to write a step: the built-in `load_batch`, `forward` and
`compute` do it from configuration. [Building an
iteration](03-building-an-iteration.md) shows how; write a step when you need
something they do not do.

A component that does not need constructor configuration or other initialization
may omit `__init__` entirely:

```python
@step("validate")
class ValidationStep(Step):
    def run(self, session: TrainingSession) -> None:
        ...
```

The inherited constructor accepts and ignores the component's configuration
mapping. Components that need configuration should continue to implement
`__init__(self, config)`.

## Stateful components

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

The framework captures each component's constructor arguments and uses them to
reconstruct the component before calling `set_state()`. Reconstruction happens
prerequisite-first, so a component that took hold of another one while it was
being built holds it again, and state is restored only after every component
exists.

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
- keep component names stable across checkpoint save and restore; and
- raise a component's `state_version` when what it checkpoints changes, with a
  `migrate_state` for older checkpoints (see
  [when a component changes](05-checkpoints-and-resume.md#when-a-component-changes)).

---

**Next:** [Wiring components together](02-wiring-components.md) — activating
components from YAML, binding roles to implementations, and declaring
dependencies between them.
