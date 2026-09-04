# Components

[← Documentation index](README.md) · [Project README](../README.md)

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

    def rollback_setup(self, session: TrainingSession) -> None:
        self.dataset = None

    def teardown(self, session: TrainingSession) -> None:
        self.dataset = None
```

Resources are set up before session hooks and torn down in reverse resource order.

If a resource's own `setup()` raises, the framework calls its
`rollback_setup()` before tearing down resources that completed setup earlier.
The rollback method has a no-op default, so existing resource subclasses remain
valid. Override it when setup can leave partial external state, and make it safe
for every point at which setup can fail. Because setup did not complete,
rollback must not depend on context-guarded resource members.

### Hook

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
constructor configuration. Use an empty mapping to activate a root component
that needs no settings:

```yaml
train:
  gradient_accumulation: 4
metrics: {}
```

The framework recursively activates resources, hooks, steps, and wrapped hooks
required by those roots. A missing dependency is constructed automatically and
without arguments only when its effective constructor is the inherited
`Component.__init__`. If its class or a component base class defines another
constructor, add a top-level mapping for it. Unrelated registered components
stay inactive. The former top-level `components` list is no longer supported;
configs that contain it receive a migration error.

`TrainingSession` activates `logger` and `checkpointer` by default.
`AnalysisSession` instead activates `trained_model` and its analysis-specific
`logger`. Both special entries and component dependencies support component
bindings.

### Component bindings

Use the session-level `component_bindings` mapping to bind a component role
used by defaults and dependency decorators to a registered implementation:

```yaml
component_bindings:
  optimizer: my_custom_optimizer

my_custom_optimizer:
  learning_rate: 0.001
```

The mapping direction is `role_name: registered_name`. In this example,
`my_custom_optimizer` must be registered, while `optimizer` may be either a
registered name or a virtual role. Configuration belongs under the registered
implementation name; defining `optimizer` as a top-level component is rejected.
Without explicit configuration, the role may still be activated by a dependency
or built-in default when normal constructor rules permit it. Dependencies such
as `@requires_step("optimizer")` resolve to `my_custom_optimizer`. The
registered name is used in component state and shown in the execution graph,
which also includes a `COMPONENT BINDINGS` section.

Bindings are session-scoped and one-to-one. Binding chains, cycles, unknown or
ambiguous targets, category changes, and top-level role configuration are
rejected. Built-in defaults such as `logger` and `checkpointer` can be replaced
through the same mechanism. `ddp.parallel_components` may contain either role
or implementation names; a bound DDP resource must support the same `config`
and `rank` construction interface as the built-in resource.

The former `aliases` key is deprecated but temporarily accepted with the same
role-to-implementation direction. It cannot be combined with
`component_bindings`, and configuration still belongs under the implementation
name.

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

If `outer` wraps `inner`, pre-session and pre-iteration callbacks run outer
then inner, while post-iteration and post-session callbacks run inner then outer.
Wrapping names support session component bindings. Hooks must share a lifecycle
phase. For
two iteration-capable hooks, the wrapper's `call_every` must be a positive
multiple of the wrapped hook's value. This lets the wrapper run less often while
ensuring it never runs on an iteration where the wrapped hook does not run.

The framework builds a registry-wide ordering graph and performs a topological
sort. It rejects missing, incorrectly typed, or unconfigured targets; wrapping
relationships without a shared lifecycle phase; cadence mismatches; and cycles.

The resulting order controls resource setup, hook callbacks, and step execution.
Teardown and post-iteration hook callbacks use reverse order.

Activating a component automatically activates its recursive dependency and
wrapping-target closure when each omitted dependency uses the inherited
component constructor. Activation follows dependency edges outward: activating
a wrapped hook alone does not activate hooks that wrap it. For DDP, secondary
ranks retain the same closure for each root named in `ddp.parallel_components`.