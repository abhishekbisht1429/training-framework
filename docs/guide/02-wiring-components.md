# Wiring Components Together

[← Docs](../README.md) · [Project README](../../README.md)

Writing a component class is only half the job: the session has to know which
components to build, which implementation fills each role, and what order to
run them in. This page covers activating components from YAML, binding a role
to an implementation, declaring an abstract role, and declaring dependencies
between components. It assumes you have read
[Resources, hooks, and steps](01-resources-hooks-steps.md).

## Selecting components

A top-level component mapping both activates a component and supplies its
constructor configuration. Use an empty mapping to activate a root component
that needs no settings:

```yaml
train:
  gradient_accumulation: 4
metrics: {}
```

The framework recursively activates resources, hooks, steps, and wrapped hooks
required by those roots, constructing each one after everything it declared. A
missing dependency is constructed automatically and without arguments only when
its effective constructor is the inherited `Component.__init__`. If its class
or a component base class defines another constructor, add a top-level mapping
for it. Unrelated registered components stay inactive. A dependency cycle has
no valid construction order and is rejected, naming the chain that closed it.
The former top-level `components` list is no longer supported; configs that
contain it receive a migration error.

`session.activate_component(name, config)` does the same thing programmatically
for a component added after the session was built.

`TrainingSession` activates `logger` and `checkpointer` by default.
`AnalysisSession` instead activates `trained_model` and its analysis-specific
`logger`. Both special entries and component dependencies support component
bindings.

## Component bindings

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

## Declaring abstract roles

Some component names, like `dataset` and `model`, are required by built-in
dependencies (`@requires_resource(...)`) with no built-in implementation.
Declare such a name explicitly with `role(name, category, *, description=None,
session_type=None, overwrite=False)` from `training_framework.components`:

```python
from training_framework.components import Resource, role

role(
    "dataset",
    Resource,
    description="the training dataset; a Resource yielding batches",
    session_type="training",
)
```

`category` must be `Resource`, `Hook`, or `Step`.
`training_framework.components.builtin.data` declares `dataset` this way;
`training_framework.components.builtin.distributed` declares `model`.
Declaring a role is optional — `@requires_resource`, `@requires_hook`,
`@requires_step`, and `@wraps` accept any name whether or not it is
declared — but a declared role improves the error raised when nothing
satisfies it, naming the expected category and description and explaining
how to implement it (`@resource('name', ...)`, etc.) or bind an existing
implementation (`component_bindings: {name: implementation}`). Redeclaring a
name without `overwrite=True` raises `ValueError`, matching
`@resource`/`@hook`/`@step`. `role_registry(session_type=None)` returns the
roles declared and visible to a session type, mirroring `component_registry()`.

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

---

**Next:** [Configuration](03-configuration.md) — the full `sessions` YAML
structure and `session_config` fields.

**Going deeper:** [The component model](../concepts/component-model.md)
explains what a component may do while it is being constructed, how to declare
a configuration schema, and how to read a missing-dependency error.
