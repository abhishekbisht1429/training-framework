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

`dataset` above is an example of a declared role: the built-in `DataManager`
requires a `dataset` Resource but the framework ships no implementation for
it. See [Component bindings](#component-bindings) for how such roles are
declared and satisfied.

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

### Declaring abstract roles

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

### Linking components to each other

`setup(session)` is the first point where a component can reach another one,
and it does not run when a session is restored from a checkpoint. A component
that needs to *hold* another component therefore wires itself up in the link
phase instead.

`Component.link(components)` runs in prerequisite-first order on every
construction path -- fresh configuration, checkpoint restore and worker
fix-up -- so a reference captured at save time exists again at load time. The
default is a no-op, so components that do not need it are unaffected.

```python
@requires_resource("text_encoder")
@resource("model")
class CaptionedImageModel(ModuleResource):
    linked_modules = ("text_encoder",)

    def build(self):
        dim = self.text_encoder.embed_dim   # already attached
        self.head = nn.Linear(2 * dim, self._config["num_classes"])
```

The link phase is deliberately weaker than `setup`:

- There is no session, so no device, no iteration context, no
  `@requires_context` access and no initialised process group. Work that needs
  those stays in `setup`.
- Lookup is restricted to declared prerequisites. Asking for a resource the
  class did not declare with `@requires_resource` raises `ComponentLinkError`,
  which is what makes the prerequisite-first order a guarantee.
- It may run more than once -- registering or replacing a component marks
  links stale, and the session re-links when it is entered -- so
  implementations must be idempotent. Links are frozen once setup has run;
  `session.relink_components()` raises after that.

`ModuleResource` (see [built-in components](built-in-components.md#moduleresource))
implements all of this for `nn.Module` resources.

### Missing-dependency errors

When a dependency, wrapping target, configured root, binding target, or
`session.get_resource(name)` lookup cannot be satisfied, the error keeps its
short headline and adds an indented explanation: which component required it,
any `component_bindings` redirection, a `Reason:` and a `Fix:`. The reason
distinguishes:

- **Not registered anywhere** — not in the shared registry, the session type's
  registry, or any other session type's registry. The fix suggests the
  decorator, reminds you to import the defining module (for example through
  `session_config.components_package`), and lists close name matches.
- **Registered only for another session type** — e.g. the training-only `ddp`
  requested from an analysis session. The fix offers using that session type,
  registering a scoped implementation, or registering it as shared.
- **Registered with a different category** — e.g. a Hook named where a
  Resource is required, including a session-scoped component that shadows a
  shared one of the expected category.
- **Registered but not active in this session** — the component exists but is
  not configured; add a top-level mapping for it.
- **Declared role without an implementation** — the role message is kept and
  the reason names the scope the role was declared in, or notes that it is
  declared only for another session type.

```text
unmet prerequisite! Resource 'ddp' resolves to 'ddp', which is not registered as a Resource.
  Required by: Step 'embed_batches' (EmbedBatches)
  Reason: 'ddp' is not in the shared registry or the 'analysis' registry; it is registered only for session type(s) 'training' (as Resource), so it is unavailable to 'analysis' sessions.
  Fix: Use it from a 'training' session, register a Resource for this session type with @resource('ddp', session_type='analysis'), or register it as shared with @resource('ddp').
```

`session.get_resource()` raises `ComponentNotFoundError`, a `KeyError`
subclass, so existing `except KeyError` handlers keep working.

## Session-extension configuration

Component configuration is immutable when extending a checkpoint unless the
component implements the exported `ExtendableComponent` interface. Its
`apply_extension_config(config, changed_paths)` method receives the merged
component mapping and the component-relative leaf paths that changed. The
method must reject unsafe paths and update any persisted state that duplicates
configuration. The framework updates the component's captured constructor
configuration after the method succeeds so later checkpoints reconstruct it
with the effective values.

The built-in optimizer uses this contract to retain optimizer tensors and step
counters while changing explicitly overridden parameter-group values, and to
allow the `lr_scheduler` configuration to change entirely (a different
scheduler class, stages, milestones, or `metric_key`). A changed `lr_scheduler`
restarts its schedule from the extension point (`last_epoch` and any per-stage
progress reset) since old scheduler state cannot be assumed compatible with a
different schedule shape; optimizer tensors and step counters are unaffected.
Changing `optimizer.optimizer.kwargs.lr` while `lr_scheduler` is left unchanged
instead scales the currently active stage's stored base learning rate(s) by
the same ratio as the override, preserving its schedule progress and whatever
multiplier it is currently applying (mid-warmup, a decay factor, ...). A
multi-stage schedule's not-yet-reached stage is left untouched by this and
runs its own originally configured base once it activates, since it has no
"current lr" of its own to preserve. Logger and checkpointer use the contract
for safe cadence changes. Model, DDP, data-manager, and all custom components
that do not opt in remain immutable.
