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
required by those roots, plus any [companions](#companions) they declare, and
constructs each one after everything it declared. A missing dependency is
constructed automatically when that needs no decision: its class declares a
[`config_schema`](../concepts/component-model.md#declaring-a-configuration-schema)
whose every field has a default (it is then built with `{}`, so it can still be
changed by `--extend-session`), or its effective constructor is the inherited
`Component.__init__` (built without arguments). Otherwise, add a top-level
mapping for it. Unrelated registered components stay inactive. Every component
the roots need is known and checked before any constructor runs. A dependency
cycle has no valid construction order and is rejected, naming the chain that
closed it.
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
as `@requires_resource("optimizer")` resolve to `my_custom_optimizer`. The
registered name is used in component state and shown in the execution graph,
which also includes a `COMPONENT BINDINGS` section.

Bindings are session-scoped, and two roles may not bind to the same target. A
target may name a particular *instance* of a component (see
[Configuring a component more than once](#configuring-a-component-more-than-once));
a role name may not, because a role is what a component class declares and a
class cannot know which instance it will be handed. Binding chains, cycles,
unknown or ambiguous targets, category changes, and top-level role
configuration are rejected. Built-in defaults such as `logger` and `checkpointer` can be replaced
through the same mechanism. `ddp.rank_zero_components` may contain either role
or implementation names; a bound DDP resource must support the same `config`
and `rank` construction interface as the built-in resource.

The former `aliases` key is deprecated but temporarily accepted with the same
role-to-implementation direction. It cannot be combined with
`component_bindings`, and configuration still belongs under the implementation
name.

## Configuring a component more than once

A session may hold several instances of one component. Add an instance suffix
to the top-level key, after a `#`:

```yaml
data_manager#train:
  batch_size: 64
data_manager#validation:
  batch_size: 256
```

Each instance is built separately, keeps its own configuration and its own
checkpointed state, and is its own node in the execution graph, where it
appears under its full name (`Resource.data_manager#validation`). A suffix is
one or more letters, digits or underscores; `data_manager#validation` reads
better in a graph or an error than `data_manager#2`, and both are legal.

A key without a suffix is still an instance — the only one of its component —
so nothing about an existing configuration changes.

### Which instance a dependency gets

A component declares the *role* it needs, never the instance:
`@requires_resource("data_manager")` is on the class, shared by every instance
of it, so it cannot name one. The session decides, in this order:

1. **The consumer's own wiring**, if it has any.
2. **An exact name match.** A component named `data_manager` answers to
   `data_manager` however many `data_manager#...` instances exist alongside it,
   so adding an instance never silently rewires anything.
3. **The sole instance** of that component, when there is exactly one.
4. Otherwise it is **an error** naming the candidates. The framework does not
   pick one: wiring a component to something the session never chose gives a
   run that works and is quietly wrong.

This is decided once, when the consumer is built. From then on the instance it
was given is the answer wherever the session needs one -- ordering, the
execution graph, which components a rank keeps, and a restored checkpoint --
so an instance added later (`activate_component("data_manager#b")`) does not
make an already-wired consumer ambiguous or move it to another instance.

### Naming the instance a component should use

Wiring for one consumer is a nested entry in `component_bindings`, keyed by the
consumer's instance name:

```yaml
component_bindings:
  evaluator:
    data_manager: data_manager#validation

data_manager#train: {batch_size: 64}
data_manager#validation: {batch_size: 256}
evaluator: {}
```

`evaluator` is built with `data_manager#validation`; every other consumer of
`data_manager` resolves on its own. A target that names an instance must be
configured: wiring to `data_manager#valdation` is an error naming the
configured instances, never a fresh instance and never a sibling. The
execution graph lists this wiring under `COMPONENT BINDINGS` as
`evaluator: data_manager -> data_manager#validation`, and each `requires:`
annotation names the instance that component is actually given. The wiring lives here rather than inside
`evaluator`'s own configuration because a component's configuration is passed
to its constructor unchanged, and because the session has to know how things
are wired before anything is constructed. The flat
`role: implementation` form is unchanged and can be mixed with it freely.

The wiring holds wherever the consumer takes its prerequisite with
`self.get_dependency(name)` -- in its constructor, in `setup`, or while an
iteration runs -- because the session resolves it for that consumer and hands
it over before the consumer is constructed. See
[taking a prerequisite](../concepts/component-model.md#taking-a-prerequisite).

### Components that must stay unique

Some components cannot have a second instance, and their class says so with
`@singleton`:

```python
@singleton
@resource("my_resource")
class MyResource(Resource):
    ...
```

Configuring one twice is rejected before anything is built, naming every
instance involved. The built-in `ddp` resource is marked this way: it owns the
process group, and the engine, worker and session all expect exactly one.

### Built-ins with more than one instance

A component that writes somewhere named after itself has to keep its instances
apart. The built-ins do this with their instance suffix, leaving the
single-instance path exactly where it was:

- `checkpointer` writes to `<session_dir>/checkpoints`, and a suffixed instance
  to `<session_dir>/checkpoints_<suffix>`. An explicit `checkpoints_dir` still
  wins.
- `tensorboard` appends the suffix to its event directory. Its `port` is
  **not** adjusted: two servers cannot share one, so a second instance needs a
  port of its own in configuration.

`ddp.rank_zero_components` accepts an instance name, so one instance can be
kept off the secondary ranks while its sibling runs on all of them.

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
wrapping-target closure when each omitted dependency can be built without
configuration (see [Selecting components](#selecting-components)). Activation
follows dependency edges outward: activating a wrapped hook alone does not
activate hooks that wrap it. Under DDP, every rank activates the same
components except those declared rank-zero-only — see
[What each rank builds](06-distributed-training.md#what-each-rank-builds).

## Companions

A resource cannot require a step, so a resource whose work is done by a step
would be active with nothing driving it. `@activates(name)` declares such a
*companion*: activating the component activates `name` too.

```python
from training_framework.components import activates, requires_resource, resource, step


@activates("optimizer_step")
@resource("optimizer")
class Optimizer(Resource):
    ...


@requires_resource("optimizer")
@step("optimizer_step")
class OptimizerStep(Step):
    ...
```

A companion only says what must be active. It is not a prerequisite -- it is
not constructed first, not handed over by `get_dependency`, and adds no
ordering edge -- so it may require the component that activates it, as
`optimizer_step` requires `optimizer`, without forming a cycle. Two components
may activate each other.

The companion name is resolved like a dependency, so per-consumer
`component_bindings` redirect it (`optimizer: {optimizer_step: my_step}`).
It is kept on every rank that keeps the component activating it. A session
that holds a component without its companion -- for instance one registered by
hand -- is rejected when it orders its components, and the execution graph
shows the relationship as `activates: Step.optimizer_step`.

## Ordering by dataflow

Steps pass values through `session.iteration_context`. A step or an
iteration hook can declare the keys it reads and writes, and the session then
orders and checks steps by them. This section gives the rules for writing
such a component; for the everyday case -- built-in steps configured in YAML
-- see [Building an iteration](03-building-an-iteration.md).

```python
from training_framework.components import Step, reads, step, writes


@reads("logits", "targets")
@writes("loss")
@step("my_loss")
class MyLoss(Step):
    def run(self, session, logits, targets):
        return loss_fn(logits, targets)
```

The declarations drive the values:

- **Reads arrive as keyword arguments** of `run` (a step) or
  `post_iteration_callback` (a hook), one per declared key, named after it.
  The callback must take every declared read -- by name, or through
  `**kwargs` for a key that is not a Python identifier -- and must not have
  any other parameter without a default. A read may not share its name with
  `self` or the session parameter, which the session passes positionally,
  unless they are positional-only (`def run(self, session, /, **values)`, as
  the generic steps do). All of this is checked when the session is built, so
  a renamed parameter is an error rather than a read of some other key.
- **Writes are returned**, from `run` or `pre_iteration_callback`. One
  declared key: the return value itself, never unpacked, so a step may write
  a tuple or a dict. Several: a tuple in declaration order, or a mapping with
  exactly the declared names. A declared output may not be `None` (that is
  what a forgotten `return` gives), and a component that declares no writes
  must return `None`. Returning is the only way to write a declared key:
  storing it in `session.iteration_context` yourself is refused.

A component whose keys come from its configuration overrides
`context_reads()` / `context_writes()`, returning a mapping of *name* ->
*key*: the name is the parameter (or output) the code uses, the key is where
it lives in the context. The built-in `backward` does this for `loss_key`:

```python
def context_reads(self):
    return {"loss": self._cfg.loss_key}

def run(self, session, loss): ...
```

The [generic steps](../reference/generic-steps.md) do it too, which is how
several `compute#...` instances of one class are ordered. The execution graph
shows such a read as `reads: total_loss (as loss)`.

- **Order.** A step that reads a key runs after the step that writes it. These
  edges join the ones from `@requires_*` and `@wraps`; a contradiction is a
  cycle, reported with its chain and the reason for each link, e.g.
  `Step.a -> Step.b (reads 'z') -> Step.a (reads 'y')`.
- **One writer per key.** Two components writing one key are rejected, naming
  both: their order would be a guess.
- **No update in place.** A step that reads and writes the same key is
  rejected; write a new key instead.
- **Every declared read has a writer**, or the error names the key and lists
  the keys that are written.
- **Iteration hooks.** An iteration hook's writes happen in
  `pre_iteration_callback`, before every step, and its reads in
  `post_iteration_callback`, after every step, so hooks are checked but not
  ordered.
- **Cadence.** The context is cleared after every iteration, so a reader may
  only run on iterations its writer runs on. Everything runs on the first and
  final iteration and on multiples of its cadence -- 1 for a step,
  `call_every` for a hook -- so a reader's cadence must be a multiple of its
  writer's: a step cannot read what a `call_every: 5` hook writes, and a hook
  reading it needs `call_every: 5`, `10`, ... Every iteration hook's
  `call_every` must be a positive integer; that is checked when the session
  is built, for every hook. A step that sets `call_every` is rejected
  (steps run every iteration), so a periodic writer is always a hook whose
  readers are checked against its real cadence. Only steps and iteration hooks take part in an
  iteration: declaring keys on anything else -- a session hook, a resource --
  is rejected.

These checks run when a session is built from configuration -- with the
engine, in the parent process, before any worker starts -- and again when a
session is entered, which covers components registered by hand. While a
session runs, a step or hook that returns something other than what it
declared fails right after that callback, naming what it declared and what
came back. A rank of a distributed run
keeps the writer of every key its components read -- see
[What each rank builds](06-distributed-training.md#what-each-rank-builds).

Declaring is optional: a component that declares nothing is neither ordered
nor checked by keys, and may still use `session.iteration_context` directly
(it must return `None`). The built-in `backward` does
declare a read of `loss`, so a step of yours that produces the loss has to
declare `@writes("loss")`.

The execution graph shows each component's `reads:` / `writes:` and a
`DATAFLOW` section listing every key with its writer and readers.

---

**Next:** [Building an iteration](03-building-an-iteration.md) — a complete
training run from the built-in steps, configured in YAML.

**Going deeper:** [The component model](../concepts/component-model.md)
explains what a component may do while it is being constructed, how to declare
a configuration schema, and how to read a missing-dependency error.
