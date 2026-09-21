# The Component Model

[← Docs](../README.md) · [Project README](../../README.md)

What the framework guarantees about a component while it is being built: that
its prerequisites already exist, that it may keep hold of them, and what it may
not do. This page also covers declaring a configuration schema, reading a
missing-dependency error, and opting a component into `--extend-session`.

It assumes you have read [Resources, hooks, and steps](../guide/01-resources-hooks-steps.md)
and [Wiring components together](../guide/02-wiring-components.md).

## Two names a component has

A component answers to two different names, and telling them apart is what
lets a session hold more than one instance of it:

| | |
|---|---|
| `component.implementation_name` | the name the **class** was registered under, with `@resource("logger")`. Every instance of it shares this. |
| `component.name` | the name of this **instance** within the session — `logger`, or `logger#validation`. The session sets it when it constructs the component. |

`component.id` is the instance name with its category, `Hook.logger#validation`,
and is what the execution graph prints and what the dependency graph uses as a
node key. `component.instance_suffix` is the part after the `#`, or `None` for
a component that is the only instance of itself.

Registration writes `name` and `id` onto the class; the session then writes
them onto the instance, shadowing the class's. A component built by hand,
outside a session, therefore still reports its class's name and has no suffix.

A component **never names an instance itself**. `@requires_resource("model")`
is evaluated at import time and lives on the class, so it names a role; if it
named an instance, every instance of that class would be handed the same one
and configuring the class twice would be pointless. Which instance fills the
role is decided by the session — see
[Configuring a component more than once](../guide/02-wiring-components.md#configuring-a-component-more-than-once).

`linked_components` records the *instance* a component was handed, not merely
the class, so a consumer rewired between two instances of one component is
caught when its state is restored rather than silently loading the other
instance's weights.

## Taking a prerequisite

A component declares what it needs with `@requires_resource(name)` and takes it
with `self.get_dependency(name)`. That is the only way, and it works at every
point in the component's life -- in `__init__`, in `setup`, in a hook callback,
in a running step.

The session resolves each declared name **for the component that declared
it**, honouring that component's own per-component wiring in
`component_bindings`, and hands the results to the component *before* its
`__init__` runs. So construction, `setup` and every later call see the same
instance, and a component wired to `dataset#b` is given `dataset#b` even when a
session-wide binding points the role somewhere else.

```python
@requires_resource("dataset")
@resource("data_manager")
class DataManager(Resource):
    def setup(self, session):
        dataset = self.get_dependency("dataset")
        ...
```

Components are constructed prerequisite-first on every path -- fresh
configuration, checkpoint restore and worker start-up -- so by the time a
constructor runs, everything it declared already exists. A component that
needs to *hold* another component takes it there, so that a reference captured
at save time exists again at load time:

```python
@requires_resource("text_encoder")
@resource("model")
class CaptionedImageModel(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.text_encoder = self.get_dependency("text_encoder")
        dim = self.text_encoder.embed_dim
        self.head = nn.Linear(2 * dim, self._config["num_classes"])
```

What `get_dependency` hands out is recorded, so the framework knows the two
components are wired together wherever the reference ends up -- an attribute, a
container module, or nowhere at all.

Lookup is restricted to declared prerequisites. Asking for a resource the
class did not declare raises `ComponentDependencyError` naming the
`@requires_resource` to add, and `self.has_dependency(name)` reports whether a
name was declared. The declaration is not a formality: it is the edge in the
graph that orders setup before the consumer and keeps the prerequisite on
every rank that builds the consumer.

Construction is deliberately weaker than `setup`:

- There is no session, so no device, no iteration context, no
  `@requires_context` access and no initialised process group. Work that needs
  those stays in `setup`.
- A dependency graph with a cycle has no valid construction order and is
  rejected, naming the chain that closed it.

Configuration is the usual way for a component to join a session, and
`session.activate_component(name, config)` is the programmatic one; both
resolve bindings and activate the dependency closure. A component you
construct yourself and hand to `session.register_resource()` /
`register_hook()` / `add_step()` is given its prerequisites when it is
registered, so they are available from `setup` onwards. Only a component that
calls `get_dependency` in its own `__init__` must be built by the session;
constructing one by hand raises `ComponentDependencyError` pointing at
`activate_component`. Registering a component under a name that consumers are
already wired to -- replacing it -- hands them the new one.

### There is no session-wide lookup

`Session` has no `get_resource` / `has_resource`. A lookup handed only a name
cannot know which component is asking, so it could only resolve session-wide,
and a component wired to one instance of a component configured twice would
be given whichever instance the session-wide binding picks. Declare the
prerequisite and use `self.get_dependency(name)`.

That is stricter than the old lookup, not a rename of it: `get_dependency`
serves only declared names, so a step or hook that fetched something it never
declared must add `@requires_resource`. That adds an edge to the graph, which can change
setup order and what a secondary DDP rank builds -- see
[what each rank builds](../guide/05-distributed-training.md#what-each-rank-builds).

[`ModuleResource`](module-resource.md) implements all of this for `nn.Module`
resources, including the rules for owning another component's weights versus
holding one privately.

## Declaring a configuration schema

A component may set `config_schema` to a dataclass. The framework then parses
the configuration mapping into `self._cfg` during construction, so the
component stops hand-checking keys:

```python
@dataclass(frozen=True)
class EncoderConfig:
    embed_dim: int
    num_heads: int = 8

    def __post_init__(self):
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")


@resource("text_encoder")
class TextEncoder(ModuleResource):
    config_schema = EncoderConfig

    def __init__(self, config=None):
        super().__init__(config)
        self.attention = nn.MultiheadAttention(
            self._cfg.embed_dim, self._cfg.num_heads,
        )
```

Fields validate the *shape* of the configuration -- which keys are required,
which are accepted, and what they are coerced to -- and `__post_init__`
validates *meaning*. Unknown and missing keys are reported with the component
name and the keys it accepts, OmegaConf containers are normalised, and a field
annotated as a `tuple` accepts a YAML list. Every failure is raised as
`Invalid <component> config: ...`.

`self._config` remains the raw mapping, because that is what a checkpoint
replays to rebuild the component; `self._cfg` is a derived view. Nested specs
whose entries are arbitrary user-supplied values stay hand-parsed by the
component that understands them.

## Missing-dependency errors

When a dependency, wrapping target, configured root, binding target, or
resource lookup cannot be satisfied, the error keeps its
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

A failed resource lookup raises `ComponentNotFoundError`, a `KeyError`
subclass, so existing `except KeyError` handlers keep working.

## Opting into extension

Component configuration is immutable when extending a checkpoint unless the
component implements the exported `ExtendableComponent` interface. Its
`apply_extension_config(config, changed_paths)` method receives the merged
component mapping and the component-relative leaf paths that changed. The
method must reject unsafe paths and update any persisted state that duplicates
configuration. The framework updates the component's captured constructor
configuration after the method succeeds so later checkpoints reconstruct it
with the effective values.

The built-in logger and checkpointer use the contract for safe cadence changes.
The built-in optimizer uses it to retain optimizer tensors and step counters
while changing explicitly overridden parameter-group values — see the
[`optimizer` reference](../reference/builtin-components.md#optimizer) for
exactly what it allows. Model, DDP, data-manager, and all custom components
that do not opt in remain immutable.

For the operator's view of what an extend may change, see
[Checkpoints, resume, and extend](../guide/04-checkpoints-and-resume.md#extend).

---

**See also:** [`ModuleResource`](module-resource.md) implements all of this for
`nn.Module` resources, including the rules for owning another component's
weights versus holding one privately.
