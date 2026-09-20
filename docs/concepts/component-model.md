# The Component Model

[← Docs](../README.md) · [Project README](../../README.md)

What the framework guarantees about a component while it is being built: that
its prerequisites already exist, that it may keep hold of them, and what it may
not do. This page also covers declaring a configuration schema, reading a
missing-dependency error, and opting a component into `--extend-session`.

It assumes you have read [Resources, hooks, and steps](../guide/01-resources-hooks-steps.md)
and [Wiring components together](../guide/02-wiring-components.md).

## Holding another component

`setup(session)` is the first point where a component can reach another one,
and it does not run when a session is restored from a checkpoint. A component
that needs to *hold* another component therefore takes it during construction.

Components are constructed prerequisite-first on every path -- fresh
configuration, checkpoint restore and worker start-up -- so by the time a
constructor runs, everything it declared with `@requires_resource` already
exists. `self.get_dependency(name)` returns it, and a reference captured at
save time exists again at load time.

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

Construction is deliberately weaker than `setup`:

- There is no session, so no device, no iteration context, no
  `@requires_context` access and no initialised process group. Work that needs
  those stays in `setup`.
- Lookup is restricted to declared prerequisites. Asking for a resource the
  class did not declare with `@requires_resource` raises
  `ComponentDependencyError`, which is what makes the prerequisite-first order
  a guarantee.
- A dependency graph with a cycle has no valid construction order and is
  rejected, naming the chain that closed it.

Because wiring happens at construction, a component that declares dependencies
must be built *by the session*. Configuration is the usual way;
`session.activate_component(name, config)` is the programmatic one, and it
resolves bindings and activates the dependency closure the same way. Handing
`session.register_resource()` an instance you constructed yourself stays
supported for components that declare no dependencies. Constructing one that
does raises `ComponentDependencyError` pointing at `activate_component`.

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
