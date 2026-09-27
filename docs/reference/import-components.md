# Importing components from another run

[← Docs](../README.md) · [Checkpoints and resume](../guide/05-checkpoints-and-resume.md)

`import_components` takes a resource of an earlier run -- and every instance
it was wired to -- into a new session as that session's own components. The
session restores them from the source checkpoint's stored session state
(their constructor arguments, wiring, `state_version` and state), on the same
path a resume uses, and from then on drives them like any other component: it
sets them up and tears them down, saves and restores their state, moves them
to the device and checks their wiring.

A resume, each DDP worker, and anything that later loads this run restores
the imported components from this run's own checkpoint. **The source
checkpoint is read only when a fresh session is built**, and may be moved or
deleted afterwards.

## Example

```yaml
import_components:
  pretrained:                                  # a name for this import
    checkpoint: runs/pretrain/checkpoints/last # a checkpoint directory
    resource: model                            # a resource of that run
    role: backbone                             # bind this run's `backbone` to it

component_bindings:
  model: fine_tuned_model
  head: cifar_head
```

If the pretraining run's `model` was a `pooled_patch_transformer` wired to
five blocks, this session now holds `pooled_patch_transformer`,
`conv_patch_embedding`, `attention_pooling` and the rest, each with its
pretrained weights, each checkpointing its own.

## Keys

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | required | The source run's checkpoint directory |
| `resource` | `model` | Which resource of that run to import, named as that run named it: an instance, a role of its `component_bindings`, a name one of its components asked for, or the only instance of an implementation |
| `role` | none | A role of this session to bind to the imported resource; saved with the session state, so a resume and every rank resolve it too |
| `bind` | `{}` | Source prerequisite -> this session's component (below) |
| `suffix` | none | Rename the imported instances (below) |

`import_components` cannot be changed by a session extension, and neither
can an imported component. Imports are restored in the order they are listed.

## Names

Imported instances keep the names they had in the source run. A name this
session already configures, or that another import also brings in, is an
error naming both. Two ways out:

- `bind` it (below), when this session's component should serve instead;
- `suffix: s` renames every imported instance: `impl` becomes `impl#s`, and
  `impl#t` becomes `impl#s_t`. This is what two imports sharing an
  implementation need -- a teacher and a student, say -- since both names
  come from checkpoints.

## Reaching the imported components

An imported component fills a dependency of this session's own components
**only when a binding names it**: the import's `role`, or -- for an import
with a `suffix` -- a binding naming the suffixed instance
(`{my_model: {encoder: conv_patch_embedding#pre}}`). Resolution never hands
one over by itself -- not by exact name, and not as the only instance of an
implementation -- so importing never quietly rewires the session's own
components. This also holds for a component activated later with
`activate_component`, and after a resume; the refusal comes before anything
is built.

A binding to an *unsuffixed* imported name -- `net: conv_patch_embedding` -- is
refused as ambiguous: without the import it would build a new
`conv_patch_embedding`, so it cannot be told which is meant. Use the import's
`role`, or give the import a `suffix` and bind the suffixed name.

## `bind`: this session's component instead of the source's

```yaml
import_components:
  pretrained:
    checkpoint: runs/pretrain/checkpoints/last
    role: backbone
    bind: {dataset: dataset}
```

A key names a prerequisite of the source run (resolved there, like
`resource`); its value names a resource of this session, resolved as a
dependency would be (a role, or the only configured instance of an
implementation; several are an error asking to name one). An instance an
earlier-listed import brought in is taken only when named exactly by its
suffixed name -- never as the only instance there is, and an unsuffixed one
is refused as ambiguous. The same holds for a per-consumer binding of an
imported component. A `bind` target may also be the `role` of an import
listed earlier (it names that import's instance, under the same rule); the
role of a later import, or the import's own, is an error. The import stops
there: the source's dataset is never built, and whatever was wired to it is
given this session's `dataset`. This is also how a `@singleton` such as `ddp`
is handled -- it is never imported -- and how to avoid rebuilding a dataset
just to count classes.

A `bind` key the import never reaches is an error, so a misspelt one cannot
silently import the source's component after all. What a `bind` target needs
cannot be one of this import's components, or a later import's: it is built
first. For a later import's, the error says to list that import first.

## What is refused

When the session is built, before anything is constructed:

- a `bind` key the import never reaches;
- a hook or step: only resources are imported;
- a `@singleton` component, and one that declares `@activates` companions
  (those carry out the source run's behaviour and would be activated anew
  here); `bind` what needs it instead;
- a name clash (above);
- a single-file checkpoint written before 0.5.0 (convert it with
  `Checkpointer.save_checkpoint(Checkpointer.load_checkpoint(path), new_path)`).

Then restore's own checks, as for a resume: a class no longer registered, or
registered as another kind; a saved state or constructor arguments the
current version cannot migrate (`migrate_state`, `migrate_init_args`).

And, for the wiring:

- a prerequisite a class declares now that the source run never wired it to
  must be wired by a per-consumer binding --
  `{imp_block: {tokenizer: my_tokenizer}}` -- or the import is refused;
- a per-consumer binding for an imported component may only do that: one
  for a name the source run already wired is refused (use `bind`), as is one
  for a name the class does not declare.

## Instance names inside a saved state

When `bind` or `suffix` changes an instance's name, every saved state is
passed to its class's `Stateful.rename_instances(state, names)` after it is
migrated, with `names` mapping the source's instance names to this session's.
`ModuleResource` renames the prerequisites its state records (`linked`); the
default returns the state unchanged. A component whose own state names other
instances overrides it. Names inside constructor arguments -- a config value
naming another instance -- are not renamed.

## Random numbers

Building the imported components draws from this session's random state
before their saved weights replace what was drawn, and so does building what
a `bind` points at, which happens first. Every later draw -- the
initialisation of a new head, say -- shifts with it. A run is still
deterministic, but it differs from the same configuration without the
import.
