# Importing components from another run

[← Docs](../README.md) · [Checkpoints and resume](../guide/05-checkpoints-and-resume.md)

`import_components` takes resources of an earlier run -- each with every
instance it was wired to -- into a new session as that session's own
components. The session restores them from the source checkpoint's stored
session state (their constructor arguments, wiring, `state_version` and
state), on the same path a resume uses, and from then on drives them like any
other component: it sets them up and tears them down, saves and restores their
state, moves them to the device and checks their wiring.

A resume, each DDP worker, and anything that later loads this run restores
the imported components from this run's own checkpoint. **The source
checkpoint is read only when a fresh session is built**, and may be moved or
deleted afterwards.

## Example

```yaml
import_components:
  pooled_patch_transformer:                    # an instance of the source run
    checkpoint: runs/pretrain/checkpoints/last # a checkpoint directory

role_bindings:
  model: fine_tuned_model
  backbone: pooled_patch_transformer#imported  # the imported instance
  head: cifar_head
```

If the pretraining run's `pooled_patch_transformer` was wired to five blocks,
this session now holds `pooled_patch_transformer#imported`,
`conv_patch_embedding#imported`, `attention_pooling#imported` and the rest,
each with its pretrained weights, each checkpointing its own.

## Keys

Each key of `import_components` is an **instance name of the source run**,
exactly as that run named it (`pooled_patch_transformer`, `imagenet#train`).
A role of the source (`model`) is refused, naming the instance that role was
bound to. A name the source does not hold is an error listing what it holds
-- except, while the deprecated labelled form is still accepted, for an entry
with only `checkpoint` and `overwritten_dependencies`: that one is read as a
label and imports the source's `model`, with a `FutureWarning` (shown by
default). A mistyped key does the same, so heed that warning
([deprecated form](#deprecated-labelled-imports)).

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | required | The source run's checkpoint directory |
| `instance_name` | `imported` | The suffix every imported instance gets (below) |
| `overwritten_dependencies` | `{}` | A dependency the source run gave its components -> this session's component (below) |

A key's value may also be a **list** of such mappings, to import the same
source name from several checkpoints -- a teacher and a student, say:

```yaml
import_components:
  pooled_patch_transformer:
    - {checkpoint: runs/teacher/checkpoints/last, instance_name: teacher}
    - {checkpoint: runs/student/checkpoints/last, instance_name: student}
```

Imports are restored in the order they are listed, a list's entries in list
order. Errors and each imported component's saved `imported_by` name an
import `import_components.<key>`, or `import_components.<key>[<index>]` in a
list. `import_components` cannot be changed by a session extension, and
neither can an imported component.

## Names

Every instance an import brings in is renamed with a suffix: its
`instance_name`, or `imported` when it has none. `impl` becomes
`impl#imported` (or `impl#s` with `instance_name: s`), and `impl#t` becomes
`impl#imported_t` (`impl#s_t`), so the source's own suffix is kept.

`imported` is reserved: a name this session writes -- a configured key, a
default component's binding, an `instance_name`, a component activated or
registered by hand -- may not use it, alone or at the start of a suffix
(`imported_train`). Names restored from a checkpoint may.

This breaks compatibility with earlier versions for such names: a
configuration that named an instance `x#imported` or `x#imported_*` is
refused, and a run resumed from an earlier version's checkpoint holding one
keeps it, but it is no longer offered as the only instance of its
implementation. Rename it with another suffix.

A name this session already configures, or that another import also brings
in, is an error naming both. Two ways out:

- overwrite it (`overwritten_dependencies`, below), when this session's
  component should serve instead;
- give the import an `instance_name`. Two imports of one implementation -- a
  teacher and a student -- need one each.

## Reaching the imported components

**Without `instance_name`**, an imported instance fills a dependency only
when something names it: a binding (`role_bindings`, or a component's own
`dependencies_role_bindings`) or an `overwritten_dependencies` target. It is
never handed over as the only instance of its implementation: a component
asking for `data_manager` while the import brought `data_manager#imported`
gets a `data_manager` of its own, built as usual, or an error if one cannot
be built without configuration.

**With `instance_name`**, imported instances are ordinary instances of this
session: a dependency resolves to one exactly as it would to an instance this
session configures -- by a binding naming it, or as the only instance of its
implementation. Several candidates and no binding are an error naming them.

The execution graph lists every imported instance and the import that
brought it in, under `IMPORTED`.

This session's bindings cannot rewire an imported component itself: it keeps
the wiring its checkpoint recorded. To give it one of this session's
components instead, use `overwritten_dependencies`.

## `overwritten_dependencies`: this session's component instead of the source's

```yaml
import_components:
  pooled_patch_transformer:
    checkpoint: runs/pretrain/checkpoints/last
    overwritten_dependencies: {dataset: dataset}
```

A key names a prerequisite of the source run (an instance name, a role of its
`role_bindings`, a name one of its components asked for, or the only instance
of an implementation -- not an `#imported` one, which the source itself
imported: name it); its value names a resource of this session, resolved
as a dependency would be -- a role, an instance name, or the only instance of
an implementation, including what an earlier-listed import brought in (an
`#imported` instance only when named). The
import stops there: the source's dataset is never built, and whatever was
wired to it is given this session's `dataset`. This is also how a
`@singleton` such as `ddp` is handled -- it is never imported -- and how to
avoid rebuilding a dataset just to count classes.

A key the import never reaches is an error, so a misspelt one cannot
silently import the source's component after all. A target is built before
the import, so it cannot be one of this import's own components, or a later
import's: for a later import's, the error says to list that import first.

## What is refused

When the session is built, before anything is constructed:

- a key the source does not hold (but see the label fallback under
  [Keys](#keys)), or that is only a role of the source;
- an `overwritten_dependencies` key the import never reaches;
- a hook or step: only resources are imported;
- a `@singleton` component, and one that declares `@activates` companions
  (those carry out the source run's behaviour and would be activated anew
  here); overwrite what needs it instead;
- a name clash (above);
- a single-file checkpoint written before 0.5.0 (convert it with
  `Checkpointer.save_checkpoint(Checkpointer.load_checkpoint(path), new_path)`).

Then restore's own checks, as for a resume: a class no longer registered, or
registered as another kind; a saved state or constructor arguments the
current version cannot migrate (`migrate_state`, `migrate_init_args`).

And, for the wiring:

- an imported component is given exactly the wiring its checkpoint recorded.
  A prerequisite its class declares now that the source run never wired is
  refused: the class has changed since the checkpoint was saved, and an import
  restores only what was recorded -- import from a checkpoint saved with this
  class;
- this session's bindings cannot wire an imported component (a legacy nested
  `component_bindings` entry, or `role_bindings=` in Python, naming it): to
  give it one of this session's components instead of one the source run gave
  it, use `overwritten_dependencies`.

## Instance names inside a saved state

When `overwritten_dependencies` or `instance_name` changes an instance's name,
every saved state is passed to its class's
`Stateful.rename_instances(state, names)` after it is migrated, with `names`
mapping the source's instance names to this session's. `ModuleResource`
renames the prerequisites its state records (`linked`); the default returns
the state unchanged. A component whose own state names other instances
overrides it. Names inside constructor arguments -- a config value naming
another instance -- are not renamed.

## Random numbers

Building the imported components draws from this session's random state
before their saved weights replace what was drawn, and so does building what
`overwritten_dependencies` points at, which happens first. Every later draw -- the
initialisation of a new head, say -- shifts with it. A run is still
deterministic, but it differs from the same configuration without the
import.

## Deprecated: labelled imports

The earlier form keyed each import by a free label and named what to import
with `resource`. It still works, unchanged, with a `DeprecationWarning`:

```yaml
import_components:
  pretrained:                                  # a label
    checkpoint: runs/pretrain/checkpoints/last
    resource: model                            # default `model`; a role or an instance
    role: backbone                             # bind this session's `backbone` to it
    suffix: pre                                # rename, as `instance_name` does
```

Unlike the keyed form, a labelled import without `suffix` keeps the source's
names.

An entry is labelled when it has `resource`, `role` or `suffix`. An entry
with none of them whose key the source holds neither as an instance nor as a
role is read as a label too (importing the source's `model`), with a
`FutureWarning`; with `instance_name`, or in a list, it is always keyed. So a
label-only entry whose label happens to be an instance or role name of the
source is now read as keyed. A labelled import's `suffix` is not checked
against the reserved `imported`.

A labelled import's instances are held apart, not members of the namespace:
one fills a dependency of this session's only through the import's `role`, or
a binding naming a suffixed instance (`suffix` set). Resolution never hands
one over by exact name or as the only instance, and a binding to an
unsuffixed name it brought in is refused as ambiguous. An
`overwritten_dependencies` target may name an earlier labelled import's
`role`, or its instance by a suffixed name only.

To move to the keyed form: key the entry by the instance `resource` resolved
to, replace `suffix` with `instance_name`, and put `role` in `role_bindings`
(`backbone: <impl>#imported`, or `<impl>#<instance_name>`).
