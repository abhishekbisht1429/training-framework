# `ModuleResource`

[← Docs](../README.md) · [Project README](../../README.md)

`ModuleResource` is the bridge between the component model and PyTorch: an
`nn.Module` that is also a `StatefulResource`, for models assembled from other
resources. This page covers how children are held, who owns which weights
at checkpoint time, and when a component class may be used as a plain
submodule.

It assumes you have read [The component model](component-model.md).

## Holding children

`ModuleResource` is an `nn.Module` that is also a `StatefulResource`, for
models assembled from other resources. Subclasses create their own parameters
in `__init__`, like any other PyTorch module, and ask for the resources they
declared with `self.get_dependency(name)` once `super().__init__()` has run.
The reference is theirs to place: an attribute, a container module, or nowhere
at all if the constructor only needed to read a width off it. Components are
constructed [prerequisite-first on every
path](component-model.md#holding-another-component), so a restored model is usable
without `setup()` -- which is what `trained_model` relies on.

```python
@resource("text_encoder")
class TextEncoder(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.embedding = nn.Embedding(self._config["vocab_size"], 256)


@requires_resource("text_encoder")
@resource("model")
class CaptionedImageModel(ModuleResource):
    def __init__(self, config=None):
        super().__init__(config)
        self.text_encoder = self.get_dependency("text_encoder")
        self.head = nn.Linear(512, self._config["num_classes"])

    def forward(self, image, tokens):
        ...
```

`@requires_resource` is the whole declaration: it orders construction, and a
component may only ask for what it declared. Holding a child conditionally is
an ordinary `if` in the constructor, and `has_dependency(name)` reports whether
an optional prerequisite is active.

## Members

| Member | Purpose |
|---|---|
| `get_dependency(name)` | The live prerequisite, during construction; records that this component was wired to it |
| `linked_components` | The asked name -> implementation name map; recorded in the checkpoint and checked on restore |
| `captured_tensors()` | The live tensors this component checkpoints, by state key |
| `config_schema` | Optional dataclass; parsed into `self._cfg` (see [declaring a configuration schema](component-model.md#declaring-a-configuration-schema)) |
| `usable_as_plain_module(cls)` | Whether a component class may be owned privately as an ordinary submodule |
| `plain_module_api` | Members a privately owned component may not override |

## Ownership

A child held in the module tree is a real submodule, so `model.parameters()`
covers it and `ddp`, the optimizer and `layer_inspector` see one module tree,
with each tensor appearing once. State is split the other way: a parent
excludes everything reachable from a prerequisite it was handed, so each
component checkpoints only the weights it created. `set_state` loads in place,
which is what keeps a parent's references valid.

Prerequisites are recognised by identity rather than by attribute, so a child
may sit anywhere in the tree -- under its own attribute, inside an
`nn.Sequential`, or in a `ModuleList` -- and is still excluded correctly.

A child may be shared by several models and is restored as one shared
instance. Capturing *another component's* weights without going through
`get_dependency` raises `ComponentDependencyError` when the session captures
state, naming both components and the shared tensor, because it would
otherwise be stored twice.

## What is recorded

`linked_components` is the asked name -> implementation name map built by
`get_dependency`: the key is the name the component asked for (the role name),
the value is the registered name of the implementation that filled it.

```python
{"patch_embedding": "conv_patch_embedding",
 "positional_embedding": "learned_positional_embedding_2d",
 "sequence_encoder": "torch_transformer_encoder"}
```

The map does three jobs: it says which tensors belong to someone else, it tells
the ownership walk which subtrees to skip, and it is saved in the checkpoint
under `linked`, alongside `version: 2`. `set_state` compares the saved map
against the current one and refuses state captured from a differently wired
instance -- a component whose constructor asks for different prerequisites than
it did when the checkpoint was written, or state moved between instances by
hand. A `version: 1` state is keyed by the attribute a child was attached
under, which no longer exists, so it is compared on the implementation names
alone. It is not a
configuration guard: a restore rebuilds components from the checkpoint's own
bindings, so rebinding a role in a later config never reaches this check, and a
renamed component is caught earlier, when the checkpoint entry no longer
matches a registered name.

## Owning a component privately

A component class may also be used as an
ordinary submodule -- constructed and owned by another component, never
registered, its weights checkpointed inside its owner. That is allowed as long
as the session drives nothing about it: it must declare no prerequisites with
`@requires_resource` and override none of `plain_module_api` (`get_state`,
`set_state`, `rollback_setup`, `setup`, `teardown`). One that does is rejected
with an explanation, because the session never calls those for a module it
does not know about. `ModuleResource.usable_as_plain_module(cls)` answers the
same question in code.

---

**See also:** the [transformer blocks](../reference/transformer-blocks.md) are
built entirely from `ModuleResource`, and are the fullest worked example of
composition and shared ownership.
