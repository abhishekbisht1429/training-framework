# `ModuleResource`

[← Docs](../README.md) · [Project README](../../README.md)

`ModuleResource` is the bridge between the component model and PyTorch: an
`nn.Module` that is also a `StatefulResource`, for models assembled from other
resources. This page covers how children are attached, who owns which weights
at checkpoint time, and when a component class may be used as a plain
submodule.

It assumes you have read [The component model](component-model.md).

## Attaching children

`ModuleResource` is an `nn.Module` that is also a `StatefulResource`, for
models assembled from other resources. Subclasses create their own parameters
in `__init__`, like any other PyTorch module, and name the resources they
attach in `linked_modules`. Those children are attached by `super().__init__()`
before the subclass body runs, so a constructor can size its own weights from
them. Components are constructed [prerequisite-first on every
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
    linked_modules = ("text_encoder",)

    def __init__(self, config=None):
        super().__init__(config)
        self.head = nn.Linear(512, self._config["num_classes"])

    def forward(self, image, tokens):
        ...
```

## Members

| Member | Purpose |
|---|---|
| `linked_modules` | Resource names attached as submodules under the same attribute |
| `attach_dependencies()` | Override to attach conditionally or under another attribute name |
| `attach_linked_module(attribute, component)` | Attach one component; rejects non-modules and re-attaching a different child |
| `linked_components` | The attribute -> component name map; recorded in the checkpoint and checked on restore |
| `captured_tensors()` | The live tensors this component checkpoints, by state key |
| `config_schema` | Optional dataclass; parsed into `self._cfg` (see [declaring a configuration schema](component-model.md#declaring-a-configuration-schema)) |
| `usable_as_plain_module(cls)` | Whether a component class may be owned privately as an ordinary submodule |
| `plain_module_api` | Members a privately owned component may not override |

## Ownership

An attached child is a real submodule, so `model.parameters()`
covers it and `ddp`, the optimizer and `layer_inspector` see one module tree,
with each tensor appearing once. State is split the other way: a parent
excludes everything reachable from an attached child, so each component
checkpoints only the weights it created. `set_state` loads in place, which is
what keeps a parent's references valid.

A child may be shared by several models and is restored as one shared
instance. Attaching *another component's* weights as a plain submodule
instead of through `attach_linked_module` raises `ComponentDependencyError`
when the session captures state, naming both components and the shared tensor,
because it would otherwise be stored twice.

## What is recorded

`linked_components` is a copy of the attribute ->
component name map built by `attach_linked_module`: the key is the attribute
the child hangs on (normally the role name), the value is the registered name
of the implementation that filled it.

```python
{"patch_embedding": "conv_patch_embedding",
 "positional_embedding": "learned_positional_embedding_2d",
 "sequence_encoder": "torch_transformer_encoder"}
```

The map does three jobs: it says which tensors belong to someone else, it tells
the ownership walk which subtrees to skip, and it is saved in the checkpoint
under `linked`. `set_state` compares the saved map against the current one and
refuses state captured from a differently wired instance -- a component class
whose `linked_modules` or `attach_dependencies` changed since the checkpoint
was written, or state moved between instances by hand. It is not a
configuration guard: a restore rebuilds components from the checkpoint's own
bindings, so rebinding a role in a later config never reaches this check, and a
renamed component is caught earlier, when the checkpoint entry no longer
matches a registered name.

## Owning a component privately

A component class may also be used as an
ordinary submodule -- constructed and owned by another component, never
registered, its weights checkpointed inside its owner. That is allowed as long
as the session drives nothing about it: it must declare no `linked_modules`
and override none of `plain_module_api` (`attach_dependencies`, `get_state`,
`set_state`, `rollback_setup`, `setup`, `teardown`). One that does is rejected
with an explanation, because the session never calls those for a module it
does not know about. `ModuleResource.usable_as_plain_module(cls)` answers the
same question in code.

---

**See also:** the [transformer blocks](../reference/transformer-blocks.md) are
built entirely from `ModuleResource`, and are the fullest worked example of
composition and shared ownership.
