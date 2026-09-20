# Documentation

[Project README](../README.md)

The docs are in three tracks. Read the **guide** in order if you are new;
dip into **reference** when you need a specific key or signature; read
**concepts** when you need to know why something behaves the way it does.

## Learn it

A task-ordered path. It picks up where the
[quick start](../README.md#quick-start) in the project README ends.

1. [Resources, hooks, and steps](guide/01-resources-hooks-steps.md) — the three
   kinds of component, giving one state that survives a checkpoint, and how the
   framework discovers your classes.
2. [Wiring components together](guide/02-wiring-components.md) — activating
   components from YAML, binding a role to an implementation, declaring
   dependencies and wrapping.
3. [Configuration](guide/03-configuration.md) — the `sessions` structure,
   `session_config` fields, and command-line overrides.
4. [Checkpoints, resume, and extend](guide/04-checkpoints-and-resume.md) — what
   a checkpoint holds, continuing a run, and changing hyperparameters safely.
5. [Distributed training](guide/05-distributed-training.md) — DDP configuration,
   how a launch decides its topology, and coordinated stopping.
6. [Analysis sessions](guide/06-analysis-sessions.md) — driving a trained
   checkpoint through an analysis workflow.

## Look it up

- [Built-in components](reference/builtin-components.md) — every component the
  framework ships with, and its configuration keys.
- [Transformer blocks](reference/transformer-blocks.md) — the swappable
  transformer building blocks and the two composite models.
- [Infinite samplers](reference/samplers.md) — `InfiniteSampler` and
  `DistributedInfiniteSampler`.
- [CLI reference](reference/cli.md) — every command-line flag.
- [API summary](reference/api.md) — public imports and type members.

## Understand it

- [Architecture and process model](concepts/architecture.md) — parent and
  worker responsibilities, spawning, supervision, heartbeats.
- [Session lifecycle](concepts/session-lifecycle.md) — exact phase ordering,
  failure and rollback, the two shared contexts.
- [The component model](concepts/component-model.md) — what a component may do
  while being constructed, configuration schemas, dependency errors.
- [`ModuleResource`](concepts/module-resource.md) — composing `nn.Module`
  resources and who owns which weights.
- [Current behavior and limitations](concepts/limitations.md) — known
  constraints, grouped by area.

## Contributing

- [Development and testing](development.md) — running the suite, project layout.
