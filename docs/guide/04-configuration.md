# Configuration

[← Docs](../README.md) · [Project README](../../README.md)

Every run is described by a YAML file containing a `sessions` list. This page
covers that structure, the `session_config` fields each session must supply,
and how to override values from the command line when starting a new session.

Resuming or extending an existing checkpoint uses a different set of rules —
see [Checkpoints, resume, and extend](05-checkpoints-and-resume.md). For the
complete flag list, see the [CLI reference](../reference/cli.md).

## YAML structure

The configuration root must contain a `sessions` list:

```yaml
sessions:
  - session_config:
      rng_seed: 42
      sessions_dir: ./runs
      max_iterations: 1000
      components_package: my_project.components
      device: cuda:0

    model:
      hidden_size: 512

    train:
      learning_rate: 0.0003

    logger:
      log_every: 10
```

`session_config` fields:

| Field | Required | Meaning |
|---|:---:|---|
| `rng_seed` | Yes | Seed used for Python, NumPy, PyTorch, and CUDA RNG initialization |
| `sessions_dir` | Yes | Parent directory for timestamped session directories |
| `max_iterations` | Yes | Number of training or analysis iterations |
| `components_package` | Yes | Importable package recursively scanned for decorated components |
| `device` | No | Requested device string; defaults to CPU, and unavailable CUDA requests currently fall back to CPU |
| `show_execution_graph` | No | Print the resolved lifecycle graph on entry; defaults to `true` |

A session directory is created as:

```text
<sessions_dir>/session_YYYYMMDD_HHMMSS/
```

The resolved session configuration is written to `config.yaml` in that directory.

Besides `session_config` and component mappings, a session entry may hold
`role_bindings` ([wiring](02-wiring-components.md)) and
`import_components`, which takes components of an earlier run into the session
([importing components](../reference/import-components.md)).

These top-level names are reserved and cannot name a component:
`session_config`, `session_type`, `session_kwargs`, `role_bindings`,
`import_components`, `resources`, `hooks`, `steps`, and the deprecated
`component_bindings`, `aliases` and `components`.
`role_bindings`, `resources`, `hooks` and `steps` are newly reserved: a
component registered under one of them is refused when it is registered, with
an error naming the reserved word; rename it. A reserved name listed inside a
group is refused too.

### Grouping components by kind

Components may be listed under `resources`, `hooks` and `steps` instead of
directly in the session entry:

```yaml
sessions:
  - session_config: {...}

    role_bindings:
      model: classifier
      dataset: my_dataset
      data_manager: data_manager#train   # what every other consumer gets

    resources:
      classifier:                 # an empty value: configured, no settings
      my_dataset: {root: ./data}
      ddp: {world_size: 1, backend: gloo, master_addr: localhost, master_port: 12355}
      data_manager#train: {batch_size: 32, num_workers: 2, pin_memory: true}
      data_manager#validation: {batch_size: 256, num_workers: 2, pin_memory: true}

    steps:
      load_batch: {fields: [inputs, targets]}
      forward: {args: [inputs], outputs: logits}
      compute#loss: {function: cross_entropy, args: [logits, targets], outputs: loss}

    hooks:
      logger: {log_every: 10}
      evaluator:                  # wired to the validation data
        dependencies_role_bindings: {data_manager: data_manager#validation}
        every: 100
```

`classifier`, `my_dataset` and `evaluator` stand for components of your own
project.

- A group is checked, not guessed: a step listed under `resources` is an error
  that names the group it belongs in -- when the session is built and when
  `--extend-session` lists a component in a group.
- A component is listed once. Two groups, or a group and the top level, is an
  error naming both places.
- The two layouts may be mixed, so a configuration can be moved over a part at
  a time.
- Listing order is not execution order; steps are ordered by the values they
  read and write.
- The configuration is stored as written: `config.yaml` in the session
  directory and every checkpoint keep the grouped layout.
- An override addresses a component where it is listed:
  `sessions[0].steps.forward.args=[x]` on the command line,
  `steps.forward.args=[x]` for `--extend-session`. Overriding
  `forward.args` when `forward` is listed under `steps` is refused with the
  path to use. The launch-topology overrides stay `ddp.<key>` either way.

An empty value (`classifier:`) configures a component with no settings, the
same as `classifier: {}`, at the top level or in a group.

`component_bindings` and `aliases` are deprecated names for `role_bindings`;
see [role bindings](02-wiring-components.md#role-bindings).

Every `sessions[]` entry is registered for the same engine run. Entries may use different registered session types; all resulting worker wrappers start together.

## New session

```bash
python -m my_project.train --config my_project/config.yaml
```

## OmegaConf overrides

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --override \
  'sessions[0].session_config.max_iterations=2000' \
  'sessions[0].logger.log_every=25'
```

With `--config`, overrides are applied to the selected `sessions[]`
definitions.

The `ddp.world_size`, `ddp.master_addr` and `ddp.master_port` keys are an
exception to everything on this page: they describe the machine a run is
launched on rather than the run itself, and are resolved fresh on every launch.
See [The launch decides the topology](06-distributed-training.md#the-launch-decides-the-topology).

---

**Next:** [Checkpoints, resume, and extend](05-checkpoints-and-resume.md) —
saving a session and continuing it later.
