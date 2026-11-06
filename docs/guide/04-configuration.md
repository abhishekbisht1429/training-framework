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
| `max_iterations` | One of these two | Number of training or analysis iterations |
| `epochs` | One of these two | Length of the run in epochs of the data; the session turns it into `max_iterations` (see below) |
| `components_package` | Yes | Importable package recursively scanned for decorated components |
| `device` | No | Requested device string; defaults to CPU, and unavailable CUDA requests currently fall back to CPU |
| `show_execution_graph` | No | Print the resolved lifecycle graph on entry; defaults to `true` |

### Run length in epochs

Give the run's length as exactly one of `max_iterations` or `epochs`. Giving
both, or neither, is an error. With `epochs`, the session asks its
`data_manager` resource how many iterations that many epochs take. It asks
once, when the session is built, and the answer becomes
`session_config.max_iterations`, which everything else reads (the iteration
loop, `$max_iterations` in schedules, the loggers). The execution graph shows
both values, e.g. `Max iterations: 8 (from 3 epochs)`, and the checkpoint
stores both.

```yaml
sessions:
  - session_config:
      rng_seed: 42
      sessions_dir: ./runs
      epochs: 90
      components_package: my_project.components
```

- **Training.** An epoch is what the training `DataManager`'s sampler counts
  as one: the dataset padded to a multiple of `ddp.world_size`. The total is
  `ceil(epochs * padded_size / batch_size)`, where `batch_size` is the global
  batch. The sampler never stops between epochs, so the last batch may start
  the next epoch, and every sample is seen at least `epochs` times. An empty
  dataset or a `batch_size` that `ddp.world_size` does not divide is refused
  when the session is built.
- **Analysis.** The analysis `DataManager` makes a single in-order pass, so
  only `epochs: 1` is accepted. It means one full pass: `ceil(N /
  batch_size)` batches, or `floor` with `drop_last`.
- **Your own data manager.** If you bind `data_manager` to your own resource,
  it counts epochs only if it subclasses
  `training_framework.components.EpochSource` and implements
  `iterations_for_epochs(epochs)`. Otherwise `epochs` is refused.
- **Resuming** keeps the total that was stored, even under a different world
  size, so schedules do not move mid-run.

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

## Sharing configuration between files

A session entry can take part of its configuration from other files, with
`includes`. Each included file is a *fragment*: a mapping shaped like (part
of) a session entry, such as a model, an optimizer recipe, or data settings.

<!-- docs-test: skip: the included files are not next to this page -->
```yaml
# configs/dino_small.yaml
sessions:
  - includes: [common/vit_s16.yaml, common/dino_optimizer.yaml]
    session_config:
      rng_seed: 0
      sessions_dir: ./runs/dino
      max_iterations: 1000
      components_package: my_project.components
    optimizer:
      optimizer: {kwargs: {lr: 0.0005}}   # overrides the fragment's lr
```

```yaml
# configs/common/dino_optimizer.yaml
optimizer:
  optimizer: {kind: adamw, kwargs: {lr: 0.001, weight_decay: 0.04}}
clip_gradients: {max_norm: 3.0}
```

- Fragments are merged **in list order**, then the session entry itself on
  top. Mappings merge key by key; **every other value, lists included, is
  replaced whole**, so a fragment's `steps` order or freezing `rules` is
  replaced, not appended to.
- Paths are **relative to the file that names them**. A fragment may have
  its own `includes`, merged under the fragment's own settings.
- **Two fragments may not set one value differently**: that is an error
  naming both files and the setting, so reordering or adding a fragment
  can never change a setting without anyone noticing. Set it in one of
  them, or override it in the file that includes them. An empty value
  (`ddp:`) merges with a mapping, as everywhere.
- A missing file, a fragment that is not a mapping, a fragment with
  `sessions`, and includes that form a cycle are errors naming the chain
  of files. `includes` belongs in a session entry, not at the top of the
  file.
- `--override` applies after the includes are merged.

The run records one complete config: its `config.yaml` has every included
setting and no `includes`. Next to it, `config_sources.yaml` lists the
files it was built from, each with its SHA-256 and the file that included
it. For HPC batches the includes are read from the batch's commit and must
stay inside the repository; see
[Running batches on an HPC cluster](08-hpc-batches.md).

A Python program can load a file the same way:
`training_framework.config_files.load_config(path).config`. A session given a
mapping that still lists `includes` refuses it.

### No `${...}` interpolation

OmegaConf interpolation (`${common.lr}`, `${oc.env:USER}`) is refused when
a config is loaded, from a file or an `--override`: values are never
resolved, so it would reach the component as that literal text. Share
settings with includes instead. Write `\${` for a value that must contain
a literal `${`.

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
