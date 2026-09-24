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
