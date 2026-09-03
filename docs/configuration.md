# Configuration and CLI

[← Documentation index](README.md) · [Project README](../README.md)

## Configuration and command-line interface

### YAML structure

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

### New session

```bash
python -m my_project.train --config my_project/config.yaml
```

### OmegaConf overrides

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --override \
  'sessions[0].session_config.max_iterations=2000' \
  'sessions[0].logger.log_every=25'
```

Overrides are applied to `--config` session definitions.

### Resume a checkpoint

```bash
python -m my_project.train \
  --resume-session ./runs/session_.../checkpoints/<checkpoint-name>
```

### Extend a checkpoint

To restore a checkpoint and replace its maximum iteration count:

```bash
python -m my_project.train \
  --extend-session ./runs/session_.../checkpoints/<checkpoint-name> 5000
```

## Process-monitoring options

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --heartbeat-timeout 60 \
  --process_timeout_on_join 30
```

| Argument | Default | Meaning |
|---|---:|---|
| `--heartbeat-timeout` | `30.0` | Maximum seconds a live worker may go without a heartbeat |
| `--process_timeout_on_join` | `30.0` | Graceful-shutdown period before surviving workers are terminated |

The three operations, `--config`, `--resume-session`, and
`--extend-session`, are mutually exclusive and one is required. New-session
type selection and constructor-specific arguments belong to each
`sessions[]` entry.