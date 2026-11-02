# CLI Reference

[← Docs](../README.md) · [Project README](../../README.md)

Every flag the entry point accepts. For how to write the configuration these
flags load, see [Configuration](../guide/04-configuration.md); for what resume
and extend actually do, see
[Checkpoints, resume, and extend](../guide/05-checkpoints-and-resume.md).

## Operations

The three operations, `--config`, `--resume-session`, and
`--extend-session`, are mutually exclusive and one is required. New-session
type selection and constructor-specific arguments belong to each
`sessions[]` entry.

| Argument | Operation | Notes |
|---|---|---|
| `--config PATH` | Start new sessions from a YAML file | Its session entries may [include other files](../guide/04-configuration.md#sharing-configuration-between-files); overrides are addressed with a `sessions[i].` prefix and applied after the includes; see [Configuration](../guide/04-configuration.md#omegaconf-overrides) |
| `--resume-session PATH` | Continue a checkpoint unchanged | Accepts only the three launch-topology overrides; see [Resume](../guide/05-checkpoints-and-resume.md#resume) |
| `--extend-session PATH` | Continue a checkpoint with changed, extension-safe settings | Overrides are session-relative, without a `sessions[0]` prefix; see [Extend](../guide/05-checkpoints-and-resume.md#extend) |

The launch-topology keys `ddp.world_size`, `ddp.master_addr` and
`ddp.master_port` are accepted by every operation, because they describe the
machine rather than the session — see
[The launch decides the topology](../guide/06-distributed-training.md#the-launch-decides-the-topology).

## Debugging

Pass `--debug` for debugger-managed worker processes. Workers are spawned
normally, and the parent waits for them using plain process joins without
heartbeat, failure, timeout, or termination monitoring:

```bash
python -m my_project.train --config my_project/config.yaml --debug
```

## Process-monitoring options

```bash
python -m my_project.train \
  --config my_project/config.yaml \
  --heartbeat-timeout 60 \
  --stop-sync-grace-period 0.01 \
  --stop-sync-poll-interval 0.005 \
  --process_timeout_on_join 30
```

| Argument | Default | Meaning |
|---|---:|---|
| `--heartbeat-timeout` | `30.0` | Maximum seconds a live worker may go without a stage change or `session.send_heartbeat()` call |
| `--stop-sync-grace-period` | `0.01` | Seconds to poll a DDP stop vote still pending when it is read before sleeping |
| `--stop-sync-poll-interval` | `0.005` | Sleep duration between later polls of a pending DDP stop vote |
| `--process_timeout_on_join` | `30.0` | Graceful-shutdown period before surviving workers are terminated |

The stop-sync grace period must be a finite non-negative value. The poll
interval must be finite and greater than zero. These are process-runtime
settings and are not written into session configuration or checkpoints.

## Run summaries

`summarize` is a separate program, not a flag of your training entry point:
it reads finished (or running) runs and compares them. See
[Comparing runs](../guide/12-comparing-runs.md) for what the report holds.

```bash
python -m training_framework.reporting summarize ROOT [ROOT ...] [options]
```

| Argument | Default | Meaning |
|---|---|---|
| `ROOT` | (required) | A directory to look for runs (`session_*` directories) in: a `sessions_dir`, a directory above several, one session directory, an HPC batch, or `hpc-results/<batch-id>` |
| `--metric TAG[:min\|:max]` | every tag, last only | A scalar tag to report; with a direction its best value too (repeatable) |
| `--tags GLOB` | | Also report every tag matching `GLOB`, last value only (repeatable) |
| `--last-n N` | | Also report the mean of each tag's last `N` points |
| `--sort TAG` | run id | Order groups and runs by the best value of this `--metric` (it needs a direction) |
| `--include GLOB` / `--exclude GLOB` | | Choose runs by id (repeatable) |
| `--seed-key KEY` | `session_config.rng_seed` | A seed: a config path, or `component:path` inside that component's entry wherever it is listed; replaces the default (repeatable) |
| `--no-seed-groups` | | Report runs only, without grouping over seeds |
| `--ignore-key PATH` | | A config path not to compare, added to the defaults `session_config.sessions_dir`, `ddp.master_addr`, `ddp.master_port`, `resources.ddp.master_addr`, `resources.ddp.master_port` (repeatable) |
| `--no-default-ignores` | | Compare the default ignored keys too |
| `--format markdown\|csv\|json` | `markdown` | Output format |
| `--table groups\|runs` | `groups` (`runs` with `--no-seed-groups`) | With `--format csv`: which table |
| `--output PATH`, `-o PATH` | standard output | Where to write the report; never inside a run directory |
| `--strict` | | Exit with status 1 when the report lists problems |

Exit status: `0` the report was written; `1` it was written, has problems,
and `--strict` was given; `2` no report could be made (no runs, a tag no run
writes, an unknown seed key, conflicting options), with the reason on
standard error.

---

**See also:** [Architecture and process model](../concepts/architecture.md)
explains what these timeouts are measuring.
