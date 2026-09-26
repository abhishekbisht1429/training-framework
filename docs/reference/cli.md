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
| `--config PATH` | Start new sessions from a YAML file | Overrides are addressed with a `sessions[i].` prefix; see [Configuration](../guide/04-configuration.md#omegaconf-overrides) |
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
| `--stop-sync-grace-period` | `0.01` | Seconds to poll a DDP stop collective before sleeping |
| `--stop-sync-poll-interval` | `0.005` | Sleep duration between later DDP stop-collective polls |
| `--process_timeout_on_join` | `30.0` | Graceful-shutdown period before surviving workers are terminated |

The stop-sync grace period must be a finite non-negative value. The poll
interval must be finite and greater than zero. These are process-runtime
settings and are not written into session configuration or checkpoints.

---

**See also:** [Architecture and process model](../concepts/architecture.md)
explains what these timeouts are measuring.
