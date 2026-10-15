# Profiler

[← Docs](../README.md) · [Built-in components](builtin-components.md)

`profiler` records [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html)
traces of chosen iterations: every operator, CUDA kernel and collective, with
its timing. It works in training and analysis sessions, on one rank or several.

```yaml
profiler:
  activities: [cpu, cuda]   # required
  start_iteration: 20
  warmup: 1
  active: 3
```

This records iterations 21-23 on rank 0: iteration 20 warms the profiler up,
and is not written.

## Keys

| Key | Default | Meaning |
|---|---|---|
| `activities` | required | What to record: `cpu`, `cuda`, or both. `cuda` in a session that does not run on a CUDA device is an error |
| `start_iteration` | `1` | The session iteration the first window starts (warms up) on |
| `warmup` | `1` | Iterations the profiler runs, unrecorded, before each window; at least 1, which torch needs for accurate CUDA timings |
| `active` | `3` | Iterations recorded in each window |
| `wait` | `0` | Iterations between one window's end and the next one's start |
| `repeat` | `1` | Number of windows |
| `ranks` | `[0]` | The ranks that record. A rank the session does not have is an error |
| `record_shapes` | `false` | Record the input shapes of each operator |
| `profile_memory` | `false` | Record tensor memory allocation and release |
| `with_stack` | `false` | Record the Python stack of each operator: which of your code ran it. Slower, and makes much larger traces |
| `summary_sort_by` | `self_device_time_total` with `cuda`, else `self_cpu_time_total` | Column the summary is sorted by: `cpu_time_total`, `self_cpu_time_total`, `cpu_memory_usage`, `self_cpu_memory_usage`, `count`, and with `cuda` also `device_time_total`, `self_device_time_total`, `device_memory_usage`, `self_device_memory_usage` |
| `summary_rows` | `25` | Rows in the summary |

Window `k` (from 0) starts on `start_iteration + k * (warmup + active + wait)`.

## What it writes

Each window is written to `<session_dir>/profiler/` (`profiler_<suffix>/` for
a suffixed instance), named after the rank and the iterations it recorded:

- `rank<r>_iter<a>-<b>.pt.trace.json.gz` -- a Chrome trace. Open it in
  [Perfetto](https://ui.perfetto.dev) (or `chrome://tracing`).
- `rank<r>_iter<a>-<b>.summary.txt` -- time and calls per operator, summed
  over the window.

A window the session ends inside of, or fails during, is written with the
iterations it recorded. A session that ends during a window's warm-up writes
nothing for it.

## Reading a trace

torch labels parts of an iteration itself:
`enumerate(DataLoader)#...` for fetching a batch, `autograd::engine::...`
for each backward operation, `Optimizer.step#<Optimizer>.step` and
`Optimizer.zero_grad#...` for the optimizer, and DDP's forward and
communication. The forward pass, losses and your own steps appear as the
operators they call (`aten::addmm`, ...), between the batch fetch and the first
backward operation. To see which of your code ran an operator, set
`with_stack: true`, or wrap code in `torch.profiler.record_function("name")`.

## Resuming and extending

Windows are counted in session iterations, so a resumed session records the
same iterations it was configured to. A window that started before the
session resumed is not restarted; if every window did, nothing is recorded,
and the profiler says so. Nothing about a run in progress is checkpointed.

An extended session may change `start_iteration`, `warmup`, `active`, `wait`
and `repeat`, for instance to profile later iterations:

```bash
training-framework \
  --extend-session ./runs/session_.../checkpoints/<checkpoint-name> \
  --override \
  session_config.max_iterations=6000 \
  profiler.start_iteration=5000
```

## Distributed sessions

The profiler is built on every rank; only the ranks in `ranks` record. Each
writes its own files, so the traces of two ranks can be compared side by side.
A session that still lists `ddp.parallel_components` (deprecated) must list
`profiler` there for ranks other than 0 to have one.
