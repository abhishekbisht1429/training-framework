# Running batches on an HPC cluster

`training_framework.hpc` turns one committed training configuration into a
batch of experiments and submits them to a Slurm cluster: one job array per
group of experiments that share resources, each task a single-node run of the
engine (with DDP inside the node if configured). Everything is decided on your
workstation; the cluster only checks and executes.

```bash
pip install 'training-framework[hpc]'     # adds `packaging`, used on the workstation
python -m training_framework.hpc validate experiments/sweep.yaml
python -m training_framework.hpc render   experiments/sweep.yaml --output /tmp/preview
python -m training_framework.hpc submit   experiments/sweep.yaml
```

Batches always run an exact, clean commit: commit the manifest, the training
config, and the code first. Uncommitted or untracked files are an error.

## 1. The profile (site settings, on your machine)

`~/.config/training-framework/hpc.yaml` (or `--profiles PATH`) holds
everything specific to a cluster, so the committed manifest runs unchanged
elsewhere. Pick one with `--profile NAME`, else `default`.

```yaml
default: bigred200
profiles:
  bigred200:
    ssh_host: bigred200                 # an alias from ~/.ssh/config
    roots:
      batches: /N/scratch/me/hpc/batches
      envs: /N/scratch/me/hpc/envs
      data: /N/scratch/me/hpc/data
    scheduler: slurm
    slurm:
      defaults: {account: r00000}
      partitions: {gpu: gpu, cpu: general}
      max_array_size: 1000              # the site's MaxArraySize
    hooks:                              # shell lines, per phase
      common: ["module load python/3.12"]
      task: ["module load cudatoolkit"]
    env:
      python_source: site               # site | uv | conda
      site_python_identity: ...         # printed by probe-python (below)
    setup_location: job                 # job | login (login: compute nodes lack internet)
    prepare_location: job
    path_map: {/home/me/datasets: /N/project/mylab/datasets}
    upload_limit: 20G
```

SSH uses your own OpenSSH configuration, agent, and control connections; no
credentials are stored and MFA is never automated (`BatchMode=yes` fails fast
instead of prompting). Open a control connection first if your site needs MFA.

For a managed environment built from the site's Python, record that
interpreter's identity once and paste the output into the profile:

```bash
python -m training_framework.hpc probe-python
```

If the site later changes its Python, `submit` stops with the new block to
paste; the new identity gives a new, separate environment.

## 2. The manifest (experiments, committed)

```yaml
version: 1
name: vit-fine-tuning
source: {strategy: bundle, ref: HEAD}   # or fetch, with url:
code:
  framework: dependency                 # checkout when this repo *is* training-framework
  paths: ["."]                          # PYTHONPATH entries in the checkout
environment:
  requirements: requirements/hpc.txt    # fully pinned (see section 4)
  python: "3.12"                        # optional pin; required for uv/conda
training:
  config: configs/fine-tuning.yaml
  session: 0
groups:
  probes:
    resources: {walltime: "02:00:00", gpus: 1, cpus: 8, memory: 64G}
    max_parallel: 4
    experiments:
      - name: frozen-backbone
        overrides:
          - {component: fine_tuned_model, path: frozen, value: ["module.*"]}
        matrix:
          - {component: optimizer, path: optimizer.kwargs.lr, values: [0.001, 0.0001]}
          - {path: session_config.rng_seed, values: [41, 42]}
```

- An address is `{component?, path}`: a component's entry wherever it is
  listed (top level or a `resources`/`hooks`/`steps` group), or a path in the
  session. Use a list path when a key contains a dot.
- Each experiment is: base config, then `overrides`, then one combination of
  the `matrix` (the last entry varies fastest). The example gives four tasks,
  `probes/frozen-backbone/000` to `003`.
- `resources` are portable; partitions, QoS, and account come from the
  profile. Scheduler-specific options go under `scheduler_options: {slurm: {...}}`.
- `ddp.world_size` must be set explicitly: with `nccl` it must equal `gpus`;
  with `gloo` it must fit in `cpus`.
- `training.engine_options` accepts `--heartbeat-timeout`,
  `--stop-sync-grace-period`, `--stop-sync-poll-interval`, and
  `--process_timeout_on_join`; config changes belong in overrides.

## 3. Data

Only config values listed in `data.sources` are treated as data:

```yaml
data:
  sources:
    - name: flowers
      target: {component: flowers102, path: root}
      upload: {}                         # the value is a local path: upload it once
    - name: inat
      target: {component: inaturalist, path: root}
      cluster: {}                        # already on the cluster (via path_map)
    - name: places
      target: {component: places, path: root}
      prepare:                           # built or downloaded on the cluster
        command: ["python", "-m", "my_project.prepare_places", "{path}"]
        version: "1"
        inputs: ["my_project/prepare_places.py"]   # optional: re-prepare only when this changes
        resources: {walltime: "04:00:00", cpus: 4, memory: 16G}
```

`validate --suggest-data` prints `upload: {}` entries for absolute local
paths found in your configs. Uploads are keyed by content and stored once
(under `roots.data`, or the `path_map` target); later batches reuse them.
A ready artifact is never modified. Keep `download: false` for uploaded or
prepared datasets; ImageNet must be uploaded or placed on the cluster.

## 4. Environments

With `env.existing` in the profile, tasks use your own environment as is.
Otherwise a managed environment is built once per distinct input (pinned
requirements, backend, interpreter identity, hooks) by a `setup` step and
shared by later batches. The requirements file must pin everything, for
example:

```bash
uv pip compile pyproject.toml --extra vision -o requirements/hpc.txt
```

In `dependency` mode it must pin `training-framework` exactly; in `checkout`
mode it must not contain it. Your project's code is never installed: tasks
see it through `code.paths`.

Tasks, preflight, and `prepare` commands run with the environment's `bin`
first on `PATH`, so `python` in a prepare command is the environment's
interpreter. For conda environments that need activation, set
`env.conda_run: true`: every command then runs under `conda run -p <env>`.

## 5. What `submit` does

1. Compiles the batch (the same checks as `validate`).
2. Checks the site interpreter's identity, then which environment, uploads,
   and prepared data are already ready.
3. Claims a new batch directory, stages the exact commit (a Git bundle), and
   writes the plan files.
4. Shows a summary (tasks, environment, uploads and their size) and asks for
   confirmation (`--yes` to skip; required when not interactive).
5. Uploads pending data, runs login-node steps, preflights a ready
   environment, then submits the jobs: `setup` -> `prepare` -> one array per
   group. A failed job cancels its dependants.

Each step prints a timestamped progress line (staging, each upload with
rsync's live progress on a terminal, login-node steps, every submitted job).

The receipt (`~/.local/state/training-framework/submissions/<batch>.json`,
mirrored as `receipt.json` in the batch directory) lists every job submitted.
If a submission's outcome is uncertain (a dropped connection), nothing is
retried; the receipt says `needs reconciliation` and prints the `squeue
--name=...` query that finds the job.

## 6. Following a batch

```bash
python -m training_framework.hpc status <batch-id>      # or the receipt's path
```

```
batch vit-fine-tuning-20261011T030035Z-2926f55be79e: submitted | /N/scratch/...
task               values                 state            detail
setup                                     built            /N/scratch/me/hpc/envs/ea3c...
probes/frozen/000  lr=0.001 rng_seed=41   done             .../sessions/session_...
probes/frozen/001  lr=0.001 rng_seed=42   running          job 4711_1
probes/frozen/002  lr=0.0001 rng_seed=41  failed (exit 1)  ValueError: ...
```

`status` only reads: the receipt, each group's `tasks.tsv`, the records on
the cluster, and `squeue` for jobs still queued or running. A task whose job
ended without writing a record (killed at the time limit, out of memory)
points at its log file. (Values are shortened above; `status` prints full
addresses such as `optimizer.optimizer.kwargs.lr`.)

Every log line written by the HPC layer starts with `[hpc ...]` (`[hpc setup
...]`, `[hpc prepare ...]`) and a UTC time. A task log opens with the task's
name, matrix values, commit, host, and GPUs, and closes with `finished: exit
N (ok|failed|interrupted)` and the record's path; the wrapper adds when the
scheduler's advance signal arrived and the task's exit status. Setup and
prepare logs show each step, whom they are waiting for when another job
holds the lease, and, on failure, a single `FAILED:` line.

## 7. Results

```
<roots.batches>/<batch-id>/
  groups/<group>/tasks.tsv        # array index -> task, matrix values, config
  tasks/<group>/<experiment>/<NNN>/
    config.yaml  variant.json  record.json
    sessions/session_<timestamp>/ # checkpoints, events, ...
  logs/<group>/<job>_<index>.out
```

`record.json` ties each result to its commit, config, environment, and exit
status. In HPC configs set the `tensorboard` component's `serve: false`, then
view a whole batch at once:

```bash
ssh -L 6006:localhost:6006 bigred200
tensorboard --logdir <batch dir>/tasks --port 6006
```

## 8. Interrupts and time limits

Each array job asks Slurm for an early signal (`process_timeout_on_join` plus
the group's `stop_grace`, default 120s, before the time limit). The task
wrapper turns it into a cooperative stop of the engine; the task records
`interrupted` and exits non-zero. The latest periodic checkpoint is kept.
Slurm's own SIGTERM at the limit stops everything without that grace, so
leave enough `stop_grace` for one iteration.

## Limitations

Single-node tasks only; Slurm only; no cancel, retry, or result download
commands yet (use `scancel`, `rsync`); components that
bind fixed ports (other than `tensorboard` with `serve: false`) fail when
tasks share a node; explicitly configured component output directories are
not redirected per task.
