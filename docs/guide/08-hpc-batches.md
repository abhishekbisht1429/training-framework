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
python -m training_framework.hpc batches  --live        # all your batches and what they are doing
python -m training_framework.hpc status   <batch-id>    # one batch, task by task
```

Every key of the profile and the manifest is described in the
[HPC configuration reference](../reference/hpc-configuration.md).

Batches always run an exact, clean commit: commit the manifest, the training
config, and the code first. Uncommitted or untracked files are an error.

For a first run on Big Red 200, follow the staged checklist in
[First runs on Big Red 200](09-hpc-big-red-200.md).

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
Sources whose values are the same local path (for example the train and
val components of one dataset) are one upload, counted once.

## 4. Environments

With `env.existing` in the profile, tasks use your own environment as is.
Otherwise a managed environment is built once per distinct input (pinned
requirements, backend, interpreter identity, hooks) by a `setup` step and
shared by later batches. The requirements file must pin everything, for
example:

```bash
uv pip compile pyproject.toml --extra vision -o requirements/hpc.txt
```

[Setting up a project for HPC batches](10-hpc-projects.md) gives the full
steps, for this repository and for projects that depend on it.

In `dependency` mode it must pin `training-framework` exactly; in `checkout`
mode it must not contain it. Your project's code is never installed: tasks
see it through `code.paths`.

Tasks, preflight, and `prepare` commands run with the environment's `bin`
first on `PATH`, so `python` in a prepare command is the environment's
interpreter. For conda environments that need activation, set
`env.conda_run: true`: every command then runs under `conda run -p <env>`.

## 5. Checking a batch: `validate` and `render`

```bash
python -m training_framework.hpc validate batch.yaml [--suggest-data]
python -m training_framework.hpc render   batch.yaml --output /tmp/preview
```

`validate` runs the same compiler as `submit`, entirely on your machine: it
never connects to the cluster, uploads, or submits. It builds the whole
plan and reports every problem at once, each with the file and key it
concerns, instead of stopping at the first. In order, it checks:

1. **The profile**: every key strictly (unknown keys, types, values), and
   the Slurm settings (no options the HPC layer owns, no control
   characters).
2. **The repository**: the working tree is clean (no uncommitted or
   untracked files), the manifest is committed and identical at `HEAD` and
   `source.ref`, and the ref resolves to one commit. Everything after this
   is read from that commit, not from your working files.
3. **The manifest**: every key strictly; site keys (`account`, `partition`,
   ...) are refused; `code.paths` exist in the commit.
4. **The environment**: the interpreter (`site_python_identity` or
   `environment.python`) against `requires-python`; a valid
   `python_source`/`backend` pair; the requirements file fully pinned, with
   the framework rule for the code mode; the environment key.
5. **The experiments**: every group, experiment, and matrix combination
   becomes a task config; every override and matrix address must exist; two
   identical tasks are an error.
6. **Topology**: `ddp.world_size` against `gpus` (`nccl`) or `cpus`
   (`gloo`); only the allowed engine options.
7. **Data**: every source's target resolves in every task; `upload` paths
   exist and are allowed, and are hashed for their key (see below);
   `cluster` and `prepare` paths go through `path_map`. Absolute local paths
   that are not data sources give a note; `--suggest-data` prints entries
   for them.
8. **Job scripts**: every Slurm script is rendered (resources, partitions,
   array sizes against `max_array_size`, dependencies, the early-stop
   signal), so mistakes `sbatch` would reject surface here.

It prints any `note:` lines, then `ok: N task(s) in G group(s); commit ...;
environment ...` and exits 0, or `error: N problems:` with the list and
exits 2.

Hashing an `upload` source reads every file the first time; later runs only
compare each file's size and timestamps with a cache
(`~/.cache/training-framework/hashes.sqlite3`) and read only what changed.
For large datasets the progress line says which is happening:
`checking ...: 120,000/1,281,167 files, 14.2 GiB (0 hashed, 120,000
unchanged since the last run)`. An interrupted run keeps what it hashed.
Give a huge, fixed dataset a `version` to skip hashing entirely.

What `validate` cannot know, because it stays offline, `submit` checks
next: the site interpreter's identity over SSH, which environments and data
are already on the cluster, that `cluster` paths exist (preflight), and
whether Slurm accepts the account, partition, and limits.

`render` runs the same checks and writes the batch directory it would
stage (job scripts, task configs, `plan.json`, `tasks.tsv`) to `--output`,
which must be empty or missing, so you can read exactly what would be sent.

## 6. What `submit` does

1. Checks that `ssh` and `rsync` are installed, then compiles the batch
   (the same checks as `validate`).
2. Checks the site interpreter's identity, then which environment, uploads,
   and prepared data are already ready.
3. Claims a new batch directory, stages the exact commit (a Git bundle), and
   writes the plan files.
4. Shows a summary (tasks, environment, uploads and their size) and asks for
   confirmation (`--yes` to skip; required when not interactive).
5. Uploads pending data, runs login-node steps, preflights a ready
   environment, then submits the jobs: `setup` -> `prepare` -> one array per
   group. A failed job cancels its dependants.

Each step prints a timestamped progress line (staging, login-node steps,
every submitted job). Uploads show rsync's live progress on a terminal and,
elsewhere (an IDE console, a log), a line every 30 seconds; the last line
says how many files were sent and how many were already there.

**Interrupted uploads resume.** An upload goes to a fixed location and is
marked ready only when it completes, so nothing ever runs on partial data.
Run `submit` again: rsync skips the files already there and continues the
partial one. While an upload is incomplete, `<location>.tf-hpc-pending`
records which content it is; if the local data has changed since, `submit`
refuses to resume (rsync never deletes at the destination, so files you
removed would stay) and says what to delete.

**Ctrl-C** during `submit` is recorded in the receipt. Jobs already
submitted are cancelled, so no partial batch runs. If an `sbatch` was in
flight, the receipt says `needs reconciliation` with the `squeue` query to
check it; otherwise `failed`.

The receipt (`~/.local/state/training-framework/submissions/<batch>.json`,
mirrored as `receipt.json` in the batch directory) lists every job submitted.
If a submission's outcome is uncertain (a dropped connection), nothing is
retried; the receipt says `needs reconciliation` and prints the `squeue
--name=...` query that finds the job.

## 7. Keeping track of your batches

Every `submit` leaves a **receipt**: a small JSON file describing the batch
(when and from where it was submitted, profile, cluster, commit, task count)
and every job it submitted, with its Slurm job ID. Receipts live in
`~/.local/state/training-framework/submissions/<batch-id>.json` (or
`$XDG_STATE_HOME/...`), and a copy is kept in the batch directory on the
cluster as `receipt.json`. Two commands read them; neither changes anything.

| Command | Shows | Connects to the cluster |
|---|---|---|
| `batches` | every batch, one line each | no |
| `batches --live` | ... plus what each batch is doing now | yes, once per cluster |
| `batches --tasks` | ... plus every task of every batch | yes, once per cluster |
| `status <batch-id>` | one batch in detail | yes |

### All your batches: `batches`

```bash
python -m training_framework.hpc batches
```

```
batch                                   submitted         profile    tasks  submit
ssl-dino-20261015T091200Z-3f2a1c9e4b7d  2026-10-15 09:12  bigred200  4      submitted
vit-probes-20261014T220100Z-9a8b7c6d5e  2026-10-14 22:01  bigred200  24     submitted
vit-probes-20261014T214000Z-9a8b7c6d5e  2026-10-14 21:40  bigred200  24     staged (declined at confirmation)
imagenet-20261013T014700Z-1c2d3e4f5a6b  2026-10-13 01:47  bigred200  8      failed (rsync not found on this machine's PATH; ...)
```

Newest first, from the receipts on this machine; it works offline and
needs no profile. The `submit` column is how the submission itself ended:

| `submit` | Meaning |
|---|---|
| `submitted` | every job was accepted by Slurm |
| `staged (...)` | the batch was prepared but not submitted (declined, or over `upload_limit`) |
| `failed (...)` | submission stopped with an error; jobs already submitted were cancelled |
| `needs reconciliation (...)` | an `sbatch` had an uncertain outcome; check with the `squeue --name` query in the receipt |
| `submitting` | `submit` was killed before it could record an outcome |

### What they are doing now: `--live`

```bash
python -m training_framework.hpc batches --live
```

```
batch                                   submitted         profile    tasks  submit     now
ssl-dino-20261015T091200Z-3f2a1c9e4b7d  2026-10-15 09:12  bigred200  4      submitted  running: 2 running, 2 pending
vit-probes-20261014T220100Z-9a8b7c6d5e  2026-10-14 22:01  bigred200  24     submitted  finished: 20 done, 3 interrupted, 1 failed
vit-probes-20261014T214000Z-9a8b7c6d5e  2026-10-14 21:40  bigred200  24     staged     not submitted
```

`--live` makes one SSH connection per cluster, whatever the number of
batches: one `squeue` for all your jobs, and one command that reads every
listed batch's task records. The `now` column summarizes the tasks:

- `running: ...` / `queued: ...`: some tasks are still in Slurm's queue
  (`queued` when none has started yet);
- `finished: ...`: none is left in the queue; the counts say how they ended;
- `(setup failed)`, `(prepare-... failed)`: a step the tasks depended on
  failed, so they were cancelled;
- `not submitted`, `submit failed`: nothing ran.

A batch on a cluster that cannot be reached shows `? (cannot read <host>:
...)` instead; the others are unaffected.

### Every task: `--tasks`

```bash
python -m training_framework.hpc batches --tasks --since 1d
```

```
batch                                  submitted         profile    tasks  submit     now
listing-20261015T012123Z-97dfad756615  2026-10-15 01:21  bigred200  3      submitted  running: 2 running, 1 done
    task           values                     state    detail
    cpu/seeds/000  session_config.rng_seed=1  done     /N/scratch/.../tasks/cpu/seeds/000/sessions/session_...
    cpu/seeds/001  session_config.rng_seed=2  running  job 8565732_1
    cpu/seeds/002  session_config.rng_seed=3  running  job 8565732_2
```

Each batch is followed by its steps (`setup`, `prepare-...`) and tasks:
the task's name (`<group>/<experiment>/<NNN>`), its matrix values, its
state, and a detail. `--tasks` implies `--live`. Task states:

| State | Meaning | Detail |
|---|---|---|
| `pending`, `running` (and other Slurm states) | in Slurm's queue | the job or array element, e.g. `job 4711_3` |
| `done` | exited 0 | its run directory |
| `failed (exit N)` | exited non-zero | the last line of the error |
| `interrupted` | stopped by the early-stop signal before the time limit, or by `scancel --signal=USR1` | |
| `ended without a record` | the job ended without writing its record: killed for memory or time, or the node failed | the log to read |
| `cancelled` | cancelled before it ran (for example, its setup failed) | |
| `built`, `already ready`, `prepared`, `ran on login` | (steps) the environment or data step's outcome | its location |

### Choosing what to list

| Option | Lists |
|---|---|
| `--since 30m` / `12h` / `7d` / `2w` | batches submitted within that time |
| `--name TEXT` | batches whose ID contains `TEXT` (IDs start with the manifest's `name`) |
| `--active` | only batches with queued or running tasks (implies `--live`) |
| `--limit N` | the newest `N` |
| `--json` | the same information as JSON (with `--tasks`, every task), for scripts and notebooks |
| `--receipts DIR` | receipts from another directory |

For example, `batches --active` answers "what is running right now", and
`batches --tasks --name vit-probes --limit 1` shows the latest sweep task by
task.

### Batches submitted from other machines: `--remote`

Receipts are per machine: a batch submitted from your laptop is not in the
list on a workstation. `--remote` also reads the receipts kept in the batch
directories on the cluster (under the profile's `roots.batches`) and adds
the batches that are missing here, with a `from` column naming the machine
that submitted them:

```bash
python -m training_framework.hpc batches --remote --live
```

`--remote` needs a profile (`--profile`, or the default in your profile
file). So do `--live` and `--tasks` for receipts written before receipts
recorded their cluster; newer receipts carry their SSH host.

### One batch in detail: `status`

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

The same rows as `batches --tasks`, for one batch, with its directory on
the cluster. (Values are shortened above; both print full addresses such
as `optimizer.optimizer.kwargs.lr`.)

### Logs

A task's log is `<batch-dir>/logs/<group>/<jobid>_<index>.out` on the
cluster: `status` gives the batch directory and, for a running task, the
`<jobid>_<index>`. Follow it with `ssh <host> 'tail -f <log>'`.

Every log line written by the HPC layer starts with `[hpc ...]` (`[hpc setup
...]`, `[hpc prepare ...]`) and a UTC time. A task log opens with the task's
name, matrix values, commit, host, and GPUs, and closes with `finished: exit
N (ok|failed|interrupted)` and the record's path; the wrapper adds when the
scheduler's advance signal arrived and the task's exit status. Setup and
prepare logs show each step, whom they are waiting for when another job
holds the lease, and, on failure, a single `FAILED:` line.

## 8. Results

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

## 9. Interrupts and time limits

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
