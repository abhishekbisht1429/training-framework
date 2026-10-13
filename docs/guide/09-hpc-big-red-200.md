# First runs on Big Red 200

A checklist for bringing up the [HPC batch layer](08-hpc-batches.md) on
Indiana University's Big Red 200, one new risk per stage. Site details
(account, partitions, modules, limits) change; the commands in step 1 read
the current values from the cluster instead of assuming them.

Tasks need a Python environment on the cluster. There are two ways to get
one, and the stages below use both:

- **Existing**: an environment you made yourself; the profile points at it
  (`env.existing`). Nothing is built. Quickest for a first run.
- **Managed**: the HPC layer builds an environment from your pinned
  requirements file, once, and every later batch with the same inputs reuses
  it. Reproducible and no manual setup; see [section 5](#5-managed-environments-building-on-the-cluster).

## 0. On your workstation

```bash
pip install "training-framework[hpc] @ git+ssh://git@github.com/abhishekbisht1429/training-framework.git@dev"
# or, from a clone: pip install -e '.[hpc]'
python -m training_framework.hpc --help         # validate, render, submit, status, probe-python
```

## 1. SSH and site facts (once)

Big Red 200 uses Duo two-factor authentication. The HPC layer never prompts
(`BatchMode=yes`), so keep a control connection open in `~/.ssh/config`:

```
Host bigred200
    HostName bigred200.uits.iu.edu
    User <your-iu-username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

```bash
ssh bigred200 true                              # answer Duo once
ssh -o BatchMode=yes bigred200 hostname         # must work without a prompt
```

Read the facts the profile needs on the cluster:

```bash
ssh bigred200
sinfo -s                                         # partition names (general, gpu)
sacctmgr -nP show assoc user=$USER format=account,partition,qos   # your accounts
scontrol show config | grep -i MaxArraySize
module avail python                              # the Python module to load
which rsync; echo /N/scratch/$USER               # rsync, scratch path
```

### Finding your Slurm account

`--account` (`-A`) names the allocation a job is charged to; it is set once
in the profile (`slurm.defaults.account`) and added to every job. At IU,
Slurm accounts come from RT Projects (projects.rt.iu.edu): a PI's project
with a Big Red 200 allocation, which you must be added to. Names usually
look like `r00123`.

If `sacctmgr` above prints nothing, either your user has no account yet or
the site hides accounting data from users. These tell you which:

```bash
sacctmgr show user $USER withassoc format=user,defaultaccount,account,partition,qos
sshare -U -u $USER                                # accounts you belong to
sacct -u $USER -S 2025-01-01 -X --format=jobid,account,partition | tail   # accounts of past jobs
sbatch --test-only -p general -t 00:05:00 --wrap="true"                 # valid without --account?
sbatch --test-only -p general -A <candidate> -t 00:05:00 --wrap="true"  # valid with this account?
```

`--test-only` checks a job without submitting it.

- An account shown by `sshare` or `sacct`: put it in `slurm.defaults.account`.
- The test without `-A` succeeds: you have a default account, and
  `defaults: {}` works; setting the account explicitly is still safer.
- `Invalid account or account/partition combination specified`: you need an
  account. Check RT Projects, ask your PI to add you, or ask UITS Research
  Technologies (rtech@iu.edu) for the account name.

After setting it, `render` the batch and check that the job script has
`#SBATCH --account=...`.

Scratch is purged after a period without access; keep results you need
elsewhere, and prefer a non-purged location (Slate, `/N/slate/$USER`, if you
have it) for managed environments, since a purge can remove files from an
environment that is still marked ready.

## 2. The profile

`~/.config/training-framework/hpc.yaml` (every key is described in the
[HPC configuration reference](../reference/hpc-configuration.md)).
Everything except `env` is the same
for both kinds of environment:

```yaml
default: bigred200
profiles:
  bigred200:
    ssh_host: bigred200
    roots:
      batches: /N/scratch/<user>/hpc/batches
      envs: /N/slate/<user>/hpc/envs             # managed environments (or scratch)
      data: /N/scratch/<user>/hpc/data
    scheduler: slurm
    slurm:
      defaults: {account: <your-account>}
      partitions: {gpu: gpu, cpu: general}
      max_array_size: <MaxArraySize>
    hooks:
      common: ["module load python/<version>"]   # from `module avail python`
    env:
      existing: /N/u/<user>/BigRed200/envs/tf    # stages A-C; section 5 replaces this
    setup_location: login
    prepare_location: login
```

An existing environment needs torch and the framework's dependencies. In
`framework: checkout` mode the framework itself comes from the staged
commit; in `dependency` mode the environment must have it installed.

## 3. A smoke-test batch

A small committed repository: this repository itself in `checkout` mode, or
your project in `dependency` mode. In the training config use a small
`max_iterations`, set `ddp.world_size` explicitly, and set
`tensorboard.serve: false` if the config has a `tensorboard` component.

```yaml
version: 1
name: br200-smoke
code: {framework: checkout, paths: ["src", "."]}   # or dependency + your project
training: {config: configs/smoke.yaml}
groups:
  cpu:
    resources: {walltime: "00:15:00", cpus: 2, memory: 8G}
    experiments:
      - name: seeds
        matrix: [{path: session_config.rng_seed, values: [1, 2]}]
```

Commit everything, then check the generated scripts before submitting:

```bash
python -m training_framework.hpc validate batch.yaml
python -m training_framework.hpc render batch.yaml --output /tmp/br200-preview
less /tmp/br200-preview/groups/cpu/group-cpu.sbatch   # account, partition, module hooks
```

## 4. Stages

Go on only when a stage passes. Stages A-C use the existing environment;
D repeats them with a managed one.

**A. CPU smoke test** (SSH, staging, `sbatch`, arrays, logs, `status`):

```bash
python -m training_framework.hpc submit batch.yaml
python -m training_framework.hpc status <batch-id>        # pending -> running -> done
ssh bigred200 'cat <batch-dir>/logs/cpu/*.out'
ssh bigred200 'cat <batch-dir>/tasks/cpu/seeds/000/record.json'
```

Passes when both tasks are `done`, each log has `task cpu/seeds/00N | ...`
and `finished: exit 0 (ok)`, and each record names a run directory.

**B. GPUs.** Add a group with `resources: {walltime: "00:20:00", gpus: 1,
cpus: 8, memory: 32G}` (with `nccl`, `ddp.world_size: 1`). The log header
shows `GPUs 0` instead of `not set`. Then `gpus: 2` with `world_size: 2` for
DDP inside a node.

**C. Cooperative stop.** One task with a large `max_iterations`, then:

```bash
scancel --signal=USR1 --batch <jobid>_0
```

The log shows `received the scheduler advance signal...` and `finished:
exit 75 (interrupted)`; `status` shows `interrupted`. For the real time
limit: `walltime: "00:05:00"` with `stop_grace: 60`.

**D. A managed environment.** Follow [section 5](#5-managed-environments-building-on-the-cluster):
build on the login node (D1), reuse it (D2), then try a build in a job (D3),
and repeat stages A and B on it.

**E. Data.** Add an `upload: {}` source pointing at a small local directory.
`submit` shows rsync's progress; a second batch skips the upload.

## 5. Managed environments (building on the cluster)

### Choose the interpreter

`env.python_source` says where the environment's Python comes from;
`env.backend` (`venv`, default, or `conda`) says what kind of environment is
built from it. A `venv` uses that interpreter itself; a `conda` environment
installs its own Python at `environment.python`, so it is refused with
`python_source: site`. Usual pairs: `site` + `venv` (the default), `uv` +
`venv`, and `conda` + `conda`.

| `python_source` | Python from | Needs |
|---|---|---|
| `site` (default) | the site module (`module load python/...` in `hooks.common`) | `env.site_python_identity`, from `probe-python` |
| `uv` | `uv python install <environment.python>` | `uv` on `PATH` (via a hook), `environment.python` in the manifest |
| `conda` | conda's base interpreter | conda on `PATH` (via a hook, e.g. `module load conda`), `environment.python` |

Start with `site`. Record the interpreter's identity once and paste the
printed block into the profile (it replaces `env.existing`):

```bash
python -m training_framework.hpc probe-python
```

```yaml
    env:
      python_source: site
      site_python_identity: {implementation: cpython, version: "3.12.x",
                             executable: /N/soft/.../python3.12, platform: linux-x86_64}
```

If the site later changes that Python, `submit` stops and prints the new
block; pasting it gives a new, separate environment. The interpreter must
satisfy your project's `requires-python`, and `environment.python` (if set)
must match it.

### Write the pinned requirements file

Step by step, including installing uv and a project that depends on
training-framework: [Setting up a project for HPC batches](10-hpc-projects.md).

The manifest names a fully pinned file:

```yaml
environment:
  requirements: requirements/hpc.txt
  python: "3.12"                       # optional with site; required with uv/conda
```

Every line is `name==X.Y.Z` or a direct reference with a full 40-character
commit SHA (or an archive hash); the only option lines allowed are
`--index-url`, `--extra-index-url`, `--find-links`, and `--hash` after a
requirement. Anything else is reported with its line number by `validate`.

- **`checkout` mode** (this repository): the file must *not* contain
  `training-framework`; it comes from the staged commit.

  ```bash
  uv pip compile pyproject.toml --extra vision -o requirements/hpc.txt
  ```

- **`dependency` mode** (your project): the file must pin
  `training-framework` exactly once. From git, that is a direct reference
  to a full commit SHA, which `uv pip compile` writes when your project
  depends on the git URL:

  ```
  training-framework @ git+https://github.com/abhishekbisht1429/training-framework.git@<40-character sha>
  ```

  `git rev-parse dev` prints the full SHA. A private repository needs its
  own access from Big Red 200 (a deploy key or token).

- **CUDA torch**: add the PyTorch index to the file, e.g.
  `--extra-index-url https://download.pytorch.org/whl/cu124` (or compile
  with `--emit-index-url`), and pin `torch` to that index's version.

Commit the requirements file with the manifest.

### D1. Build on the login node

With `setup_location: login`, `submit` builds the environment on the login
node before submitting (installing torch takes several minutes), then
preflights it:

```bash
python -m training_framework.hpc submit batch.yaml
```

`submit` prints `environment: will be built` and `building the environment
on the login node`. The setup record is at `<batch-dir>/nodes/setup/record.json`
(`status: built`), and `status` shows a `setup` row. Then repeat stages A
and B.

### D2. Reuse

Submit the batch again (or another batch with the same requirements):
`submit` prints `environment: ready` and builds nothing. The environment
lives at `<roots.envs>/<key>`; the key is a hash of the pinned requirements,
the backend, the interpreter's identity, and the setup hooks. Changing any
of them gives a new environment next to the old one; old ones are never
deleted automatically. `submit --fresh-env` builds a private environment for
one batch.

### D3. Build in a job

Login nodes are shared; a large build is better in a job. Switch to
`setup_location: job` and give the setup job resources:

```yaml
    setup_location: job
    setup_resources: {walltime: "01:00:00", cpus: 4, memory: 16G}
```

Change a pin (or use `--fresh-env`) so there is something to build. The
batch now starts with a `setup` job; the arrays wait for it
(`afterok`) and are cancelled if it fails. Its log is
`<batch-dir>/logs/setup/<job>.out`:

```
[hpc setup ...] creating the venv environment at ...
[hpc setup ...] installing the pinned requirements from requirements/hpc.txt (this can take several minutes)
[hpc setup ...] checking installed dependencies (pip check)
[hpc setup ...] running preflight in the new environment
[hpc setup ...] preflight ok: Python 3.12.x | training_framework ...
[hpc setup ...] environment ready
```

A pip network error in that log means compute nodes cannot reach the
package index: keep `setup_location: login`. Two batches that need the same
new environment at once build it once: the second setup job logs `waiting
for the lease on ... held by ...` and then `another job finished building
the environment meanwhile`.

## 6. When something goes wrong

- `submit` prints the batch directory and the receipt path on failure; the
  receipt is `~/.local/state/training-framework/submissions/<batch>.json`.
- `sbatch` refusing with `Invalid account or account/partition combination
  specified`: the profile's account is missing, wrong, or not allowed on that
  partition; see [Finding your Slurm account](#finding-your-slurm-account).
  Nothing ran, and jobs submitted before it were cancelled.
- `needs reconciliation` prints a `squeue --name=...` query: run it before
  submitting again.
- Logs: `<batch-dir>/logs/<group>/<job>_<index>.out` and `logs/setup/`.
- `status` shows `ended without a record` (with the log's path) for a job
  killed before it could write its record: out of memory, or the time limit
  without enough `stop_grace`.
- A failed environment build ends its log with `FAILED: ...`; the pip output
  above it says why. It leaves no ready marker, so the next submit rebuilds.
- Clean up: `scancel -u $USER --name=<batch-id>.group-<group>`, then remove
  the batch directory. Environments under `roots.envs` are shared by
  batches; remove one only when no batch uses it.

Not yet verified on Big Red 200 itself: its `squeue` output as read by
`status`, its module names, and network access from compute nodes. Stages A
and D settle them.
