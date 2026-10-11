# First runs on Big Red 200

A checklist for bringing up the [HPC batch layer](08-hpc-batches.md) on
Indiana University's Big Red 200, one new risk per stage. Site details
(account, partitions, modules, limits) change; the commands in step 1 read
the current values from the cluster instead of assuming them.

## 0. On your workstation

```bash
pip install -e '.[hpc]'                         # or 'training-framework[hpc]'
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

Scratch is purged after a period without access; keep results you need
elsewhere.

## 2. The profile

`~/.config/training-framework/hpc.yaml`. The first runs use an environment
you already have on the cluster (with torch and the framework's
dependencies), so nothing is built:

```yaml
default: bigred200
profiles:
  bigred200:
    ssh_host: bigred200
    roots:
      batches: /N/scratch/<user>/hpc/batches
      envs: /N/scratch/<user>/hpc/envs
      data: /N/scratch/<user>/hpc/data
    scheduler: slurm
    slurm:
      defaults: {account: <your-account>}
      partitions: {gpu: gpu, cpu: general}
      max_array_size: <MaxArraySize>
    hooks:
      common: ["module load python/<version>"]   # from `module avail python`
    env:
      existing: /N/u/<user>/BigRed200/envs/tf    # your venv or conda env
    setup_location: login
    prepare_location: login
```

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

Go on only when a stage passes.

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

**D. A managed environment.** Remove `env.existing`, set `python_source:
site`, run `python -m training_framework.hpc probe-python`, and paste its
block into the profile. Add a fully pinned `environment.requirements` to the
manifest (see [section 4 of the HPC guide](08-hpc-batches.md#4-environments)),
commit, and submit. With `setup_location: login` the environment is built
during `submit` (torch takes a while); a second batch reports `environment:
ready`.

To learn whether compute nodes can install packages, run once with
`setup_location: job` and `setup_resources: {walltime: "01:00:00", cpus: 4,
memory: 16G}` and read `logs/setup/*.out`: a pip network error means compute
nodes have no internet; keep `login`.

**E. Data.** Add an `upload: {}` source pointing at a small local directory.
`submit` shows rsync's progress; a second batch skips the upload.

## 5. When something goes wrong

- `submit` prints the batch directory and the receipt path on failure; the
  receipt is `~/.local/state/training-framework/submissions/<batch>.json`.
- `needs reconciliation` prints a `squeue --name=...` query: run it before
  submitting again.
- Logs: `<batch-dir>/logs/<group>/<job>_<index>.out` and `logs/setup/`.
- `status` shows `ended without a record` (with the log's path) for a job
  killed before it could write its record: out of memory, or the time limit
  without enough `stop_grace`.
- Clean up: `scancel -u $USER --name=<batch-id>.group-<group>`, then remove
  the batch directory. Environments under `roots.envs` are shared by
  batches; remove one only when no batch uses it.

Not yet verified on Big Red 200 itself: its `squeue` output as read by
`status`, its module names, and network access from compute nodes. Stages A
and D settle them.
