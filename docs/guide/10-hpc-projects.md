# Setting up a project for HPC batches

How to prepare a repository for the [HPC batch layer](08-hpc-batches.md):
which code mode to use, and how to produce the pinned requirements file a
managed environment is built from. Every key used here is described in the
[HPC configuration reference](../reference/hpc-configuration.md).

## Two code modes

| `code.framework` | When | The framework comes from | The requirements file |
|---|---|---|---|
| `checkout` | the repository *is* training-framework | the staged commit (`code.paths: ["src"]`) | must **not** contain `training-framework` |
| `dependency` | your project uses training-framework as a dependency | the cluster environment | must pin `training-framework` exactly once |

In both modes your own code is never pip-installed: `code.paths` puts it on
`PYTHONPATH` from the staged commit, so code changes do not rebuild the
environment. Only dependency changes do.

## Why a pinned requirements file

`pyproject.toml` says which versions your code is *compatible* with
(`numpy>=2.4`); a managed environment needs the *exact* versions to install.
The requirements file is that resolution, done once on your machine and
committed:

- The environment's key hashes it, so the same file always means the same
  packages, and batches with the same file share one environment.
- It is installed with `pip install --no-deps`: pip resolves nothing on the
  cluster, so the file must list every package, indirect dependencies
  included.
- `validate` checks it before anything reaches the cluster: every line is
  `name==X.Y.Z` or a direct reference with a full 40-character commit SHA or
  an archive hash, and the only options are `--index-url`,
  `--extra-index-url`, `--find-links`, and `--hash=`.

You do not need it with `env.existing` (an environment you manage yourself).

## Install uv

[uv](https://docs.astral.sh/uv/) resolves `pyproject.toml` for another
platform and Python version, which is what the cluster needs. Any one of:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # standalone, into ~/.local/bin
pip install uv
conda install -c conda-forge uv
```

```bash
uv --version
```

## The cluster's Python version

Resolve for the Python that runs on the cluster:

```bash
ssh <cluster> 'module load python/<version> && python3 --version'
# or, with a profile: python -m training_framework.hpc probe-python
```

The examples below use 3.12; use what the cluster reports. With
`python_source: site` it must match the profile's `site_python_identity`;
with `uv` or `conda` it is `environment.python`.

## `checkout` mode: this repository

```bash
cd training-framework
mkdir -p requirements
uv pip compile pyproject.toml --extra vision \
    --python-version 3.12 \
    --python-platform linux \
    -o requirements/hpc.txt
grep -i '^training-framework' requirements/hpc.txt    # must print nothing
```

`uv pip compile pyproject.toml` writes the dependencies only, not the
project itself. Leave out `--extra vision` if no batch uses the image
datasets. `--python-platform linux` resolves for Linux x86_64 even on a Mac.

```yaml
code: {framework: checkout, paths: ["src"]}
environment:
  requirements: requirements/hpc.txt
```

## `dependency` mode: a project that uses training-framework

### 1. Make the framework commit fetchable

The cluster installs the framework from Git, so the commit must be pushed:

```bash
cd training-framework
git push origin <branch>
git rev-parse <branch>               # the full 40-character SHA
```

Always pin the full SHA: branch names and short SHAs are refused, since they
can point at different code later.

### 2. Project layout

```
my-project/
  pyproject.toml
  my_project/
    __init__.py
    components.py          # session_config.components_package
  configs/train.yaml
  batch.yaml
  requirements/hpc.txt     # generated in step 4
```

### 3. Depend on the framework at that SHA

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "training-framework[vision] @ git+https://github.com/abhishekbisht1429/training-framework.git@<40-character sha>",
  # your own dependencies
]
```

Leave out `[vision]` if you do not use the built-in image datasets.

### 4. Generate the requirements

```bash
cd my-project
uv pip compile pyproject.toml \
    --python-version 3.12 \
    --python-platform linux \
    -o requirements/hpc.txt
grep -i '^training-framework' requirements/hpc.txt
# training-framework @ git+https://github.com/abhishekbisht1429/training-framework.git@<the same sha>
```

The framework must appear exactly once, at the full SHA; torch,
torchvision, and the rest are pinned with it.

### 5. Install the same framework on your workstation

The CLI runs locally, from the same commit, with the `hpc` extra:

```bash
pip install "training-framework[hpc] @ git+https://github.com/abhishekbisht1429/training-framework.git@<the same sha>"
```

### 6. The manifest

```yaml
version: 1
name: my-sweep
code:
  framework: dependency          # the framework comes from the environment
  paths: ["."]                   # puts my_project/ on PYTHONPATH (["src"] for a src layout)
environment:
  requirements: requirements/hpc.txt
  python: "3.12"                 # optional with python_source: site
training:
  config: configs/train.yaml
groups:
  cpu:
    resources: {walltime: "00:15:00", cpus: 2, memory: 8G}
    experiments:
      - name: seeds
        matrix: [{path: session_config.rng_seed, values: [1, 2]}]
```

In `configs/train.yaml`, set `session_config.components_package:
my_project.components`.

### 7. The profile

Let the HPC layer build the environment:

```yaml
env:
  python_source: site
  site_python_identity: {...}     # from probe-python
setup_location: login             # pip fetches from GitHub and PyPI on the login node
```

Or use `env.existing: /path/to/env`, an environment that already has this
framework commit and its dependencies (install them there with the same
`pip install ... @<sha>`).

### 8. Commit, validate, submit

```bash
git add -A && git commit -m "HPC batch setup"
python -m training_framework.hpc validate batch.yaml
python -m training_framework.hpc submit batch.yaml
python -m training_framework.hpc status <batch-id>
```

`validate` checks that the framework is pinned exactly once and everything
else is pinned; the setup's preflight checks that `training_framework`
imports from the environment.

### A private framework repository

`pip install` on the cluster must then authenticate to GitHub. Either:

- an SSH URL, `git+ssh://git@github.com/...@<sha>`, with a deploy key on the
  cluster; or
- a wheel: build it (`python -m build` in the framework repository), copy it
  where the cluster can read it, and pin it with its archive hash:
  `training-framework @ file:///N/.../training_framework-0.5.0-py3-none-any.whl#sha256=<hash>`
  (`sha256sum` prints the hash).

## CUDA builds of torch

torch's Linux wheels on PyPI include CUDA, so no extra index is needed for
the default build. For a specific CUDA build, add PyTorch's index (find the
right `cuXXX` for your torch version on pytorch.org):

```bash
uv pip compile ... \
    --extra-index-url https://download.pytorch.org/whl/<cuXXX> \
    --index-strategy unsafe-best-match \
    --emit-index-url
```

`--emit-index-url` writes the index into the file, where it is allowed.

## Without uv: freezing a tested environment

If you have a working environment on Linux x86_64 with the cluster's Python
version (for example a conda environment on a training machine):

```bash
python -m pip list --format=freeze --exclude-editable > requirements/hpc.txt
```

Use `pip list --format=freeze`, not `pip freeze`: for packages installed by
conda, `pip freeze` writes `name @ file:///...` paths that do not exist on
the cluster (and `validate` refuses). Then edit the file:

- remove conda tooling (`conda`, `conda-package-handling`, `mkl-service`,
  ...), whose versions may not exist on PyPI;
- `checkout` mode: remove `training-framework`; `dependency` mode: replace
  it with the `git+...@<sha>` line;
- a `torch==X.Y.Z+cuXXX` line needs `--extra-index-url
  https://download.pytorch.org/whl/cuXXX` at the top.

This is less reliable than `uv pip compile`: conda builds can differ from
the PyPI ones `pip install --no-deps` will fetch.

A conda environment *on the cluster* is a separate choice: `env.backend:
conda` creates it with Python and `env.conda_packages`, then installs this
file into it with pip, so the file holds pip packages only.

## Updating

When `pyproject.toml` changes (or, in `dependency` mode, to move to a new
framework commit: push it, put its SHA in `pyproject.toml`, and reinstall
the CLI at that SHA), run the same `uv pip compile` command again (add
`--upgrade` to take newer versions within your ranges) and commit the file.
The next batch builds a new environment once; earlier batches keep theirs.
