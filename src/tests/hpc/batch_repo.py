"""A throwaway Git repository with a batch manifest, for HPC tests."""

from __future__ import annotations

import copy
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from training_framework.hpc.compiler import compile_plan
from training_framework.hpc.repository import GitRepository
from training_framework.hpc.schema import parse_profiles

CLOCK = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

TRAINING_CONFIG = {
    "sessions": [{
        "session_config": {
            "rng_seed": 0,
            "sessions_dir": "./runs",
            "max_iterations": 10,
            "components_package": "my_project.components",
        },
        "resources": {
            "optimizer": {
                "optimizer": {"name": "AdamW", "kwargs": {"lr": 0.001}},
            },
        },
        "fine_tuned_model": {"frozen": []},
        "flowers102": {"root": "/placeholder/flowers", "split": "train"},
    }],
}

MANIFEST = {
    "version": 1,
    "name": "sweep",
    "code": {"framework": "dependency", "paths": ["."]},
    "environment": {"requirements": "requirements/hpc.txt"},
    "training": {"config": "configs/train.yaml"},
    "groups": {
        "probes": {
            "resources": {"walltime": "01:00:00", "gpus": 1, "cpus": 8, "memory": "64G"},
            "max_parallel": 2,
            "experiments": [{
                "name": "frozen",
                "overrides": [
                    {"component": "fine_tuned_model", "path": "frozen", "value": ["module.*"]},
                ],
                "matrix": [
                    {"component": "optimizer", "path": "optimizer.kwargs.lr", "values": [0.001, 0.0001]},
                    {"path": "session_config.rng_seed", "values": [41, 42]},
                ],
            }],
        },
    },
}

REQUIREMENTS = "torch==2.13.0\ntraining-framework==0.5.0\n"
PYPROJECT = '[project]\nname = "my-project"\nrequires-python = ">=3.12"\n'

PROFILE = {
    "default": "cluster",
    "profiles": {
        "cluster": {
            "ssh_host": "cluster",
            "roots": {"batches": "/scratch/me/batches", "envs": "/scratch/me/envs", "data": "/scratch/me/data"},
            "scheduler": "slurm",
            "slurm": {
                "defaults": {"account": "lab", "qos": "normal"},
                "partitions": {"gpu": "gpu", "cpu": "general"},
                "max_array_size": 1000,
            },
            "hooks": {"common": ["module load python/3.12"], "task": ["module load cuda"]},
            "env": {
                "python_source": "site",
                "site_python_identity": {
                    "implementation": "cpython", "version": "3.12.4",
                    "executable": "/opt/python/bin/python3", "platform": "linux-x86_64",
                },
            },
        },
    },
}


class FakeFacts:
    """`LocalFacts` with chosen existing paths and content keys."""

    def __init__(self, repository_root, home="/home/me", existing=(), contents=None):
        self.repository_root = str(repository_root)
        self.home = home
        self.existing = set(existing)
        self.contents = dict(contents or {})
        self.content_calls = []

    def exists(self, path):
        return path in self.existing

    def content(self, path):
        self.content_calls.append(path)
        return self.contents.get(path, (f"key-of-{path.strip('/').replace('/', '-')}", 3, 300))


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def make_repo(tmp_path: Path, *, manifest=None, config=None, requirements=REQUIREMENTS,
              pyproject=PYPROJECT, extra_files=None) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    files = {
        "batch.yaml": yaml.safe_dump(manifest if manifest is not None else MANIFEST),
        "configs/train.yaml": yaml.safe_dump(config if config is not None else TRAINING_CONFIG),
        "my_project/__init__.py": "",
    }
    if requirements is not None:
        files["requirements/hpc.txt"] = requirements
    if pyproject is not None:
        files["pyproject.toml"] = pyproject
    files.update(extra_files or {})
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(root, "init", "-q")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return root


def manifest(**changes):
    data = copy.deepcopy(MANIFEST)
    data.update(changes)
    return data


def profile(**env_changes):
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    for key, value in env_changes.items():
        if value is None:
            cluster.pop(key, None)
        else:
            cluster[key] = value
    return data


def compile_repo(root: Path, *, profile_data=None, facts=None, clock=CLOCK, fresh_env=False):
    return compile_plan(
        manifest_path="batch.yaml",
        repository=GitRepository(root),
        profile=parse_profiles(profile_data or PROFILE, None, "profiles.yaml"),
        facts=facts or FakeFacts(root),
        clock=clock,
        fresh_env=fresh_env,
    )
