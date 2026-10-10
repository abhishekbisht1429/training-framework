"""Fake `ssh`, `rsync`, `sbatch`, `squeue`, and `scancel` executables.

The fake `ssh` runs the remote command string with `sh -c` on this machine,
exactly as a remote shell would receive it, so quoting is really exercised.
The fake Slurm keeps its state in `$FAKE_SLURM_DIR`. Failure switches:
`FAKE_SSH_DOWN` (exit 255), `FAKE_SBATCH` = `refuse` | `garbage`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SSH = r'''
import os, sys
args = sys.argv[1:]
i = 0
while i < len(args) and args[i].startswith("-"):
    i += 2 if args[i] in ("-o", "-p", "-i", "-l") else 1
host, rest = args[i], args[i + 1:]
if rest and rest[0] == "--":
    rest = rest[1:]
if os.environ.get("FAKE_SSH_DOWN"):
    sys.stderr.write("ssh: connect to host %s: Connection refused\n" % host)
    sys.exit(255)
log = os.environ.get("FAKE_SSH_LOG")
if log:
    with open(log, "a") as handle:
        handle.write(host + "\n")
os.execvp("sh", ["sh", "-c", " ".join(rest)])
'''

RSYNC = r'''
import os, shutil, subprocess, sys
args = sys.argv[1:]
rsync_path = args[args.index("--rsync-path") + 1]
source, target = args[-2], args[-1].split(":", 1)[1]
subprocess.run(["sh", "-c", rsync_path.rsplit("&&", 1)[0]], check=True)
copied = 0
if source.endswith("/"):
    for directory, _, files in os.walk(source):
        for name in files:
            path = os.path.join(directory, name)
            destination = os.path.join(target, os.path.relpath(path, source))
            if os.path.exists(destination) and os.path.getsize(destination) == os.path.getsize(path):
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(path, destination)
            copied += os.path.getsize(path)
else:
    shutil.copy2(source, target)
    copied = os.path.getsize(source)
print("Number of files: 1\nTotal transferred file size: {:,} bytes".format(copied))
'''

SBATCH = r'''
import json, os, sys
state = os.environ["FAKE_SLURM_DIR"]
mode = os.environ.get("FAKE_SBATCH", "")
if mode == "refuse":
    sys.stderr.write("sbatch: error: invalid partition specified\n"); sys.exit(1)
counter = os.path.join(state, "counter")
job = int(open(counter).read()) + 1 if os.path.exists(counter) else 5000
open(counter, "w").write(str(job))
with open(os.path.join(state, "jobs.jsonl"), "a") as handle:
    handle.write(json.dumps({"job": str(job), "argv": sys.argv[1:]}) + "\n")
print("garbage" if mode == "garbage" else "%d;cluster" % job)
'''

SQUEUE = r'''
import json, os, sys
state = os.environ["FAKE_SLURM_DIR"]
job = sys.argv[sys.argv.index("--jobs") + 1]
jobs = set()
path = os.path.join(state, "jobs.jsonl")
if os.path.exists(path):
    jobs = {json.loads(line)["job"] for line in open(path)}
gone = set(open(os.path.join(state, "cancelled")).read().split()) if os.path.exists(os.path.join(state, "cancelled")) else set()
if job not in jobs:
    sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n"); sys.exit(1)
if job not in gone:
    print("RUNNING")
'''

SCANCEL = r'''
import json, os, sys
state = os.environ["FAKE_SLURM_DIR"]
job = sys.argv[1]
path = os.path.join(state, "jobs.jsonl")
jobs = {json.loads(line)["job"] for line in open(path)} if os.path.exists(path) else set()
if job not in jobs:
    sys.stderr.write("scancel: error: Invalid job id %s\n" % job); sys.exit(1)
with open(os.path.join(state, "cancelled"), "a") as handle:
    handle.write(job + "\n")
'''


@pytest.fixture
def fake_cluster(tmp_path, monkeypatch):
    """Puts the fakes first on PATH; returns the fake Slurm state dir."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    for name, body in (("ssh", SSH), ("rsync", RSYNC), ("sbatch", SBATCH),
                       ("squeue", SQUEUE), ("scancel", SCANCEL)):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)
    state = tmp_path / "fake-slurm"
    state.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(state))
    return state


def submitted_jobs(state: Path) -> list[dict]:
    import json

    path = state / "jobs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
