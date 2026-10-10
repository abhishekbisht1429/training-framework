"""`probe-python` and `submit` through the real cluster adapters (SSH, rsync,
Slurm, Git staging), against fake `ssh`/`rsync`/`sbatch` executables on this
machine. The bootstrap really builds a venv (with a stub framework wheel)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

from training_framework.hpc.__main__ import main
from tests.hpc.batch_repo import MANIFEST, PROFILE, make_repo
from tests.hpc.fake_cluster import fake_cluster, submitted_jobs  # noqa: F401 (fixture)
from tests.hpc.stub_framework import FILES, build_wheel, pinned_line


def setup(tmp_path, monkeypatch, identity=None):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    wheels = tmp_path / "wheels"
    wheels.mkdir(parents=True)
    requirements = pinned_line("training-framework",
                               build_wheel(wheels, "training-framework", "0.5.0", FILES))
    root = make_repo(tmp_path, manifest=copy.deepcopy(MANIFEST), requirements=requirements,
                     extra_files={"my_project/components.py": "COMPONENTS = True\n"})
    profile = copy.deepcopy(PROFILE)
    cluster = profile["profiles"]["cluster"]
    cluster["roots"] = {key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    cluster["hooks"] = {}
    cluster["setup_location"] = "login"
    cluster["env"] = {"bootstrap_python": sys.executable}
    if identity is not None:
        cluster["env"]["site_python_identity"] = identity
    path = tmp_path / "hpc.yaml"
    path.write_text(yaml.safe_dump(profile))
    return root, path


def probed_identity(path, capsys):
    assert main(["probe-python", "--profiles", str(path)]) == 0
    return yaml.safe_load(capsys.readouterr().out)["env"]["site_python_identity"]


def test_probe_python_then_submit_runs_the_whole_protocol(fake_cluster, tmp_path, monkeypatch, capsys):
    root, path = setup(tmp_path, monkeypatch)
    identity = probed_identity(path, capsys)
    root, path = setup(tmp_path / "second", monkeypatch, identity)

    assert main(["submit", str(root / "batch.yaml"), "--profiles", str(path), "--yes"]) == 0
    output = capsys.readouterr().out
    assert "environment: will be built" in output and "submitted:" in output

    (job,) = submitted_jobs(fake_cluster)
    batch_dir = Path(output.split("submitted: ")[1].splitlines()[0])
    assert job["argv"] == ["--parsable", "--kill-on-invalid-dep=yes",
                           f"{batch_dir}/groups/probes/group-probes.sbatch"]
    setup_record = json.loads((batch_dir / "nodes" / "setup" / "record.json").read_text())
    assert setup_record["status"] == "built"
    receipt = json.loads((batch_dir / "receipt.json").read_text())
    assert receipt["status"] == "submitted"
    assert receipt["nodes"]["setup"]["status"] == "ran on login"
    assert (batch_dir / "source" / "batch.yaml").exists()      # staged by bundle


def test_an_identity_mismatch_stops_submit_with_the_block_to_paste(fake_cluster, tmp_path, monkeypatch, capsys):
    root, path = setup(tmp_path, monkeypatch)
    stale = dict(probed_identity(path, capsys), executable="/old/python3")
    root, path = setup(tmp_path / "second", monkeypatch, stale)
    assert main(["submit", str(root / "batch.yaml"), "--profiles", str(path), "--yes"]) == 2
    error = capsys.readouterr().err
    assert "does not match the profile's site_python_identity" in error
    assert "site_python_identity:" in error
    assert not submitted_jobs(fake_cluster)


def test_a_non_interactive_submit_needs_yes(fake_cluster, tmp_path, monkeypatch, capsys):
    root, path = setup(tmp_path, monkeypatch)
    identity = probed_identity(path, capsys)
    root, path = setup(tmp_path / "second", monkeypatch, identity)
    monkeypatch.setattr("sys.stdin", open("/dev/null"))
    assert main(["submit", str(root / "batch.yaml"), "--profiles", str(path)]) == 1
    captured = capsys.readouterr()
    assert "pass --yes" in captured.err and "staged:" in captured.out
    assert not submitted_jobs(fake_cluster)
