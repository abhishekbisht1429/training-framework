"""The prepare runner on the cluster: lease, command, marker, record."""

import copy
import json
import sys
from pathlib import Path

from training_framework.hpc.compile.render import plan_files
from training_framework.hpc.ports.local import LocalFilesystem
from training_framework.hpc.remote.prepare import run_prepare
from tests.hpc.batch_repo import PROFILE, TRAINING_CONFIG, compile_repo, make_repo, manifest

MAKE_DIR = "import os, sys; os.makedirs(sys.argv[1]); open(os.path.join(sys.argv[1], 'done'), 'w').close()"


def batch(tmp_path, command):
    data = manifest()
    data["data"] = {"sources": [{
        "name": "inat", "target": {"component": "flowers102", "path": "root"},
        "prepare": {"command": command, "version": "1",
                    "resources": {"walltime": "01:00:00", "cpus": 1, "memory": "1G"}}}]}
    config = copy.deepcopy(TRAINING_CONFIG)
    config["sessions"][0]["flowers102"]["root"] = str(tmp_path / "cluster" / "inat")
    profile = copy.deepcopy(PROFILE)
    profile["profiles"]["cluster"]["roots"] = {
        key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    root = make_repo(tmp_path, manifest=data, config=config)
    plan = compile_repo(root, profile_data=profile).plan
    filesystem = LocalFilesystem()
    filesystem.makedirs(f"{plan.batch_dir}/source")
    for relative, content in plan_files(plan, {}).items():
        filesystem.write(f"{plan.batch_dir}/{relative}", content.encode())
    return plan


def record(plan):
    return json.loads(Path(plan.batch_dir, "nodes", "prepare-inat-0", "record.json").read_text())


def test_a_prepared_path_gets_a_marker_and_is_then_skipped(tmp_path):
    plan = batch(tmp_path, [sys.executable, "-c", MAKE_DIR, "{path}"])
    assert run_prepare(f"{plan.batch_dir}/plan.json", "prepare-inat-0") == 0
    assert record(plan)["status"] == "prepared"
    location = tmp_path / "cluster" / "inat"
    assert (location / "done").exists() and Path(f"{location}.tf-hpc-ready").exists()
    assert not Path(f"{location}.tf-hpc-lock").exists()
    assert run_prepare(f"{plan.batch_dir}/plan.json", "prepare-inat-0") == 0
    assert record(plan)["status"] == "already ready"


def test_a_failing_command_leaves_no_marker_and_releases_the_lease(tmp_path):
    plan = batch(tmp_path, [sys.executable, "-c", "raise SystemExit(4)", "{path}"])
    assert run_prepare(f"{plan.batch_dir}/plan.json", "prepare-inat-0") == 1
    assert "exited 4" in record(plan)["error"]
    location = tmp_path / "cluster" / "inat"
    assert not Path(f"{location}.tf-hpc-ready").exists()
    assert not Path(f"{location}.tf-hpc-lock").exists()


def test_success_without_the_path_is_a_failure(tmp_path):
    plan = batch(tmp_path, [sys.executable, "-c", "pass", "{path}"])
    assert run_prepare(f"{plan.batch_dir}/plan.json", "prepare-inat-0") == 1
    assert "did not create" in record(plan)["error"]
