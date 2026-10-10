"""The generated bootstrap on a "cluster" (this machine) that has only a bare
python3: no environment and no installed training_framework."""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training_framework.hpc.remote.bootstrap import render_bootstrap
from training_framework.hpc.ports.local import LocalFilesystem, LocalSourceStager
from training_framework.hpc.compile.render import plan_files
from training_framework.hpc.remote.task import interpreter_facts
from tests.hpc.batch_repo import MANIFEST, PROFILE, compile_repo, make_repo
from tests.hpc.stub_framework import FILES, build_wheel, pinned_line


def test_the_bundle_is_standard_library_python_3_8():
    source = render_bootstrap()
    tree = ast.parse(source, feature_version=(3, 8))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert "training_framework" not in roots
    assert roots <= set(sys.stdlib_module_names)
    for name in ("class Lease", "class StdlibFilesystem", "class ProcessProbe",
                 "class SlurmJobProbe", "class OwnerProbeRouter", "def compare", "def main"):
        assert name in source


def make_cluster_batch(tmp_path, mode):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    manifest = copy.deepcopy(MANIFEST)
    extra = {"my_project/components.py": "COMPONENTS = True\n"}
    if mode == "dependency":
        requirements = pinned_line("training-framework",
                                   build_wheel(wheels, "training-framework", "0.5.0", FILES))
    else:
        manifest["code"] = {"framework": "checkout", "paths": [".", "src"]}
        extra.update({f"src/{path}": text for path, text in FILES.items()})
        requirements = pinned_line("helper-pkg", build_wheel(wheels, "helper-pkg", "1.0", {"helper_pkg.py": ""}))
    root = make_repo(tmp_path, manifest=manifest, requirements=requirements, extra_files=extra)
    profile = copy.deepcopy(PROFILE)
    cluster = profile["profiles"]["cluster"]
    cluster["roots"] = {key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    cluster["env"]["site_python_identity"] = interpreter_facts()
    cluster["hooks"] = {}
    plan = compile_repo(root, profile_data=profile).plan
    filesystem = LocalFilesystem()
    filesystem.makedirs(plan.batch_dir)
    LocalSourceStager(root).stage(plan.sha, f"{plan.batch_dir}/source")
    for relative, content in plan_files(plan, {}).items():
        filesystem.write(f"{plan.batch_dir}/{relative}", content.encode())
    return plan


def run_bootstrap(plan, **env):
    environ = {key: value for key, value in os.environ.items() if key != "SLURM_JOB_ID"}
    environ.update(env)
    return subprocess.run([sys.executable, f"{plan.batch_dir}/bootstrap.py", "--plan",
                           f"{plan.batch_dir}/plan.json"], capture_output=True, text=True, env=environ)


def record(plan):
    return json.loads(Path(plan.batch_dir, "nodes", "setup", "record.json").read_text())


def marker(plan):
    path = Path(plan.environment["location"] + ".tf-hpc-ready")
    return json.loads(path.read_text())["key"] if path.exists() else None


def lock_exists(plan):
    return Path(plan.environment["location"] + ".tf-hpc-lock").exists()


@pytest.mark.parametrize("mode", ["dependency", "checkout"])
def test_setup_builds_an_environment_from_a_bare_python(tmp_path, mode):
    plan = make_cluster_batch(tmp_path, mode)
    result = run_bootstrap(plan)
    assert result.returncode == 0, result.stderr
    assert record(plan)["status"] == "built"
    assert marker(plan) == plan.environment["key"]
    assert not lock_exists(plan)
    framework = record(plan)["facts"]["framework"]
    if mode == "checkout":
        assert framework["file"].startswith(f"{plan.batch_dir}/source/src/")
        assert framework["distribution"] is None
    else:
        assert framework["distribution"] == "0.5.0"
    again = run_bootstrap(plan)
    assert again.returncode == 0 and record(plan)["status"] == "already ready"


def test_an_unsupported_protocol_fails_cleanly(tmp_path):
    plan = make_cluster_batch(tmp_path, "dependency")
    path = Path(plan.batch_dir, "plan.json")
    data = json.loads(path.read_text())
    data["protocol_version"] = 99
    path.write_text(json.dumps(data))
    result = run_bootstrap(plan)
    assert result.returncode == 1
    assert "plan uses 99" in record(plan)["error"]
    assert marker(plan) is None and not lock_exists(plan)


def test_a_failed_install_releases_the_lease_and_leaves_no_marker(tmp_path):
    plan = make_cluster_batch(tmp_path, "dependency")
    for wheel in (tmp_path / "wheels").iterdir():
        wheel.unlink()
    result = run_bootstrap(plan)
    assert result.returncode == 1
    assert "pip install failed" in record(plan)["error"]
    assert marker(plan) is None and not lock_exists(plan)


def test_a_failed_preflight_leaves_no_marker(tmp_path):
    plan = make_cluster_batch(tmp_path, "dependency")
    result = run_bootstrap(plan, STUB_PROTOCOLS="[]")
    assert result.returncode == 1
    assert "preflight failed" in record(plan)["error"] and "protocol" in record(plan)["error"]
    assert marker(plan) is None and not lock_exists(plan)


def test_a_dead_builders_partial_environment_is_replaced(tmp_path):
    plan = make_cluster_batch(tmp_path, "dependency")
    location = Path(plan.environment["location"])
    (location / "half-built").mkdir(parents=True)
    finished = subprocess.Popen(["true"])
    finished.wait()
    lock = Path(f"{location}.tf-hpc-lock")
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps(
        {"host": os.uname().nodename, "pid": finished.pid, "start_time": "1"}))
    result = run_bootstrap(plan)
    assert result.returncode == 0, result.stderr
    assert not (location / "half-built").exists()
    assert marker(plan) == plan.environment["key"]
