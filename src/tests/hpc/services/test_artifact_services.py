"""Submit with managed environments, uploads, and prepared data."""

from __future__ import annotations

import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import ExecResult
from training_framework.hpc.ports.local import LocalFilesystem, LocalFileTransfer, LocalProcessProbe
from training_framework.hpc.remote.lease import write_marker
from training_framework.hpc.services.artifacts import WorkstationProbe
from training_framework.hpc.services.submit import submit
from tests.hpc.batch_repo import CLOCK, PROFILE, TRAINING_CONFIG, FakeFacts, compile_repo, make_repo, manifest
from tests.hpc.fakes import FakeJobProbe, FakeSubmitter

IDENTITY = {"implementation": "cpython", "version": "3.12.4",
            "executable": "/opt/python/bin/python3", "platform": "linux-x86_64"}
FS = LocalFilesystem()


class ScriptedExecutor:
    """Answers the identity probe, preflight, and login-node steps."""

    def __init__(self, plan, identity=None, fail=()):
        self.plan, self.identity, self.fail = plan, identity or IDENTITY, set(fail)
        self.calls = []

    def run(self, argv, context=None, timeout=None):
        self.calls.append(argv)
        if "-c" in argv:
            return ExecResult(0, json.dumps(self.identity) + "\n", "")
        if "--preflight" in argv:
            facts = {
                "protocol_versions": [1], "interpreter": IDENTITY,
                "framework": {"file": "/env/training_framework/__init__.py", "distribution": "0.5.0"},
                "components": {p: None for p in self.plan.preflight["components_packages"]},
                "cluster_paths": {p: True for p in self.plan.preflight["cluster_paths"]},
            }
            return ExecResult(0, json.dumps(facts) + "\n", "")
        if argv[1].endswith("bootstrap.py"):
            if "setup" in self.fail:
                return ExecResult(1, "", "pip install failed")
            write_marker(FS, self.plan.environment["location"], self.plan.environment["key"])
            return ExecResult(0, "", "")
        if "--prepare" in argv:
            node_id = argv[argv.index("--prepare") + 1]
            artifact = next(a for a in self.plan.artifacts if a.key == self.plan.node(node_id).artifact)
            os.makedirs(artifact.location, exist_ok=True)
            write_marker(FS, artifact.location, artifact.key)
            return ExecResult(0, "", "")
        raise AssertionError(argv)


class Stager:
    def stage(self, sha, target):
        os.makedirs(target)


def profile_for(tmp_path, **cluster_changes):
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    cluster["roots"] = {key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    cluster.update(cluster_changes)
    return data


def compiled_for(tmp_path, data=None, config=None, facts=None, clock=CLOCK, **cluster_changes):
    root = make_repo(tmp_path, manifest=data, config=config)
    return compile_repo(root, profile_data=profile_for(tmp_path, **cluster_changes),
                        facts=facts or FakeFacts(root), clock=clock)


def run(compiled, tmp_path, submitter=None, executor=None, **kwargs):
    plan = compiled.plan
    submitter = submitter or FakeSubmitter()
    kwargs.setdefault("transfer", LocalFileTransfer())
    kwargs.setdefault("probe", WorkstationProbe(FakeJobProbe(), LocalProcessProbe()))
    kwargs.setdefault("owner", LocalProcessProbe().current_owner().as_json())
    receipt = submit(plan, compiled.manifest.raw, stager=Stager(), filesystem=FS,
                     executor=executor or ScriptedExecutor(plan), submitter=submitter,
                     receipts_dir=str(tmp_path / "receipts"), **kwargs)
    return receipt, submitter


def test_an_identity_mismatch_fails_before_any_cache_lookup(tmp_path):
    compiled = compiled_for(tmp_path)
    other = dict(IDENTITY, version="3.12.9")
    with pytest.raises(HpcError, match="does not match the profile's site_python_identity") as error:
        run(compiled, tmp_path, executor=ScriptedExecutor(compiled.plan, identity=other))
    assert 'version: "3.12.9"' in str(error.value)          # the block to paste
    assert not Path(compiled.plan.batch_dir).exists()        # nothing created


def test_a_missing_environment_is_built_by_a_setup_job_the_groups_wait_for(tmp_path):
    receipt, submitter = run(compiled_for(tmp_path), tmp_path)
    assert [(node, deps) for _, node, _, deps in submitter.submitted] == [
        ("setup", []), ("group-probes", ["1000"])]
    assert receipt.status == "submitted"


def test_a_ready_environment_skips_setup_and_is_preflighted(tmp_path):
    compiled = compiled_for(tmp_path)
    write_marker(FS, compiled.plan.environment["location"], compiled.plan.environment["key"])
    executor = ScriptedExecutor(compiled.plan)
    receipt, submitter = run(compiled, tmp_path, executor=executor)
    assert [node for _, node, _, _ in submitter.submitted] == ["group-probes"]
    assert submitter.submitted[0][3] == []
    assert receipt.node("setup")["status"] == "already ready"
    assert any("--preflight" in call for call in executor.calls)


def test_setup_on_the_login_node_runs_during_submit(tmp_path):
    compiled = compiled_for(tmp_path, setup_location="login")
    receipt, submitter = run(compiled, tmp_path)
    assert receipt.node("setup")["status"] == "ran on login"
    assert [node for _, node, _, _ in submitter.submitted] == ["group-probes"]


def test_a_failed_login_setup_fails_the_submit_with_no_jobs(tmp_path):
    compiled = compiled_for(tmp_path, setup_location="login")
    submitter = FakeSubmitter()
    with pytest.raises(HpcError, match="environment setup failed"):
        run(compiled, tmp_path, submitter=submitter,
            executor=ScriptedExecutor(compiled.plan, fail={"setup"}))
    assert not submitter.submitted


# Uploads -----------------------------------------------------------------------


def upload_batch(tmp_path, local, **cluster_changes):
    data = manifest()
    data["data"] = {"sources": [{"name": "flowers",
                                 "target": {"component": "flowers102", "path": "root"},
                                 "upload": {}}]}
    config = copy.deepcopy(TRAINING_CONFIG)
    config["sessions"][0]["flowers102"]["root"] = str(local)
    root_dir = tmp_path / "repo-dir"
    root_dir.mkdir(exist_ok=True)

    def compiled(clock=CLOCK):
        root = root_dir / "repo"
        if not root.exists():
            make_repo(root_dir, manifest=data, config=config)
        facts = FakeFacts(root, existing={str(local)},
                          contents={str(local): ("contentkey", 2, 9)})
        return compile_repo(root, profile_data=profile_for(tmp_path, **cluster_changes),
                            facts=facts, clock=clock)
    return compiled


def local_data(tmp_path):
    local = tmp_path / "datasets" / "flowers"
    local.mkdir(parents=True)
    (local / "a.jpg").write_bytes(b"12345")
    (local / "b.jpg").write_bytes(b"6789")
    return local


def test_an_upload_is_transferred_once_and_then_reused(tmp_path):
    local = local_data(tmp_path)
    make = upload_batch(tmp_path, local)
    first = make()
    run(first, tmp_path)
    upload = next(a for a in first.plan.artifacts if a.kind == "upload")
    assert Path(upload.location, "a.jpg").read_bytes() == b"12345"
    assert Path(upload.location + ".tf-hpc-ready").exists()
    reviewed = []
    second = make(clock=CLOCK + timedelta(minutes=1))
    run(second, tmp_path, review=lambda plan, summary: reviewed.append(summary) or True)
    assert reviewed[0].uploads == [] and reviewed[0].upload_bytes == 0


def test_an_upload_over_the_limit_is_declined_by_default(tmp_path):
    local = local_data(tmp_path)
    compiled = upload_batch(tmp_path, local)()
    receipt, submitter = run(compiled, tmp_path, upload_limit=5)
    assert receipt.status == "staged" and "upload_limit" in receipt.data["error"]
    upload = next(a for a in compiled.plan.artifacts if a.kind == "upload")
    assert not Path(upload.location).exists() and not submitter.submitted


def test_a_changed_upload_at_a_mapped_location_is_refused(tmp_path):
    local = local_data(tmp_path)
    compiled = upload_batch(tmp_path, local,
                            path_map={str(tmp_path / "datasets"): str(tmp_path / "cluster" / "project")})()
    upload = next(a for a in compiled.plan.artifacts if a.kind == "upload")
    write_marker(FS, upload.location, "an-older-key")
    with pytest.raises(HpcError, match="ready with key an-older-key"):
        run(compiled, tmp_path)


def test_an_upload_lease_held_on_another_host_is_never_taken_over(tmp_path):
    local = local_data(tmp_path)
    compiled = upload_batch(tmp_path, local)()
    upload = next(a for a in compiled.plan.artifacts if a.kind == "upload")
    lock = Path(upload.location + ".tf-hpc-lock")
    lock.mkdir(parents=True)
    (lock / "owner.json").write_text(json.dumps({"host": "laptop", "pid": 1, "start_time": "1"}))

    class NoAnswer:
        def is_alive(self, owner):
            return None

    with pytest.raises(HpcError, match="by hand"):
        run(compiled, tmp_path, probe=WorkstationProbe(NoAnswer(), LocalProcessProbe()))


# Prepared data -------------------------------------------------------------------


def prepare_batch(tmp_path, **cluster_changes):
    data = manifest()
    spec = {"command": ["python", "-m", "prep", "{path}"], "version": "1"}
    if cluster_changes.get("prepare_location") != "login":
        spec["resources"] = {"walltime": "01:00:00", "cpus": 2, "memory": "4G"}
    data["data"] = {"sources": [{"name": "inat", "target": {"component": "flowers102", "path": "root"},
                                 "prepare": spec}]}
    config = copy.deepcopy(TRAINING_CONFIG)
    config["sessions"][0]["flowers102"]["root"] = str(tmp_path / "cluster" / "inat")
    return compiled_for(tmp_path, data=data, config=config, **cluster_changes)


def test_a_ready_prepared_path_is_skipped_and_not_waited_for(tmp_path):
    compiled = prepare_batch(tmp_path)
    artifact = next(a for a in compiled.plan.artifacts if a.kind == "prepare")
    write_marker(FS, artifact.location, artifact.key)
    receipt, submitter = run(compiled, tmp_path)
    assert [(node, deps) for _, node, _, deps in submitter.submitted] == [
        ("setup", []), ("group-probes", ["1000"])]
    assert receipt.node("prepare-inat-0")["status"] == "already ready"


def test_a_pending_prepare_job_feeds_the_groups(tmp_path):
    receipt, submitter = run(prepare_batch(tmp_path), tmp_path)
    assert [(node, deps) for _, node, _, deps in submitter.submitted] == [
        ("setup", []), ("prepare-inat-0", ["1000"]), ("group-probes", ["1000", "1001"])]


def test_prepare_on_the_login_node_needs_the_environment_ready_first(tmp_path):
    compiled = prepare_batch(tmp_path, prepare_location="login")
    with pytest.raises(HpcError, match="setup_location: login"):
        run(compiled, tmp_path)
    compiled = prepare_batch(tmp_path / "2", prepare_location="login", setup_location="login")
    receipt, submitter = run(compiled, tmp_path / "2")
    assert receipt.node("prepare-inat-0")["status"] == "ran on login"
    assert [node for _, node, _, _ in submitter.submitted] == ["group-probes"]
