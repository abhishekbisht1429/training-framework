"""The submit service and the transactional submission receipt (B11)."""

from __future__ import annotations

import copy
import json
import os

import pytest

from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.local import LocalFilesystem
from training_framework.hpc.ports import ExecResult
from training_framework.hpc.services.submit import PreflightFailed, submit
from tests.hpc.batch_repo import PROFILE, compile_repo, make_repo, manifest
from tests.hpc.fakes import Crash, FakeSubmitter


def profile_for(tmp_path):
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    cluster["roots"] = {key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    cluster["env"] = {"existing": str(tmp_path / "env")}
    return data


def two_groups():
    data = manifest(environment={})
    data["groups"]["second"] = {
        "resources": {"walltime": "01:00:00", "gpus": 1, "cpus": 2, "memory": "8G"},
        "experiments": [{"name": "solo", "matrix": [
            {"path": "session_config.rng_seed", "values": [99]}]}],
    }
    return data


class FakeStager:
    def __init__(self):
        self.staged = []

    def stage(self, sha, target):
        os.makedirs(target)
        self.staged.append((sha, target))


class FactsExecutor:
    """Answers preflight with facts matching the plan (or `override`)."""

    def __init__(self, plan, override=None, status=0):
        self.plan = plan
        self.override = override or {}
        self.status = status
        self.calls = []

    def run(self, argv, context=None, timeout=None):
        self.calls.append(argv)
        facts = {
            "protocol_versions": [self.plan.protocol_version],
            "interpreter": {"implementation": "cpython", "version": "3.12.4",
                            "executable": "/x", "platform": "linux-x86_64"},
            "framework": {"file": "/env/training_framework/__init__.py", "distribution": "0.5.0"},
            "components": {p: None for p in self.plan.preflight["components_packages"]},
            "cluster_paths": {p: True for p in self.plan.preflight["cluster_paths"]},
        }
        facts.update(self.override)
        return ExecResult(self.status, json.dumps(facts) + "\n", "boom" if self.status else "")


@pytest.fixture
def compiled(tmp_path):
    root = make_repo(tmp_path, manifest=two_groups(), requirements=None)
    return compile_repo(root, profile_data=profile_for(tmp_path))


def run(compiled, tmp_path, submitter, executor=None, **kwargs):
    return submit(
        compiled.plan, compiled.manifest.raw, stager=FakeStager(),
        filesystem=LocalFilesystem(), executor=executor or FactsExecutor(compiled.plan),
        submitter=submitter, receipts_dir=str(tmp_path / "receipts"), **kwargs,
    )


def receipt_on_disk(tmp_path, plan):
    return json.loads((tmp_path / "receipts" / f"{plan.batch_id}.json").read_text())


def test_a_successful_submit_records_every_job_in_order(compiled, tmp_path):
    submitter = FakeSubmitter()
    receipt = run(compiled, tmp_path, submitter)
    assert receipt.status == "submitted"
    assert receipt.submitted_jobs() == [("group-probes", "1000"), ("group-second", "1001")]
    plan = compiled.plan
    assert submitter.submitted[0][2] == f"{plan.batch_dir}/groups/probes/group-probes.sbatch"
    mirrored = json.loads((tmp_path / "cluster" / "batches" / plan.batch_id / "receipt.json").read_text())
    assert mirrored == receipt_on_disk(tmp_path, plan)
    assert (tmp_path / "cluster" / "batches" / plan.batch_id / "plan.json").exists()


@pytest.mark.parametrize("k", [0, 1])
def test_a_definite_failure_cancels_every_earlier_job(compiled, tmp_path, k):
    submitter = FakeSubmitter(fail_at=k, failure="definite")
    with pytest.raises(HpcError, match="failed"):
        run(compiled, tmp_path, submitter)
    data = receipt_on_disk(tmp_path, compiled.plan)
    assert data["status"] == "failed"
    assert [job_ref for job_ref, *_ in submitter.submitted] == [str(1000 + i) for i in range(k)]
    assert submitter.cancelled == [str(1000 + i) for i in reversed(range(k))]
    for job_ref, node_id, *_ in submitter.submitted:
        assert data["nodes"][node_id]["job_ref"] == job_ref
        assert data["nodes"][node_id]["status"] == "cancelled"


def test_an_uncertain_result_is_never_retried_and_needs_reconciliation(compiled, tmp_path):
    submitter = FakeSubmitter(fail_at=1, failure="uncertain")
    with pytest.raises(HpcError, match="squeue --name="):
        run(compiled, tmp_path, submitter)
    data = receipt_on_disk(tmp_path, compiled.plan)
    assert data["status"] == "needs reconciliation"
    assert data["nodes"]["group-second"]["status"] == "unknown"
    assert data["nodes"]["group-second"]["job_name"] == f"{compiled.plan.batch_id}.group-second"
    assert len(submitter.submitted) == 1 and submitter.cancelled == ["1000"]


def test_a_crash_after_an_accepted_call_leaves_the_node_findable(compiled, tmp_path):
    with pytest.raises(Crash):
        run(compiled, tmp_path, FakeSubmitter(fail_at=1, failure="crash"))
    data = receipt_on_disk(tmp_path, compiled.plan)
    assert data["status"] == "submitting"
    assert data["nodes"]["group-probes"]["job_ref"] == "1000"
    second = data["nodes"]["group-second"]
    assert second["status"] == "submitting"
    assert second["job_name"] == f"{compiled.plan.batch_id}.group-second"


def test_a_failed_cancel_is_recorded(compiled, tmp_path):
    with pytest.raises(HpcError):
        run(compiled, tmp_path, FakeSubmitter(fail_at=1, cancel_fails={"1000"}))
    cancel = receipt_on_disk(tmp_path, compiled.plan)["nodes"]["group-probes"]
    assert cancel["status"] == "cancel failed"
    assert "not found" in cancel["cancel"]["detail"]


def test_a_preflight_failure_ends_failed_with_no_jobs(compiled, tmp_path):
    submitter = FakeSubmitter()
    executor = FactsExecutor(compiled.plan, {"protocol_versions": [7]})
    with pytest.raises(PreflightFailed, match="protocol"):
        run(compiled, tmp_path, submitter, executor)
    data = receipt_on_disk(tmp_path, compiled.plan)
    assert data["status"] == "failed" and data["staged"] is True
    assert not submitter.submitted
    assert all(node["job_ref"] is None for node in data["nodes"].values())


def test_requires_python_is_checked_for_an_existing_environment(compiled, tmp_path):
    executor = FactsExecutor(compiled.plan, {"interpreter": {
        "implementation": "cpython", "version": "3.11.9", "executable": "/x", "platform": "p"}})
    with pytest.raises(PreflightFailed, match="does not satisfy requires-python"):
        run(compiled, tmp_path, FakeSubmitter(), executor)


def test_declining_at_confirmation_leaves_a_staged_receipt(compiled, tmp_path):
    submitter = FakeSubmitter()
    receipt = run(compiled, tmp_path, submitter, confirm=lambda plan: False)
    assert receipt.status == "staged" and not submitter.submitted


def test_the_receipt_exists_before_the_first_remote_mutation(compiled, tmp_path):
    seen = []

    class SpyFilesystem(LocalFilesystem):
        def makedirs(self, path):
            seen.append((tmp_path / "receipts" / f"{compiled.plan.batch_id}.json").exists())
            super().makedirs(path)

    submit(compiled.plan, compiled.manifest.raw, stager=FakeStager(), filesystem=SpyFilesystem(),
           executor=FactsExecutor(compiled.plan), submitter=FakeSubmitter(),
           receipts_dir=str(tmp_path / "receipts"))
    assert seen and seen[0] is True


def test_a_batch_directory_is_never_reused(compiled, tmp_path):
    os.makedirs(compiled.plan.batch_dir)
    with pytest.raises(HpcError, match="never reused"):
        run(compiled, tmp_path, FakeSubmitter())
    assert receipt_on_disk(tmp_path, compiled.plan)["status"] == "failed"


def test_receipt_updates_leave_no_temporary_files(compiled, tmp_path):
    run(compiled, tmp_path, FakeSubmitter())
    assert os.listdir(tmp_path / "receipts") == [f"{compiled.plan.batch_id}.json"]


def test_log_directories_exist_before_any_job_is_submitted(compiled, tmp_path):
    seen = []

    class CheckingSubmitter(FakeSubmitter):
        def submit(self, node, script_path, dependencies):
            seen.append(os.path.isdir(f"{compiled.plan.batch_dir}/logs/{node.group}"))
            return super().submit(node, script_path, dependencies)

    run(compiled, tmp_path, CheckingSubmitter())
    assert seen == [True, True]


def test_an_error_after_the_receipt_exists_carries_it(compiled, tmp_path):
    with pytest.raises(HpcError) as error:
        run(compiled, tmp_path, FakeSubmitter(fail_at=0))
    assert error.value.receipt.status == "failed"
    assert error.value.receipt.path.endswith(f"{compiled.plan.batch_id}.json")
