"""`resume` review fixes: failed attempts do not hide tasks, staging
failures record nothing, the batch's own scheduler settings render later
attempts, strict receipt shapes, and strict --checkpoint names."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from training_framework.hpc.__main__ import main
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.local import LocalFilesystem
from training_framework.hpc.services import resume as resuming
from training_framework.hpc.services.receipt import Receipt
from training_framework.hpc.services.status import receipt_problem
from tests.hpc.batch_repo import compile_repo, make_repo
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)
from tests.hpc.fakes import FakeSubmitter
from tests.hpc.services.test_resume import (
    DIALECT,
    chosen_tasks,
    plan_json,
    rows_of,
    submitted_with_checkpoints,
)
from tests.hpc.test_logs_and_status import _batch

FS = LocalFilesystem()


# 1. A failed attempt does not take over its tasks ---------------------------------------


@pytest.mark.parametrize("status", ["pending", "submitting", "refused", "cancelled"])
def test_an_attempt_that_never_ran_leaves_the_tasks_resumable(tmp_path, status):
    batch = _batch(tmp_path)
    batch["nodes"]["group-g.resume-2"] = {"status": status, "job_ref": None, "attempt": 2,
                                          "group": "g", "tasks": [2, 3]}
    files, rows = rows_of(batch)
    states = {row.name: row.state for row in rows}
    assert states["g/e/002"] == "interrupted" and states["g/e/003"] == "ended without a record"
    chosen = resuming.select_tasks(batch, rows, resuming.task_index(files.tables))
    assert [c.task for c in chosen] == ["g/e/002", "g/e/003"]


def test_an_attempt_whose_cancel_failed_still_counts(tmp_path):
    batch = _batch(tmp_path)
    batch["nodes"]["group-g.resume-2"] = {"status": "cancel failed", "job_ref": "12", "attempt": 2,
                                          "group": "g", "tasks": [2]}
    _, rows = rows_of(batch, {"12_2": "running"})
    assert {row.name: row.state for row in rows}["g/e/002"] == "running"


# 2. Staging and unexpected failures ----------------------------------------------------


class FailingWrites(LocalFilesystem):
    def write(self, path, data):
        raise PermissionError(f"cannot write {path}")


def attempt_for(tmp_path):
    batch_dir = tmp_path / "b"
    (batch_dir / "groups" / "g").mkdir(parents=True)
    data = dict(_batch(tmp_path / "x"), batch_id="b", staged=True, submission_order=[])
    attempt = resuming.build_attempt(plan_json(str(batch_dir)), data, chosen_tasks(), DIALECT,
                                     resuming.ResourceChanges())
    return Receipt(str(tmp_path / "receipt.json"), data), attempt


def test_a_staging_failure_records_and_submits_nothing(tmp_path):
    receipt, attempt = attempt_for(tmp_path)
    before = json.loads(json.dumps(receipt.data))
    submitter = FakeSubmitter()
    with pytest.raises(HpcError, match="nothing was submitted"):
        resuming.submit_attempt(receipt, attempt, FailingWrites(), submitter)
    assert receipt.data == before and not submitter.submitted


class ExplodingSubmitter(FakeSubmitter):
    def submit(self, node, script_path, dependencies):
        raise RuntimeError("connection reset while reading the reply")


def test_an_unexpected_submit_error_needs_reconciliation(tmp_path):
    receipt, attempt = attempt_for(tmp_path)
    with pytest.raises(HpcError, match="failed unexpectedly"):
        resuming.submit_attempt(receipt, attempt, FS, ExplodingSubmitter())
    entry = receipt.data["attempts"][-1]
    assert entry["status"] == "needs reconciliation" and "squeue --name=" in entry["error"]
    assert receipt.data["nodes"]["group-g.resume-2"]["status"] == "unknown"


# 3. The batch's own scheduler settings render later attempts ----------------------------


def test_the_plan_records_the_scheduler_settings_it_was_rendered_with(tmp_path):
    plan = compile_repo(make_repo(tmp_path)).plan.as_json()
    settings = plan["scheduler_settings"]
    assert set(settings) == {"defaults", "partitions", "max_array_size"}
    dialect = resuming.plan_dialect(plan)
    assert dialect.settings.defaults == settings["defaults"]
    with pytest.raises(HpcError, match="scheduler settings in plan.json"):
        resuming.plan_dialect(dict(plan, scheduler_settings={}))


def test_resume_keeps_the_original_account_after_the_profile_changes(
        fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_with_checkpoints(fake_cluster, tmp_path, monkeypatch, capsys)
    monkeypatch.setenv("STUB_CAPABILITIES", '["resume"]')
    original = json.loads((batch_dir / "plan.json").read_text())["scheduler_settings"]["defaults"]
    data = yaml.safe_load(profiles.read_text())
    cluster = data["profiles"][data["default"]]
    cluster["slurm"]["defaults"] = {"account": "a-different-allocation"}
    profiles.write_text(yaml.safe_dump(data))
    assert main(["resume", batch_dir.name, "--profiles", str(profiles), "--dry-run"]) == 0
    script = capsys.readouterr().out
    assert "a-different-allocation" not in script
    for key, value in original.items():
        assert f"#SBATCH --{key}={value}" in script
    # Without any profile file at all, resume still works from the receipt and plan.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    assert main(["resume", batch_dir.name, "--dry-run"]) == 0


# 4. Strict receipt shapes ---------------------------------------------------------------


@pytest.mark.parametrize("change", [
    lambda r: r["nodes"]["group-g.resume-2"].update(tasks=["1"]),
    lambda r: r["nodes"]["group-g.resume-2"].update(attempt="2"),
    lambda r: r["nodes"]["group-g.resume-2"].update(attempt=1),
    lambda r: r.update(attempts=[{"attempt": 2}]),                       # no status
    lambda r: r.update(attempts=[{"attempt": 2, "status": "submitted", "tasks": []}]),
    lambda r: r.update(attempts=["2"]),
])
def test_malformed_attempts_make_a_receipt_invalid(tmp_path, change):
    receipt = _batch(tmp_path)
    receipt["nodes"]["group-g.resume-2"] = {"status": "submitted", "job_ref": "12", "attempt": 2,
                                            "group": "g", "tasks": [2]}
    receipt["attempts"] = [{"attempt": 2, "status": "submitted", "tasks": {}}]
    assert receipt_problem(receipt) is None
    change(receipt)
    assert receipt_problem(receipt) is not None


# 5. --checkpoint must name a task being resumed -----------------------------------------


def test_a_checkpoint_for_a_task_not_being_resumed_is_an_error(
        fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_with_checkpoints(fake_cluster, tmp_path, monkeypatch, capsys)
    monkeypatch.setenv("STUB_CAPABILITIES", '["resume"]')
    task = sorted(batch_dir.glob("tasks/*/*/*/sessions/session_1/checkpoints/x"))[0]
    relative = task.relative_to(batch_dir)
    assert main(["resume", batch_dir.name, "--profiles", str(profiles), "--dry-run",
                 "--checkpoint", f"probes/typo/000={relative}"]) == 2
    error = capsys.readouterr().err
    assert "--checkpoint names probes/typo/000, which this command does not resume" in error
