"""`resume`: selection, checkpoint discovery, cluster checks, rendering, the
receipt's attempts, attempt-aware status, and a real resumed engine run."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from training_framework.hpc.__main__ import main
from training_framework.hpc.config.schema import SlurmSettings, parse_profiles
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import ExecResult
from training_framework.hpc.ports.local import LocalExecutor, LocalFilesystem, LocalSourceStager
from training_framework.hpc.remote.task import _previous_attempts
from training_framework.hpc.scheduler.slurm import SlurmDialect
from training_framework.hpc.services import resume as resuming
from training_framework.hpc.services.receipt import Receipt
from training_framework.hpc.services.status import fetch_batches, task_rows
from training_framework.hpc.services.submit import submit
from tests.hpc.batch_repo import PROFILE
from tests.hpc.fake_cluster import fake_cluster, submitted_jobs  # noqa: F401 (fixture)
from tests.hpc.fakes import FakeSubmitter, LocalScheduler
from tests.hpc.test_logs_and_status import _batch

FS = LocalFilesystem()
DIALECT = SlurmDialect(SlurmSettings({"account": "lab"}, {"gpu": "gpu", "cpu": "general"}, 1000))


def write_checkpoint(directory: Path, iteration: int) -> None:
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"format_version": 1, "iteration": iteration},
                                                        indent=2))


def task_dir(batch: dict, name: str) -> Path:
    return Path(batch["batch_dir"]) / "tasks" / name


def with_checkpoints(tmp_path, max_iterations=1000):
    batch = _batch(tmp_path)
    for name in ("g/e/002", "g/e/003"):
        directory = task_dir(batch, name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(json.dumps({"sessions": [{
            "session_config": {"max_iterations": max_iterations},
            "checkpointer": {"checkpoint_every": 100}}]}))
        checkpoints = directory / "sessions" / "session_1" / "checkpoints"
        write_checkpoint(checkpoints / "a", 100)
        write_checkpoint(checkpoints / "b", 300)
        write_checkpoint(checkpoints / "c.tmp", 500)          # still being written
        (checkpoints / "d").mkdir()                          # no manifest: incomplete
    return batch


def rows_of(batch, live=None):
    files = fetch_batches(LocalExecutor(), [batch["batch_dir"]])[batch["batch_dir"]]
    return files, task_rows(batch, files, live or {})


# Selection --------------------------------------------------------------------------


def test_the_default_selection_is_interrupted_and_ended_without_a_record(tmp_path):
    batch = _batch(tmp_path)
    files, rows = rows_of(batch)
    index = resuming.task_index(files.tables)
    chosen = resuming.select_tasks(batch, rows, index)
    assert [(c.task, c.state, c.index) for c in chosen] == [
        ("g/e/002", "interrupted", 2), ("g/e/003", "ended without a record", 3)]
    with_failed = resuming.select_tasks(batch, rows, index, include_failed=True)
    assert [c.task for c in with_failed] == ["g/e/001", "g/e/002", "g/e/003"]
    assert [c.task for c in resuming.select_tasks(batch, rows, index, ["g/e/001"])] == ["g/e/001"]


def test_finished_and_unknown_tasks_cannot_be_named(tmp_path):
    batch = _batch(tmp_path)
    files, rows = rows_of(batch)
    with pytest.raises(HpcError) as error:
        resuming.select_tasks(batch, rows, resuming.task_index(files.tables), ["g/e/000", "nope"])
    assert "'g/e/000' finished (done)" in str(error.value)
    assert "'nope' is not a task of this batch" in str(error.value)


def test_jobs_still_queued_are_found_across_attempts():
    receipt = {"nodes": {"group-g": {"status": "submitted", "job_ref": "9"},
                         "group-g.resume-2": {"status": "submitted", "job_ref": "12",
                                              "attempt": 2, "group": "g", "tasks": [2]}}}
    live = {"9_0": "completed", "12_2": "pending", "77_0": "running"}
    assert resuming.active_jobs(receipt, live) == ["12_2"]


# Checkpoints ------------------------------------------------------------------------


def test_the_newest_complete_checkpoint_is_chosen(tmp_path):
    batch = with_checkpoints(tmp_path)
    found, configs = resuming.find_checkpoints(LocalExecutor(), batch["batch_dir"], ["g/e/002"])
    paths = sorted(Path(path).name for path, _ in found["g/e/002"])
    assert paths == ["a", "b"]                               # not c.tmp, not d
    candidate = resuming.Candidate("g/e/002", "g", 2, "interrupted")
    resuming.choose_checkpoints([candidate], found, configs)
    assert Path(candidate.checkpoint).name == "b" and candidate.iteration == 300
    assert candidate.max_iterations == 1000 and candidate.reason is None


def test_reasons_when_a_task_cannot_be_resumed(tmp_path):
    batch = with_checkpoints(tmp_path, max_iterations=300)
    found, configs = resuming.find_checkpoints(LocalExecutor(), batch["batch_dir"],
                                               ["g/e/002", "g/e/001"])
    at_end = resuming.Candidate("g/e/002", "g", 2, "interrupted")
    no_config = resuming.Candidate("g/e/001", "g", 1, "failed (exit 1)")
    resuming.choose_checkpoints([at_end, no_config], found, configs)
    assert "nothing is left to run" in at_end.reason
    assert "no checkpointer" in no_config.reason
    configs["g/e/001"] = {"sessions": [{"checkpointer": {"checkpoints_dir": "/elsewhere"}}]}
    explicit_dir = resuming.Candidate("g/e/001", "g", 1, "failed (exit 1)")
    resuming.choose_checkpoints([explicit_dir], found, configs)
    assert "explicit checkpoints_dir is not searched" in explicit_dir.reason


def test_explicit_checkpoints_are_checked(tmp_path):
    batch = with_checkpoints(tmp_path)
    root = batch["batch_dir"]
    chosen = resuming.explicit_checkpoints(FS, root, ["g/e/002=tasks/g/e/002/sessions/session_1/checkpoints/a"])
    assert chosen["g/e/002"] == (f"{root}/tasks/g/e/002/sessions/session_1/checkpoints/a", 100)
    for bad, message in (("x=/etc", "outside the batch directory"),
                         ("x=tasks/g/e/002/sessions/session_1/checkpoints/c.tmp", "incomplete"),
                         ("x=tasks/g/e/002/sessions/session_1/checkpoints/d", "no manifest.json"),
                         ("no-equals", "must be NAME=PATH")):
        with pytest.raises(HpcError, match=message):
            resuming.explicit_checkpoints(FS, root, [bad])


# Rendering --------------------------------------------------------------------------


def plan_json(batch_dir: str, with_resources=True) -> dict:
    node = {"id": "group-g", "kind": "task_set", "group": "g", "max_parallel": None,
            "context": {"phase": "task", "interpreter": "/env/bin/python", "environment": "/env",
                        "working_directory": f"{batch_dir}/source", "pythonpath": [f"{batch_dir}/source"],
                        "set_variables": [["PYTHONNOUSERSITE", "1"]], "unset_variables": ["WORLD_SIZE"],
                        "hooks": ["module load python"], "path_prepend": ["/env/bin"], "launcher": []},
            "stop_grace_seconds": 120}
    if with_resources:
        node["resources"] = {"walltime": "01:00:00", "walltime_seconds": 3600, "gpus": 1,
                             "cpus": 8, "memory": "64G", "scheduler_options": {}}
    return {"batch_id": "b", "batch_dir": batch_dir, "engine_options": ["--heartbeat-timeout", "600"],
            "nodes": [node]}


def chosen_tasks():
    return [resuming.Candidate("g/e/001", "g", 1, "interrupted", "/b/ck/1", 300, 1000),
            resuming.Candidate("g/e/003", "g", 3, "interrupted", "/b/ck/3", 200, 1000)]


def test_an_attempt_renders_only_its_tasks_with_the_new_resources():
    receipt = {"attempts": []}
    attempt = resuming.build_attempt(plan_json("/b"), receipt, chosen_tasks(), DIALECT,
                                     resuming.ResourceChanges(walltime="2-00:00:00", max_parallel=1))
    (group,) = attempt.groups
    assert attempt.attempt == 2 and group.node.id == "group-g.resume-2"
    script = group.script
    for line in ("#SBATCH --array=1,3%1", "#SBATCH --time=2-00:00:00", "#SBATCH --cpus-per-task=8",
                 "#SBATCH --gres=gpu:1", "#SBATCH --job-name=b.group-g.resume-2",
                 "#SBATCH --signal=B:USR1@150", "#SBATCH --account=lab"):
        assert line in script, line
    assert "/b/groups/g/resume-2.tsv" in script and '--resume "$checkpoint" --attempt 2' in script
    assert "--heartbeat-timeout 600" in script and "module load python" in script
    assert group.table.splitlines() == ["index\ttask\tcheckpoint\titeration",
                                        "1\tg/e/001\t/b/ck/1\t300", "3\tg/e/003\t/b/ck/3\t200"]


def test_rendering_refuses_old_plans_and_a_walltime_shorter_than_the_signal_lead():
    with pytest.raises(HpcError, match="submitted before batches recorded what resume needs"):
        resuming.build_attempt(plan_json("/b", with_resources=False), {}, chosen_tasks(), DIALECT,
                               resuming.ResourceChanges())
    with pytest.raises(HpcError, match="not shorter than the walltime"):
        resuming.build_attempt(plan_json("/b"), {}, chosen_tasks(), DIALECT,
                               resuming.ResourceChanges(walltime="00:02:00"))
    with pytest.raises(HpcError, match="--memory must be a size"):
        resuming.ResourceChanges(memory="lots").check()


def test_the_resume_table_feeds_the_wrapper_its_checkpoint(tmp_path):
    batch_dir = tmp_path / "b"
    (batch_dir / "groups" / "g").mkdir(parents=True)
    (batch_dir / "groups" / "g" / "tasks.tsv").write_text(
        "index\ttask\tvariant\tconfig\tconfig_hash\n3\tg/e/003\t[]\ttasks/g/e/003/config.yaml\th\n")
    attempt = resuming.build_attempt(plan_json(str(batch_dir)), {}, chosen_tasks()[1:], DIALECT,
                                     resuming.ResourceChanges())
    (group,) = attempt.groups
    Path(group.table_path).write_text(group.table)
    from training_framework.hpc.scheduler.slurm import _task_body

    body = _task_body(str(batch_dir), "g", [sys.executable, "-c",
                                            "import sys; print('ARGS', sys.argv[1:])",
                                            "--resume", "{checkpoint}", "--config", "{config}"],
                      group.table_path)
    script = tmp_path / "job.sh"
    script.write_text("\n".join(["#!/bin/bash", *body]) + "\n")
    import subprocess
    result = subprocess.run(["bash", str(script)], env={**os.environ, "SLURM_ARRAY_TASK_ID": "3"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"'--resume', '/b/ck/3', '--config', '{batch_dir}/tasks/g/e/003/config.yaml'" in result.stdout


# The cluster check ------------------------------------------------------------------


class PreflightExecutor:
    def __init__(self, capabilities):
        self.capabilities = capabilities

    def run(self, argv, context=None, timeout=None):
        facts = {"protocol_versions": [1], "capabilities": self.capabilities,
                 "interpreter": {}, "components": {}, "cluster_paths": {},
                 "framework": {"file": "/env/lib/training_framework/__init__.py",
                               "distribution": "0.5.0"}}
        return ExecResult(0, json.dumps(facts) + "\n", "")


def test_an_environment_without_resume_support_is_refused(tmp_path):
    batch_dir = tmp_path / "b"
    (batch_dir / "source").mkdir(parents=True)
    plan = dict(plan_json(str(batch_dir)), environment={"managed": False},
                preflight={"protocol_version": 1,
                           "framework": {"location": "environment", "distribution": "0.5.0"}})
    with pytest.raises(HpcError, match="predates resume"):
        resuming.check_cluster(plan, FS, PreflightExecutor([]))
    resuming.check_cluster(plan, FS, PreflightExecutor(["resume"]))
    shutil.rmtree(batch_dir / "source")
    with pytest.raises(HpcError, match="source checkout .* is gone"):
        resuming.check_cluster(plan, FS, PreflightExecutor(["resume"]))


# Submitting and recording an attempt --------------------------------------------------


def staged_attempt(tmp_path, receipt_data, submitter):
    batch_dir = tmp_path / "b"
    (batch_dir / "groups" / "g").mkdir(parents=True, exist_ok=True)
    (batch_dir / "groups" / "h").mkdir(parents=True, exist_ok=True)
    tasks = chosen_tasks() + [resuming.Candidate("h/e/000", "h", 0, "interrupted", "/b/ck/h", 5, 10)]
    plan = plan_json(str(batch_dir))
    plan["nodes"].append(dict(plan["nodes"][0], id="group-h", group="h"))
    attempt = resuming.build_attempt(plan, receipt_data, tasks, DIALECT, resuming.ResourceChanges())
    receipt = Receipt(str(tmp_path / "receipt.json"), receipt_data)
    return receipt, attempt


def test_a_submitted_attempt_is_recorded_in_the_receipt(tmp_path):
    data = dict(_batch(tmp_path / "x"), batch_id="b", staged=True, submission_order=["group-g"])
    receipt, attempt = staged_attempt(tmp_path, data, None)
    submitter = FakeSubmitter()
    entry = resuming.submit_attempt(receipt, attempt, FS, submitter, submitted_from="laptop")
    assert entry["status"] == "submitted" and entry["attempt"] == 2
    assert entry["tasks"]["g/e/003"] == {"group": "g", "index": 3, "checkpoint": "/b/ck/3",
                                         "iteration": 200}
    node = receipt.data["nodes"]["group-g.resume-2"]
    assert node["job_ref"] == "1000" and node["tasks"] == [1, 3] and node["attempt"] == 2
    assert Path(attempt.groups[0].script_path).exists() and Path(attempt.groups[0].table_path).exists()
    saved = json.loads(Path(receipt.path).read_text())
    assert saved["attempts"][0]["status"] == "submitted"


def test_a_failed_attempt_cancels_only_its_own_jobs(tmp_path):
    data = dict(_batch(tmp_path / "x"), batch_id="b", staged=True, submission_order=["group-g"])
    data["nodes"]["group-g"]["job_ref"] = "9"
    receipt, attempt = staged_attempt(tmp_path, data, None)
    submitter = FakeSubmitter(fail_at=1)                      # the second group's sbatch fails
    with pytest.raises(HpcError, match="failed"):
        resuming.submit_attempt(receipt, attempt, FS, submitter)
    assert submitter.cancelled == ["1000"]                    # not the first attempt's job 9
    assert receipt.data["attempts"][-1]["status"] == "failed"
    assert receipt.data["status"] == "submitted"              # the batch itself is unchanged


def test_an_uncertain_attempt_needs_reconciliation(tmp_path):
    data = dict(_batch(tmp_path / "x"), batch_id="b", staged=True, submission_order=[])
    receipt, attempt = staged_attempt(tmp_path, data, None)
    with pytest.raises(HpcError, match="uncertain outcome"):
        resuming.submit_attempt(receipt, attempt, FS, FakeSubmitter(fail_at=0, failure="uncertain"))
    assert receipt.data["attempts"][-1]["status"] == "needs reconciliation"
    assert receipt.data["nodes"]["group-g.resume-2"]["status"] == "unknown"


# Status follows the latest attempt ----------------------------------------------------


def test_status_rows_follow_the_latest_attempt(tmp_path):
    batch = _batch(tmp_path)
    batch["nodes"]["group-g.resume-2"] = {"status": "submitted", "job_ref": "12", "attempt": 2,
                                          "group": "g", "tasks": [2, 3]}
    files, rows = rows_of(batch, {"12_3": "running"})
    states = {row.name: (row.state, row.detail) for row in rows}
    # Task 2's record is from attempt 1: attempt 2 ended without writing one.
    assert states["g/e/002"] == ("ended without a record", "see logs/g/12_2.out (attempt 2)")
    assert states["g/e/003"] == ("running", "job 12_3 (attempt 2)")
    assert states["g/e/000"] == ("done", "/runs/a")           # not part of attempt 2
    record = task_dir(batch, "g/e/002") / "record.json"
    record.write_text(json.dumps({"exit_status": 0, "interrupted": False, "attempt": 2,
                                  "run_directories": ["/runs/x"]}))
    _, rows = rows_of(batch)
    assert {row.name: row.detail for row in rows}["g/e/002"] == "/runs/x (attempt 2)"


def test_a_records_history_keeps_earlier_attempts(tmp_path):
    path = tmp_path / "record.json"
    assert _previous_attempts(str(path)) == []
    path.write_text(json.dumps({"exit_status": 75, "previous_attempts": [{"attempt": 1}],
                                "attempt": 2}))
    assert _previous_attempts(str(path)) == [{"attempt": 1}, {"exit_status": 75, "attempt": 2}]


# End to end with the real engine ------------------------------------------------------


def test_a_resumed_task_continues_from_its_checkpoint_and_finishes(tmp_path, tmp_path_factory):
    pytest.importorskip("torch")
    from tests.hpc import test_end_to_end as e2e

    environment = e2e.environment.__wrapped__(tmp_path_factory)
    session = e2e.session(max_iterations=6)
    session["checkpointer"] = {"checkpoint_every": 3}
    root, compiled = e2e.batch(tmp_path, environment, session, [5])
    receipt = submit(compiled.plan, compiled.manifest.raw, stager=LocalSourceStager(root),
                     filesystem=FS, executor=LocalExecutor(), submitter=LocalScheduler(),
                     receipts_dir=str(tmp_path / "receipts"))
    plan = compiled.plan
    task = Path(plan.batch_dir, "tasks/cpu/seeds/000")
    checkpoints = sorted(task.glob("sessions/session_*/checkpoints/*/manifest.json"))
    iterations = {json.loads(path.read_text())["iteration"]: path.parent for path in checkpoints}
    assert sorted(iterations) == [3, 6]
    # As if the time limit stopped it after iteration 3's checkpoint.
    shutil.rmtree(iterations[6])
    record = task / "record.json"
    first = json.loads(record.read_text())
    record.write_text(json.dumps(dict(first, exit_status=75, interrupted=True)))

    executor = LocalExecutor()
    files = fetch_batches(executor, [plan.batch_dir])[plan.batch_dir]
    rows = task_rows(receipt.data, files, {})
    candidates = resuming.select_tasks(receipt.data, rows, resuming.task_index(files.tables))
    found, configs = resuming.find_checkpoints(executor, plan.batch_dir, [c.task for c in candidates])
    resuming.choose_checkpoints(candidates, found, configs)
    assert [(c.task, c.iteration, c.max_iterations) for c in candidates] == [("cpu/seeds/000", 3, 6)]
    plan_data = resuming.read_plan(FS, plan.batch_dir)
    dialect = SlurmDialect(parse_profiles(PROFILE, None, "p").slurm)
    attempt = resuming.build_attempt(plan_data, receipt.data, candidates, dialect,
                                     resuming.ResourceChanges())
    resuming.check_cluster(plan_data, FS, executor)
    scheduler = LocalScheduler()
    resuming.submit_attempt(receipt, attempt, FS, scheduler)
    assert scheduler.results == {"group-cpu.resume-2": [0]}, scheduler.outputs
    output = scheduler.outputs["group-cpu.resume-2"][0]
    assert "attempt 2: resuming from" in output and str(iterations[3]) in output

    resumed = json.loads(record.read_text())
    assert resumed["exit_status"] == 0 and resumed["attempt"] == 2
    assert resumed["resumed_from"] == str(iterations[3])
    assert resumed["previous_attempts"][0]["exit_status"] == 75
    assert resumed["run_directories"] == first["run_directories"]       # the same session
    final = [json.loads(path.read_text())["iteration"]
             for path in task.glob("sessions/session_*/checkpoints/*/manifest.json")]
    assert sorted(final) == [3, 6]                                       # iteration 6 written again
    files = fetch_batches(executor, [plan.batch_dir])[plan.batch_dir]
    (row,) = [r for r in task_rows(receipt.data, files, {}) if r.name == "cpu/seeds/000"]
    assert row.state == "done" and row.detail.endswith("(attempt 2)")


# The CLI, through the fake cluster ------------------------------------------------------

SQUEUE_IDLE = "import sys\n"          # nothing of the user's is queued


def submitted_with_checkpoints(fake_cluster, tmp_path, monkeypatch, capsys):
    from tests.hpc.services.test_batches import submitted_batch

    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    squeue = tmp_path / "squeue-bin" / "squeue"
    squeue.write_text(f"#!{sys.executable}\n{SQUEUE_IDLE}")
    for config in sorted(batch_dir.glob("tasks/*/*/*/config.yaml"))[:2]:
        directory = config.parent
        (directory / "record.json").write_text(json.dumps({"exit_status": 75, "interrupted": True}))
        write_checkpoint(directory / "sessions" / "session_1" / "checkpoints" / "x", 4)
    return batch_dir, profiles


def test_resume_refuses_an_environment_that_predates_it(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_with_checkpoints(fake_cluster, tmp_path, monkeypatch, capsys)
    assert main(["resume", batch_dir.name, "--profiles", str(profiles), "--yes"]) == 2
    captured = capsys.readouterr()
    assert "predates resume" in captured.err
    # The two tasks with checkpoints are resumable; the other two (no record,
    # no checkpointer in their config) are selected by default and skipped.
    skipped = [line for line in captured.out.splitlines() if line.startswith("skip ")]
    assert len(skipped) == 2 and all("no checkpointer" in line for line in skipped)


def test_resume_submits_an_attempt_and_status_follows_it(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_with_checkpoints(fake_cluster, tmp_path, monkeypatch, capsys)
    monkeypatch.setenv("STUB_CAPABILITIES", '["resume"]')
    before = len(submitted_jobs(fake_cluster))
    assert main(["resume", batch_dir.name, "--profiles", str(profiles), "--walltime", "02:00:00",
                 "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert "as attempt 2: 2 task(s)" in dry and "#SBATCH --array=0,1" in dry
    assert len(submitted_jobs(fake_cluster)) == before        # a dry run submits nothing

    assert main(["resume", batch_dir.name, "--profiles", str(profiles), "--walltime", "02:00:00",
                 "--yes"]) == 0
    out = capsys.readouterr().out
    assert "resumed: attempt 2" in out
    job = submitted_jobs(fake_cluster)[-1]
    assert job["argv"][-1] == f"{batch_dir}/groups/probes/group-probes.resume-2.sbatch"
    receipt = json.loads((batch_dir / "receipt.json").read_text())
    assert receipt["attempts"][0]["status"] == "submitted"
    assert receipt["nodes"]["group-probes.resume-2"]["tasks"] == [0, 1]

    assert main(["batches", "--profiles", str(profiles), "--name", batch_dir.name]) == 0
    assert "submitted (attempt 2)" in capsys.readouterr().out
    # Resuming again while attempt 2 is queued is refused.
    squeue = tmp_path / "squeue-bin" / "squeue"
    squeue.write_text(f"#!{sys.executable}\nprint('{job['job']}_0|PENDING')\n")
    assert main(["resume", batch_dir.name, "--profiles", str(profiles), "--yes"]) == 2
    assert "still has jobs in the queue" in capsys.readouterr().err
