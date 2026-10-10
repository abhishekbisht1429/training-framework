"""Contract suites: every implementation of a port runs the same tests.
Later adapters (SSH, rsync, Slurm) are added to these parameter lists."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from training_framework.hpc.local import (
    LocalExecutor,
    LocalFilesystem,
    LocalFileTransfer,
    LocalProcessProbe,
)
from training_framework.hpc.plan import ExecutionContext, JobNode
from training_framework.hpc.ports import Owner, SubmissionFailed, SubmissionUncertain
from tests.hpc.fakes import FakeJobProbe, FakeSubmitter

EXECUTORS = [LocalExecutor]
FILESYSTEMS = [LocalFilesystem]
TRANSFERS = [LocalFileTransfer]
SUBMITTERS = [FakeSubmitter]


# RemoteExecutor -------------------------------------------------------------


@pytest.fixture(params=EXECUTORS)
def executor(request):
    return request.param()


def context(tmp_path, **changes):
    values = dict(
        phase="task", interpreter="python3", environment=None,
        working_directory=str(tmp_path), pythonpath=(str(tmp_path / "src"),),
        set_variables=(("PYTHONNOUSERSITE", "1"),),
        unset_variables=("WORLD_SIZE",), hooks=("export HOOKED=yes",),
    )
    values.update(changes)
    return ExecutionContext(**values)


def test_arguments_are_never_interpreted_by_a_shell(executor, tmp_path):
    tricky = "$(echo pwned); `id` && 'quoted' \"double\" *"
    result = executor.run(["printf", "%s", tricky], context(tmp_path))
    assert result.status == 0 and result.stdout == tricky


def test_the_context_is_established(executor, tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    script = "import os; print(os.getcwd(), os.environ.get('PYTHONPATH'), os.environ.get('PYTHONNOUSERSITE'), os.environ.get('WORLD_SIZE'), os.environ.get('HOOKED'))"
    result = executor.run(["python3", "-c", script], context(tmp_path))
    assert result.stdout.split() == [str(tmp_path), str(tmp_path / "src"), "1", "None", "yes"]


def test_exit_status_and_stderr_are_reported(executor, tmp_path):
    result = executor.run(["sh", "-c", "echo oops >&2; exit 3"], context(tmp_path))
    assert result.status == 3 and "oops" in result.stderr


def test_a_failing_hook_stops_the_command(executor, tmp_path):
    result = executor.run(["echo", "ran"], context(tmp_path, hooks=("false",)))
    assert result.status != 0 and "ran" not in result.stdout


# RemoteFilesystem -------------------------------------------------------------


@pytest.fixture(params=FILESYSTEMS)
def filesystem(request):
    return request.param()


def test_mkdir_is_atomic_and_exclusive(filesystem, tmp_path):
    path = str(tmp_path / "lock")
    assert filesystem.mkdir_atomic(path) is True
    assert filesystem.mkdir_atomic(path) is False


def test_write_is_atomic_and_leaves_no_temporary_files(filesystem, tmp_path):
    path = str(tmp_path / "dir" / "file.json")
    filesystem.write(path, b"one")
    filesystem.write(path, b"two")
    assert filesystem.read(path) == b"two"
    assert os.listdir(tmp_path / "dir") == ["file.json"]


def test_read_stat_exists_of_missing_paths(filesystem, tmp_path):
    missing = str(tmp_path / "missing")
    assert filesystem.read(missing) is None
    assert filesystem.stat(missing) is None
    assert not filesystem.exists(missing)


def test_rename_replaces_and_delete_removes_trees(filesystem, tmp_path):
    filesystem.write(str(tmp_path / "a"), b"a")
    filesystem.write(str(tmp_path / "b"), b"b")
    filesystem.rename(str(tmp_path / "a"), str(tmp_path / "b"))
    assert filesystem.read(str(tmp_path / "b")) == b"a"
    filesystem.write(str(tmp_path / "tree" / "x" / "y"), b"y")
    filesystem.delete(str(tmp_path / "tree"))
    filesystem.delete(str(tmp_path / "tree"))   # missing is fine
    assert not filesystem.exists(str(tmp_path / "tree"))
    assert filesystem.stat(str(tmp_path / "b")).size == 1


# FileTransfer -------------------------------------------------------------------


@pytest.fixture(params=TRANSFERS)
def transfer(request):
    return request.param()


def test_sync_copies_one_way_and_resumes(transfer, tmp_path):
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "a").write_bytes(b"12345")
    (source / "sub" / "b").write_bytes(b"678")
    target = tmp_path / "target"
    assert transfer.sync(str(source), str(target)) == 8
    assert (target / "sub" / "b").read_bytes() == b"678"
    assert transfer.sync(str(source), str(target)) == 0     # nothing left to send
    (target / "extra").write_bytes(b"x")
    transfer.sync(str(source), str(target))
    assert not (source / "extra").exists()                 # one-way


def test_sync_of_a_single_file(transfer, tmp_path):
    (tmp_path / "model.pt").write_bytes(b"w")
    assert transfer.sync(str(tmp_path / "model.pt"), str(tmp_path / "out" / "model.pt")) == 1


# SchedulerSubmitter ------------------------------------------------------------


@pytest.fixture(params=SUBMITTERS)
def submitter_class(request):
    return request.param


def node(node_id):
    return JobNode(id=node_id, kind="setup", location="job", depends_on=(),
                   context=context(Path(os.getcwd())))


def test_submissions_return_distinct_references(submitter_class):
    submitter = submitter_class()
    first = submitter.submit(node("a"), "/b/a.sbatch", [])
    second = submitter.submit(node("b"), "/b/b.sbatch", [first])
    assert first != second


def test_definite_and_uncertain_failures_are_distinct(submitter_class):
    with pytest.raises(SubmissionFailed):
        submitter_class(fail_at=0, failure="definite").submit(node("a"), "/s", [])
    with pytest.raises(SubmissionUncertain):
        submitter_class(fail_at=0, failure="uncertain").submit(node("a"), "/s", [])
    assert not issubclass(SubmissionUncertain, SubmissionFailed)


def test_cancel_reports_its_result(submitter_class):
    submitter = submitter_class(cancel_fails={"1001"})
    assert submitter.cancel("1000").ok
    assert not submitter.cancel("1001").ok


# LeaseOwnerProbe -----------------------------------------------------------------


def test_the_process_probe_follows_a_process_and_its_start_time(tmp_path):
    probe = LocalProcessProbe()
    me = probe.current_owner()
    assert probe.is_alive(me) is True
    assert probe.is_alive(Owner(host=me.host, pid=me.pid, start_time="0")) is False
    child = subprocess.Popen(["true"])
    child.wait()
    assert probe.is_alive(Owner(host=me.host, pid=child.pid, start_time=me.start_time)) is False


def test_probes_answer_none_for_owners_they_cannot_judge():
    process_probe = LocalProcessProbe(host="here")
    assert process_probe.is_alive(Owner(host="elsewhere", pid=1, start_time="1")) is None
    assert process_probe.is_alive(Owner(job_ref="42")) is None
    job_probe = FakeJobProbe(running={"42"})
    assert job_probe.is_alive(Owner(job_ref="42")) is True
    assert job_probe.is_alive(Owner(job_ref="43")) is False
    assert job_probe.is_alive(Owner(host="here", pid=1)) is None
