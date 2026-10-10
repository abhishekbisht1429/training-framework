import ast
import inspect
import os
import subprocess
import sys

import pytest

from training_framework.hpc.remote import lease as lease_module
from training_framework.hpc.remote import runtime
from training_framework.hpc.remote.lease import (
    KeyMismatch,
    Lease,
    LeaseError,
    check_marker,
    read_marker,
    write_marker,
)
from training_framework.hpc.remote.runtime import (
    OwnerProbeRouter,
    ProcessProbe,
    SlurmJobProbe,
    StdlibFilesystem,
    current_process_owner,
)

FS = StdlibFilesystem()


class ScriptedProbe:
    """Reports the given answers in turn, then the last one forever."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def is_alive(self, owner):
        self.asked.append(owner)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_markers_are_written_atomically_and_never_reinterpreted(tmp_path):
    location = str(tmp_path / "env")
    assert read_marker(FS, location) is None
    assert check_marker(FS, location, "k1") is False
    write_marker(FS, location, "k1")
    assert check_marker(FS, location, "k1") is True
    with pytest.raises(KeyMismatch, match="ready with key k1 but this batch needs key k2"):
        check_marker(FS, location, "k2")
    assert sorted(os.listdir(tmp_path)) == ["env.tf-hpc-ready"]


def test_an_acquired_lease_records_its_owner_and_is_released(tmp_path):
    location = str(tmp_path / "env")
    lease = Lease(FS, ScriptedProbe(True), location, {"job_ref": "7"})
    lease.acquire()
    assert (tmp_path / "env.tf-hpc-lock" / "owner.json").read_text() == '{"job_ref": "7"}'
    lease.release()
    assert not (tmp_path / "env.tf-hpc-lock").exists()


def test_a_live_owner_is_waited_for_and_a_dead_one_taken_over(tmp_path):
    location = str(tmp_path / "env")
    Lease(FS, ScriptedProbe(True), location, {"job_ref": "1"}).acquire()
    sleeps = []
    probe = ScriptedProbe(True, True, False)
    second = Lease(FS, probe, location, {"job_ref": "2"}, poll=0.5, sleep=sleeps.append)
    second.acquire()
    assert sleeps == [0.5, 0.5]
    assert second.took_over and second.held
    assert probe.asked[0] == {"job_ref": "1"}


def test_an_owner_that_cannot_be_checked_is_never_taken_over(tmp_path):
    location = str(tmp_path / "env")
    Lease(FS, ScriptedProbe(True), location,
          {"host": "login2", "pid": 9, "start_time": "1"}).acquire()
    with pytest.raises(LeaseError, match="remove .*env.tf-hpc-lock by hand"):
        Lease(FS, ScriptedProbe(None), location, {"job_ref": "2"}).acquire()
    assert (tmp_path / "env.tf-hpc-lock").exists()


def test_job_and_process_owners_go_through_the_same_router(tmp_path):
    calls = []

    def fake_squeue(argv):
        calls.append(argv)
        running = argv[argv.index("--jobs") + 1] == "11"
        return subprocess.CompletedProcess(argv, 0, "RUNNING\n" if running else "", "")

    router = OwnerProbeRouter(SlurmJobProbe(fake_squeue), ProcessProbe())
    me = current_process_owner()
    assert router.is_alive({"job_ref": "11"}) is True
    assert router.is_alive({"job_ref": "12"}) is False
    assert router.is_alive(me) is True
    assert router.is_alive(dict(me, start_time="0")) is False
    assert router.is_alive(dict(me, host="elsewhere")) is None


def test_squeue_reporting_an_unknown_job_means_dead():
    probe = SlurmJobProbe(lambda argv: subprocess.CompletedProcess(
        argv, 1, "", "slurm_load_jobs error: Invalid job id specified"))
    assert probe.is_alive({"job_ref": "5"}) is False


@pytest.mark.parametrize("module", [lease_module, runtime])
def test_bundled_modules_use_python_3_8_syntax_and_the_standard_library(module):
    tree = ast.parse(inspect.getsource(module), feature_version=(3, 8))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names)
