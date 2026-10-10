import os
import socket

import pytest

from training_framework.hpc.compile.plan import ExecutionContext, JobNode
from training_framework.hpc.ports import Owner, SubmissionUncertain
from training_framework.hpc.remote.runtime import current_process_owner
from training_framework.hpc.ports.ssh import SshExecutor, TransportError
from training_framework.hpc.scheduler.slurm_submitter import RemoteOwnerProbe, SlurmSubmitter
from tests.hpc.fake_cluster import fake_cluster, submitted_jobs  # noqa: F401 (fixture)

NODE = JobNode(id="group-g", kind="task_set", location="job", depends_on=(),
               context=ExecutionContext("task", "python", None, "/", (), (), (), ()))


def test_dependencies_and_cancellation_of_dependants_reach_sbatch(fake_cluster):
    submitter = SlurmSubmitter(SshExecutor("cluster"))
    first = submitter.submit(NODE, "/b/setup.sbatch", [])
    second = submitter.submit(NODE, "/b/g.sbatch", [first, "77"])
    jobs = submitted_jobs(fake_cluster)
    assert jobs[0]["argv"] == ["--parsable", "--kill-on-invalid-dep=yes", "/b/setup.sbatch"]
    assert jobs[1]["argv"][-2:] == [f"--dependency=afterok:{first}:77", "/b/g.sbatch"]
    assert second == jobs[1]["job"]


def test_unparseable_sbatch_output_is_uncertain(fake_cluster, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH", "garbage")
    with pytest.raises(SubmissionUncertain, match="unparseable"):
        SlurmSubmitter(SshExecutor("cluster")).submit(NODE, "/s", [])


def test_a_dropped_connection_is_a_transport_error(fake_cluster, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DOWN", "1")
    with pytest.raises(TransportError, match="Connection refused"):
        SshExecutor("cluster").run(["true"])


def test_remote_probes_check_jobs_anywhere_and_processes_only_on_their_host(fake_cluster):
    executor = SshExecutor("cluster")
    job = SlurmSubmitter(executor).submit(NODE, "/s", [])
    probe = RemoteOwnerProbe(executor)
    assert probe.is_alive(Owner(job_ref=job)) is True
    SlurmSubmitter(executor).cancel(job)
    assert probe.is_alive(Owner(job_ref=job)) is False
    assert probe.is_alive(Owner(job_ref="1")) is False          # unknown job id
    me = Owner(**current_process_owner())
    assert probe.is_alive(me) is True                           # fake ssh lands here
    assert probe.is_alive(Owner(host=me.host, pid=me.pid, start_time="0")) is False
    assert probe.is_alive(Owner(host="other-login", pid=os.getpid(), start_time=me.start_time)) is None
    assert me.host == socket.gethostname()
