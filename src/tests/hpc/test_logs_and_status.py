"""Readable logs (task header and footer, wrapper lines, setup and prepare
steps, lease waits, submit progress, rsync progress) and `status`."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from training_framework.hpc.__main__ import main
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.local import LocalExecutor, LocalFilesystem, LocalSourceStager
from training_framework.hpc.ports import ExecResult
from training_framework.hpc.ports.ssh import RsyncTransfer
from training_framework.hpc.remote.lease import Lease
from training_framework.hpc.remote.prepare import run_prepare
from training_framework.hpc.remote.runtime import StdlibFilesystem
from training_framework.hpc.scheduler import get_scheduler
from training_framework.hpc.scheduler.slurm import _task_body
from training_framework.hpc.scheduler.slurm_submitter import job_states
from training_framework.hpc.services.status import batch_status, format_rows
from training_framework.hpc.services.submit import submit
from tests.hpc.fake_cluster import fake_cluster, submitted_jobs  # noqa: F401 (fixture)
from tests.hpc.fakes import FakeSubmitter, LocalScheduler
from tests.hpc.remote import test_bootstrap, test_prepare
from tests.hpc.services import test_submission
from tests.hpc.test_cli_submit import probed_identity, setup as cli_setup


# Task wrapper and task entry point -----------------------------------------------


def test_the_wrapper_logs_the_advance_signal_and_the_exit_status(tmp_path):
    batch_dir = tmp_path / "batch"
    (batch_dir / "groups" / "g").mkdir(parents=True)
    (batch_dir / "groups" / "g" / "tasks.tsv").write_text(
        "index\ttask\tvariant\tconfig\tconfig_hash\n0\tg/e/000\t[]\ttasks/g/e/000/config.yaml\th\n")
    child = ("import signal, sys, time\n"
             "signal.signal(signal.SIGINT, lambda *a: sys.exit(3))\n"
             "print('ready', flush=True)\n"
             "time.sleep(60)\n")
    script = tmp_path / "job.sh"
    script.write_text("\n".join(["#!/bin/bash", *_task_body(str(batch_dir), "g",
                                                            [sys.executable, "-c", child])]) + "\n")
    job = subprocess.Popen(["bash", str(script)], env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert job.stdout.readline().strip() == "ready"
    job.send_signal(signal.SIGUSR1)
    output = job.stdout.read()
    assert job.wait(timeout=30) == 75
    assert "] received the scheduler advance signal; stopping the task cooperatively" in output
    assert "] task process exited with status 3" in output
    assert output.splitlines()[0].startswith("[hpc 20")


def _end_to_end():
    pytest.importorskip("torch")
    from tests.hpc import test_end_to_end

    return test_end_to_end


def test_each_task_log_starts_with_a_header_and_ends_with_a_footer(tmp_path, tmp_path_factory):
    e2e = _end_to_end()
    environment = e2e.environment.__wrapped__(tmp_path_factory)
    root, compiled = e2e.batch(tmp_path, environment, e2e.session(), [5])
    scheduler = LocalScheduler()
    receipt = submit(compiled.plan, compiled.manifest.raw, stager=LocalSourceStager(root),
                     filesystem=LocalFilesystem(), executor=LocalExecutor(), submitter=scheduler,
                     receipts_dir=str(tmp_path / "receipts"))
    assert scheduler.results == {"group-cpu": [0]}, scheduler.outputs
    output = scheduler.outputs["group-cpu"][0]
    plan = compiled.plan
    assert "] task cpu/seeds/000 | session_config.rng_seed=5" in output
    assert f"] batch {plan.batch_id} | commit {plan.sha[:12]} | host " in output
    record = f"{plan.batch_dir}/tasks/cpu/seeds/000/record.json"
    assert f"] finished: exit 0 (ok) | record {record}" in output
    assert "] task process exited with status 0" in output

    # `status` reads the same records back.
    rows = batch_status(receipt.data, LocalFilesystem(), LocalExecutor(), lambda refs: {})
    (task,) = [row for row in rows if row.name == "cpu/seeds/000"]
    assert task.values == "session_config.rng_seed=5" and task.state == "done"
    assert task.detail.startswith(f"{plan.batch_dir}/tasks/cpu/seeds/000/sessions/session_")


# Setup, prepare, and the lease -----------------------------------------------------


def test_setup_logs_each_step(tmp_path):
    plan = test_bootstrap.make_cluster_batch(tmp_path, "dependency")
    result = test_bootstrap.run_bootstrap(plan)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("[hpc setup ")]
    text = "\n".join(lines)
    for step in ("batch ", "creating the venv environment", "installing the pinned requirements",
                 "running preflight", "preflight ok: Python", "environment ready"):
        assert step in text, text
    again = test_bootstrap.run_bootstrap(plan)
    assert "the environment is already ready; nothing to do" in again.stdout


def test_a_failed_setup_says_where_the_output_is(tmp_path):
    plan = test_bootstrap.make_cluster_batch(tmp_path, "dependency")
    for wheel in (tmp_path / "wheels").iterdir():
        wheel.unlink()
    result = test_bootstrap.run_bootstrap(plan)
    assert result.returncode == 1
    assert "FAILED: " in result.stdout and "its output is in this log" in result.stdout


def test_prepare_logs_each_step(tmp_path, capsys):
    plan = test_prepare.batch(tmp_path, [sys.executable, "-c", test_prepare.MAKE_DIR, "{path}"])
    assert run_prepare(f"{plan.batch_dir}/plan.json", "prepare-inat-0") == 0
    output = capsys.readouterr().out
    assert "[hpc prepare " in output
    assert "prepare-inat-0: data source 'inat' at " in output
    assert "running: " in output and "] prepared" in output
    assert run_prepare(f"{plan.batch_dir}/plan.json", "prepare-inat-0") == 0
    assert "already ready; nothing to do" in capsys.readouterr().out


class ScriptedProbe:
    def __init__(self, *answers):
        self.answers = list(answers)

    def is_alive(self, owner):
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_the_lease_says_once_whom_it_waits_for_and_when_it_takes_over(tmp_path):
    fs = StdlibFilesystem()
    location = str(tmp_path / "env")
    Lease(fs, ScriptedProbe(True), location, {"job_ref": "1"}).acquire()
    messages = []
    Lease(fs, ScriptedProbe(True, True, True, False), location, {"job_ref": "2"},
          poll=0, sleep=lambda seconds: None, notify=messages.append).acquire()
    assert messages == [
        f'waiting for the lease on {location} held by {{"job_ref": "1"}}',
        'taking over the lease from {"job_ref": "1"}, which is no longer running',
    ]


# Submit progress and rsync progress -------------------------------------------------


def test_submit_reports_its_progress(tmp_path):
    compiled = test_submission.compiled.__wrapped__(tmp_path)
    messages = []
    receipt = test_submission.run(compiled, tmp_path, FakeSubmitter(), progress=messages.append)
    assert receipt.status == "submitted"
    plan = compiled.plan
    assert f"staging commit {plan.sha[:12]} into {plan.batch_dir}" in messages
    assert any(message.startswith("preflight") for message in messages)
    submitted = [message for message in messages if message.startswith("submitted ")]
    assert submitted[0].startswith("submitted group-probes as job 1000")
    assert submitted[1].startswith("submitted group-second as job 1001")


FAKE_RSYNC = r'''
import os, sys
open(os.environ["RSYNC_ARGV"], "w").write("\n".join(sys.argv[1:]))
if os.environ.get("RSYNC_FAIL"):
    sys.stderr.write("rsync: connection unexpectedly closed\n"); sys.exit(12)
sys.stdout.write("      1,024 100%    1.00MB/s    0:00:00\r")
print("\nTotal transferred file size: 1,024 bytes")
'''


@pytest.fixture
def fake_rsync(tmp_path, monkeypatch):
    path = tmp_path / "rsync"
    path.write_text(f"#!{sys.executable}\n{FAKE_RSYNC}")
    path.chmod(0o755)
    monkeypatch.setenv("RSYNC_ARGV", str(tmp_path / "argv"))
    return path


def test_rsync_shows_live_progress_and_still_counts_bytes(tmp_path, fake_rsync, capfd):
    source = tmp_path / "data"
    source.mkdir()
    transfer = RsyncTransfer("cluster", rsync=str(fake_rsync), show_progress=True)
    assert transfer.sync(str(source), "/remote/data") == 1024
    assert "--info=progress2" in (tmp_path / "argv").read_text().splitlines()
    assert "100%" in capfd.readouterr().err


def test_a_failed_rsync_with_progress_reports_its_error(tmp_path, fake_rsync, monkeypatch):
    monkeypatch.setenv("RSYNC_FAIL", "1")
    transfer = RsyncTransfer("cluster", rsync=str(fake_rsync), show_progress=True)
    with pytest.raises(HpcError, match="connection unexpectedly closed"):
        transfer.sync(str(tmp_path), "/remote/data")


def test_rsync_without_a_terminal_shows_no_progress(tmp_path, fake_rsync):
    transfer = RsyncTransfer("cluster", rsync=str(fake_rsync))
    assert transfer.sync(str(tmp_path), "/remote/data") == 1024
    assert "--info=progress2" not in (tmp_path / "argv").read_text().splitlines()


# Status ------------------------------------------------------------------------------


class SqueueExecutor:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def run(self, argv, context=None, timeout=None):
        self.calls.append(argv)
        job = argv[argv.index("--jobs") + 1]
        if job not in self.answers:
            return ExecResult(1, "", "slurm_load_jobs error: Invalid job id specified\n")
        return ExecResult(0, self.answers[job], "")


def test_slurm_job_states_cover_array_elements_and_finished_jobs():
    executor = SqueueExecutor({"7": "7_0|RUNNING\n7_[1-3%2]|PENDING\n"})
    states = job_states(executor)(["7", "8"])
    assert states == {"7_0": "running", "7_[1-3%2]": "pending"}
    assert executor.calls[0] == ["squeue", "--noheader", "--array", "--jobs", "7",
                                 "--format", "%i|%T"]
    assert get_scheduler("slurm").job_states is job_states


def test_a_failing_squeue_is_an_error():
    class Broken:
        def run(self, argv, context=None, timeout=None):
            return ExecResult(1, "", "squeue: error: slurm_receive_msg: timeout\n")
    with pytest.raises(RuntimeError, match="timeout"):
        job_states(Broken())(["7"])


def _batch(tmp_path, node_status="submitted"):
    batch_dir = tmp_path / "batch"
    rows = ["index\ttask\tvariant\tconfig\tconfig_hash"]
    variant = '[{"address":{"path":["optimizer","lr"]},"value":0.1}]'
    for index in range(4):
        rows.append(f"{index}\tg/e/00{index}\t{variant}\ttasks/g/e/00{index}/config.yaml\th")
    (batch_dir / "groups" / "g").mkdir(parents=True)
    (batch_dir / "groups" / "g" / "tasks.tsv").write_text("\n".join(rows) + "\n")
    records = {
        0: {"exit_status": 0, "interrupted": False, "run_directories": ["/runs/a"]},
        1: {"exit_status": 1, "interrupted": False, "error": "Traceback\nValueError: bad lr\n"},
        2: {"exit_status": 75, "interrupted": True},
    }
    for index, record in records.items():
        directory = batch_dir / "tasks" / "g" / "e" / f"00{index}"
        directory.mkdir(parents=True)
        (directory / "record.json").write_text(json.dumps(record, indent=2))
    (batch_dir / "nodes" / "setup").mkdir(parents=True)
    (batch_dir / "nodes" / "setup" / "record.json").write_text(
        json.dumps({"status": "built", "location": "/envs/x"}))
    return {"batch_id": "b", "batch_dir": str(batch_dir), "status": "submitted", "nodes": {
        "setup": {"status": "ran on login", "job_ref": None},
        "group-g": {"status": node_status, "job_ref": "9"},
    }}


def test_status_combines_records_with_the_live_queue(tmp_path):
    receipt = _batch(tmp_path)
    rows = batch_status(receipt, LocalFilesystem(), LocalExecutor(), lambda refs: {})
    table = {row.name: row for row in rows}
    assert (table["setup"].state, table["setup"].detail) == ("built", "/envs/x")
    assert (table["g/e/000"].state, table["g/e/000"].detail) == ("done", "/runs/a")
    assert table["g/e/000"].values == "optimizer.lr=0.1"
    assert (table["g/e/001"].state, table["g/e/001"].detail) == ("failed (exit 1)", "ValueError: bad lr")
    assert table["g/e/002"].state == "interrupted"
    assert table["g/e/003"].state == "ended without a record"
    assert table["g/e/003"].detail == "see logs/g/9_3.out"

    running = batch_status(receipt, LocalFilesystem(), LocalExecutor(),
                           lambda refs: {"9_3": "running"})
    assert {row.name: row.state for row in running}["g/e/003"] == "running"
    pending = batch_status(receipt, LocalFilesystem(), LocalExecutor(),
                           lambda refs: {"9_[3]": "pending", "9": "pending"})
    assert {row.name: row.state for row in pending}["g/e/003"] == "pending"


def test_status_of_a_cancelled_group_says_so(tmp_path):
    receipt = _batch(tmp_path, node_status="cancelled")
    rows = batch_status(receipt, LocalFilesystem(), LocalExecutor(), lambda refs: {})
    assert {row.name: row.state for row in rows}["g/e/003"] == "cancelled"
    text = format_rows(rows)
    assert text.splitlines()[0].split() == ["task", "values", "state", "detail"]
    assert "g/e/001  optimizer.lr=0.1  failed (exit 1)  ValueError: bad lr" in text


SQUEUE = r'''
import os, sys
job = sys.argv[sys.argv.index("--jobs") + 1]
print("%s_0|RUNNING" % job)
print("%s_[1-3]|PENDING" % job)
'''


def test_the_status_command_after_a_submit(fake_cluster, tmp_path, monkeypatch, capsys):
    root, path = cli_setup(tmp_path, monkeypatch)
    identity = probed_identity(path, capsys)
    root, path = cli_setup(tmp_path / "second", monkeypatch, identity)
    assert main(["submit", str(root / "batch.yaml"), "--profiles", str(path), "--yes"]) == 0
    output = capsys.readouterr().out
    (job,) = submitted_jobs(fake_cluster)
    batch_id = Path(output.split("submitted: ")[1].splitlines()[0]).name

    bin_dir = tmp_path / "squeue-bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text(f"#!{sys.executable}\n{SQUEUE}")
    (bin_dir / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert main(["status", batch_id, "--profiles", str(path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"batch {batch_id}: submitted | ")
    states = {line.split()[0]: line for line in lines[2:]}
    assert "built" in states["setup"]
    tasks = [line for name, line in states.items() if name.startswith("probes/")]
    assert tasks and "running" in tasks[0] and f"job {job['job']}_0" in tasks[0]


def test_status_of_an_unknown_batch_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    path = tmp_path / "hpc.yaml"
    from tests.hpc.batch_repo import PROFILE
    path.write_text(yaml.safe_dump(PROFILE))
    assert main(["status", "nope", "--profiles", str(path)]) == 2
    assert "no receipt for nope" in capsys.readouterr().err


SLOW_RSYNC = r'''
import sys
for done, percent, left in ((1_000_000, 1, "1:00:00"), (2_000_000, 2, "0:59:00"),
                            (50_000_000, 50, "0:30:00"), (100_000_000, 100, "0:00:00")):
    sys.stdout.write("%15s %3d%%   10.00MB/s    %s (xfr#1, ir-chk=1/2)\r" % (f"{done:,}", percent, left))
    sys.stdout.flush()
print("\nTotal transferred file size: 100,000,000 bytes")
'''


def test_without_a_terminal_an_upload_reports_progress_lines(tmp_path, fake_rsync):
    fake_rsync.write_text(f"#!{sys.executable}\n{SLOW_RSYNC}")
    times = iter([0, 10, 40, 45, 100])          # one clock reading per output chunk
    messages = []
    transfer = RsyncTransfer("cluster", rsync=str(fake_rsync), report=messages.append,
                             report_interval=30, clock=lambda: next(times, 100))
    assert transfer.sync(str(tmp_path), "/remote/data") == 100_000_000
    assert messages[0].startswith("upload to cluster:/remote/data: ")
    assert "% of the files listed so far), 10.00MB/s, about " in messages[0]
    assert messages[-1] == "upload to cluster:/remote/data finished: 95.4 MiB in place"
    assert len(messages) <= 4


RESUMED_RSYNC = r'''
import sys
sys.stdout.write("    640,000,000 100%   98.86MB/s    0:00:06 (xfr#32, to-chk=0/41)\r")
print("\nNumber of files: 41 (reg: 40, dir: 1)")
print("Number of regular files transferred: 32")
print("Total transferred file size: 640,000,000 bytes")
'''


def test_the_final_upload_line_says_how_much_was_already_there(tmp_path, fake_rsync):
    fake_rsync.write_text(f"#!{sys.executable}\n{RESUMED_RSYNC}")
    messages = []
    RsyncTransfer("cluster", rsync=str(fake_rsync), report=messages.append).sync(str(tmp_path), "/r")
    assert messages[-1] == ("upload to cluster:/r finished: 610.4 MiB in place; "
                            "32 of 40 files sent now, 8 were already there")


def test_a_dataset_named_by_two_sources_is_hashed_once(tmp_path, monkeypatch):
    from training_framework.hpc import __main__ as cli

    data = tmp_path / "data"
    data.mkdir()
    (data / "a.bin").write_bytes(b"a")
    calls = []
    real = cli.content_summary
    monkeypatch.setattr(cli, "content_summary", lambda *args: calls.append(args[0]) or real(*args))
    facts = cli.WorkstationFacts(tmp_path, cache_path=tmp_path / "cache.sqlite3")
    assert facts.content(str(data)) == facts.content(str(data))
    assert calls == [str(data)]
