"""`fetch`: selection, the file list, verification, the index, transfers,
a real fetched checkpoint that loads, and the CLI through the fake cluster."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from training_framework.hpc.__main__ import main
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import FetchSummary
from training_framework.hpc.ports.local import LocalExecutor, LocalFileTransfer, LocalFilesystem, LocalSourceStager
from training_framework.hpc.ports.ssh import RsyncTransfer
from training_framework.hpc.services import fetch as fetching
from training_framework.hpc.services.resume import task_index
from training_framework.hpc.services.status import Row, fetch_batches, task_rows
from training_framework.hpc.services.submit import submit
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)
from tests.hpc.fakes import LocalScheduler
from tests.hpc.test_logs_and_status import _batch


def write_checkpoint(directory: Path, iteration: int, payload: bytes = b"weights") -> None:
    (directory / "components").mkdir(parents=True)
    (directory / "components" / "model.pt").write_bytes(payload)
    (directory / "session.pt").write_bytes(b"session")
    (directory / "manifest.json").write_text(json.dumps({
        "iteration": iteration,
        "components": {"model": {"file": "components/model.pt",
                                 "sha256": hashlib.sha256(payload).hexdigest()}},
    }, indent=2))


def batch_with_results(tmp_path) -> dict:
    """`_batch` (0 done, 1 failed, 2 interrupted, 3 no record) with run files,
    checkpoints, and logs."""
    batch = _batch(tmp_path)
    root = Path(batch["batch_dir"])
    for index in range(4):
        task = root / "tasks" / "g" / "e" / f"00{index}"
        session = task / "sessions" / "session_1"
        (session / "events").mkdir(parents=True)
        (session / "events" / "tfevents").write_text("events")
        (task / "config.yaml").write_text("sessions: []\n")
        write_checkpoint(session / "checkpoints" / "a", 100)
        write_checkpoint(session / "checkpoints" / "b", 200)
        (session / "checkpoints" / "c.tmp").mkdir()
        (root / "logs" / "g").mkdir(parents=True, exist_ok=True)
        (root / "logs" / "g" / f"9_{index}.out").write_text(f"log {index}")
    (root / "plan.json").write_text("{}")
    (root / "receipt.json").write_text("{}")
    return batch


def selection(batch, live=None, names=None, state="done"):
    files = fetch_batches(LocalExecutor(), [batch["batch_dir"]])[batch["batch_dir"]]
    rows = task_rows(batch, files, live or {})
    return fetching.select(batch, rows, task_index(files.tables), names, state)


# Selection -----------------------------------------------------------------------------


def test_by_default_only_successful_tasks_are_fetched(tmp_path):
    batch = batch_with_results(tmp_path)
    chosen, skipped = selection(batch)
    assert [t.task for t in chosen] == ["g/e/000"]
    assert {name for name, _ in skipped} == {"g/e/001", "g/e/002", "g/e/003"}
    chosen, _ = selection(batch, state="finished")
    assert [t.task for t in chosen] == ["g/e/000", "g/e/001", "g/e/002", "g/e/003"]
    chosen, _ = selection(batch, names=["g/e/001"])                    # named: fetched whatever its state
    assert [t.task for t in chosen] == ["g/e/001"]
    with pytest.raises(HpcError, match="not tasks of this batch: nope"):
        selection(batch, names=["nope"])


def test_tasks_still_running_are_skipped_unless_asked(tmp_path):
    batch = batch_with_results(tmp_path)
    _, skipped = selection(batch, live={"9_3": "running"}, state="finished")
    assert dict(skipped)["g/e/003"].startswith("still running")
    chosen, _ = selection(batch, live={"9_3": "running"}, state="all")
    assert "g/e/003" in [t.task for t in chosen]


def test_logs_of_every_attempt():
    receipt = {"nodes": {"group-g": {"status": "submitted", "job_ref": "9"},
                         "group-g.resume-2": {"status": "submitted", "job_ref": "12", "attempt": 2,
                                              "group": "g", "tasks": [2]},
                         "group-h": {"status": "submitted", "job_ref": "10"}}}
    assert fetching.logs_for(receipt, "g", 2) == ["logs/g/9_2.out", "logs/g/12_2.out"]
    assert fetching.logs_for(receipt, "g", 1) == ["logs/g/9_1.out"]


# The file list --------------------------------------------------------------------------


def listing(batch, mode="latest", includes=None, names=("g/e/000",)):
    tasks = [fetching.FetchTask(name, "g", int(name[-1]), "done") for name in names]
    files = fetching.plan_files(LocalExecutor(), batch["batch_dir"], tasks, mode, includes)
    fetching.add_logs(LocalExecutor(), batch["batch_dir"], batch, tasks, files)
    return tasks, files


def test_the_list_has_run_files_the_latest_checkpoint_and_logs(tmp_path):
    batch = batch_with_results(tmp_path)
    tasks, files = listing(batch)
    root = "tasks/g/e/000"
    assert set(files) == {
        "plan.json", "receipt.json", "groups/g/tasks.tsv", "logs/g/9_0.out",
        f"{root}/record.json", f"{root}/config.yaml", f"{root}/sessions/session_1/events/tfevents",
        f"{root}/sessions/session_1/checkpoints/b/manifest.json",
        f"{root}/sessions/session_1/checkpoints/b/session.pt",
        f"{root}/sessions/session_1/checkpoints/b/components/model.pt",
    }
    assert tasks[0].checkpoints == [(f"{root}/sessions/session_1/checkpoints/b", 200)]
    assert files["logs/g/9_0.out"] == len("log 0")


def test_all_or_no_checkpoints_and_extra_files(tmp_path):
    batch = batch_with_results(tmp_path)
    tasks, files = listing(batch, "all")
    assert [it for _, it in tasks[0].checkpoints] == [100, 200]
    assert any("/checkpoints/a/" in path for path in files)
    tasks, files = listing(batch, "none", includes=["nodes/*/record.json"])
    assert tasks[0].checkpoints == [] and not any("/checkpoints/" in path for path in files)
    assert "nodes/setup/record.json" in files


def test_files_already_here_are_counted(tmp_path):
    files = {"a": 3, "b": 5}
    (tmp_path / "a").write_text("abc")
    (tmp_path / "b").write_text("xy")                                  # different size
    assert fetching.already_here(files, str(tmp_path)) == (1, 3)


# Verification and the index --------------------------------------------------------------


def test_checkpoints_are_verified_against_their_manifest(tmp_path):
    good = tmp_path / "good"
    write_checkpoint(good, 5)
    assert fetching.verify_checkpoint(str(good)) == []
    (good / "components" / "model.pt").write_bytes(b"tampered")
    assert "does not match its SHA-256" in fetching.verify_checkpoint(str(good))[0]
    (good / "components" / "model.pt").unlink()
    (good / "session.pt").unlink()
    problems = fetching.verify_checkpoint(str(good))
    assert any("session.pt is missing" in p for p in problems)
    assert any("model.pt is missing" in p for p in problems)
    assert "unreadable" in fetching.verify_checkpoint(str(tmp_path / "nothing"))[0]


def test_the_index_keeps_earlier_fetches(tmp_path):
    receipt = {"batch_id": "b", "batch_dir": "/cluster/b"}
    first = fetching.FetchTask("g/e/000", "g", 0, "done", [("tasks/x/ck", 6)], ["logs/g/9_0.out"])
    fetching.write_index(str(tmp_path), receipt, "host", [first], {"tasks/x/ck": []},
                         FetchSummary(3, 3, 30))
    second = fetching.FetchTask("g/e/001", "g", 1, "failed (exit 1)")
    path = fetching.write_index(str(tmp_path), receipt, "host", [second], {}, FetchSummary(1, 0, 0))
    index = json.loads(Path(path).read_text())
    assert set(index["tasks"]) == {"g/e/000", "g/e/001"}
    assert index["tasks"]["g/e/000"]["checkpoints"][0] == {
        "path": "tasks/x/ck", "iteration": 6, "verified": True, "problems": []}
    assert index["cluster"] == "host" and index["last_transfer"] == {"files": 1, "fetched": 0, "bytes": 0}


# Transfers -------------------------------------------------------------------------------


def test_a_local_fetch_copies_only_what_is_new(tmp_path):
    remote = tmp_path / "remote"
    (remote / "d").mkdir(parents=True)
    (remote / "d" / "x").write_text("12345")
    (remote / "y").write_text("ab")
    transfer = LocalFileTransfer()
    assert transfer.fetch(str(remote), ["d/x", "y"], str(tmp_path / "local")) == FetchSummary(2, 2, 7)
    assert (tmp_path / "local" / "d" / "x").read_text() == "12345"
    assert transfer.fetch(str(remote), ["d/x", "y"], str(tmp_path / "local")) == FetchSummary(2, 0, 0)


FAKE_RSYNC = r'''
import os, shutil, sys
args = sys.argv[1:]
open(os.environ["RSYNC_ARGV"], "w").write("\n".join(args))
listing = next(a for a in args if a.startswith("--files-from=")).split("=", 1)[1]
open(os.environ["RSYNC_LIST"], "w").write(open(listing).read())
source, target = args[-2].split(":", 1)[1], args[-1]
sent = size = 0
for path in open(listing).read().split():
    shutil.copy2(os.path.join(source, path), os.path.join(target, path)) if os.makedirs(
        os.path.dirname(os.path.join(target, path)), exist_ok=True) is None else None
    sent += 1
    size += os.path.getsize(os.path.join(source, path))
print("Number of files: %d (reg: %d)" % (sent, sent))
print("Number of regular files transferred: %d" % sent)
print("Total transferred file size: {:,} bytes".format(size))
'''


def test_rsync_fetch_uses_one_files_from_list(tmp_path, monkeypatch):
    rsync = tmp_path / "rsync"
    rsync.write_text(f"#!{sys.executable}\n{FAKE_RSYNC}")
    rsync.chmod(0o755)
    monkeypatch.setenv("RSYNC_ARGV", str(tmp_path / "argv"))
    monkeypatch.setenv("RSYNC_LIST", str(tmp_path / "list"))
    remote = tmp_path / "remote"
    (remote / "a").mkdir(parents=True)
    (remote / "a" / "f").write_text("1234")
    summary = RsyncTransfer("cluster", rsync=str(rsync)).fetch(str(remote), ["a/f"], str(tmp_path / "here"))
    assert summary == FetchSummary(1, 1, 4)
    argv = (tmp_path / "argv").read_text().splitlines()
    assert "--partial" in argv and argv[-2] == f"cluster:{remote}/" and argv[-1] == f"{tmp_path}/here/"
    assert (tmp_path / "list").read_text() == "a/f\n"
    assert (tmp_path / "here" / "a" / "f").read_text() == "1234"
    assert RsyncTransfer("cluster", rsync=str(rsync)).fetch(str(remote), [], str(tmp_path)) == \
        FetchSummary(0, 0, 0)


# A real fetched checkpoint loads ---------------------------------------------------------


def test_a_fetched_checkpoint_loads(tmp_path, tmp_path_factory, monkeypatch):
    pytest.importorskip("torch")
    from training_framework.components.builtin.checkpointing import Checkpointer
    from tests.hpc import test_end_to_end as e2e

    environment = e2e.environment.__wrapped__(tmp_path_factory)
    session = e2e.session(max_iterations=6)
    session["checkpointer"] = {"checkpoint_every": 3}
    root, compiled = e2e.batch(tmp_path, environment, session, [5])
    receipt = submit(compiled.plan, compiled.manifest.raw, stager=LocalSourceStager(root),
                     filesystem=LocalFilesystem(), executor=LocalExecutor(), submitter=LocalScheduler(),
                     receipts_dir=str(tmp_path / "receipts"))
    batch_dir = compiled.plan.batch_dir
    executor = LocalExecutor()
    files = fetch_batches(executor, [batch_dir])[batch_dir]
    tasks, _ = fetching.select(receipt.data, task_rows(receipt.data, files, {}), task_index(files.tables))
    listing_ = fetching.plan_files(executor, batch_dir, tasks)
    fetching.add_logs(executor, batch_dir, receipt.data, tasks, listing_)
    local = tmp_path / "results" / compiled.plan.batch_id
    LocalFileTransfer().fetch(batch_dir, sorted(listing_), str(local))
    ((relative, iteration),) = tasks[0].checkpoints
    assert iteration == 6
    assert fetching.verify_checkpoint(str(local / relative)) == []
    # Loading imports the project's components, as on any workstation with the project.
    monkeypatch.syspath_prepend(str(root))
    loaded = Checkpointer.load_checkpoint(str(local / relative))
    assert loaded.iteration == 6
    assert (local / "tasks/cpu/seeds/000/record.json").exists()


# The CLI, through the fake cluster -------------------------------------------------------

CLI_RSYNC = FAKE_RSYNC.replace('open(os.environ["RSYNC_ARGV"], "w").write("\\n".join(args))\n', "").replace(
    'open(os.environ["RSYNC_LIST"], "w").write(open(listing).read())\n', "")


def submitted_and_finished(fake_cluster, tmp_path, monkeypatch, capsys):
    from tests.hpc.services.test_batches import submitted_batch

    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    (tmp_path / "squeue-bin" / "squeue").write_text(f"#!{sys.executable}\nimport sys\n")   # all ended
    for number, config in enumerate(sorted(batch_dir.glob("tasks/*/*/*/config.yaml"))):
        directory = config.parent
        (directory / "record.json").write_text(json.dumps({
            "exit_status": 0 if number < 2 else 1, "interrupted": False,
            "run_directories": [str(directory / "sessions/session_1")]}))
        write_checkpoint(directory / "sessions" / "session_1" / "checkpoints" / "x", 10)
    # A fetch-capable rsync first on PATH (the fake cluster's only uploads).
    (tmp_path / "squeue-bin" / "rsync").write_text(f"#!{sys.executable}\n{CLI_RSYNC}")
    (tmp_path / "squeue-bin" / "rsync").chmod(0o755)
    return batch_dir, profiles


def test_fetch_through_the_cli_and_again(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_and_finished(fake_cluster, tmp_path, monkeypatch, capsys)
    dest = tmp_path / "results"
    assert main(["fetch", batch_dir.name, "--dest", str(dest), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 task(s)" in out and out.count("iteration 10") == 2 and out.count("skip ") == 2
    assert not dest.exists()

    assert main(["fetch", batch_dir.name, "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "checkpoints: 2 verified" in out and "view: tensorboard --logdir" in out
    local = dest / batch_dir.name
    index = json.loads((local / "fetched.json").read_text())
    assert len(index["tasks"]) == 2 and index["cluster_batch_dir"] == str(batch_dir)
    fetched = sorted(p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file())
    assert "plan.json" in fetched and "receipt.json" in fetched
    assert sum("/checkpoints/x/components/model.pt" in p for p in fetched) == 2
    assert main(["fetch", batch_dir.name, "--dest", str(dest), "--dry-run"]) == 0
    again = capsys.readouterr().out
    total = again.split(" file(s), ")[0].rsplit("\n", 1)[-1]
    assert f"already here: {total} file(s)" in again             # nothing new to download
