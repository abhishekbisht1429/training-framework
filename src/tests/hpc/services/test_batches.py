"""`batches`: listing receipts, filters, live state per cluster, output."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training_framework.hpc.__main__ import main
from training_framework.hpc.ports.local import LocalExecutor
from training_framework.hpc.services import batches as listing
from training_framework.hpc.services.status import Row, summarize
from tests.hpc.fake_cluster import fake_cluster, submitted_jobs  # noqa: F401 (fixture)
from tests.hpc.test_cli_submit import probed_identity, setup as cli_setup
from tests.hpc.test_logs_and_status import _batch

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def receipt(batch_id, hours_ago=1, status="submitted", **extra):
    return {"batch_id": batch_id, "created": (NOW - timedelta(hours=hours_ago)).isoformat(),
            "status": status, "nodes": {}, "batch_dir": f"/b/{batch_id}", **extra}


def write_receipts(directory: Path, *receipts):
    directory.mkdir(parents=True, exist_ok=True)
    for item in receipts:
        (directory / f"{item['batch_id']}.json").write_text(json.dumps(item))


# Local receipts and filters ---------------------------------------------------------


def test_receipts_are_listed_newest_first_and_unreadable_ones_are_shown(tmp_path):
    write_receipts(tmp_path, receipt("old", hours_ago=48), receipt("new", hours_ago=1))
    (tmp_path / "broken.json").write_text("{not json")
    views = listing.load_receipts(str(tmp_path))
    assert [view.batch_id for view in views] == ["new", "old", "broken"]
    assert views[-1].receipt["status"] == "unreadable receipt"
    assert listing.load_receipts(str(tmp_path / "missing")) == []


@pytest.mark.parametrize("text, expected", [("30m", timedelta(minutes=30)), ("12h", timedelta(hours=12)),
                                            ("7d", timedelta(days=7)), ("2w", timedelta(weeks=2))])
def test_ages(text, expected):
    assert listing.parse_age(text) == expected


def test_a_bad_age_is_an_error():
    with pytest.raises(ValueError, match="such as 30m"):
        listing.parse_age("7 days")


def test_filters(tmp_path):
    write_receipts(tmp_path, receipt("vit-a", 1), receipt("vit-b", 30), receipt("ssl-c", 2))
    views = listing.load_receipts(str(tmp_path))
    ids = lambda chosen: [view.batch_id for view in chosen]   # noqa: E731
    assert ids(listing.select(views, since=timedelta(days=1), now=NOW)) == ["vit-a", "ssl-c"]
    assert ids(listing.select(views, name="vit", now=NOW)) == ["vit-a", "vit-b"]
    assert ids(listing.select(views, limit=1, now=NOW)) == ["vit-a"]
    views[0].now, views[1].now = "running: 1 running", "finished: 2 done"
    assert ids(listing.select(views, active=True, now=NOW)) == ["vit-a"]


# Summaries ------------------------------------------------------------------------


def test_summaries_say_what_a_batch_is_doing():
    submitted = receipt("b", nodes={"setup": {}, "group-g": {"job_ref": "9"}})
    rows = [Row("setup", "", "built", ""), Row("g/e/000", "", "running", "job 9_0", True),
            Row("g/e/001", "", "pending", "job 9_1", True), Row("g/e/002", "", "done", ""),
            Row("g/e/003", "", "failed (exit 1)", "")]
    assert summarize(submitted, rows) == "running: 1 running, 1 pending, 1 done, 1 failed"
    assert summarize(submitted, rows[3:]) == "finished: 1 done, 1 failed"
    assert summarize(submitted, [Row("setup", "", "failed", "pip"), Row("g/e/0", "", "cancelled", "")]) \
        == "finished: 1 cancelled (setup failed)"
    assert summarize(receipt("b", status="staged"), []) == "not submitted"
    assert summarize(receipt("b", status="failed", nodes={"group-g": {"job_ref": None}}), []) \
        == "submit failed"


# Live state -----------------------------------------------------------------------


def test_live_state_reads_records_tables_and_the_queue_per_cluster(tmp_path):
    batch = dict(_batch(tmp_path), ssh_host="cluster", scheduler="slurm",
                 created=NOW.isoformat())
    view = listing.BatchView(batch, "local")
    connections = []

    def connect(host, scheduler):
        connections.append((host, scheduler))
        return LocalExecutor(), lambda: {"9_3": "running"}

    listing.add_live_state([view], connect, fallback=None)
    assert connections == [("cluster", "slurm")]
    assert view.now == "running: 1 running, 1 done, 1 interrupted, 1 failed"
    states = {row.name: row.state for row in view.rows}
    assert states == {"setup": "built", "g/e/000": "done", "g/e/001": "failed (exit 1)",
                      "g/e/002": "interrupted", "g/e/003": "running"}


def test_an_unreachable_cluster_is_reported_on_its_batches(tmp_path):
    view = listing.BatchView(receipt("b", ssh_host="down"), "local")

    def connect(host, scheduler):
        raise ConnectionError("ssh: connect to host down: Connection refused")

    listing.add_live_state([view], connect, fallback=None)
    assert view.now is None and "cannot read down" in view.problem


def test_older_receipts_use_the_selected_profile_or_say_what_is_missing(tmp_path):
    batch = dict(_batch(tmp_path), created=NOW.isoformat())     # no ssh_host recorded
    with_profile, without = listing.BatchView(batch, "local"), listing.BatchView(dict(batch), "local")
    listing.add_live_state([with_profile], lambda h, s: (LocalExecutor(), None),
                           lambda: ("cluster", "slurm"))
    assert with_profile.now.startswith("finished")
    listing.add_live_state([without], lambda h, s: (LocalExecutor(), None), None)
    assert "pass --profile" in without.problem


def test_the_table_with_tasks_and_the_json(tmp_path):
    batch = dict(_batch(tmp_path), ssh_host="cluster", created=NOW.isoformat(),
                 profile="bigred200", task_count=4)
    view = listing.BatchView(batch, "local")
    listing.add_live_state([view], lambda h, s: (LocalExecutor(), lambda: {}), None)
    text = listing.format_views([view], live=True, tasks=True, origins=False)
    lines = text.splitlines()
    assert lines[0].split() == ["batch", "submitted", "profile", "tasks", "submit", "now"]
    assert lines[1].split()[0] == "b" and "bigred200" in lines[1] and lines[1].endswith(view.now)
    assert any(line.startswith("    g/e/001") and "failed (exit 1)" in line for line in lines)
    (entry,) = listing.as_json([view], tasks=True)
    assert entry["now"] == view.now and entry["tasks"][0]["name"] == "setup"
    assert listing.format_views([], live=False, tasks=False, origins=False) == "no batches"


# The CLI, through the fake cluster --------------------------------------------------

SQUEUE = r'''
import json, os, sys
state = os.environ["FAKE_SLURM_DIR"]
if "--user" in sys.argv:
    for line in open(os.path.join(state, "jobs.jsonl")):
        job = json.loads(line)["job"]
        print("%s_0|RUNNING" % job)
        print("%s_[1-3]|PENDING" % job)
'''


def submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys):
    root, path = cli_setup(tmp_path, monkeypatch)
    identity = probed_identity(path, capsys)
    root, path = cli_setup(tmp_path / "second", monkeypatch, identity)
    assert main(["submit", str(root / "batch.yaml"), "--profiles", str(path), "--yes"]) == 0
    output = capsys.readouterr().out
    batch_dir = Path(output.split("submitted: ")[1].splitlines()[0])
    bin_dir = tmp_path / "squeue-bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text(f"#!{sys.executable}\n{SQUEUE}")
    (bin_dir / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return batch_dir, path


def test_batches_lists_live_states_and_tasks_after_a_submit(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    receipt_file = Path(os.environ["XDG_STATE_HOME"]) / "training-framework" / "submissions" / \
        f"{batch_dir.name}.json"
    saved = json.loads(receipt_file.read_text())
    assert saved["profile"] == "cluster" and saved["task_count"] == 4
    assert saved["ssh_host"] and saved["manifest"] == "batch.yaml" and saved["submitted_from"]

    assert main(["batches", "--profiles", str(profiles)]) == 0
    offline = capsys.readouterr().out.splitlines()
    assert offline[0].split() == ["batch", "submitted", "profile", "tasks", "submit"]
    assert offline[1].split()[0] == batch_dir.name and offline[1].rstrip().endswith("submitted")

    assert main(["batches", "--tasks", "--profiles", str(profiles)]) == 0
    live = capsys.readouterr().out
    assert "running: 1 running, 3 pending" in live
    assert "    probes/" in live and "pending" in live

    assert main(["batches", "--json", "--active", "--profiles", str(profiles)]) == 0
    (entry,) = json.loads(capsys.readouterr().out)
    assert entry["batch_id"] == batch_dir.name and entry["now"].startswith("running")


def test_batches_remote_finds_batches_submitted_elsewhere(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    receipts = Path(os.environ["XDG_STATE_HOME"]) / "training-framework" / "submissions"
    for item in receipts.iterdir():
        item.unlink()                                   # as if submitted from another machine
    assert main(["batches", "--profiles", str(profiles)]) == 0
    assert capsys.readouterr().out.strip() == "no batches"
    assert main(["batches", "--remote", "--profiles", str(profiles)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:4] == ["batch", "submitted", "profile", "from"]
    assert lines[1].split()[0] == batch_dir.name


def test_batches_needs_no_profile_for_the_offline_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-config"))
    write_receipts(tmp_path / "receipts", receipt("solo"))
    assert main(["batches", "--receipts", str(tmp_path / "receipts")]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[0] == "solo"
    assert main(["batches", "--remote", "--receipts", str(tmp_path / "receipts")]) == 2
    assert "hpc.yaml does not exist" in capsys.readouterr().err      # --remote needs the profile


def test_pending_array_ranges_count_for_each_element():
    from training_framework.hpc.services.status import expand_ranges

    live = expand_ranges({"7_0": "running", "7_[1-3,5%2]": "pending", "8": "running"})
    assert [live[f"7_{i}"] for i in (0, 1, 2, 3, 5)] == ["running"] + ["pending"] * 4
    assert "7_4" not in live and live["8"] == "running"
