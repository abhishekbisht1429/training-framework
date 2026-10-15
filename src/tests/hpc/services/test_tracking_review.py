"""`batches` and `status` from the review: missing cluster files, the
receipt's own cluster, offline listing without a profile, malformed
receipts, Slurm state classification, and uncertain submissions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training_framework.hpc.__main__ import main
from training_framework.hpc.ports.local import LocalExecutor
from training_framework.hpc.services import batches as listing
from training_framework.hpc.services.status import Row, summarize
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)
from tests.hpc.services.test_batches import NOW, receipt, submitted_batch, write_receipts
from tests.hpc.test_logs_and_status import _batch


def live_view(tmp_path, **changes):
    batch = dict(_batch(tmp_path), ssh_host="cluster", created=NOW.isoformat(), **changes)
    return listing.BatchView(batch, "local")


def local(states=None):
    return lambda host, scheduler: (LocalExecutor(), lambda: dict(states or {}))


# 1. Missing cluster files are a problem, not a lifecycle -------------------------------


def test_a_missing_batch_directory_is_a_problem_not_finished(tmp_path):
    view = live_view(tmp_path, staged=True, batch_dir=str(tmp_path / "purged"))
    listing.add_live_state([view], local({"9_3": "running"}), None)
    assert view.now is None and view.rows is None
    assert "not found (deleted, purged, or another cluster) on cluster" in view.problem
    assert "its jobs are still in the queue" in view.problem
    assert listing.select([view], active=True) == []        # no summary, so not claimed active


def test_a_missing_task_table_is_a_problem(tmp_path):
    view = live_view(tmp_path, staged=True)
    os.remove(Path(view.receipt["batch_dir"]) / "groups" / "g" / "tasks.tsv")
    listing.add_live_state([view], local(), None)
    assert view.now is None and "task table missing for group(s) g" in view.problem


def test_a_batch_that_never_reached_the_cluster_still_summarizes(tmp_path):
    view = listing.BatchView(dict(receipt("b", status="staged", ssh_host="cluster"),
                                  staged=False), "local")
    listing.add_live_state([view], local(), None)
    assert view.now == "not submitted" and view.problem is None


# 2. `status` goes where the receipt says ------------------------------------------------


def test_status_uses_the_receipts_cluster_and_refuses_a_conflicting_profile(
        fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    receipt_file = (Path(os.environ["XDG_STATE_HOME"]) / "training-framework" / "submissions"
                    / f"{batch_dir.name}.json")
    data = json.loads(receipt_file.read_text())
    data["ssh_host"] = "other-cluster"
    receipt_file.write_text(json.dumps(data))
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    # The default profile file points at another host; status ignores it.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config = tmp_path / "config" / "training-framework"
    config.mkdir(parents=True)
    (config / "hpc.yaml").write_text(profiles.read_text())
    assert main(["status", batch_dir.name]) == 0
    assert set(log.read_text().split()) == {"other-cluster"}
    assert "other-cluster:" in capsys.readouterr().out
    assert main(["status", batch_dir.name, "--profiles", str(profiles)]) == 2
    error = capsys.readouterr().err
    assert "was submitted to other-cluster" in error and "drop --profile" in error


def test_status_of_a_purged_batch_is_an_error(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    import shutil
    shutil.rmtree(batch_dir)
    assert main(["status", batch_dir.name]) == 2
    assert "not found (deleted, purged, or another cluster)" in capsys.readouterr().err


# 3. The offline listing never reads the profile file ------------------------------------


def test_offline_batches_works_with_a_broken_profile_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config = tmp_path / "config" / "training-framework"
    config.mkdir(parents=True)
    (config / "hpc.yaml").write_text("profiles: [not: a mapping\n")
    write_receipts(tmp_path / "receipts", receipt("legacy"))   # no recorded cluster
    assert main(["batches", "--receipts", str(tmp_path / "receipts")]) == 0
    assert capsys.readouterr().out.splitlines()[1].split()[0] == "legacy"
    # A live lookup needs the profile only for this older receipt, and says why.
    assert main(["batches", "--live", "--receipts", str(tmp_path / "receipts")]) == 0
    out = capsys.readouterr().out
    assert "predates receipts recording their cluster" in out and "pass --profile" in out


# 4. Malformed receipts are isolated -----------------------------------------------------


def test_malformed_receipts_are_listed_as_unreadable_and_do_not_hide_others(tmp_path):
    good = dict(_batch(tmp_path / "good"), ssh_host="cluster", created=NOW.isoformat())
    no_dir = {k: v for k, v in good.items() if k != "batch_dir"} | {"batch_id": "no-dir"}
    no_nodes = {k: v for k, v in good.items() if k != "nodes"} | {"batch_id": "no-nodes"}
    write_receipts(tmp_path / "receipts", dict(good, batch_id="good"), no_dir, no_nodes)
    views = {view.batch_id: view for view in listing.load_receipts(str(tmp_path / "receipts"))}
    assert "missing or malformed: batch_dir" in views["no-dir"].receipt["error"]
    assert "missing or malformed: nodes" in views["no-nodes"].receipt["error"]
    listing.add_live_state(list(views.values()), local(), None)
    assert views["good"].now.startswith("finished")
    assert views["no-dir"].problem and views["no-nodes"].problem


def test_remote_receipts_skip_malformed_ones(tmp_path):
    root = tmp_path / "batches"
    for name, data in (("ok", receipt("ok")), ("bad", {"batch_id": "bad"})):
        (root / name).mkdir(parents=True)
        (root / name / "receipt.json").write_text(json.dumps(data))
    assert [r["batch_id"] for r in listing.remote_receipts(LocalExecutor(), str(root))] == ["ok"]


# 5. Slurm states --------------------------------------------------------------------------

SUBMITTED = receipt("b", nodes={"group-g": {"status": "submitted", "job_ref": "9"}})


def task(state, live=True):
    return Row("g/e/000", "", state, "job 9_0" if live else "", live)


@pytest.mark.parametrize("state", ["stopped", "requeue_hold", "requeue_fed", "special_exit",
                                   "suspended", "some_future_state"])
def test_non_terminal_slurm_states_keep_a_batch_active(state):
    assert not summarize(SUBMITTED, [task(state)]).startswith("finished")


@pytest.mark.parametrize("states, phase", [
    (["completing"], "running"),
    (["pending", "requeue_hold"], "queued"),
    (["pending", "running"], "running"),
    (["timeout"], "finished"),          # terminal, still briefly shown by squeue
])
def test_the_phase_follows_the_live_states(states, phase):
    rows = [Row(f"g/e/{i:03d}", "", state, f"job 9_{i}", True) for i, state in enumerate(states)]
    assert summarize(SUBMITTED, rows).startswith(f"{phase}:")


# 6. Uncertain submissions -----------------------------------------------------------------


def test_needs_reconciliation_is_never_called_finished():
    uncertain = receipt("b", status="needs reconciliation",
                        nodes={"group-a": {"status": "cancelled", "job_ref": "8"},
                               "group-b": {"status": "unknown", "job_ref": None}})
    rows = [Row("a/e/000", "", "cancelled", ""), Row("b/e/000", "", "unknown", "")]
    assert summarize(uncertain, rows) == "needs reconciliation: 1 cancelled, 1 unknown"
