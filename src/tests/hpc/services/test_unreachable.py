"""An unreachable cluster is reported, with SSH's reason and what to check,
instead of an empty or noisy listing."""

from __future__ import annotations

import json
import os
from pathlib import Path

from training_framework.hpc.__main__ import main
from training_framework.hpc.ports.ssh import TransportError, ssh_reason
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)
from tests.hpc.services.test_batches import submitted_batch


def test_ssh_noise_is_dropped_from_the_reason():
    stderr = ("Warning: Permanently added '[h]:22' (ED25519) to the list of known hosts.\n"
              "Load key \"/home/me/.ssh/id\": bad permissions\n"
              "me@h: Permission denied (publickey).\n")
    assert ssh_reason(stderr) == ('Load key "/home/me/.ssh/id": bad permissions; '
                                  "me@h: Permission denied (publickey).")
    assert ssh_reason("") == "ssh exited 255 without a message"
    error = TransportError("bigred200", "me@h: Permission denied (publickey).")
    assert str(error) == "cannot reach bigred200 over SSH: me@h: Permission denied (publickey)."


def test_every_command_says_why_the_cluster_cannot_be_reached(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    monkeypatch.setenv("FAKE_SSH_DOWN", "1")

    assert main(["batches", "--live", "--profiles", str(profiles)]) == 1
    captured = capsys.readouterr()
    assert "Connection refused" in captured.out                     # the batch's row says why
    assert "error: cannot reach " in captured.err and " over SSH: " in captured.err
    assert captured.err.count("hint: check that `ssh ") == 1         # once per cluster

    assert main(["status", batch_dir.name]) == 2
    captured = capsys.readouterr()
    assert "cannot reach" in captured.err and "Connection refused" in captured.err
    assert "hint: " in captured.err

    assert main(["batches", "--remote", "--profiles", str(profiles)]) == 2
    assert "Connection refused" in capsys.readouterr().err


def test_an_empty_live_listing_still_checks_the_cluster(fake_cluster, tmp_path, monkeypatch, capsys):
    batch_dir, profiles = submitted_batch(fake_cluster, tmp_path, monkeypatch, capsys)
    receipts = Path(os.environ["XDG_STATE_HOME"]) / "training-framework" / "submissions"
    for item in receipts.iterdir():
        item.unlink()                          # e.g. every submit failed before its receipt
    assert main(["batches", "--profiles", str(profiles)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "no batches"
    assert "no batches were submitted from this machine" in captured.err and "--remote" in captured.err
    assert main(["batches", "--live", "--profiles", str(profiles)]) == 0      # reachable: just empty
    monkeypatch.setenv("FAKE_SSH_DOWN", "1")
    assert main(["batches", "--live", "--profiles", str(profiles)]) == 2
    err = capsys.readouterr().err
    assert "cannot reach" in err and "Connection refused" in err and "hint: " in err
