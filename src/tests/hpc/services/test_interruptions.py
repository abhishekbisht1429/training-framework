"""Interrupted uploads and submissions: resuming is safe, Ctrl-C is recorded,
and missing local tools fail before any slow work."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from training_framework.hpc.__main__ import main
from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.local import LocalFileTransfer
from training_framework.hpc.services.artifacts import PENDING_SUFFIX
from tests.hpc.fakes import FakeSubmitter
from tests.hpc.services import test_submission
from tests.hpc.services.test_artifact_services import local_data, run, upload_batch


def mapped_upload(tmp_path):
    local = local_data(tmp_path)
    compiled = upload_batch(tmp_path, local, path_map={
        str(tmp_path / "datasets"): str(tmp_path / "cluster" / "project")})()
    return compiled, next(a for a in compiled.plan.artifacts if a.kind == "upload")


class InterruptedTransfer(LocalFileTransfer):
    """Copies, then is interrupted (Ctrl-C) before finishing, once."""

    def __init__(self):
        self.interrupted = False

    def sync(self, local, remote):
        sent = super().sync(local, remote)
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return sent


def test_an_interrupted_upload_records_its_key_and_resumes(tmp_path):
    compiled, upload = mapped_upload(tmp_path)
    transfer = InterruptedTransfer()
    with pytest.raises(KeyboardInterrupt) as interrupt:
        run(compiled, tmp_path, transfer=transfer)
    pending = Path(upload.location + PENDING_SUFFIX)
    assert json.loads(pending.read_text()) == {"key": upload.key}
    assert not Path(upload.location + ".tf-hpc-ready").exists()
    assert not Path(upload.location + ".tf-hpc-lock").exists()      # the lease was released
    receipt = interrupt.value.receipt
    assert receipt.status == "failed" and "interrupted" in receipt.data["error"]

    receipt, _ = run(_later(tmp_path), tmp_path, transfer=transfer)   # a new batch, same data
    assert receipt.status == "submitted"
    assert Path(upload.location + ".tf-hpc-ready").exists() and not pending.exists()


def _later(tmp_path):
    from datetime import timedelta

    from tests.hpc.batch_repo import CLOCK

    return upload_batch(tmp_path, tmp_path / "datasets" / "flowers", path_map={
        str(tmp_path / "datasets"): str(tmp_path / "cluster" / "project")})(
        clock=CLOCK + timedelta(minutes=1))


def test_resuming_an_interrupted_upload_with_other_content_is_refused(tmp_path):
    compiled, upload = mapped_upload(tmp_path)
    Path(upload.location).mkdir(parents=True)
    Path(upload.location + PENDING_SUFFIX).write_text(json.dumps({"key": "older-content-key"}))
    submitter = FakeSubmitter()
    with pytest.raises(HpcError, match="interrupted upload of different content") as error:
        run(compiled, tmp_path, submitter=submitter)
    assert f"Delete {upload.location} and {upload.location}{PENDING_SUFFIX}" in str(error.value)
    assert not submitter.submitted


class InterruptingSubmitter(FakeSubmitter):
    """Ctrl-C arrives while the `at`-th sbatch is in flight."""

    def __init__(self, at):
        super().__init__()
        self.at = at
        self.calls = 0

    def submit(self, node, script_path, dependencies):
        self.calls += 1
        if self.calls == self.at:
            raise KeyboardInterrupt
        return super().submit(node, script_path, dependencies)


def test_ctrl_c_while_submitting_cancels_earlier_jobs_and_needs_reconciliation(tmp_path):
    compiled = test_submission.compiled.__wrapped__(tmp_path)
    submitter = InterruptingSubmitter(at=2)
    with pytest.raises(KeyboardInterrupt) as interrupt:
        test_submission.run(compiled, tmp_path, submitter)
    receipt = interrupt.value.receipt
    assert receipt.status == "needs reconciliation"
    assert "interrupted while submitting group-second" in receipt.data["error"]
    assert receipt.node("group-second")["status"] == "unknown"
    assert receipt.node("group-probes")["status"] == "cancelled"
    assert submitter.cancelled == ["1000"]


def test_ctrl_c_during_the_first_sbatch_needs_reconciliation(tmp_path):
    compiled = test_submission.compiled.__wrapped__(tmp_path)
    with pytest.raises(KeyboardInterrupt) as interrupt:
        test_submission.run(compiled, tmp_path, InterruptingSubmitter(at=1))
    receipt = interrupt.value.receipt
    # The first sbatch was in flight: it may have been accepted.
    assert receipt.status == "needs reconciliation"
    assert receipt.node("group-probes")["status"] == "unknown"


def test_submit_checks_for_ssh_and_rsync_before_any_work(tmp_path, monkeypatch, capsys):
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "rsync" else real_which(name))
    profiles = tmp_path / "hpc.yaml"
    profiles.write_text(json.dumps(test_submission.profile_for(tmp_path)))
    assert main(["submit", str(tmp_path / "no-such-batch.yaml"), "--profiles", str(profiles)]) == 2
    error = capsys.readouterr().err
    assert "rsync not found on this machine's PATH" in error
    assert "no-such-batch" not in error          # stopped before compiling anything
