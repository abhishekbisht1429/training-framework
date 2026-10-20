"""rsync never preserves owner or group: copies belong to the receiving user
(a root receiver otherwise chowns to the sender's numeric IDs, which some
filesystems and containers reject)."""

from __future__ import annotations

import sys

from training_framework.hpc.ports.ssh import RsyncTransfer

RECORDING_RSYNC = r'''
import os, sys
open(os.environ["RSYNC_ARGV"], "a").write(" ".join(sys.argv[1:]) + "\n")
print("Number of files: 1 (reg: 1)\nNumber of regular files transferred: 0\nTotal transferred file size: 0 bytes")
'''


def test_uploads_and_downloads_keep_neither_owner_nor_group(tmp_path, monkeypatch):
    rsync = tmp_path / "rsync"
    rsync.write_text(f"#!{sys.executable}\n{RECORDING_RSYNC}")
    rsync.chmod(0o755)
    log = tmp_path / "argv"
    monkeypatch.setenv("RSYNC_ARGV", str(log))
    (tmp_path / "data").mkdir()
    transfer = RsyncTransfer("cluster", rsync=str(rsync))
    transfer.sync(str(tmp_path / "data"), "/remote/data")
    transfer.fetch("/remote/batch", ["a/b"], str(tmp_path / "here"))
    upload, download = (line.split() for line in log.read_text().splitlines())
    for argv in (upload, download):
        assert argv[:3] == ["-a", "--no-owner", "--no-group"], argv
