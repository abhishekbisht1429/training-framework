"""Local implementations of the outbound ports, and local source staging."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

from training_framework.hpc.errors import HpcError
from training_framework.hpc.compile.plan import ExecutionContext
from training_framework.hpc.ports import ExecResult, Owner, Stat
from training_framework.hpc.remote.runtime import (
    ProcessProbe,
    StdlibFilesystem,
    current_process_owner,
)
from training_framework.hpc.ports.shell import exec_script


class LocalExecutor:
    def run(self, argv: list[str], context: ExecutionContext | None = None,
            timeout: float | None = None) -> ExecResult:
        if not argv:
            raise ValueError("argv must not be empty")
        command = list(argv)
        if context is not None:
            command = ["bash", "-c", exec_script(context), "bash", *argv]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return ExecResult(result.returncode, result.stdout, result.stderr)


class LocalFilesystem(StdlibFilesystem):
    """`RemoteFilesystem` on this machine (the stdlib implementation the
    bootstrap also uses), plus `stat`."""

    def stat(self, path: str) -> Stat | None:
        try:
            info = os.stat(path)
        except FileNotFoundError:
            return None
        return Stat(info.st_size, os.path.isdir(path))


class LocalFileTransfer:
    """`FileTransfer` between local paths; resumable in that files already
    present with the same size and mtime are skipped."""

    def sync(self, local: str, remote: str) -> int:
        transferred = 0
        if os.path.isfile(local):
            pairs = [(local, remote)]
        else:
            pairs = []
            for directory, _, files in os.walk(local):
                for name in files:
                    source = os.path.join(directory, name)
                    pairs.append((source, os.path.join(remote, os.path.relpath(source, local))))
        for source, target in pairs:
            source_stat = os.stat(source)
            try:
                target_stat = os.stat(target)
                if (target_stat.st_size == source_stat.st_size
                        and int(target_stat.st_mtime) == int(source_stat.st_mtime)):
                    continue
            except FileNotFoundError:
                pass
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            shutil.copy2(source, target)
            transferred += source_stat.st_size
        return transferred


class LocalProcessProbe:
    """`LeaseOwnerProbe` for process owners on this host."""

    def __init__(self, host: str | None = None):
        self._probe = ProcessProbe(host)
        self.host = self._probe.host

    def is_alive(self, owner: Owner) -> bool | None:
        return self._probe.is_alive(owner.as_json())

    def current_owner(self) -> Owner:
        return Owner(**current_process_owner())


class LocalSourceStager:
    """Writes the exact commit's tree (`git archive`) to a local directory."""

    def __init__(self, repository_root: Path):
        self.root = repository_root

    def stage(self, sha: str, target: str) -> None:
        if os.path.exists(target):
            raise HpcError(f"{target} already exists")
        archive = subprocess.run(["git", "archive", "--format=tar", sha],
                                 cwd=self.root, capture_output=True, check=False)
        if archive.returncode != 0:
            raise HpcError(f"git archive failed: {archive.stderr.decode().strip()}")
        os.makedirs(target)
        with tarfile.open(fileobj=BytesIO(archive.stdout)) as tar:
            tar.extractall(target, filter="data")
