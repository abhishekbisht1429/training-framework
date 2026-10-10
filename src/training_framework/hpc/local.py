"""Local implementations of the outbound ports, and local source staging."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

from training_framework.hpc.errors import HpcError
from training_framework.hpc.plan import ExecutionContext
from training_framework.hpc.ports import ExecResult, Owner, Stat
from training_framework.hpc.shell import exec_script


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


class LocalFilesystem:
    def mkdir_atomic(self, path: str) -> bool:
        try:
            os.mkdir(path)
        except FileExistsError:
            return False
        return True

    def makedirs(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def rename(self, source: str, target: str) -> None:
        os.replace(source, target)

    def read(self, path: str) -> bytes | None:
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return None

    def write(self, path: str, data: bytes) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def delete(self, path: str) -> None:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.unlink(path)

    def exists(self, path: str) -> bool:
        return os.path.lexists(path)

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


def process_start_time(pid: int) -> str | None:
    """The kernel's start time of `pid` (clock ticks since boot), or None."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        return fields[19]   # field 22 overall; 20th after "(comm)"
    except (FileNotFoundError, IndexError, ProcessLookupError):
        return None


class LocalProcessProbe:
    """`LeaseOwnerProbe` for process owners on this host."""

    def __init__(self, host: str | None = None):
        self.host = host or socket.gethostname()

    def is_alive(self, owner: Owner) -> bool | None:
        if owner.job_ref is not None or owner.host != self.host or owner.pid is None:
            return None
        start = process_start_time(owner.pid)
        return start is not None and start == owner.start_time

    def current_owner(self) -> Owner:
        return Owner(host=self.host, pid=os.getpid(),
                     start_time=process_start_time(os.getpid()))


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
