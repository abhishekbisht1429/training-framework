"""Cluster adapters over the user's OpenSSH: executor, filesystem, and
rsync transfer (Slurm-specific adapters live in `scheduler/`).

SSH uses the user's own configuration, agent, host keys, and control
connections; nothing is stored and MFA is never automated (`BatchMode=yes`
fails fast instead of prompting). Remote commands are argument vectors,
each argument quoted for the remote shell.
"""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import subprocess
import uuid

from training_framework.hpc.errors import HpcError
from training_framework.hpc.compile.plan import ExecutionContext
from training_framework.hpc.ports import ExecResult, Stat
from training_framework.hpc.ports.shell import exec_script

SSH_OPTIONS = ("-o", "BatchMode=yes", "-T")


class TransportError(HpcError):
    """The SSH connection itself failed (exit 255); the remote command may or
    may not have run."""


class SshExecutor:
    def __init__(self, host: str, ssh: tuple[str, ...] = ("ssh",)):
        self.host = host
        self.ssh = tuple(ssh)

    def _command(self, remote: list[str]) -> list[str]:
        return [*self.ssh, *SSH_OPTIONS, self.host, "--", shlex.join(remote)]

    def run_bytes(self, argv: list[str], data: bytes | None = None,
                  timeout: float | None = None) -> tuple[int, bytes, str]:
        result = subprocess.run(self._command(argv), input=data, capture_output=True,
                                timeout=timeout, check=False)
        stderr = result.stderr.decode(errors="replace")
        if result.returncode == 255:
            raise TransportError(f"ssh {self.host} failed: {stderr.strip()}")
        return result.returncode, result.stdout, stderr

    def run(self, argv: list[str], context: ExecutionContext | None = None,
            timeout: float | None = None) -> ExecResult:
        if not argv:
            raise ValueError("argv must not be empty")
        remote = list(argv)
        if context is not None:
            remote = ["bash", "-c", exec_script(context), "bash", *argv]
        status, stdout, stderr = self.run_bytes(remote, timeout=timeout)
        return ExecResult(status, stdout.decode(errors="replace"), stderr)


class SshFilesystem:
    def __init__(self, executor: SshExecutor):
        self.executor = executor

    def _ok(self, argv: list[str]) -> bool:
        return self.executor.run_bytes(argv)[0] == 0

    def _must(self, argv: list[str], what: str) -> None:
        status, _, stderr = self.executor.run_bytes(argv)
        if status != 0:
            raise HpcError(f"{what} failed on {self.executor.host}: {stderr.strip()}")

    def mkdir_atomic(self, path: str) -> bool:
        if self._ok(["mkdir", path]):
            return True
        if self.exists(path):
            return False
        raise HpcError(f"mkdir {path} failed on {self.executor.host}")

    def makedirs(self, path: str) -> None:
        self._must(["mkdir", "-p", path], f"mkdir -p {path}")

    def rename(self, source: str, target: str) -> None:
        self._must(["mv", "-f", source, target], f"mv {source} {target}")

    def read(self, path: str) -> bytes | None:
        status, stdout, stderr = self.executor.run_bytes(["cat", path])
        if status == 0:
            return stdout
        if not self.exists(path):
            return None
        raise HpcError(f"reading {path} failed on {self.executor.host}: {stderr.strip()}")

    def write(self, path: str, data: bytes) -> None:
        directory = posixpath.dirname(path) or "."
        temporary = f"{directory}/.tmp-{uuid.uuid4().hex}"
        status, _, stderr = self.executor.run_bytes(
            ["sh", "-c", 'mkdir -p "$1" && cat > "$2" && mv -f "$2" "$3"',
             "sh", directory, temporary, path],
            data,
        )
        if status != 0:
            raise HpcError(f"writing {path} failed on {self.executor.host}: {stderr.strip()}")

    def delete(self, path: str) -> None:
        self._must(["rm", "-rf", "--", path], f"rm -rf {path}")

    def exists(self, path: str) -> bool:
        return self._ok(["sh", "-c", 'test -e "$1" || test -L "$1"', "sh", path])

    def stat(self, path: str) -> Stat | None:
        status, stdout, _ = self.executor.run_bytes(["stat", "-c", "%s %F", path])
        if status != 0:
            return None
        size, kind = stdout.decode().strip().split(" ", 1)
        return Stat(int(size), kind == "directory")


_TRANSFERRED = re.compile(r"Total transferred file size: ([\d,]+) bytes")


class RsyncTransfer:
    """Resumable one-way sync over SSH (`--partial`)."""

    def __init__(self, host: str, ssh: tuple[str, ...] = ("ssh",), rsync: str = "rsync"):
        self.host = host
        self.ssh = tuple(ssh)
        self.rsync = rsync

    def sync(self, local: str, remote: str) -> int:
        is_dir = os.path.isdir(local)
        source = local.rstrip("/") + "/" if is_dir else local
        target = remote.rstrip("/") + "/" if is_dir else remote
        parent = remote.rstrip("/") if is_dir else posixpath.dirname(remote)
        command = [
            self.rsync, "-a", "--partial", "--protect-args", "--stats",
            "-e", shlex.join([*self.ssh, *SSH_OPTIONS]),
            "--rsync-path", f"mkdir -p {shlex.quote(parent)} && rsync",
            source, f"{self.host}:{target}",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise HpcError(f"rsync to {self.host}:{remote} failed: {result.stderr.strip()}")
        match = _TRANSFERRED.search(result.stdout)
        return int(match.group(1).replace(",", "")) if match else 0
