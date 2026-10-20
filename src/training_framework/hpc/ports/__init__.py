"""Outbound ports: the only ways the HPC layer touches the outside world.

Each is narrow and takes argument vectors and paths, never shell strings, so
none can become a generic shell escape hatch. Every implementation of a port
passes that port's contract suite (`src/tests/hpc/test_contract_*.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from training_framework.hpc.errors import HpcError
from training_framework.hpc.compile.plan import ExecutionContext, JobNode


@dataclass(frozen=True)
class ExecResult:
    status: int
    stdout: str
    stderr: str


class RemoteExecutor(Protocol):
    def run(self, argv: list[str], context: ExecutionContext | None = None,
            timeout: float | None = None) -> ExecResult: ...


@dataclass(frozen=True)
class Stat:
    size: int
    is_dir: bool


class RemoteFilesystem(Protocol):
    def mkdir_atomic(self, path: str) -> bool:
        """Create `path` (parents must exist); False if it already exists."""

    def makedirs(self, path: str) -> None: ...

    def rename(self, source: str, target: str) -> None:
        """Atomic replace of `target`."""

    def read(self, path: str) -> bytes | None: ...

    def write(self, path: str, data: bytes) -> None:
        """Atomic: a temporary file renamed over `path`."""

    def delete(self, path: str) -> None:
        """Remove a file or directory tree; missing is not an error."""

    def exists(self, path: str) -> bool: ...

    def stat(self, path: str) -> Stat | None: ...


@dataclass(frozen=True)
class FetchSummary:
    files: int            # files listed
    fetched: int          # files copied this time (the rest were already here)
    bytes: int            # bytes copied this time


class FileTransfer(Protocol):
    def sync(self, local: str, remote: str) -> int:
        """Resumable one-way copy of `local` to `remote`; returns the bytes
        transferred."""

    def fetch(self, remote_root: str, paths: list[str], local_root: str) -> FetchSummary:
        """Resumable copy of `paths` (relative to `remote_root`) from the
        cluster into the same relative paths under `local_root`; files
        already here with the same size and modification time are skipped."""


class SubmissionFailed(HpcError):
    """The scheduler definitely refused the job (nothing was submitted)."""


class SubmissionUncertain(HpcError):
    """The outcome is unknown (e.g. a lost connection): the job may exist."""


@dataclass(frozen=True)
class CancelResult:
    ok: bool
    detail: str


class SchedulerSubmitter(Protocol):
    def submit(self, node: JobNode, script_path: str,
               dependencies: list[str]) -> str:
        """Submit one staged node script after `dependencies` (job
        references); return its job reference, or raise SubmissionFailed /
        SubmissionUncertain."""

    def cancel(self, job_ref: str) -> CancelResult: ...


@dataclass(frozen=True)
class Owner:
    """Who holds a lease: a scheduler job, or a process on a host."""

    job_ref: str | None = None
    host: str | None = None
    pid: int | None = None
    start_time: str | None = None

    def as_json(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class LeaseOwnerProbe(Protocol):
    def is_alive(self, owner: Owner) -> bool | None:
        """True or False, or None when this probe cannot tell (e.g. a process
        owner on another host)."""
