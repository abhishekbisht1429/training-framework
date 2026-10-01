"""The error the engine raises when a worker process fails.

The worker's traceback travels as text. It is attached as the error's cause,
a `RemoteTraceback`, the way `concurrent.futures` does it: Python prints it
verbatim before the parent's own frames, so its `File "...", line N` lines
read -- and link in an IDE -- like any traceback.
"""

from __future__ import annotations

import signal
from collections.abc import Mapping
from typing import Any


class RemoteTraceback(Exception):
    """A worker's traceback, as the worker formatted it."""

    def __init__(self, header: str, text: str) -> None:
        super().__init__(header, text)
        self.header = header
        self.text = text

    def __str__(self) -> str:
        return f"{self.header}\n{self.text.rstrip()}"


class WorkerFailedError(RuntimeError):
    """A worker process failed.

    `str()` names the worker and what it raised; the worker's traceback is
    `worker_traceback`, and also this error's `__cause__`, so it is printed
    with the parent's traceback. Without a report from the worker -- killed
    by a signal, or exited without raising -- `exitcode` says how it ended
    and the exception fields are None.
    """

    def __init__(
            self,
            summary: str,
            *,
            rank: int | None,
            pid: int | None,
            exitcode: int | None = None,
            exception_type: str | None = None,
            worker_message: str | None = None,
            worker_traceback: str | None = None,
    ) -> None:
        super().__init__(summary)
        self.rank = rank
        self.pid = pid
        self.exitcode = exitcode
        self.exception_type = exception_type
        self.worker_message = worker_message
        self.worker_traceback = worker_traceback
        if worker_traceback:
            self.__cause__ = RemoteTraceback(
                f"Traceback of worker rank {rank} (pid {pid}):",
                worker_traceback,
            )


def _worker(pid: int | None, rank: int | None) -> str:
    return f"Worker pid={pid} (rank {rank})"


def failure_from_report(
        report: Mapping[str, Any],
        *,
        pid: int | None,
        rank: int | None,
        exitcode: int | None = None,
) -> WorkerFailedError:
    """The error for a worker that reported `report` before it ended.

    `pid` and `rank` are what the parent knows of the worker; the report's
    own are used when it has them.
    """
    pid = report.get("pid", pid)
    rank = report.get("rank", rank)
    exception_type = report.get("exception_type")
    message = report.get("message")
    raised = exception_type or "an exception"
    summary = f"{_worker(pid, rank)} failed: {raised}"
    if message:
        summary = f"{summary}: {message}"
    return WorkerFailedError(
        summary,
        rank=rank,
        pid=pid,
        exitcode=exitcode,
        exception_type=exception_type,
        worker_message=message,
        worker_traceback=report.get("traceback"),
    )


def failure_without_report(
        *,
        pid: int | None,
        rank: int | None,
        exitcode: int,
) -> WorkerFailedError:
    """The error for a worker that ended with `exitcode` and reported
    nothing: killed by a signal (a negative exit code, as multiprocessing
    gives it -- SIGKILL is what the out-of-memory killer sends), or exited
    without raising."""
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = "an unknown signal"
        ended = f"was killed by signal {-exitcode} ({name})"
    else:
        ended = f"exited with code {exitcode}"
    return WorkerFailedError(
        f"{_worker(pid, rank)} failed: it {ended} without reporting an "
        "error",
        rank=rank,
        pid=pid,
        exitcode=exitcode,
    )


__all__ = [
    "RemoteTraceback",
    "WorkerFailedError",
    "failure_from_report",
    "failure_without_report",
]
