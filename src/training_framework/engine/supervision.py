import time
from collections.abc import Callable, Mapping
from multiprocessing.connection import wait
from typing import Any

from training_framework.engine.failures import (
    failure_from_report,
    failure_without_report,
)
from training_framework.session.failure_report import FAILURE_REPORT_TYPE


_MAX_PROGRESS_POLL_INTERVAL = 1.0
_MAX_STATUS_INTERVAL = 10.0


def join_or_terminate(wrappers: list, timeout: float) -> None:
    print(
        f"Waiting up to {timeout:.1f}s for "
        f"{len(wrappers)} worker process(es) to exit gracefully.",
        flush=True,
    )

    deadline = time.monotonic() + timeout

    for wrapper in wrappers:
        process = wrapper.process
        remaining = max(0.0, deadline - time.monotonic())

        print(
            f"Joining rank={wrapper.rank}, pid={process.pid}, "
            f"remaining_timeout={remaining:.2f}s.",
            flush=True,
        )
        process.join(timeout=remaining)
        print(
            f"Join completed for rank={wrapper.rank}, pid={process.pid}, "
            f"alive={process.is_alive()}, exitcode={process.exitcode}.",
            flush=True,
        )

    survivors = [
        wrapper for wrapper in wrappers if wrapper.process.is_alive()
    ]
    print(
        f"Graceful shutdown complete. "
        f"{len(survivors)} worker process(es) still alive.",
        flush=True,
    )

    for wrapper in survivors:
        print(
            f"Terminating rank={wrapper.rank}, pid={wrapper.process.pid}.",
            flush=True,
        )
        wrapper.process.terminate()

    for wrapper in survivors:
        print(
            f"Waiting for terminated rank={wrapper.rank}, "
            f"pid={wrapper.process.pid}.",
            flush=True,
        )
        wrapper.process.join(timeout=1.0)

    stubborn = [
        wrapper for wrapper in survivors if wrapper.process.is_alive()
    ]
    print(
        f"Termination phase complete. "
        f"{len(stubborn)} worker process(es) still alive.",
        flush=True,
    )

    for wrapper in stubborn:
        print(
            f"Killing rank={wrapper.rank}, pid={wrapper.process.pid}.",
            flush=True,
        )
        wrapper.process.kill()

    for wrapper in stubborn:
        print(
            f"Waiting for killed rank={wrapper.rank}, "
            f"pid={wrapper.process.pid}.",
            flush=True,
        )
        wrapper.process.join(timeout=1.0)
        print(
            f"Final state for rank={wrapper.rank}, "
            f"pid={wrapper.process.pid}, "
            f"alive={wrapper.process.is_alive()}, "
            f"exitcode={wrapper.process.exitcode}.",
            flush=True,
        )

    print("Worker shutdown sequence finished.", flush=True)


def process_ready_waitables(waitables, ready_waitables):
    """Read what is ready; return the first worker failure, or None.

    A worker reports a failure on its error connection before it exits;
    one that exits with a non-zero code and no report (killed, or exited
    without raising) is a failure too. Only the first failure is kept.
    """
    failure = None
    ready_waitables.sort(key=lambda key: waitables[key][0] == "sentinel")

    for ready_waitable in ready_waitables:
        entry = waitables.get(ready_waitable)
        if entry is None:
            continue

        ready_waitable_type, wrapper = entry

        if ready_waitable_type == "connection":
            try:
                message = ready_waitable.recv()
                if _is_failure_report(message):
                    waitables.pop(ready_waitable, None)
                    ready_waitable.close()
                    failure = failure or failure_from_report(
                        message,
                        pid=wrapper.process.pid,
                        rank=wrapper.rank,
                    )
                else:
                    print(f"Unknown message type received! {message}")
            except EOFError:
                waitables.pop(ready_waitable, None)
                ready_waitable.close()
        elif ready_waitable_type == "sentinel":
            waitables.pop(ready_waitable, None)
            wrapper.process.join()
            exitcode = wrapper.process.exitcode

            if exitcode != 0:
                report = None
                while (
                        not wrapper.error_conn.closed
                        and wrapper.error_conn.poll()
                ):
                    try:
                        message = wrapper.error_conn.recv()
                    except EOFError:
                        break
                    if report is None and _is_failure_report(message):
                        report = message
                failure = failure or (
                    failure_from_report(
                        report,
                        pid=wrapper.process.pid,
                        rank=wrapper.rank,
                        exitcode=exitcode,
                    )
                    if report is not None else
                    failure_without_report(
                        pid=wrapper.process.pid,
                        rank=wrapper.rank,
                        exitcode=exitcode,
                    )
                )
            waitables.pop(wrapper.error_conn, None)
            if not wrapper.error_conn.closed:
                wrapper.error_conn.close()
        else:
            raise RuntimeError("Unknown ready waitable type!")

    return failure


def _is_failure_report(message) -> bool:
    return (
        isinstance(message, Mapping)
        and message.get("type") == FAILURE_REPORT_TYPE
    )


def _report_progress(wrapper, now: float, last_status_times: dict) -> None:
    status_interval = min(
        _MAX_STATUS_INTERVAL,
        wrapper.heartbeat_timeout / 3,
    )
    last_status_time = last_status_times.get(wrapper.rank)
    if (
            last_status_time is not None
            and now - last_status_time < status_interval
    ):
        return
    last_status_times[wrapper.rank] = now
    print(
        f"Worker rank={wrapper.rank}: iteration {wrapper.last_iteration}, "
        f"stage {wrapper.last_stage!r}",
        flush=True,
    )


def monitor_processes(
        wrappers: list,
        *,
        process_ready: Callable[[dict, list], Any],
        request_stop_all: Callable[[], None],
        shutdown: Callable[..., None],
        process_timeout_on_join: float,
) -> None:
    waitables = {}
    for wrapper in wrappers:
        waitables[wrapper.error_conn] = ("connection", wrapper)
        waitables[wrapper.process.sentinel] = ("sentinel", wrapper)

    failure = None
    interrupted = False
    last_status_times = {}
    try:
        while waitables:
            if failure:
                break

            active = [
                wrapper
                for waitable_type, wrapper in waitables.values()
                if waitable_type == "sentinel"
            ]
            if not active:
                break

            poll_interval = min(
                _MAX_PROGRESS_POLL_INTERVAL,
                min(wrapper.heartbeat_timeout for wrapper in active) / 10,
            )
            timeout = max(
                0.0,
                min(
                    poll_interval,
                    min(wrapper.deadline for wrapper in active)
                    - time.monotonic(),
                ),
            )
            ready = wait(waitables, timeout=timeout)

            if ready:
                failure = process_ready(waitables, ready)

            now = time.monotonic()
            for wrapper in active:
                if wrapper.check_progress(now):
                    _report_progress(wrapper, now, last_status_times)
            timed_out_wrappers = [
                wrapper
                for wrapper in active
                if wrapper.deadline <= now and wrapper.process.is_alive()
            ]
            if timed_out_wrappers:
                wrapper = min(
                    timed_out_wrappers,
                    key=lambda item: item.deadline,
                )
                failure = TimeoutError(
                    f"Worker rank={wrapper.rank} pid={wrapper.process.pid} "
                    f"made no progress for "
                    f"{now - wrapper.last_progress_time:.1f}s "
                    f"(iteration {wrapper.last_iteration}, "
                    f"stage {wrapper.last_stage!r})"
                )
    except KeyboardInterrupt:
        print("Interrupted!")
        interrupted = True

    if interrupted or failure:
        request_stop_all()
        active = [
            wrapper
            for waitable_type, wrapper in waitables.values()
            if waitable_type == "sentinel"
        ]
        shutdown(active, timeout=process_timeout_on_join)

    if failure:
        raise failure
