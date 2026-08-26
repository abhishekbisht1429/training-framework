import time
from collections.abc import Callable
from multiprocessing.connection import wait
from typing import Any


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
                if isinstance(message, dict):
                    if message.get("type") == "heartbeat":
                        print(
                            f"Heartbeat received from Process rank "
                            f"{wrapper.rank}. - {message}"
                        )
                        wrapper.reset_deadline()
                    else:
                        waitables.pop(ready_waitable, None)
                        ready_waitable.close()
                        failure = RuntimeError(
                            f"Worker pid={wrapper.process.pid} failed:\n"
                            f"{message}"
                        )
                else:
                    print(f"Unknown message type received! {message}")
            except EOFError:
                waitables.pop(ready_waitable, None)
                ready_waitable.close()
        elif ready_waitable_type == "sentinel":
            waitables.pop(ready_waitable, None)
            wrapper.process.join()

            if wrapper.process.exitcode != 0:
                while (
                        not wrapper.error_conn.closed
                        and wrapper.error_conn.poll()
                ):
                    try:
                        message = wrapper.error_conn.recv()
                    except EOFError:
                        break

                    if message.get("type") == "error":
                        failure = RuntimeError(
                            f"Worker pid={wrapper.process.pid} failed:\n{message}"
                            if message is not None
                            else (
                                f"Worker pid={wrapper.process.pid} failed with "
                                f"exit code {wrapper.process.exitcode}"
                            )
                        )
            waitables.pop(wrapper.error_conn, None)
            if not wrapper.error_conn.closed:
                wrapper.error_conn.close()
        else:
            raise RuntimeError("Unknown ready waitable type!")

    return failure


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

            timeout = max(
                0.0,
                min(wrapper.deadline for wrapper in active) - time.monotonic(),
            )
            ready = wait(waitables, timeout=timeout)

            if ready:
                failure = process_ready(waitables, ready)

            now = time.monotonic()
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
                    f"Worker pid={wrapper.process.pid} missed its "
                    f"heartbeat deadline by "
                    f"{now - wrapper.deadline:.1f}s"
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
