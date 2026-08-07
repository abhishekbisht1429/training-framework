import collections
import signal
import time
from multiprocessing import Event, Process
from typing import Iterator

from training_framework.configurator import create_session_from_config
from training_framework.util import context_entry, context_exit, requires_context


def session_process_worker(
    config: dict,
    session_id: int,
    rank: int,
    stop_event: Event,
) -> None:
    # The parent coordinates graceful interrupt handling.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    session = create_session_from_config(config, rank=rank)

    with session:
        while not stop_event.is_set():
            try:
                next(session)
            except StopIteration:
                break

    print(f"Session {session_id}[{rank}] exiting.", flush=True)


class SessionProcessWrapper:
    def __init__(self, config: dict, session_id: int, rank: int):
        self._config = dict(config)
        self._session_id = session_id
        self._rank = rank
        self._stop_event = Event()
        self._started = False

        self._session_process = Process(
            name=f"training-session-{session_id}-rank-{rank}",
            target=session_process_worker,
            args=(
                self._config,
                session_id,
                rank,
                self._stop_event,
            ),
        )

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def process(self) -> Process:
        return self._session_process

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError(
                f"Session {self._session_id}[{self._rank}] "
                "has already been started"
            )

        self._session_process.start()
        self._started = True

    def request_stop(self) -> None:
        self._stop_event.set()


class TrainingEngine:
    def __init__(self, config: dict):
        self._config = config
        self._session_processes: list[list[SessionProcessWrapper]] = []

    def _iter_wrappers(self) -> Iterator[SessionProcessWrapper]:
        for wrapper_list in self._session_processes:
            yield from wrapper_list

    def register_session(self, config: dict) -> int:
        if not isinstance(config, collections.abc.Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}"
            )

        session_id = len(self._session_processes)

        if "ddp" in config:
            try:
                world_size = config["ddp"]["world_size"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "DDP configuration must contain ddp.world_size"
                ) from exc
        else:
            world_size = 1

        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size < 1
        ):
            raise ValueError("ddp.world_size must be a positive integer")

        wrappers = [
            SessionProcessWrapper(
                config=dict(config),
                session_id=session_id,
                rank=rank,
            )
            for rank in range(world_size)
        ]

        self._session_processes.append(wrappers)

        return session_id

    @requires_context
    def start_session(self, session_id: int) -> None:
        wrappers = self._session_processes[session_id]
        started: list[SessionProcessWrapper] = []

        try:
            for wrapper in wrappers:
                wrapper.start()
                started.append(wrapper)
        except BaseException:
            for wrapper in started:
                wrapper.request_stop()

            self._join_or_terminate(started, timeout=5.0)
            raise

    @requires_context
    def start_all(self) -> None:
        for session_id in range(len(self._session_processes)):
            self.start_session(session_id)

    def request_stop_all(self) -> None:
        for wrapper in self._iter_wrappers():
            if wrapper.started:
                wrapper.request_stop()

    def _join_or_terminate(
        self,
        wrappers: list[SessionProcessWrapper] | None = None,
        *,
        timeout: float,
    ) -> None:
        selected = (
            wrappers
            if wrappers is not None
            else [
                wrapper
                for wrapper in self._iter_wrappers()
                if wrapper.started
            ]
        )

        deadline = time.monotonic() + timeout

        # Give all workers one shared graceful-shutdown period.
        for wrapper in selected:
            process = wrapper.process
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)

        survivors = [
            wrapper
            for wrapper in selected
            if wrapper.process.is_alive()
        ]

        # Terminate all survivors before waiting again.
        for wrapper in survivors:
            wrapper.process.terminate()

        for wrapper in survivors:
            wrapper.process.join(timeout=1.0)

        stubborn = [
            wrapper
            for wrapper in survivors
            if wrapper.process.is_alive()
        ]

        for wrapper in stubborn:
            wrapper.process.kill()

        for wrapper in stubborn:
            wrapper.process.join(timeout=1.0)

    def _raise_worker_failures(self) -> None:
        failures = []

        for wrapper in self._iter_wrappers():
            process = wrapper.process

            if (
                wrapper.started
                and process.exitcode not in (0, None)
            ):
                failures.append(
                    f"session={wrapper.session_id}, "
                    f"rank={wrapper.rank}, "
                    f"exitcode={process.exitcode}"
                )

        if failures:
            raise RuntimeError(
                "One or more training workers failed: "
                + "; ".join(failures)
            )

    def _close_resources(self) -> None:
        for wrapper in self._iter_wrappers():
            process = wrapper.process

            if wrapper.started and not process.is_alive():
                process.close()

    @context_entry
    def __enter__(self):
        return self

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        interrupted_during_join = False

        try:
            if exc_type is not None:
                # Preserve and propagate the original context-body exception,
                # but stop children first.
                self.request_stop_all()
                self._join_or_terminate(timeout=5.0)
            else:
                try:
                    # Normal behavior: wait for finite training to finish.
                    for wrapper in self._iter_wrappers():
                        if wrapper.started:
                            wrapper.process.join()
                except KeyboardInterrupt:
                    interrupted_during_join = True
                    self.request_stop_all()
                    self._join_or_terminate(timeout=5.0)

            if exc_type is None and not interrupted_during_join:
                self._raise_worker_failures()

        finally:
            self._close_resources()

        if interrupted_during_join:
            raise KeyboardInterrupt

        # Never suppress an exception from the with block.
        return False