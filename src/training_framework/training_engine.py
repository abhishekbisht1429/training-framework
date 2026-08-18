import collections
import traceback
from multiprocessing.connection import wait

from torch import multiprocessing
import signal
import time
from typing import Iterator
import os

from training_framework.builtin_components import Checkpointer, DDPResource
from training_framework.configurator import Configurator
from training_framework.training_session import TrainingSession
from training_framework.util import context_entry, context_exit, requires_context


def load_session_for_worker(session_state, rank, session_update_params: dict | None=None):
    # session = Checkpointer.load_checkpoint(path=session)
    # session = deepcopy(session)
    session = TrainingSession.from_state(session_state)

    if session_update_params is not None:
        if "max_iterations" in session_update_params:
            session.update_max_iters(session_update_params["max_iterations"])

    if session.has_resource("ddp"):
        placeholder_ddp_resource: DDPResource = session.get_resource("ddp")
        ddp_resource = DDPResource(
            config=placeholder_ddp_resource.config,
            rank=rank
        )
        session.unregister_resource(placeholder_ddp_resource.name)
        session.register_resource(ddp_resource)

        # for processes with rank > 0, remove the non-parallel components (they should run only on rank 0)
        if rank > 0:
            parallel_components = set(ddp_resource.parallel_components + ["ddp"])
            for hook in session.get_all_hooks():
                if hook.name not in parallel_components:
                    session.unregister_hook(hook.name)
            for resource in session.get_all_resources():
                if resource.name not in parallel_components:
                    session.unregister_resource(resource.name)
            for step in session.get_all_steps():
                if step.name not in parallel_components:
                    session.remove_step(step.name)

    return session


def session_process_worker(
    session_state,
    # session_id: int,
    rank: int,
    stop_event,
    error_conn,
    **kwargs
) -> None:
    # The parent coordinates graceful interrupt handling.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        session_update_params = kwargs["session_update_params"] if "session_update_params" in kwargs else None
        # important so that model doesn't share the tensors between processes
        session = load_session_for_worker(session_state, rank, session_update_params=session_update_params)
        session.set_dist_manager_err_conn(error_conn)
        session.set_heartbeat_interval(10.0)
        with session:
            # try:
            while not stop_event.is_set():
                try:
                    next(session)
                except StopIteration:
                    break
            # except KeyboardInterrupt:
            #     # print(f"Session {session_id}[{rank}] is interrupted.")
            #     print(f"Session rank[{rank}] is interrupted.")

        # print(f"Session {session_id}[{rank}] exiting.", flush=True)
    except BaseException as exc:
        try:
            error_conn.send(
                {
                    'type': 'error',
                    "rank": rank,
                    "pid": os.getpid(),
                    "exception_type": (
                        f"{type(exc).__module__}."
                        f"{type(exc).__qualname__}"
                    ),
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            error_conn.close()

            # Critical: preserve a non-zero process exit code.
        raise
    else:
        # Normal worker completion. No success message is required.
        error_conn.close()
    print(f"Session rank[{rank}] exiting.", flush=True)

class SessionProcessWrapper:
    def __init__(self,
                 session: TrainingSession,
                 # session_id: int,
                 rank: int,
                 **kwargs
                 ):
        # self._worker_config = deepcopy(worker_config)
        self._session = session
        # self._session_id = session_id
        self._rank = rank

        ctx = multiprocessing.get_context("spawn")

        self._stop_event = ctx.Event()

        self._recv_conn, self._send_conn = ctx.Pipe(duplex=False)

        self._session_process = ctx.Process(
            # name=f"training-session-{session_id}-rank-{rank}",
            name=f"training-session-rank-{rank}",
            target=session_process_worker,
            args=(
                self._session.get_state(),
                # session_id,
                rank,
                self._stop_event,
                self._send_conn
            ),
            kwargs=kwargs
        )
        self._started = False

        self._heartbeat_timeout = kwargs['heartbeat_timeout']
        self._deadline = time.monotonic() + self._heartbeat_timeout

    # @property
    # def session_id(self) -> int:
    #     return self._session_id
    @property
    def error_conn(self):
        return self._recv_conn

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def process(self):
        return self._session_process

    @property
    def started(self) -> bool:
        return self._started

    @property
    def deadline(self) -> float:
        return self._deadline

    def reset_deadline(self):
        self._deadline = time.monotonic() + self._heartbeat_timeout

    def start(self) -> None:
        if self._started:
            raise RuntimeError(
                f"Session {self._session_id}[{self._rank}] "
                "has already been started"
            )

        self._session_process.start()
        self._send_conn.close()
        self._started = True

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self):
        if self._session_process.alive():
            if self._recv_conn.poll():
                try:
                    error = self._recv_conn.recv()
                    print(error)
                except EOFError:
                    print(f"Session {self.session_id}[{self.rank}] finished successfully.")
                    pass
            else:
                print(f"Session {self.session_id}[{self.rank}] failed with exit code {self._session_process.exitcode}")
        self._recv_conn.close()



class TrainingEngine:
    def __init__(self, configurator: Configurator):
        self._configurator = configurator
        self._timeout_on_interrupt = configurator.process_timeout_on_join
        # self._session_processes: list[list[SessionProcessWrapper]] = []
        self._session_process_wrappers: list[SessionProcessWrapper] = []
        # self._sentinels: list = []

    # def _iter_wrappers(self) -> Iterator[SessionProcessWrapper]:
    #     for wrapper_list in self._session_processes:
    #         yield from wrapper_list

    def load_session(self, checkpoint_path: str, session_update_params: dict | None=None):
        # session_id = len(self._session_processes)

        # load checkpoint here to get ddp info
        session = Checkpointer.load_checkpoint(checkpoint_path)

        if session.has_resource("ddp"):
            world_size = session.get_resource("ddp").world_size
        else:
            world_size = 1

        self._session_process_wrappers = [
            SessionProcessWrapper(
                session=session,
                # session_id=session_id,
                rank=rank,
                session_update_params=session_update_params,
                heartbeat_timeout=self._configurator.heartbeat_timeout
            )
            for rank in range(world_size)
        ]

        # self._session_processes.append(wrappers)
        # self._session_process_wrappers.append(wrappers)

        # return session_id

    def register_session(self, config: dict) -> int:
        if not isinstance(config, collections.abc.Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}"
            )

        # session_id = len(self._session_processes)

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

        # worker_config = { "session_config": config }
        self._session_process_wrappers = [
            SessionProcessWrapper(
                session=TrainingSession(config),
                # worker_config=worker_config,
                # session_id=session_id,
                rank=rank,
                heartbeat_timeout=self._configurator.heartbeat_timeout
            )
            for rank in range(world_size)
        ]

        # self._session_processes.append(wrappers)
        # self._session_process_wrappers.append(wrappers)

        # return session_id

    @requires_context
    def start_session(self) -> None:
        started: list[SessionProcessWrapper] = []

        try:
            for wrapper in self._session_process_wrappers:
                wrapper.start()
                started.append(wrapper)
        except BaseException:
            for wrapper in started:
                wrapper.request_stop()
            self._join_or_terminate(started, timeout=self._timeout_on_interrupt)
            raise

    # @requires_context
    # def start_all(self) -> None:
    #     # for session_id in range(len(self._session_processes)):
    #     #     self.start_session(session_id)

    def request_stop_all(self) -> None:
        for wrapper in self._session_process_wrappers:
            if wrapper.started:
                wrapper.request_stop()

    def _join_or_terminate(self, wrappers: list[SessionProcessWrapper] | None = None, timeout: float=5.0) -> None:
        selected = (
            wrappers
            if wrappers is not None
            else [
                wrapper
                for wrapper in self._session_process_wrappers
                if wrapper.started
            ]
        )

        print(
            f"Waiting up to {timeout:.1f}s for "
            f"{len(selected)} worker process(es) to exit gracefully.",
            flush=True,
        )

        deadline = time.monotonic() + timeout

        # Give all workers one shared graceful-shutdown period.
        for wrapper in selected:
            process = wrapper.process
            remaining = max(0.0, deadline - time.monotonic())

            print(
                # f"Joining session={wrapper.session_id}, "
                f"Joining "
                f"rank={wrapper.rank}, "
                f"pid={process.pid}, "
                f"remaining_timeout={remaining:.2f}s.",
                flush=True,
            )

            process.join(timeout=remaining)

            print(
                # f"Join completed for session={wrapper.session_id}, "
                f"Join completed for "
                f"rank={wrapper.rank}, "
                f"pid={process.pid}, "
                f"alive={process.is_alive()}, "
                f"exitcode={process.exitcode}.",
                flush=True,
            )

        survivors = [
            wrapper
            for wrapper in selected
            if wrapper.process.is_alive()
        ]

        print(
            f"Graceful shutdown complete. "
            f"{len(survivors)} worker process(es) still alive.",
            flush=True,
        )

        # Terminate all survivors before waiting again.
        for wrapper in survivors:
            print(
                # f"Terminating session={wrapper.session_id}, "
                f"Terminating "
                f"rank={wrapper.rank}, "
                f"pid={wrapper.process.pid}.",
                flush=True,
            )

            wrapper.process.terminate()

        for wrapper in survivors:
            print(
                # f"Waiting for terminated session={wrapper.session_id}, "
                f"Waiting for terminated "
                f"rank={wrapper.rank}, "
                f"pid={wrapper.process.pid}.",
                flush=True,
            )

            wrapper.process.join(timeout=1.0)

        stubborn = [
            wrapper
            for wrapper in survivors
            if wrapper.process.is_alive()
        ]

        print(
            f"Termination phase complete. "
            f"{len(stubborn)} worker process(es) still alive.",
            flush=True,
        )

        for wrapper in stubborn:
            print(
                # f"Killing session={wrapper.session_id}, "
                f"Killing "
                f"rank={wrapper.rank}, "
                f"pid={wrapper.process.pid}.",
                flush=True,
            )

            wrapper.process.kill()

        for wrapper in stubborn:
            print(
                # f"Waiting for killed session={wrapper.session_id}, "
                f"Waiting for killed "
                f"rank={wrapper.rank}, "
                f"pid={wrapper.process.pid}.",
                flush=True,
            )

            wrapper.process.join(timeout=1.0)

            print(
                # f"Final state for session={wrapper.session_id}, "
                f"Final state for "
                f"rank={wrapper.rank}, "
                f"pid={wrapper.process.pid}, "
                f"alive={wrapper.process.is_alive()}, "
                f"exitcode={wrapper.process.exitcode}.",
                flush=True,
            )

        print("Worker shutdown sequence finished.", flush=True)

    # def _raise_worker_failures(self) -> None:
    #     failures = []
    #
    #     for wrapper in self._iter_wrappers():
    #         process = wrapper.process
    #
    #         if (
    #             wrapper.started
    #             and process.exitcode not in (0, None)
    #         ):
    #             failures.append(
    #                 f"session={wrapper.session_id}, "
    #                 f"rank={wrapper.rank}, "
    #                 f"exitcode={process.exitcode}"
    #             )
    #
    #     if failures:
    #         raise RuntimeError(
    #             "One or more training workers failed: "
    #             + "; ".join(failures)
    #         )

    def _close_resources(self) -> None:
        for wrapper in self._session_process_wrappers:
            process = wrapper.process

            if wrapper.started and not process.is_alive():
                process.close()

    @context_entry
    def __enter__(self):
        if self._configurator.mode == "new":
            for session_config in self._configurator.session_configs:
                self.register_session(session_config)
        elif self._configurator.mode == "extend":
            self.load_session(
                checkpoint_path=self._configurator.checkpoint_path,
                session_update_params={"max_iterations": self._configurator.new_max_iters}
            )
        elif self._configurator.mode == "resume":
            self.load_session(self._configurator.checkpoint_path)
        else:
            raise RuntimeError("Invalid mode!")

        return self

    def _process_ready_waitables(self, waitables, ready_waitables):
        # Read messages before handling process exits.
        failure = None

        ready_waitables.sort(key=lambda key: waitables[key][0] == "sentinel")

        for ready_waitable in ready_waitables:
            entry = waitables.get(ready_waitable)
            if entry is None:
                continue

            ready_waitable_type, wrapper = entry

            if ready_waitable_type == "connection":
                # handle connection waitable
                try:
                    message = ready_waitable.recv()
                    if isinstance(message, dict):
                        if message.get('type', None) == 'heartbeat':
                            print(f"Heartbeat received from Process rank {wrapper.rank}. - {message}")
                            wrapper.reset_deadline()
                        else:
                            # treat all messages other than heartbeat as errors
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
                # handle sentinel waitables
                waitables.pop(ready_waitable, None)
                # Sentinel is ready, so join returns immediately.
                wrapper.process.join()

                if wrapper.process.exitcode != 0:
                    # Drain any queued error after an abnormal exit.
                    while not wrapper.error_conn.closed and wrapper.error_conn.poll():
                        try:
                            message = wrapper.error_conn.recv()
                        except EOFError:
                            break

                        # get the last error message sent
                        if message.get('type', None) == 'error':
                            failure = RuntimeError(
                                f"Worker pid={wrapper.process.pid} failed:\n{message}"
                                if message is not None
                                else (
                                    f"Worker pid={wrapper.process.pid} failed with "
                                    f"exit code {wrapper.process.exitcode}"
                                )
                            )
                # remove the connection for this process from waitables as it is already finished.
                waitables.pop(wrapper.error_conn, None)
                if not wrapper.error_conn.closed:
                    wrapper.error_conn.close()
            else:
                raise RuntimeError("Unknown ready waitable type!")

        return failure

    def _monitor_processes(self):
        waitables = {}
        failure = None

        # Watch worker messages and process termination.
        for wrapper in self._session_process_wrappers:
            waitables[wrapper.error_conn] = ("connection", wrapper)
            waitables[wrapper.process.sentinel] = ("sentinel", wrapper)

        interrupt_time = None
        while waitables:
            if failure:
                break
            elif interrupt_time and (time.monotonic() - interrupt_time) > self._configurator.wait_time_after_interrupt:
                break

            try:
                active = [wrapper for waitable_type, wrapper in waitables.values() if waitable_type == "sentinel"]
                if not active:
                    break

                timeout = max(0.0, min(wrapper.deadline for wrapper in active) - time.monotonic())
                ready = wait(waitables, timeout=timeout)

                if ready:
                    failure = self._process_ready_waitables(waitables, ready)

                now = time.monotonic()
                timed_out_wrappers = [
                    wrapper
                    for wrapper in active
                    if wrapper.deadline <= now and wrapper.process.is_alive()
                ]
                if timed_out_wrappers:
                    # wait timed out
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
                print("Interrupted! Requesting proccesses to stop.")
                self.request_stop_all()
                interrupt_time = time.monotonic()

        if interrupt_time or failure:
            active = [wrapper for waitable_type, wrapper in waitables.values() if waitable_type == "sentinel"]
            self._join_or_terminate(active, timeout=self._configurator.process_timeout_on_join)


        return failure

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                failure = self._monitor_processes()
                if failure:
                    raise failure
            else:
                # The parent failed, so stop all workers.
                self.request_stop_all()
                self._join_or_terminate(
                    self._session_process_wrappers,
                    timeout=self._timeout_on_interrupt,
                )
        finally:
            self._close_resources()

        # Never suppress an exception from the with block.
        return False