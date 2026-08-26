import os
import signal
import time
import traceback

from torch import multiprocessing

from training_framework.session import Session, TrainingSession


def load_session_for_worker(
        session_state,
        rank,
        session_update_params: dict | None = None,
):
    session = Session.from_state(session_state)

    if (
            session_update_params is not None
            and "max_iterations" in session_update_params
    ):
        if not isinstance(session, TrainingSession):
            raise TypeError("max_iterations updates require a TrainingSession")
        session.update_max_iters(session_update_params["max_iterations"])

    if session.has_resource("ddp"):
        placeholder_ddp_resource = session.get_resource("ddp")
        ddp_resource = type(placeholder_ddp_resource)(
            config=placeholder_ddp_resource.config,
            rank=rank,
        )
        session.unregister_resource(placeholder_ddp_resource.name)
        session.register_resource(ddp_resource)

        if rank > 0:
            parallel_components = session._component_dependency_closure(
                ddp_resource.parallel_components + ["ddp"]
            )
            hooks = session.get_all_hooks()
            resources = session.get_all_resources()
            steps = session.get_all_steps()
            for hook in hooks:
                if hook.name not in parallel_components:
                    session.unregister_hook(hook.name)
            for resource in resources:
                if resource.name not in parallel_components:
                    session.unregister_resource(resource.name)
            for step in steps:
                if step.name not in parallel_components:
                    session.remove_step(step.name)

    return session


def session_process_worker(
        session_state,
        rank: int,
        stop_event,
        error_conn,
        **kwargs,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        session = load_session_for_worker(
            session_state,
            rank,
            session_update_params=kwargs.get("session_update_params"),
        )
        session.set_dist_manager_err_conn(error_conn)
        session.set_heartbeat_interval(10.0)
        with session:
            while not stop_event.is_set():
                try:
                    next(session)
                except StopIteration:
                    break
    except BaseException as error:
        try:
            error_conn.send({
                "type": "error",
                "rank": rank,
                "pid": os.getpid(),
                "exception_type": (
                    f"{type(error).__module__}."
                    f"{type(error).__qualname__}"
                ),
                "message": str(error),
                "traceback": traceback.format_exc(),
            })
        finally:
            error_conn.close()
        raise
    else:
        error_conn.close()
    print(f"Session rank[{rank}] exiting.", flush=True)


class SessionProcessWrapper:
    def __init__(
            self,
            session: Session,
            rank: int,
            **kwargs,
    ):
        self._session = session
        self._rank = rank

        context = multiprocessing.get_context("spawn")
        self._stop_event = context.Event()
        self._recv_conn, self._send_conn = context.Pipe(duplex=False)
        self._session_process = context.Process(
            name=f"training-session-rank-{rank}",
            target=session_process_worker,
            args=(
                self._session.get_state(),
                rank,
                self._stop_event,
                self._send_conn,
            ),
            kwargs=kwargs,
        )
        self._started = False
        self._heartbeat_timeout = kwargs["heartbeat_timeout"]
        self._deadline = time.monotonic() + self._heartbeat_timeout

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
                f"Session rank[{self._rank}] has already been started"
            )
        self._session_process.start()
        self._send_conn.close()
        self._started = True

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self):
        self._session_process.join()
        self._recv_conn.close()
