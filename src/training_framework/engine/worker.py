import os
import signal
import time
import traceback
from collections.abc import Mapping

import torch
from torch import distributed, multiprocessing

from training_framework.components.config import component_bindings_from_config
from training_framework.engine.topology import (
    LaunchTopology,
    pin_process_device,
)
from training_framework.session import Session, TrainingSession
from training_framework.session.components import SessionComponents
from training_framework.session.config import normalize_session_type
from training_framework.session.progress import ProgressBeacon
from training_framework.session.state import configuration_from_state
from training_framework.util import import_all_modules


_STOP_SYNC_GRACE_PERIOD = 0.01
_STOP_SYNC_POLL_INTERVAL = 0.005


def prepare_worker_state(
        session_state,
        rank: int,
        topology: LaunchTopology | None = None,
):
    """Return `session_state` adjusted for `rank`, before anything is built.

    Components are wired to each other as they are constructed, so a rank's
    component set and the rank-specific `ddp` configuration have to be settled
    first. Both are decided by data the state already carries: the dependency
    graph is class-level, so the closure a rank needs can be resolved without
    constructing a single component.

    `topology` is this launch's process topology. It overrides whatever the
    state carries, because a checkpoint's own topology describes the machine
    that wrote it and says nothing about this one.
    """
    components_state = session_state.get("components_state") or {}
    if not components_state:
        return session_state

    _, session_settings, _ = configuration_from_state(session_state)
    import_all_modules(session_settings["components_package"])
    components = SessionComponents(
        component_bindings=component_bindings_from_config(
            session_state["config"],
        ),
        session_type=normalize_session_type(session_state["session_type"]),
    )
    ddp_name = components.resolve_name("ddp")
    if ddp_name not in components_state:
        return session_state

    session_state = dict(session_state)
    components_state = {
        name: dict(info) for name, info in components_state.items()
    }
    session_state["components_state"] = components_state

    ddp_info = components_state[ddp_name]
    ddp_info["init_args"] = _topology_init_args(
        ddp_info["init_args"],
        rank,
        topology,
    )

    if topology is not None:
        # The session config is diffed against by a later --extend-session,
        # so it has to agree with the arguments the component was built from.
        session_state["config"] = _topology_config(
            session_state["config"],
            ddp_name,
            topology,
        )

    if rank > 0:
        ddp_config = _ddp_config(ddp_info["init_args"])
        keep = components.dependency_closure(
            list(ddp_config.get("parallel_components", [])) + ["ddp"],
            active_names=components_state,
        )
        for name in list(components_state):
            if name not in keep:
                del components_state[name]

    return session_state


def _topology_init_args(init_args, rank: int, topology) -> dict:
    """Settle the rank-specific and launch-specific `ddp` constructor args."""
    args = list(init_args["args"])
    kwargs = dict(init_args["kwargs"])
    kwargs["rank"] = rank

    if topology is not None:
        kwargs["local_rank"] = topology.local_rank(rank)
        overlay = topology.config_overlay()
        if args and isinstance(args[0], Mapping):
            args[0] = {**dict(args[0]), **overlay}
        elif isinstance(kwargs.get("config"), Mapping):
            kwargs["config"] = {**dict(kwargs["config"]), **overlay}

    return {"args": tuple(args), "kwargs": kwargs}


def _topology_config(config, ddp_name: str, topology) -> Mapping:
    if not isinstance(config, Mapping) or ddp_name not in config:
        return config
    ddp_config = config[ddp_name]
    if not isinstance(ddp_config, Mapping):
        return config

    patched = dict(config)
    patched[ddp_name] = {**dict(ddp_config), **topology.config_overlay()}
    return patched


def _ddp_config(init_args) -> dict:
    args = init_args["args"]
    if args and isinstance(args[0], Mapping):
        return dict(args[0])
    config = init_args["kwargs"].get("config")
    return dict(config) if isinstance(config, Mapping) else {}


def load_session_for_worker(
        session_state,
        rank,
        session_update_params: dict | None = None,
        launch_topology: LaunchTopology | None = None,
):
    # Before anything is constructed: no component can then be built or
    # restored against the wrong device, and the session's CUDA RNG stream
    # has a definite device to land on.
    pin_process_device(launch_topology, rank, session_state)

    session = Session.from_state(
        prepare_worker_state(session_state, rank, launch_topology)
    )

    if (
            session_update_params is not None
            and "max_iterations" in session_update_params
    ):
        if not isinstance(session, TrainingSession):
            raise TypeError("max_iterations updates require a TrainingSession")
        session.update_max_iters(session_update_params["max_iterations"])

    return session


def _stop_requested(
        session: Session,
        stop_event,
        *,
        stop_sync_grace_period: float = _STOP_SYNC_GRACE_PERIOD,
        stop_sync_poll_interval: float = _STOP_SYNC_POLL_INTERVAL,
) -> bool:
    local_stop_requested = stop_event.is_set()
    if not session.has_resource("ddp"):
        return local_stop_requested

    ddp_resource = session.get_resource("ddp")
    control_device = (
        session.device
        if ddp_resource.backend == "nccl"
        else torch.device("cpu")
    )
    stop_flag = torch.tensor(
        int(local_stop_requested),
        dtype=torch.int32,
        device=control_device,
    )
    session.send_heartbeat("Synchronizing worker stop state")
    stop_sync = distributed.all_reduce(
        stop_flag,
        op=distributed.ReduceOp.MAX,
        async_op=True,
    )
    poll_started = time.monotonic()
    while not stop_sync.is_completed():
        session.send_heartbeat("Synchronizing worker stop state")
        if time.monotonic() - poll_started >= stop_sync_grace_period:
            time.sleep(stop_sync_poll_interval)
    stop_sync.wait()
    return bool(stop_flag.item())


def session_process_worker(
        session_state,
        rank: int,
        stop_event,
        error_conn,
        progress_beacon,
        **kwargs,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    session = None
    try:
        session = load_session_for_worker(
            session_state,
            rank,
            session_update_params=kwargs.get("session_update_params"),
            launch_topology=kwargs.get("launch_topology"),
        )
        session.set_dist_manager_err_conn(error_conn)
        session.set_progress_beacon(progress_beacon)
        stop_sync_grace_period = kwargs.get(
            "stop_sync_grace_period",
            _STOP_SYNC_GRACE_PERIOD,
        )
        stop_sync_poll_interval = kwargs.get(
            "stop_sync_poll_interval",
            _STOP_SYNC_POLL_INTERVAL,
        )
        with session:
            while True:
                if _stop_requested(
                        session,
                        stop_event,
                        stop_sync_grace_period=stop_sync_grace_period,
                        stop_sync_poll_interval=stop_sync_poll_interval,
                ):
                    break
                try:
                    next(session)
                except StopIteration:
                    break
    except BaseException as error:
        try:
            # Session.__exit__ already reports failures raised while the
            # session was active; only report the rest (e.g. load or setup).
            if session is None or not session.worker_exception_reported:
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
        except OSError:
            # The parent already closed the pipe; re-raise the real error.
            pass
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
        self._progress_beacon = ProgressBeacon(context)
        self._session_process = context.Process(
            name=f"training-session-rank-{rank}",
            target=session_process_worker,
            args=(
                self._session.get_state(),
                rank,
                self._stop_event,
                self._send_conn,
                self._progress_beacon,
            ),
            kwargs=kwargs,
        )
        self._started = False
        self._heartbeat_timeout = kwargs["heartbeat_timeout"]
        self._deadline = time.monotonic() + self._heartbeat_timeout
        self._last_seq = 0
        self._last_iteration = 0
        self._last_stage = "Starting worker"
        self._last_progress_time = time.monotonic()

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

    @property
    def heartbeat_timeout(self) -> float:
        return self._heartbeat_timeout

    @property
    def last_iteration(self) -> int:
        return self._last_iteration

    @property
    def last_stage(self) -> str:
        return self._last_stage

    @property
    def last_progress_time(self) -> float:
        return self._last_progress_time

    def reset_deadline(self):
        self._deadline = time.monotonic() + self._heartbeat_timeout

    def check_progress(self, now: float | None = None) -> bool:
        seq, iteration, stage = self._progress_beacon.snapshot()
        if seq == self._last_seq:
            return False
        if now is None:
            now = time.monotonic()
        self._last_seq = seq
        self._last_iteration = iteration
        self._last_stage = stage
        self._last_progress_time = now
        self._deadline = now + self._heartbeat_timeout
        return True

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
