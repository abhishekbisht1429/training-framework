import collections
import os

from training_framework.components.builtin import Checkpointer
from training_framework.engine.config import Configurator
from training_framework.engine.supervision import (
    join_or_terminate,
    monitor_processes,
    process_ready_waitables,
)
from training_framework.engine.worker import SessionProcessWrapper
from training_framework.session import (
    AnalysisSession,
    Session,
    SessionMode,
    TrainingSession,
    normalize_session_mode,
)
from training_framework.util import context_entry, context_exit, requires_context


class TrainingEngine:
    def __init__(self, configurator: Configurator):
        self._configurator = configurator
        self._timeout_on_interrupt = configurator.process_timeout_on_join
        self._session_process_wrappers: list[SessionProcessWrapper] = []

    def load_session(self, checkpoint_path: str, session_update_params: dict | None=None):
        session = Checkpointer.load_checkpoint(checkpoint_path)

        if session.has_resource("ddp"):
            world_size = session.get_resource("ddp").world_size
        else:
            world_size = 1

        self._session_process_wrappers = [
            SessionProcessWrapper(
                session=session,
                rank=rank,
                session_update_params=session_update_params,
                heartbeat_timeout=self._configurator.heartbeat_timeout
            )
            for rank in range(world_size)
        ]

    def register_session(
            self,
            config: dict,
            *,
            session_mode: SessionMode | str = SessionMode.TRAINING,
            model_checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(config, collections.abc.Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}"
            )

        mode = normalize_session_mode(session_mode)

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

        def create_session() -> Session:
            if mode is SessionMode.TRAINING:
                return TrainingSession(config)
            return AnalysisSession(
                config,
                model_checkpoint_path=model_checkpoint_path,
            )

        self._session_process_wrappers = [
            SessionProcessWrapper(
                session=create_session(),
                rank=rank,
                heartbeat_timeout=self._configurator.heartbeat_timeout
            )
            for rank in range(world_size)
        ]

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
        join_or_terminate(selected, timeout)

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
        elif self._configurator.mode == "analysis":
            for session_config in self._configurator.session_configs:
                self.register_session(
                    session_config,
                    session_mode=SessionMode.ANALYSIS,
                    model_checkpoint_path=self._configurator.model_checkpoint_path,
                )
        else:
            raise RuntimeError("Invalid mode!")

        return self

    def _process_ready_waitables(self, waitables, ready_waitables):
        return process_ready_waitables(waitables, ready_waitables)

    def _monitor_processes(self):
        monitor_processes(
            self._session_process_wrappers,
            process_ready=self._process_ready_waitables,
            request_stop_all=self.request_stop_all,
            shutdown=self._join_or_terminate,
            process_timeout_on_join=(
                self._configurator.process_timeout_on_join
            ),
        )

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self._monitor_processes()
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