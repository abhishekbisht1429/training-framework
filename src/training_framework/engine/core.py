import collections
from collections.abc import Mapping
from copy import deepcopy

from training_framework.components.builtin import Checkpointer
from training_framework.engine.config import Configurator
from training_framework.engine.supervision import (
    join_or_terminate,
    monitor_processes,
    process_ready_waitables,
)
from training_framework.engine.worker import SessionProcessWrapper
from training_framework.session import (
    TRAINING_SESSION_TYPE,
    Session,
    normalize_session_type,
    session_class_for_type,
)
from training_framework.util import context_entry, context_exit, requires_context


class TrainingEngine:
    def __init__(self, configurator: Configurator):
        self._configurator = configurator
        self._timeout_on_interrupt = configurator.process_timeout_on_join
        self._session_process_wrappers: list[SessionProcessWrapper] = []

    def load_session(
            self,
            checkpoint_path: str,
            session_update_params: dict | None = None,
    ):
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
                heartbeat_timeout=self._configurator.heartbeat_timeout,
            )
            for rank in range(world_size)
        ]

    def register_session(
            self,
            config: dict,
            *,
            session_type: str = TRAINING_SESSION_TYPE,
            session_kwargs: Mapping | None = None,
    ) -> None:
        if not isinstance(config, collections.abc.Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}"
            )
        if session_kwargs is None:
            session_kwargs = {}
        if not isinstance(session_kwargs, Mapping):
            raise TypeError("session_kwargs must be a mapping")

        normalized_type = normalize_session_type(session_type)
        session_class = session_class_for_type(normalized_type)

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
                session=session_class(
                    deepcopy(dict(config)),
                    **deepcopy(dict(session_kwargs)),
                ),
                rank=rank,
                heartbeat_timeout=self._configurator.heartbeat_timeout,
            )
            for rank in range(world_size)
        ]
        self._session_process_wrappers.extend(wrappers)

    @staticmethod
    def _split_session_definition(
            definition: Mapping,
    ) -> tuple[dict, str, dict]:
        if not isinstance(definition, Mapping):
            raise TypeError("Each sessions entry must be a mapping")

        session_type = normalize_session_type(
            definition.get("session_type", TRAINING_SESSION_TYPE),
        )
        session_kwargs = definition.get("session_kwargs", {})
        if not isinstance(session_kwargs, Mapping):
            raise TypeError("'session_kwargs' must be a mapping")

        config = {
            key: deepcopy(value)
            for key, value in definition.items()
            if key not in {"session_type", "session_kwargs"}
        }
        return config, session_type, deepcopy(dict(session_kwargs))

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

    def _join_or_terminate(
            self,
            wrappers: list[SessionProcessWrapper] | None = None,
            timeout: float = 5.0,
    ) -> None:
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
            for definition in self._configurator.session_configs:
                config, session_type, session_kwargs = (
                    self._split_session_definition(definition)
                )
                self.register_session(
                    config,
                    session_type=session_type,
                    session_kwargs=session_kwargs,
                )
        elif self._configurator.mode == "extend":
            self.load_session(
                checkpoint_path=self._configurator.checkpoint_path,
                session_update_params={
                    "max_iterations": self._configurator.new_max_iters,
                },
            )
        elif self._configurator.mode == "resume":
            self.load_session(self._configurator.checkpoint_path)
        else:
            raise RuntimeError("Invalid operation!")

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

    def _join_started_processes(self) -> None:
        for wrapper in self._session_process_wrappers:
            if wrapper.started:
                wrapper.join()

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.request_stop_all()
                self._join_or_terminate(
                    self._session_process_wrappers,
                    timeout=self._timeout_on_interrupt,
                )
            elif getattr(self._configurator, "debug", False):
                self._join_started_processes()
            else:
                self._monitor_processes()
        finally:
            self._close_resources()
        return False
