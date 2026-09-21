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
from training_framework.engine.topology import resolve_launch_topology
from training_framework.engine.worker import SessionProcessWrapper
from training_framework.engine.worker import (
    _STOP_SYNC_GRACE_PERIOD,
    _STOP_SYNC_POLL_INTERVAL,
)
from training_framework.session import (
    TRAINING_SESSION_TYPE,
    Session,
    TrainingSession,
    normalize_session_type,
    session_class_for_type,
)
from training_framework.util import context_entry, context_exit, requires_context


class TrainingEngine:
    def __init__(self, configurator: Configurator):
        self._configurator = configurator
        self._timeout_on_interrupt = configurator.process_timeout_on_join
        self._stop_sync_grace_period = getattr(
            configurator,
            "stop_sync_grace_period",
            _STOP_SYNC_GRACE_PERIOD,
        )
        self._stop_sync_poll_interval = getattr(
            configurator,
            "stop_sync_poll_interval",
            _STOP_SYNC_POLL_INTERVAL,
        )
        self._session_process_wrappers: list[SessionProcessWrapper] = []

    @property
    def _topology_overrides(self):
        return getattr(self._configurator, "topology_overrides", None)

    def _wrapper_kwargs(self, topology) -> dict:
        kwargs = {
            "heartbeat_timeout": self._configurator.heartbeat_timeout,
            "stop_sync_grace_period": self._stop_sync_grace_period,
            "stop_sync_poll_interval": self._stop_sync_poll_interval,
        }
        # A single-process session has no topology to convey.
        if topology is not None:
            kwargs["launch_topology"] = topology
        return kwargs

    def load_session(
            self,
            checkpoint_path: str,
            session_update_params: dict | None = None,
    ):
        session = Checkpointer.load_checkpoint(checkpoint_path)

        if session_update_params is not None:
            if not isinstance(session, TrainingSession):
                raise TypeError(
                    "Session extension updates require a TrainingSession"
                )
            if "overrides" in session_update_params:
                session.apply_extension_overrides(
                    session_update_params["overrides"]
                )
            elif "max_iterations" in session_update_params:
                session.apply_extension_overrides((
                    "session_config.max_iterations="
                    f"{session_update_params['max_iterations']}",
                ))
            else:
                raise ValueError(
                    "Unsupported session extension update parameters"
                )

        # The checkpoint's own topology describes the machine that wrote it,
        # so this launch decides how many processes to run and where they
        # meet, not the stored configuration.
        topology = resolve_launch_topology(
            session._components.get_resource("ddp").config
            if session._components.has_resource("ddp")
            else None,
            overrides=self._topology_overrides,
            from_checkpoint=True,
        )
        world_size = 1 if topology is None else topology.world_size
        self._check_rank_component_plan(session, world_size)

        self._session_process_wrappers = [
            SessionProcessWrapper(
                session=session,
                rank=rank,
                **self._wrapper_kwargs(topology),
            )
            for rank in range(world_size)
        ]

    @staticmethod
    def _check_rank_component_plan(session, world_size: int) -> None:
        """Settle what the secondary ranks will build, before any of them run.

        The workers resolve this for themselves, but by then rank 0 is on its
        way into `init_process_group`: a name that does not resolve would
        abort one worker while the others wait out the join timeout. Doing it
        here turns that into a launch-time error, and surfaces the
        rank-zero-only warnings where they can still be acted on.
        """
        if not session._components.has_resource("ddp"):
            return
        ddp_resource = session._components.get_resource("ddp")

        if world_size <= 1:
            # There is no rank to prune for, so no plan to settle. The names
            # are still resolved: a typo here is dormant until the same
            # configuration is run on more than one rank, and it should not
            # take that launch to find it. The collective diagnostics stay
            # off, because with one rank there is nobody left waiting.
            for names, source in (
                    (ddp_resource.rank_zero_components,
                     "ddp.rank_zero_components"),
                    (ddp_resource.parallel_components,
                     "ddp.parallel_components"),
            ):
                session.validate_component_names(names, source=source)
            return

        session.rank_parallel_names(
            parallel_components=(
                ddp_resource.parallel_components
                if ddp_resource.declares_parallel_components
                else None
            ),
            rank_zero_components=ddp_resource.rank_zero_components,
        )

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

        if "ddp" in config and not isinstance(config["ddp"], Mapping):
            raise ValueError("DDP configuration must contain ddp.world_size")

        topology = resolve_launch_topology(
            config.get("ddp"),
            overrides=self._topology_overrides,
            from_checkpoint=False,
        )
        world_size = 1 if topology is None else topology.world_size
        if topology is not None:
            # Keep the parent's session agreeing with the workers when the
            # launch resolved a different topology than the file states.
            config = dict(config)
            config["ddp"] = {
                **dict(config["ddp"]),
                **topology.config_overlay(),
            }

        sessions = [
            session_class(
                deepcopy(dict(config)),
                **deepcopy(dict(session_kwargs)),
            )
            for _ in range(world_size)
        ]
        self._check_rank_component_plan(sessions[0], world_size)

        wrappers = [
            SessionProcessWrapper(
                session=session,
                rank=rank,
                **self._wrapper_kwargs(topology),
            )
            for rank, session in enumerate(sessions)
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
            overrides = getattr(
                self._configurator,
                "extension_overrides",
                None,
            )
            if overrides:
                update_params = {"overrides": overrides}
            elif hasattr(self._configurator, "extension_overrides"):
                # Only launch-topology overrides were given, and those are
                # applied when the workers are built, not by extending the
                # session configuration.
                update_params = None
            else:
                update_params = {
                    "max_iterations": self._configurator.new_max_iters,
                }
            self.load_session(
                checkpoint_path=self._configurator.checkpoint_path,
                session_update_params=update_params,
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
