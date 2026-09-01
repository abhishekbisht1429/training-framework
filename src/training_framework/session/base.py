import os
import random
from abc import abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any, cast, override

import numpy as np
import torch

from training_framework.components import (
    Component,
    Hook,
    Resource,
    Stateful,
    Step,
    format_execution_graph,
)
from training_framework.session.components import SessionComponents
from training_framework.session.config import (
    SessionConfig,
    SessionPhase,
    normalize_session_config,
    normalize_session_type,
)
from training_framework.session.io import write_session_config
from training_framework.session.registry import session_class_for_type
from training_framework.session.state import (
    capture_rng_state,
    configuration_from_state,
    restore_rng_state,
)
from training_framework.session.runtime import (
    clear_iteration_state,
    report_worker_exception,
    run_iteration,
    send_heartbeat as send_worker_heartbeat,
    setup_resources,
    setup_session_hooks,
    teardown_resources,
    teardown_session_hooks,
)
from training_framework.util import (
    CaptureInitMeta,
    context_entry,
    context_exit,
    import_all_modules,
    requires_context,
)


class Session(Stateful, metaclass=CaptureInitMeta):
    _registered_session_type: str | None = None

    def __init__(self, config: dict):
        registered_type = type(self)._registered_session_type
        if registered_type is None:
            raise TypeError(
                f"{type(self).__name__} is not registered with "
                "@register_session_type"
            )
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        if "session_config" not in config:
            raise ValueError("config must contain 'session_config'")

        self._session_type = registered_type
        self._config = deepcopy(dict(config))
        self._session_settings = normalize_session_config(
            self._config["session_config"],
        )
        self._config["session_config"] = deepcopy(self._session_settings)

        session_dir = os.path.join(
            self._session_settings["sessions_dir"],
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        self._session_config = SessionConfig(
            rng_seed=self._session_settings["rng_seed"],
            session_dir=session_dir,
            max_iterations=self._session_settings["max_iterations"],
        )

        self._iteration = 0

        torch.manual_seed(self._session_config.rng_seed)
        random.seed(self._session_config.rng_seed)
        torch.cuda.manual_seed(self._session_config.rng_seed)
        np.random.seed(self._session_config.rng_seed)

        self._session_context: dict[str, Any] = {}

        self._init_transient_infra()
        self._set_component_collections()

        self._phase = SessionPhase.NEW
        self._register_components()

    @classmethod
    @abstractmethod
    def _default_component_configs(cls) -> Mapping[str, Mapping]:
        raise NotImplementedError

    def _set_component_collections(
            self,
            *,
            aliases: dict[str, str] | None = None,
    ) -> None:
        if aliases is None:
            aliases = self._config.get("aliases", {})
        self._components = SessionComponents(
            aliases=aliases,
            session_type=self._session_type,
        )

    def _register_components(self) -> None:
        self._components.register_from_config(
            self._config,
            default_configs=self._default_component_configs(),
        )

    def _get_session_type_state(self) -> dict[str, Any]:
        return {}

    def _restore_session_type_state(self, state: Mapping[str, Any]) -> None:
        pass

    def _restore_and_validate_session_type(
            self,
            state: Mapping[str, Any],
    ) -> None:
        if "session_type" not in state:
            raise ValueError(
                "Checkpoint does not contain the required 'session_type'"
            )
        stored_session_type = normalize_session_type(state["session_type"])
        expected_session_type = type(self)._registered_session_type
        if stored_session_type != expected_session_type:
            raise ValueError(
                f"{type(self).__name__} cannot restore "
                f"{stored_session_type} session-type state"
            )
        self._session_type = stored_session_type
        self._restore_session_type_state(state)

    def _init_transient_infra(self):
        self._device = self._check_and_get_device()
        self._shared_state: dict[str, Any] = {}
        import_all_modules(self._session_settings["components_package"])

        self._successfully_setup_resource_names = set()
        self._successfully_setup_hook_names = set()

        self._dist_manager_err_conn = None
        self._heartbeat_interval = None
        self._last_heartbeat_time = 0.0

    @override
    def get_state(self):
        state = {
            "session_type": self._session_type,
            "config": deepcopy(self._config),
            "session_config": self._session_config,
            "iteration": self._iteration,
            "components_state": self._components.get_state(),
            "session_context": deepcopy(self._session_context),
            "init_args": self._init_args,
        }
        state.update(self._get_session_type_state())
        state.update(capture_rng_state())
        return state

    @staticmethod
    def _configuration_from_state(state):
        return configuration_from_state(state)

    @override
    def set_state(self, state):
        if "components_state" not in state:
            raise ValueError(
                "Checkpoint uses an unsupported component state schema; "
                "expected 'components_state'"
            )
        self._restore_and_validate_session_type(state)
        (
            self._config,
            self._session_settings,
            self._session_config,
        ) = self._configuration_from_state(state)
        self._iteration = state["iteration"]

        self._init_transient_infra()
        restored_components = SessionComponents(
            aliases=self._config.get("aliases", {}),
            session_type=self._session_type,
        )
        restored_components.set_state(state["components_state"])
        self._components = restored_components

        self._session_context = state["session_context"]
        restore_rng_state(state)

    def _prepare_for_state_restore(self, state) -> None:
        self._init_args = state["init_args"]
        self._restore_and_validate_session_type(state)
        (
            self._config,
            self._session_settings,
            self._session_config,
        ) = self._configuration_from_state(state)
        self._session_context = {}
        self._phase = SessionPhase.NEW
        import_all_modules(self._session_settings["components_package"])

    @override
    def __setstate__(self, state):
        self._prepare_for_state_restore(state)
        self.set_state(state)

    @classmethod
    def from_state(cls, session_state):
        if "session_type" not in session_state:
            raise ValueError(
                "Checkpoint does not contain the required 'session_type'"
            )
        session_type = normalize_session_type(session_state["session_type"])
        target_cls = (
            session_class_for_type(session_type)
            if cls is Session
            else cls
        )
        if target_cls._registered_session_type != session_type:
            raise ValueError(
                f"{target_cls.__name__} cannot restore "
                f"{session_type} session-type state"
            )
        obj = target_cls.__new__(target_cls)
        obj.__setstate__(session_state)
        return obj

    @property
    def full_config(self):
        config = deepcopy(self._config)
        config["session_type"] = self._session_type
        session_kwargs = deepcopy(self._init_args.get("kwargs", {}))
        session_kwargs.pop("config", None)
        if session_kwargs:
            config["session_kwargs"] = session_kwargs
        return config

    @property
    def session_type(self) -> str:
        return self._session_type

    @property
    def session_config(self) -> SessionConfig:
        return self._session_config

    @property
    def iteration(self):
        return self._iteration

    @property
    def device(self):
        return self._device

    @property
    def session_context(self):
        return self._session_context

    @property
    @requires_context
    def iteration_context(self):
        return self._shared_state

    # --------------------------------------------------------------------

    @property
    def _sorted_hooks(self):
        return self._components.ordered_hooks

    @property
    def _hooks(self):
        return self._components.hooks

    @property
    def _resources(self):
        return self._components.resources

    @property
    def _steps(self):
        return self._components.steps

    @property
    def _sorted_resources(self):
        return self._components.ordered_resources

    @property
    def _sorted_steps(self):
        return self._components.ordered_steps

    @property
    def _stateful_hooks(self):
        return [
            component
            for component in self._sorted_hooks
            if isinstance(component, Stateful)
        ]

    @property
    def _iteration_hooks(self):
        return self._components.iteration_hooks

    @property
    def _session_hooks(self):
        return self._components.session_hooks

    @requires_context
    def _clear_iteration_state(self):
        clear_iteration_state(self)

    def get_resource(self, key: str):
        return self._components.get_resource(key)

    def has_resource(self, resource_name):
        return self._components.has_resource(resource_name)

    @property
    def component_aliases(self) -> dict[str, str]:
        return self._components.alias_bindings

    def resolve_component_name(self, name: str) -> str:
        return self._components.resolve_name(name)

    def _component_dependency_closure(self, names) -> set[str]:
        return self._components.dependency_closure(names)

    def get_all_hooks(self):
        return list(self._components.hooks.values())

    def get_all_resources(self):
        return list(self._components.resources.values())

    def get_all_steps(self):
        return list(self._components.steps.values())

    def execution_graph(self) -> str:
        return format_execution_graph(
            resources=self.get_all_resources(),
            hooks=self.get_all_hooks(),
            steps=self.get_all_steps(),
            max_iterations=self.session_config.max_iterations,
            aliases=self._components.aliases,
            session_type=self._session_type,
        )

    def print_execution_graph(self, *, file=None) -> None:
        print(self.execution_graph(), file=file)

    def register_resource(self, component: Resource, overwrite=False):
        return self._components.register_resource(component, overwrite=overwrite)

    def register_hook(self, component: Hook, overwrite=False):
        return self._components.register_hook(component, overwrite=overwrite)

    def add_step(self, component: Step, overwrite=False):
        return self._components.add_step(component, overwrite=overwrite)

    def remove_step(self, step_name):
        self._components.remove_step(step_name)

    def unregister_hook(self, hook_name):
        self._components.unregister_hook(hook_name)

    def unregister_resource(self, resource_name):
        self._components.unregister_resource(resource_name)

    def _check_and_get_device(self):
        if 'device' not in self._session_settings:
            return torch.device('cpu')
        if self._session_settings['device'].startswith('cuda') and torch.cuda.is_available():
            return torch.device(self._session_settings['device'])
        else:
            return torch.device('cpu')

    def _raise_if_not_new(self):
        if self._phase is not SessionPhase.NEW:
            raise RuntimeError("Session should be in NEW phase!")

    def _raise_if_new(self):
        if self._phase is SessionPhase.NEW:
            raise RuntimeError('Use within "with"!')

    def _raise_if_finished(self):
        if self._phase is SessionPhase.FINISHED:
            raise RuntimeError('Attempting to run a finished session!')

    def set_device(self, device: torch.device):
        self._raise_if_not_new()
        self._device = device

    @requires_context
    def __iter__(self):
        return self

    @requires_context
    def __next__(self):
        self._raise_if_new()
        return run_iteration(self)

    def _setup_resources(self) -> None:
        setup_resources(self)

    def _setup_session_hooks(self) -> None:
        setup_session_hooks(self)

    def _teardown_resources(self, *, after_exception: bool = False) -> None:
        teardown_resources(self, after_exception=after_exception)

    def _teardown_session_hooks(self, *, after_exception: bool = False) -> None:
        teardown_session_hooks(self, after_exception=after_exception)

    @context_entry
    def __enter__(self):
        self._raise_if_finished()

        ddp_resource = self.get_resource("ddp") if self.has_resource("ddp") else None
        if self._session_settings.get("show_execution_graph", True):
            # print only for rank zero
            if ddp_resource is None or cast(Any, ddp_resource).rank == 0:
                self.print_execution_graph()
        self._successfully_setup_resource_names.clear()
        self._successfully_setup_hook_names.clear()

        try:
            self._setup_resources()
        except Exception:
            print("Failed to setup resources!")
            self._teardown_resources(after_exception=True)
            self._session_context.clear()
            raise

        try:
            self._setup_session_hooks()
        except Exception:
            print("Failed to setup session hooks!")
            self._teardown_session_hooks(after_exception=True)
            self._teardown_resources(after_exception=True)
            self._session_context.clear()
            raise

        # Dump config (only for rank 0)
        if ddp_resource is None or cast(Any, ddp_resource).rank == 0:
            write_session_config(
                self.session_config.session_dir,
                self.full_config,
            )

        self._phase = SessionPhase.READY
        return self

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        report_worker_exception(self, exc_type, exc_val)

        self._teardown_session_hooks()
        self._teardown_resources()
        self._session_context.clear()

        if self._phase is SessionPhase.READY:
            self._phase = SessionPhase.NEW
        elif self._phase is SessionPhase.RUNNING:
            self._phase = SessionPhase.PAUSED

        return False

    def set_dist_manager_err_conn(self, err_conn):
        self._dist_manager_err_conn = err_conn

    def set_heartbeat_interval(self, interval):
        self._heartbeat_interval = interval

    def send_heartbeat(self, stage):
        send_worker_heartbeat(self, stage)
