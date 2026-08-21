import os
import random
import time
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any, cast, override

import numpy as np
import torch

from training_framework.builtin_components import Logger, Checkpointer
from training_framework.components import (
    Hook,
    IterationHook,
    LifecycleHook,
    Resource,
    SessionHook,
    Stateful,
    StatefulIterationHook,
    StatefulLifecycleHook,
    StatefulLifeCycleHook,
    StatefulResource,
    StatefulSessionHook,
    StatefulStep,
    Step,
)
from training_framework.registry import (
    HOOK_REGISTRY,
    RESOURCE_REGISTRY,
    STEP_REGISTRY,
    hook,
    make_registry,
    requires_hook,
    requires_resource,
    requires_step,
    resource,
    step,
    topological_sort_of_components,
)
from training_framework.session_components import SessionComponents
from training_framework.session_config import SessionConfig, SessionPhase
from training_framework.session_io import write_session_config
from training_framework.session_state import (
    capture_component_collection,
    capture_rng_state,
    restore_component_collection,
    restore_rng_state,
)
from training_framework.util import (
    CaptureInitMeta,
    context_entry,
    context_exit,
    import_all_modules,
    requires_context,
)

__all__ = [
    "HOOK_REGISTRY",
    "Hook",
    "IterationHook",
    "LifecycleHook",
    "RESOURCE_REGISTRY",
    "Resource",
    "STEP_REGISTRY",
    "SessionConfig",
    "SessionHook",
    "SessionPhase",
    "Stateful",
    "StatefulIterationHook",
    "StatefulLifeCycleHook",
    "StatefulLifecycleHook",
    "StatefulResource",
    "StatefulSessionHook",
    "StatefulStep",
    "Step",
    "TrainingSession",
    "hook",
    "make_registry",
    "requires_hook",
    "requires_resource",
    "requires_step",
    "resource",
    "step"
]


class TrainingSession(Stateful, metaclass=CaptureInitMeta):

    def __init__(self, config: dict):
        self._config = config
        self._base_config = config["base_config"]

        session_dir = os.path.join(
            self._base_config['sessions_dir'],
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._session_config = SessionConfig(
            rng_seed=self._base_config['rng_seed'],
            session_dir=session_dir,
            max_iterations=self._base_config['max_iterations'],
        )

        self._set_component_collections()

        # session essentials
        self._iteration = 0

        torch.manual_seed(self._session_config.rng_seed)
        random.seed(self._session_config.rng_seed)
        torch.cuda.manual_seed(self._session_config.rng_seed)
        np.random.seed(self._session_config.rng_seed)

        # shared session context
        self._session_context: dict[str, Any] = {}

        self._init_transient_infra()

        self._phase = SessionPhase.NEW

        self._register_default_components()
        self._register_components()

    def _set_component_collections(
            self,
            *,
            resources: dict[str, Resource] | None = None,
            steps: dict[str, Step] | None = None,
            hooks: dict[str, Hook] | None = None,
    ) -> None:
        self._components = SessionComponents(
            resources=resources,
            steps=steps,
            hooks=hooks,
        )
        self._resources = self._components.resources
        self._steps = self._components.steps
        self._hooks = self._components.hooks

    def _register_default_components(self) -> None:
        # Register logger
        self.register_hook(
            Logger({
                'log_every': 10
            })
        )

        # Register Checkpointer
        self.register_hook(
            Checkpointer({
                'checkpoint_every': 100,
            })
        )

    def _register_components(self) -> None:
        self._components.register_from_config(self._config)

    # will contain only those attributes which need to be recreated after state load
    def _init_transient_infra(self):
        # TODO: should we use default device if requested device is not available ?
        self._device = self._check_and_get_device()

        # shared state for a single iteration
        self._shared_state: dict[str, Any] = {}

        # import all component modules
        import_all_modules(self._base_config['components_package'])

        self._successfully_setup_resource_names = set()
        self._successfully_setup_hook_names = set()

        self._dist_manager_err_conn = None
        self._heartbeat_interval = None
        self._last_heartbeat_time = 0.0

    @override
    def get_state(self):
        state = {
            "config": deepcopy(self._config),
            "base_config": deepcopy(self._base_config),
            "session_config": self._session_config,
            "iteration": self._iteration,
            "resources_state": capture_component_collection(self._resources),
            "steps_state": capture_component_collection(self._steps),
            "hooks_state": capture_component_collection(self._hooks),
            "session_context": deepcopy(self._session_context),
            "init_args": self._init_args,
        }
        state.update(capture_rng_state())
        return state

    @staticmethod
    def _configuration_from_state(state):
        if "session_config" in state:
            return (
                state["config"],
                state["base_config"],
                state["session_config"],
            )

        init_args = state["init_args"]
        if init_args["args"]:
            config = init_args["args"][0]
        else:
            config = init_args["kwargs"]["config"]

        return config, state["config"], state["base_config"]

    @override
    def set_state(self, state):
        (
            self._config,
            self._base_config,
            self._session_config,
        ) = self._configuration_from_state(state)
        self._iteration = state["iteration"]

        self._set_component_collections(
            resources=restore_component_collection(
                state["resources_state"],
                RESOURCE_REGISTRY,
            ),
            steps=restore_component_collection(
                state["steps_state"],
                STEP_REGISTRY,
            ),
            hooks=restore_component_collection(
                state["hooks_state"],
                HOOK_REGISTRY,
            ),
        )

        self._session_context = state["session_context"]
        self._init_transient_infra()
        restore_rng_state(state)

    def _prepare_for_state_restore(self, state) -> None:
        self._init_args = state["init_args"]
        (
            self._config,
            self._base_config,
            self._session_config,
        ) = self._configuration_from_state(state)
        self._session_context = {}
        self._phase = SessionPhase.NEW
        import_all_modules(self._base_config["components_package"])

    @override
    def __setstate__(self, state):
        self._prepare_for_state_restore(state)
        self.set_state(state)

    @classmethod
    def from_state(cls, session_state):
        obj = cls.__new__(cls)
        obj.__setstate__(session_state)
        return obj

    # --------------------------- Public properties----------------------
    @property
    def full_config(self):
        return deepcopy(self._config)

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
        self._shared_state.clear()

    def get_resource(self, key: str):
        return self._components.get_resource(key)

    def has_resource(self, resource_name):
        return resource_name in self._resources

    def get_all_hooks(self):
        return list(self._hooks.values())

    def get_all_resources(self):
        return list(self._resources.values())

    def get_all_steps(self):
        return list(self._steps.values())

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
        if 'device' not in self._base_config:
            return torch.device('cpu')
        if self._base_config['device'].startswith('cuda') and torch.cuda.is_available():
            return torch.device(self._base_config['device'])
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

        iteration_complete = False
        try:
            self._iteration += 1
            self._phase = SessionPhase.RUNNING

            if self._iteration > self.session_config.max_iterations:
                self._phase = SessionPhase.FINISHED
                raise StopIteration

            # Execution order or callbacks A, B, C
            # A_pre -> B_pre -> C_pre -> A -> B -> C -> C_post -> B_post -> A_post

            # 1. Run pre iteration methods
            for iter_hook in self._iteration_hooks:
                if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % iter_hook.call_every == 0:
                    self.send_heartbeat(f"Running {iter_hook.id}")
                    iter_hook.pre_iteration_callback(self)

            # 2. Run iteration components
            for step in self._sorted_steps:
                self.send_heartbeat(f"Running {step.id}")
                step.run(self)

            # 3. Run post iteration methods
            for iter_hook in reversed(self._iteration_hooks):
                if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % iter_hook.call_every == 0:
                    self.send_heartbeat(f"Running {iter_hook.id}")
                    iter_hook.post_iteration_callback(self)

            iteration_complete = True
        finally:
            self._clear_iteration_state()

            # rollback in case of failure
            if not iteration_complete:
                self._iteration -= 1

        return self._iteration

    def _setup_resources(self) -> None:
        for component in self._sorted_resources:
            self.send_heartbeat(f"Running setup {component.id}")
            component.setup(self)
            self._successfully_setup_resource_names.add(component.name)

    def _setup_session_hooks(self) -> None:
        for component in self._session_hooks:
            component.setup(self)
            self.send_heartbeat(f"Running setup {component.id}")
            self._successfully_setup_hook_names.add(component.name)

    def _teardown_resources(self, *, after_exception: bool = False) -> None:
        stage_suffix = " after exception" if after_exception else ""
        for component in reversed(self._sorted_resources):
            if component.name not in self._successfully_setup_resource_names:
                continue
            try:
                self.send_heartbeat(
                    f"Running teardown {component.id}{stage_suffix}"
                )
                component.teardown(self)
            except Exception as error:
                print(f"Error releasing resource '{component.id}': {error}")

    def _teardown_session_hooks(self, *, after_exception: bool = False) -> None:
        stage_suffix = " after exception" if after_exception else ""
        for component in reversed(self._session_hooks):
            if component.name not in self._successfully_setup_hook_names:
                continue
            try:
                self.send_heartbeat(
                    f"Running teardown {component.id}{stage_suffix}"
                )
                component.teardown(self)
            except Exception as error:
                print(f"Error running teardown '{component.name}': {error}")

    @context_entry
    def __enter__(self):
        self._raise_if_finished()
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
        ddp_resource = self.get_resource("ddp") if self.has_resource("ddp") else None
        if ddp_resource is None or cast(Any, ddp_resource).rank == 0:
            write_session_config(
                self.session_config.session_dir,
                self.full_config,
            )

        self._phase = SessionPhase.READY
        return self

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._dist_manager_err_conn is not None and exc_type is not None:
            self._dist_manager_err_conn.send({
                "type": "error",
                "rank": cast(Any, self.get_resource("ddp")).rank,
                "pid": os.getpid(),
                "exception_type": str(exc_type),
                "message": str(exc_val),
                "traceback": traceback.format_exc(),
            })

        self._teardown_resources()
        self._teardown_session_hooks()
        self._session_context.clear()

        if self._phase is SessionPhase.READY:
            self._phase = SessionPhase.NEW
        elif self._phase is SessionPhase.RUNNING:
            self._phase = SessionPhase.PAUSED

        return False

    def update_max_iters(self, new_max_iters):
        self._session_config = SessionConfig(
            rng_seed=self.session_config.rng_seed,
            session_dir=self.session_config.session_dir,
            max_iterations=new_max_iters
        )

    def set_dist_manager_err_conn(self, err_conn):
        self._dist_manager_err_conn = err_conn

    def set_heartbeat_interval(self, interval):
        self._heartbeat_interval = interval

    def send_heartbeat(self, stage):
        if self._dist_manager_err_conn is not None:
            if (time.monotonic() - self._last_heartbeat_time) >= self._heartbeat_interval:
                self._dist_manager_err_conn.send({
                    'type': 'heartbeat',
                    'pid': os.getpid(),
                    'iteration': self._iteration,
                    'stage': stage,
                })
                self._last_heartbeat_time = time.monotonic()
