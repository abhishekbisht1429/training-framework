import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List, Any, override

import numpy as np
import torch

from training_framework.util import context_entry, context_exit, requires_context, CaptureInitMeta


# ==================== Hook Registry ================
def make_registry(kind: str):
    registry = {}

    def register(name: str):
        def wrapper(cls):
            if name in registry:
                raise ValueError(f"{kind} with name '{name}' already registered")
            registry[name] = cls
            cls.name = name
            return cls
        return wrapper

    return registry, register


HOOK_REGISTRY, hook = make_registry("Hook")
RESOURCE_REGISTRY, resource = make_registry("Resource")
STEP_REGISTRY, step = make_registry("Step")
# ====================================================

@dataclass(frozen=True)
class SessionConfig:
    rng_seed: int
    session_dir: str
    max_iterations: int

class Stateful(ABC):
    @abstractmethod
    def get_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: Any) -> None:
        raise NotImplementedError

    def __getstate__(self) -> Any:
        return self.get_state()


    def __setstate__(self, state: Any) -> None:
        self.set_state(state)

class Hook(ABC, metaclass=CaptureInitMeta):
    name: str
    pass

class SessionHook(Hook, ABC):
    @abstractmethod
    def setup(self,  session: "TrainingSession"):
        pass

    @abstractmethod
    def teardown(self):
        pass

class IterationHook(Hook, ABC):
    call_every: int
    @abstractmethod
    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def post_iteration_callback(self, session: "TrainingSession") -> None:
        pass

class LifecycleHook(SessionHook, IterationHook, ABC):
    """
    An instance of this class wraps two callbacks (pre and post) around training iteration.
    The callbacks would be called for an iteration that is multiple of 'call_every'.
    pre would be called before the iteration starts and post would be called after it is finished.
    """
    pass

class Resource(ABC, metaclass=CaptureInitMeta):
    name: str

    @abstractmethod
    def setup(self, session: "TrainingSession"):
        pass

    @abstractmethod
    def teardown(self):
        pass


class Step(ABC, metaclass=CaptureInitMeta):
    name: str

    @abstractmethod
    def run(self, session: "TrainingSession") -> None:
        pass

class StatefulIterationHook(IterationHook, Stateful, ABC):
    pass

class StatefulSessionHook(SessionHook, Stateful, ABC):
    pass

class StatefulLifeCycleHook(LifecycleHook, Stateful, ABC):
    pass

class StatefulStep(Step, Stateful, ABC):
    pass

class StatefulResource(Resource, Stateful, ABC):
    pass

class SessionPhase(Enum):
    NEW = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()
    INTERRUPTED = auto()


class TrainingSession(Stateful, metaclass=CaptureInitMeta):

    def __init__(self, config: dict):
        self._config = config

        session_dir = os.path.join(
            self._config['sessions_dir'],
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._session_config = SessionConfig(
            rng_seed=self._config['rng_seed'],
            session_dir=session_dir,
            max_iterations=self._config['max_iterations'],
        )

        # callbacks
        self._resources: dict[str, Resource] = {}
        self._steps: List[Step] = []
        self._hooks: List[Hook] = []

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

    # will contain only those attributes which need to be recreated after state load
    def _init_transient_infra(self):
        # TODO: should we use default device if requested device is not available ?
        self._device = self._check_and_get_device()

        # shared state for a single iteration
        self._shared_state: dict[str, Any] = {}

    @override
    def get_state(self):
        state = {
            'config': self._config,
            'iteration': self._iteration,
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'np_rng_state': np.random.get_state(),
            'session_config': self._session_config,
            'resources_state': {
                key: {
                    'name': resource.name,
                    'state': resource.get_state() if isinstance(resource, Stateful) else None,
                    'init_args': resource._init_args
                } for key, resource in self._resources.items()
            },
            'steps_state': [
                {
                    'name': step.name,
                    'state': step.get_state() if isinstance(step, Stateful) else None,
                    'init_args': step._init_args
                } for step in self._steps
            ],
            'hooks_state': [
                {
                    'name': hook.name,
                    'state': hook.get_state() if isinstance(hook, Stateful) else None,
                    'init_args': hook._init_args
                }
                for hook in self._hooks
            ],
            'session_context': self._session_context,
            'init_args': self._init_args
        }

        return state

    @override
    def set_state(self, state):
        # 1. Restore configuration and tracking variables
        self._config = state['config']
        self._iteration = state['iteration']
        self._session_config = state['session_config']

        # 2. Restore Global RNG (Random Number Generator) States
        torch.set_rng_state(state['torch_rng_state'])
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['np_rng_state'])

        # Guard CUDA restoration in case code is loaded on a CPU-only machine
        if torch.cuda.is_available() and 'cuda_rng_state' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])

        # 3. Dynamically Reconstruct Polymorphic Nested Collections

        # Rebuild resources
        self._resources: dict[str, Resource] = {}
        for key, resource_info in state['resources_state'].items():
            cls = RESOURCE_REGISTRY[resource_info['name']]
            init_args = resource_info['init_args']
            obj = cls(*init_args['args'], **init_args['kwargs'])
            if issubclass(cls, Stateful):
                resource_state = resource_info['state']
                obj.set_state(resource_state)
            self._resources[key] = obj


        # Rebuild steps
        self._steps: List[Step] = []
        for step_info in state['steps_state']:
            cls = STEP_REGISTRY[step_info['name']]
            init_args = step_info['init_args']
            obj = cls(*init_args['args'], **init_args['kwargs'])
            if issubclass(cls, Stateful):
                step_state = step_info['state']
                obj.set_state(step_state)
            self._steps.append(obj)

        # Rebuild hooks
        self._hooks: List[Hook] = []
        for hook_info in state['hooks_state']:
            cls = HOOK_REGISTRY[hook_info['name']]
            init_args = hook_info['init_args']
            obj = cls(*init_args['args'], **init_args['kwargs'])
            if issubclass(cls, Stateful):
                hook_state = hook_info['state']
                obj.set_state(hook_state)
            self._hooks.append(obj)

        # Restore session context
        self._session_context = state['session_context']
        self._init_transient_infra()

    @override
    def __setstate__(self, state):
        init_args = state['init_args']
        self.__init__(*init_args['args'], **init_args['kwargs'])
        self.set_state(state)

    # --------------------------- Public properties----------------------
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

    # ---------------------- Helper private attributes ----------------------

    @property
    def _stateful_hooks(self):
        return [hook for hook in self._hooks if isinstance(hook, Stateful)]

    @property
    def _iteration_hooks(self):
        return [hook for hook in self._hooks if isinstance(hook, IterationHook)]

    @property
    def _session_hooks(self):
        return [hook for hook in self._hooks if isinstance(hook, SessionHook)]

    # ------------------------------------------------------------------------

    # ---------------------- Shared Context Management -----------------------

    # @requires_context
    # def share_value(self, key: str, value: Any):
    #     self._shared_state[key] = value
    #
    # @requires_context
    # def get_shared_value(self, key: str) -> Any:
    #     if key not in self._shared_state:
    #         raise KeyError(f"{key} not found in shared state")
    #     return self._shared_state[key]

    @requires_context
    def _clear_iteration_state(self):
        self._shared_state.clear()

    # ----------------------------------------------------------------------

    def get_resource(self, key: str):
        if key not in self._resources:
            raise KeyError(f"{key} not found in resources")
        return self._resources[key]

    def register_resource(self, resource: Resource):
        if not isinstance(resource, Resource):
            raise TypeError(f"The provided object '{type(resource).__name__}' is not an instance of {Resource.__name__}!")
        if not hasattr(resource, "name") or resource.name not in RESOURCE_REGISTRY:
            raise ValueError(f"Resource '{type(resource).__name__}' not registered in RESOURCE_REGISTRY!")

        hex_id = f"0x{time.time_ns():x}"
        resource_id = f"{ resource.__class__.__name__}_{hex_id}"

        self._resources[resource_id] = resource

        return resource_id

    def add_step(self, step: Step):
        if not isinstance(step, Step):
            raise TypeError(f"The provided object '{type(step).__name__}' is not an instance of {Step.__name__}!")
        if not hasattr(step, "name") or step.name not in STEP_REGISTRY:
            raise ValueError(f"Step '{type(step).__name__}' not registered in STEP_REGISTRY!")
        self._steps.append(step)

    def register_hook(self, hook: Hook):
        if not isinstance(hook, Hook):
            raise TypeError(f"The provided object '{type(hook).__name__}' is not an instance of {Hook.__name__}!")
        if not hasattr(hook, "name") or hook.name not in HOOK_REGISTRY:
            raise ValueError(f"Hook '{type(hook).__name__}' not registered in HOOK_REGISTRY!")
        self._hooks.append(hook)

    def _check_and_get_device(self):
        if self._config['device'].startswith('cuda'):
            if torch.cuda.is_available():
                return torch.device(self._config['device'])
            else:
                raise ValueError(f"device '{self._config['device']}' not available")
        elif self._config['device'] == 'cpu':
            return torch.device('cpu')
        else:
            raise ValueError(f"Unknown device '{self._config['device']}'")

    def _check_ready(self):
        if self._phase is SessionPhase.NEW:
            raise RuntimeError('Use within "with"!')

    def _check_finished(self):
        if self._phase is SessionPhase.FINISHED:
            raise RuntimeError('Attempting to run a finished session!')

    @requires_context
    def __iter__(self):
        return self

    @requires_context
    def __next__(self):
        self._check_ready()

        if self._iteration == self.session_config.max_iterations:
            self._phase = SessionPhase.FINISHED
            raise StopIteration

        self._phase = SessionPhase.RUNNING
        self._iteration += 1

        # Execution order or callbacks A, B, C
        # A_pre -> B_pre -> C_pre -> A -> B -> C -> C_post -> B_post -> A_post

        # 1. Run pre iteration methods
        for iter_hook in self._iteration_hooks:
            if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % iter_hook.call_every == 0:
                iter_hook.pre_iteration_callback(self)

        # 2. Run iteration components
        for step in self._steps:
            step.run(self)

        # 3. Run post iteration methods
        for iter_hook in reversed(self._iteration_hooks):
            if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % iter_hook.call_every == 0:
                iter_hook.post_iteration_callback(self)

        self._clear_iteration_state()

        return self._iteration

    @context_entry
    def __enter__(self):
        self._check_finished()

        # 1. Setup Resources
        for resource_key in self._resources:
            self._resources[resource_key].setup(self)

        # 2. Call Session Hooks
        for session_hook in self._session_hooks:
            session_hook.setup(self)

        # 3. Update Phase to READY
        self._phase = SessionPhase.READY
        return self

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Clean up in reverse order of acquisition to respect dependencies
        for resource_key in reversed(self._resources):
            try:
                self._resources[resource_key].teardown()
            except Exception as e:
                print(f"Error releasing resource '{resource_key}': {e}")

        # 2. Call session teardown hooks
        for session_hook in self._session_hooks:
            session_hook.teardown()

        # 3. clear session context
        self._session_context.clear()

        # 4. Update the phase
        if self._phase is SessionPhase.READY:
            self._phase = SessionPhase.NEW
        elif self._phase is SessionPhase.RUNNING:
            self._phase = SessionPhase.PAUSED

        return False

