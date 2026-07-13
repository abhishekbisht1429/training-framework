import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Any, override

import numpy as np
import torch

@dataclass(frozen=True)
class SessionConfig:
    rng_seed: int
    session_dir: str
    max_iterations: int

class Resource(ABC):
    @abstractmethod
    def setup(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def __getstate__(self) -> Any:
        pass

    @abstractmethod
    def __setstate__(self, state: Any) -> None:
        pass

class IterationComponent(ABC):

    @abstractmethod
    def run(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def __getstate__(self) -> Any:
        pass

    @abstractmethod
    def __setstate__(self, state: Any) -> None:
        pass

class Wrapper(Resource, ABC):
    """
    An instance of this class wraps two callbacks (pre and post) around training iteration.
    The callbacks would be called for an iteration that is multiple of 'call_wrapper_every'.
    pre would be called before the iteration starts and post would be called after it is finished.
    """
    call_wrapper_every: int

    @override
    def setup(self, session: "TrainingSession") -> None:
        pass

    @override
    def release(self) -> None:
        pass

    @abstractmethod
    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def post_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    @abstractmethod
    def __getstate__(self) -> Any:
        pass

    @abstractmethod
    def __setstate__(self, state: Any) -> None:
        pass

class TrainingSession:

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
        self._iteration_components: List[IterationComponent] = []
        self._iteration_wrappers: List[Wrapper] = []

        # session essentials
        self._iteration = 0

        torch.manual_seed(self._session_config.rng_seed)
        random.seed(self._session_config.rng_seed)
        torch.cuda.manual_seed(self._session_config.rng_seed)
        np.random.seed(self._session_config.rng_seed)

        self._ready = False

        self._init_transient_infra()

    # will contain only those attributes which need to be recreated after state load
    def _init_transient_infra(self):
        # TODO: should we use default device if requested device is not available ?
        self._device = self._check_and_get_device()

        # shared state for a single iteration
        self._shared_state: dict[str, Any] = {}

    def __getstate__(self):
        state = {
            'config': self._config,
            'iteration': self._iteration,
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'np_rng_state': np.random.get_state(),
            'session_config': self._session_config,
            'resources': self._resources,
            'iteration_components': self._iteration_components,
            'iteration_wrappers': self._iteration_wrappers,
        }

        return state

    def __setstate__(self, state):
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

        # Rebuild Resource Providers
        self._resources = state['resources']

        # Rebuild Iteration Components / Callbacks
        self._iteration_components = state['iteration_components']

        # Rebuild Iteration Wrappers
        self._iteration_wrappers = state['iteration_wrappers']

        # 4. Run _init_non_serailizable
        self._init_transient_infra()

    @property
    def session_config(self) -> SessionConfig:
        return self._session_config

    @property
    def iteration(self):
        return self._iteration

    @property
    def device(self):
        return self._device

    def get_resource(self, key: str):
        if key not in self._resources:
            raise KeyError(f"{key} not found in resources")
        return self._resources[key]

    def share_value(self, key: str, value: Any):
        self._shared_state[key] = value

    def get_shared_value(self, key: str) -> Any:
        if key not in self._shared_state:
            raise KeyError(f"{key} not found in shared state")
        return self._shared_state[key]

    def _clear_iteration_state(self):
        self._shared_state.clear()

    def register_resource(self, resource: Resource):
        if not isinstance(resource, Resource):
            raise TypeError(
                f"The provided provider '{type(resource).__name__}' "
                "does not implement the required acquire() and release() Protocol methods."
            )

        hex_id = f"0x{id(resource.__class__.__name__):x}"
        resource_key = f"{ resource.__class__.__name__}_{hex_id}"

        self._resources[resource_key] = resource

        return resource_key

    def add_iteration_component(self, component: IterationComponent):
        if not isinstance(component, IterationComponent):
            raise TypeError(f"The provided object '{type(component).__name__}' is not callable!")
        self._iteration_components.append(component)

    def add_iteration_wrapper(self, wrapper: Wrapper):
        if not isinstance(wrapper, Wrapper):
            raise TypeError(f"The provided object '{type(wrapper).__name__}' does not implement IterationWrapper Protocol methods.!")
        self._iteration_wrappers.append(wrapper)

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

    def _setup_resources(self):
        for resource_key in self._resources:
            self._resources[resource_key].setup(self)

        for wrapper in self._iteration_wrappers:
            wrapper.setup(self)

    def _release_resources(self):
        # Clean up in reverse order of acquisition to respect dependencies
        for resource_key in reversed(self._resources):
            # 1. Trigger the resource's internal cleanup logic
            try:
                self._resources[resource_key].release()
            except Exception as e:
                print(f"Error releasing resource '{resource_key}': {e}")

        # clean up wrappers
        for wrapper in self._iteration_wrappers:
            wrapper.release()

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError('Use within "with"!')

    def __iter__(self):
        self._check_ready()
        return self


    def __next__(self):
        self._check_ready()

        if self._iteration == self.session_config.max_iterations:
            raise StopIteration
        self._iteration += 1

        # Execution order or callbacks A, B, C
        # A_pre -> B_pre -> C_pre -> A -> B -> C -> C_post -> B_post -> A_post

        # 1. Run pre iteration methods
        for wrapper in self._iteration_wrappers:
            if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % wrapper.call_wrapper_every == 0:
                wrapper.pre_iteration_callback(self)

        # 2. Run iteration components
        for component in self._iteration_components:
            component.run(self)

        # 3. Run post iteration methods
        for wrapper in self._iteration_wrappers:
            if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % wrapper.call_wrapper_every == 0:
                wrapper.post_iteration_callback(self)

        self._clear_iteration_state()

        return self._iteration

    def __enter__(self):
        self._setup_resources()
        self._ready = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release_resources()
        self._ready = False

        return True

    def start(self):
        try:
            for _ in self:
                pass
        except KeyboardInterrupt:
            pass