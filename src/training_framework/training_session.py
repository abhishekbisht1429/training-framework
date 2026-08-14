import os
import random
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, override, List

import numpy as np
import torch
from training_framework.util import context_entry, context_exit, requires_context, CaptureInitMeta, import_all_modules


#TODO: check for circular dependencies later

@dataclass(frozen=True)
class SessionConfig:
    rng_seed: int
    session_dir: str
    max_iterations: int


class SessionPhase(Enum):
    NEW = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()
    INTERRUPTED = auto()


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
    def teardown(self, session: "TrainingSession"):
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
    def teardown(self, session: "TrainingSession"):
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

# ==================== Registry ================
def make_registry(type):
    registry = {}

    def register(name: str):
        def wrapper(cls):
            if not issubclass(cls, type):
                raise TypeError(f"{cls.__name__} must be subclass of {type.__name__}")
            if name in registry:
                raise ValueError(f"{type} with name '{name}' already registered")
            registry[name] = cls
            cls.name = name
            cls.id = f"{type.__name__}.{cls.name}"
            return cls
        return wrapper

    return registry, register

HOOK_REGISTRY, hook = make_registry(Hook)
RESOURCE_REGISTRY, resource = make_registry(Resource)
STEP_REGISTRY, step = make_registry(Step)
# ====================================================

def requires_step(step_name: str):
    def wrapper(cls):
        # A step can only be required by another step.
        if not issubclass(cls, Step):
            raise TypeError(
                f"@requires_step can only be applied to Step subclasses. "
                f"'{cls.__name__}' is not a Step."
            )

        if "required_steps" not in cls.__dict__:
            cls.required_steps = []

        cls.required_steps.append(step_name)
        return cls

    return wrapper


def requires_hook(hook_name: str):
    def wrapper(cls):
        # A hook can only be required by a step or another hook.
        if not issubclass(cls, (Step, Hook)):
            raise TypeError(
                f"@requires_hook can only be applied to Step or Hook subclasses. "
                f"'{cls.__name__}' is neither."
            )

        if "required_hooks" not in cls.__dict__:
            cls.required_hooks = []

        cls.required_hooks.append(hook_name)
        return cls

    return wrapper

def requires_resource(resource_name: str):
    def wrapper(cls):
        if not issubclass(cls, (Step, Hook, Resource)):
            raise TypeError(
                "@requires_resource can only be applied to Step, Hook, "
                f"or Resource subclasses. '{cls.__name__}' is neither."
            )

        if "required_resources" not in cls.__dict__:
            cls.required_resources = list(
                getattr(cls, "required_resources", ())
            )

        cls.required_resources.append(resource_name)
        return cls

    return wrapper

def topological_sort_of_components():
    # NOTE: the requires_<component> decorator ensure the correct dependency order between
    # the different types of components, so we are not checking for that here again
    components = list(STEP_REGISTRY.values()) + list(HOOK_REGISTRY.values()) + list(RESOURCE_REGISTRY.values())

    prerequisites_graph: dict[str, List[str]] = {component.id: [] for component in components}
    for component in components:
        for required_hook_name in getattr(component, 'required_hooks', []):
            if required_hook_name not in HOOK_REGISTRY:
                raise RuntimeError(f"unmet prerequisite! '{required_hook_name}' not registered.")
            prerequisites_graph[component.id].append(HOOK_REGISTRY[required_hook_name].id)

        for required_step_name in getattr(component, 'required_steps', []):
            if required_step_name not in STEP_REGISTRY:
                raise RuntimeError(f"unmet prerequisite! '{required_step_name}' not registered.")
            prerequisites_graph[component.id].append(STEP_REGISTRY[required_step_name].id)

        for required_resource_name in getattr(component, 'required_resources', []):
            if required_resource_name not in RESOURCE_REGISTRY:
                raise RuntimeError(f"unmet prerequisite! '{required_resource_name}' not registered.")
            prerequisites_graph[component.id].append(RESOURCE_REGISTRY[required_resource_name].id)

    # create dependents graph
    dependents_graph: dict[str, List[str]] = {component_id: [] for component_id in prerequisites_graph.keys()}
    for component_id, prerequisites in prerequisites_graph.items():
        for prerequisite in prerequisites:
            dependents_graph[prerequisite].append(component_id)

    queue = deque()

    # find prereq count
    prereq_count = {}
    for component_id, prereqs in prerequisites_graph.items():
        prereq_count[component_id] = len(prereqs)
        if prereq_count[component_id] == 0:
            queue.append(component_id)

    sorted_components = []

    while len(queue) > 0:
        front_node = queue.popleft()
        sorted_components.append(front_node)

        for dependent_id in dependents_graph[front_node]:
            prereq_count[dependent_id] -= 1

            if prereq_count[dependent_id] == 0:
                queue.append(dependent_id)

    if len(sorted_components) != len(prerequisites_graph):
        raise RuntimeError("Cyclic dependency detected in the component graph!")


    index_of_component: dict[str, int] = {}
    for i, component_id in enumerate(sorted_components):
        index_of_component[component_id] = i

    return index_of_component

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

        # callbacks
        self._resources: dict[str, Resource] = {}
        self._steps: dict[str, Step] = {}
        self._hooks: dict[str, Hook] = {}

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

        self._register_components()

    def _register_components(self):
        for name in self._config.keys():
            if name in ["base_config"]:
                continue
            if name in STEP_REGISTRY:
                step_config = self._config[name]
                step_cls = STEP_REGISTRY[name]
                step_obj = step_cls(step_config)
                self.add_step(step_obj)
            elif name in HOOK_REGISTRY:
                hook_config = self._config[name]
                hook_cls = HOOK_REGISTRY[name]
                hook_obj = hook_cls(hook_config)
                self.register_hook(hook_obj)
            elif name in RESOURCE_REGISTRY:
                resource_config = self._config[name]
                resource_cls = RESOURCE_REGISTRY[name]
                resource_obj = resource_cls(resource_config)
                self.register_resource(resource_obj)
            else:
                raise ValueError(f"No step, hook or resource registered with name '{name}'!")

    # will contain only those attributes which need to be recreated after state load
    def _init_transient_infra(self):
        # TODO: should we use default device if requested device is not available ?
        self._device = self._check_and_get_device()

        # shared state for a single iteration
        self._shared_state: dict[str, Any] = {}

        # import all component modules
        import_all_modules(self._base_config['components_package'])

        # self._order_of_components = topological_sort_of_components()

    @override
    def get_state(self):
        state = {
            'config': self._base_config,
            'iteration': self._iteration,
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'np_rng_state': np.random.get_state(),
            "base_config": self._session_config,
            'resources_state': {
                name: {
                    'state': resource.get_state() if isinstance(resource, Stateful) else None,
                    'init_args': resource._init_args
                } for name, resource in self._resources.items()
            },
            'steps_state': {
                name: {
                    'state': step.get_state() if isinstance(step, Stateful) else None,
                    'init_args': step._init_args
                } for name, step in self._steps.items()
            },
            'hooks_state': {
                name: {
                    'state': hook.get_state() if isinstance(hook, Stateful) else None,
                    'init_args': hook._init_args
                } for name, hook in self._hooks.items()
            },
            'session_context': deepcopy(self._session_context),
            'init_args': self._init_args
        }

        return state

    @override
    def set_state(self, state):
        # 1. Restore configuration and tracking variables
        self._base_config = state['config']
        self._iteration = state['iteration']
        self._session_config = state["base_config"]

        # Guard CUDA restoration in case code is loaded on a CPU-only machine
        if torch.cuda.is_available() and 'cuda_rng_state' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])

        # Dynamically Reconstruct Polymorphic Nested Collections

        # Rebuild resources
        self._resources: dict[str, Resource] = {}
        for name, resource_info in state['resources_state'].items():
            cls = RESOURCE_REGISTRY[name]
            init_args = resource_info['init_args']
            obj = cls(*init_args['args'], **init_args['kwargs'])
            if issubclass(cls, Stateful):
                resource_state = resource_info['state']
                obj.set_state(resource_state)
            self._resources[name] = obj


        # Rebuild steps
        self._steps: dict[str, Step] = {}
        for name, step_info in state['steps_state'].items():
            cls = STEP_REGISTRY[name]
            init_args = step_info['init_args']
            obj = cls(*init_args['args'], **init_args['kwargs'])
            if issubclass(cls, Stateful):
                step_state = step_info['state']
                obj.set_state(step_state)
            self._steps[name] = obj

        # Rebuild hooks
        self._hooks: dict[str, "Hook"] = {}
        for name, hook_info in state['hooks_state'].items():
            cls = HOOK_REGISTRY[name]
            init_args = hook_info['init_args']
            obj = cls(*init_args['args'], **init_args['kwargs'])
            if issubclass(cls, Stateful):
                hook_state = hook_info['state']
                obj.set_state(hook_state)
            self._hooks[name] = obj

        # Restore session context
        self._session_context = state['session_context']
        self._init_transient_infra()

        # Restore Global RNG (Random Number Generator) States
        torch.set_rng_state(state['torch_rng_state'])
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['np_rng_state'])
        # TODO: maybe change the session phase to paused after restoration (by default it is set to new)

    @override
    def __setstate__(self, state):
        init_args = state['init_args']
        self.__init__(*init_args['args'], **init_args['kwargs'])
        self.set_state(state)

    @classmethod
    def from_state(cls, session_state):
        init_args = session_state['init_args']
        obj = cls(*init_args['args'], **init_args['kwargs'])
        obj.set_state(session_state)
        return obj

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
    def _sorted_hooks(self):
        order_of_components = topological_sort_of_components()
        return sorted(list(self._hooks.values()), key=lambda hook: order_of_components[hook.id])

    @property
    def _sorted_resources(self):
        order_of_components = topological_sort_of_components()
        return sorted(list(self._resources.values()), key=lambda resource: order_of_components[resource.id])

    @property
    def _sorted_steps(self):
        order_of_components = topological_sort_of_components()
        return sorted(list(self._steps.values()), key=lambda step: order_of_components[step.id])

    @property
    def _stateful_hooks(self):
        return [hook for hook in self._sorted_hooks if isinstance(hook, Stateful)]

    @property
    def _iteration_hooks(self):
        return [hook for hook in self._sorted_hooks if isinstance(hook, IterationHook)]

    @property
    def _session_hooks(self):
        return [hook for hook in self._sorted_hooks if isinstance(hook, SessionHook)]

    # ------------------------------------------------------------------------

    # ---------------------- Shared Context Management -----------------------

    @requires_context
    def _clear_iteration_state(self):
        self._shared_state.clear()

    # ----------------------------------------------------------------------
    # TODO: can we make a check that only those resources which are labeled as required can be accessed through the method below ?
    def get_resource(self, key: str):
        if key not in self._resources:
            raise KeyError(f"{key} not found in resources!")
        return self._resources[key]

    def has_resource(self, resource_name):
        return resource_name in self._resources

    def get_all_hooks(self):
        return list(self._hooks.values())

    def get_all_resources(self):
        return list(self._resources.values())

    def get_all_steps(self):
        return list(self._steps.values())

    # ================= component modifiers ======================

    def register_resource(self, resource: Resource):
        if not isinstance(resource, Resource):
            raise TypeError(
                f"The provided object '{type(resource).__name__}' "
                f"is not an instance of {Resource.__name__}!"
            )

        if not hasattr(resource, "name") or resource.name not in RESOURCE_REGISTRY:
            raise ValueError(
                f"Resource '{type(resource).__name__}' "
                "not registered in RESOURCE_REGISTRY!"
            )

        if resource.name in self._resources:
            raise ValueError(f"Resource '{resource.name}' already registered!")

        self._resources[resource.name] = resource

        return resource.name

    def register_hook(self, hook: Hook):
        if not isinstance(hook, Hook):
            raise TypeError(
                f"The provided object '{type(hook).__name__}' "
                f"is not an instance of {Hook.__name__}!"
            )

        if not hasattr(hook, "name") or hook.name not in HOOK_REGISTRY:
            raise ValueError(
                f"Hook '{type(hook).__name__}' "
                "not registered in HOOK_REGISTRY!"
            )

        if hook.name in self._hooks:
            raise ValueError(f"Hook '{hook.name}' already registered!")

        self._hooks[hook.name] = hook

    def add_step(self, step: Step):
        if not isinstance(step, Step):
            raise TypeError(
                f"The provided object '{type(step).__name__}' "
                f"is not an instance of {Step.__name__}!"
            )

        if not hasattr(step, "name") or step.name not in STEP_REGISTRY:
            raise ValueError(
                f"Step '{type(step).__name__}' "
                "not registered in STEP_REGISTRY!"
            )

        if step.name in self._steps:
            raise ValueError(f"Step '{step.name}' already registered!")

        self._steps[step.name] = step

    def remove_step(self, step_name):
        if step_name not in STEP_REGISTRY:
            raise ValueError(f"Step '{step_name}' not in STEP_REGISTRY!")
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' is not added to this session!")
        del self._steps[step_name]

    def unregister_hook(self, hook_name):
        if hook_name not in HOOK_REGISTRY:
            raise ValueError(f"Hook '{hook_name}' not in HOOK_REGISTRY!")
        if hook_name not in self._hooks:
            raise ValueError(f"Hook '{hook_name}' not registered with current session!")
        del self._hooks[hook_name]

    def unregister_resource(self, resource_name):
        if resource_name not in RESOURCE_REGISTRY:
            raise ValueError(f"Resource '{resource_name}' not in RESOURCE_REGISTRY!")
        if resource_name not in self._resources:
            raise ValueError(f"Resource '{resource_name}' not registered with current session!")
        del self._resources[resource_name]

    # ===========================================================

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
                    iter_hook.pre_iteration_callback(self)

            # 2. Run iteration components
            for step in self._sorted_steps:
                step.run(self)

            # 3. Run post iteration methods
            for iter_hook in reversed(self._iteration_hooks):
                if self._iteration == 1 or self._iteration == self.session_config.max_iterations or self._iteration % iter_hook.call_every == 0:
                    iter_hook.post_iteration_callback(self)

            iteration_complete = True
        finally:
            self._clear_iteration_state()

            # rollback in case of failure
            if not iteration_complete:
                self._iteration -= 1

        return self._iteration

    @context_entry
    def __enter__(self):
        self._raise_if_finished()

        # 1. Setup Resources
        successful_resource_setups = []
        try:
            for resource in self._sorted_resources:
                resource.setup(self)
                successful_resource_setups.append(resource)
        except Exception as e:
            print("Failed to setup resources!")

            for resource in reversed(successful_resource_setups):
                resource.teardown(self)

            self._session_context.clear()

            raise


        # 2. Call Session Hooks
        successful_hook_setups = []
        try:
            for session_hook in self._session_hooks:
                session_hook.setup(self)
        except Exception:
            print("Falied to setup session hooks!")
            for session_hook in reversed(successful_hook_setups):
                session_hook.teardown(self)

            self._session_context.clear()

            raise


        # 3. Update Phase to READY
        self._phase = SessionPhase.READY
        return self

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Clean up in reverse order of acquisition to respect dependencies
        for resource in reversed(self._sorted_resources):
            try:
                resource.teardown(self)
            except Exception as e:
                print(f"Error releasing resource '{resource.id}': {e}")

        # 2. Call session teardown hooks
        for session_hook in reversed(self._session_hooks):
            try:
                session_hook.teardown(self)
            except Exception as e:
                print(f"Error running teardown '{session_hook.name}': {e}")

        # 3. clear session context
        self._session_context.clear()

        # 4. Update the phase
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