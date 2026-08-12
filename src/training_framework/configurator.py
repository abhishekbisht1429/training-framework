import argparse
from copy import deepcopy
from typing import Mapping

from omegaconf import OmegaConf

from training_framework.builtin_components import Checkpointer, DDPResource
from training_framework.training_session import TrainingSession
from training_framework.training_session import HOOK_REGISTRY, STEP_REGISTRY, RESOURCE_REGISTRY


def copy_and_modify_session_for_worker(session, rank, session_update_params: dict | None=None):
    # session = Checkpointer.load_checkpoint(path=session)
    session = deepcopy(session)

    if session_update_params is not None:
        if "max_iterations" in session_update_params:
            session.update_max_iters(session_update_params["max_iterations"])

    if session.has_resource("ddp") and rank > 0:
        placeholder_ddp_resource: DDPResource = session.get_resource("ddp")
        ddp_resource = DDPResource(
            config=placeholder_ddp_resource.config,
            rank=rank
        )
        parallel_components = set(ddp_resource.parallel_components)

        session.unregister_resource(placeholder_ddp_resource.name)
        session.register_resource(ddp_resource)

        for hook in session.get_all_hooks():
            if hook.name not in parallel_components:
                session.unregister_hook(hook.name)
        for resource in session.get_all_resources():
            if resource.name not in parallel_components:
                session.unregister_resource(resource.name)
        for step in session.get_all_steps():
            if step.name not in parallel_components:
                session.remove_step(step.name)

    return session

def create_session_from_config(config):
    # # import all component modules
    # import_all_modules(config['components_package'])

    # create session
    session = TrainingSession(config["base_config"])
    # non_parallel_components = set(config.keys())
    # if "ddp" in config:
    #     # placeholder ddp_resource - each worker will modify this resource accordingly
    #     ddp_resource = DDPResource(config['ddp'], rank=-1)
    #     # if 'parallel_components' in config['ddp']:
    #     #     for component in config['ddp']['parallel_components']:
    #     #         non_parallel_components.remove(component)
    #     session.register_resource(ddp_resource)
    for name in config.keys():
        if name in ["base_config"]:
            continue
        # # for parallel session skip registration of components which are not marked as parallel
        # if rank > 0 and name in non_parallel_components:
        #     continue
        if name in STEP_REGISTRY:
            step_config = config[name]
            step_cls = STEP_REGISTRY[name]
            step_obj = step_cls(step_config)
            session.add_step(step_obj)
        elif name in HOOK_REGISTRY:
            hook_config = config[name]
            hook_cls = HOOK_REGISTRY[name]
            hook_obj = hook_cls(hook_config)
            session.register_hook(hook_obj)
        elif name in RESOURCE_REGISTRY:
            resource_config = config[name]
            resource_cls = RESOURCE_REGISTRY[name]
            resource_obj = resource_cls(resource_config)
            session.register_resource(resource_obj)
        else:
            raise ValueError(f"No step, hook or resource registered with name '{name}'!")
    return session


class Configurator:
    def __init__(self):
        self._parser = argparse.ArgumentParser()

        group = self._parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--config", help="Path to config file")
        group.add_argument("--extend-session", nargs=2, help="Path to session checkpoint to extend")
        group.add_argument("--resume-session", help="Path to checkpoint to resume the session from")

        self._parser.add_argument('--override', type=str, nargs='*', default=None)
        self._parser.add_argument('--process_timeout_on_join', type=float, default=5.0)

        self._args = self._parser.parse_args()

        self._session_configs = None
        self._checkpoint_path = None
        self._new_max_iters = None
        self._mode = None

        if self._args.config:
            self._mode = "new"
            config = OmegaConf.load(self._args.config)
            if self._args.override is not None:
                # cli_config = OmegaConf.from_dotlist(self._args.override)
                config.merge_with_dotlist(self._args.override)
            self._session_configs = OmegaConf.to_container(config)['sessions']
        elif self._args.extend_session:
            self._mode = "extend"
            self._checkpoint_path = self._args.extend_session[0]
            self._new_max_iters = int(self._args.extend_session[1])
        elif self._args.resume_session:
            self._mode = "resume"
            self._checkpoint_path = self._args.resume_session

    def get_base_config(self, index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current mode!")
        return self._session_configs[index]

    def get_component_config(self, session_index: int, key: str):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current mode!")
        session_config = self._session_configs[session_index]
        if key not in session_config:
            raise KeyError(key)
        if not isinstance(session_config[key], Mapping):
            raise ValueError("The value corresponding to the key '{}' is not a mapping".format(key))
        return deepcopy(session_config[key])


    def get_all_component_configs(self, session_index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current mode!")
        component_configs = {}
        session_config = self._session_configs[session_index]

        for key in session_config:
            if key == "base_config":
                continue
            component_configs[key] = self.get_component_config(session_index, key)

        return component_configs

    @property
    def session_configs(self):
        if not self._session_configs:
            raise KeyError("Cannot use this property in the current mode!")
        return deepcopy(self._session_configs)

    @property
    def checkpoint_path(self):
        if not self._checkpoint_path:
            raise KeyError("Cannot use this property in the current mode!")
        return self._checkpoint_path

    @property
    def new_max_iters(self):
        if not self._new_max_iters:
            raise KeyError("Cannot use this property in the current mode!")
        return self._new_max_iters

    @property
    def process_timeout_on_join(self):
        return self._args.process_timeout_on_join

    @property
    def mode(self):
        return self._mode