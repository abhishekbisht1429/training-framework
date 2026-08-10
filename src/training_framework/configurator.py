import argparse
import collections
import sys
from copy import deepcopy
from typing import Mapping, List, Any

from omegaconf import OmegaConf
from tensorboard.program import TensorBoard

from training_framework.builtin_components import Logger, Checkpointer, Tensorboard, DDPResource
from training_framework.training_session import TrainingSession
from training_framework.training_session import HOOK_REGISTRY, STEP_REGISTRY, RESOURCE_REGISTRY

import importlib
import pkgutil


def import_all_modules(package_name: str) -> None:
    package = importlib.import_module(package_name)

    if not hasattr(package, "__path__"):
        return

    prefix = package.__name__ + "."

    for module_info in pkgutil.walk_packages(
        package.__path__,
        prefix=prefix,
    ):
        importlib.import_module(module_info.name)

def create_session_from_config(config, rank=0):
    # import all component modules
    import_all_modules(config['components_package'])

    # create session
    session = TrainingSession(config["base_config"])
    if "ddp" in config:
        ddp_resource = DDPResource(config['ddp'], rank=rank)
        session.register_resource(ddp_resource)
    for name in config.keys():
        if name in ["ddp", "base_config", "components_package"]:
            continue
        # for parallel session skip registration of components which are not marked as parallel
        if rank > 0 and config[name].get('parallel', False) == False:
            continue
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
        self._parser.add_argument('config', type=str)
        self._parser.add_argument('--override', type=str, nargs='*', default=None)

        self._args = self._parser.parse_args()
        config = OmegaConf.load(self._args.config)
        if self._args.override is not None:
            # cli_config = OmegaConf.from_dotlist(self._args.override)
            config.merge_with_dotlist(self._args.override)

        self._session_configs = OmegaConf.to_container(config)['sessions']


    def get_base_config(self, index):
        return self._session_configs[index]

    def get_component_config(self, session_index: int, key: str):
        session_config = self._session_configs[session_index]
        if key not in session_config:
            raise KeyError(key)
        if not isinstance(session_config[key], Mapping):
            raise ValueError("The value corresponding to the key '{}' is not a mapping".format(key))
        return deepcopy(session_config[key])


    def get_all_component_configs(self, session_index):
        component_configs = {}
        session_config = self._session_configs[session_index]

        for key in session_config:
            if key == "base_config":
                continue
            component_configs[key] = self.get_component_config(session_index, key)

        return component_configs

    @property
    def session_configs(self):
        return deepcopy(self._session_configs)