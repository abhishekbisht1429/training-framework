import argparse
from copy import deepcopy
from typing import Mapping, List, Any

from omegaconf import OmegaConf
from tensorboard.program import TensorBoard

from training_framework.resources import Logger, Checkpointer, Tensorboard, DDPHook
from training_framework.training_session import TrainingSession, HOOK_REGISTRY, STEP_REGISTRY, RESOURCE_REGISTRY

def create_session_from_config(config, rank=0):
    session = TrainingSession(config['session_config'])
    for name in config.keys():
        # for parallel session skip registration of components which are not marked as parallel
        if rank > 0 and config[name].get('parallel', False) == False:
            continue
        if name in ["ddp", "session_config"]:
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

    if "ddp" in config:
        ddp_hook = DDPHook(config['ddp'], rank=rank)
        session.register_hook(ddp_hook)
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


    def get_session_config(self, index):
        return self._session_configs[index]

    def get_sub_config(self, session_index: int, key: str):
        session_config = self._session_configs[session_index]
        if key not in session_config:
            raise KeyError(key)
        if not isinstance(session_config[key], Mapping):
            raise ValueError("The value corresponding to the key '{}' is not a mapping".format(key))
        return deepcopy(session_config[key])

    def create_sessions(self) -> List[TrainingSession]:
        sessions = []
        for config in self._session_configs:
            sessions.append(create_session_from_config(config))

        return sessions

