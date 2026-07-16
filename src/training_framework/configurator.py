import argparse
from copy import deepcopy
from typing import Mapping, List, Any

from omegaconf import OmegaConf
from tensorboard.program import TensorBoard

from training_framework.resources import Logger, Checkpointer, Tensorboard
from training_framework.training_session import TrainingSession


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

    def get_resource_config(self, session_index: int, key: str):
        session_config = self._session_configs[session_index]
        if key not in session_config:
            raise KeyError(key)
        if not isinstance(session_config[key], Mapping):
            raise ValueError("The value corresponding to the key '{}' is not a mapping".format(key))
        return deepcopy(session_config[key])

    def create_sessions(self):
        sessions = []
        for config in self._session_configs:
            session = TrainingSession(config)
            if "logger" in config:
                session.register_hook(Logger(config["logger"]))

            if "checkpointer" in config:
                session.register_hook(Checkpointer(config["checkpointer"]))

            if "tensorboard" in config:
                session.register_resource(Tensorboard(config["tensorboard"]))

            sessions.append(session)

        return sessions

