import argparse
from copy import deepcopy
from typing import Mapping

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
            cli_config = OmegaConf.from_dotlist(self._args.override)
            config = OmegaConf.merge(config, cli_config)

        self._config = OmegaConf.to_container(config)


    def get_session_config(self):
        return self._config

    def get_resource_config(self, key: str):
        if key not in self._config:
            raise KeyError(key)
        if not isinstance(self._config[key], Mapping):
            raise ValueError("The value corresponding to the key '{}' is not a mapping".format(key))
        return deepcopy(self._config[key])

    def create_session(self):
        session = TrainingSession(self._config)
        if "logger" in self._config:
            session.add_wrapper(Logger(self.get_resource_config("logger")))

        if "checkpointer" in self._config:
            session.add_wrapper(Checkpointer(self.get_resource_config("checkpointer")))

        if "tensorboard" in self._config:
            session.register_resource(Tensorboard(self.get_resource_config("tensorboard")))

        return session

