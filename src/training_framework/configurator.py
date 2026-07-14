import argparse
from copy import deepcopy
from typing import Mapping

from omegaconf import OmegaConf


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

        self._config = config


    def get_session_config(self):
        return OmegaConf.to_container(self._config)

    def get_resource_config(self, key: str):
        if key not in self._config:
            raise KeyError(key)
        if not isinstance(self._config[key], Mapping):
            raise ValueError("The value corresponding to the key '{}' is not a mapping".format(key))
        return deepcopy(self._config[key])
