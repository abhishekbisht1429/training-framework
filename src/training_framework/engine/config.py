import argparse
from copy import deepcopy
from typing import Mapping

from omegaconf import OmegaConf

from training_framework.components.config import (
    RESERVED_CONFIG_NAMES,
    selected_component_names,
)


class Configurator:
    def __init__(self):
        self._parser = argparse.ArgumentParser()

        group = self._parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--config", help="Path to config file")
        group.add_argument(
            "--analysis-config",
            help="Path to analysis session config file",
        )
        group.add_argument("--extend-session", nargs=2, help="Path to session checkpoint to extend")
        group.add_argument("--resume-session", help="Path to checkpoint to resume the session from")

        self._parser.add_argument('--override', type=str, nargs='*', default=None)
        self._parser.add_argument(
            "--model-checkpoint",
            help="Training-session checkpoint to analyze",
        )
        self._parser.add_argument('--heartbeat-timeout', type=float, default=30.0)
        self._parser.add_argument('--process_timeout_on_join', type=float, default=30.0)
        # self._parser.add_argument('--wait-time-after-interrupt', type=float, default=10.0)

        self._args = self._parser.parse_args()

        self._session_configs = None
        self._checkpoint_path = None
        self._model_checkpoint_path = None
        self._new_max_iters = None
        self._mode = None

        if self._args.config:
            if self._args.model_checkpoint:
                self._parser.error(
                    "--model-checkpoint requires --analysis-config"
                )
            self._mode = "new"
            config = OmegaConf.load(self._args.config)
            if self._args.override is not None:
                # cli_config = OmegaConf.from_dotlist(self._args.override)
                config.merge_with_dotlist(self._args.override)
            self._session_configs = OmegaConf.to_container(config)['sessions']
        elif self._args.analysis_config:
            if not self._args.model_checkpoint:
                self._parser.error(
                    "--analysis-config requires --model-checkpoint"
                )
            self._mode = "analysis"
            self._model_checkpoint_path = self._args.model_checkpoint
            config = OmegaConf.load(self._args.analysis_config)
            if self._args.override is not None:
                config.merge_with_dotlist(self._args.override)
            self._session_configs = OmegaConf.to_container(config)['sessions']
        elif self._args.extend_session:
            if self._args.model_checkpoint:
                self._parser.error(
                    "--model-checkpoint requires --analysis-config"
                )
            self._mode = "extend"
            self._checkpoint_path = self._args.extend_session[0]
            self._new_max_iters = int(self._args.extend_session[1])
        elif self._args.resume_session:
            if self._args.model_checkpoint:
                self._parser.error(
                    "--model-checkpoint requires --analysis-config"
                )
            self._mode = "resume"
            self._checkpoint_path = self._args.resume_session

    def get_base_config(self, index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current mode!")
        return self._session_configs[index]

    @staticmethod
    def _selected_component_names(session_config: Mapping) -> list[str]:
        return selected_component_names(session_config)

    def get_component_config(self, session_index: int, key: str):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current mode!")
        session_config = self._session_configs[session_index]
        if key in session_config and key not in RESERVED_CONFIG_NAMES:
            if not isinstance(session_config[key], Mapping):
                raise ValueError(
                    f"The value corresponding to the key '{key}' is not a mapping"
                )
            return deepcopy(session_config[key])
        if key in self._selected_component_names(session_config):
            return {}
        raise KeyError(key)

    def get_all_component_configs(self, session_index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current mode!")
        session_config = self._session_configs[session_index]
        component_configs = {
            name: {}
            for name in self._selected_component_names(session_config)
        }

        for key in session_config:
            if key in RESERVED_CONFIG_NAMES:
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
    def model_checkpoint_path(self):
        if not self._model_checkpoint_path:
            raise KeyError("Cannot use this property in the current mode!")
        return self._model_checkpoint_path

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

    @property
    def heartbeat_timeout(self):
        return self._args.heartbeat_timeout