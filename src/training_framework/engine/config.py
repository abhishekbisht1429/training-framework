import argparse
import warnings
from collections.abc import Mapping
from copy import deepcopy

from omegaconf import OmegaConf

from training_framework.components.config import (
    reject_legacy_components_entry,
    reserved_config_names,
)


class Configurator:
    def __init__(self):
        self._parser = argparse.ArgumentParser()

        group = self._parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--config", help="Path to session config file")
        group.add_argument(
            "--extend-session",
            nargs="+",
            metavar="VALUE",
            help=(
                "Path to session checkpoint to extend, optionally followed "
                "by the deprecated positional maximum iteration count"
            ),
        )
        group.add_argument(
            "--resume-session",
            help="Path to checkpoint to resume the session from",
        )

        self._parser.add_argument("--override", type=str, nargs="*", default=None)
        self._parser.add_argument(
            "--debug",
            action="store_true",
            help="Wait for worker processes using joins without monitoring",
        )
        self._parser.add_argument("--heartbeat-timeout", type=float, default=30.0)
        self._parser.add_argument(
            "--process_timeout_on_join",
            type=float,
            default=30.0,
        )

        self._args = self._parser.parse_args()

        self._session_configs = None
        self._checkpoint_path = None
        self._new_max_iters = None
        self._extension_overrides = None
        self._mode = None

        if self._args.config:
            self._mode = "new"
            config = OmegaConf.load(self._args.config)
            if self._args.override is not None:
                config.merge_with_dotlist(self._args.override)
            self._session_configs = OmegaConf.to_container(config)["sessions"]
        elif self._args.extend_session:
            self._mode = "extend"
            values = self._args.extend_session
            if len(values) > 2:
                self._parser.error(
                    "--extend-session accepts CHECKPOINT and an optional "
                    "deprecated NEW_MAX_ITERATIONS value"
                )
            self._checkpoint_path = values[0]
            overrides = list(self._args.override or [])
            if len(values) == 2:
                warnings.warn(
                    "The positional NEW_MAX_ITERATIONS argument is deprecated; "
                    "use --override session_config.max_iterations=VALUE",
                    DeprecationWarning,
                    stacklevel=2,
                )
                try:
                    self._new_max_iters = int(values[1])
                except ValueError:
                    self._parser.error(
                        "The deprecated positional NEW_MAX_ITERATIONS value "
                        "must be an integer"
                    )
                if any(
                    item.split("=", 1)[0].strip()
                    == "session_config.max_iterations"
                    for item in overrides
                ):
                    self._parser.error(
                        "max_iterations cannot be supplied both positionally "
                        "and through --override"
                    )
                overrides.append(
                    f"session_config.max_iterations={self._new_max_iters}"
                )
            if not overrides:
                self._parser.error(
                    "--extend-session requires at least one --override"
                )
            self._extension_overrides = tuple(overrides)
        elif self._args.resume_session:
            self._mode = "resume"
            self._checkpoint_path = self._args.resume_session

    def get_session_definition(self, index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current operation!")
        session_definition = self._session_configs[index]
        reject_legacy_components_entry(session_definition)
        return deepcopy(session_definition)

    @staticmethod
    def _reserved_config_names(session_config: Mapping) -> frozenset[str]:
        return reserved_config_names(session_config.get("session_type"))

    def get_component_config(self, session_index: int, key: str):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current operation!")
        session_config = self._session_configs[session_index]
        reject_legacy_components_entry(session_config)
        reserved_names = self._reserved_config_names(session_config)
        if key in session_config and key not in reserved_names:
            if not isinstance(session_config[key], Mapping):
                raise ValueError(
                    f"The value corresponding to the key '{key}' is not a mapping"
                )
            return deepcopy(session_config[key])
        raise KeyError(key)

    def get_all_component_configs(self, session_index):
        if not self._session_configs:
            raise KeyError("Cannot use this function in the current operation!")
        session_config = self._session_configs[session_index]
        reject_legacy_components_entry(session_config)
        reserved_names = self._reserved_config_names(session_config)
        component_configs = {}

        for key in session_config:
            if key in reserved_names:
                continue
            component_configs[key] = self.get_component_config(session_index, key)

        return component_configs

    @property
    def session_configs(self):
        if not self._session_configs:
            raise KeyError("Cannot use this property in the current operation!")
        for session_config in self._session_configs:
            reject_legacy_components_entry(session_config)
        return deepcopy(self._session_configs)

    @property
    def checkpoint_path(self):
        if not self._checkpoint_path:
            raise KeyError("Cannot use this property in the current operation!")
        return self._checkpoint_path

    @property
    def new_max_iters(self):
        if self._new_max_iters is None:
            raise KeyError("Cannot use this property in the current operation!")
        return self._new_max_iters

    @property
    def extension_overrides(self):
        if self._extension_overrides is None:
            raise KeyError("Cannot use this property in the current operation!")
        return tuple(self._extension_overrides)

    @property
    def process_timeout_on_join(self):
        return self._args.process_timeout_on_join

    @property
    def mode(self):
        return self._mode

    @property
    def heartbeat_timeout(self):
        return self._args.heartbeat_timeout

    @property
    def debug(self):
        return self._args.debug
