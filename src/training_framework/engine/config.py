import argparse
import math
import warnings
from collections.abc import Mapping
from copy import deepcopy

from omegaconf import OmegaConf

from training_framework.components.config import (
    reject_legacy_components_entry,
    reserved_config_names,
)
from training_framework.engine.topology import TOPOLOGY_KEYS


#: Overrides that describe the launch rather than the session, and so may be
#: given when resuming a checkpoint that changes nothing else.
_TOPOLOGY_OVERRIDES = frozenset(f"ddp.{name}" for name in TOPOLOGY_KEYS)


class Configurator:
    @staticmethod
    def _non_negative_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise argparse.ArgumentTypeError(
                "must be a finite, non-negative number"
            )
        return parsed

    @staticmethod
    def _positive_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise argparse.ArgumentTypeError(
                "must be a finite number greater than zero"
            )
        return parsed

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
            "--stop-sync-grace-period",
            type=self._non_negative_finite_float,
            default=0.01,
            help=(
                "Seconds to poll a DDP stop collective before sleeping"
            ),
        )
        self._parser.add_argument(
            "--stop-sync-poll-interval",
            type=self._positive_finite_float,
            default=0.005,
            help=(
                "Seconds to sleep between DDP stop-collective polls after "
                "the grace period"
            ),
        )
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
        self._topology_overrides = {}
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
            self._topology_overrides, overrides = (
                self._split_topology_overrides(overrides)
            )
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
            if not overrides and not self._topology_overrides:
                self._parser.error(
                    "--extend-session requires at least one --override"
                )
            self._extension_overrides = tuple(overrides)
        elif self._args.resume_session:
            self._mode = "resume"
            self._checkpoint_path = self._args.resume_session
            self._topology_overrides, unsupported = (
                self._split_topology_overrides(self._args.override or [])
            )
            if unsupported:
                names = ", ".join(sorted(unsupported))
                self._parser.error(
                    "--resume-session only accepts launch-topology "
                    f"overrides ({', '.join(sorted(_TOPOLOGY_OVERRIDES))}), "
                    f"but got: {names}. Use --extend-session to change the "
                    "session configuration."
                )

    @staticmethod
    def _split_topology_overrides(
            overrides,
    ) -> tuple[dict[str, str], list[str]]:
        """Separate launch-topology overrides from session-config ones.

        Topology overrides never reach the session extension machinery: they
        describe the machine this launch runs on, not the run itself.
        """
        topology: dict[str, str] = {}
        remaining: list[str] = []
        for item in overrides:
            key, separator, value = item.partition("=")
            key = key.strip()
            if separator and key in _TOPOLOGY_OVERRIDES:
                topology[key.split(".", 1)[1]] = value
            else:
                remaining.append(item)
        return topology, remaining

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
    def topology_overrides(self):
        return dict(self._topology_overrides)

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
    def stop_sync_grace_period(self):
        return self._args.stop_sync_grace_period

    @property
    def stop_sync_poll_interval(self):
        return self._args.stop_sync_poll_interval

    @property
    def debug(self):
        return self._args.debug
