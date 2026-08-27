from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


TRAINING_SESSION_TYPE = "training"
ANALYSIS_SESSION_TYPE = "analysis"

_REQUIRED_SESSION_CONFIG_FIELDS = frozenset({
    "rng_seed",
    "sessions_dir",
    "max_iterations",
    "components_package",
})
_OPTIONAL_SESSION_CONFIG_DEFAULTS = {
    "device": "cpu",
    "show_execution_graph": True,
}
_SESSION_CONFIG_FIELDS = (
    _REQUIRED_SESSION_CONFIG_FIELDS
    | _OPTIONAL_SESSION_CONFIG_DEFAULTS.keys()
)


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


def normalize_session_type(session_type: str) -> str:
    if not isinstance(session_type, str):
        raise TypeError("session_type must be a string")
    normalized = session_type.strip()
    if not normalized:
        raise ValueError("session_type must not be empty")
    return normalized


def normalize_session_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("'session_config' must be a mapping")

    missing = sorted(_REQUIRED_SESSION_CONFIG_FIELDS - config.keys())
    if missing:
        raise ValueError(
            "Missing required session_config fields: " + ", ".join(missing)
        )
    unknown = sorted(config.keys() - _SESSION_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            "Unknown session_config fields: " + ", ".join(unknown)
        )

    normalized = dict(config)
    for name, default in _OPTIONAL_SESSION_CONFIG_DEFAULTS.items():
        normalized.setdefault(name, default)

    rng_seed = normalized["rng_seed"]
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool):
        raise TypeError("session_config.rng_seed must be an integer")
    max_iterations = normalized["max_iterations"]
    if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations < 0
    ):
        raise ValueError(
            "session_config.max_iterations must be a non-negative integer"
        )
    for name in ("sessions_dir", "components_package", "device"):
        value = normalized[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"session_config.{name} must be a non-empty string")
    if not isinstance(normalized["show_execution_graph"], bool):
        raise TypeError("session_config.show_execution_graph must be a boolean")

    return normalized
