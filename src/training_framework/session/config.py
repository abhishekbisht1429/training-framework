from dataclasses import dataclass
from enum import Enum, auto


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


class SessionMode(str, Enum):
    TRAINING = "training"
    ANALYSIS = "analysis"


def normalize_session_mode(mode: SessionMode | str) -> SessionMode:
    if isinstance(mode, SessionMode):
        return mode
    try:
        return SessionMode(mode)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in SessionMode)
        raise ValueError(f"Invalid session mode {mode!r}; expected one of: {allowed}") from error
