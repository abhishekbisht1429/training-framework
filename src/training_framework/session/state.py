import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from training_framework.session.config import (
    SessionConfig,
    normalize_session_config,
)


#: Bumped when the shape of a captured session state changes. Version 2
#: records a single CUDA RNG stream rather than one per device visible to the
#: writer, which is what lets a checkpoint move between machines with
#: different GPU counts. A state without the key is version 1.
CHECKPOINT_VERSION = 2

_RESTORE_RNG = ContextVar("training_framework_restore_rng", default=True)


@contextmanager
def rng_restore_suppressed():
    """Load a session's state without adopting its RNG.

    Reading a checkpoint for the weights it holds should not reseed the
    process doing the reading. Pickle drives `__setstate__`, so the intent
    cannot be passed as an argument and travels in the context instead.
    """
    token = _RESTORE_RNG.set(False)
    try:
        yield
    finally:
        _RESTORE_RNG.reset(token)


def rng_restore_enabled() -> bool:
    return _RESTORE_RNG.get()


@contextmanager
def rng_preserved():
    """Leave the process's RNG exactly as it was on entry.

    Suppressing the restore only keeps a checkpoint's saved RNG out; the
    components rebuilt while loading still draw from the generators -- a
    constructor that initialises weights the saved state then overwrites,
    say. Putting the caller's state back afterwards makes the load invisible
    to whatever random work comes next, however much the load drew.

    A CUDA stream is only put back if CUDA was already initialised on entry;
    a load that is the first to touch CUDA has no earlier stream to return to.
    """
    saved = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(saved)


def capture_rng_state(carried_cuda_state: Any = None) -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": _capture_cuda_rng_state(carried_cuda_state),
        "np_rng_state": _capture_numpy_rng_state(),
    }


def _capture_numpy_rng_state() -> tuple:
    """numpy's state with its key as a list rather than an ndarray.

    A checkpoint holds plain data only, so it loads with
    `torch.load(weights_only=True)`; `np.random.set_state` takes either.
    """
    kind, key, *rest = np.random.get_state()
    return (kind, key.tolist(), *rest)


def _capture_cuda_rng_state(carried: Any) -> Any:
    """The CUDA stream of the device this process pinned, if it pinned one.

    Only one stream is recorded. Capturing every visible device, as version 1
    did, tied the checkpoint to the GPU count of the writing machine, and
    recorded a stale stream for every device but the current one -- a process
    only ever advances the generator of the device it actually uses.

    A process that pinned no device has no stream of its own to record. The
    parent is such a process: it restores a session only to hand its state to
    the workers, so it passes on whatever it was restored with rather than
    dropping it.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.cuda.get_rng_state()
    return carried


def restore_rng_state(
        state: dict[str, Any],
        *,
        rng_seed: int | None = None,
) -> Any:
    """Adopt the RNG recorded in `state`.

    Returns the CUDA stream that could not be applied here, for the caller to
    carry, or `None` once it has been applied.
    """
    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["np_rng_state"])

    return _restore_cuda_rng_state(state.get("cuda_rng_state"), rng_seed)


def _restore_cuda_rng_state(stored: Any, rng_seed: int | None) -> Any:
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        # No device is pinned here, so there is nowhere definite to put the
        # stream. Touching CUDA now would take a context this process does
        # not need and queue a deferred call that fails somewhere unrelated.
        return stored

    stream = _cuda_stream_for_this_device(stored)
    if stream is None:
        # Nothing recorded: seed this device from the session's own seed so
        # CUDA randomness stays reproducible instead of falling back to
        # whatever the default generator was seeded with.
        if rng_seed is not None:
            torch.cuda.manual_seed(rng_seed)
        return None

    torch.cuda.set_rng_state(stream)
    return None


def _cuda_stream_for_this_device(stored: Any) -> Any:
    if stored is None:
        return None
    if isinstance(stored, torch.Tensor):
        return stored
    if isinstance(stored, Sequence) and len(stored) > 0:
        # Version 1 recorded one entry per device visible to the writer.
        # Only the writer's own device holds a live stream; the others were
        # frozen when the run started. Checkpoints are written by rank 0,
        # whose device is the first visible one, so entry 0 is the live one.
        # Indexing by the restoring rank's ordinal would pick a stale entry
        # -- and fail outright when fewer devices are visible than were
        # recorded, which is the failure this version exists to fix.
        return stored[0]
    return None


def configuration_from_state(
        state: Mapping[str, Any],
) -> tuple[dict, dict, Any]:
    if "config" not in state or "session_config" not in state:
        raise ValueError(
            "Checkpoint uses an unsupported configuration state schema"
        )
    config = deepcopy(state["config"])
    if "session_config" not in config:
        raise ValueError(
            "Checkpoint config does not contain the required 'session_config'"
        )
    session_settings = normalize_session_config(config["session_config"])
    config["session_config"] = deepcopy(session_settings)
    session_config = state["session_config"]
    if isinstance(session_config, Mapping):
        # Stored as a mapping so a checkpoint holds no framework class; a
        # state written before that holds the dataclass itself.
        session_config = SessionConfig(**session_config)
    return config, session_settings, session_config
