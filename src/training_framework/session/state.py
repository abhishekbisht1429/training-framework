import random
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from training_framework.session.config import normalize_session_config


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "np_rng_state": np.random.get_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["np_rng_state"])


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
    return config, session_settings, state["session_config"]
