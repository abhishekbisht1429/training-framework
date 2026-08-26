import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


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
    """Decode current and legacy session configuration state."""
    if "session_config" in state:
        return (
            state["config"],
            state["base_config"],
            state["session_config"],
        )

    init_args = state["init_args"]
    if init_args["args"]:
        config = init_args["args"][0]
    else:
        config = init_args["kwargs"]["config"]

    return config, state["config"], state["base_config"]
