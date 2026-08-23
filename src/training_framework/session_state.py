import random
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


