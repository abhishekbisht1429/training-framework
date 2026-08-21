import random
from typing import Any

import numpy as np
import torch

from training_framework.components import Stateful


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


def capture_component_collection(components: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "state": (
                component.get_state()
                if isinstance(component, Stateful)
                else None
            ),
            "init_args": getattr(component, "_init_args"),
        }
        for name, component in components.items()
    }


def restore_component_collection(
        component_states: dict[str, Any],
        registry: dict[str, type],
) -> dict[str, Any]:
    components = {}
    for name, component_info in component_states.items():
        component_class = registry[name]
        init_args = component_info["init_args"]
        component = component_class(
            *init_args["args"],
            **init_args["kwargs"],
        )
        if issubclass(component_class, Stateful):
            component.set_state(component_info["state"])
        components[name] = component

    return components
