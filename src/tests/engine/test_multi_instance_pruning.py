"""What a secondary rank builds when a component is configured twice.

`prepare_worker_state` settles a rank's component set before anything is
constructed, from the state and the configured bindings alone. With more than
one instance of a component that set has to be decided per instance: pruning
by component would take both or neither, and a consumer wired to one of them
has to keep the one it was wired to.
"""

from __future__ import annotations

import socket
import warnings
from typing import Any

import pytest

from training_framework.components import (
    Resource,
    requires_resource,
    resource,
)
from training_framework.engine.worker import prepare_worker_state
from training_framework.session import TrainingSession
from tests.test_utils import COMPONENTS_PACKAGE, register_test_components


def _bindable_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _declare_components() -> None:
    @resource("mi_dep")
    class Dependency(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.tag = (config or {}).get("tag")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass

    @requires_resource("mi_dep")
    @resource("mi_consumer")
    class Consumer(Resource):
        def __init__(self, config=None):
            super().__init__(config)
            self.dependency = self.get_dependency("mi_dep")

        def setup(self, session) -> None:
            pass

        def teardown(self, session) -> None:
            pass


def _config(tmp_path, **entries: Any) -> dict[str, Any]:
    register_test_components()
    _declare_components()
    ddp_config = dict(entries.pop("ddp", {}))
    config: dict[str, Any] = {
        "session_config": {
            "rng_seed": 7,
            "sessions_dir": str(tmp_path),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "component_bindings": {"model": "it_3d45_model"},
        "ddp": {
            "world_size": 2,
            "backend": "gloo",
            "master_addr": "127.0.0.1",
            "master_port": _bindable_port(),
            **ddp_config,
        },
        "it_3d45_model": {},
    }
    config.update(entries)
    return config


def _rank_one_components(config) -> set[str]:
    state = TrainingSession(config).get_state()
    with warnings.catch_warnings():
        # Pruning warnings are the subject of their own tests.
        warnings.simplefilter("ignore")
        prepared = prepare_worker_state(state, 1)
    return set(prepared["components_state"])


def test_a_rank_zero_only_instance_is_pruned_on_its_own(tmp_path):
    config = _config(
        tmp_path,
        **{"logger": {"log_every": 1}, "logger#2": {"log_every": 5}},
    )

    components = _rank_one_components(config)

    # The logger is rank-zero-only by class, so both instances go.
    assert "logger" not in components
    assert "logger#2" not in components


def test_an_instance_may_be_named_rank_zero_only(tmp_path):
    config = _config(
        tmp_path,
        ddp={"rank_zero_components": ["mi_dep#a"]},
        **{"mi_dep#a": {"tag": "a"}, "mi_dep#b": {"tag": "b"}},
    )

    components = _rank_one_components(config)

    # Named per instance, so its sibling is untouched.
    assert "mi_dep#a" not in components
    assert "mi_dep#b" in components


def test_a_consumer_keeps_the_instance_it_was_wired_to(tmp_path):
    config = _config(
        tmp_path,
        **{
            "component_bindings": {
                "model": "it_3d45_model",
                "mi_consumer": {"mi_dep": "mi_dep#b"},
            },
            "mi_dep#a": {"tag": "a"},
            "mi_dep#b": {"tag": "b"},
            "mi_consumer": {},
        },
    )

    components = _rank_one_components(config)

    assert "mi_consumer" in components
    assert "mi_dep#b" in components


def test_naming_an_instance_that_is_not_configured_is_reported(tmp_path):
    config = _config(
        tmp_path,
        ddp={"rank_zero_components": ["mi_dep#missing"]},
        **{"mi_dep#a": {"tag": "a"}},
    )
    state = TrainingSession(config).get_state()

    with pytest.raises(RuntimeError, match="not configured in this session"):
        prepare_worker_state(state, 1)


def test_rank_zero_pruning_leaves_rank_zero_state_alone(tmp_path):
    config = _config(
        tmp_path,
        **{"mi_dep#a": {"tag": "a"}, "mi_dep#b": {"tag": "b"}},
    )
    state = TrainingSession(config).get_state()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepare_worker_state(state, 1)

    # Preparing a worker must not mutate the state rank 0 still holds.
    assert {"mi_dep#a", "mi_dep#b"} <= set(state["components_state"])


# -- session extension -----------------------------------------------------


def test_an_extension_override_reaches_one_instance(tmp_path):
    config = _config(
        tmp_path,
        **{"logger": {"log_every": 1}, "logger#2": {"log_every": 5}},
    )
    session = TrainingSession(config)

    session.apply_extension_overrides(("logger#2.log_every=9",))

    components = session._components
    assert components.config_for_extension("logger#2") == {"log_every": 9}
    assert components.config_for_extension("logger") == {"log_every": 1}
