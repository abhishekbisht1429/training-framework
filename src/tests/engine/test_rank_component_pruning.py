"""What a worker's state carries once a secondary rank has settled it.

`prepare_worker_state` decides a rank's component set before anything is
constructed, so these tests read the prepared state rather than a session.
"""

from __future__ import annotations

import socket
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from training_framework.engine import TrainingEngine
from training_framework.engine.worker import prepare_worker_state
from training_framework.session import TrainingSession
from tests.test_utils import COMPONENTS_PACKAGE, register_test_components


def _bindable_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _session_config(tmp_path, **ddp_overrides: Any) -> dict[str, Any]:
    register_test_components()
    ddp_config: dict[str, Any] = {
        "world_size": 2,
        "backend": "gloo",
        "master_addr": "127.0.0.1",
        "master_port": _bindable_port(),
    }
    ddp_config.update(ddp_overrides)
    return {
        "session_config": {
            "rng_seed": 7,
            "sessions_dir": str(tmp_path),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "component_bindings": {"model": "it_3d45_model"},
        "ddp": ddp_config,
        "it_3d45_model": {},
        "it_3d45_train": {},
        "it_3d45_rank0_hook": {"call_every": 1},
        "logger": {"log_every": 1},
    }


def _session_state(tmp_path, **ddp_overrides: Any) -> dict[str, Any]:
    return TrainingSession(_session_config(tmp_path, **ddp_overrides)).get_state()


def _components_of(state: dict[str, Any]) -> set[str]:
    return set(state["components_state"])


def test_a_secondary_rank_drops_only_the_rank_zero_only_components(tmp_path):
    state = _session_state(tmp_path)

    prepared = prepare_worker_state(state, 1)

    assert _components_of(prepared) == {
        "ddp",
        "it_3d45_model",
        "it_3d45_train",
        "it_3d45_rank0_hook",
    }


def test_rank_zero_keeps_every_configured_component(tmp_path):
    state = _session_state(tmp_path)

    prepared = prepare_worker_state(state, 0)

    # `checkpointer` is activated by default, and is rank-zero-only too.
    assert _components_of(prepared) == {
        "ddp",
        "it_3d45_model",
        "it_3d45_train",
        "it_3d45_rank0_hook",
        "logger",
        "checkpointer",
    }


def test_the_config_key_keeps_a_component_off_a_secondary_rank(tmp_path):
    state = _session_state(
        tmp_path,
        rank_zero_components=["it_3d45_rank0_hook"],
    )

    prepared = prepare_worker_state(state, 1)

    assert _components_of(prepared) == {
        "ddp",
        "it_3d45_model",
        "it_3d45_train",
    }


def test_the_deprecated_list_still_decides_the_rank_set(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        state = _session_state(tmp_path, parallel_components=["it_3d45_train"])

    prepared = prepare_worker_state(state, 1)

    assert _components_of(prepared) == {"ddp", "it_3d45_model", "it_3d45_train"}


def test_configuring_the_deprecated_list_warns(tmp_path):
    with pytest.warns(FutureWarning, match="parallel_components"):
        _session_state(tmp_path, parallel_components=["it_3d45_train"])


def _engine() -> TrainingEngine:
    """An engine with only the configurator attributes it actually reads."""
    return TrainingEngine(SimpleNamespace(
        process_timeout_on_join=10,
        heartbeat_timeout=30,
        topology_overrides=None,
    ))


def test_an_unknown_rank_zero_name_fails_the_launch_before_any_worker(tmp_path):
    """Rank 0 would otherwise be inside init_process_group when rank 1 died."""
    engine = _engine()
    config = _session_config(tmp_path, rank_zero_components=["it_3d45_typo"])

    with pytest.raises(ValueError, match="it_3d45_typo"):
        engine.register_session(config)

    assert engine._session_process_wrappers == []


def test_a_rank_zero_name_that_is_not_in_this_session_fails_the_launch(tmp_path):
    engine = _engine()
    config = _session_config(tmp_path, rank_zero_components=["it_3d45_metrics"])

    with pytest.raises(RuntimeError, match="not configured in this session"):
        engine.register_session(config)


def test_a_single_process_launch_still_resolves_the_names(tmp_path):
    """A typo is dormant until the config is scaled; fail on it now."""
    engine = _engine()
    config = _session_config(
        tmp_path,
        world_size=1,
        rank_zero_components=["it_3d45_typo"],
    )

    with pytest.raises(ValueError, match="it_3d45_typo"):
        engine.register_session(config)

    assert engine._session_process_wrappers == []


def test_a_single_process_launch_resolves_the_deprecated_list_too(tmp_path):
    engine = _engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        config = _session_config(
            tmp_path,
            world_size=1,
            parallel_components=["it_3d45_typo"],
        )

        with pytest.raises(ValueError, match="it_3d45_typo"):
            engine.register_session(config)


def test_a_single_process_launch_settles_no_rank_plan(tmp_path):
    """Nothing is pruned and nothing can hang, so the diagnostics stay off."""
    engine = _engine()
    config = _session_config(
        tmp_path,
        world_size=1,
        rank_zero_components=["it_3d45_rank0_hook"],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.register_session(config)

    assert len(engine._session_process_wrappers) == 1
    assert [
        str(warning.message) for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ] == []


def test_rank_zero_components_must_be_a_list_of_names(tmp_path):
    """`rank_zero_components: logger` would ask for components 'l', 'o', ..."""
    config = _session_config(tmp_path, rank_zero_components="logger")

    with pytest.raises(ValueError, match="must be a list of component names"):
        TrainingSession(config)


def test_the_deprecated_list_must_be_a_list_of_names_too(tmp_path):
    config = _session_config(tmp_path, parallel_components="it_3d45_train")

    with pytest.raises(ValueError, match="must be a list of component names"):
        TrainingSession(config)
