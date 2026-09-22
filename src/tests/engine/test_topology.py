from __future__ import annotations

import socket
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from training_framework.engine import (
    Configurator,
    LaunchTopology,
    TrainingEngine,
    host_rendezvous,
    load_session_for_worker,
    resolve_launch_topology,
)
from training_framework.engine.topology import pin_process_device
from training_framework.session import TrainingSession
from tests.test_utils import (
    COMPONENTS_PACKAGE,
    register_test_components,
    resource_named,
)


def _cuda(monkeypatch, device_count: int) -> None:
    """Pretend this machine has `device_count` visible CUDA devices."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)


def _bindable_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _can_bind(port) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def _ddp_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "world_size": 2,
        "backend": "gloo",
        "master_addr": "127.0.0.1",
        "master_port": _bindable_port(),
        "parallel_components": [],
    }
    config.update(overrides)
    return config


def _clear_topology_env(monkeypatch) -> None:
    for name in ("WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    _clear_topology_env(monkeypatch)


def test_a_session_without_ddp_has_no_topology():
    assert resolve_launch_topology(None) is None


def test_a_new_sessions_config_outranks_the_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")

    topology = resolve_launch_topology(
        _ddp_config(world_size=2),
        from_checkpoint=False,
    )

    assert topology.world_size == 2


def test_a_checkpoints_world_size_yields_to_the_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")

    topology = resolve_launch_topology(
        _ddp_config(world_size=8),
        from_checkpoint=True,
    )

    assert topology.world_size == 4


def test_an_override_outranks_both(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")

    topology = resolve_launch_topology(
        _ddp_config(world_size=8),
        overrides={"world_size": "3"},
        from_checkpoint=True,
    )

    assert topology.world_size == 3


def test_resuming_falls_back_to_the_visible_device_count(monkeypatch):
    """The reported failure: trained on eight GPUs, resumed on four."""
    _cuda(monkeypatch, 4)

    with pytest.warns(UserWarning, match="only 4 CUDA device"):
        topology = resolve_launch_topology(
            _ddp_config(world_size=8, backend="nccl"),
            from_checkpoint=True,
        )

    assert topology.world_size == 4
    assert [topology.local_rank(rank) for rank in range(4)] == [0, 1, 2, 3]


def test_a_new_session_asking_for_absent_devices_is_rejected(monkeypatch):
    _cuda(monkeypatch, 4)

    with pytest.raises(ValueError, match="only 4 CUDA device"):
        resolve_launch_topology(
            _ddp_config(world_size=8, backend="nccl"),
            from_checkpoint=False,
        )


def test_an_override_asking_for_absent_devices_is_rejected(monkeypatch):
    _cuda(monkeypatch, 4)

    with pytest.raises(ValueError, match="only 4 CUDA device"):
        resolve_launch_topology(
            _ddp_config(world_size=4, backend="nccl"),
            overrides={"world_size": "8"},
            from_checkpoint=True,
        )


def test_gloo_never_claims_a_gpu(monkeypatch):
    _cuda(monkeypatch, 4)

    topology = resolve_launch_topology(_ddp_config(world_size=6))

    assert not topology.uses_cuda
    assert topology.world_size == 6
    assert [topology.local_rank(rank) for rank in range(6)] == [0, 1, 2, 3, 4, 5]
    assert pin_process_device(topology, 3) is None


def test_pinning_is_skipped_without_a_topology():
    assert pin_process_device(None, 0) is None


def test_local_rank_wraps_over_the_visible_devices(monkeypatch):
    _cuda(monkeypatch, 4)

    topology = LaunchTopology(
        world_size=8,
        backend="nccl",
        master_addr="127.0.0.1",
        master_port="29500",
        devices_per_node=4,
    )

    assert [topology.local_rank(rank) for rank in range(8)] == [
        0, 1, 2, 3, 0, 1, 2, 3,
    ]


def test_a_missing_world_size_is_rejected_for_a_new_session():
    config = _ddp_config()
    del config["world_size"]

    with pytest.raises(ValueError, match="must contain ddp.world_size"):
        resolve_launch_topology(config, from_checkpoint=False)


def test_a_configured_world_size_keeps_its_strict_typing():
    with pytest.raises(ValueError, match="positive integer"):
        resolve_launch_topology(_ddp_config(world_size="4"))


def test_an_environment_world_size_may_be_a_string(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    config = _ddp_config()
    del config["world_size"]

    assert resolve_launch_topology(config).world_size == 4


@pytest.mark.parametrize("world_size", [0, -1, True])
def test_an_unusable_world_size_is_rejected(world_size):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_launch_topology(_ddp_config(world_size=world_size))


def test_an_absent_master_port_is_left_to_the_launcher():
    config = _ddp_config()
    del config["master_port"]

    assert resolve_launch_topology(config).master_port is None


# ---------------------------------------------------------------------------
# Hosting the rendezvous
# ---------------------------------------------------------------------------


def test_an_absent_master_port_is_allocated_and_held_until_closed():
    config = _ddp_config()
    del config["master_port"]

    with host_rendezvous(resolve_launch_topology(config)) as hosted:
        port = hosted.topology.master_port
        assert 0 < int(port) < 65536
        assert hosted.topology.store_hosted
        # Bound from the moment it was chosen, so nothing else can take it
        # before the workers meet on it.
        assert not _can_bind(port)

    assert _can_bind(port)


def test_a_free_master_port_is_reused_quietly(recwarn):
    port = _bindable_port()

    with host_rendezvous(resolve_launch_topology(
        _ddp_config(master_port=port),
        from_checkpoint=True,
    )) as hosted:
        assert hosted.topology.master_port == port

    assert [str(warning.message) for warning in recwarn] == []


def test_a_taken_master_port_from_a_checkpoint_is_replaced():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = str(taken.getsockname()[1])

        with pytest.warns(UserWarning, match="already in use"):
            hosted = host_rendezvous(resolve_launch_topology(
                _ddp_config(master_port=port),
                from_checkpoint=True,
            ))

        with hosted:
            assert hosted.topology.master_port != port
            assert 0 < int(hosted.topology.master_port) < 65536


def test_a_configured_master_port_that_is_taken_is_refused():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = str(taken.getsockname()[1])

        with pytest.raises(
                RuntimeError,
                match=rf"ddp\.master_port={port} is already in use",
        ):
            host_rendezvous(resolve_launch_topology(
                _ddp_config(master_port=port),
            ))


@pytest.mark.parametrize("from_checkpoint", [False, True])
def test_an_address_this_machine_lacks_is_reported_at_once(from_checkpoint):
    # 192.0.2.0/24 is reserved for documentation and never assigned.
    config = _ddp_config(master_addr="192.0.2.1")

    with pytest.raises(RuntimeError, match="not an address of this machine"):
        host_rendezvous(resolve_launch_topology(
            config,
            from_checkpoint=from_checkpoint,
        ))


def _launch_config(tmp_path, **ddp_overrides: Any) -> dict[str, Any]:
    register_test_components()
    ddp = _ddp_config(**ddp_overrides)
    del ddp["parallel_components"]
    return {
        "session_config": {
            "rng_seed": 7,
            "sessions_dir": str(tmp_path),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "component_bindings": {"model": "it_3d45_model"},
        "ddp": ddp,
        "it_3d45_model": {},
    }


def _launcher(config) -> TrainingEngine:
    """An engine that registers `config` on entry and starts no worker."""
    return TrainingEngine(SimpleNamespace(
        mode="new",
        session_configs=[config],
        process_timeout_on_join=10,
        heartbeat_timeout=30,
        topology_overrides=None,
        debug=True,
    ))


def test_the_launcher_holds_the_rendezvous_port_until_it_exits(tmp_path):
    port = _bindable_port()

    with _launcher(_launch_config(tmp_path, master_port=port)):
        # Registered, no worker started yet: the port is already held.
        assert not _can_bind(port)

    assert _can_bind(port)


class _StopAtRendezvous(Exception):
    pass


def _rendezvous_call(monkeypatch, session) -> dict[str, Any]:
    """Enter `session` and return how `ddp` rendezvouses.

    Stops at the first rendezvous step, so nothing waits on a port: joining
    a store shows up as `{"store": ...}`, a plain rendezvous as the
    `init_process_group` keyword arguments.
    """
    calls = []

    def join_store(*args, **kwargs):
        calls.append({"store": (args, kwargs)})
        raise _StopAtRendezvous

    def init_process_group(*args, **kwargs):
        calls.append(kwargs)
        raise _StopAtRendezvous

    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, "TCPStore", join_store)
        patch.setattr(
            torch.distributed,
            "init_process_group",
            init_process_group,
        )
        with pytest.raises(_StopAtRendezvous):
            with session:
                pass
    [call] = calls
    return call


def test_only_a_launched_worker_joins_the_hosted_store(tmp_path, monkeypatch):
    config = _launch_config(tmp_path, world_size=1)
    state = TrainingSession(config).get_state()

    with host_rendezvous(resolve_launch_topology(config["ddp"])) as hosted:
        worker = load_session_for_worker(
            state,
            0,
            launch_topology=hosted.topology,
        )
        saved = worker.get_state()
        assert "store" in _rendezvous_call(monkeypatch, worker)

    # The worker's checkpoint says nothing about a store that is gone: driven
    # by hand, it rendezvouses on its own port as before.
    restored = TrainingSession.from_state(saved)
    call = _rendezvous_call(monkeypatch, restored)
    assert "store" not in call
    assert call["init_method"].startswith("tcp://127.0.0.1:")


def test_a_launch_whose_configured_port_is_taken_fails_at_registration(
        tmp_path,
):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = str(taken.getsockname()[1])

        with pytest.raises(RuntimeError, match=f"master_port={port}"):
            with _launcher(_launch_config(tmp_path, master_port=port)):
                pass


def test_an_override_replaces_the_master_endpoint():
    topology = resolve_launch_topology(
        _ddp_config(),
        overrides={"master_addr": "10.0.0.1", "master_port": "29777"},
        from_checkpoint=True,
    )

    assert (topology.master_addr, topology.master_port) == (
        "10.0.0.1",
        "29777",
    )


# ---------------------------------------------------------------------------
# Injection into a worker's session state
# ---------------------------------------------------------------------------


def _ddp_session_state(tmp_path, world_size: int = 2) -> dict[str, Any]:
    register_test_components()
    session = TrainingSession({
        "session_config": {
            "rng_seed": 7,
            "sessions_dir": str(tmp_path),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": COMPONENTS_PACKAGE,
        },
        "component_bindings": {"model": "it_3d45_model"},
        "ddp": _ddp_config(
            world_size=world_size,
            parallel_components=["model"],
        ),
        "it_3d45_model": {},
    })
    return session.get_state()


def _ddp_init_args(state: dict[str, Any]) -> dict[str, Any]:
    return state["components_state"]["ddp"]["init_args"]


def test_the_launch_topology_replaces_the_stored_one(tmp_path):
    state = _ddp_session_state(tmp_path, world_size=8)
    topology = LaunchTopology(
        world_size=4,
        backend="gloo",
        master_addr="10.0.0.1",
        master_port="29777",
        devices_per_node=0,
    )

    ddp = resource_named(
        load_session_for_worker(state, 2, launch_topology=topology),
        "ddp",
    )

    assert (ddp.rank, ddp.local_rank, ddp.world_size) == (2, 2, 4)
    assert ddp.config["master_addr"] == "10.0.0.1"
    assert ddp.config["master_port"] == "29777"


def test_the_session_config_agrees_with_the_constructor_arguments(tmp_path):
    """A later --extend-session diffs against the config, so it must match."""
    state = _ddp_session_state(tmp_path, world_size=8)
    topology = LaunchTopology(
        world_size=4,
        backend="gloo",
        master_addr="10.0.0.1",
        master_port="29777",
        devices_per_node=0,
    )

    session = load_session_for_worker(state, 0, launch_topology=topology)

    assert session.full_config["ddp"]["world_size"] == 4
    assert session.full_config["ddp"]["master_port"] == "29777"
    assert session.full_config["ddp"]["parallel_components"] == ["model"]


def test_the_original_state_is_left_alone(tmp_path):
    state = _ddp_session_state(tmp_path, world_size=8)
    topology = LaunchTopology(
        world_size=4,
        backend="gloo",
        master_addr="127.0.0.1",
        master_port="29500",
        devices_per_node=0,
    )

    load_session_for_worker(state, 1, launch_topology=topology)

    assert _ddp_init_args(state)["args"][0]["world_size"] == 8
    assert state["config"]["ddp"]["world_size"] == 8


def test_without_a_topology_only_the_rank_is_settled(tmp_path):
    state = _ddp_session_state(tmp_path, world_size=2)

    ddp = resource_named(load_session_for_worker(state, 1), "ddp")

    # With no topology to place it, the rank is also its local device index.
    assert (ddp.rank, ddp.local_rank, ddp.world_size) == (1, 1, 2)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_resuming_accepts_topology_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "pytest",
        "--resume-session", "checkpoint",
        "--override", "ddp.world_size=4", "ddp.master_port=29777",
    ])

    configurator = Configurator()

    assert configurator.mode == "resume"
    assert configurator.topology_overrides == {
        "world_size": "4",
        "master_port": "29777",
    }


def test_resuming_rejects_session_configuration_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "pytest",
        "--resume-session", "checkpoint",
        "--override", "optimizer.kwargs.lr=0.1",
    ])

    with pytest.raises(SystemExit):
        Configurator()


def test_extending_separates_topology_from_session_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "pytest",
        "--extend-session", "checkpoint",
        "--override",
        "ddp.world_size=4",
        "session_config.max_iterations=10",
    ])

    configurator = Configurator()

    assert configurator.topology_overrides == {"world_size": "4"}
    assert configurator.extension_overrides == (
        "session_config.max_iterations=10",
    )


def test_extending_with_only_a_topology_override_is_allowed(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "pytest",
        "--extend-session", "checkpoint",
        "--override", "ddp.world_size=4",
    ])

    configurator = Configurator()

    assert configurator.topology_overrides == {"world_size": "4"}
    assert configurator.extension_overrides == ()


def test_an_error_inside_the_engine_is_not_masked(tmp_path):
    with pytest.raises(ValueError, match="original body error"):
        with _launcher(_launch_config(tmp_path)):
            # No worker has started; leaving must not trip over that.
            raise ValueError("original body error")


def test_a_failed_entry_releases_the_ports_it_held(tmp_path):
    first = _launch_config(tmp_path / "first")
    second = deepcopy(first)
    second["ddp"]["master_port"] = _bindable_port()
    second["session_config"]["sessions_dir"] = str(tmp_path / "second")
    second["not_a_registered_component"] = {}
    engine = TrainingEngine(SimpleNamespace(
        mode="new",
        session_configs=[first, second],
        process_timeout_on_join=10,
        heartbeat_timeout=30,
        topology_overrides=None,
        debug=True,
    ))

    with pytest.raises(ValueError, match="not_a_registered_component"):
        with engine:
            pass

    # The first session's port was bound before the second one failed.
    assert _can_bind(first["ddp"]["master_port"])
