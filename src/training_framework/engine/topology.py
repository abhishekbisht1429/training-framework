from __future__ import annotations

import errno
import os
import socket
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from training_framework.components.builtin.distributed import (
    RENDEZVOUS_TIMEOUT,
)


#: `ddp` configuration keys that describe the launch rather than the run.
TOPOLOGY_KEYS = ("world_size", "master_addr", "master_port")

_DEFAULT_MASTER_ADDR = "127.0.0.1"


@dataclass(frozen=True)
class LaunchTopology:
    """How many processes a launch runs, and where they find each other.

    A checkpoint records what was learned and how far it got, never the
    machine it ran on, so none of this is restored from one. It is resolved
    fresh on every launch and injected into the `ddp` component just before
    the workers are built, which is what lets a run trained on eight GPUs
    resume on four.
    """

    world_size: int
    backend: str
    master_addr: str
    #: `None` until the launcher binds one: see `host_rendezvous`.
    master_port: str | None
    devices_per_node: int
    #: The port came from a checkpoint, which describes the machine that
    #: wrote it, so a taken one may be replaced rather than refused.
    master_port_from_checkpoint: bool = False
    #: The launcher holds a rendezvous store on `master_addr:master_port`,
    #: and the workers join it as clients.
    store_hosted: bool = False

    @property
    def uses_cuda(self) -> bool:
        return self.backend == "nccl" and torch.cuda.is_available()

    def local_rank(self, rank: int) -> int:
        """The CUDA ordinal `rank` runs on.

        Ordinals are relative to `CUDA_VISIBLE_DEVICES`, so this is a
        position within the visible devices, not a physical GPU.
        """
        if not self.uses_cuda or self.devices_per_node <= 0:
            return rank
        return rank % self.devices_per_node

    def config_overlay(self) -> dict[str, Any]:
        """The `ddp` config entries this launch decides."""
        return {
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
        }


def _pick(
        name: str,
        env_name: str,
        overrides: Mapping[str, Any],
        stored: Mapping[str, Any],
        *,
        stored_is_stale: bool,
        default: Any = None,
) -> tuple[Any, str]:
    """Resolve one field, returning the value and where it came from.

    The environment outranks the stored value only when that value came out
    of a checkpoint and is therefore stale. A config file is this launch's
    explicit statement, so it wins over an ambient environment variable.
    """
    if overrides.get(name) is not None:
        return overrides[name], "override"

    env_value = os.environ.get(env_name) or None
    stored_value = stored.get(name)

    if stored_is_stale and env_value is not None:
        return env_value, "env"
    if stored_value is not None:
        return stored_value, "config"
    if env_value is not None:
        return env_value, "env"
    return default, "default"


def _coerce_world_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("ddp.world_size must be a positive integer")
    try:
        world_size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ddp.world_size must be a positive integer"
        ) from error
    if world_size < 1:
        raise ValueError("ddp.world_size must be a positive integer")
    return world_size


def resolve_launch_topology(
        ddp_config: Mapping[str, Any] | None,
        *,
        overrides: Mapping[str, Any] | None = None,
        from_checkpoint: bool = False,
) -> LaunchTopology | None:
    """Decide this launch's process topology.

    `ddp_config` is the `ddp` component configuration, either from the config
    file of a new session or from the checkpoint of one being resumed;
    `from_checkpoint` says which, because a stored value is stale in the
    second case and authoritative in the first. Returns `None` for a session
    with no `ddp` component, which runs in a single process.
    """
    if ddp_config is None:
        return None

    overrides = dict(overrides or {})
    stored = dict(ddp_config)
    backend = str(stored.get("backend", "gloo"))
    devices_per_node = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    uses_cuda = backend == "nccl" and torch.cuda.is_available()

    raw_world_size, world_size_source = _pick(
        "world_size",
        "WORLD_SIZE",
        overrides,
        stored,
        stored_is_stale=from_checkpoint,
        # Falling back to the visible device count is a convenience for
        # resuming on an unknown machine. A new session still has to say how
        # many processes it wants.
        default=devices_per_node if (uses_cuda and from_checkpoint) else None,
    )
    if raw_world_size is None:
        raise ValueError("DDP configuration must contain ddp.world_size")
    if world_size_source == "config":
        # A configured value keeps the stricter typing it has always had;
        # only an environment variable or a command-line override may be a
        # string.
        if (
                isinstance(raw_world_size, bool)
                or not isinstance(raw_world_size, int)
                or raw_world_size < 1
        ):
            raise ValueError("ddp.world_size must be a positive integer")
        world_size = raw_world_size
    else:
        world_size = _coerce_world_size(raw_world_size)

    if uses_cuda and world_size > devices_per_node:
        if world_size_source == "config" and from_checkpoint:
            warnings.warn(
                f"The checkpoint was written with ddp.world_size="
                f"{world_size}, but only {devices_per_node} CUDA device(s) "
                f"are visible; resuming with ddp.world_size="
                f"{devices_per_node}. Pass --override ddp.world_size=N to "
                "choose a different size.",
                stacklevel=2,
            )
            world_size = devices_per_node
        else:
            raise ValueError(
                f"ddp.world_size is {world_size}, but only "
                f"{devices_per_node} CUDA device(s) are visible to this "
                "launch. Reduce ddp.world_size or make more devices visible "
                "through CUDA_VISIBLE_DEVICES."
            )

    master_addr, _ = _pick(
        "master_addr",
        "MASTER_ADDR",
        overrides,
        stored,
        stored_is_stale=from_checkpoint,
        default=_DEFAULT_MASTER_ADDR,
    )
    master_port, master_port_source = _pick(
        "master_port",
        "MASTER_PORT",
        overrides,
        stored,
        stored_is_stale=from_checkpoint,
    )
    # No port is chosen here. Choosing one and binding it later, in a
    # worker, leaves a window in which anything on the machine can take it;
    # `host_rendezvous` binds it and keeps it bound instead.
    return LaunchTopology(
        world_size=world_size,
        backend=backend,
        master_addr=str(master_addr),
        master_port=None if master_port is None else str(master_port),
        devices_per_node=devices_per_node,
        master_port_from_checkpoint=(
            master_port_source == "config" and from_checkpoint
        ),
    )


class HostedRendezvous:
    """A rendezvous store the launcher holds for one launch's workers.

    The port is bound from the moment it is chosen until `close`, so no other
    process can take it before the workers meet on it.
    """

    def __init__(self, store, topology: LaunchTopology):
        self._store = store
        self.topology = topology

    def close(self) -> None:
        """Release the port. Call once every worker has left the group."""
        self._store = None

    def __enter__(self) -> "HostedRendezvous":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _listen(address: str, port: int) -> socket.socket:
    """Bind and listen on `address:port`, or raise the `OSError` saying why."""
    family, kind, proto, _, sockaddr = socket.getaddrinfo(
        address,
        port,
        type=socket.SOCK_STREAM,
    )[0]
    sock = socket.socket(family, kind, proto)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(sockaddr)
        sock.listen(socket.SOMAXCONN)
    except BaseException:
        sock.close()
        raise
    return sock


def _bind_failure(topology: LaunchTopology, port: int, error: OSError) -> str:
    address = topology.master_addr
    if isinstance(error, socket.gaierror):
        return f"ddp.master_addr={address} could not be resolved: {error}."
    if error.errno == errno.EADDRINUSE:
        return (
            f"ddp.master_port={port} is already in use on {address}. Choose "
            "another port, or leave ddp.master_port out and the launcher "
            "picks a free one."
        )
    if error.errno == errno.EADDRNOTAVAIL:
        return (
            f"ddp.master_addr={address} is not an address of this machine. "
            "The engine runs every rank here, so the rendezvous has to be on "
            "a local address such as 127.0.0.1."
        )
    return (
        f"Could not open the rendezvous on {address}:{port}: "
        f"{error.strerror or error}."
    )


def host_rendezvous(topology: LaunchTopology) -> HostedRendezvous:
    """Bind this launch's rendezvous port and hold it for the workers.

    With no port configured, the operating system picks a free one. A
    configured port that cannot be bound is an error, reported now rather
    than by rank 0 once the other ranks are already waiting -- except a port
    that came from a checkpoint and is merely in use: it belonged to the
    machine that wrote the checkpoint, so it is replaced with a warning.
    """
    requested = 0 if topology.master_port is None else int(topology.master_port)
    # The socket is bound here, not by `TCPStore`, so the reason a bind fails
    # is known (a store asked for an address this machine lacks waits out
    # its whole timeout instead of failing).
    try:
        sock = _listen(topology.master_addr, requested)
    except OSError as error:
        if not (
                error.errno == errno.EADDRINUSE
                and topology.master_port_from_checkpoint
        ):
            raise RuntimeError(
                _bind_failure(topology, requested, error),
            ) from error
        sock = _listen(topology.master_addr, 0)
        warnings.warn(
            f"ddp.master_port={requested} from the checkpoint is already in "
            f"use; rendezvous will use port {sock.getsockname()[1]} instead.",
            stacklevel=2,
        )

    port = sock.getsockname()[1]
    try:
        store = dist.TCPStore(
            topology.master_addr,
            port,
            topology.world_size,
            is_master=True,
            wait_for_workers=False,
            timeout=RENDEZVOUS_TIMEOUT,
            master_listen_fd=sock.fileno(),
        )
    except BaseException:
        sock.close()
        raise
    # The store owns the descriptor now and closes it when released.
    sock.detach()
    return HostedRendezvous(
        store,
        replace(topology, master_port=str(port), store_hosted=True),
    )


def _configured_cuda_device(session_state: Any) -> torch.device | None:
    """The CUDA device the session configuration asks for, if any.

    A session can use a GPU without nccl -- a single-process run, or a gloo
    group over CUDA tensors -- and its RNG stream has to land on that device
    just the same.
    """
    if not isinstance(session_state, Mapping):
        return None
    config = session_state.get("config")
    if not isinstance(config, Mapping):
        return None
    session_config = config.get("session_config")
    if not isinstance(session_config, Mapping):
        return None

    device = session_config.get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        return None
    return torch.device(device)


def pin_process_device(
        topology: LaunchTopology | None,
        rank: int,
        session_state: Any = None,
) -> torch.device | None:
    """Pin this worker to its CUDA device before anything is constructed.

    Doing it here rather than in `DDPResource.setup` means no component can
    be built or restored against the wrong device, and the RNG restore has
    somewhere definite to land. A run that wants no GPU pins none, so a
    `gloo` group over CPU tensors never claims one.
    """
    if not torch.cuda.is_available():
        return None

    if topology is not None and topology.uses_cuda:
        local_rank = topology.local_rank(rank)
        if local_rank >= topology.devices_per_node:
            raise ValueError(
                f"Rank {rank} resolves to CUDA device {local_rank}, but only "
                f"{topology.devices_per_node} device(s) are visible"
            )
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    device = _configured_cuda_device(session_state)
    if device is None:
        return None

    torch.cuda.set_device(device)
    return device
