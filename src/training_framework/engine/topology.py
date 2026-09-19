from __future__ import annotations

import os
import socket
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


#: `ddp` configuration keys that describe the launch rather than the run.
TOPOLOGY_KEYS = ("world_size", "master_addr", "master_port")

_DEFAULT_MASTER_ADDR = "127.0.0.1"


def available_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_DEFAULT_MASTER_ADDR, 0))
        return str(sock.getsockname()[1])


def _port_is_free(port: Any) -> bool:
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((_DEFAULT_MASTER_ADDR, port_number))
    except OSError:
        return False
    return True


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
    master_port: str
    devices_per_node: int

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
    if master_port is None:
        master_port = available_local_port()
    elif (
            master_port_source == "config"
            and from_checkpoint
            and not _port_is_free(master_port)
    ):
        # The stored port belonged to the machine that wrote the checkpoint
        # and is taken here. Say so rather than failing in the rendezvous.
        replacement = available_local_port()
        warnings.warn(
            f"ddp.master_port={master_port} from the checkpoint is already "
            f"in use; rendezvous will use port {replacement} instead.",
            stacklevel=2,
        )
        master_port = replacement

    return LaunchTopology(
        world_size=world_size,
        backend=backend,
        master_addr=str(master_addr),
        master_port=str(master_port),
        devices_per_node=devices_per_node,
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
