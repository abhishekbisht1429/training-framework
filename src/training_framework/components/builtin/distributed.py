from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import timedelta
from typing import TYPE_CHECKING, Any, override

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from training_framework.components import (
    Resource,
    requires_resource,
    resource,
    role,
    singleton,
)
from training_framework.util import requires_context

#: How long a worker waits to reach the launcher's rendezvous store.
RENDEZVOUS_TIMEOUT = timedelta(minutes=5)

if TYPE_CHECKING:
    from training_framework.session import Session


role(
    "model",
    Resource,
    description="the model being trained; a Resource exposing an nn.Module",
    session_type="training",
)


def _tcp_init_method(master_addr: str, master_port) -> str:
    """Return the `tcp://` rendezvous address for a process group.

    Handed to `init_process_group` directly so that nothing is written to
    `MASTER_ADDR` / `MASTER_PORT`: the environment is process-wide, and launch
    topology deliberately lets it outrank a checkpoint's stored port, so a
    value left behind would become the next run's rendezvous. An IPv6 host is
    bracketed, as a URL requires.
    """
    host = str(master_addr)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"tcp://{host}:{master_port}"


def _component_name_list(config: Mapping, key: str) -> list[str] | None:
    """Return `config[key]` as a list of component names, or None if absent.

    A bare string is the mistake worth naming: `rank_zero_components: logger`
    is a valid YAML scalar that would otherwise be iterated one character at
    a time, quietly asking for components called 'l', 'o', 'g'.
    """
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str) or isinstance(value, Mapping) or not isinstance(
            value, Iterable,
    ):
        raise ValueError(
            f"ddp.{key} must be a list of component names, got "
            f"{type(value).__name__}. Write it as a YAML list, one name per "
            "line."
        )
    names = list(value)
    invalid = [name for name in names if not isinstance(name, str)]
    if invalid:
        raise ValueError(
            f"ddp.{key} must contain component names, got {invalid}"
        )
    return names


@singleton
@requires_resource("model")
@resource("ddp", session_type="training")
class DDPResource(Resource):

    def __init__(self, config: dict, rank: int = -1, local_rank: int | None = None):
        self._config = config
        self._world_size = config["world_size"]
        self._backend = config["backend"]
        self._rank = rank
        # The CUDA ordinal this rank runs on. It is the rank itself on a
        # single-node launch, and the engine settles it from the launch
        # topology; a hand-constructed resource keeps the old behaviour.
        self._local_rank = rank if local_rank is None else local_rank
        self._parallel_components = _component_name_list(
            config,
            "parallel_components",
        )
        self._rank_zero_components = (
            _component_name_list(config, "rank_zero_components") or []
        )
        if self._parallel_components is not None:
            warnings.warn(
                "ddp.parallel_components is deprecated. Secondary ranks now "
                "build every configured component except those marked with "
                "@rank_zero_only or named in ddp.rank_zero_components; remove "
                "parallel_components to get that behaviour.",
                FutureWarning,
                stacklevel=2,
            )
        self._master_addr = config["master_addr"]
        self._master_port = config["master_port"]
        self._ddp_wrapped_model = None
        # Set by the launcher's worker, never by configuration, and never
        # saved; see `join_hosted_store`.
        self._joins_hosted_store = False
        self._store = None

    @property
    def backend(self):
        return self._backend

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def parallel_components(self):
        """The deprecated opt-in rank list; empty when the session omits it."""
        return deepcopy(self._parallel_components or [])

    @property
    def declares_parallel_components(self) -> bool:
        """Whether the session set the deprecated list at all.

        An explicit empty list still decides the rank set, so it has to be
        told apart from the key being absent.
        """
        return self._parallel_components is not None

    @property
    def rank_zero_components(self):
        """Components this session keeps off ranks other than zero."""
        return deepcopy(self._rank_zero_components)

    @property
    def config(self):
        return deepcopy(self._config)

    def join_hosted_store(self) -> None:
        """Meet the other ranks on the store the launcher is holding.

        The engine binds the rendezvous port before it starts any worker and
        keeps it bound, so no other process can take it in between. Without
        this, `setup` has rank 0 bind the port itself, as a session driven by
        hand needs.
        """
        self._joins_hosted_store = True

    @property
    @requires_context
    def wrapped_model(self):
        return self._ddp_wrapped_model

    @override
    def setup(self, session: Session) -> Any:
        uses_cuda = self._backend == "nccl" and torch.cuda.is_available()
        if uses_cuda:
            device_count = torch.cuda.device_count()
            if self._local_rank >= device_count:
                raise ValueError(
                    f"Rank {self._rank} needs CUDA device "
                    f"{self._local_rank}, but only {device_count} device(s) "
                    "are visible. Reduce ddp.world_size or make more "
                    "devices visible through CUDA_VISIBLE_DEVICES."
                )
            # Usually already pinned before the session was built; repeating
            # it keeps a hand-driven session working.
            torch.cuda.set_device(self._local_rank)
            session.set_device(torch.device("cuda", self._local_rank))

        if self._joins_hosted_store:
            self._store = torch.distributed.TCPStore(
                self._master_addr,
                int(self._master_port),
                self._world_size,
                is_master=False,
                timeout=RENDEZVOUS_TIMEOUT,
            )
            torch.distributed.init_process_group(
                backend=self._backend,
                store=self._store,
                rank=self._rank,
                world_size=self._world_size,
            )
        else:
            torch.distributed.init_process_group(
                backend=self._backend,
                init_method=_tcp_init_method(
                    self._master_addr,
                    self._master_port,
                ),
                rank=self._rank,
                world_size=self._world_size,
            )

        try:
            model = self.get_dependency("model")
            if uses_cuda:
                model.to(session.device)
            device_ids = [self._local_rank] if uses_cuda else None
            self._ddp_wrapped_model = DDP(model, device_ids=device_ids)
        except Exception:
            torch.distributed.destroy_process_group()
            self._store = None
            raise

    @override
    def teardown(self, session):
        self._ddp_wrapped_model = None
        torch.distributed.destroy_process_group()
        self._store = None
