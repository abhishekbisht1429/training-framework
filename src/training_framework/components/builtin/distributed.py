from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from training_framework.components import Resource, requires_resource, resource
from training_framework.util import (
    context_entry,
    context_exit,
    requires_context,
)

if TYPE_CHECKING:
    from training_framework.session import Session


@requires_resource("model")
@resource("ddp", session_type="training")
class DDPResource(Resource):

    def __init__(self, config: dict, rank: int = -1):
        self._config = config
        self._world_size = config["world_size"]
        self._backend = config["backend"]
        self._rank = rank
        self._parallel_components = config.get("parallel_components", [])
        self._master_addr = config["master_addr"]
        self._master_port = config["master_port"]
        self._ddp_wrapped_model = None

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
    def parallel_components(self):
        return deepcopy(self._parallel_components)

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    @requires_context
    def wrapped_model(self):
        return self._ddp_wrapped_model

    @context_entry
    @override
    def setup(self, session: Session) -> Any:
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = self._master_port

        uses_cuda = self._backend == "nccl" and torch.cuda.is_available()
        if uses_cuda:
            torch.cuda.set_device(self._rank)
            session.set_device(torch.device("cuda", self._rank))

        torch.distributed.init_process_group(
            backend=self._backend,
            rank=self._rank,
            world_size=self._world_size,
        )

        try:
            model = session.get_resource("model")
            if uses_cuda:
                model.to(session.device)
            device_ids = [self.rank] if uses_cuda else None
            self._ddp_wrapped_model = DDP(model, device_ids=device_ids)
        except Exception:
            torch.distributed.destroy_process_group()
            raise

    @context_exit
    @override
    def teardown(self, session):
        self._ddp_wrapped_model = None
        torch.distributed.destroy_process_group()
