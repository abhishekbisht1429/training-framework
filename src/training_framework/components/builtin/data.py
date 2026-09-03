from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch
from torch.utils.data import DataLoader

from training_framework.components import StatefulResource
from training_framework.dataloader import DistributedInfiniteSampler
from training_framework.components import requires_resource, resource

if TYPE_CHECKING:
    from training_framework.session import Session


class _ManagedDataIterator:
    """Track delivered batches and own the DataLoader iterator lifecycle."""

    def __init__(self, data_manager: DataManager, dataloader: DataLoader):
        self._data_manager = data_manager
        self._iterator = iter(dataloader)
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration

        batch = next(self._iterator)
        self._data_manager._record_batch_delivery()
        return batch

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        iterator = self._iterator
        self._iterator = None
        shutdown_workers: Any = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()


@requires_resource("ddp")
@requires_resource("dataset")
@resource("data_manager", session_type="training")
class DataManager(StatefulResource):

    def __init__(self, config):
        self._batch_size = config["batch_size"]
        self._num_workers = config["num_workers"]
        self._pin_memory = config["pin_memory"]
        self._data_iter: _ManagedDataIterator | None = None
        self._sampler_state: dict[str, Any] | None = None
        self._local_batch_size: int | None = None
        self._samples_per_rank: int | None = None

        if (
                isinstance(self._batch_size, bool)
                or not isinstance(self._batch_size, int)
                or self._batch_size <= 0
        ):
            raise ValueError(
                "DataManager batch_size must be a positive integer"
            )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def data_iter(self):
        return self._data_iter

    def _validate_setup(self, dataset_size: int, world_size: int) -> None:
        if dataset_size <= 0:
            raise ValueError("DataManager requires a non-empty dataset")
        if (
                isinstance(world_size, bool)
                or not isinstance(world_size, int)
                or world_size <= 0
        ):
            raise ValueError(
                "DataManager requires a positive integer world_size"
            )
        if self.batch_size < world_size:
            raise ValueError(
                "DataManager batch_size must be at least the DDP world_size "
                "so every rank receives a sample"
            )
        if self.batch_size % world_size != 0:
            raise ValueError(
                "DataManager batch_size must be divisible by the DDP "
                "world_size"
            )

    def _restore_sampler(
            self,
            sampler: DistributedInfiniteSampler,
            dataset_size: int,
            rank: int,
            world_size: int,
    ) -> None:
        if self._sampler_state is None:
            return

        restored_state = deepcopy(self._sampler_state)
        if restored_state.get("num_samples") != dataset_size:
            raise ValueError(
                "Cannot restore DataManager state with a different dataset "
                "size"
            )
        if restored_state.get("world_size") != world_size:
            raise ValueError(
                "Cannot restore DataManager state with a different DDP "
                "world_size"
            )

        restored_state["rank"] = rank
        sampler.set_state(restored_state)

    def _record_batch_delivery(self) -> None:
        if (
                self._sampler_state is None
                or self._local_batch_size is None
                or self._samples_per_rank is None
        ):
            raise RuntimeError("DataManager is not set up")

        position = (
            self._sampler_state["index_within_epoch"]
            + self._local_batch_size
        )
        completed_epochs, index_within_epoch = divmod(
            position,
            self._samples_per_rank,
        )
        self._sampler_state["epoch"] += completed_epochs
        self._sampler_state["index_within_epoch"] = index_within_epoch

    @override
    def setup(self, session: Session):
        ddp = session.get_resource("ddp")
        dataset = session.get_resource("dataset")
        dataset_size = len(dataset)
        world_size = ddp.world_size
        self._validate_setup(dataset_size, world_size)
        collate_fn = getattr(dataset, "collate_fn", torch.stack)
        if not callable(collate_fn):
            raise TypeError(
                f"Dataset resource '{type(dataset).__name__}' collate_fn "
                "must be callable"
            )

        sampler = DistributedInfiniteSampler(
            num_samples=dataset_size,
            rank=ddp.rank,
            world_size=world_size,
        )
        self._restore_sampler(
            sampler,
            dataset_size,
            ddp.rank,
            world_size,
        )

        self._local_batch_size = self.batch_size // world_size
        self._samples_per_rank = sampler.num_samples_per_rank
        self._sampler_state = sampler.get_state()

        dataloader = DataLoader(
            dataset,
            batch_size=self._local_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        self._data_iter = _ManagedDataIterator(self, dataloader)

    @override
    def teardown(self, session: Session):
        try:
            if self._data_iter is not None:
                self._data_iter.close()
        finally:
            self._data_iter = None
            self._local_batch_size = None
            self._samples_per_rank = None

    @override
    def get_state(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "sampler_state": deepcopy(self._sampler_state),
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:
        if state["batch_size"] != self.batch_size:
            raise ValueError(
                "Cannot restore DataManager state with a different batch_size"
            )
        self._sampler_state = deepcopy(state["sampler_state"])
