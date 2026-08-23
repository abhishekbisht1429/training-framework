from __future__ import annotations

import os
import subprocess
import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training_framework.dataloader import DistributedInfiniteSampler
from training_framework.registry import hook, resource, requires_resource
from training_framework.components import Stateful, LifecycleHook, Resource, \
    StatefulResource, StatefulLifeCycleHook
from training_framework.util import timestamp_str, context_entry, context_exit, requires_context
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from training_framework.training_session import TrainingSession


@hook("checkpointer")
class Checkpointer(LifecycleHook, Stateful):

    def __init__(self, config: dict):
        self._config = config
        self._checkpoints_dir = None
        self.call_every = config['checkpoint_every']

    @override
    def setup(self, session: TrainingSession) -> Any:
        if "checkpoints_dir" in self._config:
            self._checkpoints_dir = self._config["checkpoints_dir"]
        else:
            self._checkpoints_dir = os.path.join(session.session_config.session_dir, 'checkpoints')

        # Create directory if it doesn't exist
        os.makedirs(self._checkpoints_dir, exist_ok=True)

    @override
    def teardown(self, session):
        pass

    @override
    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    @override
    def post_iteration_callback(self, session: "TrainingSession") -> None:
        if (
            session.iteration == 1
            and session.session_config.max_iterations > 1
            and not self._config.get("checkpoint_first", False)
        ):
            return

        print("Creating checkpoint...")
        # File path
        filepath = os.path.join(self._checkpoints_dir, timestamp_str())

        # Save
        torch.save(session, filepath)


    @override
    def get_state(self) -> Any:
        return {
            'config': self._config,
        }

    @override
    def set_state(self, state: Any) -> None:
        self._config = state['config']
        self.call_every = self._config['checkpoint_every']

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu") -> TrainingSession:
        return torch.load(path, map_location=map_location, weights_only=False)

@hook("logger")
class Logger(LifecycleHook):

    def __init__(self, config: dict):
        self._config = config
        self.call_every = config['log_every']
        self._log_file = config['log_file'] if 'log_file' in config else sys.stdout

    def setup(self, session: TrainingSession) -> Any:
        if self._log_file is not sys.stdout:
            try:
                self._log_file = open(self._config['log_file'], 'w')
            except FileNotFoundError:
                print(f"Unable to open log file for writing to {self._config['log_file']}")
                pass

    def teardown(self, session) -> None:
        if self._log_file is not sys.stdout:
            self._log_file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self._log_file)

    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        self.print(f"Iteration {session.iteration}/{session.session_config.max_iterations}")

    def post_iteration_callback(self, session: "TrainingSession") -> None:
        pass

    def __getstate__(self) -> Any:
        return {
            'config': self._config,
        }

    def __setstate__(self, state: Any) -> None:
        self._config = state['config']
        self.call_every = self._config['log_every']

@resource("tensorboard")
class Tensorboard(Resource):

    def __init__(self, config: dict):
        self._config = config
        self._tb_process: subprocess.Popen | None = None
        self._tb_summary_writer = None

    @property
    def summary_writer(self):
        return self._tb_summary_writer

    def setup(self, session: "TrainingSession"):
        if "logdir" in self._config:
            logdir = self._config["logdir"]
        else:
            logdir = session.session_config.session_dir

        tensorboard_args = [
            "tensorboard",
            "--logdir", logdir,
            "--host", self._config["host"],
            "--port", str(self._config["port"]),
        ]
        print("Tensorboard Arguments: ", " ".join(tensorboard_args))
        self._tb_process = subprocess.Popen(tensorboard_args)
        self._tb_summary_writer = SummaryWriter(
            log_dir=os.path.join(session.session_config.session_dir, f"{type(self).__name__}_tensorboard"))
        time.sleep(3)
        if self._tb_process.poll() is not None:
            print("Failed to start tensorboard process...")
            raise RuntimeError("Failed to start tensorboard process...")

    def teardown(self, session):
        print("releasing resources...")
        self._tb_summary_writer.close()
        self._tb_process.terminate()
        print("resources released...")

        self._tb_summary_writer = None

        if self._tb_process is not None:
            self._tb_process.terminate()

@requires_resource("model")
@resource("ddp")
class DDPResource(Resource):

    def __init__(self, config: dict, rank: int=-1):
        self._config = config
        self._world_size = config['world_size']
        self._backend = config['backend']
        self._rank = rank
        self._parallel_components = config['parallel_components'] if 'parallel_components' in config else []
        self._master_addr = config['master_addr']
        self._master_port = config['master_port']

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
    def setup(self, session: TrainingSession) -> Any:
        # Address and port where Rank 0 is hosted (must be reachable by all processes)
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = self._master_port

        # Explicitly set the CUDA device for this process (1 process per GPU strategy)
        uses_cuda = self._backend == "nccl" and torch.cuda.is_available()
        if uses_cuda:
            torch.cuda.set_device(self._rank)
            session.set_device(torch.device("cuda", self._rank))

        # Initialize the default process group
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=self._rank,
            world_size=self._world_size
        )

        try:
            # Wrap the model only after the process group is available.
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

@hook("optimizer")
@requires_resource("ddp")
class OptimizerHook(StatefulLifeCycleHook):

    def __init__(self, config):
        # ====== required, do not alter ===========
        self.call_every = 1
        # =========================================

        self._learning_rate = config['learning_rate']
        self._weight_decay = config['weight_decay']
        self._warmup_iters = config['warmup_iters']
        self._optimizer = None
        self._lr_scheduler = None
        self._restored_state = None

    def _prepare_scheduler(self, optimizer, max_iter):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=self._warmup_iters
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iter - self._warmup_iters
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self._warmup_iters]
        )

        return scheduler

    @override
    def setup(self, session: "TrainingSession"):
        ddp_model: nn.Module = session.get_resource("ddp")
        self._optimizer = optim.AdamW(
            ddp_model.wrapped_model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay
        )
        self._lr_scheduler = self._prepare_scheduler(self._optimizer, session.session_config.max_iterations)
        self._restore_state()

    def _restore_state(self):
        if self._restored_state is None:
            return

        optimizer_state = self._restored_state.get('optimizer_state')
        if optimizer_state is not None:
            self._optimizer.load_state_dict(optimizer_state)

        scheduler_state = self._restored_state.get('lr_scheduler_state')
        if scheduler_state is not None:
            self._lr_scheduler.load_state_dict(scheduler_state)

        self._restored_state = None

    @override
    def pre_iteration_callback(self, session: "TrainingSession") -> None:
        self._optimizer.zero_grad()

    @override
    def post_iteration_callback(self, session: "TrainingSession") -> None:
        # print("calling optimizing step...")
        loss = session.iteration_context['loss']
        loss.backward()

        self._optimizer.step()
        self._lr_scheduler.step()

    @override
    def teardown(self, session: "TrainingSession"):
        self._restored_state = self.get_state()
        self._optimizer = None
        self._lr_scheduler = None

    @override
    def set_state(self, state: Any) -> None:
        self._restored_state = deepcopy(state)
        if self._optimizer is not None and self._lr_scheduler is not None:
            self._restore_state()

    @override
    def get_state(self) -> Any:
        if self._optimizer is None and self._lr_scheduler is None:
            if self._restored_state is not None:
                return deepcopy(self._restored_state)
            return {
                'optimizer_state': None,
                'lr_scheduler_state': None,
            }
        return {
            'optimizer_state': self._optimizer.state_dict() if self._optimizer else None,
            'lr_scheduler_state': self._lr_scheduler.state_dict() if self._lr_scheduler else None,
        }


class _ManagedDataIterator:
    """Track delivered batches and own the DataLoader iterator lifecycle."""

    def __init__(self, data_manager: "DataManager", dataloader: DataLoader):
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
        shutdown_workers: Any = getattr(
            iterator,
            "_shutdown_workers",
            None,
        )
        if callable(shutdown_workers):
            shutdown_workers()


@requires_resource("ddp")
@requires_resource("dataset")
@resource("data_manager")
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
            raise ValueError("DataManager batch_size must be a positive integer")

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def data_iter(self):
        return self._data_iter

    @classmethod
    def collate_fn(cls, batch):
        return torch.stack(batch)

    def _validate_setup(self, dataset_size: int, world_size: int) -> None:
        if dataset_size <= 0:
            raise ValueError("DataManager requires a non-empty dataset")
        if (
                isinstance(world_size, bool)
                or not isinstance(world_size, int)
                or world_size <= 0
        ):
            raise ValueError("DataManager requires a positive integer world_size")
        if self.batch_size < world_size:
            raise ValueError(
                "DataManager batch_size must be at least the DDP world_size "
                "so every rank receives a sample"
            )
        if self.batch_size % world_size != 0:
            raise ValueError(
                "DataManager batch_size must be divisible by the DDP world_size"
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
                "Cannot restore DataManager state with a different dataset size"
            )
        if restored_state.get("world_size") != world_size:
            raise ValueError(
                "Cannot restore DataManager state with a different DDP world_size"
            )

        # A rank-zero checkpoint is restored independently in every spawned
        # worker. Preserve its logical position while selecting this worker's
        # rank-local shard.
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
    def setup(self, session: TrainingSession):
        ddp = session.get_resource("ddp")
        dataset = session.get_resource("dataset")
        dataset_size = len(dataset)
        world_size = ddp.world_size
        self._validate_setup(dataset_size, world_size)

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
        # Capture state before DataLoader workers can prefetch sampler indices.
        self._sampler_state = sampler.get_state()

        dataloader = DataLoader(
            dataset,
            batch_size=self._local_batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        self._data_iter = _ManagedDataIterator(self, dataloader)

    @override
    def teardown(self, session: TrainingSession):
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
