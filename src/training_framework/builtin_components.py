from __future__ import annotations

import os
import subprocess
import sys
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, override

import torch
from torch.utils.tensorboard import SummaryWriter

from training_framework.registry import hook, resource
from training_framework.components import Stateful, LifecycleHook, Resource, \
    StatefulResource
from training_framework.util import timestamp_str

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

    def setup(self, session: TrainingSession) -> Any:
        # Address and port where Rank 0 is hosted (must be reachable by all processes)
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = self._master_port

        # Explicitly set the CUDA device for this process (1 process per GPU strategy)
        if self._backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(self._rank)
            session.set_device(torch.device("cuda", self._rank))

        # Initialize the default process group
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=self._rank,
            world_size=self._world_size
        )

    def teardown(self, session):
        torch.distributed.destroy_process_group()
