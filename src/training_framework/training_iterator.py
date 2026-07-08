import os
import random
import subprocess
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpointer import save_checkpoint
import yaml


class TrainingIterator(ABC):
    def __init__(self, config: dict):
        self._config = config
        self._iteration = 0
        self._rng_seed = config['rng_seed']
        self._checkpoint_every = config['checkpoint_every']
        self._log_every = config['log_every']
        self._session_dir = os.path.join(
            config['sessions_dir'],
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._checkpoints_dir = os.path.join(self._session_dir, 'checkpoints')

        torch.manual_seed(self._rng_seed)
        random.seed(self._rng_seed)
        torch.cuda.manual_seed(self._rng_seed)
        np.random.seed(self._rng_seed)

        self._device = self._check_and_get_device()
        self._ready = False

        # dump current configuration
        self._dump_config()

    def _dump_config(self):
        os.makedirs(self._session_dir, exist_ok=True)
        config_dump_file = os.path.join(self._session_dir, "session_config.yaml")
        with open(config_dump_file, "w") as config_dump_file:
            yaml.dump(self._config, config_dump_file)

    @abstractmethod
    def _run_iteration(self):
        pass

    def create_checkpoint(self):
        state = self._get_state()

        checkpoint = {
            'config': self._config,
            'iteration': self._iteration,
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'np_rng_state': np.random.get_state(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'session_dir': self._session_dir,
            'state': state
        }

        return checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint):
        obj = cls(checkpoint['config'])
        obj._iteration = checkpoint['iteration']
        obj._session_dir = checkpoint['session_dir']

        torch.set_rng_state(checkpoint['torch_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])

        obj._set_state(checkpoint['state'])

        return obj


    @abstractmethod
    def _get_state(self):
        raise NotImplementedError

    @abstractmethod
    def _set_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def _pre_iteration(self):
        pass

    @abstractmethod
    def _post_iteration(self, iter_result):
        pass

    def _check_and_get_device(self):
        if self._config['device'].startswith('cuda'):
            if torch.cuda.is_available():
                return torch.device(self._config['device'])
            else:
                raise ValueError(f"device '{self._config['device']}' not available")
        elif self._config['device'] == 'cpu':
            return torch.device('cpu')
        else:
            raise ValueError(f"Unknown device '{self._config['device']}'")


    def __iter__(self):
        if not self._ready:
            raise RuntimeError('Use within "with"!')
        return self

    def __next__(self):
        if not self._ready:
            raise RuntimeError('Use within "with"!')

        if self._iteration == self._config['max_iter']:
            raise StopIteration
        self._iteration += 1

        if self._iteration == 1 or self._iteration == self._config['max_iter'] or self._iteration % self._log_every == 0:
            self._pre_iteration()

        iter_result = self._run_iteration()

        # create checkpoints
        if self._iteration == 1 or self._iteration == self._config['max_iter'] or self._iteration % self._checkpoint_every == 0:
            checkpoint = self.create_checkpoint()
            save_checkpoint(checkpoint, save_dir=self._checkpoints_dir, prefix=type(self).__name__)
            print(f'checkpoint created at iteration {self._iteration}')

        # post iteration hook call
        if self._iteration == 1 or self._iteration == self._config['max_iter'] or self._iteration % self._log_every == 0:
            self._post_iteration(iter_result=iter_result)

        return iter_result

    @abstractmethod
    def _acquire_resources(self):
        pass

    @abstractmethod
    def _release_resources(self):
        pass

    def __enter__(self):
        self._acquire_resources()
        self._ready = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Creating checkpoint...")
        checkpoint = self.create_checkpoint()
        if exc_type is KeyboardInterrupt:
            prefix = "KeyboardInterrupt"
        elif exc_type is not None:
            prefix = "Exception"
            checkpoint["exception"] = {
                'type': exc_type,
                'msg': str(exc_val),
                'traceback': traceback.print_tb(exc_tb)
            }
        else:
            prefix = "Final"
        save_checkpoint(checkpoint, save_dir=self._checkpoints_dir, prefix=f"{prefix}_{type(self).__name__}")
        print("Checkpoint created.")

        self._release_resources()
        self._ready = False


class TensorboardTI(TrainingIterator, ABC):
    def __init__(self, config):
        super().__init__(config)
        self._tb_process = None
        self._tb_summary_writer = None


    def _acquire_resources(self):
        tensorboard_args = [
            "tensorboard",
            "--logdir", self._config['sessions_dir'],
            "--host", "0.0.0.0",
            "--port", str(self._config["tensorboard_port"]),
        ]
        print("Tensorboard Arguments: ", " ".join(tensorboard_args))
        self._tb_process = subprocess.Popen(tensorboard_args)
        self._tb_summary_writer = SummaryWriter(log_dir=os.path.join(self._session_dir, f"{type(self).__name__}_tensorboard"))
        time.sleep(3)
        if self._tb_process.poll() is not None:
            print("Failed to start tensorboard process...")
            raise RuntimeError("Failed to start tensorboard process...")

    def _release_resources(self):
        print("releasing resources...")
        self._tb_summary_writer.close()
        self._tb_process.terminate()
        print("resources released...")
