from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from torch.utils.tensorboard import SummaryWriter

from training_framework.components import LifecycleHook, Resource
from training_framework.registry import hook, resource, wraps
from training_framework.util import format_execution_time

if TYPE_CHECKING:
    from training_framework.training_session import TrainingSession


@hook("logger")
class Logger(LifecycleHook):

    def __init__(self, config: dict):
        self._config = config
        self.call_every = config["log_every"]
        self._log_file = config.get("log_file", sys.stdout)

    def setup(self, session: TrainingSession) -> Any:
        if self._log_file is not sys.stdout:
            try:
                self._log_file = open(self._config["log_file"], "w")
            except FileNotFoundError:
                print(
                    "Unable to open log file for writing to "
                    f"{self._config['log_file']}"
                )

    def teardown(self, session) -> None:
        if self._log_file is not sys.stdout:
            self._log_file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self._log_file)

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        self.print(
            f"Iteration {session.iteration}/"
            f"{session.session_config.max_iterations}"
        )

    def post_iteration_callback(self, session: TrainingSession) -> None:
        pass

    def __getstate__(self) -> Any:
        return {"config": self._config}

    def __setstate__(self, state: Any) -> None:
        self._config = state["config"]
        self.call_every = self._config["log_every"]


@resource("tensorboard")
class Tensorboard(Resource):

    def __init__(self, config: dict):
        self._config = config
        self._tb_process: subprocess.Popen | None = None
        self._tb_summary_writer = None

    @property
    def summary_writer(self):
        return self._tb_summary_writer

    def setup(self, session: TrainingSession):
        logdir = self._config.get(
            "logdir",
            session.session_config.session_dir,
        )
        tensorboard_args = [
            "tensorboard",
            "--logdir", logdir,
            "--host", self._config["host"],
            "--port", str(self._config["port"]),
        ]
        print("Tensorboard Arguments: ", " ".join(tensorboard_args))
        self._tb_process = subprocess.Popen(tensorboard_args)
        self._tb_summary_writer = SummaryWriter(
            log_dir=os.path.join(
                session.session_config.session_dir,
                f"{type(self).__name__}_tensorboard",
            )
        )
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


@wraps("optimizer")
@hook("timer")
class Timer(LifecycleHook):

    def __init__(self, config) -> None:
        self.call_every = config["call_every"]
        self._session_start = None
        self._iter_start = None

    def setup(self, session: TrainingSession):
        self._session_start = time.time_ns()

    def teardown(self, session: TrainingSession):
        pass

    def pre_iteration_callback(self, session: TrainingSession) -> None:
        self._iter_start = time.time_ns()

    def post_iteration_callback(self, session: TrainingSession) -> None:
        iter_end = time.time_ns()
        iter_duration = iter_end - self._iter_start
        elapsed_time = iter_end - self._session_start
        print(
            f"Time taken for the iteration {session.iteration}: "
            f"{format_execution_time(iter_duration)}"
        )
        print(f"Elapsed time: {format_execution_time(elapsed_time)}")
        print()
