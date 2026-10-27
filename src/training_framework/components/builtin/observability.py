from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import torch

from torch.utils.tensorboard import SummaryWriter

from training_framework.components import (
    ExtendableComponent,
    LifecycleHook,
    Resource,
    hook,
    rank_zero_only,
    requires_resource,
    resource,
)
from training_framework.util import format_execution_time

if TYPE_CHECKING:
    from training_framework.session import Session, TrainingSession


def reported(session: Session, attribute: str) -> Any:
    """The first non-None `attribute` any hook or resource exposes.

    Duck-typed so the value is found whatever the component is called or
    bound as -- `current_lrs`, `grad_norm`, `last_loss` and
    `completed_steps` come from the optimizer resource, and a hook may
    expose them too.
    """
    for listing in ("get_all_hooks", "get_all_resources"):
        get_all = getattr(session, listing, None)
        if get_all is None:
            continue
        for component in get_all():
            value = getattr(component, attribute, None)
            if value is not None:
                return value
    return None


@rank_zero_only
@hook("logger")
class Logger(LifecycleHook, ExtendableComponent):

    def __init__(self, config: dict):
        self._config = config
        self.call_every = config["log_every"]
        # The configured path (None: stdout), and the stream written to,
        # which is stdout whenever no file is open.
        self._log_path = config.get("log_file")
        self._log_file = sys.stdout

    def pre_session(self, session: Session) -> Any:
        if self._log_path is None:
            return
        # A session resumed from a checkpoint continues the log of the run
        # it came from; a fresh one starts it clean.
        mode = "a" if session.iteration > 0 else "w"
        try:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            self._log_file = open(self._log_path, mode)
        except OSError as error:
            raise OSError(
                f"Logger cannot open log_file {self._log_path!r} for "
                f"writing: {error}"
            ) from error

    def post_session(self, session) -> None:
        log_file, self._log_file = self._log_file, sys.stdout
        if log_file is not sys.stdout:
            log_file.close()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self._log_file)

    def pre_iteration_callback(self, session: Session) -> None:
        line = (
            f"Iteration {session.iteration}/"
            f"{session.session_config.max_iterations}"
        )
        lrs = self._reported(session, "current_lrs")
        if lrs is not None:
            line += " | lr: " + ", ".join(f"{lr:.3e}" for lr in lrs)
        grad_norm = self._reported(session, "grad_norm")
        if grad_norm is not None:
            line += f" | grad_norm: {grad_norm:.3e}"
        loss = self._reported(session, "last_loss")
        if loss is not None:
            line += f" | loss: {loss:.4e}"
        self.print(line)

    @staticmethod
    def _reported(session: Session, attribute: str) -> Any:
        return reported(session, attribute)

    def post_iteration_callback(self, session: Session) -> None:
        pass

    def __getstate__(self) -> Any:
        return {"config": self._config}

    def __setstate__(self, state: Any) -> None:
        self.__init__(state["config"])

    def apply_extension_config(
            self,
            config: Mapping,
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        unsupported = changed_paths - {("log_every",)}
        if unsupported:
            names = ", ".join(".".join(path) for path in sorted(unsupported))
            raise ValueError(
                "Logger session extension does not allow changes to: " + names
            )
        self._config = deepcopy(dict(config))
        self.call_every = self._config["log_every"]


@rank_zero_only
@resource("tensorboard")
class Tensorboard(Resource):
    # Keys that only configure the server; with `serve: false` there is no
    # server for them to configure.
    _SERVER_KEYS = ("host", "port", "logdir")

    def __init__(self, config: dict):
        serve = config.get("serve", True)
        if not isinstance(serve, bool):
            raise ValueError(f"serve must be a boolean; got {serve!r}")
        if not serve:
            given = [key for key in self._SERVER_KEYS if key in config]
            if given:
                raise ValueError(
                    "tensorboard starts no server with serve: false, so "
                    f"{', '.join(given)} would have no effect; remove "
                    f"{'it' if len(given) == 1 else 'them'}"
                )
        self._config = config
        self._tb_process: subprocess.Popen | None = None
        self._tb_summary_writer = None

    @property
    def summary_writer(self):
        return self._tb_summary_writer

    def setup(self, session: Session):
        serve = self._config.get("serve", True)
        if serve:
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
        # Named per instance, so two writers do not merge their scalars into
        # one event directory. The port cannot be shared and is not defaulted:
        # a second serving instance must be given one of its own.
        writer_dir = f"{type(self).__name__}_tensorboard"
        if self.instance_suffix is not None:
            writer_dir = f"{writer_dir}_{self.instance_suffix}"
        self._tb_summary_writer = SummaryWriter(
            log_dir=os.path.join(
                session.session_config.session_dir,
                writer_dir,
            )
        )
        if serve:
            time.sleep(3)
            if self._tb_process.poll() is not None:
                print("Failed to start tensorboard process...")
                raise RuntimeError("Failed to start tensorboard process...")

    def teardown(self, session):
        print("releasing resources...")
        self._release_partial_setup()
        print("resources released...")

    def rollback_setup(self, session: Session) -> None:
        self._release_partial_setup()

    def _release_partial_setup(self) -> None:
        try:
            if self._tb_summary_writer is not None:
                self._tb_summary_writer.close()
        finally:
            self._tb_summary_writer = None

            process = self._tb_process
            self._tb_process = None
            if process is not None and process.poll() is None:
                process.terminate()


# -- tensorboard_summaries ------------------------------------------------------


@dataclass
class TensorboardSummariesConfig:
    """What is written, by kind of summary.

    `scalars` maps a TensorBoard tag to the iteration_context key whose
    value is written under it. `optimizer` adds the optimizer's learning
    rates (one tag per group), gradient norm, loss and completed steps.
    """

    call_every: int = 1
    scalars: Any = field(default_factory=dict)
    optimizer: bool = False

    def __post_init__(self):
        if (
                isinstance(self.call_every, bool)
                or not isinstance(self.call_every, int)
                or self.call_every <= 0
        ):
            raise ValueError(
                f"call_every must be a positive integer; got {self.call_every!r}"
            )
        if not isinstance(self.scalars, Mapping):
            raise ValueError(
                "scalars must be a mapping of TensorBoard tag to "
                f"iteration_context key; got {self.scalars!r}"
            )
        for tag, key in self.scalars.items():
            for name, value in (("tag", tag), ("key", key)):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"scalars maps non-empty tags to non-empty keys; got "
                        f"{name} {value!r}"
                    )
        self.scalars = dict(self.scalars)
        if not isinstance(self.optimizer, bool):
            raise ValueError(
                f"optimizer must be true or false; got {self.optimizer!r}"
            )
        if not self.scalars and not self.optimizer:
            raise ValueError(
                "tensorboard_summaries writes nothing: give scalars, or set "
                "optimizer: true"
            )


def _scalars(tag: str, value: Any) -> list[tuple[str, float]]:
    """`value` as (tag, number) pairs: a number or one-element tensor under
    `tag`, a mapping as one tag per entry (`tag/name`)."""
    if isinstance(value, Mapping):
        pairs = []
        for name, item in value.items():
            pairs.extend(_scalars(f"{tag}/{name}", item))
        return pairs
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                f"tensorboard_summaries scalar {tag!r} is a tensor of shape "
                f"{tuple(value.shape)}; a scalar needs one element"
            )
        return [(tag, float(value.detach()))]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [(tag, float(value))]
    raise TypeError(
        f"tensorboard_summaries scalar {tag!r} is a {type(value).__name__}, "
        "not a number, a one-element tensor or a mapping of them"
    )


@rank_zero_only
@requires_resource("tensorboard")
@hook("tensorboard_summaries")
class TensorboardSummaries(LifecycleHook):
    """Write summaries of the run to the `tensorboard` resource's writer.

    Every `call_every` iterations, after the iteration's steps, each value
    `scalars` names is written at `session.iteration`. The values are read
    from the iteration context like any step's inputs, so a key no step
    writes is an error when the session is built. With `optimizer: true`,
    the optimizer's learning rates, gradient norm, loss and completed steps
    are written too (`optimizer/...`). Written on rank 0 only.
    """

    config_schema = TensorboardSummariesConfig

    def __init__(self, config: Mapping | None = None) -> None:
        super().__init__(config)
        self.call_every = self._cfg.call_every

    @override
    def context_reads(self) -> dict[str, str]:
        # Keys need not be identifiers, so values arrive by key.
        return {key: key for key in dict.fromkeys(self._cfg.scalars.values())}

    def pre_session(self, session: Session) -> None:
        pass

    def post_session(self, session: Session) -> None:
        writer = self.get_dependency("tensorboard").summary_writer
        if writer is not None:
            writer.flush()

    def pre_iteration_callback(self, session: Session) -> None:
        return None

    def post_iteration_callback(self, session: Session, /, *args: Any, **reads: Any) -> None:
        writer = self.get_dependency("tensorboard").summary_writer
        if writer is None:
            raise RuntimeError(
                "tensorboard_summaries has no writer: the tensorboard resource "
                "is not set up"
            )
        step = session.iteration
        for tag, key in self._cfg.scalars.items():
            for name, value in _scalars(tag, reads[key]):
                writer.add_scalar(name, value, step)
        if self._cfg.optimizer:
            for name, value in self._optimizer_scalars(session):
                writer.add_scalar(name, value, step)

    @staticmethod
    def _optimizer_scalars(session: Session) -> list[tuple[str, float]]:
        values = []
        lrs = reported(session, "current_lrs")
        for index, lr in enumerate(lrs or ()):
            values.append((f"optimizer/lr/group_{index}", float(lr)))
        for attribute in ("grad_norm", "last_loss", "completed_steps"):
            value = reported(session, attribute)
            if value is not None:
                values.append((f"optimizer/{attribute}", float(value)))
        return values


@rank_zero_only
@hook("timer", session_type="training")
class Timer(LifecycleHook):

    def __init__(self, config) -> None:
        self.call_every = config["call_every"]
        self._session_start = None
        self._iter_start = None

    def pre_session(self, session: TrainingSession):
        self._session_start = time.time_ns()

    def post_session(self, session: Session):
        pass

    def pre_iteration_callback(self, session: Session) -> None:
        self._iter_start = time.time_ns()

    def post_iteration_callback(self, session: Session) -> None:
        iter_end = time.time_ns()
        iter_duration = iter_end - self._iter_start
        elapsed_time = iter_end - self._session_start
        print(
            f"Time taken for the iteration {session.iteration}: "
            f"{format_execution_time(iter_duration)}"
        )
        print(f"Elapsed time: {format_execution_time(elapsed_time)}")
        print()
