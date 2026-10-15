"""`profiler`: `torch.profiler` traces of chosen iterations, per rank."""

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import distributed
from torch.profiler import ProfilerAction, ProfilerActivity, profile

from training_framework.components import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    ExtendableComponent,
    LifecycleHook,
    hook,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from training_framework.session import Session

ACTIVITIES = {"cpu": ProfilerActivity.CPU, "cuda": ProfilerActivity.CUDA}

CPU_SORT_KEYS = (
    "cpu_time_total",
    "self_cpu_time_total",
    "cpu_memory_usage",
    "self_cpu_memory_usage",
    "count",
)
DEVICE_SORT_KEYS = (
    "device_time_total",
    "self_device_time_total",
    "device_memory_usage",
    "self_device_memory_usage",
)

# What an extended session may change: when and how much is recorded, not
# what is recorded or where it goes.
EXTENDABLE_KEYS = frozenset(
    {"start_iteration", "warmup", "active", "wait", "repeat"}
)


def _int_at_least(value: Any, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer of at least {minimum}; got {value!r}"
        )
    return value


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false; got {value!r}")
    return value


@dataclass
class ProfilerConfig:
    activities: Any
    start_iteration: Any = 1
    warmup: Any = 1
    active: Any = 3
    wait: Any = 0
    repeat: Any = 1
    ranks: Any = (0,)
    record_shapes: Any = False
    profile_memory: Any = False
    with_stack: Any = False
    summary_sort_by: Any = None
    summary_rows: Any = 25

    def __post_init__(self):
        activities = self.activities
        if (
                not isinstance(activities, (list, tuple)) or not activities
                or any(activity not in ACTIVITIES for activity in activities)
        ):
            raise ValueError(
                f"activities must be a non-empty list of {list(ACTIVITIES)}; "
                f"got {activities!r}"
            )
        self.activities = tuple(dict.fromkeys(activities))
        self.start_iteration = _int_at_least(
            self.start_iteration, 1, "start_iteration",
        )
        # torch documents a warm-up step as needed for accurate CUDA timings.
        self.warmup = _int_at_least(self.warmup, 1, "warmup")
        self.active = _int_at_least(self.active, 1, "active")
        self.wait = _int_at_least(self.wait, 0, "wait")
        self.repeat = _int_at_least(self.repeat, 1, "repeat")
        ranks = self.ranks
        if (
                not isinstance(ranks, (list, tuple)) or not ranks
                or any(
                    isinstance(rank, bool) or not isinstance(rank, int)
                    or rank < 0
                    for rank in ranks
                )
        ):
            raise ValueError(
                "ranks must be a non-empty list of non-negative integers; "
                f"got {ranks!r}"
            )
        self.ranks = tuple(sorted(set(ranks)))
        self.record_shapes = _flag(self.record_shapes, "record_shapes")
        self.profile_memory = _flag(self.profile_memory, "profile_memory")
        self.with_stack = _flag(self.with_stack, "with_stack")
        records_device = "cuda" in self.activities
        if self.summary_sort_by is None:
            self.summary_sort_by = (
                "self_device_time_total" if records_device
                else "self_cpu_time_total"
            )
        allowed = CPU_SORT_KEYS + (DEVICE_SORT_KEYS if records_device else ())
        if self.summary_sort_by not in allowed:
            raise ValueError(
                f"summary_sort_by must be one of {list(allowed)} with "
                f"activities {list(self.activities)}; "
                f"got {self.summary_sort_by!r}"
            )
        self.summary_rows = _int_at_least(self.summary_rows, 1, "summary_rows")

    def window_starts(self) -> list[int]:
        """The first (warm-up) iteration of each profiling window."""
        period = self.warmup + self.active + self.wait
        return [
            self.start_iteration + index * period
            for index in range(self.repeat)
        ]


class _RunState:
    """What one run of a `Profiler` holds: the live `torch.profiler.profile`,
    the windows still to come, where traces go, what was written.

    It belongs to one hook and never travels: a pickle or deep copy of the
    hook gets an empty one (`__reduce__`), and a shallow copy, which shares
    the hook's `__dict__`, is given its own on first use (`owner`).
    """

    def __init__(self, owner: "Profiler | None" = None) -> None:
        self.owner = weakref.ref(owner) if owner is not None else None
        self.profiler: profile | None = None
        self.session: "Session | None" = None
        self.rank = 0
        self.first_iteration = 1
        self.windows: list[int] = []
        self.last_iteration = 0
        self.directory: str | None = None
        self.written: list[str] = []

    def __reduce__(self):
        return (_RunState, ())


_RUN_STATE_ATTR = "_profiler_run"


@hook("profiler", session_type=TRAINING_SESSION_TYPE)
@hook("profiler", session_type=ANALYSIS_SESSION_TYPE)
class Profiler(LifecycleHook, ExtendableComponent):
    """Record `torch.profiler` traces of chosen iterations.

    Each window warms up for `warmup` iterations starting at
    `start_iteration` (counted in session iterations, so a resumed run
    profiles the same ones), then records `active` iterations; `repeat`
    windows are taken `wait` iterations apart. A window a resumed run has
    already started or passed is skipped.

    Built on every rank; only the ranks in `ranks` record. Each recorded
    window is written to `<session_dir>/profiler[_<suffix>]/` as
    `rank<r>_iter<a>-<b>.pt.trace.json.gz` (a Chrome trace: open it in
    Perfetto) and `rank<r>_iter<a>-<b>.summary.txt` (operator totals sorted
    by `summary_sort_by`). A window cut short by the end of the session, or
    by a failure, is written with the iterations it did record.
    """

    config_schema = ProfilerConfig
    call_every = 1

    @property
    def _run(self) -> _RunState:
        run = self.__dict__.get(_RUN_STATE_ATTR)
        if run is None or run.owner is None or run.owner() is not self:
            run = self.__dict__[_RUN_STATE_ATTR] = _RunState(self)
        return run

    @property
    def written(self) -> list[str]:
        """Files written so far in this run."""
        return list(self._run.written)

    @property
    def recording(self) -> bool:
        """Whether this rank has a profiler running in this session."""
        return self._run.profiler is not None

    def pre_session(self, session: "Session") -> None:
        cfg = self._cfg
        run = self._run
        rank, world_size = _rank_and_world_size()
        out_of_range = [r for r in cfg.ranks if r >= world_size]
        if out_of_range:
            raise ValueError(
                f"{self.name}.ranks lists {out_of_range}, but the session "
                f"has {world_size} rank(s)"
            )
        if "cuda" in cfg.activities and session.device.type != "cuda":
            raise ValueError(
                f"{self.name}.activities includes 'cuda', but the session "
                f"runs on {session.device}"
            )
        if rank not in cfg.ranks:
            return

        run.rank = rank
        run.first_iteration = session.iteration + 1
        run.last_iteration = session.iteration
        run.windows = [
            start for start in cfg.window_starts()
            if start >= run.first_iteration
        ]
        if not run.windows:
            print(
                f"{self.name}: every profiling window starts before iteration "
                f"{run.first_iteration}, where this session resumes; nothing "
                "will be recorded"
            )
            return

        suffix = self.instance_suffix
        run.directory = os.path.join(
            session.session_config.session_dir,
            "profiler" + (f"_{suffix}" if suffix else ""),
        )
        os.makedirs(run.directory, exist_ok=True)
        run.session = session
        run.profiler = profile(
            activities=[ACTIVITIES[name] for name in cfg.activities],
            schedule=self._action_for_step,
            on_trace_ready=self._write_window,
            record_shapes=cfg.record_shapes,
            profile_memory=cfg.profile_memory,
            with_stack=cfg.with_stack,
        )
        run.profiler.start()

    def _action_for_step(self, step: int) -> ProfilerAction:
        """torch counts steps from 0 at `start()`; step `n` is the session's
        iteration `first_iteration + n`."""
        run = self._run
        cfg = self._cfg
        iteration = run.first_iteration + step
        for start in run.windows:
            record_start = start + cfg.warmup
            record_end = record_start + cfg.active - 1
            if start <= iteration < record_start:
                return ProfilerAction.WARMUP
            if record_start <= iteration < record_end:
                return ProfilerAction.RECORD
            if iteration == record_end:
                return ProfilerAction.RECORD_AND_SAVE
        return ProfilerAction.NONE

    def _write_window(self, prof: profile) -> None:
        run = self._run
        cfg = self._cfg
        last = run.last_iteration
        start = max(s for s in run.windows if s <= last)
        record_start = start + cfg.warmup
        if last < record_start:
            # The session ended after the warm-up, before any iteration of
            # the window was recorded: there is nothing to write.
            return
        stem = os.path.join(
            run.directory,
            f"rank{run.rank}_iter{record_start}-{last}",
        )
        if run.session is not None:
            run.session.send_heartbeat(f"Writing {self.name} trace")
        trace_path = f"{stem}.pt.trace.json.gz"
        prof.export_chrome_trace(trace_path)
        summary_path = f"{stem}.summary.txt"
        with open(summary_path, "w") as summary:
            summary.write(
                prof.key_averages().table(
                    sort_by=cfg.summary_sort_by,
                    row_limit=cfg.summary_rows,
                )
            )
            summary.write("\n")
        run.written.extend([trace_path, summary_path])

    def pre_iteration_callback(self, session: "Session") -> None:
        pass

    def post_iteration_callback(self, session: "Session") -> None:
        run = self._run
        if run.profiler is None:
            return
        run.last_iteration = session.iteration
        with warnings.catch_warnings():
            # Each window is written on its own, so torch dropping one
            # window's events before the next is what is wanted.
            warnings.filterwarnings(
                "ignore", message="Warning: Profiler clears events",
            )
            run.profiler.step()

    def post_session(self, session: "Session") -> None:
        # Stopping writes a window the session ended (or failed) inside of.
        self._close(export=True)

    def rollback_pre_session(self, session: "Session") -> None:
        self._close(export=False)

    def _close(self, *, export: bool) -> None:
        run = self._run
        prof, run.profiler = run.profiler, None
        try:
            if prof is not None:
                if not export:
                    prof.on_trace_ready = None
                prof.stop()
        finally:
            run.session = None

    def apply_extension_config(
            self,
            config: "Mapping",
            changed_paths: frozenset[tuple[str, ...]],
    ) -> None:
        unsupported = {
            ".".join(path) for path in changed_paths
            if path[0] not in EXTENDABLE_KEYS
        }
        if unsupported:
            raise ValueError(
                f"{self.name} session extension may change only "
                f"{sorted(EXTENDABLE_KEYS)}; not "
                + ", ".join(sorted(unsupported))
            )
        self._parse_config_schema(config)


def _rank_and_world_size() -> tuple[int, int]:
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_rank(), distributed.get_world_size()
    return 0, 1
