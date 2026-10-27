"""Values that change over a run, counted in optimizer steps.

A schedule goes from `start` to `end` over `steps` optimizer steps --
linearly, along a half cosine, or not at all -- and holds `end` after that.
Without `steps` it spans the whole run: the number of optimizer steps in
`session_config.max_iterations`. That length is resolved once, on first use,
and kept in the owner's state, so a run extended past its original length
holds the final value instead of stretching (or restarting) the schedule --
the rule the optimizer's lr schedule follows too.

A config gives a schedule as a number (constant) or as a mapping::

    momentum: {start: 0.996, end: 1.0, kind: cosine}
    teacher_temp: {start: 0.04, end: 0.07, kind: linear, steps: 1000}
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: How a schedule moves from `start` to `end`.
SCHEDULE_KINDS = ("constant", "linear", "cosine")

_KEYS = ("start", "end", "kind", "steps")


def _finite(value: Any, name: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Schedule:
    """A value scheduled over optimizer steps; see the module docstring."""

    start: float
    end: float
    kind: str
    steps: int | None = None

    def length(self, run_steps: int) -> int:
        """The steps this schedule spans: its own `steps`, else the run's."""
        return self.steps if self.steps is not None else run_steps

    def value(self, step: int, length: int) -> float:
        """The value at the 0-based optimizer `step` of a schedule `length`
        steps long: `start` at step 0, `end` from step `length` on."""
        if self.kind == "constant" or length <= 0:
            return self.start if self.kind == "constant" else self.end
        progress = min(max(step, 0) / length, 1.0)
        if self.kind == "linear":
            return self.start + (self.end - self.start) * progress
        return self.end + (self.start - self.end) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def values(self) -> tuple[float, float]:
        return self.start, self.end


def parse_schedule(value: Any, name: str) -> Schedule:
    """Read a schedule from config: a number, or a mapping of `start`,
    `end`, `kind` and `steps`."""
    if isinstance(value, Schedule):
        return value
    if not isinstance(value, Mapping):
        number = _finite(value, name)
        return Schedule(number, number, "constant")
    unknown = sorted(set(map(str, value)) - set(_KEYS))
    if unknown:
        raise ValueError(
            f"{name} got unknown keys {unknown}; a schedule takes {list(_KEYS)}"
        )
    if "start" not in value:
        raise ValueError(f"{name} needs a start value")
    start = _finite(value["start"], f"{name}.start")
    kind = value.get("kind", "constant")
    if kind not in SCHEDULE_KINDS:
        raise ValueError(
            f"{name}.kind must be one of {list(SCHEDULE_KINDS)}; got {kind!r}"
        )
    if kind == "constant":
        if "end" in value and value["end"] != value["start"]:
            raise ValueError(
                f"{name} is constant but sets an end ({value['end']!r}) "
                f"different from its start; give kind: linear or cosine"
            )
        if value.get("steps") is not None:
            raise ValueError(f"{name} is constant, so steps means nothing")
        return Schedule(start, start, kind)
    if "end" not in value:
        raise ValueError(f"{name} is {kind}, so it needs an end value")
    end = _finite(value["end"], f"{name}.end")
    steps = value.get("steps")
    if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
    ):
        raise ValueError(
            f"{name}.steps must be a positive number of optimizer steps, or "
            f"null for the whole run; got {steps!r}"
        )
    return Schedule(start, end, kind, steps)


def check_range(
        schedule: Schedule,
        name: str,
        *,
        low: float,
        high: float,
        low_open: bool = False,
) -> Schedule:
    """Refuse a schedule whose start or end falls outside [low, high] (or
    (low, high] with `low_open`)."""
    for value in schedule.values():
        if value > high or value < low or (low_open and value == low):
            bracket = "(" if low_open else "["
            raise ValueError(
                f"{name} values must be in {bracket}{low}, {high}]; got "
                f"start={schedule.start}, end={schedule.end}"
            )
    return schedule


__all__ = ["SCHEDULE_KINDS", "Schedule", "check_range", "parse_schedule"]
