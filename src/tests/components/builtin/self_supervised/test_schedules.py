"""Schedules over optimizer steps: values, holding, and config errors."""

from __future__ import annotations

import math

import pytest

from training_framework.components.builtin.self_supervised import (
    Schedule,
    parse_schedule,
)
from training_framework.components.builtin.self_supervised.schedules import (
    check_range,
)


def test_a_number_is_a_constant_schedule():
    schedule = parse_schedule(0.04, "teacher_temp")

    assert schedule == Schedule(0.04, 0.04, "constant")
    assert schedule.value(0, 10) == schedule.value(99, 10) == 0.04


def test_a_linear_schedule_reaches_its_end_after_its_steps_and_holds():
    schedule = parse_schedule(
        {"start": 0.04, "end": 0.07, "kind": "linear", "steps": 4}, "t",
    )

    values = [schedule.value(step, schedule.length(100)) for step in range(7)]

    assert values[:5] == pytest.approx([0.04, 0.0475, 0.055, 0.0625, 0.07])
    assert values[5:] == pytest.approx([0.07, 0.07])


def test_a_cosine_schedule_without_steps_spans_the_run():
    schedule = parse_schedule({"start": 0.996, "end": 1.0, "kind": "cosine"}, "m")
    length = schedule.length(10)

    assert length == 10
    assert schedule.value(0, length) == pytest.approx(0.996)
    assert schedule.value(5, length) == pytest.approx(0.998)
    assert schedule.value(10, length) == pytest.approx(1.0)
    # Past its length -- an extended run -- it holds the end.
    assert schedule.value(25, length) == pytest.approx(1.0)
    expected = 1.0 + (0.996 - 1.0) * 0.5 * (1 + math.cos(math.pi * 0.3))
    assert schedule.value(3, length) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"start": 1, "speed": 2}, r"unknown keys \['speed'\]"),
        ({"end": 1}, "needs a start"),
        ({"start": 1, "end": 2}, "constant but sets an end"),
        ({"start": 1, "kind": "linear"}, "needs an end"),
        ({"start": 1, "end": 2, "kind": "step"}, "kind must be one of"),
        ({"start": 1, "end": 2, "kind": "linear", "steps": 0}, "steps must be a positive"),
        ({"start": 1, "end": 2, "kind": "linear", "steps": True}, "steps must be a positive"),
        ({"start": 1, "steps": 5}, "constant, so steps means nothing"),
        ("fast", "must be a finite number"),
        (float("nan"), "must be a finite number"),
        (True, "must be a finite number"),
    ],
)
def test_invalid_schedules_are_reported_by_name(value, message):
    with pytest.raises(ValueError, match=message):
        parse_schedule(value, "momentum")


def test_a_schedule_outside_its_range_is_refused():
    schedule = parse_schedule({"start": 0.9, "end": 1.1, "kind": "cosine"}, "m")

    with pytest.raises(ValueError, match=r"momentum values must be in \[0.0, 1.0\]"):
        check_range(schedule, "momentum", low=0.0, high=1.0)
    with pytest.raises(ValueError, match=r"must be in \(0.0"):
        check_range(Schedule(0.0, 0.0, "constant"), "t", low=0.0, high=1.0, low_open=True)
