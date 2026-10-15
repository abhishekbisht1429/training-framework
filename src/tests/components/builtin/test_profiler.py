"""The `profiler` built-in: which iterations it records and what it writes."""

from __future__ import annotations

import gzip
import json
import os

import pytest

from tests.test_utils import (
    component_named,
    register_test_components,
    session_config,
)
from training_framework.engine import load_session_for_worker
from training_framework.session import TrainingSession


def profiled_config(tmp_path, *, max_iterations, **profiler):
    config = session_config(tmp_path, max_iterations=max_iterations)
    config["session_config"]["show_execution_graph"] = False
    config["profiler"] = {"activities": ["cpu"], **profiler}
    return config


def run_to_end(session):
    with session:
        for _ in session:
            pass
    return session


def written_names(session):
    return sorted(
        os.path.basename(path)
        for path in component_named(session, "profiler").written
    )


@pytest.fixture(autouse=True)
def _test_components():
    register_test_components()


def test_each_window_writes_a_chrome_trace_and_a_summary(tmp_path):
    session = run_to_end(TrainingSession(profiled_config(
        tmp_path, max_iterations=12,
        start_iteration=3, warmup=1, active=2, wait=2, repeat=2,
    )))

    # Window 1 warms up on 3 and records 4-5; window 2 starts 5 later.
    assert written_names(session) == [
        "rank0_iter4-5.pt.trace.json.gz",
        "rank0_iter4-5.summary.txt",
        "rank0_iter9-10.pt.trace.json.gz",
        "rank0_iter9-10.summary.txt",
    ]
    directory = os.path.join(session.session_config.session_dir, "profiler")
    assert sorted(os.listdir(directory)) == written_names(session)
    with gzip.open(os.path.join(directory, "rank0_iter4-5.pt.trace.json.gz")) as trace:
        assert json.load(trace)["traceEvents"]
    with open(os.path.join(directory, "rank0_iter4-5.summary.txt")) as summary:
        assert "Self CPU" in summary.read()


def test_a_window_the_session_ends_inside_is_written_as_far_as_it_got(tmp_path):
    session = run_to_end(TrainingSession(profiled_config(
        tmp_path, max_iterations=4, start_iteration=2, warmup=1, active=5,
    )))

    assert written_names(session) == [
        "rank0_iter3-4.pt.trace.json.gz",
        "rank0_iter3-4.summary.txt",
    ]


def test_a_session_ending_during_warm_up_writes_nothing(tmp_path):
    session = run_to_end(TrainingSession(profiled_config(
        tmp_path, max_iterations=3, start_iteration=3, warmup=2,
    )))

    assert written_names(session) == []


def test_a_resumed_session_profiles_the_same_iterations(tmp_path):
    config = profiled_config(
        tmp_path, max_iterations=4,
        start_iteration=3, warmup=1, active=2, wait=0, repeat=2,
    )
    first = run_to_end(TrainingSession(config))
    assert written_names(first) == [
        "rank0_iter4-4.pt.trace.json.gz",
        "rank0_iter4-4.summary.txt",
    ]

    resumed = run_to_end(load_session_for_worker(
        first.get_state(), rank=0,
        session_update_params={"max_iterations": 10},
    ))

    # The window that started on 3 is not restarted; the next is on 6.
    assert written_names(resumed) == [
        "rank0_iter7-8.pt.trace.json.gz",
        "rank0_iter7-8.summary.txt",
    ]


def test_a_resumed_session_past_every_window_records_nothing(tmp_path, capsys):
    first = run_to_end(TrainingSession(profiled_config(
        tmp_path, max_iterations=5, start_iteration=1, warmup=1, active=2,
    )))

    resumed = run_to_end(load_session_for_worker(
        first.get_state(), rank=0,
        session_update_params={"max_iterations": 8},
    ))

    assert written_names(resumed) == []
    assert "nothing will be recorded" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("profiler", "message"),
    [
        ({"activities": None}, "activities"),
        ({"activities": []}, "activities"),
        ({"activities": ["gpu"]}, "activities"),
        ({"warmup": 0}, "warmup"),
        ({"active": 0}, "active"),
        ({"repeat": 0}, "repeat"),
        ({"wait": -1}, "wait"),
        ({"start_iteration": 0}, "start_iteration"),
        ({"ranks": []}, "ranks"),
        ({"ranks": [-1]}, "ranks"),
        ({"with_stack": "yes"}, "with_stack"),
        ({"summary_sort_by": "self_device_time_total"}, "summary_sort_by"),
        ({"summary_sort_by": "bogus"}, "summary_sort_by"),
    ],
)
def test_an_invalid_configuration_is_refused(tmp_path, profiler, message):
    with pytest.raises((TypeError, ValueError), match=message):
        TrainingSession(profiled_config(tmp_path, max_iterations=1, **profiler))


def test_activities_must_be_given(tmp_path):
    config = profiled_config(tmp_path, max_iterations=1)
    del config["profiler"]["activities"]

    with pytest.raises((TypeError, ValueError), match="activities"):
        TrainingSession(config)


def test_cuda_activity_needs_a_cuda_session(tmp_path):
    session = TrainingSession(profiled_config(
        tmp_path, max_iterations=1, activities=["cpu", "cuda"],
    ))

    with pytest.raises(ValueError, match="'cuda'.*runs on cpu"):
        run_to_end(session)


def test_a_rank_the_session_does_not_have_is_refused(tmp_path):
    session = TrainingSession(profiled_config(
        tmp_path, max_iterations=1, ranks=[0, 1],
    ))

    with pytest.raises(ValueError, match=r"ranks lists \[1\].*1 rank"):
        run_to_end(session)


def test_an_extended_session_may_move_the_windows(tmp_path):
    first = run_to_end(TrainingSession(profiled_config(
        tmp_path, max_iterations=4, start_iteration=1, warmup=1, active=2,
    )))
    resumed = load_session_for_worker(
        first.get_state(), rank=0,
        session_update_params={"max_iterations": 10},
    )

    resumed.apply_extension_overrides(
        ["profiler.start_iteration=7", "profiler.active=3"],
    )
    run_to_end(resumed)

    assert written_names(resumed) == [
        "rank0_iter8-10.pt.trace.json.gz",
        "rank0_iter8-10.summary.txt",
    ]


def test_an_extended_session_may_not_change_what_is_recorded(tmp_path):
    first = run_to_end(TrainingSession(profiled_config(
        tmp_path, max_iterations=1,
    )))
    resumed = load_session_for_worker(first.get_state(), rank=0)

    with pytest.raises(ValueError, match="record_shapes"):
        resumed.apply_extension_overrides(["profiler.record_shapes=true"])
