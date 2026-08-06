"""Unit tests for the parallel training engine.

The implementation is assumed to be importable as ``engine``.  When it lives
somewhere else, set ``ENGINE_MODULE`` before running pytest, for example::

    ENGINE_MODULE=my_package.parallel_engine pytest -q test_training_engine.py

The tests replace multiprocessing queues/processes with synchronous fakes, so
no child processes are created during the test run.
"""

from __future__ import annotations

from collections import deque
import importlib
import os
from typing import Any, Callable, Deque, Iterable, Optional

import pytest
from training_framework import training_engine


class ScriptedQueue:
    """Small queue fake with optional scripted ``empty()`` responses."""

    def __init__(
        self,
        maxsize: int = 1,
        *,
        items: Iterable[Any] = (),
        empty_results: Iterable[bool] = (),
    ) -> None:
        self.maxsize = maxsize
        self.items: Deque[Any] = deque(items)
        self.empty_results: Deque[bool] = deque(empty_results)
        self.put_calls: list[Any] = []
        self.get_calls = 0

    def empty(self) -> bool:
        if self.empty_results:
            return self.empty_results.popleft()
        return not self.items

    def put(self, value: Any) -> None:
        self.put_calls.append(value)
        self.items.append(value)

    def get(self) -> Any:
        self.get_calls += 1
        if not self.items:
            raise AssertionError("The worker attempted to read an empty test queue")
        return self.items.popleft()


class FakeSession:
    """Iterator/context-manager fake used by the worker tests."""

    def __init__(self, values: Iterable[Any] = ()) -> None:
        self._values = iter(values)
        self.entered = False
        self.exited = False
        self.next_calls = 0
        self.exit_args: Optional[tuple[Any, Any, Any]] = None

    def __enter__(self) -> "FakeSession":
        self.entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exited = True
        self.exit_args = (exc_type, exc_val, exc_tb)

    def __iter__(self) -> "FakeSession":
        return self

    def __next__(self) -> Any:
        self.next_calls += 1
        return next(self._values)


class FakeProcess:
    """Synchronous stand-in for ``multiprocessing.Process``."""

    def __init__(self, *, target: Callable[..., Any], args: tuple[Any, ...]) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.join_timeouts: list[float] = []
        self.alive = False
        self.terminated = False

    def start(self) -> None:
        self.started = True
        self.alive = True

    def join(self, timeout: float) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


def install_multiprocessing_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[ScriptedQueue], list[FakeProcess]]:
    """Patch the module and return every queue/process it creates."""

    queues: list[ScriptedQueue] = []
    processes: list[FakeProcess] = []

    def queue_factory(maxsize: int = 1) -> ScriptedQueue:
        queue = ScriptedQueue(maxsize=maxsize)
        queues.append(queue)
        return queue

    def process_factory(
        *, target: Callable[..., Any], args: tuple[Any, ...]
    ) -> FakeProcess:
        process = FakeProcess(target=target, args=args)
        processes.append(process)
        return process

    monkeypatch.setattr(training_engine, "Queue", queue_factory)
    monkeypatch.setattr(training_engine, "Process", process_factory)
    return queues, processes


def test_ddp_proc_worker_advances_session_until_signal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue = ScriptedQueue(
        items=["pause"],
        empty_results=[True, True, False],
    )
    session = FakeSession(values=["step-1", "step-2"])
    create_calls: list[tuple[dict[str, Any], int]] = []

    def create_session(config: dict[str, Any], *, rank: int) -> FakeSession:
        create_calls.append((config, rank))
        return session

    monkeypatch.setattr(training_engine, "create_session_from_config", create_session)

    config = {"name": "training"}
    training_engine.ddp_proc_worker(config, rank=3, queue=queue)

    assert create_calls == [(config, 3)]
    assert session.entered is True
    assert session.exited is True
    assert session.next_calls == 2
    assert queue.get_calls == 1
    assert "Parallel Session 3: signal pause received." in capsys.readouterr().out


def test_ddp_proc_worker_handles_session_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ``empty()`` reports no signal for the first check.  The session then
    # raises StopIteration, after which the worker still consumes the signal.
    queue = ScriptedQueue(items=[99], empty_results=[True])
    session = FakeSession()

    monkeypatch.setattr(
        training_engine,
        "create_session_from_config",
        lambda config, *, rank: session,
    )

    training_engine.ddp_proc_worker({}, rank=1, queue=queue)

    assert session.next_calls == 1
    assert session.exited is True
    assert queue.get_calls == 1
    assert "Parallel Session 1: signal 99 received." in capsys.readouterr().out


def test_proc_worker_without_ddp_runs_until_signal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue = ScriptedQueue(
        items=["stop"],
        empty_results=[True, True, False],
    )
    session = FakeSession(values=[1, 2])
    create_calls: list[dict[str, Any]] = []

    def create_session(config: dict[str, Any]) -> FakeSession:
        create_calls.append(config)
        return session

    monkeypatch.setattr(training_engine, "create_session_from_config", create_session)

    config = {"epochs": 2}
    training_engine.proc_worker(config, session_id=7, queue=queue)

    assert create_calls == [config]
    assert session.entered is True
    assert session.exited is True
    assert session.next_calls == 2
    assert queue.get_calls == 1

    output = capsys.readouterr().out
    assert "Starting session 7" in output
    assert "signal stop received." in output


def test_proc_worker_without_ddp_handles_session_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = ScriptedQueue(items=["finished"], empty_results=[True])
    session = FakeSession()

    monkeypatch.setattr(
        training_engine,
        "create_session_from_config",
        lambda config: session,
    )

    training_engine.proc_worker({}, session_id=0, queue=queue)

    assert session.next_calls == 1
    assert session.exited is True
    assert queue.get_calls == 1


def test_register_session_assigns_incrementing_ids_and_process_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queues, processes = install_multiprocessing_fakes(monkeypatch)
    engine = training_engine.TrainingEngine(config={"engine": "test"})

    first_config = {"name": "first"}
    second_config = {"name": "second"}

    assert engine.register_session(first_config) == 0
    assert engine.register_session(second_config) == 1

    assert len(queues) == 2
    assert len(processes) == 2
    assert all(queue.maxsize == 1 for queue in queues)

    assert processes[0].target is training_engine.proc_worker
    assert processes[0].args == (first_config, 0, queues[0])
    assert processes[1].target is training_engine.proc_worker
    assert processes[1].args == (second_config, 1, queues[1])


def test_start_session_starts_only_the_requested_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, processes = install_multiprocessing_fakes(monkeypatch)
    engine = training_engine.TrainingEngine(config={})
    first_id = engine.register_session({"name": "first"})
    engine.register_session({"name": "second"})

    with engine:
        engine.start_session(first_id)

    assert processes[0].started is True
    assert processes[1].started is False


def test_start_session_rejects_unknown_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_multiprocessing_fakes(monkeypatch)
    engine = training_engine.TrainingEngine(config={})

    with pytest.raises(IndexError):
        with engine:
            engine.start_session(0)


def test_exit_signals_all_sessions_and_terminates_only_live_processes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queues, processes = install_multiprocessing_fakes(monkeypatch)
    engine = training_engine.TrainingEngine(config={})
    engine.register_session({"name": "first"})
    engine.register_session({"name": "second"})

    # Simulate the first process exiting during join and the second remaining
    # alive, which should trigger the forceful-termination path.
    processes[0].alive = False
    processes[1].alive = True

    result = engine.__exit__(None, None, None)

    assert result is None
    assert [queue.put_calls for queue in queues] == [[1], [1]]
    assert processes[0].join_timeouts == [2.0]
    assert processes[1].join_timeouts == [2.0]
    assert processes[0].terminated is False
    assert processes[1].terminated is True
    assert "terminating process forcefully" in capsys.readouterr().out

def test_ddp_branch_starts_one_worker_for_each_nonzero_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Specification test for the likely intended DDP behavior.

    This remains an expected failure against the code in the question.  Remove
    the xfail marker after correcting the DDP branch.
    """

    queues, processes = install_multiprocessing_fakes(monkeypatch)
    session = FakeSession()
    monkeypatch.setattr(
        training_engine,
        "create_session_from_config",
        lambda config: session,
    )

    parent_queue = ScriptedQueue(items=["stop"])
    config = {"ddp": {"n_proc": 4}}
    training_engine.proc_worker(config, session_id=0, queue=parent_queue)

    assert len(queues) == 3
    assert len(processes) == 3
    assert [process.target for process in processes] == [
        training_engine.ddp_proc_worker,
        training_engine.ddp_proc_worker,
        training_engine.ddp_proc_worker,
    ]
    assert [process.args[1] for process in processes] == [1, 2, 3]
    assert all(process.started for process in processes)