from __future__ import annotations

import time

import pytest

from training_framework.engine import supervision


class _Process:
    pid = 1234
    sentinel = None

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @staticmethod
    def is_alive():
        return True


class _FakeWrapper:
    """Stands in for SessionProcessWrapper with a scripted beacon."""

    def __init__(self, sentinel, *, progress_until: float):
        self.rank = 0
        self.heartbeat_timeout = 0.2
        self.process = _Process(sentinel)
        self.error_conn = object()
        self.last_iteration = 0
        self.last_stage = "Starting worker"
        self.last_progress_time = time.monotonic()
        self.deadline = self.last_progress_time + self.heartbeat_timeout
        self._progress_until = progress_until
        self.progress_checks = 0

    def check_progress(self, now):
        self.progress_checks += 1
        if now >= self._progress_until:
            return False
        self.last_iteration += 1
        self.last_stage = "Running train"
        self.last_progress_time = now
        self.deadline = now + self.heartbeat_timeout
        return True


def _monitor(monkeypatch, wrapper):
    monkeypatch.setattr(supervision, "wait", lambda waitables, timeout: [])
    shutdowns = []
    monitor_error = None
    try:
        supervision.monitor_processes(
            [wrapper],
            process_ready=lambda waitables, ready: None,
            request_stop_all=lambda: None,
            shutdown=lambda active, timeout: shutdowns.append(active),
            process_timeout_on_join=1.0,
        )
    except TimeoutError as error:
        monitor_error = error
    return monitor_error, shutdowns


def test_progress_extends_deadline_then_timeout_names_stuck_stage(monkeypatch):
    progress_until = time.monotonic() + 0.4
    wrapper = _FakeWrapper(object(), progress_until=progress_until)

    error, shutdowns = _monitor(monkeypatch, wrapper)

    assert error is not None
    # The worker survived past its initial deadline because it kept marking.
    assert time.monotonic() >= progress_until + wrapper.heartbeat_timeout
    assert shutdowns == [[wrapper]]
    message = str(error)
    assert "rank=0" in message
    assert "stage 'Running train'" in message
    assert f"iteration {wrapper.last_iteration}" in message


def test_monitor_polls_progress_without_pipe_traffic(monkeypatch):
    wrapper = _FakeWrapper(object(), progress_until=0.0)

    error, _ = _monitor(monkeypatch, wrapper)

    assert error is not None
    assert "stage 'Starting worker'" in str(error)
    assert wrapper.progress_checks > 1


def test_heartbeat_messages_are_no_longer_accepted_on_error_pipe():
    class Connection:
        closed = False

        @staticmethod
        def recv():
            return {"type": "error", "message": "boom"}

        def close(self):
            self.closed = True

    connection = Connection()
    wrapper = _FakeWrapper(object(), progress_until=0.0)
    waitables = {connection: ("connection", wrapper)}

    failure = supervision.process_ready_waitables(waitables, [connection])

    assert isinstance(failure, RuntimeError)
    assert "boom" in str(failure)
    assert connection.closed
