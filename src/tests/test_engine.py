from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

import training_framework.training_engine as sut


@pytest.fixture
def fake_mp(monkeypatch):
    """Replace multiprocessing primitives with deterministic in-memory fakes."""

    class FakeEvent:
        instances = []

        def __init__(self):
            self._is_set = False
            self.set_calls = 0
            type(self).instances.append(self)

        def set(self) -> None:
            self.set_calls += 1
            self._is_set = True

        def is_set(self) -> bool:
            return self._is_set

    class FakeProcess:
        instances = []
        plans = deque()
        log = []

        def __init__(self, *, name, target, args):
            plan = dict(self.plans.popleft()) if self.plans else {}

            self.name = name
            self.target = target
            self.args = args
            self.fail_on_start = plan.get("fail_on_start")
            self.survive_graceful = plan.get("survive_graceful", False)
            self.survive_terminate = plan.get("survive_terminate", False)
            self.survive_kill = plan.get("survive_kill", False)
            self.graceful_exitcode = plan.get("graceful_exitcode", 0)
            self.terminate_exitcode = plan.get("terminate_exitcode", -15)
            self.kill_exitcode = plan.get("kill_exitcode", -9)
            self.unbounded_exitcode = plan.get("unbounded_exitcode", 0)
            self.interrupt_on_unbounded_join = plan.get(
                "interrupt_on_unbounded_join", False
            )

            self.start_calls = 0
            self.join_calls = []
            self.terminate_calls = 0
            self.kill_calls = 0
            self.close_calls = 0
            self.exitcode = None
            self._alive = False
            self._join_interrupted = False

            type(self).instances.append(self)

        @property
        def pid(self):
            return 0

        def start(self) -> None:
            self.start_calls += 1
            type(self).log.append(("start", self.name))
            if self.fail_on_start is not None:
                raise self.fail_on_start
            self._alive = True

        def join(self, timeout=None) -> None:
            self.join_calls.append(timeout)
            type(self).log.append(("join", self.name, timeout))

            if (
                timeout is None
                and self.interrupt_on_unbounded_join
                and not self._join_interrupted
            ):
                self._join_interrupted = True
                raise KeyboardInterrupt

            if not self._alive:
                return

            if timeout is None:
                self.complete(self.unbounded_exitcode)
            elif self.kill_calls:
                if not self.survive_kill:
                    self.complete(self.kill_exitcode)
            elif self.terminate_calls:
                if not self.survive_terminate:
                    self.complete(self.terminate_exitcode)
            elif not self.survive_graceful:
                self.complete(self.graceful_exitcode)

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminate_calls += 1
            type(self).log.append(("terminate", self.name))

        def kill(self) -> None:
            self.kill_calls += 1
            type(self).log.append(("kill", self.name))

        def close(self) -> None:
            if self._alive:
                raise ValueError("cannot close a running process")
            self.close_calls += 1
            type(self).log.append(("close", self.name))

        def complete(self, exitcode=0) -> None:
            self._alive = False
            self.exitcode = exitcode

    class FakeConfigurator:
        def __init__(self):
            self.session_configs = []

        @property
        def process_timeout_on_join(self):
            return 5.0

        @property
        def mode(self):
            return "new"

    monkeypatch.setattr(sut, "Event", FakeEvent)
    monkeypatch.setattr(sut, "Process", FakeProcess)
    monkeypatch.setattr(sut, "Configurator", FakeConfigurator)
    return SimpleNamespace(Event=FakeEvent, Process=FakeProcess, Configurator=FakeConfigurator)


class IteratorSession:
    def __init__(self, values=(), *, on_next=None, error=None):
        self._values = iter(values)
        self._on_next = on_next
        self._error = error
        self.next_calls = 0
        self.entered = False
        self.exit_args = None

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        return False

    def __next__(self):
        self.next_calls += 1
        if self._on_next is not None:
            self._on_next()
        if self._error is not None:
            raise self._error
        return next(self._values)


def test_session_process_worker_runs_until_stop_iteration(monkeypatch, capsys):
    stop_event = SimpleNamespace(is_set=lambda: False)
    session = IteratorSession(["step-1", "step-2"])
    signal_calls = []
    factory_calls = []

    monkeypatch.setattr(
        sut.signal,
        "signal",
        lambda sig, handler: signal_calls.append((sig, handler)),
    )

    def factory(config, *, rank):
        factory_calls.append((config, rank))
        return session

    monkeypatch.setattr(sut, "create_session_from_config", factory)

    config = {"session_config": {"model": "tiny"}}
    sut.session_process_worker(config, session_id=8, rank=3, stop_event=stop_event)

    assert signal_calls == [(sut.signal.SIGINT, sut.signal.SIG_IGN)]
    assert factory_calls == [(config["session_config"], 3)]
    assert session.entered is True
    assert session.next_calls == 3  # Two values, then StopIteration.
    assert session.exit_args == (None, None, None)
    assert capsys.readouterr().out == "Session 8[3] exiting.\n"


def test_session_process_worker_observes_stop_event_between_steps(
    monkeypatch, capsys
):
    class StopEvent:
        def __init__(self):
            self.value = False

        def is_set(self):
            return self.value

        def set(self):
            self.value = True

    stop_event = StopEvent()
    session = IteratorSession(["unused-second-step"], on_next=stop_event.set)

    monkeypatch.setattr(sut.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        sut,
        "create_session_from_config",
        lambda config, *, rank: session,
    )

    sut.session_process_worker({"session_config": {}}, session_id=1, rank=0, stop_event=stop_event)

    assert session.next_calls == 1
    assert capsys.readouterr().out == "Session 1[0] exiting.\n"


def test_session_process_worker_propagates_session_errors(monkeypatch, capsys):
    error = ValueError("bad training step")
    session = IteratorSession(error=error)

    monkeypatch.setattr(sut.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        sut,
        "create_session_from_config",
        lambda config, *, rank: session,
    )

    with pytest.raises(ValueError, match="bad training step"):
        sut.session_process_worker(
            {"session_config": {}},
            session_id=2,
            rank=0,
            stop_event=SimpleNamespace(is_set=lambda: False),
        )

    assert session.exit_args[0] is ValueError
    assert capsys.readouterr().out == ""


def test_wrapper_builds_named_process_with_copied_config(fake_mp):
    original = {"epochs": 3}

    wrapper = sut.SessionProcessWrapper(original, session_id=4, rank=2)
    process = wrapper.process

    assert wrapper.session_id == 4
    assert wrapper.rank == 2
    assert wrapper.started is False
    assert process.name == "training-session-4-rank-2"
    assert process.target is sut.session_process_worker
    assert process.args[0] == original
    assert process.args[0] is not original
    assert process.args[1:3] == (4, 2)
    assert process.args[3] is fake_mp.Event.instances[0]

    original["epochs"] = 99
    assert process.args[0]["epochs"] == 3


def test_wrapper_starts_once_and_can_request_stop(fake_mp):
    wrapper = sut.SessionProcessWrapper({}, session_id=5, rank=1)

    wrapper.start()
    wrapper.request_stop()

    assert wrapper.started is True
    assert wrapper.process.start_calls == 1
    assert wrapper.process.args[3].is_set() is True

    with pytest.raises(RuntimeError, match=r"Session 5\[1\].*already been started"):
        wrapper.start()

    assert wrapper.process.start_calls == 1


def test_wrapper_is_not_marked_started_when_process_start_fails(fake_mp):
    fake_mp.Process.plans.append({"fail_on_start": OSError("spawn failed")})
    wrapper = sut.SessionProcessWrapper({}, session_id=0, rank=0)

    with pytest.raises(OSError, match="spawn failed"):
        wrapper.start()

    assert wrapper.started is False


@pytest.mark.parametrize("bad_config", [None, [], 7, "not-a-mapping"])
def test_register_session_rejects_non_mapping(fake_mp, bad_config):
    engine = sut.TrainingEngine(fake_mp.Configurator())

    with pytest.raises(TypeError, match="config must be a mapping"):
        engine.register_session(bad_config)


@pytest.mark.parametrize(
    "bad_config",
    [
        {"ddp": {}},
        {"ddp": None},
        {"ddp": []},
    ],
)
def test_register_session_requires_ddp_n_proc(fake_mp, bad_config):
    engine = sut.TrainingEngine(fake_mp.Configurator())

    with pytest.raises(ValueError, match=r"ddp\.world_size"):
        engine.register_session(bad_config)


@pytest.mark.parametrize("world_size", [0, -1, 1.5, "2", True, None])
def test_register_session_rejects_invalid_process_counts(fake_mp, world_size):
    engine = sut.TrainingEngine(fake_mp.Configurator())

    with pytest.raises(ValueError, match="positive integer"):
        engine.register_session({"ddp": {"world_size": world_size}})


def test_register_session_assigns_ids_ranks_and_default_process_count(fake_mp):
    engine = sut.TrainingEngine(fake_mp.Configurator())

    first_id = engine.register_session({"name": "single"})
    second_id = engine.register_session(
        {"name": "distributed", "ddp": {"world_size": 3}}
    )

    assert (first_id, second_id) == (0, 1)
    assert len(engine._session_processes[0]) == 1
    assert [w.rank for w in engine._session_processes[1]] == [0, 1, 2]
    assert [w.session_id for w in engine._session_processes[1]] == [1, 1, 1]
    assert len(fake_mp.Process.instances) == 4

    configs = [w.process.args[0] for w in engine._session_processes[1]]
    assert all(config == {"session_config": {"name": "distributed", "ddp": {"world_size": 3}}} for config in configs)
    assert len({id(config) for config in configs}) == 3


def test_start_all_starts_every_registered_worker_and_normal_exit_closes_them(
    fake_mp,
):
    configurator = fake_mp.Configurator()
    configurator.session_configs.append({"ddp": {"world_size": 2}})
    engine = sut.TrainingEngine(fake_mp.Configurator())

    with engine:
        engine.start_all()
        wrappers = list(engine._iter_wrappers())
        assert all(wrapper.started for wrapper in wrappers)
        assert all(wrapper.process.is_alive() for wrapper in wrappers)

    assert all(wrapper.process.exitcode == 0 for wrapper in wrappers)
    assert all(wrapper.process.close_calls == 1 for wrapper in wrappers)


def test_start_session_cleans_up_workers_started_before_a_start_failure(fake_mp):
    fake_mp.Process.plans.extend(
        [
            {},
            {"fail_on_start": OSError("cannot spawn rank 1")},
            {},
        ]
    )
    engine = sut.TrainingEngine(fake_mp.Configurator())
    session_id = engine.register_session({"ddp": {"world_size": 3}})
    wrappers = engine._session_processes[session_id]

    with engine:
        with pytest.raises(OSError, match="cannot spawn rank 1"):
            engine.start_session(session_id)

        assert wrappers[0].started is True
        assert wrappers[0].process.args[3].is_set() is True
        assert wrappers[0].process.join_calls
        assert wrappers[0].process.join_calls[0] is not None
        assert wrappers[1].started is False
        assert wrappers[2].started is False


def test_request_stop_all_only_signals_started_workers(fake_mp):
    engine = sut.TrainingEngine(fake_mp.Configurator())
    engine.register_session({"ddp": {"world_size": 2}})
    first, second = engine._session_processes[0]
    first.start()

    engine.request_stop_all()

    assert first.process.args[3].is_set() is True
    assert second.process.args[3].is_set() is False

    first.process.complete()


def test_join_or_terminate_uses_one_shared_grace_period(fake_mp, monkeypatch):
    fake_mp.Process.plans.extend(
        [
            {"survive_graceful": True},
            {"survive_graceful": True},
        ]
    )
    engine = sut.TrainingEngine(fake_mp.Configurator())
    engine.register_session({"ddp": {"world_size": 2}})
    first, second = engine._session_processes[0]
    first.start()
    second.start()

    ticks = iter([100.0, 101.0, 104.25])
    monkeypatch.setattr(sut.time, "monotonic", lambda: next(ticks))

    engine._join_or_terminate([first, second], timeout=5.0)

    assert first.process.join_calls == [pytest.approx(4.0), 1.0]
    assert second.process.join_calls == [pytest.approx(0.75), 1.0]
    assert first.process.terminate_calls == 1
    assert second.process.terminate_calls == 1

    shutdown_actions = [
        entry[0]
        for entry in fake_mp.Process.log
        if entry[0] in {"join", "terminate"}
    ]
    assert shutdown_actions == [
        "join",
        "join",
        "terminate",
        "terminate",
        "join",
        "join",
    ]


def test_join_or_terminate_kills_process_that_survives_terminate(
    fake_mp, monkeypatch
):
    fake_mp.Process.plans.append(
        {
            "survive_graceful": True,
            "survive_terminate": True,
            "survive_kill": False,
        }
    )
    engine = sut.TrainingEngine(fake_mp.Configurator())
    engine.register_session({})
    wrapper = engine._session_processes[0][0]
    wrapper.start()

    monkeypatch.setattr(sut.time, "monotonic", lambda: 0.0)
    engine._join_or_terminate([wrapper], timeout=2.0)

    assert wrapper.process.join_calls == [2.0, 1.0, 1.0]
    assert wrapper.process.terminate_calls == 1
    assert wrapper.process.kill_calls == 1
    assert wrapper.process.is_alive() is False
    assert wrapper.process.exitcode == -9


def test_raise_worker_failures_aggregates_session_rank_and_exitcode(fake_mp):
    engine = sut.TrainingEngine(fake_mp.Configurator())
    engine.register_session({"ddp": {"world_size": 2}})
    engine.register_session({})
    wrappers = list(engine._iter_wrappers())

    for wrapper in wrappers:
        wrapper.start()

    wrappers[0].process.complete(0)
    wrappers[1].process.complete(7)
    wrappers[2].process.complete(-9)

    with pytest.raises(RuntimeError) as exc_info:
        engine._raise_worker_failures()

    message = str(exc_info.value)
    assert "session=0, rank=1, exitcode=7" in message
    assert "session=1, rank=0, exitcode=-9" in message
    assert "session=0, rank=0" not in message


def test_raise_worker_failures_ignores_unstarted_running_and_successful_workers(
    fake_mp,
):
    engine = sut.TrainingEngine(fake_mp.Configurator())
    engine.register_session({"ddp": {"world_size": 3}})
    successful, running, unstarted = engine._session_processes[0]

    successful.start()
    successful.process.complete(0)
    running.start()  # exitcode is None while alive.
    unstarted.process.exitcode = 12

    engine._raise_worker_failures()  # Must not raise.
    running.process.complete(0)


def test_close_resources_only_closes_started_dead_processes(fake_mp):
    engine = sut.TrainingEngine(fake_mp.Configurator())
    engine.register_session({"ddp": {"world_size": 3}})
    dead, alive, unstarted = engine._session_processes[0]

    dead.start()
    dead.process.complete(0)
    alive.start()

    engine._close_resources()

    assert dead.process.close_calls == 1
    assert alive.process.close_calls == 0
    assert unstarted.process.close_calls == 0

    alive.process.complete(0)


def test_normal_context_exit_reports_worker_failure_and_still_closes_process(
    fake_mp,
):
    fake_mp.Process.plans.append({"unbounded_exitcode": 23})
    engine = sut.TrainingEngine(fake_mp.Configurator())
    session_id = engine.register_session({})
    process = engine._session_processes[session_id][0].process

    with pytest.raises(RuntimeError, match=r"exitcode=23"):
        with engine:
            engine.start_session(session_id)

    assert process.join_calls == [None]
    assert process.close_calls == 1


def test_context_body_exception_is_preserved_after_children_are_stopped(fake_mp):
    fake_mp.Process.plans.append({"graceful_exitcode": 17})
    engine = sut.TrainingEngine(fake_mp.Configurator())
    session_id = engine.register_session({})
    wrapper = engine._session_processes[session_id][0]

    with pytest.raises(LookupError, match="body failed"):
        with engine:
            engine.start_session(session_id)
            raise LookupError("body failed")

    assert wrapper.process.args[3].is_set() is True
    assert wrapper.process.join_calls[0] is not None
    assert wrapper.process.exitcode == 17
    assert wrapper.process.close_calls == 1


def test_keyboard_interrupt_while_waiting_stops_children_and_is_reraised(fake_mp):
    fake_mp.Process.plans.append({"interrupt_on_unbounded_join": True})
    engine = sut.TrainingEngine(fake_mp.Configurator())
    session_id = engine.register_session({})
    wrapper = engine._session_processes[session_id][0]

    with pytest.raises(KeyboardInterrupt):
        with engine:
            engine.start_session(session_id)

    assert wrapper.process.args[3].is_set() is True
    assert wrapper.process.join_calls[0] is None
    assert wrapper.process.join_calls[1] is not None
    assert wrapper.process.close_calls == 1