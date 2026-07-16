import threading
import time

import pytest

from training_framework.training_engine import TrainingEngine
from training_framework.training_session import TrainingSession, Step, Stateful, step, SessionPhase


@step("blocking_step_for_threading")
class BlockingStep(Step, Stateful):
    def __init__(self, started_event=None, release_event=None):
        self.started_event = started_event or threading.Event()
        self.release_event = release_event or threading.Event()
        self.run_calls = 0

    def get_state(self):
        return {"run_calls": self.run_calls}

    def set_state(self, state):
        self.run_calls = state["run_calls"]

    def run(self, session):
        self.run_calls += 1
        session.iteration_context["ran"] = True
        self.started_event.set()
        self.release_event.wait(timeout=2)

def make_session(tmp_path, max_iterations=1):
    return TrainingSession(
        {
            "rng_seed": 123,
            "sessions_dir": str(tmp_path),
            "max_iterations": max_iterations,
            "device": "cpu",
        }
    )

def test_run_all_wait_false_starts_thread_and_does_not_join(tmp_path):
    session = TrainingSession(
        {
            "rng_seed": 123,
            "sessions_dir": str(tmp_path),
            "max_iterations": 1,
            "device": "cpu",
        }
    )

    started = threading.Event()
    release = threading.Event()
    step_obj = BlockingStep(started_event=started, release_event=release)
    session.add_step(step_obj)

    engine = TrainingEngine({})

    engine.register_session(session)

    with engine:
        engine.run_all(wait=False)

        assert started.wait(timeout=1), "session thread never reached the step"
        assert engine._session_threads[0].is_alive()

        release.set()
        engine._session_threads[0].join(timeout=2)

    assert not engine._session_threads[0].is_alive()
    assert step_obj.run_calls == 1
    assert session.iteration == 1
    assert session._phase is SessionPhase.FINISHED


def test_engine_context_exit_stops_session_after_current_iteration(tmp_path):
    session = TrainingSession(
        {
            "rng_seed": 123,
            "sessions_dir": str(tmp_path),
            "max_iterations": 3,
            "device": "cpu",
        }
    )

    started = threading.Event()
    release = threading.Event()
    step_obj = BlockingStep(started_event=started, release_event=release)
    session.add_step(step_obj)

    engine = TrainingEngine({})
    engine.register_session(session)

    with engine:
        engine.run_all(wait=False)
        assert started.wait(timeout=1), "session thread never reached the first iteration"
        # Exit the engine context while the session is still blocked in the step.

    release.set()
    engine._session_threads[0].join(timeout=2)

    assert session.iteration == 1
    assert step_obj.run_calls == 1
    assert session._phase is SessionPhase.PAUSED or session._phase is SessionPhase.FINISHED


def test_register_session_rejects_same_instance_twice(tmp_path):
    session = make_session(tmp_path)
    engine = TrainingEngine({})

    engine.register_session(session)

    with pytest.raises(RuntimeError):
        engine.register_session(session)


def test_engine_run_all_wait_false_returns_before_blocking_step_finishes(tmp_path):
    session = make_session(tmp_path, max_iterations=1)
    started = threading.Event()
    release = threading.Event()
    step_obj = BlockingStep(started_event=started, release_event=release)
    session.add_step(step_obj)

    engine = TrainingEngine({})
    engine.register_session(session)

    with engine:
        engine.run_all(wait=False)

        assert started.wait(timeout=1), "Session thread never reached the blocking step"
        assert engine._session_threads[0].is_alive()

        release.set()
        engine._session_threads[0].join(timeout=2)

    assert step_obj.run_calls == 1
    assert session.iteration == 1
    assert session._phase in (SessionPhase.FINISHED, SessionPhase.PAUSED)


def test_engine_exit_stops_after_current_iteration_not_mid_iteration(tmp_path):
    session = make_session(tmp_path, max_iterations=3)
    started = threading.Event()
    release = threading.Event()
    step_obj = BlockingStep(started_event=started, release_event=release)
    session.add_step(step_obj)

    engine = TrainingEngine({})
    engine.register_session(session)

    with engine:
        engine.run_all(wait=False)
        assert started.wait(timeout=1), "Session thread never started"
        time.sleep(0.1)

    release.set()
    engine._session_threads[0].join(timeout=2)

    assert step_obj.run_calls == 1
    assert session.iteration == 1


def test_session_does_not_start_second_iteration_after_stop_flag_is_set(tmp_path):
    session = make_session(tmp_path, max_iterations=10)
    started = threading.Event()
    release = threading.Event()
    step_obj = BlockingStep(started_event=started, release_event=release)
    session.add_step(step_obj)

    engine = TrainingEngine({})
    engine.register_session(session)

    with engine:
        engine.run_all(wait=False)
        assert started.wait(timeout=1), "Session thread never reached the first iteration"

        # Signal shutdown for this specific session while the first iteration is still running.
        with engine._session_active_flag_locks[0]:
            engine._session_active_flags[0] = False

        release.set()
        engine._session_threads[0].join(timeout=2)

    assert step_obj.run_calls == 1
    assert session.iteration == 1

# TODO: Uncomment this test after implementing session registry
# def test_two_engines_can_not_share_same_session_object_if_ownership_is_tracked(tmp_path):
#     session = make_session(tmp_path)
#     engine1 = TrainingEngine({})
#     engine2 = TrainingEngine({})
#
#     engine1.register_session(session)
#
#     # If you add session ownership tracking, this should raise.
#     # Keep this test if you've added session._engine or session._attach_engine().
#     with pytest.raises(RuntimeError):
#         engine2.register_session(session)