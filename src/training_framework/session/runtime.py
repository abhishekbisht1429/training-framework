import os
import traceback
from typing import TYPE_CHECKING, Any, cast

from training_framework.session.config import SessionPhase

if TYPE_CHECKING:
    from training_framework.session.base import Session


def clear_iteration_state(session: "Session") -> None:
    session._shared_state.clear()


def _check_declared_writes(session: "Session", writer) -> None:
    """Hold a step or iteration hook to the keys it declared writing, right
    after the callback that writes them: the order and the checks made when
    the session was built rely on that promise."""
    missing = [
        key for key in writer.context_writes()
        if key not in session._shared_state
    ]
    if missing:
        raise RuntimeError(
            f"{writer.id} declares it writes iteration_context {missing} but "
            "did not write them; its readers are ordered and checked on that "
            "promise."
        )


def run_iteration(session: "Session") -> int:
    iteration_complete = False
    try:
        session._iteration += 1
        session._phase = SessionPhase.RUNNING

        if session._iteration > session.session_config.max_iterations:
            session._phase = SessionPhase.FINISHED
            raise StopIteration

        for iteration_hook in session._iteration_hooks:
            if (
                    session._iteration == 1
                    or session._iteration == session.session_config.max_iterations
                    or session._iteration % iteration_hook.call_every == 0
            ):
                session.send_heartbeat(f"Running {iteration_hook.id}")
                iteration_hook.pre_iteration_callback(session)
                _check_declared_writes(session, iteration_hook)

        for step in session._sorted_steps:
            session.send_heartbeat(f"Running {step.id}")
            step.run(session)
            _check_declared_writes(session, step)

        for iteration_hook in reversed(session._iteration_hooks):
            if (
                    session._iteration == 1
                    or session._iteration == session.session_config.max_iterations
                    or session._iteration % iteration_hook.call_every == 0
            ):
                session.send_heartbeat(f"Running {iteration_hook.id}")
                iteration_hook.post_iteration_callback(session)

        iteration_complete = True
    finally:
        session._clear_iteration_state()

        if not iteration_complete:
            session._iteration -= 1

    return session._iteration


def setup_resources(session: "Session") -> None:
    for component in session._sorted_resources:
        session.send_heartbeat(f"Running setup {component.id}")
        try:
            component.setup(session)
        except Exception:
            try:
                session.send_heartbeat(
                    f"Running setup rollback {component.id}"
                )
                component.rollback_setup(session)
            except Exception as error:
                print(
                    f"Error rolling back setup for resource "
                    f"'{component.id}': {error}"
                )
            raise
        session._successfully_setup_resource_names.add(component.name)


def setup_session_hooks(session: "Session") -> None:
    for component in session._session_hooks:
        session.send_heartbeat(f"Running pre-session {component.id}")
        try:
            component.pre_session(session)
        except Exception:
            try:
                session.send_heartbeat(
                    f"Running pre-session rollback {component.id}"
                )
                component.rollback_pre_session(session)
            except Exception as error:
                print(
                    f"Error rolling back pre-session for hook "
                    f"'{component.id}': {error}"
                )
            raise
        session._successfully_setup_hook_names.add(component.name)


def teardown_resources(
        session: "Session",
        *,
        after_exception: bool = False,
) -> None:
    stage_suffix = " after exception" if after_exception else ""
    for component in reversed(session._sorted_resources):
        if component.name not in session._successfully_setup_resource_names:
            continue
        try:
            session.send_heartbeat(
                f"Running teardown {component.id}{stage_suffix}"
            )
            component.teardown(session)
        except Exception as error:
            print(f"Error releasing resource '{component.id}': {error}")


def teardown_session_hooks(
        session: "Session",
        *,
        after_exception: bool = False,
) -> None:
    stage_suffix = " after exception" if after_exception else ""
    for component in reversed(session._session_hooks):
        if component.name not in session._successfully_setup_hook_names:
            continue
        try:
            session.send_heartbeat(
                f"Running post-session {component.id}{stage_suffix}"
            )
            component.post_session(session)
        except Exception as error:
            print(f"Error running post-session '{component.name}': {error}")


def report_worker_exception(
        session: "Session",
        exc_type,
        exc_val,
) -> None:
    if session._dist_manager_err_conn is None or exc_type is None:
        return
    if session._worker_exception_reported:
        return
    rank = (
        cast(Any, session._components.get_resource("ddp")).rank
        if session._components.has_resource("ddp")
        else 0
    )
    try:
        session._dist_manager_err_conn.send({
            "type": "error",
            "rank": rank,
            "pid": os.getpid(),
            "exception_type": str(exc_type),
            "message": str(exc_val),
            "traceback": traceback.format_exc(),
        })
    except OSError:
        # The parent already closed the pipe; don't mask the real exception.
        return
    session._worker_exception_reported = True


def send_heartbeat(session: "Session", stage) -> None:
    if session._progress_beacon is None:
        return
    session._progress_beacon.mark(stage, session._iteration)
