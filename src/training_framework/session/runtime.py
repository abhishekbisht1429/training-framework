import os
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from training_framework.components.edges import context_mapping
from training_framework.session.config import SessionPhase

if TYPE_CHECKING:
    from training_framework.session.base import Session


def clear_iteration_state(session: "Session") -> None:
    session._shared_state.clear()
    session._iteration_generation += 1


def _reads_for(session: "Session", reader) -> dict[str, Any]:
    """The values `reader` declared reading, by the name it takes them as."""
    context = session._shared_state
    values = {}
    for name, key in context_mapping(reader, "reads").items():
        try:
            values[name] = context[key]
        except KeyError:
            raise RuntimeError(
                f"{reader.id} reads iteration_context '{key}' (as '{name}'), "
                "which has not been written this iteration."
            ) from None
    return values


def _describe_result(result: Any) -> str:
    if isinstance(result, tuple):
        return f"a tuple of {len(result)}"
    if isinstance(result, Mapping):
        return f"a mapping of {sorted(map(str, result))}"
    return f"a {type(result).__name__}"


def _store_writes(session: "Session", writer, result: Any, callback: str) -> None:
    """Put what `writer` returned from `callback` under the keys it declared.

    One declared output is the returned value itself, never unpacked, so a
    step can write a tuple or a dict. Several are a tuple in declaration
    order or a mapping by output name, exactly; nothing to write means None.
    No declared output may be None, which is what a forgotten `return`
    produces. Returning is the only way a key is written.
    """
    outputs = context_mapping(writer, "writes")
    shown = f"{writer.id}.{callback}"
    if not outputs:
        if result is not None:
            raise RuntimeError(
                f"{shown} returned {_describe_result(result)}, but declares "
                "no iteration_context writes. Declare them with "
                "@writes(...) (or in context_writes()), or return None."
            )
        return
    names = list(outputs)
    if len(names) == 1:
        values = {names[0]: result}
    elif isinstance(result, tuple) and len(result) == len(names):
        values = dict(zip(names, result))
    elif isinstance(result, Mapping) and set(result) == set(names):
        values = {name: result[name] for name in names}
    else:
        raise RuntimeError(
            f"{shown} declares writing {names}, so it must return a tuple of "
            f"{len(names)} in that order or a mapping with exactly those "
            f"names; it returned {_describe_result(result)}."
        )
    unset = [name for name in names if values[name] is None]
    if unset:
        raise RuntimeError(
            f"{writer.id} declares it writes iteration_context {names}, but "
            f"{callback} returned None for {unset}. Its readers are ordered "
            "and checked on that promise; return a value for each (a "
            "forgotten `return` looks like this)."
        )
    context = session._shared_state
    for name, key in outputs.items():
        context[key] = values[name]


def _nothing_returned(component, result: Any, callback: str) -> None:
    if result is not None:
        raise RuntimeError(
            f"{component.id}.{callback} returned {_describe_result(result)}; "
            "it must return None. An iteration hook writes in its "
            "pre-iteration callback."
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
                _store_writes(
                    session,
                    iteration_hook,
                    iteration_hook.pre_iteration_callback(session),
                    "pre_iteration_callback",
                )

        for step in session._sorted_steps:
            session.send_heartbeat(f"Running {step.id}")
            _store_writes(
                session,
                step,
                step.run(session, **_reads_for(session, step)),
                "run",
            )

        for iteration_hook in reversed(session._iteration_hooks):
            if (
                    session._iteration == 1
                    or session._iteration == session.session_config.max_iterations
                    or session._iteration % iteration_hook.call_every == 0
            ):
                session.send_heartbeat(f"Running {iteration_hook.id}")
                _nothing_returned(
                    iteration_hook,
                    iteration_hook.post_iteration_callback(
                        session, **_reads_for(session, iteration_hook),
                    ),
                    "post_iteration_callback",
                )

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
