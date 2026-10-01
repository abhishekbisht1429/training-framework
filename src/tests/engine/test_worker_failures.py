"""What the parent raises when a spawned worker fails.

The error is one readable line; the worker's traceback is printed verbatim
as the error's cause, so its `File "...", line N` lines read like any
traceback.
"""

from __future__ import annotations

import importlib
import signal
import sys
import traceback
from dataclasses import dataclass
from typing import Any

import pytest

from training_framework.engine import (
    RemoteTraceback,
    TrainingEngine,
    WorkerFailedError,
)
from training_framework.engine.failures import failure_from_report


_COMPONENTS_PACKAGE = "tests.engine.worker_failure_components"


@dataclass
class _EngineConfig:
    mode: str
    process_timeout_on_join: float = 5.0
    session_configs: tuple[dict[str, Any], ...] = ()
    checkpoint_path: str | None = None
    new_max_iters: int | None = None
    heartbeat_timeout: float = 10.0
    stop_sync_grace_period: float = 0.01
    stop_sync_poll_interval: float = 0.005
    debug: bool = False


@pytest.fixture(autouse=True)
def _components():
    existing = sys.modules.get(_COMPONENTS_PACKAGE)
    if existing is None:
        importlib.import_module(_COMPONENTS_PACKAGE)
    else:
        importlib.reload(existing)


def run_failing(tmp_path, **components) -> WorkerFailedError:
    config = {
        "session_config": {
            "rng_seed": 3,
            "sessions_dir": str(tmp_path),
            "max_iterations": 2,
            "device": "cpu",
            "components_package": _COMPONENTS_PACKAGE,
            "show_execution_graph": False,
        },
        **components,
    }
    with pytest.raises(WorkerFailedError) as caught:
        with TrainingEngine(_EngineConfig(mode="new", session_configs=(config,))) as engine:
            engine.start_session()
    return caught.value


def test_a_worker_failure_is_one_line_with_the_worker_traceback_as_its_cause(tmp_path):
    error = run_failing(tmp_path, wf_raise={"message": "boom in a step"})

    assert isinstance(error, RuntimeError)
    assert str(error) == (
        f"Worker pid={error.pid} (rank 0) failed: RuntimeError: boom in a step"
    )
    assert (error.rank, error.exception_type, error.worker_message) == (
        0, "RuntimeError", "boom in a step",
    )
    cause = error.__cause__
    assert isinstance(cause, RemoteTraceback)
    assert cause.text == error.worker_traceback
    # Real lines in Python's own format, pointing at the raising file.
    assert "\\n" not in str(cause)
    lines = str(cause).splitlines()
    assert lines[0] == f"Traceback of worker rank 0 (pid {error.pid}):"
    assert any(
        line.startswith('  File "') and "worker_failure_components.py\", line " in line
        for line in lines
    )
    assert lines[-1] == "RuntimeError: boom in a step"


def test_the_printed_error_shows_the_worker_traceback_before_the_parents(tmp_path):
    error = run_failing(tmp_path, wf_raise={})

    printed = "".join(traceback.format_exception(error))

    worker_frame = printed.index("worker_failure_components.py")
    assert printed.index("The above exception was the direct cause") > worker_frame
    assert printed.rstrip().endswith(str(error))
    assert "{'type'" not in printed


def test_a_failure_while_entering_the_session_is_reported_too(tmp_path):
    error = run_failing(tmp_path, wf_raise_on_setup={})

    assert error.exception_type == "ValueError"
    assert str(error).endswith("failed: ValueError: raised while entering the session")
    assert "worker_failure_components.py" in error.worker_traceback


def test_a_worker_that_exits_without_raising_is_a_failure(tmp_path):
    error = run_failing(tmp_path, wf_exit={"code": 3})

    assert str(error) == (
        f"Worker pid={error.pid} (rank 0) failed: it exited with code 3 "
        "without reporting an error"
    )
    assert error.exitcode == 3
    assert error.exception_type is None and error.worker_traceback is None
    assert error.__cause__ is None


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
def test_a_killed_worker_names_the_signal(tmp_path):
    error = run_failing(tmp_path, wf_kill={})

    assert str(error).endswith(
        "failed: it was killed by signal 9 (SIGKILL) without reporting an error"
    )
    assert error.exitcode == -signal.SIGKILL


def test_a_report_names_a_non_builtin_type_by_its_module():
    error = failure_from_report(
        {
            "type": "error", "rank": 1, "pid": 42,
            "exception_type": "pkg.errors.Bad", "message": "", "traceback": "",
        },
        pid=7, rank=0,
    )

    assert str(error) == "Worker pid=42 (rank 1) failed: pkg.errors.Bad"
    assert error.__cause__ is None
