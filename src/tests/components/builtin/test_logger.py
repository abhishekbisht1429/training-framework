from types import SimpleNamespace

import pytest

from training_framework.components.builtin.analysis import AnalysisLogger
from training_framework.components.builtin.observability import Logger


def _session(iteration):
    return SimpleNamespace(
        iteration=iteration,
        session_config=SimpleNamespace(max_iterations=10),
    )


def _run(logger_type, log_path, iteration):
    """Drive a logger through one session that logs a single iteration."""
    logger = logger_type({"log_every": 1, "log_file": str(log_path)})
    session = _session(iteration)
    logger.pre_session(session)
    session.iteration = iteration + 1
    logger.pre_iteration_callback(session)
    logger.post_session(session)


@pytest.mark.parametrize("logger_type", [Logger, AnalysisLogger])
def test_log_file_in_missing_directory_is_created(tmp_path, logger_type):
    log_path = tmp_path / "runs" / "nested" / "train.log"

    _run(logger_type, log_path, iteration=0)

    assert log_path.read_text().endswith("teration 1/10\n")


@pytest.mark.parametrize("logger_type", [Logger, AnalysisLogger])
def test_unopenable_log_file_fails_before_the_run(tmp_path, logger_type):
    log_path = tmp_path / "a_directory"
    log_path.mkdir()
    logger = logger_type({"log_every": 1, "log_file": str(log_path)})
    session = _session(0)

    with pytest.raises(OSError, match="a_directory"):
        logger.pre_session(session)

    logger.post_session(session)


@pytest.mark.parametrize("logger_type", [Logger, AnalysisLogger])
def test_resumed_session_appends_to_the_log(tmp_path, logger_type):
    log_path = tmp_path / "train.log"

    _run(logger_type, log_path, iteration=0)
    _run(logger_type, log_path, iteration=5)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("teration 1/10")
    assert lines[1].endswith("teration 6/10")


def test_fresh_session_starts_the_log_clean(tmp_path):
    log_path = tmp_path / "train.log"

    _run(Logger, log_path, iteration=0)
    _run(Logger, log_path, iteration=0)

    assert log_path.read_text() == "Iteration 1/10\n"


def test_post_session_twice_does_not_raise(tmp_path):
    logger = Logger({"log_every": 1, "log_file": str(tmp_path / "train.log")})
    session = _session(0)
    logger.pre_session(session)

    logger.post_session(session)
    logger.post_session(session)
