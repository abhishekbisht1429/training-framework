from __future__ import annotations

import pytest

from training_framework.components import Resource, Step, resource, step
from training_framework.session import TrainingSession


class _Conn:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class _TeardownRecorder(Resource):
    torn_down = False

    def __init__(self, config):
        pass

    def setup(self, session):
        pass

    def teardown(self, session):
        type(self).torn_down = True


class _RankStub(Resource):
    def __init__(self, config):
        self.rank = config["rank"]

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


class _Explode(Step):
    def __init__(self, config):
        pass

    def run(self, session):
        raise RuntimeError("step exploded")


def _session(tmp_path, *, with_ddp: bool) -> TrainingSession:
    resource("report_teardown_recorder")(_TeardownRecorder)
    resource("report_rank_stub")(_RankStub)
    step("report_explode")(_Explode)
    _TeardownRecorder.torn_down = False
    config = {
        "session_config": {
            "rng_seed": 1,
            "sessions_dir": str(tmp_path),
            "max_iterations": 1,
            "device": "cpu",
            "components_package": "training_framework.components.builtin",
            "show_execution_graph": False,
        },
        "report_teardown_recorder": {},
        "report_explode": {},
    }
    if with_ddp:
        config["component_bindings"] = {"ddp": "report_rank_stub"}
        config["report_rank_stub"] = {"rank": 3}
    return TrainingSession(config)


@pytest.mark.parametrize(("with_ddp", "rank"), [(False, 0), (True, 3)])
def test_worker_exception_is_reported_and_resources_torn_down(
        tmp_path,
        with_ddp,
        rank,
):
    session = _session(tmp_path, with_ddp=with_ddp)
    conn = _Conn()
    session.set_dist_manager_err_conn(conn)

    with pytest.raises(RuntimeError, match="step exploded"):
        with session:
            next(session)

    [message] = conn.messages
    assert message["type"] == "error"
    assert message["rank"] == rank
    assert "step exploded" in message["message"]
    assert _TeardownRecorder.torn_down is True


def test_teardown_runs_even_if_reporting_fails(tmp_path):
    session = _session(tmp_path, with_ddp=False)

    class _BrokenConn:
        def send(self, message):
            raise BrokenPipeError("parent gone")

    session.set_dist_manager_err_conn(_BrokenConn())

    with pytest.raises(BrokenPipeError):
        with session:
            next(session)

    assert _TeardownRecorder.torn_down is True
