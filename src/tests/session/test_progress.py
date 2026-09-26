from __future__ import annotations

from types import SimpleNamespace

from torch import multiprocessing

from training_framework.session import runtime
from training_framework.session.progress import ProgressBeacon


def _beacon() -> ProgressBeacon:
    return ProgressBeacon(multiprocessing.get_context("spawn"))


def _mark_in_child(beacon: ProgressBeacon) -> None:
    beacon.mark("Marked by child", 7)


def test_mark_increments_sequence_and_records_stage():
    beacon = _beacon()
    assert beacon.snapshot() == (0, 0, "")

    beacon.mark("Running setup model", 0)
    beacon.mark("Running train", 3)

    assert beacon.snapshot() == (2, 3, "Running train")


def test_mark_truncates_long_multibyte_stage_on_character_boundary():
    beacon = _beacon()
    beacon.mark("é" * 300, 1)

    _, _, stage = beacon.snapshot()

    assert set(stage) == {"é"}
    assert len(stage.encode("utf-8")) <= 255


def test_snapshot_sees_marks_from_spawned_child():
    context = multiprocessing.get_context("spawn")
    beacon = ProgressBeacon(context)
    process = context.Process(target=_mark_in_child, args=(beacon,))
    process.start()
    process.join(timeout=30.0)

    assert process.exitcode == 0
    assert beacon.snapshot() == (1, 7, "Marked by child")


def test_send_heartbeat_marks_every_call_without_rate_limit():
    beacon = _beacon()
    session = SimpleNamespace(_progress_beacon=beacon, _iteration=4)

    runtime.send_heartbeat(session, "first")
    runtime.send_heartbeat(session, "second")

    assert beacon.snapshot() == (2, 4, "second")


def test_send_heartbeat_without_beacon_is_a_no_op():
    runtime.send_heartbeat(
        SimpleNamespace(_progress_beacon=None, _iteration=0),
        "ignored",
    )


def test_pre_session_stage_is_marked_before_the_hook_runs():
    observed = []

    class Hook:
        id = "hook"
        name = "hook"

        def pre_session(self, session):
            observed.append(session.stages[-1])

    session = SimpleNamespace(
        _session_hooks=[Hook()],
        _successfully_setup_hook_names=set(),
        stages=[],
    )
    session.send_heartbeat = session.stages.append

    runtime.setup_session_hooks(session)

    assert observed == ["Running pre-session hook"]
