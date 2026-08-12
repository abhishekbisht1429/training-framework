# from __future__ import annotations
#
# from collections import deque
# from copy import deepcopy
# from types import SimpleNamespace
# from typing import Any
#
# import pytest
#
# import training_framework.training_engine as sut
# from unittest.mock import Mock
#
#
# class FakeConfigurator:
#     def __init__(
#         self,
#         *,
#         mode: str = "new",
#         session_configs: list[dict[str, Any]] | None = None,
#         checkpoint_path: str = "session.pkl",
#         new_max_iters: int = 100,
#         process_timeout_on_join: float = 5.0,
#     ) -> None:
#         self._mode = mode
#         self._session_configs = list(session_configs or [])
#         self._checkpoint_path = checkpoint_path
#         self._new_max_iters = new_max_iters
#         self._process_timeout_on_join = process_timeout_on_join
#
#     @property
#     def mode(self) -> str:
#         return self._mode
#
#     @property
#     def session_configs(self) -> list[dict[str, Any]]:
#         return deepcopy(self._session_configs)
#
#     @property
#     def checkpoint_path(self) -> str:
#         return self._checkpoint_path
#
#     @property
#     def new_max_iters(self) -> int:
#         return self._new_max_iters
#
#     @property
#     def process_timeout_on_join(self) -> float:
#         return self._process_timeout_on_join
#
#
# @pytest.fixture
# def fake_mp(monkeypatch: pytest.MonkeyPatch):
#     """Replace multiprocessing primitives with deterministic in-memory fakes."""
#
#     class FakeEvent:
#         instances: list["FakeEvent"] = []
#
#         def __init__(self) -> None:
#             self._is_set = False
#             self.set_calls = 0
#             type(self).instances.append(self)
#
#         def set(self) -> None:
#             self.set_calls += 1
#             self._is_set = True
#
#         def is_set(self) -> bool:
#             return self._is_set
#
#     class FakeProcess:
#         instances: list["FakeProcess"] = []
#         plans: deque[dict[str, Any]] = deque()
#         log: list[tuple[Any, ...]] = []
#
#         def __init__(
#             self,
#             *,
#             name: str,
#             target,
#             args: tuple[Any, ...],
#             kwargs: dict[str, Any] | None = None,
#         ) -> None:
#             plan = dict(self.plans.popleft()) if self.plans else {}
#             self.name = name
#             self.target = target
#             self.args = args
#             self.kwargs = dict(kwargs or {})
#             self.fail_on_start = plan.get("fail_on_start")
#             self.survive_graceful = plan.get("survive_graceful", False)
#             self.survive_terminate = plan.get("survive_terminate", False)
#             self.survive_kill = plan.get("survive_kill", False)
#             self.graceful_exitcode = plan.get("graceful_exitcode", 0)
#             self.terminate_exitcode = plan.get("terminate_exitcode", -15)
#             self.kill_exitcode = plan.get("kill_exitcode", -9)
#             self.unbounded_exitcode = plan.get("unbounded_exitcode", 0)
#             self.interrupt_on_unbounded_join = plan.get(
#                 "interrupt_on_unbounded_join",
#                 False,
#             )
#             self.start_calls = 0
#             self.join_calls: list[float | None] = []
#             self.terminate_calls = 0
#             self.kill_calls = 0
#             self.close_calls = 0
#             self.exitcode: int | None = None
#             self._alive = False
#             self._join_interrupted = False
#             self.pid = len(type(self).instances) + 1000
#             type(self).instances.append(self)
#
#         def start(self) -> None:
#             self.start_calls += 1
#             type(self).log.append(("start", self.name))
#             if self.fail_on_start is not None:
#                 raise self.fail_on_start
#             self._alive = True
#
#         def join(self, timeout: float | None = None) -> None:
#             self.join_calls.append(timeout)
#             type(self).log.append(("join", self.name, timeout))
#             if (
#                 timeout is None
#                 and self.interrupt_on_unbounded_join
#                 and not self._join_interrupted
#             ):
#                 self._join_interrupted = True
#                 raise KeyboardInterrupt
#
#             if not self._alive:
#                 return
#
#             if timeout is None:
#                 self.complete(self.unbounded_exitcode)
#             elif self.kill_calls:
#                 if not self.survive_kill:
#                     self.complete(self.kill_exitcode)
#             elif self.terminate_calls:
#                 if not self.survive_terminate:
#                     self.complete(self.terminate_exitcode)
#             elif not self.survive_graceful:
#                 self.complete(self.graceful_exitcode)
#
#         def is_alive(self) -> bool:
#             return self._alive
#
#         def terminate(self) -> None:
#             self.terminate_calls += 1
#             type(self).log.append(("terminate", self.name))
#
#         def kill(self) -> None:
#             self.kill_calls += 1
#             type(self).log.append(("kill", self.name))
#
#         def close(self) -> None:
#             if self._alive:
#                 raise ValueError("cannot close a running process")
#             self.close_calls += 1
#             type(self).log.append(("close", self.name))
#
#         def complete(self, exitcode: int = 0) -> None:
#             self._alive = False
#             self.exitcode = exitcode
#
#     def default_session_factory(config: dict[str, Any]):
#         return SimpleNamespace(config=deepcopy(config))
#
#     monkeypatch.setattr(sut, "Event", FakeEvent)
#     monkeypatch.setattr(sut, "Process", FakeProcess)
#     monkeypatch.setattr(sut, "create_session_from_config", default_session_factory)
#
#     return SimpleNamespace(
#         Event=FakeEvent,
#         Process=FakeProcess,
#         Configurator=FakeConfigurator,
#     )
#
#
# class IteratorSession:
#     def __init__(self, values=(), *, on_next=None, error=None) -> None:
#         self._values = iter(values)
#         self._on_next = on_next
#         self._error = error
#         self.next_calls = 0
#         self.entered = False
#         self.exit_args = None
#
#     def __enter__(self):
#         self.entered = True
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.exit_args = (exc_type, exc_val, exc_tb)
#         return False
#
#     def __next__(self):
#         self.next_calls += 1
#         if self._on_next is not None:
#             self._on_next()
#         if self._error is not None:
#             raise self._error
#         return next(self._values)
#
#
# # ---------------------------------------------------------------------------
# # Worker behavior: the parent supplies a session and the worker makes its copy.
# # ---------------------------------------------------------------------------
#
#
# def test_session_process_worker_copies_session_and_runs_until_stop_iteration(
#     monkeypatch: pytest.MonkeyPatch,
#     capsys: pytest.CaptureFixture[str],
# ) -> None:
#     source_session = object()
#     worker_session = IteratorSession(["step-1", "step-2"])
#     stop_event = SimpleNamespace(is_set=lambda: False)
#     signal_calls: list[tuple[Any, Any]] = []
#     copy_calls: list[tuple[Any, int, Any]] = []
#
#     monkeypatch.setattr(
#         sut.signal,
#         "signal",
#         lambda sig, handler: signal_calls.append((sig, handler)),
#     )
#
#     def copy_for_worker(session, rank, session_update_params=None):
#         copy_calls.append((session, rank, session_update_params))
#         return worker_session
#
#     monkeypatch.setattr(sut, "copy_and_modify_session_for_worker", copy_for_worker)
#
#     updates = {"max_iterations": 300}
#     sut.session_process_worker(
#         source_session,
#         session_id=8,
#         rank=3,
#         stop_event=stop_event,
#         session_update_params=updates,
#     )
#
#     assert signal_calls == [(sut.signal.SIGINT, sut.signal.SIG_IGN)]
#     assert copy_calls == [(source_session, 3, updates)]
#     assert worker_session.entered is True
#     assert worker_session.next_calls == 3
#     assert worker_session.exit_args == (None, None, None)
#     assert capsys.readouterr().out == "Session 8[3] exiting.\n"
#
#
# def test_session_process_worker_defaults_update_parameters_to_none(
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     source_session = object()
#     worker_session = IteratorSession()
#     calls: list[tuple[Any, int, Any]] = []
#
#     monkeypatch.setattr(sut.signal, "signal", lambda *_: None)
#
#     def copy_for_worker(session, rank, session_update_params=None):
#         calls.append((session, rank, session_update_params))
#         return worker_session
#
#     monkeypatch.setattr(sut, "copy_and_modify_session_for_worker", copy_for_worker)
#
#     sut.session_process_worker(
#         source_session,
#         session_id=0,
#         rank=0,
#         stop_event=SimpleNamespace(is_set=lambda: True),
#     )
#
#     assert calls == [(source_session, 0, None)]
#
#
# def test_session_process_worker_observes_stop_event_between_steps(
#     monkeypatch: pytest.MonkeyPatch,
#     capsys: pytest.CaptureFixture[str],
# ) -> None:
#     class StopEvent:
#         def __init__(self) -> None:
#             self.value = False
#
#         def is_set(self) -> bool:
#             return self.value
#
#         def set(self) -> None:
#             self.value = True
#
#     stop_event = StopEvent()
#     session = IteratorSession(["unused-second-step"], on_next=stop_event.set)
#     monkeypatch.setattr(sut.signal, "signal", lambda *_: None)
#     monkeypatch.setattr(
#         sut,
#         "copy_and_modify_session_for_worker",
#         lambda source, rank, session_update_params=None: session,
#     )
#
#     sut.session_process_worker(
#         object(),
#         session_id=1,
#         rank=0,
#         stop_event=stop_event,
#     )
#
#     assert session.next_calls == 1
#     assert capsys.readouterr().out == "Session 1[0] exiting.\n"
#
#
# def test_session_process_worker_propagates_session_errors(
#     monkeypatch: pytest.MonkeyPatch,
#     capsys: pytest.CaptureFixture[str],
# ) -> None:
#     error = ValueError("bad training step")
#     session = IteratorSession(error=error)
#     monkeypatch.setattr(sut.signal, "signal", lambda *_: None)
#     monkeypatch.setattr(
#         sut,
#         "copy_and_modify_session_for_worker",
#         lambda source, rank, session_update_params=None: session,
#     )
#
#     with pytest.raises(ValueError, match="bad training step"):
#         sut.session_process_worker(
#             object(),
#             session_id=2,
#             rank=0,
#             stop_event=SimpleNamespace(is_set=lambda: False),
#         )
#
#     assert session.exit_args[0] is ValueError
#     assert capsys.readouterr().out == ""
#
#
# # ---------------------------------------------------------------------------
# # Process wrapper
# # ---------------------------------------------------------------------------
#
#
# def test_wrapper_builds_named_process_for_supplied_session(fake_mp) -> None:
#     session = object()
#     wrapper = sut.SessionProcessWrapper(session, session_id=4, rank=2)
#     process = wrapper.process
#
#     assert wrapper.session_id == 4
#     assert wrapper.rank == 2
#     assert wrapper.started is False
#     assert process.name == "training-session-4-rank-2"
#     assert process.target is sut.session_process_worker
#     assert process.args[0] is session
#     assert process.args[1:4] == (4, 2, fake_mp.Event.instances[0])
#
#
# def test_wrapper_forwards_worker_options_as_process_kwargs(fake_mp) -> None:
#     """The process target uses ``**kwargs``; options must not be positional."""
#     session = object()
#     updates = {"max_iterations": 500}
#
#     wrapper = sut.SessionProcessWrapper(
#         session,
#         session_id=4,
#         rank=2,
#         session_update_params=updates,
#     )
#
#     assert wrapper.process.args == (
#         session,
#         4,
#         2,
#         fake_mp.Event.instances[0],
#     )
#     assert wrapper.process.kwargs == {"session_update_params": updates}
#
#
# def test_wrapper_starts_once_and_can_request_stop(fake_mp) -> None:
#     wrapper = sut.SessionProcessWrapper(object(), session_id=5, rank=1)
#
#     wrapper.start()
#     wrapper.request_stop()
#
#     assert wrapper.started is True
#     assert wrapper.process.start_calls == 1
#     assert wrapper.process.args[3].is_set() is True
#
#     with pytest.raises(
#         RuntimeError,
#         match=r"Session 5\[1\].*already been started",
#     ):
#         wrapper.start()
#
#     assert wrapper.process.start_calls == 1
#
#
# def test_wrapper_is_not_marked_started_when_process_start_fails(fake_mp) -> None:
#     fake_mp.Process.plans.append({"fail_on_start": OSError("spawn failed")})
#     wrapper = sut.SessionProcessWrapper(object(), session_id=0, rank=0)
#
#     with pytest.raises(OSError, match="spawn failed"):
#         wrapper.start()
#
#     assert wrapper.started is False
#
#
# # ---------------------------------------------------------------------------
# # Registering new sessions
# # ---------------------------------------------------------------------------
#
#
# @pytest.mark.parametrize("bad_config", [None, [], 7, "not-a-mapping"])
# def test_register_session_rejects_non_mapping(fake_mp, bad_config) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#
#     with pytest.raises(TypeError, match="config must be a mapping"):
#         engine.register_session(bad_config)
#
#
# @pytest.mark.parametrize("bad_config", [{"ddp": {}}, {"ddp": None}, {"ddp": []}])
# def test_register_session_requires_ddp_world_size(fake_mp, bad_config) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#
#     with pytest.raises(ValueError, match=r"ddp\.world_size"):
#         engine.register_session(bad_config)
#
#
# @pytest.mark.parametrize("world_size", [0, -1, 1.5, "2", True, None])
# def test_register_session_rejects_invalid_world_sizes(fake_mp, world_size) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#
#     with pytest.raises(ValueError, match="positive integer"):
#         engine.register_session({"ddp": {"world_size": world_size}})
#
#
# def test_register_session_builds_parent_session_once_and_shares_it_with_wrappers(
#     fake_mp,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     created_session = object()
#     factory_calls: list[dict[str, Any]] = []
#
#     def factory(config):
#         factory_calls.append(config)
#         return created_session
#
#     monkeypatch.setattr(sut, "create_session_from_config", factory)
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     config = {"name": "distributed", "ddp": {"world_size": 3}}
#
#     session_id = engine.register_session(config)
#     wrappers = engine._session_processes[session_id]
#
#     assert factory_calls == [config]
#     assert [wrapper.rank for wrapper in wrappers] == [0, 1, 2]
#     assert all(wrapper.process.args[0] is created_session for wrapper in wrappers)
#
#
# def test_register_session_assigns_ids_and_defaults_to_one_process(fake_mp) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#
#     first_id = engine.register_session({"name": "single"})
#     second_id = engine.register_session(
#         {"name": "distributed", "ddp": {"world_size": 3}}
#     )
#
#     assert (first_id, second_id) == (0, 1)
#     assert len(engine._session_processes[0]) == 1
#     assert [wrapper.rank for wrapper in engine._session_processes[1]] == [0, 1, 2]
#
#
# # ---------------------------------------------------------------------------
# # Loading checkpoint sessions
# # ---------------------------------------------------------------------------
#
#
# class RecordingWrapper:
#     instances: list["RecordingWrapper"] = []
#
#     def __init__(self, session, session_id: int, rank: int, **kwargs) -> None:
#         self.session = session
#         self.session_id = session_id
#         self.rank = rank
#         self.kwargs = kwargs
#         type(self).instances.append(self)
#
#
# def test_load_session_loads_checkpoint_once_and_creates_one_non_ddp_wrapper(
#     fake_mp,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     class LoadedSession:
#         def has_resource(self, name: str) -> bool:
#             return False
#
#     loaded_session = LoadedSession()
#     load_calls: list[str] = []
#     RecordingWrapper.instances.clear()
#     monkeypatch.setattr(sut, "SessionProcessWrapper", RecordingWrapper)
#     monkeypatch.setattr(
#         sut.Checkpointer,
#         "load_checkpoint",
#         lambda path: load_calls.append(path) or loaded_session,
#     )
#
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     session_id = engine.load_session("session.pkl")
#
#     assert session_id == 0
#     assert load_calls == ["session.pkl"]
#     assert len(RecordingWrapper.instances) == 1
#     assert RecordingWrapper.instances[0].session is loaded_session
#     assert RecordingWrapper.instances[0].rank == 0
#
#
# def test_load_session_uses_world_size_from_ddp_resource(
#     fake_mp,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     ddp = SimpleNamespace(world_size=4)
#
#     class LoadedSession:
#         def has_resource(self, name: str) -> bool:
#             return name == "ddp"
#
#         def get_resource(self, name: str):
#             assert name == "ddp"
#             return ddp
#
#     RecordingWrapper.instances.clear()
#     monkeypatch.setattr(sut, "SessionProcessWrapper", RecordingWrapper)
#     monkeypatch.setattr(
#         sut.Checkpointer,
#         "load_checkpoint",
#         lambda path: LoadedSession(),
#     )
#
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.load_session("ddp-session.pkl")
#
#     assert [wrapper.rank for wrapper in RecordingWrapper.instances] == [0, 1, 2, 3]
#
#
# # @pytest.mark.xfail(
# #     strict=True,
# #     reason=(
# #         "e4a2f7f accepts session_update_params in load_session() but does not "
# #         "forward them to SessionProcessWrapper"
# #     ),
# # )
# def test_load_session_forwards_extension_parameters_to_every_wrapper(
#     fake_mp,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     class LoadedSession:
#         def has_resource(self, name: str) -> bool:
#             return False
#
#     RecordingWrapper.instances.clear()
#     monkeypatch.setattr(sut, "SessionProcessWrapper", RecordingWrapper)
#     monkeypatch.setattr(
#         sut.Checkpointer,
#         "load_checkpoint",
#         lambda path: LoadedSession(),
#     )
#
#     updates = {"max_iterations": 750}
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.load_session("session.pkl", session_update_params=updates)
#
#     assert RecordingWrapper.instances[0].kwargs == {
#         "session_update_params": updates,
#     }
#
#
# # ---------------------------------------------------------------------------
# # Configurator-driven engine entry
# # ---------------------------------------------------------------------------
#
#
# def test_enter_new_mode_registers_every_config(monkeypatch: pytest.MonkeyPatch) -> None:
#     configs = [{"name": "a"}, {"name": "b"}]
#     engine = sut.TrainingEngine(FakeConfigurator(session_configs=configs))
#     calls: list[dict[str, Any]] = []
#     monkeypatch.setattr(engine, "register_session", lambda config: calls.append(config))
#
#     with engine:
#         pass
#
#     assert calls == configs
#
#
# def test_enter_resume_mode_loads_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
#     engine = sut.TrainingEngine(
#         FakeConfigurator(mode="resume", checkpoint_path="resume.pkl")
#     )
#     calls: list[tuple[str, Any]] = []
#     monkeypatch.setattr(
#         engine,
#         "load_session",
#         lambda path, session_update_params=None: calls.append(
#             (path, session_update_params)
#         ),
#     )
#
#     with engine:
#         pass
#
#     assert calls == [("resume.pkl", None)]
#
# def test_enter_extend_mode_loads_checkpoint_with_new_iteration_limit(
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     engine = sut.TrainingEngine(
#         FakeConfigurator(
#             mode="extend",
#             checkpoint_path="extend.pkl",
#             new_max_iters=900,
#         )
#     )
#
#     load_session = Mock()
#     monkeypatch.setattr(engine, "load_session", load_session)
#
#     with engine:
#         pass
#
#     load_session.assert_called_once_with(
#         checkpoint_path="extend.pkl",
#         session_update_params={
#             "max_iterations": 900,
#         },
#     )
#
# def test_enter_rejects_unknown_mode() -> None:
#     engine = sut.TrainingEngine(FakeConfigurator(mode="unknown"))
#
#     with pytest.raises(RuntimeError, match="Invalid mode"):
#         with engine:
#             pass
#
#
# # ---------------------------------------------------------------------------
# # Process lifecycle retained from the previous engine tests
# # ---------------------------------------------------------------------------
#
#
# def test_start_all_starts_every_registered_worker_and_normal_exit_closes_them(
#     fake_mp,
# ) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({})
#     engine.register_session({"ddp": {"world_size": 2}})
#
#     with engine:
#         engine.start_all()
#         wrappers = list(engine._iter_wrappers())
#         assert all(wrapper.started for wrapper in wrappers)
#         assert all(wrapper.process.is_alive() for wrapper in wrappers)
#
#     assert all(wrapper.process.exitcode == 0 for wrapper in wrappers)
#     assert all(wrapper.process.close_calls == 1 for wrapper in wrappers)
#
#
# def test_start_session_cleans_up_workers_started_before_a_start_failure(fake_mp) -> None:
#     fake_mp.Process.plans.extend(
#         [
#             {},
#             {"fail_on_start": OSError("cannot spawn rank 1")},
#             {},
#         ]
#     )
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     session_id = engine.register_session({"ddp": {"world_size": 3}})
#     wrappers = engine._session_processes[session_id]
#
#     with engine:
#         with pytest.raises(OSError, match="cannot spawn rank 1"):
#             engine.start_session(session_id)
#
#         assert wrappers[0].started is True
#         assert wrappers[0].process.args[3].is_set() is True
#         assert wrappers[0].process.join_calls[0] is not None
#         assert wrappers[1].started is False
#         assert wrappers[2].started is False
#
#
# def test_request_stop_all_only_signals_started_workers(fake_mp) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({"ddp": {"world_size": 2}})
#     first, second = engine._session_processes[0]
#     first.start()
#
#     engine.request_stop_all()
#
#     assert first.process.args[3].is_set() is True
#     assert second.process.args[3].is_set() is False
#     first.process.complete()
#
#
# def test_join_or_terminate_uses_one_shared_grace_period(
#     fake_mp,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     fake_mp.Process.plans.extend(
#         [
#             {"survive_graceful": True},
#             {"survive_graceful": True},
#         ]
#     )
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({"ddp": {"world_size": 2}})
#     first, second = engine._session_processes[0]
#     first.start()
#     second.start()
#     ticks = iter([100.0, 101.0, 104.25])
#     monkeypatch.setattr(sut.time, "monotonic", lambda: next(ticks))
#
#     engine._join_or_terminate([first, second], timeout=5.0)
#
#     assert first.process.join_calls == [pytest.approx(4.0), 1.0]
#     assert second.process.join_calls == [pytest.approx(0.75), 1.0]
#     assert first.process.terminate_calls == 1
#     assert second.process.terminate_calls == 1
#
#
# def test_join_or_terminate_kills_process_that_survives_terminate(
#     fake_mp,
#     monkeypatch: pytest.MonkeyPatch,
# ) -> None:
#     fake_mp.Process.plans.append(
#         {
#             "survive_graceful": True,
#             "survive_terminate": True,
#             "survive_kill": False,
#         }
#     )
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({})
#     wrapper = engine._session_processes[0][0]
#     wrapper.start()
#     monkeypatch.setattr(sut.time, "monotonic", lambda: 0.0)
#
#     engine._join_or_terminate([wrapper], timeout=2.0)
#
#     assert wrapper.process.join_calls == [2.0, 1.0, 1.0]
#     assert wrapper.process.terminate_calls == 1
#     assert wrapper.process.kill_calls == 1
#     assert wrapper.process.exitcode == -9
#
#
# def test_raise_worker_failures_aggregates_session_rank_and_exitcode(fake_mp) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({"ddp": {"world_size": 2}})
#     engine.register_session({})
#     wrappers = list(engine._iter_wrappers())
#
#     for wrapper in wrappers:
#         wrapper.start()
#
#     wrappers[0].process.complete(0)
#     wrappers[1].process.complete(7)
#     wrappers[2].process.complete(-9)
#
#     with pytest.raises(RuntimeError) as exc_info:
#         engine._raise_worker_failures()
#
#     message = str(exc_info.value)
#     assert "session=0, rank=1, exitcode=7" in message
#     assert "session=1, rank=0, exitcode=-9" in message
#     assert "session=0, rank=0" not in message
#
#
# def test_raise_worker_failures_ignores_unstarted_running_and_successful_workers(
#     fake_mp,
# ) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({"ddp": {"world_size": 3}})
#     successful, running, unstarted = engine._session_processes[0]
#
#     successful.start()
#     successful.process.complete(0)
#     running.start()
#     unstarted.process.exitcode = 12
#
#     engine._raise_worker_failures()
#     running.process.complete(0)
#
#
# def test_close_resources_only_closes_started_dead_processes(fake_mp) -> None:
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     engine.register_session({"ddp": {"world_size": 3}})
#     dead, alive, unstarted = engine._session_processes[0]
#
#     dead.start()
#     dead.process.complete(0)
#     alive.start()
#
#     engine._close_resources()
#
#     assert dead.process.close_calls == 1
#     assert alive.process.close_calls == 0
#     assert unstarted.process.close_calls == 0
#     alive.process.complete(0)
#
#
# def test_normal_context_exit_reports_worker_failure_and_still_closes_process(
#     fake_mp,
# ) -> None:
#     fake_mp.Process.plans.append({"unbounded_exitcode": 23})
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     session_id = engine.register_session({})
#     process = engine._session_processes[session_id][0].process
#
#     with pytest.raises(RuntimeError, match=r"exitcode=23"):
#         with engine:
#             engine.start_session(session_id)
#
#     assert process.join_calls == [None]
#     assert process.close_calls == 1
#
#
# def test_context_body_exception_is_preserved_after_children_are_stopped(fake_mp) -> None:
#     fake_mp.Process.plans.append({"graceful_exitcode": 17})
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     session_id = engine.register_session({})
#     wrapper = engine._session_processes[session_id][0]
#
#     with pytest.raises(LookupError, match="body failed"):
#         with engine:
#             engine.start_session(session_id)
#             raise LookupError("body failed")
#
#     assert wrapper.process.args[3].is_set() is True
#     assert wrapper.process.join_calls[0] is not None
#     assert wrapper.process.exitcode == 17
#     assert wrapper.process.close_calls == 1
#
#
# def test_keyboard_interrupt_while_waiting_stops_children_and_is_reraised(
#     fake_mp,
# ) -> None:
#     fake_mp.Process.plans.append({"interrupt_on_unbounded_join": True})
#     engine = sut.TrainingEngine(fake_mp.Configurator())
#     session_id = engine.register_session({})
#     wrapper = engine._session_processes[session_id][0]
#
#     with pytest.raises(KeyboardInterrupt):
#         with engine:
#             engine.start_session(session_id)
#
#     assert wrapper.process.args[3].is_set() is True
#     assert wrapper.process.join_calls[0] is None
#     assert wrapper.process.join_calls[1] is not None
#     assert wrapper.process.close_calls == 1