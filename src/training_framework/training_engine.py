import threading
from typing import List

from training_framework.training_session import TrainingSession
from training_framework.util import requires_context, context_entry, context_exit

class TrainingEngine:

    def __init__(self, config):
        self._config = config

        self._sessions: List[TrainingSession] = []
        self._session_threads: List[threading.Thread] = []
        self._session_locks: List[threading.Lock] = []
        self._session_active_flags: List[bool] = []
        self._session_active_flag_locks: List[threading.Lock] = []

    @requires_context
    def _run_session(self, session_id: int):
        session = self._sessions[session_id]
        # set active flag to true
        with self._session_active_flag_locks[session_id]:
            self._session_active_flags[session_id] = True

        # acquire session lock
        with self._session_locks[session_id], session:
            try:
                while True:
                    with self._session_active_flag_locks[session_id]:
                        if not self._session_active_flags[session_id]:
                            break
                    next(session)
            except StopIteration:
                pass

        print(f"Session {session_id} is exiting.")

    def register_session(self, session: TrainingSession):
        if not isinstance(session, TrainingSession):
            raise TypeError(f"The provided object '{type(session).__name__}' is not an instance of TrainingSession!")
        if any(existing is session for existing in self._sessions):
            raise RuntimeError(f"Session '{session}' is already registered!")
        # TODO: Implement a session registry that store current owner engine of a session

        session_id = len(self._sessions)
        self._sessions.append(session)
        self._session_threads.append(threading.Thread(target=self._run_session, args=(session_id,)))
        self._session_locks.append(threading.Lock())
        self._session_active_flags.append(False)
        self._session_active_flag_locks.append(threading.Lock())

    @requires_context
    def run_all(self, wait=True):
        if len(self._sessions) == 0:
            raise RuntimeError("There are no sessions registered to run!")

        for i, session_thread in enumerate(self._session_threads):
            print("Starting session", session_thread)
            session_thread.start()

        if wait:
            for session_thread in self._session_threads:
                session_thread.join()

    @context_entry
    def __enter__(self):
        pass

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, flag_lock in enumerate(self._session_active_flag_locks):
            with flag_lock:
                self._session_active_flags[i] = False

        # Wait for all the threads to finish before exiting context
        for session_thread in self._session_threads:
            if session_thread.is_alive():
                session_thread.join()
