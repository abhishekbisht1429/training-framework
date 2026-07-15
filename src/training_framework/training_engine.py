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

    @requires_context
    def _run_session(self, session_id: int):
        session = self._sessions[session_id]
        # acquire session lock
        with self._session_locks[session_id]:
            # enter the session context
            with session:
                try:
                    while hasattr(self, "_active") and self._active:
                            next(session)
                except StopIteration:
                    pass

        print(f"Session {session_id} is exiting.")

    def register_session(self, session: TrainingSession):
        if not isinstance(session, TrainingSession):
            raise TypeError(f"The provided object '{type(session).__name__}' is not an instance of TrainingSession!")

        session_id = len(self._sessions)
        self._sessions.append(session)
        self._session_threads.append(threading.Thread(target=self._run_session, args=(session_id,)))
        self._session_locks.append(threading.Lock())

    @requires_context
    def run_all(self, wait=True):
        if len(self._sessions) == 0:
            raise RuntimeError("There are no sessions registered to run!")

        for session_thread in self._session_threads:
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
        pass
