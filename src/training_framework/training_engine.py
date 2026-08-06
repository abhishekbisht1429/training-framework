import collections
import threading
from multiprocessing import Process, Queue
from typing import List

from training_framework.configurator import create_session_from_config
from training_framework.training_session import TrainingSession
from training_framework.util import requires_context, context_entry, context_exit

def ddp_proc_worker(config: dict, rank: int, queue):
    session = create_session_from_config(config, rank=rank)

    with session:
        try:
            # when a value is put in the queue, then pause the session
            while queue.empty():
                next(session)
        except StopIteration:
            pass

        signal = queue.get()
        print(f"Parallel Session {rank}: signal {signal} received.")

def proc_worker(config: dict, session_id: int, queue: Queue):
    print("Starting session", session_id)

    session = create_session_from_config(config)

    if "ddp" in config:
        n_proc = config["ddp"]["n_proc"]
        ddp_processes = []
        signal_queues = []
        for i in range(n_proc-1):
            rank = i + 1
            signal_queues.append(Queue(maxsize=1))
            ddp_processes.append(Process(target=ddp_proc_worker, args=(config, rank, signal_queues[i])))
            ddp_processes[-1].start()
    else:
        with session:
            try:
                # when a value is put in the queue, then pause the session
                while queue.empty():
                    next(session)
            except StopIteration:
                pass

            signal = queue.get()
            print(f"signal {signal} received.")

class TrainingEngine:
    def __init__(self, config):
        self._config = config
        self._session_processes: List[Process] = []
        self._signal_queues: List[Queue] = []

    def register_session(self, config: dict):
        if not isinstance(config, collections.abc.Mapping):
            raise TypeError(f"config is not a dict: {config}")
        session_id = len(self._session_processes)
        self._signal_queues.append(Queue(maxsize=1))
        self._session_processes.append(
            Process(
                target=proc_worker,
                args=(config, session_id, self._signal_queues[session_id])
            )
        )

        return session_id

    @requires_context
    def start_session(self, session_id: int):
        session_process: Process = self._session_processes[session_id]
        session_process.start()

    @requires_context
    def start_all(self):
        for id in range(len(self._session_processes)):
            self.start_session(id)

    @context_entry
    def __enter__(self):
        pass

    @context_exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        for queue in self._signal_queues:
            queue.put(1)

        for process in self._session_processes:
            process.join(timeout=2.0)
            if process.is_alive():
                print('terminating process forcefully')
                process.terminate()

