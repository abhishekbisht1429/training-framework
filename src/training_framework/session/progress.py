import ctypes


_STAGE_CAPACITY = 256
_SNAPSHOT_RETRIES = 5


class ProgressBeacon:
    """Lock-free shared-memory record of a worker's latest progress.

    The worker marks progress on every stage change (a few shared-memory
    writes, no syscall or pickling). The parent polls ``snapshot()`` and
    treats any change of ``seq`` as proof of progress. Instances can only be
    shared with a child process as a ``Process`` argument at spawn time.
    """

    def __init__(self, context):
        self._seq = context.RawValue(ctypes.c_uint64, 0)
        self._iteration = context.RawValue(ctypes.c_int64, 0)
        self._stage = context.RawArray(ctypes.c_char, _STAGE_CAPACITY)

    def mark(self, stage, iteration: int) -> None:
        encoded = str(stage).encode("utf-8")
        if len(encoded) >= _STAGE_CAPACITY:
            encoded = encoded[:_STAGE_CAPACITY - 1].decode(
                "utf-8", errors="ignore"
            ).encode("utf-8")
        self._stage.value = encoded
        self._iteration.value = iteration
        self._seq.value += 1

    def snapshot(self) -> tuple[int, int, str]:
        seq = self._seq.value
        for _ in range(_SNAPSHOT_RETRIES):
            iteration = self._iteration.value
            stage = self._stage.value
            current_seq = self._seq.value
            if current_seq == seq:
                break
            seq = current_seq
        return seq, iteration, stage.decode("utf-8", errors="replace")
