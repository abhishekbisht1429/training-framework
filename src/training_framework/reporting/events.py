"""Reading the scalars of one TensorBoard event file, every record of it.

TensorBoard's `EventAccumulator` is not used: by default it keeps a
10 000-point sample per tag, drops points when a step goes backwards (which
every resume does), and only logs a warning when a file is truncated or
corrupt. Here every record is read, nothing is dropped, and a damaged file
says where it is damaged.

The file is a sequence of records: an 8-byte little-endian length, the
masked CRC-32C of those 8 bytes, the serialized `Event`, and its masked
CRC-32C.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
from tensorboard.util import tensor_util

#: The plugin TensorBoard writes tensor-valued scalars under.
_SCALARS_PLUGIN = "scalars"


@dataclass(frozen=True)
class Point:
    step: int
    value: float
    wall_time: float


@dataclass
class EventFileScalars:
    """The scalars of one event file, in the order they were written.

    `damage` describes why reading stopped before the end of the file
    (truncated or corrupt record), or is None when the whole file was read.
    """

    path: str
    scalars: dict[str, list[Point]] = field(default_factory=dict)
    records: int = 0
    damage: str | None = None

    @property
    def first_wall_time(self) -> float | None:
        times = [points[0].wall_time for points in self.scalars.values() if points]
        return min(times) if times else None


def read_event_file(path: str) -> EventFileScalars:
    """Every scalar in the event file at `path`.

    Raises OSError when the file cannot be opened; a damaged record ends
    the read and is reported in `damage`, with what came before it kept.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    result = EventFileScalars(path=path)
    # A tensor-valued scalar carries its plugin only in the first value
    # of its tag.
    plugins: dict[str, str] = {}
    offset = 0
    size = len(data)
    while offset < size:
        damage = _record_damage(data, offset)
        if damage is not None:
            result.damage = (
                f"{damage} at byte {offset} of {size}; the "
                f"{size - offset} byte(s) from there were not read"
            )
            break
        (length,) = struct.unpack_from("<Q", data, offset)
        payload = data[offset + 12: offset + 12 + length]
        offset += 12 + length + 4
        try:
            event = event_pb2.Event.FromString(payload)
        except Exception as error:  # protobuf raises its own DecodeError
            result.damage = (
                f"record {result.records + 1} is not an Event ({error}); "
                f"the {size - offset} byte(s) after it were not read"
            )
            break
        result.records += 1
        if not event.HasField("summary"):
            continue
        for value in event.summary.value:
            number = _scalar(value, plugins)
            if number is not None:
                result.scalars.setdefault(value.tag, []).append(
                    Point(int(event.step), number, float(event.wall_time))
                )
    return result


def _record_damage(data: bytes, offset: int) -> str | None:
    """Why the record at `offset` cannot be read, or None when it can."""
    size = len(data)
    if size - offset < 12:
        return "truncated record header"
    header = data[offset: offset + 8]
    (header_crc,) = struct.unpack_from("<I", data, offset + 8)
    if masked_crc32c(header) != header_crc:
        return "corrupt record header (CRC mismatch)"
    (length,) = struct.unpack_from("<Q", data, offset)
    end = offset + 12 + length
    if end + 4 > size:
        return "truncated record"
    (payload_crc,) = struct.unpack_from("<I", data, end)
    if masked_crc32c(data[offset + 12: end]) != payload_crc:
        return "corrupt record (CRC mismatch)"
    return None


def _scalar(value, plugins: dict[str, str]) -> float | None:
    """The number a summary value holds, when it is a scalar."""
    kind = value.WhichOneof("value")
    if kind == "simple_value":
        return float(value.simple_value)
    if value.HasField("metadata") and value.metadata.plugin_data.plugin_name:
        plugins[value.tag] = value.metadata.plugin_data.plugin_name
    if kind != "tensor" or plugins.get(value.tag) != _SCALARS_PLUGIN:
        return None
    array = tensor_util.make_ndarray(value.tensor)
    if array.size != 1:
        return None
    return float(array.reshape(()))
