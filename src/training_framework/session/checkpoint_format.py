"""The on-disk form of a checkpoint: a directory of plain data.

A session's state (`Session.get_state()`) is already a graph of components
and each component's state. A checkpoint writes it out so that each part can
be read on its own:

    <checkpoint>/
      manifest.json          what the checkpoint holds, readable by anything
      session.pt             the session's own state and every component's
                             record except its state
      components/<name>.pt   one component's state

Every `.pt` file holds plain data only -- tensors, containers, numbers and
strings -- and is read with `torch.load(weights_only=True)`. Reading a
checkpoint therefore imports no class, and renaming or moving a class can no
longer make one unreadable. What is not plain data is refused when the
checkpoint is written, naming where it is, rather than when it is read.

The directory is written under a temporary name and renamed into place once
complete, so an interrupted save never looks like a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import torch

#: Bumped when the layout of the directory or the manifest changes.
FORMAT_VERSION = 1

MANIFEST_FILE = "manifest.json"
SESSION_FILE = "session.pt"
COMPONENTS_DIR = "components"
_TEMPORARY_SUFFIX = ".tmp"

# What `torch.load(weights_only=True)` reads back, by exact type: a subclass
# (a namedtuple, a defaultdict) is a class the reader would have to import.
_PLAIN_LEAVES = (
    type(None), bool, int, float, complex, str, bytes,
    torch.Size, torch.dtype, torch.device,
)
_PLAIN_MAPPINGS = (dict, OrderedDict)
_PLAIN_SEQUENCES = (list, tuple, set)


class CheckpointFormatError(ValueError):
    """A checkpoint directory that cannot be written or read as one."""


def is_checkpoint_directory(path) -> bool:
    return os.path.isdir(os.fspath(path))


# -- writing ---------------------------------------------------------------------


def write_checkpoint(state: Mapping[str, Any], path) -> str:
    """Write a session's state as a checkpoint directory at `path`."""
    final = os.fspath(path)
    if os.path.exists(final):
        raise FileExistsError(f"Checkpoint already exists: {final}")
    temporary = final + _TEMPORARY_SUFFIX
    shutil.rmtree(temporary, ignore_errors=True)
    os.makedirs(os.path.join(temporary, COMPONENTS_DIR))
    try:
        _write_contents(state, temporary)
        _fsync_directory(os.path.join(temporary, COMPONENTS_DIR))
        _fsync_directory(temporary)
        os.rename(temporary, final)
        _fsync_directory(os.path.dirname(os.path.abspath(final)))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _write_contents(state: Mapping[str, Any], directory: str) -> None:
    session_part = {
        key: value
        for key, value in state.items()
        if key != "components_state"
    }
    for key, value in session_part.items():
        require_plain(value, f"the session's '{key}'")

    # Everything is checked before anything is written, so a refused
    # checkpoint costs no disk writes.
    components_state = state["components_state"]
    for name, info in components_state.items():
        component_file(name)
        for key, value in info.items():
            where = "state" if key == "state" else f"'{key}'"
            require_plain(value, f"component '{name}' {where}")

    records = {}
    for name, info in components_state.items():
        file = component_file(name)
        record = {key: value for key, value in info.items() if key != "state"}
        record["file"] = file
        record["sha256"] = _save(info.get("state"), os.path.join(directory, file))
        records[name] = record
    session_part["components"] = records
    _save(session_part, os.path.join(directory, SESSION_FILE))

    # Written last: a directory without it was never finished.
    with open(os.path.join(directory, MANIFEST_FILE), "w") as manifest_file:
        json.dump(_manifest(state, records), manifest_file, indent=2, default=repr)
        manifest_file.flush()
        os.fsync(manifest_file.fileno())


def component_file(name: str) -> str:
    """The file, relative to the checkpoint, holding `name`'s state."""
    if not name or name in (".", "..") or "/" in name or os.sep in name:
        raise CheckpointFormatError(
            f"Component name {name!r} cannot be used as a checkpoint file name"
        )
    return f"{COMPONENTS_DIR}/{name}.pt"


def _manifest(state: Mapping[str, Any], records: Mapping[str, Mapping]) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "framework_version": _framework_version(),
        "state_version": state.get("checkpoint_version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_type": state.get("session_type"),
        "iteration": state.get("iteration"),
        "config": state.get("config"),
        "components": {
            name: {
                key: record[key]
                for key in (
                    "implementation",
                    "component_type",
                    "state_version",
                    "dependencies",
                    "context_reads",
                    "context_writes",
                    "file",
                    "sha256",
                )
                if key in record
            }
            for name, record in records.items()
        },
    }


def _framework_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("training-framework")
    except PackageNotFoundError:
        return None


def _save(value: Any, path: str) -> str:
    with open(path, "wb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    return _sha256(path)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: str) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def require_plain(value: Any, where: str) -> None:
    """Raise unless `value` is data `torch.load(weights_only=True)` reads.

    Checked on the way out, so a component holding something else -- an
    object, a function, a class -- is named when the checkpoint is written
    instead of making it unreadable later.
    """
    found = _first_non_plain(value, "")
    if found is not None:
        location, offending = found
        raise CheckpointFormatError(
            f"Cannot checkpoint {where}: "
            f"{('the value at ' + location) if location else 'the value'} is "
            f"a {type(offending).__module__}.{type(offending).__qualname__}, "
            "which is not plain data. A checkpoint holds only tensors, dicts, "
            "lists, tuples, sets, numbers, strings and None, so that it can "
            "be read without importing any class. Store the object's state "
            "(or the arguments that rebuild it) instead."
        )


def _first_non_plain(value: Any, location: str):
    kind = type(value)
    if kind in _PLAIN_LEAVES or kind in (torch.Tensor, torch.nn.Parameter):
        return None
    if isinstance(value, torch.Tensor):
        # A Parameter or tensor subclass: its class is pickled with it.
        return location, value
    if kind in _PLAIN_MAPPINGS:
        for key, item in value.items():
            found = _first_non_plain(key, f"{location}[key {key!r}]")
            if found is None:
                found = _first_non_plain(item, f"{location}[{key!r}]")
            if found is not None:
                return found
        return None
    if kind in _PLAIN_SEQUENCES:
        for index, item in enumerate(value):
            found = _first_non_plain(item, f"{location}[{index}]")
            if found is not None:
                return found
        return None
    return location, value


# -- reading ---------------------------------------------------------------------


def read_manifest(path) -> dict[str, Any]:
    """Return what a checkpoint holds, without reading any state."""
    directory = os.fspath(path)
    if directory.rstrip(os.sep).endswith(_TEMPORARY_SUFFIX):
        raise CheckpointFormatError(
            f"{directory} is an unfinished checkpoint: its save was "
            "interrupted"
        )
    manifest_path = _contained_file(directory, MANIFEST_FILE, "manifest")
    if not os.path.isfile(manifest_path):
        raise CheckpointFormatError(
            f"{directory} is not a checkpoint: it has no {MANIFEST_FILE}"
        )
    with open(manifest_path) as manifest_file:
        manifest = json.load(manifest_file)
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise CheckpointFormatError(
            f"{directory} has checkpoint format version {version!r}; this "
            f"version of the framework reads up to {FORMAT_VERSION}"
        )
    return manifest


def read_checkpoint(
        path,
        *,
        map_location="cpu",
        components: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return the session state a checkpoint directory holds.

    `components`, when given, limits the component states read to those
    names; the others are left out of `components_state` entirely, and
    their files are never opened.
    """
    directory = os.fspath(path)
    read_manifest(directory)
    session_part = _load(
        _contained_file(directory, SESSION_FILE, "session"), map_location,
    )
    records = session_part.pop("components")
    wanted = None if components is None else set(components)
    if wanted is not None:
        unknown = sorted(wanted - set(records))
        if unknown:
            raise KeyError(
                f"Checkpoint {directory} has no component {unknown}; it "
                f"holds {sorted(records)}"
            )

    components_state = {}
    # In stored order, which restore relies on.
    for name, record in records.items():
        if wanted is None or name in wanted:
            components_state[name] = _component_info(
                directory, name, record, map_location,
            )
    session_part["components_state"] = components_state
    return session_part


def read_session_record(path, *, map_location="cpu") -> dict[str, Any]:
    """Return the session's own state and every component's record -- what
    it is, how it was built and wired -- without any component's state."""
    directory = os.fspath(path)
    read_manifest(directory)
    return _load(
        _contained_file(directory, SESSION_FILE, "session"), map_location,
    )


def read_component_state(path, name: str, *, map_location="cpu") -> Any:
    """Return one component's saved state, reading only its own file."""
    directory = os.fspath(path)
    manifest = read_manifest(directory)
    record = manifest["components"].get(name)
    if record is None:
        raise KeyError(
            f"Checkpoint {directory} has no component '{name}'; it holds "
            f"{sorted(manifest['components'])}"
        )
    return _component_info(directory, name, record, map_location)["state"]


def _component_info(
        directory: str,
        name: str,
        record: Mapping[str, Any],
        map_location,
) -> dict[str, Any]:
    info = dict(record)
    # Where a component's state lives follows from its name alone. The
    # recorded path is only cross-checked, never followed: a checkpoint is
    # not trusted to say which file to open.
    file = component_file(name)
    recorded_file = info.pop("file", None)
    if recorded_file != file:
        raise CheckpointFormatError(
            f"Checkpoint {directory} records the state of component "
            f"'{name}' at {recorded_file!r}; it can only be at {file!r}"
        )
    expected = info.pop("sha256", None)
    file_path = _contained_file(directory, file, f"component '{name}'")
    if expected is not None and os.path.isfile(file_path):
        if _sha256(file_path) != expected:
            raise CheckpointFormatError(
                f"Checkpoint {directory}: the state of component '{name}' "
                f"({file}) does not match its checksum; the file is damaged"
            )
    info["state"] = _load(file_path, map_location, owner=f"component '{name}'")
    return info


def _contained_file(directory: str, relative: str, owner: str) -> str:
    """Return `relative` inside the checkpoint `directory`, refusing any
    path that leaves it -- through a symlink or otherwise.

    The checkpoint directory itself may be reached through a symlink; what
    it contains may not point elsewhere.
    """
    path = os.path.join(directory, relative)
    current = directory
    for part in relative.split("/"):
        current = os.path.join(current, part)
        if os.path.islink(current):
            raise CheckpointFormatError(
                f"Checkpoint {directory}: the {owner} file {relative} goes "
                f"through a symlink ({current}); a checkpoint's files must be "
                "inside it"
            )
    root = os.path.realpath(directory)
    if os.path.realpath(path) != os.path.join(root, *relative.split("/")):
        raise CheckpointFormatError(
            f"Checkpoint {directory}: the {owner} file {relative} is not "
            "inside the checkpoint"
        )
    return path


def _load(path: str, map_location, owner: str = "session") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except FileNotFoundError:
        raise CheckpointFormatError(
            f"The checkpoint file holding the {owner}'s state is missing: "
            f"{path}"
        ) from None
    except Exception as error:
        raise CheckpointFormatError(
            f"Cannot read the {owner}'s state from {path}: "
            f"{str(error).splitlines()[0]}"
        ) from error


__all__ = [
    "CheckpointFormatError",
    "FORMAT_VERSION",
    "component_file",
    "is_checkpoint_directory",
    "read_checkpoint",
    "read_component_state",
    "read_manifest",
    "read_session_record",
    "require_plain",
    "write_checkpoint",
]
