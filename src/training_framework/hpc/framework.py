"""The only module of the HPC layer that imports framework internals.

The session-config helpers below are not exported by
`training_framework.components`; wrapping them here keeps that dependency in
one place, pinned by compatibility tests. A public session-config editing API
is a deferred framework change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training_framework.components.config import (
    COMPONENT_GROUPS,
    component_entry,
    find_component_entry,
    reserved_config_names,
    with_component_entry,
)

__all__ = [
    "COMPONENT_GROUPS",
    "component_entry_of",
    "is_listed_component",
    "reserved_names",
    "with_entry",
]


def reserved_names(session_type: str | None) -> frozenset[str]:
    """Top-level session keys that are not component names."""
    return reserved_config_names(session_type)


def is_listed_component(session: Mapping, name: str) -> bool:
    """Whether `name` is listed as a component, at the top level or in a
    `resources`/`hooks`/`steps` group."""
    reserved = reserved_names(session.get("session_type"))
    if name in reserved:
        return False
    return find_component_entry(session, name) is not None


def component_entry_of(session: Mapping, name: str) -> Any:
    """`name`'s entry as written; an empty entry reads as `{}`."""
    entry = component_entry(session, name)
    return {} if entry is None else entry


def with_entry(session: Mapping, name: str, entry: Any) -> dict:
    """A copy of `session` with `name`'s entry replaced where it is listed."""
    return with_component_entry(session, name, entry)
