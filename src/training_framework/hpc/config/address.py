"""Config addresses (building block B1): one value in a session config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

from training_framework.hpc import framework

#: Session-level paths the framework owns; no address may reach them.
FRAMEWORK_OWNED = (("session_config", "sessions_dir"),)

_MISSING = object()


@dataclass(frozen=True)
class Address:
    """`component` (optional) plus a path of segments inside it, or inside
    the selected session when there is no component."""

    component: str | None
    path: tuple[str, ...]

    def describe(self) -> str:
        path = _key(self.path)
        if self.component is None:
            return path
        return f"{self.component}: {path}"

    def as_json(self) -> dict:
        data: dict[str, Any] = {"path": list(self.path)}
        if self.component is not None:
            data["component"] = self.component
        return data


def parse_address(raw: Any, where: str) -> tuple[Address | None, list[str]]:
    """Read `{component?, path}`; return the address or the problems."""
    if not isinstance(raw, Mapping):
        return None, [f"{where} must be a mapping with 'path'"]
    problems = []
    component = raw.get("component")
    if component is not None and (
            not isinstance(component, str) or not component
    ):
        problems.append(f"{where}.component must be a non-empty string")
    if "path" not in raw:
        problems.append(f"{where}.path is required")
        return None, problems
    path = _segments(raw["path"])
    if path is None:
        problems.append(
            f"{where}.path must be a dotted string or a list of non-empty "
            "strings (use a list when a segment contains a dot); segments "
            "may not contain '[' or ']'"
        )
    if problems:
        return None, problems
    return Address(component, path), []


def _segments(raw: Any) -> tuple[str, ...] | None:
    if isinstance(raw, str):
        segments = tuple(raw.split("."))
    elif isinstance(raw, list):
        segments = tuple(raw)
    else:
        return None
    if not segments or not all(
            isinstance(segment, str) and segment
            and "[" not in segment and "]" not in segment
            for segment in segments
    ):
        return None
    return segments


def check_target(address: Address, session: Mapping) -> str | None:
    """The problem with `address` as a target in `session`, or None."""
    groups = framework.COMPONENT_GROUPS
    if address.component is not None:
        if address.component in groups:
            return (
                f"'{address.component}' is a component group, not a "
                "component; name the component inside it"
            )
        if not framework.is_listed_component(session, address.component):
            return (
                f"component '{address.component}' is not listed in the "
                "selected session"
            )
        return None
    if address.path[0] in groups:
        return (
            f"'{address.path[0]}' is a component group; address a component "
            "with 'component' instead of a path through the group"
        )
    for owned in FRAMEWORK_OWNED:
        shorter = min(len(owned), len(address.path))
        if address.path[:shorter] == owned[:shorter]:
            return (
                f"'{'.'.join(owned)}' is set by the HPC layer for every task "
                "and cannot be addressed"
            )
    return None


def _key(path: tuple[str, ...]) -> str:
    """OmegaConf key syntax: `a.b`, with a dotted segment as `a[b.c]`
    (no dot before a bracket)."""
    key = ""
    for segment in path:
        if "." in segment:
            key += f"[{segment}]"
        else:
            key += f".{segment}" if key else segment
    return key


def read(session: Mapping, address: Address) -> Any:
    """The value at `address`, or `_MISSING`."""
    base = (
        framework.component_entry_of(session, address.component)
        if address.component is not None else session
    )
    if not isinstance(base, Mapping):
        return _MISSING
    value = OmegaConf.select(
        OmegaConf.create(dict(base)), _key(address.path), default=_MISSING,
    )
    if value is _MISSING:
        return _MISSING
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value)
    return value


def is_missing(value: Any) -> bool:
    return value is _MISSING


def write(session: Mapping, address: Address, value: Any) -> dict:
    """A copy of `session` with `value` at `address` (missing keys are
    created; the value replaces what was there)."""
    if address.component is None:
        updated = OmegaConf.create(dict(session))
        OmegaConf.update(updated, _key(address.path), value, merge=False)
        return OmegaConf.to_container(updated)
    entry = framework.component_entry_of(session, address.component)
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"component '{address.component}' is configured with a "
            f"non-mapping value ({entry!r}); it has no keys to address"
        )
    updated_entry = OmegaConf.create(dict(entry))
    OmegaConf.update(updated_entry, _key(address.path), value, merge=False)
    return framework.with_entry(
        session, address.component, OmegaConf.to_container(updated_entry),
    )
