"""Data sources (plan section 8): only addresses listed in `data.sources`
are data. Each source is exactly one of `upload`, `cluster`, `prepare`."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from training_framework.hpc.config import address as addresses
from training_framework.hpc import framework
from training_framework.hpc.config.address import Address
from training_framework.hpc.names import check_name, hash_value
from training_framework.hpc.compile.plan import Artifact
from training_framework.hpc.config.schema import Profile, Resources, parse_resources
from training_framework.hpc.errors import Problems

KINDS = ("upload", "cluster", "prepare")
_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{[^{}]*\}")


class LocalFacts(Protocol):
    """What the compiler may learn about the workstation."""

    repository_root: str
    home: str

    def exists(self, path: str) -> bool: ...

    def content(self, path: str) -> tuple[str, int, int]:
        """(key, files, size) of a file or directory."""


@dataclass(frozen=True)
class Source:
    name: str
    address: Address
    kind: str
    spec: dict
    resources: Resources | None = None


@dataclass
class DataResult:
    artifacts: list[Artifact]
    # task index -> {address: cluster path}
    rewrites: dict[int, dict[Address, str]]
    # prepare node id -> (artifact, source, path)
    prepare_nodes: list[tuple[str, Artifact, Source, str]]
    # prepare node id -> task indices that use its path
    users: dict[str, set[int]]
    cluster_paths: list[str]
    rows: list[dict]


def parse_sources(raw: tuple, problems: Problems, profile: Profile) -> list[Source]:
    sources: list[Source] = []
    names = set()
    for index, item in enumerate(raw):
        where = f"data.sources[{index}]"
        if not isinstance(item, Mapping):
            problems.add(f"{where} must be a mapping")
            continue
        unknown = set(item) - {"name", "target", *KINDS}
        if unknown:
            problems.add(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
        name = item.get("name")
        if problem := check_name(name, f"{where}.name"):
            problems.add(problem)
            continue
        if name in names:
            problems.add(f"{where}: source name '{name}' is used twice")
        names.add(name)
        target, target_problems = addresses.parse_address(item.get("target"), f"{where}.target")
        for problem in target_problems:
            problems.add(problem)
        kinds = [kind for kind in KINDS if kind in item]
        if len(kinds) != 1:
            problems.add(f"{where} must have exactly one of {', '.join(KINDS)}")
            continue
        kind = kinds[0]
        spec = item[kind] if item[kind] is not None else {}
        if not isinstance(spec, Mapping):
            problems.add(f"{where}.{kind} must be a mapping")
            continue
        resources = _check_spec(kind, dict(spec), f"{where}.{kind}", problems, profile)
        if target is not None:
            sources.append(Source(name, target, kind, dict(spec), resources))
    return sources


def _check_spec(kind, spec, where, problems, profile) -> Resources | None:
    allowed = {
        "upload": {"version"},
        "cluster": {"path"},
        "prepare": {"command", "version", "inputs", "resources"},
    }[kind]
    unknown = set(spec) - allowed
    if unknown:
        problems.add(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
    if kind == "upload" and "version" in spec and (
            not isinstance(spec["version"], str) or not spec["version"]):
        problems.add(f"{where}.version must be a non-empty string")
    if kind == "cluster" and "path" in spec and (
            not isinstance(spec["path"], str) or not posixpath.isabs(spec["path"])):
        problems.add(f"{where}.path must be an absolute cluster path")
    if kind != "prepare":
        return None
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(
            isinstance(part, str) for part in command):
        problems.add(f"{where}.command must be a non-empty list of strings")
    else:
        for part in command:
            for token in _PLACEHOLDER.findall(part):
                if token not in ("{path}", "{{", "}}"):
                    problems.add(
                        f"{where}.command: unknown placeholder {token} (only "
                        "{path}; write {{ and }} for literal braces)"
                    )
    if not isinstance(spec.get("version"), str) or not spec.get("version"):
        problems.add(f"{where}.version is required (a non-empty string)")
    inputs = spec.get("inputs")
    if inputs is not None and (not isinstance(inputs, list) or not inputs or not all(
            isinstance(path, str) and path and not posixpath.isabs(path) for path in inputs)):
        problems.add(f"{where}.inputs must be a non-empty list of repository paths")
    resources = None
    if "resources" in spec:
        if profile.prepare_location == "login":
            problems.add(
                f"{where}.resources: the profile runs prepare steps on the "
                "login node (prepare_location: login), where resources mean nothing"
            )
        else:
            resources = parse_resources(spec["resources"], f"{where}.resources", problems)
    elif profile.prepare_location == "job":
        problems.add(f"{where}.resources is required when prepare steps run as jobs")
    return resources


def map_path(path: str, profile: Profile) -> str | None:
    """The cluster path for `path` through the longest matching `path_map`
    prefix, or None when no prefix matches."""
    best = None
    for local, remote in profile.path_map:
        if path == local or path.startswith(local.rstrip("/") + "/"):
            if best is None or len(local) > len(best[0]):
                best = (local, remote)
    if best is None:
        return None
    return best[1] + path[len(best[0]):]


def refused_upload(path: str, facts: LocalFacts) -> str | None:
    real = os.path.realpath(path)
    repo = os.path.realpath(facts.repository_root)
    home = os.path.realpath(facts.home)
    if real == "/" or real == home:
        return "is the filesystem root or the home directory"
    if repo == real or repo.startswith(real.rstrip("/") + "/"):
        return "contains the repository"
    if real.startswith(repo + "/"):
        return "is inside the repository (it travels with the source)"
    return None


def compile_sources(
        sources: list[Source],
        sessions: list[dict],
        profile: Profile,
        facts: LocalFacts,
        environment_key: str | None,
        code_identity: Any,
        problems: Problems,
) -> DataResult:
    result = DataResult([], {}, [], {}, [], [])
    for source in sources:
        values: dict[str, list[int]] = {}
        for index, session in enumerate(sessions):
            problem = addresses.check_target(source.address, session)
            if problem:
                problems.add(f"data source '{source.name}': {problem}")
                break
            value = addresses.read(session, source.address)
            if addresses.is_missing(value):
                problems.add(
                    f"data source '{source.name}': {source.address.describe()} "
                    f"has no value in task {index}"
                )
                break
            if not isinstance(value, str):
                problems.add(
                    f"data source '{source.name}': {source.address.describe()} "
                    f"must hold a path string; got {value!r}"
                )
                break
            values.setdefault(value, []).append(index)
        else:
            handler = {"upload": _upload, "cluster": _cluster, "prepare": _prepare}[source.kind]
            handler(source, values, profile, facts, environment_key, code_identity, problems, result)
    return result


def _rewrite(result: DataResult, indices, address, path):
    for index in indices:
        result.rewrites.setdefault(index, {})[address] = path


def _upload(source, values, profile, facts, _env, _code, problems, result):
    for value, indices in sorted(values.items()):
        if not posixpath.isabs(value) or not facts.exists(value):
            problems.add(
                f"data source '{source.name}': upload value {value!r} must be "
                "an absolute path that exists on this machine"
            )
            continue
        if reason := refused_upload(value, facts):
            problems.add(f"data source '{source.name}': {value} {reason}; refused")
            continue
        version = source.spec.get("version")
        if version is not None:
            key = hash_value({"source_name": source.name, "version": version})
            files = size = None
            details = {"local": value, "version": version, "declared": True}
        else:
            try:
                key, files, size = facts.content(value)
            except (OSError, ValueError) as error:
                problems.add(f"data source '{source.name}': {error}")
                continue
            details = {"local": value, "files": files, "size": size, "declared": False}
        mapped = map_path(value, profile)
        if mapped is not None:
            location, location_kind = mapped, "user-chosen"
        else:
            location = f"{profile.roots['data']}/{key}{value}"
            location_kind = "key-derived"
        artifact = Artifact("upload", source.name, key, location, location_kind, details)
        result.artifacts.append(artifact)
        _rewrite(result, indices, source.address, location)
        result.rows.append({
            "source": source.name, "value": value, "location": location,
            "size": "" if size is None else str(size), "kind": "upload",
        })


def _cluster(source, values, profile, _facts, _env, _code, problems, result):
    for value, indices in sorted(values.items()):
        if "path" in source.spec:
            location = source.spec["path"]
        else:
            location = map_path(value, profile) or value
        if not posixpath.isabs(location):
            problems.add(
                f"data source '{source.name}': cluster path {location!r} is "
                "not absolute"
            )
            continue
        _rewrite(result, indices, source.address, location)
        if location not in result.cluster_paths:
            result.cluster_paths.append(location)
            result.artifacts.append(Artifact(
                "cluster", source.name, None, location, "given", {"value": value},
            ))
            result.rows.append({
                "source": source.name, "value": value, "location": location,
                "size": "", "kind": "cluster",
            })


def _prepare(source, values, profile, _facts, environment_key, code_identity, problems, result):
    paths: dict[str, set[int]] = {}
    for value, indices in values.items():
        location = map_path(value, profile) or value
        if not posixpath.isabs(location):
            problems.add(
                f"data source '{source.name}': prepare path {location!r} is "
                "not absolute"
            )
            continue
        paths.setdefault(location, set()).update(indices)
    for number, (location, indices) in enumerate(sorted(paths.items())):
        key = hash_value({
            "name": source.name,
            "version": source.spec.get("version"),
            "command": source.spec.get("command"),
            "environment": environment_key,
            "hooks": [list(profile.hooks["common"]), list(profile.hooks["prepare"])],
            "code": code_identity(source.spec.get("inputs")),
        })
        artifact = Artifact("prepare", source.name, key, location, "user-chosen", {
            "command": list(source.spec.get("command") or []),
            "version": source.spec.get("version"),
            "inputs": source.spec.get("inputs"),
        })
        node_id = f"prepare-{source.name}-{number}"
        result.artifacts.append(artifact)
        result.prepare_nodes.append((node_id, artifact, source, location))
        result.users[node_id] = set(indices)
        _rewrite(result, indices, source.address, location)
        result.rows.append({
            "source": source.name, "value": location, "location": location,
            "size": "", "kind": "prepare",
        })


def expand_command(command: list[str], path: str) -> list[str]:
    def substitute(part: str) -> str:
        return _PLACEHOLDER.sub(
            lambda match: {"{path}": path, "{{": "{", "}}": "}"}[match.group(0)],
            part,
        )
    return [substitute(part) for part in command]


# Suggestion helper ---------------------------------------------------------


def _entry_paths(session: Mapping) -> dict[tuple, str]:
    """Session-level path prefix of every listed component -> its name."""
    reserved = framework.reserved_names(session.get("session_type"))
    prefixes = {}
    for key, value in session.items():
        if key in framework.COMPONENT_GROUPS and isinstance(value, Mapping):
            for name in value:
                prefixes[(key, name)] = name
        elif key not in reserved:
            prefixes[(key,)] = key
    return prefixes


def _scan(value, path, found):
    if isinstance(value, str):
        found.append((path, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _scan(item, path + (str(key),), found)
    elif isinstance(value, list):
        for position, item in enumerate(value):
            _scan(item, path + (str(position),), found)


def suggest_sources(
        sessions: list[dict],
        sources: list[Source],
        facts: LocalFacts,
) -> list[dict]:
    """`upload: {}` entries for absolute local paths not yet covered."""
    suggestions: dict[tuple, dict] = {}
    taken = {source.name for source in sources}
    for session in sessions:
        prefixes = _entry_paths(session)
        claimed = set()
        for source in sources:
            if source.address.component is None:
                claimed.add(source.address.path)
            else:
                for prefix, name in prefixes.items():
                    if name == source.address.component:
                        claimed.add(prefix + source.address.path)
        found: list[tuple[tuple, str]] = []
        _scan(session, (), found)
        for path, value in found:
            if path[:2] == ("session_config", "sessions_dir") or path in claimed:
                continue
            if not posixpath.isabs(value) or not facts.exists(value):
                continue
            if refused_upload(value, facts):
                continue
            target = None
            for prefix, name in prefixes.items():
                if path[:len(prefix)] == prefix and len(path) > len(prefix):
                    target = {"component": name, "path": list(path[len(prefix):])}
                    break
            if target is None:
                target = {"path": list(path)}
            if path not in suggestions:
                suggestions[path] = {"name": _suggested_name(path, taken), "target": target,
                                     "upload": {}}
    return list(suggestions.values())


_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")


def _suggested_name(path: tuple, taken: set[str]) -> str:
    """A valid source name from the last keys of `path` (component instances
    such as `imagenet#train` contain characters names may not), unique among
    `taken`, which it extends."""
    base = "_".join(part for part in path[-2:] if not part.isdigit())
    base = _UNSAFE_NAME_CHARACTERS.sub("_", base).strip("_") or "data"
    name, number = base, 2
    while name in taken:
        name, number = f"{base}_{number}", number + 1
    taken.add(name)
    return name
