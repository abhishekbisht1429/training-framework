"""Taking components from another run's checkpoint into a new session.

`import_components` names a resource of another run. The session takes that
resource, and every instance it was wired to, in as its own components: it
restores them from the source's stored session state -- constructor
arguments, wiring, `state_version` and state, through restore's own path --
so it then drives their lifecycle, state, device and wiring like any other
component's. A resume or a rank worker restores them from this run's own
checkpoint and never reads the source again.

This module turns one import into the part of the source's session state to
restore; `SessionComponents` restores it. Nothing here builds a component.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from training_framework.components.base import ComponentDependencyError
from training_framework.components.config import component_bindings_from_config
from training_framework.components.edges import (
    EdgeKind,
    declared_edges,
    instances_of,
)
from training_framework.components.naming import (
    INSTANCE_SEPARATOR,
    _INSTANCE_SUFFIX_PATTERN,
    implementation_of,
    parse_instance_name,
)
from training_framework.session.checkpoint_format import (
    is_checkpoint_directory,
    read_checkpoint,
    read_manifest,
)
from training_framework.session.components import ComponentNotFoundError

IMPORT_COMPONENTS_KEY = "import_components"
IMPORTS_STATE_KEY = "imports"
"""The session state's record of its imports: the role bindings they added
(`bindings`). Which import brought a component in is on that component's
own entry (`imported_by`)."""

_IMPORT_KEYS = frozenset({"checkpoint", "resource", "role", "bind", "suffix"})


def instance_named_in_manifest(manifest: Mapping, name: str) -> str:
    """Resolve `name` from what a checkpoint stored, importing nothing.

    In order: an instance name; a top-level `component_bindings` entry, or a
    role an import of that run bound; the instance the checkpoint's
    components were given when they asked for `name`; the sole instance of
    that implementation.
    """
    stored = manifest["components"]
    if name in stored:
        return name
    config = manifest.get("config") or {}
    bindings = config.get("component_bindings") or config.get("aliases") or {}
    bindings = {
        **(bindings if isinstance(bindings, Mapping) else {}),
        **((manifest.get(IMPORTS_STATE_KEY) or {}).get("bindings") or {}),
    }
    bound = bindings.get(name)
    if isinstance(bound, str):
        candidates = instances_of(bound, stored)
    else:
        given = sorted({
            (info.get("dependencies") or {})[name]
            for info in stored.values()
            if name in (info.get("dependencies") or {})
        })
        candidates = given or instances_of(name, stored)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ComponentNotFoundError(
            f"The checkpoint has no component '{name}'; it holds "
            f"{sorted(stored)}"
        )
    raise ComponentDependencyError(
        f"'{name}' could be any of {candidates} in this checkpoint; name "
        "the instance"
    )


@dataclass(frozen=True)
class ComponentImport:
    """One entry of `import_components`, validated."""

    name: str
    checkpoint: str
    resource: str = "model"
    role: str | None = None
    bind: dict[str, str] = field(default_factory=dict)
    suffix: str | None = None

    @property
    def key(self) -> str:
        return f"{IMPORT_COMPONENTS_KEY}.{self.name}"


def parse_imports(value: Any) -> list[ComponentImport]:
    """Return the imports `import_components` configures, or say what is
    wrong with it."""
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{IMPORT_COMPONENTS_KEY} must be a mapping of import names to "
            f"their settings; got {value!r}"
        )
    return [_parse_import(name, entry) for name, entry in value.items()]


def _parse_import(name: Any, entry: Any) -> ComponentImport:
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{IMPORT_COMPONENTS_KEY} names must be non-empty strings; got "
            f"{name!r}"
        )
    key = f"{IMPORT_COMPONENTS_KEY}.{name}"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{key} must be a mapping; got {entry!r}")
    unknown = sorted(set(entry) - _IMPORT_KEYS)
    if unknown:
        raise ValueError(
            f"{key} has unknown keys {unknown}; it accepts "
            f"{sorted(_IMPORT_KEYS)}"
        )
    if "checkpoint" not in entry:
        raise ValueError(f"{key}.checkpoint is required")
    try:
        checkpoint = os.fspath(entry["checkpoint"])
    except TypeError:
        checkpoint = None
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(
            f"{key}.checkpoint must be a path; got {entry['checkpoint']!r}"
        )
    resource = entry.get("resource", "model")
    _require_name(resource, f"{key}.resource")
    role = entry.get("role")
    if role is not None:
        _require_name(role, f"{key}.role")
        if INSTANCE_SEPARATOR in role:
            raise ValueError(
                f"{key}.role must be a role name, not an instance; got {role!r}"
            )
    bind = entry.get("bind", {})
    if not isinstance(bind, Mapping):
        raise ValueError(
            f"{key}.bind must map a prerequisite of the source run to a "
            f"component of this one; got {bind!r}"
        )
    for source, target in bind.items():
        _require_name(source, f"{key}.bind key")
        _require_name(target, f"{key}.bind.{source}")
    suffix = entry.get("suffix")
    if suffix is not None and (
            not isinstance(suffix, str)
            or not _INSTANCE_SUFFIX_PATTERN.match(suffix)
    ):
        raise ValueError(
            f"{key}.suffix must be one or more letters, digits or "
            f"underscores; got {suffix!r}"
        )
    return ComponentImport(
        name=name,
        checkpoint=checkpoint,
        resource=resource,
        role=role,
        bind=dict(bind),
        suffix=suffix,
    )


def _require_name(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string; got {value!r}")


@dataclass(frozen=True)
class PlannedImport:
    """What one import restores, before anything is built.

    `components_state` is the part of the source's session state to restore,
    under this session's names, with each instance's wiring pointed at the
    other imported instances or at the `bind` targets. `names` maps a source
    instance to this session's name for it, for every renamed instance and
    every `bind`; it is empty when neither is configured. `bind` maps each
    `bind` key, as written, to the component of this session it resolved
    to.
    """

    spec: ComponentImport
    root: str
    components_state: dict[str, dict[str, Any]]
    names: dict[str, str]
    bind: dict[str, str]

    @property
    def instances(self) -> list[str]:
        return list(self.components_state)


def plan_import(
        spec: ComponentImport,
        registry: Mapping[str, type],
        resolve_local: Callable[[str], str],
) -> PlannedImport:
    """Select, from the source checkpoint, what `spec` imports.

    `registry` is this session's component registry, for the checks a class
    decides; `resolve_local` resolves a `bind` target in this session.
    """
    path = spec.checkpoint
    if not os.path.exists(path):
        raise FileNotFoundError(f"{spec.key}.checkpoint does not exist: {path}")
    if not is_checkpoint_directory(path):
        raise ValueError(
            f"{spec.key}: {path} is a single-file checkpoint, written before "
            "0.5.0, and cannot be imported. Convert it once with "
            "Checkpointer.save_checkpoint(Checkpointer.load_checkpoint(path), "
            "new_path)."
        )
    manifest = read_manifest(path)
    stored = manifest["components"]

    root = _resolve_in_source(manifest, spec.resource, f"{spec.key}.resource")
    bound: dict[str, str] = {}
    bind_keys: dict[str, str] = {}
    for source_name, target in spec.bind.items():
        instance = _resolve_in_source(
            manifest, source_name, f"{spec.key}.bind key '{source_name}'",
        )
        if instance in bound:
            raise ValueError(
                f"{spec.key}.bind keys '{bind_keys[instance]}' and "
                f"'{source_name}' are both the source's '{instance}'"
            )
        bind_keys[instance] = source_name
        if instance == root:
            raise ValueError(
                f"{spec.key}.bind names '{source_name}', which is the imported "
                f"resource '{root}' itself"
            )
        try:
            bound[instance] = resolve_local(target)
        except ComponentDependencyError as error:
            raise ComponentDependencyError(
                f"{spec.key}.bind.{source_name}: {error}"
            ) from None

    # The resource and every instance it was wired to, stopping at a bound
    # prerequisite, which this session provides instead.
    reached: list[str] = []
    reached_bindings: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in bound:
            reached_bindings.add(name)
            continue
        if name in reached:
            continue
        reached.append(name)
        dependencies = stored[name].get("dependencies") or {}
        pending.extend(
            target for target in reversed(list(dependencies.values()))
            if target in stored
        )

    unreached = sorted(set(bound) - reached_bindings)
    if unreached:
        keys = [bind_keys[instance] for instance in unreached]
        raise ValueError(
            f"{spec.key}.bind names {keys}, which '{root}' is not wired to, "
            "so nothing would be bound. The import brings in "
            f"{sorted(reached)}."
        )

    problems = [
        problem
        for name in reached
        if (problem := _import_problem(name, stored[name], registry))
    ]
    if problems:
        raise ComponentDependencyError(
            f"{spec.key} cannot import '{root}':\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    renamed = {name: _renamed(name, spec.suffix) for name in reached}
    names = {
        source: new for source, new in renamed.items() if source != new
    }
    names.update(bound)

    # Stored order, which is prerequisite-first as the source activated them.
    source_state = read_checkpoint(path, components=reached)["components_state"]
    components_state = {}
    for name, info in source_state.items():
        info = dict(info)
        # This import brought it in, whatever the source run says: a source
        # may itself have imported it.
        info["imported_by"] = spec.key
        info["dependencies"] = {
            asked: names.get(target, target)
            for asked, target in (info.get("dependencies") or {}).items()
        }
        components_state[renamed[name]] = info
    return PlannedImport(
        spec=spec,
        root=renamed[root],
        components_state=components_state,
        names=names,
        bind={bind_keys[instance]: target for instance, target in bound.items()},
    )


def stored_bindings(state: Mapping[str, Any]) -> dict[str, Any]:
    """The bindings a stored session state was built with.

    Its configured `component_bindings`, plus the role bindings its imports
    added (`import_components.<name>.role`), which the configuration does
    not hold: they name the imported instance, which only the import knew.
    """
    bindings = dict(component_bindings_from_config(state["config"]))
    bindings.update((state.get(IMPORTS_STATE_KEY) or {}).get("bindings") or {})
    return bindings


def _resolve_in_source(manifest: Mapping, name: str, where: str) -> str:
    try:
        return instance_named_in_manifest(manifest, name)
    except ComponentNotFoundError as error:
        raise ValueError(f"{where}: {error}") from None
    except ComponentDependencyError as error:
        raise ComponentDependencyError(f"{where}: {error}") from None


def _import_problem(
        name: str,
        info: Mapping[str, Any],
        registry: Mapping[str, type],
) -> str | None:
    """Why `name` cannot be imported, or None.

    What restore itself checks -- a class no longer registered, registered
    as another kind, or declaring a prerequisite its wiring lacks -- is left
    to restore. These are what only an import has to refuse.
    """
    if info.get("component_type") != "Resource":
        return (
            f"'{name}' is a {info.get('component_type')}; only resources can "
            "be imported"
        )
    component_class = registry.get(implementation_of(name))
    if component_class is None:
        return None
    if getattr(component_class, "singleton", False):
        return (
            f"'{name}' is @singleton: it belongs to the run it was built in, "
            "so it cannot be imported. Give what needs it this session's own "
            "instance with `bind`."
        )
    companions = sorted(
        edge.asked for edge in declared_edges(component_class)
        if edge.kind is EdgeKind.COMPANION
    )
    if companions:
        return (
            f"'{name}' activates {companions} (@activates): those carry out the "
            "source run's behaviour, and here they would be activated anew "
            "and wired by this session, so the import is refused. Give what "
            "needs it this session's own instance with `bind`."
        )
    return None


def _renamed(name: str, suffix: str | None) -> str:
    if suffix is None:
        return name
    implementation, own_suffix = parse_instance_name(name)
    if own_suffix is None:
        return f"{implementation}{INSTANCE_SEPARATOR}{suffix}"
    return f"{implementation}{INSTANCE_SEPARATOR}{suffix}_{own_suffix}"


__all__ = [
    "IMPORT_COMPONENTS_KEY",
    "ComponentImport",
    "IMPORTS_STATE_KEY",
    "PlannedImport",
    "instance_named_in_manifest",
    "parse_imports",
    "plan_import",
    "stored_bindings",
]
