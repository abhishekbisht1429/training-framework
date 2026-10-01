"""Taking components from another run's checkpoint into a new session.

`import_components` is keyed by instance names of another run. The session
takes each such resource, and every instance it was wired to, in as its own
components: it
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
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from training_framework.components.base import ComponentDependencyError
from training_framework.components.config import parse_session_config
from training_framework.components.edges import (
    EdgeKind,
    declared_edges,
    instances_of,
)
from training_framework.components.naming import (
    IMPORTED_SUFFIX,
    INSTANCE_SEPARATOR,
    _INSTANCE_SUFFIX_PATTERN,
    is_imported_suffix,
    implementation_of,
    parse_instance_name,
)
from training_framework.session.checkpoint_format import (
    IMPORTED_INTO_NAMESPACE_KEY,
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

_IMPORT_KEYS = frozenset({
    "checkpoint", "instance_name", "overwritten_dependencies",
})
_LEGACY_IMPORT_KEYS = frozenset({"resource", "role", "suffix"})
"""Keys only the deprecated form has: an entry holding any of them is keyed
by an import label, not by the source's instance name."""


def instance_named_in_manifest(manifest: Mapping, name: str) -> str:
    """Resolve `name` from what a checkpoint stored, importing nothing.

    In order: an instance name; a session-wide `role_bindings` entry, or a
    role an import of that run bound; the instance the checkpoint's
    components were given when they asked for `name`; the sole instance of
    that implementation.
    """
    stored = manifest["components"]
    if name in stored:
        return name
    bound = stored_bindings(manifest).roles.get(name)
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
    """One entry of `import_components`, validated.

    `name` is the entry's key. For a keyed import (`legacy` false) it is the
    source's instance name, `resource` is that same name and `instance_name`
    renames what it brings in (the reserved `imported` when not given);
    `index` is its position when the key holds a list. A labelled import
    (`legacy`, deprecated) is keyed by a label and uses `resource`, `role`
    and `suffix` instead.
    """

    name: str
    checkpoint: str
    resource: str = "model"
    role: str | None = None
    overwritten_dependencies: dict[str, str] = field(default_factory=dict)
    suffix: str | None = None
    instance_name: str | None = None
    legacy: bool = True
    index: int | None = None

    @property
    def key(self) -> str:
        """How errors and `imported_by` name this import."""
        position = "" if self.index is None else f"[{self.index}]"
        return f"{IMPORT_COMPONENTS_KEY}.{self.name}{position}"

    @property
    def rename_suffix(self) -> str | None:
        """The suffix every imported instance is renamed with, if any."""
        if self.legacy:
            return self.suffix
        return self.instance_name or IMPORTED_SUFFIX


def parse_imports(value: Any) -> list[ComponentImport]:
    """Return the imports `import_components` configures, or say what is
    wrong with it."""
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{IMPORT_COMPONENTS_KEY} must be a mapping of the source's "
            f"instance names to their settings; got {value!r}"
        )
    imports: list[ComponentImport] = []
    labelled: list[str] = []
    for name, entry in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{IMPORT_COMPONENTS_KEY} keys must be non-empty strings; got "
                f"{name!r}"
            )
        # Any sequence, so a configuration still held by OmegaConf
        # (a ListConfig) reads like a plain one.
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            if not entry:
                raise ValueError(
                    f"{IMPORT_COMPONENTS_KEY}.{name} is an empty list; give "
                    "one mapping per checkpoint to import it from"
                )
            imports.extend(
                _parse_import(name, item, index=index)
                for index, item in enumerate(entry)
            )
            continue
        parsed = _parse_import(name, entry)
        if parsed.legacy:
            labelled.append(name)
        imports.append(parsed)
    if labelled:
        warnings.warn(
            f"{IMPORT_COMPONENTS_KEY} entries {labelled} use the deprecated "
            "form (an import label with `resource`, `role` or `suffix`). Key "
            "each import by the source's instance name, rename with "
            "`instance_name`, and bind roles with `role_bindings`",
            DeprecationWarning,
            stacklevel=2,
        )
    return imports


def _parse_import(
        name: str,
        entry: Any,
        *,
        index: int | None = None,
) -> ComponentImport:
    key = f"{IMPORT_COMPONENTS_KEY}.{name}" + (
        "" if index is None else f"[{index}]"
    )
    if not isinstance(entry, Mapping):
        raise ValueError(f"{key} must be a mapping; got {entry!r}")
    legacy_keys = sorted(set(entry) & _LEGACY_IMPORT_KEYS)
    if legacy_keys and index is not None:
        raise ValueError(
            f"{key} uses {legacy_keys}, which only the deprecated labelled "
            "form accepts, and that form has no lists. Key the import by the "
            "source's instance name and rename it with `instance_name`."
        )
    legacy = bool(legacy_keys)
    accepted = (
        (_IMPORT_KEYS - {"instance_name"}) | _LEGACY_IMPORT_KEYS
        if legacy else _IMPORT_KEYS
    )
    unknown = sorted(set(entry) - accepted)
    if unknown:
        raise ValueError(
            f"{key} has unknown keys {unknown}; it accepts {sorted(accepted)}"
            + (
                " (`instance_name` belongs to the keyed form, which has no "
                "`resource`, `role` or `suffix`)"
                if legacy and "instance_name" in unknown else ""
            )
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
    resource = entry.get("resource", "model") if legacy else name
    if legacy:
        _require_name(resource, f"{key}.resource")
    role = entry.get("role")
    if role is not None:
        _require_name(role, f"{key}.role")
        if INSTANCE_SEPARATOR in role:
            raise ValueError(
                f"{key}.role must be a role name, not an instance; got {role!r}"
            )
    overwritten = entry.get("overwritten_dependencies", {})
    if not isinstance(overwritten, Mapping):
        raise ValueError(
            f"{key}.overwritten_dependencies must map a dependency of the "
            f"source run to a component of this one; got {overwritten!r}"
        )
    for source, target in overwritten.items():
        _require_name(source, f"{key}.overwritten_dependencies key")
        _require_name(target, f"{key}.overwritten_dependencies.{source}")
    suffix = _suffix(entry.get("suffix"), f"{key}.suffix")
    instance_name = _suffix(entry.get("instance_name"), f"{key}.instance_name")
    if is_imported_suffix(instance_name):
        raise ValueError(
            f"{key}.instance_name '{instance_name}' is reserved: an import "
            f"without `instance_name` already uses '{IMPORTED_SUFFIX}'. Leave "
            "it out, or choose another name."
        )
    return ComponentImport(
        name=name,
        checkpoint=checkpoint,
        resource=resource,
        role=role,
        overwritten_dependencies=dict(overwritten),
        suffix=suffix,
        instance_name=instance_name,
        legacy=legacy,
        index=index,
    )


def _suffix(value: Any, where: str) -> str | None:
    if value is not None and (
            not isinstance(value, str)
            or not _INSTANCE_SUFFIX_PATTERN.match(value)
    ):
        raise ValueError(
            f"{where} must be one or more letters, digits or underscores; "
            f"got {value!r}"
        )
    return value


def _require_name(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string; got {value!r}")


@dataclass(frozen=True)
class PlannedImport:
    """What one import restores, before anything is built.

    `components_state` is the part of the source's session state to restore,
    under this session's names, with each instance's wiring pointed at the
    other imported instances or at the `overwritten_dependencies` targets.
    `names` maps a source instance to this session's name for it, for every
    renamed and every overwritten instance; it is empty when neither is
    configured. `overwritten_dependencies` maps each key, as written, to the
    component of this session it resolved to.
    """

    spec: ComponentImport
    root: str
    components_state: dict[str, dict[str, Any]]
    names: dict[str, str]
    overwritten_dependencies: dict[str, str]

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
    decides; `resolve_local` resolves an `overwritten_dependencies` target in
    this session.
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
    spec = _labelled_if_not_keyed(spec, manifest)

    root = (
        _resolve_in_source(manifest, spec.resource, f"{spec.key}.resource")
        if spec.legacy else _instance_in_source(manifest, spec)
    )
    where = f"{spec.key}.overwritten_dependencies"
    bound: dict[str, str] = {}
    overwritten_keys: dict[str, str] = {}
    for source_name, target in spec.overwritten_dependencies.items():
        instance = _resolve_in_source(
            manifest, source_name, f"{where} key '{source_name}'",
        )
        if instance in bound:
            raise ValueError(
                f"{where} keys '{overwritten_keys[instance]}' and "
                f"'{source_name}' are both the source's '{instance}'"
            )
        overwritten_keys[instance] = source_name
        if instance == root:
            raise ValueError(
                f"{where} names '{source_name}', which is the imported "
                f"resource '{root}' itself"
            )
        try:
            bound[instance] = resolve_local(target)
        except ComponentDependencyError as error:
            raise ComponentDependencyError(
                f"{where}.{source_name}: {error}"
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
        keys = [overwritten_keys[instance] for instance in unreached]
        raise ValueError(
            f"{where} names {keys}, which '{root}' is not wired to, "
            "so nothing would be overwritten. The import brings in "
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

    renamed = {name: _renamed(name, spec.rename_suffix) for name in reached}
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
        if spec.legacy:
            info.pop(IMPORTED_INTO_NAMESPACE_KEY, None)
        else:
            info[IMPORTED_INTO_NAMESPACE_KEY] = True
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
        overwritten_dependencies={
            overwritten_keys[instance]: target
            for instance, target in bound.items()
        },
    )


class StoredBindings(NamedTuple):
    """The bindings a stored session state was built with, kept apart as
    `RoleBindings` takes them."""

    roles: dict[str, Any]
    """Session-wide: the configured `role_bindings` (or legacy
    `component_bindings`), plus the roles its imports bound
    (`import_components.<name>.role`), which the configuration does not
    hold: they name the imported instance, which only the import knew."""
    dependencies: dict[str, dict[str, Any]]
    """Each component's own wiring (`dependencies_role_bindings`)."""


def stored_bindings(state: Mapping[str, Any]) -> StoredBindings:
    """The bindings a stored session state was built with."""
    view = parse_session_config(
        state.get("config") or {},
        session_type=state.get("session_type"),
    )
    roles = dict(view.role_bindings)
    roles.update((state.get(IMPORTS_STATE_KEY) or {}).get("bindings") or {})
    return StoredBindings(roles=roles, dependencies=view.dependency_bindings)


def _labelled_if_not_keyed(
        spec: ComponentImport,
        manifest: Mapping[str, Any],
) -> ComponentImport:
    """Read an entry holding only keys both forms share as the deprecated
    labelled form when its key is no instance of the source.

    A keyed import must name one of the source's instances, so such a key
    can only be a label, which imports the source's `model`. An entry with
    `instance_name` or in a list is keyed whatever its key says, and so is a
    key that is a role of the source: that is a keyed import naming the role
    instead of the instance, refused by `_instance_in_source`.

    The fallback warns with a `FutureWarning`, which Python shows by default:
    a mistyped key reaches here too, and must not pass unseen.
    """
    if (
            spec.legacy or spec.index is not None
            or spec.instance_name is not None
            or spec.name in manifest["components"]
            or spec.name in stored_bindings(manifest).roles
    ):
        return spec
    holds = sorted(manifest["components"])
    if "model" not in stored_bindings(manifest).roles:
        raise ValueError(
            f"{spec.key}: the checkpoint has no component '{spec.name}'; it "
            f"holds {holds}. An import is keyed by the source's instance "
            "name. (Read as an import label -- the deprecated form -- it "
            "would import the source's `model`, which that run never bound.)"
        )
    warnings.warn(
        f"{spec.key}: '{spec.name}' is not an instance of the source run, so "
        "it is read as an import label (the deprecated form) and imports "
        "the source's `model`, under its source names. If the key is "
        "mistyped, fix it; otherwise key the import by the source's "
        f"instance name. The source holds {holds}.",
        FutureWarning,
        stacklevel=3,
    )
    return replace(spec, legacy=True, resource="model")


def _instance_in_source(manifest: Mapping, spec: ComponentImport) -> str:
    """The source instance a keyed import names: its key, exactly."""
    stored = manifest["components"]
    if spec.name in stored:
        return spec.name
    bound = stored_bindings(manifest).roles.get(spec.name)
    if isinstance(bound, str):
        instances = instances_of(bound, stored)
        raise ValueError(
            f"{spec.key}: '{spec.name}' is a role of the source run, bound to "
            f"{instances or [bound]}; key the import by the instance name"
        )
    raise ValueError(
        f"{spec.key}: the checkpoint has no component '{spec.name}'; it holds "
        f"{sorted(stored)}. An import is keyed by the source's instance name. "
        f"If '{spec.name}' is an import label (the deprecated form), key it "
        "by the component to import instead, or add `resource:`."
    )


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
            "instance with `overwritten_dependencies`."
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
            "needs it this session's own instance with "
            "`overwritten_dependencies`."
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
    "IMPORTED_INTO_NAMESPACE_KEY",
    "ComponentImport",
    "IMPORTS_STATE_KEY",
    "PlannedImport",
    "instance_named_in_manifest",
    "parse_imports",
    "plan_import",
    "stored_bindings",
]
