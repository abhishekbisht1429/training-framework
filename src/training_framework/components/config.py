import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


ROLE_BINDINGS_KEY = "role_bindings"
"""Session-wide bindings, at the top level of a session."""

DEPENDENCIES_BINDINGS_KEY = "dependencies_role_bindings"
"""The bindings of one component's dependencies, inside its own entry."""

LEGACY_BINDINGS_KEY = "component_bindings"
LEGACY_ALIASES_KEY = "aliases"
_BINDINGS_KEYS = (ROLE_BINDINGS_KEY, LEGACY_BINDINGS_KEY, LEGACY_ALIASES_KEY)

COMPONENT_GROUPS = ("resources", "hooks", "steps")
"""Top-level keys that list components of one kind. Which kind each holds is
checked where the registry is known (`SessionComponents`)."""

COMMON_RESERVED_CONFIG_NAMES = frozenset({
    *_BINDINGS_KEYS,
    *COMPONENT_GROUPS,
    "components",
    "import_components",
    "session_config",
    "session_kwargs",
    "session_type",
})


SESSION_RESERVED_CONFIG_NAMES: dict[str, frozenset[str]] = {}


def reserved_config_names(session_type: str | None = None) -> frozenset[str]:
    if session_type is None:
        normalized_type = "training"
    elif not isinstance(session_type, str):
        raise TypeError("session_type must be a string or None")
    else:
        normalized_type = session_type.strip()
        if not normalized_type:
            raise ValueError("session_type must not be empty")

    return (
        COMMON_RESERVED_CONFIG_NAMES
        | SESSION_RESERVED_CONFIG_NAMES.get(normalized_type, frozenset())
    )


def reject_legacy_components_entry(config: Mapping) -> None:
    if "components" in config:
        raise ValueError(
            "The top-level 'components' entry is no longer supported. "
            "Activate root components with top-level mappings such as "
            "'component_name: {}'. Dependencies that do not define a custom "
            "constructor are activated automatically."
        )


@dataclass(frozen=True)
class ComponentEntry:
    """One configured component, as a session configuration lists it."""

    config: dict[str, Any]
    """What the constructor receives: the entry without
    `dependencies_role_bindings`."""
    group: str | None
    """The group it is listed under, or None when listed at the top level."""
    written: dict[str, Any]
    """The entry as written (an empty value read as `{}`),
    `dependencies_role_bindings` included; what a session extension
    compares."""


@dataclass(frozen=True)
class SessionConfigView:
    """A session configuration read once, whichever layout it uses.

    The configuration itself is kept as written; everything that needs its
    components or bindings reads them from here, so the flat and grouped
    layouts and the legacy binding keys are told apart in one place.
    """

    components: dict[str, ComponentEntry]
    """Configured components by the name they are listed under, in order."""
    role_bindings: dict[str, Any]
    """Session-wide bindings, `role: implementation`."""
    dependency_bindings: dict[str, dict[str, Any]]
    """Each component's own wiring, `consumer: {role: target}`. Kept apart
    from `role_bindings`: the two are keyed differently (role, consumer), so
    merged, a role and a component of the same name would overwrite each
    other."""
    framework: dict[str, Any]
    """The reserved top-level entries other than the component groups."""

    def comparable(self) -> dict[str, Any]:
        """The configuration flattened to one key per framework entry and per
        component, so two layouts of the same run compare equal."""
        return {
            **self.framework,
            **{name: entry.written for name, entry in self.components.items()},
        }

    def where(self, name: str) -> str:
        """Where `name` is listed, for messages."""
        group = self.components[name].group
        return f"under '{group}'" if group else "at the top level"


def parse_session_config(
        config: Mapping,
        *,
        session_type: str | None = None,
        warn_legacy: bool = False,
) -> SessionConfigView:
    """Read a session configuration's components and bindings.

    `warn_legacy` warns about a top-level `component_bindings`; it is set
    when a session is built from a configuration someone wrote, and left off
    when one is read back from a checkpoint, which cannot be changed.
    """
    reject_legacy_components_entry(config)
    reserved = reserved_config_names(session_type)

    global_bindings, legacy_wiring = _top_level_bindings(config, warn_legacy)

    components: dict[str, ComponentEntry] = {}
    for key, value in config.items():
        if key in COMPONENT_GROUPS:
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"'{key}' must be a mapping of component names to their "
                    "configuration"
                )
            for name, entry in value.items():
                _add_entry(components, name, entry, group=key, reserved=reserved)
        elif key not in reserved:
            _add_entry(components, key, value, group=None, reserved=reserved)

    wiring: dict[str, dict] = {}
    for name, entry in components.items():
        own = entry.written.get(DEPENDENCIES_BINDINGS_KEY)
        if own is None:
            continue
        place = f"'{name}'" if entry.group is None else f"'{entry.group}.{name}'"
        if not isinstance(own, Mapping):
            raise TypeError(
                f"{place} dependencies_role_bindings must be a mapping of role "
                "names to the components they are bound to"
            )
        for role_name, target in own.items():
            if isinstance(target, Mapping):
                raise ValueError(
                    f"{place} dependencies_role_bindings binds '{role_name}' to "
                    "a mapping. A component's dependencies_role_bindings name "
                    "the component each of its own roles is bound to; wire "
                    "another component in that component's own entry."
                )
        if name in legacy_wiring:
            raise ValueError(
                f"'{name}' is wired both by its own dependencies_role_bindings "
                f"and by the deprecated top-level component_bindings entry "
                f"'{name}'. Keep only its dependencies_role_bindings."
            )
        if own:
            wiring[name] = dict(own)

    framework = {
        key: value for key, value in config.items()
        if key in reserved and key not in COMPONENT_GROUPS
    }
    return SessionConfigView(
        components=components,
        role_bindings=global_bindings,
        dependency_bindings={**legacy_wiring, **wiring},
        framework=framework,
    )


def _top_level_bindings(
        config: Mapping,
        warn_legacy: bool,
) -> tuple[dict, dict]:
    """Return the session-wide bindings and the legacy per-consumer wiring."""
    given = [key for key in _BINDINGS_KEYS if key in config]
    if len(given) > 1:
        raise ValueError(
            f"'{given[0]}' and '{given[1]}' are both configured. Use "
            "'role_bindings' ('component_bindings' and 'aliases' are its "
            "deprecated names), not both."
        )
    if not given:
        return {}, {}

    key = given[0]
    bindings = config[key]
    if key == LEGACY_ALIASES_KEY:
        warnings.warn(
            "The top-level 'aliases' entry is deprecated (as is "
            "'component_bindings'); use 'role_bindings' instead",
            DeprecationWarning,
            stacklevel=3,
        )
    elif key == LEGACY_BINDINGS_KEY and warn_legacy:
        warnings.warn(
            "The top-level 'component_bindings' entry is deprecated; use "
            "'role_bindings' for session-wide bindings and a component's own "
            "'dependencies_role_bindings' entry for its wiring",
            DeprecationWarning,
            stacklevel=3,
        )
    if bindings is None:
        return {}, {}
    if not isinstance(bindings, Mapping):
        raise TypeError(f"'{key}' must be a mapping of strings to strings")

    global_bindings: dict = {}
    legacy_wiring: dict = {}
    for name, value in bindings.items():
        if not isinstance(value, Mapping):
            global_bindings[name] = value
        elif key == ROLE_BINDINGS_KEY:
            raise ValueError(
                f"Top-level 'role_bindings' holds session-wide bindings "
                f"(role: implementation) only, but '{name}' maps to a "
                f"mapping. Wire one component in its own entry: "
                f"{name}: {{dependencies_role_bindings: {{...}}}}"
            )
        else:
            legacy_wiring[name] = value
    return global_bindings, legacy_wiring


def _add_entry(
        components: dict[str, ComponentEntry],
        name: Any,
        entry: Any,
        *,
        group: str | None,
        reserved: frozenset[str],
) -> None:
    """Add one listed component. Every listed component comes through here,
    at the top level or in a group, so its name is checked here, once."""
    place = f"'{name}'" if group is None else f"'{group}.{name}'"
    if not isinstance(name, str) or not name:
        raise ValueError(f"Component names must be non-empty strings; got {name!r}")
    if name in reserved:
        raise ValueError(
            f"{place} uses the reserved name '{name}', which is a framework "
            "entry, not a component. Framework entries belong at the top "
            "level of a session."
        )
    if name in components:
        first = components[name].group
        raise ValueError(
            f"Component '{name}' is listed twice: "
            f"{f'under {first!r}' if first else 'at the top level'} and "
            f"{f'under {group!r}' if group else 'at the top level'}. "
            "List it once."
        )
    if entry is None:
        written = {}
    elif isinstance(entry, Mapping):
        written = dict(entry)
    else:
        raise ValueError(f"The value corresponding to the key {place} is not a mapping")
    config = {
        key: value for key, value in written.items()
        if key != DEPENDENCIES_BINDINGS_KEY
    }
    components[name] = ComponentEntry(config=config, group=group, written=written)


def find_component_entry(config: Mapping, name: str) -> tuple[str, ...] | None:
    """Return the path of `name`'s entry in `config`, flat or grouped.

    For code that has to change one component's entry in the configuration
    as written, such as the launch overlaying the resolved ddp topology.
    """
    if name in config:
        return (name,)
    for group in COMPONENT_GROUPS:
        members = config.get(group)
        if isinstance(members, Mapping) and name in members:
            return (group, name)
    return None


def component_entry(config: Mapping, name: str) -> Any:
    """The entry `name` is listed with, flat or grouped; None when it is not
    listed or listed with an empty value (tell them apart with
    `find_component_entry`)."""
    path = find_component_entry(config, name)
    if path is None:
        return None
    value = config
    for key in path:
        value = value[key]
    return value


def with_component_entry(config: Mapping, name: str, entry: Any) -> dict:
    """A copy of `config` with `name`'s entry replaced where it is listed, or
    added at the top level when it is not; the rest is shared."""
    path = find_component_entry(config, name) or (name,)
    updated = dict(config)
    if len(path) == 1:
        updated[name] = entry
    else:
        group = path[0]
        updated[group] = {**dict(config[group]), name: entry}
    return updated
