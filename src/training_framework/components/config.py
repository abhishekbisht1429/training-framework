from collections.abc import Mapping


RESERVED_CONFIG_NAMES = frozenset({
    "aliases",
    "components",
    "session_config",
    "session_kwargs",
    "session_type",
})


def selected_component_names(config: Mapping) -> list[str]:
    """Validate and return explicitly selected component names."""
    selected = config.get("components", [])
    if not isinstance(selected, list):
        raise TypeError("'components' must be a list of component names")

    names: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if not isinstance(name, str):
            raise TypeError("'components' entries must be strings")
        if not name:
            raise ValueError("'components' entries must not be empty")
        if name in RESERVED_CONFIG_NAMES:
            raise ValueError(
                f"'{name}' is a reserved configuration name and cannot "
                "select a component"
            )
        if name in seen:
            raise ValueError(
                f"Component '{name}' appears more than once in 'components'"
            )
        seen.add(name)
        names.append(name)
    return names
